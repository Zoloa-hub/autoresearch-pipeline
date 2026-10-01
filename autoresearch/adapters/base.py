"""实验后端适配器协议。

**解决的问题**：本项目最初把「实验怎么跑」硬编码在 ``stages/s4_experiment.py``
里——生成的脚本必须接受 ``--variant/--epochs/--seed/--out-dir``、必须写出固定表头
的 ``metrics.csv``、变体只能是 ``baseline``/``method``。那套约束是为一个合成分类
任务量身定做的，导致整个项目只能跑「小型 CPU 分类实验」，换个数据集或训练框架
就跑不动。

**本模块的分工边界**（这是设计上最关键的取舍）：

==================  ====================================================
谁                  负责什么
==================  ====================================================
适配器（本模块）    「怎么跑」+「怎么看结果」：环境准备、种子模板、
                    命令行构造、指标解析、环境可用性检查
``s4_experiment``   「写什么代码」+「跑不通怎么修」：调用 LLM 生成/
                    修补代码、捕获 traceback、反思闭环、**保真度护栏**
==================  ====================================================

**为什么代码生成不放进适配器**：如果把「生成代码」交给适配器，就会出现两套生成
逻辑，而且 ``s4`` 的调试闭环与保真度检查（补丁是否删掉了指标输出、是否加了
``except: pass``、是否写死高分）对适配器产物**完全失效**——那正是自动科研里最难
被发现的一类错误。所以：适配器只提供**模板**（``seed_code``），LLM 负责按需生成
与修补，任何适配器的产物都受同一套质量门禁约束。

**受控特例**：确实存在「我已经有训练脚本，别让 LLM 碰它」的真实诉求。这时适配器
把 ``owns_code`` 置为 ``True`` 并让 ``seed_code()`` 返回真实脚本，``s4`` 就跳过
代码生成，但**调试闭环仍然生效**（只依据 traceback 做最小修复），且补丁同样要过
保真度检查。协议只有一套，不分叉。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# --------------------------------------------------------------------------- #
# 运行规格
# --------------------------------------------------------------------------- #


@dataclass
class RunSpec:
    """一次实验运行的全部参数。

    刻意做成**开放**的：``params`` 承接来自 s3 消融矩阵的任意超参，``extra_args``
    是逃生舱（例如 ``--nproc_per_node=4`` 这类必须插在程序名之后的参数）。
    """

    variant: str
    seed: int
    out_dir: str  # 相对工作区的 POSIX 路径，如 "runs/method/seed_0"
    params: dict[str, Any] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)

    def slug(self) -> str:
        """用于目录名/标签的单行标识。"""
        base = f"{self.variant}-seed{self.seed}"
        return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in base)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "seed": self.seed,
            "out_dir": self.out_dir,
            "params": dict(self.params),
            "extra_args": list(self.extra_args),
        }


class AdapterError(RuntimeError):
    """适配器契约被违反（缺方法、返回类型错误、环境不可用等）。"""


# --------------------------------------------------------------------------- #
# 指标契约
# --------------------------------------------------------------------------- #

#: 指标名 → 逐点序列（epoch / fold / step）。这是 ``parse_results`` 的唯一合法
#: 返回形态。**必须是"序列"而不是"标量"**，理由是机制而非比例：
#:
#: * 学习曲线、收敛判断、逐 epoch 统计、箱线图、指标网格这些产出**在原理上**
#:   需要逐点数据——喂给它一个长度 1 的序列，它们会退化成单点图或直接消失；
#: * 而学习曲线恰恰是判断「实验是否真的收敛」的主要依据，也是审稿人最先看的图。
#:
#: （早期版本在这里写过一个"约八成"的比例。那个数字是目测估计、不是测量结果，
#: 已删除：一个未经测量的百分比会被当成事实引用，而它并不比机制说明更有信息量。）
MetricSeries = dict[str, list[float]]

#: 指标名里出现这些子串即视为「越小越好」。**这是本模块对外的兼容别名**，
#: 真正的规范表在 :mod:`autoresearch.tools.metrics`（``_LOWER_WORDS``）。
#: 之所以不让这里自成一份：副本一旦存在就会漂移，而「同一个指标名在不同阶段被判定为
#: 不同方向」不会报错，只会让「改善」的定义在论文不同章节里悄悄改变。
LOWER_IS_BETTER_TOKENS: tuple[str, ...] = (
    "loss", "error", "rmse", "mae", "mse", "perplexity", "ppl", "nll",
    "latency", "time", "cost", "wer", "cer", "fid", "fdr",
)


def higher_is_better(metric_name: str) -> bool:
    """指标方向的统一判定：越大越好返回 ``True``。

    转调 :func:`autoresearch.tools.metrics.higher_is_better`——全项目共用同一套
    规则，适配器作者与管线内部不会得出不同结论。命名里带语义（``val_loss``、
    ``test_accuracy``）是最省事的做法；特殊指标请用 ``higher_is_better`` 的
    ``overrides`` 参数在 ``tools.metrics.summarize`` 层面声明。
    """
    try:
        from ..tools.metrics import _higher_is_better

        return bool(_higher_is_better(metric_name))
    except Exception:  # pragma: no cover - tools 层不可用时退回本地表
        name = (metric_name or "").lower()
        return not any(token in name for token in LOWER_IS_BETTER_TOKENS)


def coerce_metric_series(raw: Any, source: str = "adapter") -> MetricSeries:
    """把适配器返回的东西规范成 ``MetricSeries``，不合规就抛错。

    宽容的地方：接受标量（包成单元素序列）、接受 ``numpy`` 数组、跳过非数值。
    严格的地方：**键必须是字符串、值必须能变成 float 序列**——静默接受畸形结构
    会让错误一路漂到画图阶段才炸，那时已经很难归因了。
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise AdapterError(
            f"{source}.parse_results() 必须返回 dict[str, list[float]]，"
            f"实际是 {type(raw).__name__}"
        )

    out: MetricSeries = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        values: list[float] = []
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values = [float(value)]
        elif isinstance(value, (list, tuple)):
            for item in value:
                try:
                    values.append(float(item))
                except (TypeError, ValueError):
                    continue
        else:
            # numpy 数组 / pandas Series 等：只要求可迭代且元素可转 float
            try:
                iterator: Iterable[Any] = iter(value)  # type: ignore[arg-type]
            except TypeError:
                continue
            for item in iterator:
                try:
                    values.append(float(item))
                except (TypeError, ValueError):
                    continue
        if values:
            out[name] = values
    return out


def read_standard_metrics(out_dir: Path) -> MetricSeries:
    """读取标准 ``metrics.csv`` / ``metrics.jsonl``（管线内置的默认约定）。

    适配器可以直接复用它——多数训练脚本把指标写成 CSV 就够用了，
    没必要让每个适配器作者重复实现解析。
    """
    merged: MetricSeries = {}
    for filename in ("metrics.csv", "metrics.jsonl", "metrics.json"):
        path = Path(out_dir) / filename
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            from ..tools.metrics import parse_metrics

            parsed = parse_metrics(path) or {}
        except Exception:
            parsed = _fallback_parse(path)
        for key, values in parsed.items():
            if isinstance(values, list) and values:
                merged[str(key)] = [float(v) for v in values]
    return merged


def _fallback_parse(path: Path) -> MetricSeries:
    """不依赖 ``tools.metrics`` 的极简 CSV/JSONL 解析（适配器兜底）。"""
    import json
    import re

    out: MetricSeries = {}
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return out
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return out
    if path.suffix.lower() == ".jsonl":
        for line in lines:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                for key, value in obj.items():
                    try:
                        out.setdefault(str(key), []).append(float(value))
                    except (TypeError, ValueError):
                        continue
        return out
    header = [h.strip() for h in re.split(r"[,\t;]", lines[0])]
    for line in lines[1:]:
        cells = [c.strip() for c in re.split(r"[,\t;]", line)]
        for key, cell in zip(header, cells):
            try:
                out.setdefault(key, []).append(float(cell))
            except (TypeError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------- #
# 适配器协议
# --------------------------------------------------------------------------- #


class BaseExperimentAdapter(ABC):
    """实验后端适配器。

    子类**必须**实现 :meth:`build_command` 与 :meth:`parse_results`；
    其余方法有安全的默认实现，最小可用适配器只需实现这两个。
    """

    #: 适配器标识（CLI ``--experiment-adapter`` 可直接用这个名字）。
    name: str = "base"
    #: 人类可读的一句话说明，会写进交付报告，帮助读者判断结论的适用范围。
    description: str = ""
    #: ``True`` 表示适配器**自带代码**，s4 跳过 LLM 代码生成。
    #: 注意：调试闭环与保真度检查依然生效。
    owns_code: bool = False
    #: 默认入口脚本（相对工作区）。``None`` 表示由 s4 从生成结果里推断。
    entrypoint: str | None = None
    #: 单次实验的默认超时（秒）。``None`` 表示沿用沙箱配置。
    default_timeout: int | None = None
    #: 本适配器单次运行**合理的臂数上限**（主对照 + 消融格点）。
    #:
    #: 存在的理由：s3 会声明一整套消融矩阵，但对代价高的后端（真实训练）一次跑十几臂
    #: 是不现实的；而对廉价的合成任务，跑满也不痛。这个上限让「跑多少臂」成为
    #: 适配器可以表达的能力，而不是管线里的硬编码常量。
    #: ``None`` 表示不额外限制（仍受配置项 ``max_variants`` 约束）。
    max_variants: int | None = None

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    # -- 可选钩子 ------------------------------------------------------- #

    def prepare(self, workspace: Path, plan: dict[str, Any]) -> None:
        """准备实验环境：建目录、写数据清单、生成依赖声明等。

        允许 no-op。**不应**做重活（下载大数据集、编译）——沙箱超时会掐掉它，
        而且失败信息很难归因。需要重资源时请在 ``validate_environment`` 里
        提前把要求说清楚。
        """
        return None

    def seed_code(self, workspace: Path, plan: dict[str, Any]) -> dict[str, str]:
        """提供**模板**源码 ``{相对路径: 内容}``。

        返回空 dict 表示「交给 LLM 生成」。返回非空时，``s4`` 会先把这些文件
        写进工作区，再让 LLM 基于它们做适配与修补——这样既能复用高质量模板，
        又不会绕过调试闭环。
        """
        return {}

    @abstractmethod
    def build_command(self, spec: RunSpec) -> list[str]:
        """构造一次运行的命令。

        返回值是**完整的 argv**（``[程序, 参数...]``），由沙箱按不经 shell 的方式
        执行。第一个元素用 ``"python"`` 表示「当前解释器」，沙箱会替换成
        ``sys.executable``；也可以是 ``"torchrun"``、``"accelerate"``、
        ``"make"`` 或绝对路径。

        禁止返回带 shell 元字符（``|``、``&&``、``>``）的**单条字符串**——
        沙箱不做 shell 展开，那样只会得到「找不到该文件」这种难懂的错误。
        """
        raise NotImplementedError

    @abstractmethod
    def parse_results(self, out_dir: Path) -> MetricSeries:
        """解析一次运行的指标，返回 ``{指标名: 逐点序列}``。

        返回空 dict 表示「本次运行没有可用指标」——这是一种**合法结果**，
        管线会如实记录为「运行完成但无指标」，而不是当成功处理。
        """
        raise NotImplementedError

    def validate_environment(self) -> tuple[bool, str]:
        """检查运行所需的外部条件（解释器、依赖、GPU、数据路径）。

        返回 ``(可用, 原因)``。``s4`` 在**任何运行之前**调用一次；不可用时会
        立即失败并保留原因，而不是让 6 次运行各超时一次。
        """
        return True, ""

    def supported_variants(self) -> set[str] | None:
        """本适配器能跑的变体名集合；``None`` 表示「不限制」。

        为什么需要它：s3 的实验计划会声明任意变体名（``seed=2``、``no_budget``…），
        但一个具体脚本未必认这些名字——``argparse`` 的 ``choices`` 会直接拒绝，
        表现为「实验全挂」，而归因却指向一个与真实原因（计划声明了脚本不支持的变体）
        无关的 argparse 报错。

        声明之后，``s4`` 会在运行前把不支持的变体**过滤掉并如实记录**，
        而不是浪费 6 次运行去撞同一堵墙。
        """
        return None

    def code_files(self, workspace: Path) -> list[Path]:
        """本适配器实际要执行的代码文件（绝对路径）。

        ``owns_code=True`` 的适配器用它在「不经过 LLM」的前提下把自己的脚本交给
        ``s4`` 登记与（只读）检查。默认实现按 ``entrypoint`` 在工作区里查。

        返回空列表表示「代码由 s4 生成，适配器不提供」。
        """
        entry = str(getattr(self, "entrypoint", "") or "").strip()
        if not entry:
            return []
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = Path(workspace) / entry.replace("\\", "/")
        return [candidate] if candidate.is_file() else []

    def describe(self) -> dict[str, Any]:
        """能力声明，写进交付报告与 ``code_manifest.json``。"""
        return {
            "name": self.name,
            "description": self.description or self.__class__.__doc__ or "",
            "owns_code": bool(self.owns_code),
            "entrypoint": self.entrypoint,
            "params": dict(self.params),
            "max_variants": self.max_variants,
            "quality_note": self.quality_note(),
        }

    def quality_note(self) -> str:
        """一句话说明「本适配器的结果能支撑什么级别的结论」。

        默认实现给出保守提示；真实适配器应当覆盖它。这段文字会进入论文的
        Limitations 材料与交付报告——**读者需要知道实验的可信边界**。
        """
        return (
            "该适配器未声明结果适用范围；请人工确认实验是否足以支撑论文 claim"
            "（尤其是数据集代表性、基线强度与种子数量）。"
        )

    # -- 供 s4 构造代码生成提示词 --------------------------------------- #
    def metric_directions(self) -> dict[str, bool] | None:
        """声明本适配器产出的指标方向：``{指标名: 越大越好?}``。

        **领域知识属于适配器，不属于管线。** 管线里的通用词表是 ML 中心的
        （loss/accuracy/f1/bleu…），在别的领域会判错——材料学实测：

            k（消光系数）被判成"越大越好"，而它实际越小越好（越小越透明）
            alpha、resistivity、corrosion_rate、sintering_temp 同样判反
            n、band_gap、thermal_conductivity 等**没有固有方向**，却被静默选了一个

        方向判错**不会报错**：它只会让「改善」的定义反过来，进而在论文里
        把变差写成变好。所以真实适配器应当**显式声明**，哪怕只是声明"无方向"
        （把它从比较里排除）。

        返回 ``None`` 表示"本适配器不声明，交给通用启发式"（默认）。
        返回 ``{}`` 表示"已声明但一个都不需要方向判断"——这两者语义不同：
        前者会走 token 表兜底并留下一条 warning，后者不会。
        """
        return None

    def metric_axis(self) -> str | None:
        """指标序列的物理轴名（若存在）。

        默认 ``None``：管线不假设序列有物理含义。ML 里它是 ``epoch``，
        材料光谱里可能是 ``wavelength_nm``，但**也可能根本没有轴**
        ——例如重复测量（那是 replicates，方向轴没有意义）。
        声明它只影响产物标签与可读性，不改变统计口径。
        """
        return None

    def codegen_conventions(self) -> str:
        """告诉代码生成模型「本适配器期望什么样的脚本」。

        这是把原先硬编码在 s4 里的合成任务约定归还给适配器的关键：
        此前的提示词无条件要求「必须支持 --epochs --seed --variant --out-dir」
        与「metrics.csv 表头固定为 epoch,loss,accuracy,f1,val_loss,val_accuracy」，
        于是一个跑材料仿真的自定义适配器也会收到为小型分类任务写的规范——
        生成的代码自然对不上它真正的指标。
        """
        entry = self.entrypoint or "train.py"
        return (
            f"入口脚本为 {entry}。必须把指标写入 ./metrics.csv 与 ./metrics.jsonl，"
            "并接受 --seed 与 --out-dir 两个参数（其余超参由适配器自行决定）。"
        )

    def codegen_data_info(self) -> str:
        """告诉代码生成模型数据集从哪来（脚本内合成 / 用户提供 / 需外部路径）。"""
        return "数据集由适配器负责准备，脚本不得在运行时下载数据。"

    def codegen_variants(self) -> str:
        """告诉代码生成模型本适配器会传入哪些变体名。"""
        supported = self.supported_variants()
        if supported:
            return "|".join(sorted(supported))
        return "由 s3 的消融矩阵决定（适配器不限制）"


# --------------------------------------------------------------------------- #
# 适配器解析
# --------------------------------------------------------------------------- #

#: 第三方包可通过该 entry point 组注册适配器：
#: ``[project.entry-points."autoresearch.adapters"] my_adapter = "pkg.module:Class"``
ENTRY_POINT_GROUP = "autoresearch.adapters"


def _load_from_file(path: Path) -> type[BaseExperimentAdapter]:
    """从 ``.py`` 文件加载适配器类。

    约定：模块里的 ``ADAPTER`` 变量优先；否则找第一个 ``BaseExperimentAdapter``
    子类。找不到就报错并列出可用名字，而不是返回 None 让调用方猜。
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise AdapterError(f"适配器文件不存在：{path}")

    module_name = f"_autoresearch_adapter_{path.stem}_{abs(hash(str(path.resolve()))) % 10**8}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AdapterError(f"无法从 {path} 构造模块规格（不是合法的 Python 文件？）")
    module = importlib.util.module_from_spec(spec)
    # 让适配器文件能 import 到同目录的辅助模块
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AdapterError(f"导入适配器 {path} 失败：{type(exc).__name__}: {exc}") from exc
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass

    candidate = getattr(module, "ADAPTER", None)
    if isinstance(candidate, type) and issubclass(candidate, BaseExperimentAdapter):
        return candidate
    if isinstance(candidate, BaseExperimentAdapter):
        return type(candidate)

    found: list[type[BaseExperimentAdapter]] = []
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, BaseExperimentAdapter) \
                and value is not BaseExperimentAdapter:
            found.append(value)
    if len(found) == 1:
        return found[0]
    if not found:
        raise AdapterError(
            f"{path} 里没有找到 BaseExperimentAdapter 子类；"
            "请定义 `class MyAdapter(BaseExperimentAdapter)`，"
            "或把类赋给模块级变量 `ADAPTER`。"
        )
    names = ", ".join(sorted(c.__name__ for c in found))
    raise AdapterError(
        f"{path} 里有多个适配器类（{names}）；请用模块级 `ADAPTER = <类名>` 指定用哪个。"
    )


def _load_from_entry_points(name: str) -> type[BaseExperimentAdapter] | None:
    """从已安装包的 entry point 加载。旧 Python 无 ``importlib.metadata`` 时跳过。"""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        group = entry_points()
        selected = group.select(group=ENTRY_POINT_GROUP) if hasattr(group, "select") \
            else group.get(ENTRY_POINT_GROUP, [])  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - 元数据损坏
        return None
    for entry in selected:
        if getattr(entry, "name", "") != name:
            continue
        try:
            loaded = entry.load()
        except Exception as exc:  # pragma: no cover
            raise AdapterError(f"加载 entry point {name} 失败：{exc}") from exc
        if isinstance(loaded, type) and issubclass(loaded, BaseExperimentAdapter):
            return loaded
        if isinstance(loaded, BaseExperimentAdapter):
            return type(loaded)
    return None


def builtin_adapters() -> dict[str, type[BaseExperimentAdapter]]:
    """内置适配器注册表（延迟导入，避免循环依赖）。"""
    from .script_wrapper import ScriptWrapperAdapter
    from .synthetic_toy import SyntheticToyAdapter

    return {
        SyntheticToyAdapter.name: SyntheticToyAdapter,
        ScriptWrapperAdapter.name: ScriptWrapperAdapter,
    }


def resolve_adapter(spec: str | None, params: Mapping[str, Any] | None = None) -> BaseExperimentAdapter:
    """把 CLI 传入的 ``--experiment-adapter`` 解析成适配器实例。

    支持四种写法（按优先级）：

    1. 空 / ``default`` / ``synthetic`` / ``toy`` → 内置合成任务适配器；
    2. ``path/to/adapter.py`` 或 ``pkg.module:Class`` → 从文件/模块加载；
    3. 已安装包的 entry point 名；
    4. 内置适配器名（``synthetic-toy``、``script-wrapper``）。
    """
    params = dict(params or {})
    raw = str(spec or "").strip()

    if not raw or raw.lower() in ("default", "synthetic", "toy", "auto"):
        return builtin_adapters()["synthetic-toy"](params)

    # -- 文件路径 -------------------------------------------------------- #
    looks_like_path = (
        raw.endswith(".py")
        or os.sep in raw
        or "/" in raw
        or (len(raw) > 2 and raw[1] == ":")  # Windows 盘符
    )
    if looks_like_path:
        return _load_from_file(Path(raw))(params)

    # -- module:Class ---------------------------------------------------- #
    if ":" in raw:
        module_name, _, class_name = raw.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise AdapterError(f"无法导入模块 {module_name}：{exc}") from exc
        candidate = getattr(module, class_name, None)
        if candidate is None:
            raise AdapterError(f"模块 {module_name} 里没有 {class_name}")
        if isinstance(candidate, BaseExperimentAdapter):
            return type(candidate)(params)
        if isinstance(candidate, type) and issubclass(candidate, BaseExperimentAdapter):
            return candidate(params)
        raise AdapterError(f"{module_name}:{class_name} 不是 BaseExperimentAdapter 子类")

    # -- entry point ----------------------------------------------------- #
    from_entry_point = _load_from_entry_points(raw)
    if from_entry_point is not None:
        return from_entry_point(params)

    # -- 内置名 ---------------------------------------------------------- #
    registry = builtin_adapters()
    if raw in registry:
        return registry[raw](params)

    available = ", ".join(sorted(registry))
    raise AdapterError(
        f"未知的实验适配器 {raw!r}。可用内置名：{available}；"
        f"也可以传 .py 路径（如 adapters/my_adapter.py）或 module:Class。"
    )


__all__ = [
    "ENTRY_POINT_GROUP",
    "LOWER_IS_BETTER_TOKENS",
    "AdapterError",
    "BaseExperimentAdapter",
    "MetricSeries",
    "RunSpec",
    "builtin_adapters",
    "coerce_metric_series",
    "higher_is_better",
    "read_standard_metrics",
    "resolve_adapter",
]
