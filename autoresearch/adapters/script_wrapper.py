"""用户自带脚本适配器：把「我已经有训练脚本」接进管线。

这是 :class:`~autoresearch.adapters.base.BaseExperimentAdapter` 文档里说的
**受控特例**：``owns_code = True`` 让 ``s4`` 跳过 LLM 代码生成——用户明确要求
「别让模型改写我的训练脚本」，这个要求必须被尊重。

但受控两个字是有实质意义的：

* **调试闭环仍然生效**。运行失败时依旧捕获 traceback、反思、打最小补丁。
  只不过补丁**只针对运行环境层面的问题**（缺依赖、参数名不符、路径错误），
  而不是重写实验逻辑。
* **保真度检查仍然生效**。如果补丁删掉了指标输出、加了 ``except: pass``、
  或写死高分返回值，一样会被标记 ``validity_flag`` 并进入交付报告的未解决清单。
  「用户的脚本」不是绕开质量门禁的理由——被污染的结果进了论文，署名的是使用者。

用法（CLI）::

    python -m autoresearch.cli run --direction "..." \
        --experiment-adapter my_train.py \
        --adapter-arg program=python \
        --adapter-arg epochs=10

或在自己的适配器文件里继承本类并覆盖参数。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .base import (
    AdapterError,
    BaseExperimentAdapter,
    MetricSeries,
    RunSpec,
    coerce_metric_series,
    read_standard_metrics,
)

#: 默认参数模板。占位符在 :meth:`ScriptWrapperAdapter.build_command` 里替换。
#: ``{python}`` 由沙箱替换为当前解释器（保证与管线同一环境）。
DEFAULT_ARGS_TEMPLATE: tuple[str, ...] = (
    "{script}",
    "--variant", "{variant}",
    "--seed", "{seed}",
    "--out-dir", "{out_dir}",
)

_PLACEHOLDERS = ("script", "variant", "seed", "out_dir", "python", "workspace")


class ScriptWrapperAdapter(BaseExperimentAdapter):
    """包装一个用户提供的训练/评估脚本。

    通过 ``params`` 配置（CLI 用 ``--adapter-arg key=value`` 传入）：

    ================  ==========================================================
    参数              说明
    ================  ==========================================================
    ``script``        脚本路径（必需）。相对路径按工作区解析。
    ``own_code``      置 true 时**不复制**脚本、直接原地使用（默认复制进工作区）
    ``program``       可执行程序名或绝对路径，默认 ``python``
    ``args_template`` 参数模板，逗号分隔；默认见 ``DEFAULT_ARGS_TEMPLATE``
    ``metrics_file``  指标文件名，默认 ``metrics.csv``
    ``metrics_format````csv`` / ``jsonl`` / ``json``，默认按扩展名推断
    ``metrics_map``   列名重映射，如 ``{"val_acc":"val_accuracy"}``
    ``assets``        额外随脚本带入工作区的文件，``{"相对路径": "源路径"}``
    ``timeout``       单次运行超时（秒）
    ================  ==========================================================
    """

    name = "script-wrapper"
    description = "包装用户自带的训练/评估脚本（不做代码生成，只负责跑起来与取指标）。"
    owns_code = True
    entrypoint = None
    default_timeout = None
    #: 用户脚本通常跑得慢（真实数据集 / GPU），默认只跑主对照，不自动展开消融。
    #: 想跑消融请在子类里抬高这个值，或设 --max-variants。
    max_variants = 2

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.script_spec = str(self.params.get("script") or "").strip()
        if not self.script_spec:
            raise AdapterError(
                "script-wrapper 适配器需要 --adapter-arg script=<训练脚本路径>。"
                "例如：--experiment-adapter script-wrapper --adapter-arg script=train.py"
            )
        self.program = str(self.params.get("program") or "python").strip() or "python"
        self.copy_into_workspace = not _as_bool(self.params.get("own_code"), default=False)
        self.metrics_file = str(self.params.get("metrics_file") or "metrics.csv").strip()
        self.metrics_format = str(self.params.get("metrics_format") or "").strip().lower()
        raw_map = self.params.get("metrics_map")
        self.metrics_map: dict[str, str] = (
            {str(k): str(v) for k, v in raw_map.items()} if isinstance(raw_map, dict) else {}
        )
        raw_assets = self.params.get("assets")
        self.assets: dict[str, str] = (
            {str(k): str(v) for k, v in raw_assets.items()} if isinstance(raw_assets, dict) else {}
        )
        self.timeout = int(self.params.get("timeout") or 0) or None
        self.args_template = _parse_template(self.params.get("args_template"))
        #: 脚本进入工作区后的相对路径（``prepare`` 里确定），正斜杠形式。
        self._staged: str = ""
        #: 对应的绝对路径，供 ``code_files()`` 返回（``entrypoint`` 是给命令行用的
        #: 相对路径，不能直接拿去 ``is_file()`` 判断——工作目录不同就会误判）。
        self._staged_abs: Path | None = None

    # -- 钩子 ----------------------------------------------------------- #

    def prepare(self, workspace: Path, plan: dict[str, Any]) -> None:
        """把用户脚本（及其 assets）带进工作区。

        为什么要复制而不是原地引用：沙箱的工作目录就是工作区，原地引用会让
        「脚本相对路径」与「工作目录」脱钩，产生一类很难诊断的 ``FileNotFound``。
        复制一份让路径关系变成确定的。若用户明确要求 ``own_code=true``
        （例如脚本必须在自己的仓库里才能找到配置），则尊重原路径。
        """
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        source = self._resolve_source(workspace)

        if self.copy_into_workspace:
            target_dir = workspace / "_provided"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            try:
                shutil.copy2(source, target)
            except OSError as exc:
                raise AdapterError(f"复制脚本 {source} 到工作区失败：{exc}") from exc
            # 统一用正斜杠：这个字符串会进命令行，而反斜杠在 POSIX 上是转义字符；
            # 同时它也是产物清单里的键，跨平台保持一致才能让清单可比。
            self._staged = f"_provided/{source.name}"
            self._staged_abs = target.resolve()
            for rel, src in self.assets.items():
                src_path = Path(src)
                if not src_path.is_absolute():
                    src_path = source.parent / src_path
                dest = workspace / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src_path, dest)
                except OSError as exc:
                    raise AdapterError(f"复制 asset {src_path} → {rel} 失败：{exc}") from exc
        else:
            self._staged = source.as_posix()
            self._staged_abs = source.resolve()

        self.entrypoint = self._staged

    def seed_code(self, workspace: Path, plan: dict[str, Any]) -> dict[str, str]:
        """自带脚本时返回空——``s4`` 会依据 ``owns_code`` 跳过代码生成。

        这里刻意**不**把脚本内容读出来返回：那样 s4 会把它当成「模板」再交给
        LLM 改写，正是本适配器要避免的行为。
        """
        return {}

    def code_files(self, workspace: Path) -> list[Path]:
        """返回已带入工作区的脚本（绝对路径）。"""
        if self._staged_abs is not None and self._staged_abs.is_file():
            return [self._staged_abs]
        source = Path(self.script_spec).expanduser()
        return [source.resolve()] if source.is_file() else []

    def validate_environment(self) -> tuple[bool, str]:
        source = Path(self.script_spec).expanduser()
        if not source.is_absolute():
            # 相对路径在 prepare 时才最终确定；这里只做存在性粗查
            if not source.exists():
                return False, (
                    f"找不到训练脚本 {self.script_spec!r}（相对当前目录解析）。"
                    "请传绝对路径，或先 cd 到脚本所在目录再运行。"
                )
        elif not source.is_file():
            return False, f"训练脚本不是普通文件：{source}"

        program = self.program
        if program not in ("python", "python3", sys_python_alias()):
            candidate = shutil.which(program)
            if candidate is None and not Path(program).is_file():
                return False, (
                    f"找不到可执行程序 {program!r}。"
                    "请确认它在 PATH 中，或传绝对路径。"
                )
        return True, ""

    def build_command(self, spec: RunSpec) -> list[str]:
        if not self._staged:
            # 允许调用方跳过 prepare（例如单测）——退化到原始路径
            self._staged = self.script_spec
        values = {
            "script": self._staged,
            "variant": spec.variant,
            "seed": str(spec.seed),
            "out_dir": spec.out_dir,
            "python": "python",
            "workspace": ".",
        }
        argv = [self.program]
        for token in self.args_template:
            rendered = token
            for key in _PLACEHOLDERS:
                rendered = rendered.replace("{" + key + "}", values[key])
            # 未展开的占位符（如 {lr}）从 params 取；取不到就原样保留并让它显式报错，
            # 而不是静默丢掉参数——静默丢参会导致"实验跑了但设置不对"。
            if "{" in rendered and "}" in rendered:
                key = rendered.strip("{}").strip()
                if key in spec.params:
                    rendered = str(spec.params[key])
                else:
                    raise AdapterError(
                        f"参数模板里的占位符 {{{key}}} 无法解析：既不是内置占位符 "
                        f"({', '.join(_PLACEHOLDERS)})，也不在 params 里 "
                        f"({', '.join(sorted(map(str, spec.params))) or '空'})。"
                    )
            if rendered:
                argv.append(rendered)
        argv.extend(spec.extra_args)
        return argv

    def parse_results(self, out_dir: Path) -> MetricSeries:
        out_dir = Path(out_dir)
        if self.metrics_file and self.metrics_file != "metrics.csv":
            parsed = _parse_single_file(out_dir / self.metrics_file, self.metrics_format)
        else:
            parsed = read_standard_metrics(out_dir)
        if not parsed and self.metrics_file:
            # 声明的指标文件不存在时，退回标准约定再试一次——多数脚本其实写的是
            # metrics.csv，用户只是没意识到自己是标准格式。
            parsed = read_standard_metrics(out_dir)
        if self.metrics_map:
            remapped: MetricSeries = {}
            for key, values in parsed.items():
                remapped[self.metrics_map.get(key, key)] = values
            parsed = remapped
        return coerce_metric_series(parsed, source=f"ScriptWrapperAdapter({self.script_spec})")

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "script": self.script_spec,
                "staged_as": self._staged,
                "program": self.program,
                "metrics_file": self.metrics_file,
                "metrics_map": dict(self.metrics_map),
            }
        )
        return info

    def quality_note(self) -> str:
        return (
            f"结果来自用户提供的脚本 `{self.script_spec}`。适配器只负责执行与取指标，"
            "**不对实验设计负责**：数据集代表性、基线强度、种子数量、指标选择是否"
            "足以支撑论文 claim，需要使用者自行确认。管线的保真度检查只能发现"
            "「指标输出被补丁破坏」这类问题，无法发现实验设计本身的缺陷。"
        )

    def codegen_conventions(self) -> str:
        """用户脚本自带代码（``owns_code=True``），这里的规范只在生成辅助代码时用。"""
        return (
            f"代码由用户提供（{self.script_spec}），**不得重写或替换该脚本的实验逻辑**。"
            f"若需要新增辅助文件，只能放在工作区的 _generated/ 下，"
            f"并把指标写到 {self.metrics_file}。"
        )

    def codegen_data_info(self) -> str:
        return (
            "数据集与依赖由用户脚本自行负责；"
            "不得假设数据是管线内合成的，也不得在运行时下载数据。"
        )

    # -- 内部 ----------------------------------------------------------- #

    def _resolve_source(self, workspace: Path) -> Path:
        source = Path(self.script_spec).expanduser()
        if not source.is_absolute():
            # 相对路径优先按「调用者的当前目录」解析（符合直觉），
            # 找不到再试工作区（适配器文件与工作区同源的情况）。
            if source.is_file():
                return source.resolve()
            alt = workspace / source
            if alt.is_file():
                return alt.resolve()
            raise AdapterError(
                f"找不到训练脚本 {self.script_spec!r}；已尝试 {source.resolve()} 与 {alt}"
            )
        if not source.is_file():
            raise AdapterError(f"训练脚本不存在：{source}")
        return source


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def sys_python_alias() -> str:
    import sys

    return Path(sys.executable).name


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_template(raw: Any) -> list[str]:
    """把 ``args_template`` 解析成 token 列表。

    接受三种写法：list、逗号分隔字符串、空格分隔的 JSON 数组字符串。
    """
    if raw is None or raw == "":
        return list(DEFAULT_ARGS_TEMPLATE)
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x) != ""]
    text = str(raw).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x) != ""]
        except ValueError:
            pass
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p] or list(DEFAULT_ARGS_TEMPLATE)


def _parse_single_file(path: Path, fmt: str) -> MetricSeries:
    """解析用户声明的指标文件（csv / jsonl / json）。"""
    from .base import _fallback_parse  # 局部导入：内部辅助

    path = Path(path)
    if not path.is_file():
        return {}
    suffix = (fmt or path.suffix.lstrip(".")).lower()
    if suffix == "json" and path.suffix.lower() == ".json":
        try:
            # utf-8-sig：PowerShell 的 `Set-Content -Encoding utf8` 默认写 BOM，
            # 而 json.loads 会直接拒绝带 BOM 的文本。实测同一条路径下
            # tools.metrics.parse_metrics 能读出指标、这里却返回 {}——这种
            # 「同一个文件两处解析结果不同」的 bug 极难归因，所以吸收掉它。
            payload = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        except (OSError, ValueError):
            return {}
        return coerce_metric_series(_json_to_series(payload), source=str(path))
    return _fallback_parse(path)


def _json_to_series(payload: Any) -> MetricSeries:
    """把常见 JSON 指标结构拍成 ``{指标: 序列}``。

    支持 ``{"acc": 0.9}``（标量）、``{"acc": [..]}``、``[{"acc": ..}, ..]``、
    ``{"metrics": {...}}``。
    """
    if isinstance(payload, dict):
        for wrapper in ("metrics", "results", "summary"):
            inner = payload.get(wrapper)
            if isinstance(inner, (dict, list)):
                return _json_to_series(inner)
        out: MetricSeries = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[str(key)] = [float(value)]
            elif isinstance(value, list):
                nums = []
                for item in value:
                    try:
                        nums.append(float(item))
                    except (TypeError, ValueError):
                        continue
                if nums:
                    out[str(key)] = nums
        return out
    if isinstance(payload, list):
        out: MetricSeries = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                try:
                    out.setdefault(str(key), []).append(float(value))
                except (TypeError, ValueError):
                    continue
        return out
    return {}


__all__ = [
    "DEFAULT_ARGS_TEMPLATE",
    "ScriptWrapperAdapter",
    "sys_python_alias",
]
