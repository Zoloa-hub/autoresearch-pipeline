"""把 Lorenz-63 governance 实验接成适配器（**不改动原实验代码**）。

对应脚本：``sakanaai ctm/gov_naive_update.py``
研究问题：对已冻结的 K=32 faithful 模型做一次「朴素自然漂移更新」，
源域（rho=28）会被保留还是被破坏，同时目标域（rho=30）学到多少。

## 为什么这个脚本值得接

它是**真正意义上的多种子实验**：内部固定跑 5 个模型种子
（42 / 7 / 123 / 2024 / 5555），每个种子给出 pre / post 两组证据，
输出 ``models: {"42": {...}, "7": {...}, ...}``。这正好落在管线
「先在种子上取标量、再跨种子算 mean±std」的统计口径里——
而这个实验**不能**用单种子代表：脚本自己的输出就显示
``delta_abs_lam_ref`` 从 −0.001（seed 5555）到 −0.381（seed 7），差两个数量级。

## 接口摩擦（必须如实记录）

``RunSpec.seed`` 是管线侧必需的（决定输出目录 ``runs/<variant>/seed_<n>``），
但**这个脚本不使用外部 seed**——种子写死在脚本内部。因此：

1. 适配器**忽略** ``spec.seed``，不在命令行上传任何 seed 参数（脚本也不接受）；
2. 一次运行产出全部 5 个种子，适配器把它们拆成 ``metric@seed=<内部种子>``
   的多个序列，所以「跨种子统计」是脚本自己那 5 个种子，不需要管线多跑；
3. 管线的种子循环仍会重复调用本脚本，属于已知接口开销，写进 ``quality_note()``。

## 指标命名与方向

全部来自脚本输出，**没有编造**：

* ``source_ss_mse`` / ``source_attractor_dist`` —— 源域保持，**越小越好**
* ``target_ss_mse`` —— 目标域适应，**越小越好**
* ``source_vpt`` / ``source_lyapunov`` —— **越大越好**（预测时长 / 李雅普诺夫指数）
* ``delta_*`` —— 脚本自己算的变化量

注意 ``vpt`` 在 pre 侧常见为 ``None``（模型没到该阈值就失去预测能力），
适配器**丢弃**这类缺失值而不是填 0——填 0 会造出一个不存在的观测。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

# 双模式导入：作为包的一部分（`from autoresearch.adapters import ...`）时用相对导入；
# 被 `--experiment-adapter <这个文件的路径>` 独立加载时相对导入会失败
# （`attempted relative import with no known parent package`），此时退回绝对导入。
# 这两种用法都必须能用——文档里教的正是「把适配器写成单个 .py 文件」。
try:
    from .base import (
        BaseExperimentAdapter,
        MetricSeries,
        RunSpec,
        coerce_metric_series,
    )
except ImportError:  # pragma: no cover - 取决于加载方式
    from autoresearch.adapters.base import (  # type: ignore[no-redef]
        BaseExperimentAdapter,
        MetricSeries,
        RunSpec,
        coerce_metric_series,
    )

#: 脚本输出 ``models.<seed>.<snapshot>`` 里的字段 → 适配器指标名
_FIELD_MAP: tuple[tuple[str, str, str], ...] = (
    # (输出字段, 适配器指标名, 取哪个快照 pre|post)
    ("ss_mse", "source_ss_mse", "post"),
    ("ast_dist", "source_attractor_dist", "post"),
    ("vpt", "source_vpt", "post"),
    ("lam", "source_lyapunov", "post"),
    ("target_ss", "target_ss_mse", "post"),
)

#: ``deltas`` 里的字段直接搬运（脚本已算好，适配器不重新推导）
_DELTA_MAP: tuple[tuple[str, str], ...] = (
    ("delta_mse", "delta_source_ss_mse"),
    ("delta_vpt", "delta_source_vpt"),
    ("delta_attractor_dist", "delta_source_attractor_dist"),
    ("delta_abs_lam_ref", "delta_source_lyapunov_gap"),
)

#: 脚本硬编码的（模型）种子，用于给输出目录/序列名加后缀。**不用于命令行。**
SCRIPT_SEEDS: tuple[str, ...] = ("42", "7", "123", "2024", "5555")


class LorenzGovernanceAdapter(BaseExperimentAdapter):
    """跑 ``gov_naive_update.py``，把它的 5 种子输出接进管线。"""

    name = "lorenz-governance"
    description = (
        "Lorenz-63 governance：对冻结的 K=32 faithful 模型施加朴素自然漂移更新，"
        "度量源域（rho=28）保持与目标域（rho=30）适应。脚本内部跑 5 个模型种子。"
    )
    #: 用户自带真实实验代码 —— 跳过 LLM 代码生成，但仍走调试闭环与保真度检查。
    owns_code = True
    entrypoint = "gov_naive_update.py"
    default_timeout = 1800
    #: 脚本一次跑完全部 5 个种子；再加臂只是重复同样的计算 —— 压到 1。
    max_variants = 1

    # -- 参数解析 ------------------------------------------------------- #
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        #: 实验代码所在目录（只读取其中的 .py；输入数据/模型也从这里取）
        self.source_dir = Path(str(self.params.get("source_dir") or "")).expanduser()
        #: 预训练模型目录、治理数据目录（相对 source_dir 或绝对路径）
        self.model_dir = str(self.params.get("model_dir") or "faithful_models")
        self.data_dir = str(self.params.get("data_dir") or "lorenz_governance_data")
        self.update_epochs = int(self.params.get("update_epochs") or 100)
        self.target_frac = float(self.params.get("target_frac") or 0.8)
        #: 附加依赖模块（一般**不需要**手工指定：本地依赖由 _local_dep_closure 自动解析）
        self._extra_deps = [str(f) for f in (self.params.get("dep_files") or [])]
        self.results_file = str(self.params.get("results_file") or "gov_results.json")
        self._deps_cache: list[str] | None = None

    # -- 本地依赖解析 --------------------------------------------------- #
    def local_deps(self) -> list[str]:
        """解析入口脚本的**本地依赖闭包**（同目录下的 .py）。

        为什么不写死清单：第一版手工列了 ``train_lorenz_baseline_v2.py`` 与
        ``validate_faithful_chaos.py`` 两个文件，运行时立刻炸在**传递依赖**上——
        ``validate_faithful_chaos`` 还 import 了 ``validate_lorenz_lyapunov``。
        这类失败既慢又难查（要跑一分钟才发现），而且每次上游加一个 import
        就会复发。用 AST 走一遍 import 图，一劳永逸。

        只认**入口脚本同目录下真实存在**的模块：第三方与标准库（numpy/torch/…）
        留给环境，不进闭包。
        """
        if self._deps_cache is not None:
            return self._deps_cache

        import ast

        root = self.source_dir
        seen: set[str] = set()
        order: list[str] = []
        queue = [self.entrypoint, *self._extra_deps]
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            path = root / name
            if not path.is_file():
                seen.add(name)
                continue
            seen.add(name)
            if name != self.entrypoint:
                order.append(name)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                mods: list[str] = []
                if isinstance(node, ast.Import):
                    mods = [alias.name for alias in node.names]
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.level == 0
                ):
                    mods = [node.module]
                for mod in mods:
                    candidate = mod.split(".")[0] + ".py"
                    if (root / candidate).is_file() and candidate not in seen:
                        queue.append(candidate)

        self._deps_cache = order
        return order

    def supported_variants(self) -> set[str] | None:
        return None

    # -- 生命周期 ------------------------------------------------------- #
    def validate_environment(self) -> tuple[bool, str]:
        if not self.source_dir or not self.source_dir.is_dir():
            return False, (
                "需要 adapter-arg source_dir=<gov_naive_update.py 所在目录>；"
                f"当前值 {self.source_dir!r} 不是目录"
            )
        script = self.source_dir / self.entrypoint
        if not script.is_file():
            return False, f"找不到入口脚本 {script}"
        missing = [f for f in self.local_deps() if not (self.source_dir / f).is_file()]
        if missing:
            return False, f"缺少依赖模块: {', '.join(missing)}"
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            return False, f"需要 PyTorch（CPU 版即可）: {exc}"
        md = Path(self.model_dir)
        if not md.is_absolute():
            md = self.source_dir / md
        if not md.is_dir():
            return False, f"预训练模型目录不存在: {md}"
        n_ckpt = len(list(md.glob("k32_seed*_best.pt")))
        if n_ckpt < 2:
            return False, (
                f"模型目录 {md} 只有 {n_ckpt} 个 k32_seed*_best.pt；"
                "多种子统计至少需要 2 个种子"
            )
        return True, f"{n_ckpt} 个模型 checkpoint；数据目录 {self.data_dir}"

    def prepare(self, workspace: Path, plan: dict[str, Any]) -> None:
        """把脚本与依赖模块复制进工作区；输入数据/模型**只读取、不复制**。

        为什么不复制模型与数据：那是用户的实验输入（5 个 checkpoint、3 个 npy），
        复制一份既浪费磁盘，又会让「我跑的是不是他发布的那个版本」变得含糊。
        """
        dest = workspace / "_provided"
        dest.mkdir(parents=True, exist_ok=True)
        for fname in [self.entrypoint, *self.local_deps()]:
            src = self.source_dir / fname
            if src.is_file():
                shutil.copy2(src, dest / fname)

    def seed_code(self, workspace: Path, plan: dict[str, Any]) -> dict[str, str]:
        """``owns_code=True``：不生成代码，仅交付原文供装配清单记录。"""
        script = self.source_dir / self.entrypoint
        return {
            self.entrypoint: (
                script.read_text(encoding="utf-8", errors="replace")
                if script.is_file()
                else ""
            )
        }

    def code_files(self, workspace: Path) -> list[Path]:
        out = [workspace / "_provided" / self.entrypoint]
        out += [workspace / "_provided" / f for f in self.local_deps()]
        return [p for p in out if p.is_file()]

    def build_command(self, spec: RunSpec) -> list[str]:
        """构造命令。

        **不传任何 seed 参数**：种子写死在脚本内部，命令行没有对应开关；
        传一个它不认识的 ``--seed`` 只会让 argparse 直接失败（退出码 2）。
        ``spec.seed`` 仅用于输出目录命名。
        """
        script = f"_provided/{self.entrypoint}"
        out = f"{spec.out_dir}/{self.results_file}"
        argv = [
            "python",
            script,
            "--data",
            self._resolve_input_dir(self.data_dir),
            "--model-dir",
            self._resolve_input_dir(self.model_dir),
            "--update-epochs",
            str(self.update_epochs),
            "--target-frac",
            str(self.target_frac),
            "--out",
            out,
        ]
        argv.extend(str(a) for a in spec.extra_args)
        return argv

    def _resolve_input_dir(self, value: str) -> str:
        p = Path(value)
        if not p.is_absolute():
            p = self.source_dir / p
        return str(p)

    # -- 指标解析 ------------------------------------------------------- #
    def parse_results(self, out_dir: Path) -> MetricSeries:
        """把 ``{models: {seed: {pre/post/deltas}}}`` 摊平成多个种子序列。

        关键决定：**每个脚本内部种子产生一条独立序列**
        （``source_ss_mse@seed=42``、``source_ss_mse@seed=7`` …）。
        这样管线的跨种子统计才真是「先按种子取标量、再跨种子求均值」，
        而不是把 5 个种子的数混进一条序列算方差——后者会把「种子间差异」
        与「同一种子内的噪声」混为一谈，而这里关心的恰恰是种子间差异。
        """
        path = out_dir / self.results_file
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        models = payload.get("models")
        if not isinstance(models, dict):
            return {}

        series: dict[str, list[float]] = {}
        for seed_key, record in models.items():
            if not isinstance(record, dict):
                continue
            label = _seed_label(seed_key)
            for field, metric, snapshot in _FIELD_MAP:
                block = record.get(snapshot)
                if not isinstance(block, dict):
                    continue
                value = _number(block.get(field))
                if value is None:
                    continue  # 缺失值不填 0：宁缺勿造
                series.setdefault(f"{metric}@seed={label}", []).append(value)
            deltas = record.get("deltas")
            if isinstance(deltas, dict):
                for field, metric in _DELTA_MAP:
                    value = _number(deltas.get(field))
                    if value is None:
                        continue
                    series.setdefault(f"{metric}@seed={label}", []).append(value)
        return coerce_metric_series(series)

    # -- 自述 ----------------------------------------------------------- #
    def quality_note(self) -> str:
        return (
            "结果来自 Lorenz-63 governance 实验（gov_naive_update.py），"
            "由适配器原样调用、**未修改其实验代码**。"
            "覆盖面：5 个模型种子（42/7/123/2024/5555），每个种子报告源域保持与目标域适应。"
            "两点边界：（1）只覆盖一条「朴素自然漂移更新」路径，不是完整的风险门控审计；"
            "（2）脚本内部固定 5 个种子、不接受外部 seed，因此管线的多种子循环会重复执行"
            "同一个 5 种子实验并取到同一组数字——不会污染统计，但会浪费时间。"
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "source_dir": str(self.source_dir),
                "entrypoint": self.entrypoint,
                "script_seeds": list(SCRIPT_SEEDS),
                "update_epochs": self.update_epochs,
                "target_frac": self.target_frac,
                "note": "脚本内部固定 5 个模型种子，命令行不接受 seed 参数",
            }
        )
        return info


def _seed_label(value: Any) -> str:
    """种子键规范化（脚本用字符串键，但 JSON 里也可能是数字）。"""
    text = str(value).strip()
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text) or "unknown"


def _number(value: Any) -> float | None:
    """把脚本输出里的数值转成 float；``None`` / ``"NA"`` / 非数值返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None

#: 显式导出，便于用 `--experiment-adapter path/to/lorenz_governance.py` 直接加载。
#: （`_load_from_file` 本来也会找第一个子类，但显式声明意图更清楚。）
ADAPTER = LorenzGovernanceAdapter
