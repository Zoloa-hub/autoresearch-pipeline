r"""材料光学适配器 —— 用真实材料学脚本验证管线的领域通用性。

对应脚本：``comsol_la2ti2o7/ltp_optics.py``
（La2Ti2O7 薄膜光学常数：可见区透明 n≈2.2，深紫外吸收边 ~3.8-4.0 eV，
Kramers-Kronig 一致的 n,k 对。文献锚点：Bayart et al., Optical Materials 92 (2019)）

## 这个适配器要证明什么

它是**第一个非 ML 领域的实测案例**。此前所有实测（合成分类、Lorenz-63）在结构上
都与 ML 实验同构：跑命令 → 写逐 epoch 指标。材料光学不一样：

* **没有 epoch**。自然的自变量是**波长**（连续物理轴）。
* **没有 accuracy/f1**。指标是折射率 n、消光系数 k、吸收系数 α。
* **"越好"取决于物理目标，不由名字决定**。k 越小越透明（好），
  n 则要**靠近设计值**（2.20），靠近才叫好——它没有单调方向。

最后一条正是通用 token 表会判错的地方。实测 19 个材料学指标里
**5 个判反、6 个无方向却被静默选了一个**，合计 13/19 不可信。所以本适配器
显式声明全部方向（``metric_directions()``），并把"靠近设计值"编码成
**偏差量**（偏差越小越好）——这样单调方向判断才成立。

## 指标设计（全部有明确方向，且都是偏差/单向量）

| 指标 | 方向 | 物理含义 |
|---|---|---|
| ``n_deviation_1e3`` | 越小越好 | \|n(550nm) − 2.20\| × 1000，偏离设计折射率 |
| ``k_at_250nm`` | 越小越好 | 深紫外消光系数（吸收边内） |
| ``k_at_550nm`` | 越小越好 | 可见区消光系数（应接近 0） |
| ``alpha_visible_mean`` | 越小越好 | 可见区平均吸收系数 α = 4πk/λ |
| ``transparent_window_nm`` | **越大越好** | k < 0.01 的波长窗口宽度 |

## 序列的语义：重复测量，不是"轨迹"

``metric_axis()`` 返回 ``"replicate"``——因为这里每个指标是**重复测量**：
同一模型参数下加入确定性的制备/测量扰动，得到 N 个独立样本。
于是 ``mean±std`` 表示**不确定度**，这在材料学里是有意义的；
而在 ML 里序列轴是 epoch，mean 表示"训练过程平均"（同样有意义但含义不同）。

这条区别很重要：**管线不该假设序列一定有物理顺序**。材料学里
"把不同波长的 n 求平均"是没有物理意义的，所以我们不在波长轴上取序列，
而是在**重复**轴上取——波长只用来挑选有物理含义的代表性条件。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .sweep import SweepAxis

try:
    from .base import BaseExperimentAdapter, MetricSeries, RunSpec, coerce_metric_series
except ImportError:  # pragma: no cover - 支持用 .py 路径独立加载
    from autoresearch.adapters.base import (  # type: ignore[no-redef]
        BaseExperimentAdapter,
        MetricSeries,
        RunSpec,
        coerce_metric_series,
    )

#: 目标折射率（可见区设计值）。偏离越小越好。
N_TARGET = 2.20

#: 本适配器产出的全部指标方向。**领域知识在这里，不在管线的 token 表里。**
DIRECTIONS: dict[str, bool] = {
    "n_deviation_1e3": False,      # 偏离设计值 -> 越小越好
    "k_at_250nm": False,           # 消光系数 -> 越小越透明
    "k_at_550nm": False,
    "alpha_visible_mean": False,   # 吸收系数 -> 越小越好
    "transparent_window_nm": True,  # 透明窗口 -> 越大越好
}

#: 默认扫描轴 —— 深紫外吸收边的三个物理参数。
#:
#: ``target`` 是**实测出来的**，不是猜的：``ltp_optics.py`` 把 ``OSC_F/OSC_E0/OSC_G``
#: 绑成 ``eps_lorentz`` 的默认参数，直接 ``setattr`` 模块常量**不会生效**
#: （实测确认），必须重建该函数。而 ``N_VIS`` 在 ``nk_table`` 函数体内被读取，
#: 直接 setattr 即可。
DEFAULT_SWEEP_AXES = (
    SweepAxis("osc_g", (0.35, 0.55, 0.75), "eV", target="eps_lorentz:g"),
    SweepAxis("osc_f", (3.30, 4.10, 4.90), "eV^2", target="eps_lorentz:f"),
    SweepAxis("n_vis", (2.10, 2.20, 2.30), "", target="N_VIS"),
)

#: 计算这些指标需要材料模块提供什么
REQUIRED_API = ("nk_table",)

#: 驱动脚本模板。Adapter 提供模板（seed_code），LLM 在跑不通时可以修补。
RUNNER_TEMPLATE = r'''#!/usr/bin/env python3
"""驱动材料光学模块并算出指标 —— 由 materials-optics 适配器生成。

**不要改这里的物理模型**：参数与公式来自目标模块本身，本脚本只负责
"调用它、在多个代表性条件下取值、写到文件"。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

N_TARGET = 2.20
VISIBLE_NM = (400.0, 700.0)
UV_NM = 250.0
VIS_NM = 550.0


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("materials_target", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load materials module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nk(mod, lam_um):
    """取 n,k。优先用 nk_table()（可一次算全网格）。"""
    lam, n, k = mod.nk_table(np.asarray(lam_um, dtype=float))
    return np.asarray(n, dtype=float), np.asarray(k, dtype=float)


def nk_at_wavelength(mod, lam_nm):
    """在给定波长（nm）取一个 (n, k)。用细网格插值，避免依赖模块内部的接口细节。"""
    lo, hi = lam_nm / 1000.0 * 0.98, lam_nm / 1000.0 * 1.02
    grid = np.linspace(lo, hi, 33)
    n, k = nk(mod, grid)
    return float(np.mean(n)), float(np.mean(k))


def compute_metrics(mod, target_frac: float, jitter: float, seed: int):
    """算一组指标。jitter 模拟制备/测量扰动（确定性，只由 seed 决定）。"""
    rng = random.Random(seed)
    # 参数扰动：材料学里 n,k 会因厚度/结晶度/表面粗糙度轻微漂移
    scale_n = 1.0 + jitter * rng.gauss(0.0, 1.0)
    scale_k = 1.0 + jitter * rng.gauss(0.0, 1.0)

    n_vis, k_vis = nk_at_wavelength(mod, VIS_NM)
    n_vis *= scale_n
    k_vis = max(0.0, k_vis * scale_k)

    _, k_uv = nk_at_wavelength(mod, UV_NM)
    k_uv = max(0.0, k_uv * scale_k)

    # 可见区平均吸收系数 alpha = 4*pi*k/lambda
    grid_nm = np.linspace(VISIBLE_NM[0], VISIBLE_NM[1], 61)
    _, k_grid = nk(mod, grid_nm / 1000.0)
    k_grid = np.maximum(0.0, k_grid * scale_k)
    alpha = 4.0 * math.pi * k_grid / (grid_nm * 1e-9)
    alpha_mean = float(np.mean(alpha)) if alpha.size else 0.0

    # 透明窗口：k < 0.01 的连续波长宽度（用宽网格扫 200-1500 nm）
    wide_nm = np.linspace(200.0, 1500.0, 651)
    _, k_wide = nk(mod, wide_nm / 1000.0)
    k_wide = np.maximum(0.0, k_wide * scale_k)
    transparent = k_wide < 0.01
    window = float(np.sum(transparent) * (wide_nm[1] - wide_nm[0]))

    return {
        "n_deviation_1e3": abs(n_vis - N_TARGET) * 1000.0,
        "k_at_250nm": k_uv,
        "k_at_550nm": k_vis,
        "alpha_visible_mean": alpha_mean,
        "transparent_window_nm": window,
    }


def apply_override(mod, target: str, value) -> str:
    """按 target 把参数写进被驱动模块，返回实际应用方式的说明。

    ``target`` 两种形态：

    * ``"N_VIS"``          —— 普通模块属性，直接 ``setattr``
    * ``"eps_lorentz:g"``  —— 该参数是函数 ``eps_lorentz`` 的**默认参数**。
      这种绑定在函数定义时就固定了，``setattr`` 模块常量**没有效果**
      （在真实模块上实测确认），必须重建函数再替换回模块。

    重建用 ``sig.bind_partial`` 而不是简单的 ``kwargs.setdefault``：
    后者在调用方**按位置**传了该参数时会被忽略，于是覆盖静默失效。
    """
    import functools
    import inspect

    if ":" not in target:
        if not hasattr(mod, target):
            raise SystemExit(f"OVERRIDE_TARGET_MISSING: 模块没有属性 {target!r}")
        setattr(mod, target, value)
        return f"setattr({target})"

    fname, pname = target.split(":", 1)
    fn = getattr(mod, fname, None)
    if fn is None or not callable(fn):
        raise SystemExit(f"OVERRIDE_TARGET_MISSING: 模块没有可调用对象 {fname!r}")
    sig = inspect.signature(fn)
    if pname not in sig.parameters:
        raise SystemExit(f"OVERRIDE_TARGET_MISSING: {fname}() 没有参数 {pname!r}")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.arguments[pname] = value
        bound.apply_defaults()
        return fn(*bound.args, **bound.kwargs)

    setattr(mod, fname, wrapper)
    return f"rebind({fname}:{pname})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="materials optics metric runner")
    ap.add_argument("--module", required=True, help="材料光学模块的 .py 路径")
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--jitter", type=float, default=0.004,
                    help="制备/测量扰动幅度（相对）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="TARGET=VALUE",
                    help="覆盖被驱动模块的参数，如 N_VIS=2.3 或 eps_lorentz:g=0.35")
    ap.add_argument("--out-dir", required=True)
    args, _unknown = ap.parse_known_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_module(Path(args.module))

    # --- 覆盖参数，并**验证真的生效** -------------------------------- #
    # 覆盖失效是静默的：所有格点算出同一组数值 -> 效应全为 0 ->
    # 结论变成「这些参数都不重要」。这是一份看起来完全正常的错误结论，
    # 所以这里主动检测并报错，而不是交出去。
    baseline_probe = None
    applied: list[str] = []
    if args.sets:
        # 覆盖前的参考值（用同一套判据，避免模块自带随机性造成假阳性）
        base_mod_probe = compute_metrics(mod, 0.8, args.jitter, args.seed * 1000)
        for item in args.sets:
            if "=" not in item:
                raise SystemExit(f"OVERRIDE_BAD_SYNTAX: 需要 TARGET=VALUE，得到 {item!r}")
            target, raw_value = item.split("=", 1)
            try:
                value: object = float(raw_value)
            except ValueError:
                value = raw_value
            how = apply_override(mod, target.strip(), value)
            applied.append(f"{target.strip()}={raw_value}({how})")
        after = compute_metrics(mod, 0.8, args.jitter, args.seed * 1000)
        same = all(
            abs(after[k] - base_mod_probe[k]) <= 1e-12 * max(1.0, abs(base_mod_probe[k]))
            for k in base_mod_probe
        )
        if same:
            sys.stderr.write(
                "OVERRIDE_NO_EFFECT: 参数覆盖没有改变任何指标 —— 说明覆盖方式与被驱动"
                "模块的绑定方式不匹配。\n"
                f"  已尝试: {applied}\n"
                "  常见原因: 该参数是函数默认值（定义时已绑定），必须写成 "
                "TARGET='函数名:参数名'。\n"
                "  继续跑下去会得到一条平坦的效应曲线，并被误读成"
                "「这些参数都不重要」，因此这里直接失败。\n"
            )
            return 3
        baseline_probe = base_mod_probe
        print(f"overrides applied: {'; '.join(applied)}", flush=True)

    rows = []
    for rep in range(args.replicates):
        m = compute_metrics(mod, 0.8, args.jitter, args.seed * 1000 + rep)
        m["replicate"] = rep + 1
        rows.append(m)
        print(f"replicate={rep+1} " + " ".join(f"{k}={v:.6g}" for k, v in m.items() if k != "replicate"),
              flush=True)

    fields = ["replicate", "n_deviation_1e3", "k_at_250nm", "k_at_550nm",
              "alpha_visible_mean", "transparent_window_nm"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"variant": args.variant, "replicates": len(rows),
                   "jitter": args.jitter, "axis": "replicate", "rows": rows},
                  fh, indent=2, ensure_ascii=False)
    print(f"wrote {out_dir/'metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class MaterialsOpticsAdapter(BaseExperimentAdapter):
    """驱动材料光学模块，产出有明确物理方向的指标。"""

    name = "materials-optics"
    description = (
        "材料光学常数：驱动一个提供 nk_table() 的模块，在代表性波长上取 n/k，"
        "按重复测量给出不确定度。指标方向由适配器声明，不依赖通用词表。"
    )
    #: 适配器提供驱动脚本模板；LLM 可在跑不通时修补它（这是刻意的分工：
    #: 物理模型属于用户模块，胶水代码可以生成）。
    owns_code = False
    entrypoint = "materials_run.py"
    default_timeout = 900
    max_variants = 2

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.module_path = Path(str(self.params.get("module") or "")).expanduser()
        self.replicates = int(self.params.get("replicates") or 5)
        self.jitter = float(self.params.get("jitter") or 0.004)

    # -- 生命周期 ------------------------------------------------------- #
    def validate_environment(self) -> tuple[bool, str]:
        if not self.module_path or not self.module_path.is_file():
            return False, (
                "需要 adapter-arg module=<材料光学模块的 .py 路径>（必须提供 "
                f"nk_table()）；当前值 {str(self.module_path)!r} 不是文件"
            )
        # **先报配置错误，再报环境缺口。**
        #
        # 顺序有实际影响：module 没给、或模块 API 不对，是**使用者要改的东西**；
        # 缺 numpy 是环境事实。若把 numpy 检查放前面，一个"忘了传 module"的用户
        # 会先看到"需要 numpy"，装完 numpy 才发现真正的问题在参数上——
        # 两步才能定位一个一步就能说清的错误。
        text = self.module_path.read_text(encoding="utf-8", errors="replace")
        missing = [api for api in REQUIRED_API if f"def {api}" not in text]
        if missing:
            return False, (
                f"模块 {self.module_path.name} 缺少必需的 API: {', '.join(missing)}"
                "（本适配器约定：模块需提供 nk_table(lam_um) -> (lam, n, k)）"
            )
        try:
            import numpy  # noqa: F401
        except ImportError as exc:
            return False, f"材料光学计算需要 numpy（可选依赖 [figures]）: {exc}"
        return True, f"{self.module_path.name}；{self.replicates} 次重复测量"

    def seed_code(self, workspace: Path, plan: dict[str, Any]) -> dict[str, str]:
        return {self.entrypoint: RUNNER_TEMPLATE}

    def prepare(self, workspace: Path, plan: dict[str, Any]) -> None:
        # `s4` 会调用 seed_code() 并落盘；这里只保证目录存在。
        (workspace / "_provided").mkdir(parents=True, exist_ok=True)

    def code_files(self, workspace: Path) -> list[Path]:
        return [p for p in (workspace / self.entrypoint,) if p.is_file()]

    def build_command(self, spec: RunSpec) -> list[str]:
        argv = [
            "python",
            self.entrypoint,
            "--module",
            str(self.module_path),
            "--variant",
            spec.variant,
            "--replicates",
            str(self.replicates),
            "--jitter",
            str(self.jitter),
            "--seed",
            str(spec.seed),
            "--out-dir",
            spec.out_dir,
        ]
        # 把扫描格点的参数交给 runner。**用适配器声明的 target**，
        # 而不是假设「参数名就是模块属性名」——见 DEFAULT_SWEEP_AXES 的说明：
        # OSC_* 是函数默认参数（定义时已绑定），按属性名覆盖会静默失效。
        for axis in (self.sweep_axes() or ()):
            if axis.name in (spec.params or {}):
                argv += ["--set", f"{axis.target or axis.name}={spec.params[axis.name]}"]
        argv.extend(str(a) for a in spec.extra_args)
        return argv

    def parse_results(self, out_dir: Path) -> MetricSeries:
        """每个指标一条序列，序列轴是**重复测量**（见模块 docstring）。

        为什么不在波长轴上取序列：把不同波长的 n 求平均没有物理意义。
        代表性条件（550nm、250nm、可见区均值）把物理含义固定住，
        重复测量提供不确定度——这样 mean±std 才是材料学里成立的口径。
        """
        path = out_dir / "metrics.csv"
        if not path.is_file():
            return {}
        try:
            import csv as _csv

            with path.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(_csv.DictReader(fh))
        except Exception:
            return {}
        series: dict[str, list[float]] = {}
        for row in rows:
            for key, value in row.items():
                if key in (None, "replicate"):
                    continue
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    continue
                series.setdefault(str(key), []).append(num)
        return coerce_metric_series(series)

    # -- 参数扫描 ------------------------------------------------------- #
    #: 一次扫描最多跑多少格点（可由 AUTORESEARCH_MAX_SWEEP_RUNS 与配置收紧）
    max_sweep_runs = 12
    #: 默认 OFAT —— 3 条轴各 3 水平 => 1+2+2+2 = 7 组。
    #: 改 "grid" 是 27 组（3³），能测交互效应但要确认预算。
    default_sweep_mode = "ofat"

    def sweep_axes(self) -> tuple[SweepAxis, ...] | None:
        """默认扫描深紫外吸收边的三条参数轴。

        可用 ``adapter-arg sweeps=osc_g:0.3|0.5|0.7;n_vis:2.1|2.3`` 覆盖，
        或 ``sweep_mode=grid`` 切换设计。
        """
        raw = self.params.get("sweeps")
        if raw:
            parsed = _parse_sweep_spec(str(raw))
            if parsed:
                return parsed
        if self.params.get("no_sweep"):
            return None
        return DEFAULT_SWEEP_AXES

    def sweep_mode(self) -> str:
        mode = str(self.params.get("sweep_mode") or self.default_sweep_mode)
        return mode if mode in ("ofat", "grid") else self.default_sweep_mode

    # -- 领域知识：方向与轴 ---------------------------------------------- #
    def metric_directions(self) -> dict[str, bool] | None:
        """声明全部指标方向 —— 这是本适配器存在的核心理由。

        不声明的话，通用词表会：把 k/alpha 判成"越大越好"（**判反**），
        把 n_deviation/transparent_window 也猜错。实测同一份数据下
        supports_claim 会从 True 翻成 False。
        """
        return dict(DIRECTIONS)

    def metric_axis(self) -> str | None:
        return "replicate"

    # -- 自述 ----------------------------------------------------------- #
    def quality_note(self) -> str:
        return (
            "结果来自材料光学模型（用户提供的模块），由适配器原样调用。"
            "指标是**重复测量**（确定性扰动的多个样本），因此 mean±std 表示不确定度；"
            "代表性条件固定在 550nm / 250nm / 可见区均值，"
            "**没有**在波长轴上做统计（那在物理上没有意义）。"
            "边界：本适配器只覆盖光学常数这一类表征，不代表材料的力学/热学/电学性质；"
            "扰动幅度 jitter 是人为设定的，真实不确定度需要来自实验重复。"
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "module": str(self.module_path),
                "replicates": self.replicates,
                "jitter": self.jitter,
                "metric_directions": dict(DIRECTIONS),
                "metric_axis": self.metric_axis(),
                "note": "非 ML 领域：序列轴是重复测量，不是 epoch",
            }
        )
        return info


#: 支持用 `--experiment-adapter path/to/materials_optics.py` 直接加载
ADAPTER = MaterialsOpticsAdapter


def _parse_sweep_spec(raw: str) -> tuple[SweepAxis, ...]:
    """解析 ``"osc_g:0.3|0.5|0.7;n_vis:2.1|2.3"`` 形式的扫描规格。

    已知轴名会继承其 ``unit`` 与 ``target``（覆盖机制），未知轴名则按
    ``target=同名属性`` 处理——后者对"参数是函数默认值"的模块会静默失效，
    所以 runner 侧还有一道生效验证兜底。
    """
    axes: list[SweepAxis] = []
    known = {a.name: a for a in DEFAULT_SWEEP_AXES}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, values_raw = chunk.split(":", 1)
        name = name.strip()
        values: list[Any] = []
        for token in values_raw.split("|"):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(
                    float(token) if ("." in token or "e" in token.lower()) else int(token)
                )
            except ValueError:
                values.append(token)
        if len(values) < 2:
            continue
        src = known.get(name)
        axes.append(
            SweepAxis(
                name,
                tuple(values),
                src.unit if src else "",
                target=src.target if src else name,
            )
        )
    return tuple(axes)
