"""补齐环境变量并接 s4：扫描模式下的变体命名与参数传递。"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- 1) 环境变量 --
CFG = ROOT / "autoresearch" / "config.py"
c = CFG.read_text(encoding="utf-8")
if "MAX_SWEEP_RUNS" not in c:
    old = '    "MAX_VARIANTS": ("max_variants",),'
    new = (
        '    "MAX_VARIANTS": ("max_variants",),\n'
        '    #: 参数扫描的运行数预算。**与 MAX_VARIANTS 分开**：前者是"跑几个对照臂"，\n'
        '    #: 后者是"扫多少个参数格点"，两者的合理量级不同（2-6 vs 6-60）。\n'
        '    "MAX_SWEEP_RUNS": ("max_sweep_runs",),'
    )
    print("  config ENV_KEYS anchor:", old in c)
    c = c.replace(old, new, 1)
    CFG.write_text(c, encoding="utf-8")

# ---------------------------------------------------------------- 2) s4 --
S4 = ROOT / "autoresearch" / "stages" / "s4_experiment.py"
t = S4.read_text(encoding="utf-8")

SWEEP_HELPERS = '''

# --------------------------------------------------------------------------- #
# 多参数扫描
# --------------------------------------------------------------------------- #
def _sweep_plan(adapter: Any, max_runs: int | None) -> Any:
    """按适配器声明的轴展开扫描方案。返回 ``None`` 表示本适配器不做扫描。

    **纯函数**：``_plan_variants`` 与 ``_variant_params`` 各自重算，
    不往 s3 的 plan 里塞派生状态（那会污染上游产物）。
    """
    axes = None
    try:
        axes = adapter.sweep_axes()
    except Exception:  # noqa: BLE001 - 适配器声明失败不应拖垮阶段
        axes = None
    if not axes:
        return None
    from ..adapters.sweep import expand_sweep

    mode = "ofat"
    try:
        mode = str(adapter.sweep_mode() or "ofat")
    except Exception:  # noqa: BLE001
        mode = "ofat"
    budget = max_runs
    try:
        cap = int(getattr(adapter, "max_sweep_runs", 24) or 24)
    except Exception:  # noqa: BLE001
        cap = 24
    if budget is None:
        budget = cap
    else:
        budget = min(int(budget), cap)
    return expand_sweep(tuple(axes), mode=mode, max_runs=budget)


def _sweep_budget(cfg: Any, adapter: Any) -> int:
    """扫描运行数预算：配置与适配器取较小值（默认 24）。"""
    try:
        cfg_value = int(getattr(cfg, "max_sweep_runs", 24) or 24)
    except Exception:  # noqa: BLE001
        cfg_value = 24
    return max(2, cfg_value)


def _sweep_variant_names(plan_obj: Any) -> list[str]:
    """扫描模式下的变体名。

    **参考格点命名为 ``baseline``**，其余为 ``sw-<n>``。
    理由：材料参数研究问的不是"新方法 vs 基线"，而是"相对参考条件的偏离"，
    而把参考点叫 baseline 既贴合语义，又能直接复用既有的对照逻辑——
    不必为扫描另写一套比较代码，也就不会出现"两套比较口径"的经典漂移。
    """
    cells = list(getattr(plan_obj, "cells", []) or [])
    if not cells:
        return []
    return ["baseline"] + [f"sw-{i}" for i in range(1, len(cells))]

'''

anchor = "\ndef _plan_variants("
if "_sweep_plan(" not in t:
    t = t.replace(anchor, SWEEP_HELPERS + anchor, 1)
    print("  s4: 已插入扫描辅助函数")

# _plan_variants 接受 adapter
old_sig = "def _plan_variants(plan: dict[str, Any], max_variants: int | None = 2) -> list[str]:"
new_sig = (
    "def _plan_variants(\n"
    "    plan: dict[str, Any],\n"
    "    max_variants: int | None = 2,\n"
    "    adapter: Any = None,\n"
    ") -> list[str]:"
)
if old_sig in t:
    t = t.replace(old_sig, new_sig, 1)
    print("  s4: _plan_variants 已接受 adapter")

# 在 _plan_variants 体内最前面插入扫描分支
old_body = '''    primary: list[str] = ["baseline", "method"]
    if max_variants is not None and max_variants <= 2:
        return primary[: max(1, max_variants)]
'''
new_body = '''    # 参数扫描是另一种实验设计：变体是网格格点，不是"方法 vs 基线"。
    # 优先走它（适配器声明了轴就说明它要扫描）。
    if adapter is not None:
        sweep = _sweep_plan(adapter, getattr(adapter, "_sweep_budget_hint", None))
        names = _sweep_variant_names(sweep) if sweep is not None else []
        if names:
            return names

    primary: list[str] = ["baseline", "method"]
    if max_variants is not None and max_variants <= 2:
        return primary[: max(1, max_variants)]
'''
if old_body in t:
    t = t.replace(old_body, new_body, 1)
    print("  s4: _plan_variants 已加扫描分支")

# 调用点传 adapter
old_call = "        variants = _plan_variants(plan, max_variants=_variant_budget(self.ctx.cfg, adapter))"
new_call = (
    "        # 扫描预算与变体预算分开：前者是\"扫多少格点\"，后者是\"跑几个对照臂\"。\n"
    "        sweep_budget = _sweep_budget(self.ctx.cfg, adapter)\n"
    "        setattr(adapter, \"_sweep_budget_hint\", sweep_budget)\n"
    "        variants = _plan_variants(\n"
    "            plan, max_variants=_variant_budget(self.ctx.cfg, adapter), adapter=adapter\n"
    "        )"
)
if old_call in t:
    t = t.replace(old_call, new_call, 1)
    print("  s4: 调用点已传 adapter 与扫描预算")

S4.write_text(t, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(CFG), doraise=True)
py_compile.compile(str(S4), doraise=True)
print("  编译通过")
