"""修两处：扫描格点的参数解析，以及"参考格点必须也拿到参数"。

## 问题 1：``_variant_params`` 对扫描格点返回空

它原本只查 s3 的消融矩阵，``sw-1``/``baseline``（扫描模式）都查不到，于是返回 ``{}``。
结果：所有格点用同一套默认参数跑，扫描退化成"同一个实验跑 N 次"——
**不会有任何报错**，只会得到一条平坦的效应曲线，然后被解读成"这些参数都不重要"。

## 问题 2：参考格点（baseline）也必须拿到它的参数

扫描模式下 ``baseline`` 是**参考条件**（OFAT 的参考点 / 网格的首个格点），
不是"默认参数"。如果它拿 ``{}`` 而其余格点拿覆盖值，
对照就变成"模块默认 vs 被改过的参数"，而不是"参考条件 vs 偏离参考条件"。

## 问题 3：用 setattr 传预算是脏的

改为显式参数 ``sweep_budget``，不再往 adapter 上挂临时属性。
"""

from __future__ import annotations

import pathlib

S4 = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "stages" / "s4_experiment.py"
t = S4.read_text(encoding="utf-8")
applied = 0

# ---- 1) _variant_params 接受 adapter 并解析扫描格点 ----
old = '''def _variant_params(plan: dict[str, Any], variant: str) -> dict[str, Any]:
    """给某个变体取出该应用的超参。

    主对照（baseline/method）返回空 dict——它们必须用**同一套默认超参**，
    否则「提升」就归因不到方法本身。
    消融格点返回 ``{轴名: 取值}``，让适配器把它变成命令行参数。
    """
    for label, _axis, _value, params in _ablation_cells(plan):
        if _safe_variant(label) == variant or label == variant:
            return dict(params)
    return {}'''
new = '''def _variant_params(
    plan: dict[str, Any],
    variant: str,
    adapter: Any = None,
    sweep_budget: int | None = None,
) -> dict[str, Any]:
    """给某个变体取出该应用的超参。

    三类变体，参数来源不同：

    * **扫描格点**（``baseline`` + ``sw-<n>``，仅当适配器声明了扫描轴）：
      取自扫描方案。注意 **``baseline`` 在这里是"参考条件"，不是"默认参数"**——
      它同样要拿到参考格点的取值序列。否则对照会变成
      "模块默认 vs 被改过的参数"，而不是"参考条件 vs 偏离参考条件"。
    * **主对照**（消融模式下）返回空 dict——它们必须用**同一套默认超参**，
      否则「提升」就归因不到方法本身。
    * **消融格点**返回 ``{轴名: 取值}``，让适配器把它变成命令行参数。
    """
    if adapter is not None:
        sweep = _sweep_plan(adapter, sweep_budget)
        if sweep is not None:
            names = _sweep_variant_names(sweep)
            if variant in names:
                cell = sweep.cells[names.index(variant)]
                return dict(cell)
    for label, _axis, _value, params in _ablation_cells(plan):
        if _safe_variant(label) == variant or label == variant:
            return dict(params)
    return {}'''
if old in t:
    t = t.replace(old, new, 1)
    applied += 1
    print("  _variant_params: 已加扫描分支（含参考格点取参）")
else:
    print("  MISS _variant_params")

# ---- 2) 调用点传入 adapter 与预算 ----
old_call = "        params = _variant_params(plan, variant)"
new_call = "        params = _variant_params(plan, variant, adapter, sweep_budget)"
if old_call in t:
    # sweep_budget 需在 run() 里可见：它已经在调用 _plan_variants 之前算好
    t = t.replace(old_call, new_call, 1)
    applied += 1
    print("  调用点: 已传 adapter 与 sweep_budget")
else:
    print("  MISS 调用点")

# ---- 3) 去掉 setattr 脏传参，改为显式参数 ----
old_hint = '''        # 扫描预算与变体预算分开：前者是"扫多少格点"，后者是"跑几个对照臂"。
        sweep_budget = _sweep_budget(self.ctx.cfg, adapter)
        setattr(adapter, "_sweep_budget_hint", sweep_budget)
        variants = _plan_variants(
            plan, max_variants=_variant_budget(self.ctx.cfg, adapter), adapter=adapter
        )'''
new_hint = '''        # 扫描预算与变体预算分开：前者是"扫多少格点"，后者是"跑几个对照臂"。
        # 显式传参，不用 setattr 往适配器上挂临时属性（那会让同一适配器实例的
        # 行为依赖调用顺序）。
        sweep_budget = _sweep_budget(self.ctx.cfg, adapter)
        variants = _plan_variants(
            plan,
            max_variants=_variant_budget(self.ctx.cfg, adapter),
            adapter=adapter,
            sweep_budget=sweep_budget,
        )'''
if old_hint in t:
    t = t.replace(old_hint, new_hint, 1)
    applied += 1
    print("  调用点: 已去掉 setattr，改显式 sweep_budget")
else:
    print("  MISS setattr 段")

# ---- 4) _plan_variants 用显式 sweep_budget ----
t = t.replace(
    "    max_variants: int | None = 2,\n    adapter: Any = None,\n) -> list[str]:",
    "    max_variants: int | None = 2,\n    adapter: Any = None,\n    sweep_budget: int | None = None,\n) -> list[str]:",
    1,
)
t = t.replace(
    "        sweep = _sweep_plan(adapter, getattr(adapter, \"_sweep_budget_hint\", None))",
    "        sweep = _sweep_plan(adapter, sweep_budget)",
    1,
)

S4.write_text(t, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(S4), doraise=True)
print(f"  已改 {applied} 处；编译通过")
