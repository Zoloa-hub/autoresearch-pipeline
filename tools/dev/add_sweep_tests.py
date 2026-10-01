"""给多参数扫描加回归测试（追加到 test_adapters.py）。

测试重点放在**会静默出错**的地方，而不是正常路径：
  * 截断网格必须标记非正交（否则主效应被误读成"某轴重要"）
  * OFAT 必须说明它测不出交互
  * 参考格点必须拿到参数（否则对照退化成"默认值 vs 被改过的值"）
  * 覆盖 target 必须进命令行（否则扫描变成重复实验，效应全 0）
  * 交互效应检测：可加数据≈0、有交互数据显著——用合成数据验证判据本身
"""

from __future__ import annotations

import pathlib

TA = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "tests" / "test_adapters.py"
t = TA.read_text(encoding="utf-8")

NEW = '''

def test_parameter_sweep() -> None:
    """多参数扫描：展开、预算护栏、效应分析。

    材料/化学/器件研究最自然的实验形态是参数扫描（温度 × 成分 × 退火时间），
    而它和"方法 vs 基线"是两种不同的设计。这里测的是**会静默出错**的地方：
    截断破坏正交性、OFAT 测不出交互、参考格点拿不到参数、
    覆盖 target 没进命令行——每一个都不会报错，只会给出看起来正常的错误结论。
    """
    section("多参数扫描：展开与预算护栏")
    from autoresearch.adapters.sweep import (
        SweepAxis,
        axis_effects,
        effect_ratio,
        expand_sweep,
        interaction_effects,
        rank_axes,
    )

    # --- 参数校验 ---
    try:
        SweepAxis("", (1, 2))
        check("空轴名被拒绝", False, "未抛异常")
    except ValueError:
        check("空轴名被拒绝", True)
    try:
        SweepAxis("x", (1,))
        check("少于 2 个取值被拒绝", False, "未抛异常")
    except ValueError:
        check("少于 2 个取值被拒绝", True)

    T = SweepAxis("temp_C", (600, 700, 800), "C")
    X = SweepAxis("comp_x", (0.0, 0.1), "")
    H = SweepAxis("time_h", (1, 4), "h")

    # --- OFAT 数量 = 1 + Σ(k-1) ---
    ofat = expand_sweep([T, X, H], mode="ofat", max_runs=None)
    eq(ofat.n_runs, 1 + 2 + 1 + 1, "OFAT 运行数 = 1 + Σ(kᵢ−1)")
    eq(len(ofat.cells[0]), 3, "参考格点含全部轴")
    eq(ofat.reference["temp_C"], 600, "默认参考点取每轴首值")
    check(
        any("无法发现交互" in n for n in ofat.notes),
        "OFAT 必须说明它测不出交互效应（否则会被当成完整结论）",
        str(ofat.notes)[:120],
    )

    # --- 网格数量 = Πk ---
    grid = expand_sweep([T, X, H], mode="grid", max_runs=None)
    eq(grid.n_runs, 3 * 2 * 2, "网格运行数 = Πkᵢ")
    check(grid.orthogonal is True, "未截断的网格是正交的")
    check(not any("无法发现交互" in n for n in grid.notes), "网格不报 OFAT 的告警")

    # --- 截断护栏（核心）---
    cut = expand_sweep([T, X, H], mode="grid", max_runs=5)
    eq(cut.n_runs, 5, "截断到预算")
    eq(cut.dropped, 12 - 5, "如实报告丢了多少组")
    check(cut.truncated is True, "标记为已截断")
    check(
        cut.orthogonal is False,
        "**截断后必须标记为非正交**——否则主效应会与交互效应混淆，"
        "而结果看起来完全正常",
    )
    check(
        any("不再正交" in n for n in cut.notes),
        "截断时必须说明后果", str(cut.notes)[:140],
    )

    # --- 自定义参考点 ---
    ref = expand_sweep([T, X], mode="ofat", max_runs=None, reference={"temp_C": 700})
    eq(ref.reference["temp_C"], 700, "参考点可指定")
    eq(ref.reference["comp_x"], 0.0, "未指定的轴取首值")
    try:
        expand_sweep([T], mode="ofat", reference={"temp_C": 999})
        check("非法参考点被拒绝", False, "未抛异常")
    except ValueError:
        check("非法参考点被拒绝（不在取值内）", True)
    try:
        expand_sweep([T, X], mode="bogus")
        check("非法 mode 被拒绝", False, "未抛异常")
    except ValueError:
        check("非法 mode 被拒绝", True)

    # --- 效应分析：合成数据验证判据本身 ---
    section("多参数扫描：效应分析")
    A = SweepAxis("a", (0, 1, 2))
    B = SweepAxis("b", (0, 1))
    cells = expand_sweep([A, B], mode="grid", max_runs=None).cells
    # 可加模型：val = 10*a + 3*b，无交互
    add_obs = [(c, 10.0 * c["a"] + 3.0 * c["b"]) for c in cells]
    add_eff = rank_axes(axis_effects(add_obs, [A, B]))
    eq([e.axis for e in add_eff], ["a", "b"], "可加数据里效应排序正确")
    eq(round(add_eff[0].span, 6), 20.0, "轴 a 的效应幅度 = 10×2")
    eq(round(add_eff[1].span, 6), 3.0, "轴 b 的效应幅度 = 3×1")
    ratios = effect_ratio(add_eff)
    check(abs(ratios["b"] - 3.0 / 20.0) < 1e-9, "相对强度比例正确", str(ratios))

    add_int = interaction_effects(add_obs, [A, B])
    check(
        (not add_int) or add_int[0].strength < 1e-9,
        "可加数据的交互强度为 0（判据不会凭空造出交互）",
        str([i.to_dict() for i in add_int]),
    )

    # 有交互：b 会放大 a 的作用
    inter_obs = [(c, (10.0 * c["a"]) * (1.0 + 2.0 * c["b"])) for c in cells]
    inter_int = interaction_effects(inter_obs, [A, B])
    check(bool(inter_int), "能报出交互", str(inter_int))
    if inter_int:
        # b=0 时 a 的效应 = 20；b=1 时 = 60 -> 交互强度 = 40
        check(
            abs(inter_int[0].strength - 40.0) < 1e-6,
            "交互强度数值正确（b 放大 a 的作用）",
            f"{inter_int[0].strength}",
        )

    # --- OFAT 在有交互时会给出错误结论（这正是要告警的理由）---
    ofat_cells = expand_sweep([A, B], mode="ofat", max_runs=None).cells
    # 构造"只有 a=0,b=0 是坏的、其余都好"的交互，OFAT 从 (0,0) 出发
    tricky = [(c, 0.0 if (c["a"] == 0 and c["b"] == 0) else 100.0) for c in ofat_cells]
    tricky_eff = axis_effects(tricky, [A, B])
    check(
        bool(tricky_eff),
        "OFAT 仍会算出数值（所以必须靠 note 提示它测不出交互）",
        str([e.to_dict() for e in tricky_eff])[:150],
    )

    # --- 缺失值不参与统计 ---
    sparse = [({"a": 0}, 1.0), ({"a": 1}, float("nan")), ({"a": 2}, 3.0)]
    sp = axis_effects(sparse, [A])
    check(bool(sp), "部分缺失仍可算", str(sp))
    if sp:
        check(abs(sp[0].span - 2.0) < 1e-9, "nan 被忽略（跨度 = 3-1）", str(sp[0].span))


def test_sweep_integration() -> None:
    """扫描接进适配器协议与 s4：命令里必须有覆盖、参考格点必须有参数。"""
    section("多参数扫描：适配器与 s4 接线")
    from autoresearch.adapters.base import RunSpec
    from autoresearch.adapters.materials_optics import MaterialsOpticsAdapter
    from autoresearch.adapters.sweep import SweepAxis, expand_sweep
    from autoresearch.stages.s4_experiment import (
        _sweep_plan,
        _sweep_variant_names,
        _variant_params,
    )

    # 基类默认不扫描
    from autoresearch.adapters.base import BaseExperimentAdapter

    check(BaseExperimentAdapter({}).sweep_axes() is None,
          "基类默认不做参数扫描（不改变既有适配器行为）")
    eq(BaseExperimentAdapter({}).sweep_mode(), "ofat", "默认设计是 OFAT")

    # 材料适配器声明了 3 条轴，且带 target
    ad = MaterialsOpticsAdapter({})
    axes = ad.sweep_axes()
    check(bool(axes) and len(axes) == 3, "材料适配器声明 3 条扫描轴", str(axes))
    if axes:
        eq([a.name for a in axes], ["osc_g", "osc_f", "n_vis"], "轴名正确")
        check(
            all(a.target for a in axes),
            "**每条轴都显式声明 target**（绑定方式只有适配器知道，猜错会静默失效）",
            str([(a.name, a.target) for a in axes]),
        )
        check(
            any(":" in a.target for a in axes),
            "其中至少一条是函数默认参数绑定（必须用 函数名:参数名 形式）",
            str([a.target for a in axes]),
        )

    # 可关闭 / 可覆盖
    check(MaterialsOpticsAdapter({"no_sweep": True}).sweep_axes() is None,
          "no_sweep 可关闭扫描")
    custom = MaterialsOpticsAdapter({"sweeps": "osc_g:0.3|0.5|0.7"}).sweep_axes()
    check(bool(custom) and len(custom) == 1 and custom[0].levels == 3,
          "sweeps 可自定义轴与取值", str(custom))
    check(custom is not None and custom[0].target == "eps_lorentz:g",
          "自定义轴继承已知轴的 target（否则又会静默失效）",
          str(custom[0].target if custom else None))

    # _sweep_plan / 变体命名
    plan = _sweep_plan(ad, 12)
    check(plan is not None, "_sweep_plan 对声明了轴的适配器返回方案")
    names = _sweep_variant_names(plan) if plan else []
    eq(names[0], "baseline", "参考格点命名为 baseline")
    check(all(n.startswith("sw-") for n in names[1:]), "其余格点是 sw-<n>", str(names))

    # **参考格点必须拿到参数**，否则对照变成"默认值 vs 被改过的值"
    base_params = _variant_params({}, "baseline", ad, 12)
    check(bool(base_params) and len(base_params) == 3,
          "参考格点拿到参考条件的参数（不是空 dict）", str(base_params))
    sw1 = _variant_params({}, "sw-1", ad, 12)
    check(bool(sw1) and sw1 != base_params,
          "其余格点拿到与参考格点不同的参数", f"{base_params} vs {sw1}")

    # **覆盖必须进命令行**——否则所有格点算出同一组数值，
    # 扫描退化成重复实验，效应全为 0，然后被误读成"参数都不重要"
    argv = ad.build_command(
        RunSpec(variant="sw-1", seed=0, out_dir="runs/sw-1/seed_0", params=sw1)
    )
    eq(sum(1 for a in argv if a == "--set"), len(sw1), "每个参数都变成一次 --set")
    joined = " ".join(str(a) for a in argv)
    check("eps_lorentz:g=" in joined or "eps_lorentz:f=" in joined or "N_VIS=" in joined,
          "命令里用的是声明过的 target 形式", joined[-120:])
    check("--epochs" not in argv, "扫描命令里没有 ML 概念", joined)

    # 未声明轴的 adapter 不受影响
    from autoresearch.stages.s4_experiment import _plan_variants

    eq(_plan_variants({}, max_variants=2, adapter=BaseExperimentAdapter({})),
       ["baseline", "method"], "未声明轴的适配器仍走原有变体逻辑")
'''

anchor = "\ndef main() -> int:"
if "def test_parameter_sweep(" not in t:
    t = t.replace(anchor, NEW + anchor, 1)
    reg = """    _progress('test_parameter_sweep')
    try:
        test_parameter_sweep()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_parameter_sweep 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_sweep_integration')
    try:
        test_sweep_integration()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_sweep_integration 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    print("\\n" + "=" * 70)"""
    t = t.replace('    print("\\n" + "=" * 70)', reg, 1)
    TA.write_text(t, encoding="utf-8")
    print("  已加 2 个扫描测试并注册")
else:
    print("  已存在")

import py_compile  # noqa: E402

py_compile.compile(str(TA), doraise=True)
print("  编译通过")
