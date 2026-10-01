"""真实多参数扫描 E2E：La2Ti2O7 深紫外吸收边的三参数研究。

这是第一个**真正的参数扫描**（而非"方法 vs 基线"）：
  * 3 条物理参数轴，OFAT 设计 → 7 个格点
  * 每个格点在沙箱里真实执行
  * 主效应分析 → "哪条轴主导"
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.adapters import resolve_adapter  # noqa: E402
from autoresearch.adapters.base import RunSpec  # noqa: E402
from autoresearch.adapters.sweep import axis_effects, effect_ratio, rank_axes  # noqa: E402
from autoresearch.config import load_config  # noqa: E402
from autoresearch.stages.s4_experiment import (  # noqa: E402
    _sweep_plan,
    _sweep_variant_names,
    _variant_params,
)
from autoresearch.tools.sandbox import SubprocessSandbox  # noqa: E402

MOD = Path(r"D:\user\Documents\deepseekv4flash harness\comsol_la2ti2o7\ltp_optics.py")
WS = ROOT / ".autoresearch" / "sweep_e2e"
BUDGET = 12


def main() -> int:
    print("=" * 78)
    print("多参数扫描 E2E：La2Ti2O7 深紫外吸收边（3 参数轴，真实执行）")
    print("=" * 78)

    if not MOD.is_file():
        print("  SKIP 找不到材料模块")
        return 0
    if WS.exists():
        shutil.rmtree(WS, ignore_errors=True)
    WS.mkdir(parents=True, exist_ok=True)

    adapter = resolve_adapter(
        "autoresearch.adapters.materials_optics:MaterialsOpticsAdapter",
        {"module": str(MOD), "replicates": 3, "jitter": 0.002},
    )
    ok, why = adapter.validate_environment()
    print(f"\n  环境: {ok} — {why}")
    if not ok:
        return 1

    # 1) 扫描方案
    plan = _sweep_plan(adapter, BUDGET)
    print(f"\n1) 扫描方案: {plan.describe()}")
    print(f"   orthogonal={plan.orthogonal}  truncated={plan.truncated}")
    print("   轴:")
    for ax in plan.axes:
        print(f"     {ax.name:<8} {ax.levels} 水平 {list(ax.values)} {ax.unit}"
              f"  target={ax.target}")
    for note in plan.notes:
        print(f"   note: {note[:100]}")

    names = _sweep_variant_names(plan)
    print(f"\n2) 变体 {len(names)} 个（参考格点命名为 baseline）")

    # 2) 模板落盘
    for rel, content in adapter.seed_code(WS, {}).items():
        (WS / rel).write_text(content, encoding="utf-8")

    # 3) 逐格点执行
    print(f"\n3) 沙箱逐格点执行")
    sandbox = SubprocessSandbox(load_config().sandbox, WS)
    observations: dict[str, list[float]] = {}
    params_of: dict[str, dict] = {}
    failures: list[str] = []

    for name in names:
        params = _variant_params({}, name, adapter, BUDGET)
        params_of[name] = params
        out_dir = f"runs/{name}/seed_0"
        (WS / out_dir).mkdir(parents=True, exist_ok=True)
        spec = RunSpec(variant=name, seed=0, out_dir=out_dir, params=params)
        argv = adapter.build_command(spec)
        res = sandbox.run_command(argv, timeout=600)
        if not res.ok:
            failures.append(f"{name}: rc={res.returncode}")
            print(f"   FAIL {name:<9} rc={res.returncode}")
            for line in (res.stderr or "").splitlines()[-4:]:
                print(f"        {line[:130]}")
            continue
        series = adapter.parse_results(WS / out_dir)
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"   ok   {name:<9} [{label}]")
        for metric, values in series.items():
            observations.setdefault(metric, [])
        # 每个格点取该格点的均值（重复测量的均值）
        for metric, values in series.items():
            if values:
                observations.setdefault(f"__cell__{metric}", []).append(
                    (params, sum(values) / len(values))
                )

    if failures:
        print(f"\n   失败 {len(failures)} 个格点: {failures}")
        return 1

    # 4) 主效应分析
    print(f"\n4) 主效应分析（{len(names)} 个格点）")
    summary: dict[str, dict] = {}
    for metric in sorted(
        k[len("__cell__"):] for k in observations if k.startswith("__cell__")
    ):
        obs = observations[f"__cell__{metric}"]
        effects = axis_effects(obs, plan.axes, orthogonal=plan.orthogonal)
        if not effects:
            continue
        ranked = rank_axes(effects)
        ratios = effect_ratio(effects)
        print(f"\n   指标 {metric}")
        print(f"     {'轴':<10} {'效应幅度':>12}  {'相对最强':>8}  {'最优取值':>10}")
        for e in ranked:
            print(f"     {e.axis:<10} {e.span:>12.5g}  {ratios[e.axis]:>8.2f}  "
                  f"{str(e.best_level):>10}")
        summary[metric] = {
            "ranked": [e.to_dict() for e in ranked],
            "ratios": ratios,
        }

    # 5) 结论
    # 注意：summary 里存的是 to_dict() 的**结果**（dict），不是 AxisEffect 对象。
    print("\n5) 结论")
    for metric, info in summary.items():
        top = info["ranked"][0]
        second = info["ranked"][1] if len(info["ranked"]) > 1 else None
        unit = top.get("unit") or "无量纲"
        line = (
            f"   {metric}: 主导轴 = {top['axis']}"
            f"（{unit}，最优 {top['best_level']}）"
        )
        if second:
            line += f"；次强 {second['axis']} 为其 {info['ratios'][second['axis']]:.2f} 倍"
        print(line)

    # 结论的一句话总结：哪条轴主导**取决于指标**
    leaders = {info["ranked"][0]["axis"] for info in summary.values()}
    print(f"\n   主导轴集合: {sorted(leaders)}")
    if len(leaders) > 1:
        print("   —— 没有单一「最重要参数」：哪条轴主导取决于关心哪个性质。"
              "这正是参数研究该给出的信息。")
    else:
        print("   —— 所有指标都由同一条轴主导。")
    print(f"\n   设计说明: {plan.mode.upper()}"
          + ("（**无法检测交互效应**）" if plan.mode == "ofat" else "（可检测交互）"))

    print("\n=== 多参数扫描端到端通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
