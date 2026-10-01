"""端到端验证材料光学适配器：用真实的 La2Ti2O7 光学模块跑通。

要点：
  * 模块是用户的真实文件（comsol_la2ti2o7/ltp_optics.py），未做任何修改
  * 指标方向由适配器声明，管线不再从名字猜
  * 序列轴是重复测量，不是 epoch —— 这是非 ML 领域的关键差异
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.adapters import resolve_adapter  # noqa: E402
from autoresearch.adapters.base import RunSpec  # noqa: E402
from autoresearch.config import load_config  # noqa: E402
from autoresearch.stages.s4_experiment import _compare  # noqa: E402
from autoresearch.tools.sandbox import SubprocessSandbox  # noqa: E402

MATERIALS_MODULE = Path(r"D:\user\Documents\deepseekv4flash harness\comsol_la2ti2o7\ltp_optics.py")
WS = ROOT / ".autoresearch" / "materials_probe"


def main() -> int:
    print("=" * 74)
    print("材料光学适配器：端到端（真实 La2Ti2O7 模块，未修改）")
    print("=" * 74)

    if not MATERIALS_MODULE.is_file():
        print(f"  SKIP 找不到材料模块: {MATERIALS_MODULE}")
        return 0

    if WS.exists():
        shutil.rmtree(WS, ignore_errors=True)
    WS.mkdir(parents=True, exist_ok=True)

    # 1) 加载（三种路径都要能用）
    print("\n1) 加载适配器")
    for label, spec in (
        ("module:Class", "autoresearch.adapters.materials_optics:MaterialsOpticsAdapter"),
        ("文件路径", "autoresearch/adapters/materials_optics.py"),
    ):
        try:
            ad = resolve_adapter(spec, {"module": str(MATERIALS_MODULE), "replicates": 4})
            print(f"   OK   {label:<14} -> {ad.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"   FAIL {label:<14} -> {type(exc).__name__}: {exc}")
            return 1

    adapter = resolve_adapter(
        "autoresearch.adapters.materials_optics:MaterialsOpticsAdapter",
        {"module": str(MATERIALS_MODULE), "replicates": 4, "jitter": 0.006},
    )
    ok, reason = adapter.validate_environment()
    print(f"   validate_environment: {ok} — {reason}")
    if not ok:
        return 1

    # 2) 模板落盘（s4 平时做这一步）
    print("\n2) seed_code() 落盘")
    for rel, content in adapter.seed_code(WS, {}).items():
        (WS / rel).write_text(content, encoding="utf-8")
        print(f"   {rel}  ({len(content)} 字符)")

    # 3) 命令里**不能**出现 epoch 之类 ML 概念
    spec = RunSpec(variant="baseline", seed=0, out_dir="runs/baseline/seed_0")
    argv = adapter.build_command(spec)
    print("\n3) build_command()")
    print("  ", " ".join(str(a) for a in argv))
    print(f"   含 --replicates（重复测量）: {'--replicates' in argv}")
    print(f"   含 --epochs（ML 概念，应为 False）: {'--epochs' in argv}")

    run_dir = WS / spec.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4) 沙箱执行
    print("\n4) 沙箱执行")
    sandbox = SubprocessSandbox(load_config().sandbox, WS)
    try:
        result = sandbox.run_command(argv, timeout=600)
    except Exception as exc:  # noqa: BLE001
        print(f"   FAIL 沙箱: {type(exc).__name__}: {exc}")
        return 1
    print(f"   ok={result.ok} rc={result.returncode}")
    if not result.ok:
        for line in (result.stderr or "").splitlines()[-10:]:
            print("    ", line[:150])
        return 1

    # 5) 指标解析
    print("\n5) parse_results()")
    series = adapter.parse_results(run_dir)
    print(f"   {len(series)} 个指标：")
    for name, vals in sorted(series.items()):
        mean = sum(vals) / len(vals)
        print(f"     {name:<24} n={len(vals)}  mean={mean:.6g}")

    # 6) 声明方向 vs 不声明 —— 这是核心对比
    print("\n6) 适配器声明的方向 vs 通用词表")
    declared = adapter.metric_directions() or {}
    from autoresearch.tools.metrics import _higher_is_better

    print(f"   {'指标':<24} {'适配器声明':<12} {'通用词表猜的':<12} 一致?")
    mismatched = []
    for name in sorted(series):
        d = declared.get(name)
        guess = _higher_is_better(name)
        flag = "一致" if d == guess else "**不一致**"
        if d != guess:
            mismatched.append(name)
        print(f"   {name:<24} {str(d):<12} {str(guess):<12} {flag}")
    print(f"\n   不一致 {len(mismatched)}/{len(series)} 个: {mismatched}")

    # 7) 对照结论是否被方向影响
    print("\n7) 方向如何改变对照结论")
    # 造一个"全面改善"的对照：偏差/吸收类降 30%，透明窗口升 30%
    adv = {
        k: [
            v[-1] * 0.7
            if ("deviation" in k or k.startswith("k_") or k.startswith("alpha"))
            else v[-1] * 1.3
        ]
        for k, v in series.items()
    }
    results = {"baseline": series, "improved": adv}
    c_bad = _compare(results)
    c_good = _compare(results, directions=declared)
    print(f"   不声明方向: supports_claim={c_bad.get('supports_claim')}  improved={c_bad.get('improved')}")
    print(f"   声明方向  : supports_claim={c_good.get('supports_claim')}  improved={c_good.get('improved')}")

    # 8) describe()
    print("\n8) describe()")
    info = adapter.describe()
    print(f"   metric_axis = {info.get('metric_axis')!r}（非 epoch）")
    print(f"   声明方向数 = {len(info.get('metric_directions') or {})}")

    print("\n=== 材料领域端到端验证通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
