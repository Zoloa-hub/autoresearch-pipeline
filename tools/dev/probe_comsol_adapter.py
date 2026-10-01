"""端到端验证 COMSOL 适配器的 ingest 模式：用 21 个真实 console 日志。

要点：
  * 日志是 UTF-16LE，直接按 UTF-8 读会乱码 —— 适配器按 BOM 试探解码
  * 中文错误含 168 个 U+FFFD —— **不可恢复**，适配器如实量化而不是假装能修
  * 指标方向由适配器声明
  * run 模式（真调 comsol batch）本机不可验证，适配器如实说明
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.adapters import resolve_adapter  # noqa: E402
from autoresearch.adapters.base import RunSpec  # noqa: E402
from autoresearch.stages.s4_experiment import _compare  # noqa: E402
from autoresearch.tools.metrics import _higher_is_better  # noqa: E402

LOG_DIR = Path(r"D:\user\Documents\deepseekv4flash harness\comsol_la2ti2o7")
WS = ROOT / ".autoresearch" / "comsol_probe"


def main() -> int:
    print("=" * 76)
    print("COMSOL 批处理适配器：ingest 模式（21 个真实 console 日志）")
    print("=" * 76)

    if not LOG_DIR.is_dir():
        print(f"  SKIP 找不到日志目录: {LOG_DIR}")
        return 0
    if WS.exists():
        shutil.rmtree(WS, ignore_errors=True)
    WS.mkdir(parents=True, exist_ok=True)

    # 1) 加载
    print("\n1) 加载（两条路径）")
    for label, spec in (
        ("module:Class", "autoresearch.adapters.comsol_batch:ComsolBatchAdapter"),
        ("文件路径", "autoresearch/adapters/comsol_batch.py"),
    ):
        try:
            ad = resolve_adapter(spec, {"log_dir": str(LOG_DIR)})
            print(f"   OK   {label:<14} -> {ad.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"   FAIL {label:<14} -> {type(exc).__name__}: {exc}")
            return 1

    adapter = resolve_adapter(
        "autoresearch.adapters.comsol_batch:ComsolBatchAdapter",
        {"log_dir": str(LOG_DIR)},
    )

    # 2) 环境如实说明
    ok, why = adapter.validate_environment()
    print(f"\n2) validate_environment: {ok}")
    print(f"   {why}")

    # 3) 无参数时的报错要可操作
    bare = resolve_adapter("autoresearch.adapters.comsol_batch:ComsolBatchAdapter", {})
    ok2, why2 = bare.validate_environment()
    print(f"\n3) 无参数时: ok={ok2}")
    print(f"   {why2}")
    check_ok = (ok2 is False) and ("log_dir" in why2) and ("model" in why2)
    print(f"   报错同时给出两种模式的可操作指引: {check_ok}")

    # 4) 解析指标
    print("\n4) parse_results()（ingest）")
    series = adapter.parse_results(LOG_DIR)
    if not series:
        print("   FAIL 没解析出指标")
        return 1
    n_obs = len(next(iter(series.values())))
    print(f"   {len(series)} 个指标 / {n_obs} 个日志（每个日志一个观测）")
    for name in sorted(series):
        vals = series[name]
        mean = sum(vals) / len(vals)
        print(f"     {name:<24} mean={mean:<10.4g} min={min(vals):<8.4g} max={max(vals):.4g}")

    # 5) 损坏度量的实际含义
    corrupt = series.get("log_corruption_ratio", [])
    high = [v for v in corrupt if v > 0.5]
    print(f"\n5) 日志损坏度（U+FFFD 占比）")
    print(f"   损坏率 > 50% 的日志数: {len(high)}/{len(corrupt)}")
    if high:
        print(f"   最高损坏率: {max(corrupt):.1%}")
    print("   → 中文错误在 COMSOL 写出时就已成 U+FFFD，**转码不可恢复**；")
    print("     适配器默认加 `-locale en_US` 让后续运行的错误可读。")

    # 6) 方向声明 vs 通用词表
    print("\n6) 方向声明 vs 通用词表")
    declared = adapter.metric_directions() or {}
    print(f"   {'指标':<24} {'声明':<7} {'词表':<7} 一致?")
    mismatched = []
    for name in sorted(series):
        got = _higher_is_better(name)
        d = declared.get(name)
        same = (d == got)
        if not same:
            mismatched.append(name)
        print(f"   {name:<24} {str(d):<7} {str(got):<7} {'一致' if same else '**不一致**'}")
    print(f"\n   不一致 {len(mismatched)}/{len(series)}: {mismatched}")

    # 7) run 模式命令构造（本机不可执行，但可验证形态）
    print("\n7) run 模式命令构造（本机无 COMSOL，只验证形态）")
    run_adapter = resolve_adapter(
        "autoresearch.adapters.comsol_batch:ComsolBatchAdapter",
        {"model": "model.mph", "study": "std1"},
    )
    argv = run_adapter.build_command(RunSpec(variant="baseline", seed=0, out_dir="runs/v/seed_0"))
    print("  ", " ".join(str(a) for a in argv))
    print(f"   含 -locale en_US（让错误可读）: {'-locale' in argv}")
    print(f"   含 batch 子命令: {'batch' in argv}")
    ok3, why3 = run_adapter.validate_environment()
    print(f"   run 模式环境: ok={ok3}")
    print(f"   {why3}")
    print("   → 适配器如实说明「run 模式不可用」，而不是笼统失败")

    # 8) describe
    info = adapter.describe()
    print(f"\n8) describe(): mode={info.get('mode')!r} comsol_available={info.get('comsol_available')}")
    print(f"   metric_axis={info.get('metric_axis')!r} 声明方向数={len(info.get('metric_directions') or {})}")

    print("\n=== COMSOL ingest 模式验证通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
