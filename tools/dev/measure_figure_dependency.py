"""实测「序列 vs 标量」对出图数量的影响，用来替换未经验证的量级声明。

原来的注释写「约八成的图会失去意义」，另一个版本写「41 张图里的 30 张」。
两个数字都不是测量得来的。这个脚本给出真实结果。
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "autoresearch"))

from autoresearch.tools.figures import make_all_figures  # noqa: E402

WORK = ROOT / ".autoresearch" / "figure_measure"
METRICS = ("accuracy", "f1", "loss", "val_loss", "precision", "recall")
EPOCHS = 8
RUNS = ("baseline", "method")


def _scratch(name: str) -> Path:
    d = WORK / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    (d / "metrics").mkdir(parents=True, exist_ok=True)
    return d


def write_series(root: Path, collapse: bool) -> None:
    """写指标文件。collapse=True 时每条序列只留最后一个点（模拟"返回标量"）。"""
    rng = random.Random(7)
    for run in RUNS:
        d = root / "metrics" / run / "seed_0"
        d.mkdir(parents=True, exist_ok=True)
        lines = ["epoch," + ",".join(METRICS)]
        for epoch in range(1, EPOCHS + 1):
            vals = [f"{rng.uniform(0.5, 0.99):.4f}" for _ in METRICS]
            if collapse and epoch < EPOCHS:
                continue  # 只保留最后一个点 → 每个指标序列长度 1
            lines.append(f"{epoch}," + ",".join(vals))
        (d / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_figures(root: Path) -> dict[str, int]:
    out = root / "figures"
    produced = make_all_figures(root / "metrics", out, formats=("pdf",))
    counts: dict[str, int] = {}
    for name, paths in (produced or {}).items():
        if name.startswith("__"):
            continue
        if isinstance(paths, (list, tuple)):
            counts[name] = len(paths)
    return counts


def main() -> int:
    print("=" * 66)
    print(f"指标数 {len(METRICS)} / epoch 数 {EPOCHS} / run 数 {len(RUNS)}")
    print("=" * 66)

    full_dir = _scratch("full")
    write_series(full_dir, collapse=False)
    full = count_figures(full_dir)

    collapsed_dir = _scratch("collapsed")
    write_series(collapsed_dir, collapse=True)
    collapsed = count_figures(collapsed_dir)

    print("\n--- 序列充足（每个指标 8 个点）---")
    for name in sorted(full):
        print(f"  {name:<28} {full[name]}")
    print(f"  合计 {sum(full.values())} 个图文件")

    print("\n--- 只有最终标量（每个指标 1 个点）---")
    for name in sorted(collapsed):
        print(f"  {name:<28} {collapsed[name]}")
    print(f"  合计 {sum(collapsed.values())} 个图文件")

    total_full, total_collapsed = sum(full.values()), sum(collapsed.values())
    lost = sorted(set(full) - set(collapsed))
    print("\n--- 结论 ---")
    print(f"  序列充足: {total_full} 张；退化后: {total_collapsed} 张")
    if total_full:
        pct = 100.0 * (total_full - total_collapsed) / total_full
        print(f"  减少 {total_full - total_collapsed} 张，占 {pct:.0f}%")
    print(f"  完全消失的图类型: {lost if lost else '（无）'}")
    print(f"  仍然存在的图类型: {sorted(collapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
