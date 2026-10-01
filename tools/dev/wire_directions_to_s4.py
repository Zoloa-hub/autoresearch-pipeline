"""把适配器声明的指标方向接进 s4 的对照逻辑。

`_compare` 原本对每个指标调 `_higher_is_better(metric)` 从 token 表猜方向。
接线后：`adapter.metric_directions()` → `_compare(directions=...)` → 三处判定。

顺带让"适配器声明了方向但漏了某个指标"变成一条可见的 warning，
而不是静默回落 token 表——那正是会把方向判反的地方。
"""

from __future__ import annotations

import pathlib
import re

p = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "stages" / "s4_experiment.py"
t = p.read_text(encoding="utf-8")
applied = 0

# 1) 调用点：把适配器声明传进去
old_call = "    comparison = _compare(results)"
new_call = (
    "    # 适配器声明的指标方向优先于通用 token 表 —— 领域知识在适配器里。\n"
    "    # 材料学实测：token 表会把 k（消光系数）判成「越大越好」，而它越小越好。\n"
    "    declared = adapter.metric_directions()\n"
    "    comparison = _compare(results, directions=declared)\n"
    "    if declared:\n"
    "        # 声明了方向却没覆盖到的指标会静默回落 token 表 —— 那正是会判反的地方，\n"
    "        # 所以如实记一条 warning，让它可见。\n"
    "        declared_names = {str(k).split(\"@seed=\")[0] for k in declared}\n"
    "        seen_names = {str(k).split(\"@seed=\")[0] for blk in results.values()\n"
    "                      for k in (blk or {})}\n"
    "        uncovered = sorted(seen_names - declared_names)\n"
    "        if uncovered:\n"
    "            warnings.append(\n"
    "                f\"适配器声明了 {len(declared)} 个指标方向，但以下指标未声明、\"\n"
    "                f\"将回落通用词表（可能判反）: {', '.join(uncovered[:8])}\"\n"
    "            )"
)
if old_call in t:
    t = t.replace(old_call, new_call, 1)
    applied += 1
    print("  s4: 调用点已传 directions")
else:
    print("  MISS: _compare 调用点")

# 2) _compare 签名
old_sig = "def _compare(results: dict[str, dict[str, Any]]) -> dict[str, Any]:"
new_sig = (
    "def _compare(\n"
    "    results: dict[str, dict[str, Any]],\n"
    "    directions: dict[str, bool] | None = None,\n"
    ") -> dict[str, Any]:"
)
if old_sig in t:
    t = t.replace(old_sig, new_sig, 1)
    applied += 1
    print("  s4: _compare 已接受 directions")
else:
    print("  MISS: _compare 签名")

# 3) 三处判定传 directions
for old, new in [
    ("rels.append(rel if _higher_is_better(metric) else -rel)",
     "rels.append(rel if _higher_is_better(metric, directions) else -rel)"),
    ('direction = "higher" if _higher_is_better(name) else "lower"',
     'direction = "higher" if _higher_is_better(name, directions) else "lower"'),
    ("return min(values) if not _higher_is_better(name) else max(values)",
     "return min(values) if not _higher_is_better(name, directions) else max(values)"),
]:
    if old in t:
        t = t.replace(old, new, 1)
        applied += 1
print(f"  已改 {applied} 处")

# 4) s4 本地的 _higher_is_better 透传 declared
old_h = '''def _higher_is_better(name: str) -> bool:
    """指标方向：转调共享实现（tools.metrics 是唯一规范表）。

    此前 s3/s4/s5/s9 各有一份 token 表副本，且已经漂移（有的含 fid/fdr，
    有的含 flops/params）。副本一旦存在就会继续分叉，而方向判错不会报错——
    它只会让「改善」的定义在不同章节里不一致。
    """
    from ..tools.metrics import _higher_is_better as _shared

    return bool(_shared(name))'''
new_h = '''def _higher_is_better(name: str, directions: dict[str, bool] | None = None) -> bool:
    """指标方向：适配器声明优先，其次共享 token 表。

    此前 s3/s4/s5/s9 各有一份 token 表副本，且已经漂移。现在方向判定收敛到
    `tools.metrics` 一张表；而**领域差异**由适配器通过 `metric_directions()`
    显式声明——通用词表只是兜底，且对非 ML 领域并不可靠（材料学实测 13/19 不可信）。
    """
    from ..tools.metrics import _higher_is_better as _shared

    return bool(_shared(name, declared=directions))'''
if old_h in t:
    t = t.replace(old_h, new_h, 1)
    applied += 1
    print("  s4: _higher_is_better 已透传 declared")
else:
    print("  MISS: s4 本地 _higher_is_better")

p.write_text(t, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(p), doraise=True)
print("  编译通过")
