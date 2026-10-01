"""把多参数扫描接进适配器协议、配置与 s4。

设计要点（每条都有理由）：

1. **预算旋钮独立于 ``max_variants``。** 变体预算默认 2（baseline/method），
   而一次材料参数扫描天然要 6–18 组。共用一个旋钮会让"想跑扫描"变成
   "必须先把变体上限调到 18"，语义混淆。
   → 新增 ``cfg.max_sweep_runs``（环境变量 ``AUTORESEARCH_MAX_SWEEP_RUNS``）。

2. **扫描模式下 baseline = 参考格点**，其余是 ``sw-<n>``。
   材料参数研究不是"方法 vs 基线"，而是"相对参考条件的偏离"——
   把 OFAT 参考点（或网格首个格点）命名为 ``baseline``，
   既贴合语义，又能直接复用既有的对照逻辑，不必为扫描再写一套。
   **不**改动 ``_compare`` 的假设。

3. **扫描方案是纯函数**，在 ``_plan_variants`` 与 ``_variant_params`` 各自
   由 (adapter, 预算) 重算，而不是往 s3 的 plan 里塞派生数据——
   后者会把派生状态写进上游产物。
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- 1) 基类 --
BASE = ROOT / "autoresearch" / "adapters" / "base.py"
b = BASE.read_text(encoding="utf-8")

ANCHOR = "    def metric_directions(self)"
SWEEP_API = '''    #: 单次参数扫描的运行数上限（可由配置收紧，取两者较小值）
    max_sweep_runs: int = 24

    #: 默认扫描设计：``"ofat"``（一次一条轴，便宜但测不出交互）
    #: 或 ``"grid"``（全组合，能测交互但组合爆炸）。
    default_sweep_mode: str = "ofat"

    def sweep_axes(self) -> tuple[Any, ...] | None:
        """声明可扫描的参数轴。返回 ``None`` = 本适配器不支持参数扫描。

        材料/化学/器件研究最常见的实验形态就是参数扫描（温度 × 成分 × 退火时间），
        而它和"方法 vs 基线"是**两种不同的实验设计**：

        * 对照实验问"新方法是否更好" → 变体是离散的臂
        * 参数扫描问"哪个因素主导、最优条件在哪" → 变体是网格上的格点

        声明轴之后，管线会按 ``default_sweep_mode`` 展开成格点，
        并把每个格点的参数通过 :attr:`RunSpec.params` 交给
        :meth:`build_command` —— 适配器负责把它变成自己认识的命令行参数。

        返回 ``SweepAxis`` 的元组（见 :mod:`autoresearch.adapters.sweep`）。
        """
        return None

    def sweep_mode(self) -> str:
        """本适配器倾向的扫描设计：``"ofat"`` 或 ``"grid"``。

        选 ``"grid"`` 前请确认预算：``Π kᵢ`` 增长极快（5×4×3 = 60），
        而超预算截断会**破坏正交性**、使主效应与交互效应混淆。
        """
        return self.default_sweep_mode

''' + ANCHOR

if "def sweep_axes(" not in b:
    b = b.replace(ANCHOR, SWEEP_API, 1)
    if "\nfrom typing import Any" not in b and "Any" not in b.split("\n\n")[0]:
        pass
    BASE.write_text(b, encoding="utf-8")
    print("  base.py: 已加 sweep_axes / sweep_mode / max_sweep_runs")
else:
    print("  base.py: 已有 sweep_axes")

# ---------------------------------------------------------------- 2) 配置 --
CFG = ROOT / "autoresearch" / "config.py"
c = CFG.read_text(encoding="utf-8")

if "max_sweep_runs" not in c:
    # 找一个已有的 max_variants 声明作为锚点
    m = re.search(r"^(\s*)max_variants\s*:\s*int\s*=\s*(\d+)", c, re.M)
    if m:
        indent = m.group(1)
        insert = (
            f"{m.group(0)}\n"
            f"{indent}#: 单次参数扫描的运行数上限。**与 max_variants 分开**：\n"
            f"{indent}#: 变体预算默认 2（baseline/method），而一次材料参数扫描\n"
            f"{indent}#: 天然要 6-18 组；共用一个旋钮会让\"想跑扫描\"变成\n"
            f"{indent}#: \"必须先把变体上限调大\"，语义混淆。\n"
            f"{indent}max_sweep_runs: int = 24"
        )
        c = c.replace(m.group(0), insert, 1)
        print("  config.py: 已加 max_sweep_runs")
    else:
        print("  MISS config.py: 找不到 max_variants 锚点")

    # 环境变量
    if "AUTORESEARCH_MAX_SWEEP_RUNS" not in c:
        em = re.search(r'("max_variants"\s*:\s*[^,]+,\s*\n)', c)
        if em:
            c = c.replace(
                em.group(1),
                em.group(1)
                + '    #: 见 max_sweep_runs 的说明\n'
                + '    "max_sweep_runs": "AUTORESEARCH_MAX_SWEEP_RUNS",\n',
                1,
            )
            print("  config.py: 已加 AUTORESEARCH_MAX_SWEEP_RUNS")
        else:
            print("  MISS config.py: ENV_KEYS 锚点")
    CFG.write_text(c, encoding="utf-8")
else:
    print("  config.py: 已有 max_sweep_runs")

import py_compile  # noqa: E402

for f in (BASE, CFG):
    py_compile.compile(str(f), doraise=True)
print("  编译通过")
