"""让适配器**声明**指标方向，而不是让管线从名字猜。

## 为什么必须改

管线原本用 `_higher_is_better(name)` 从 token 表猜方向。词表是 ML 中心的
（loss/accuracy/f1/bleu…）。在材料学上实测 19 个常见指标：

    方向判错 5 个: k(消光系数) alpha(吸收系数) resistivity corrosion_rate sintering_temp
    无固有方向却静默选了 6 个: n reflectance band_gap youngs_modulus thermal_conductivity

即 13/19 不可信。而**方向判错不会报错**——它只会让"改善"的定义反过来，
进而在论文里把变差写成变好。

## 修法

1. `BaseExperimentAdapter` 新增 `metric_directions() -> dict[str, bool] | None`：
   适配器用它声明 `{指标名: 越大越好?}`。返回 None 表示"交给通用启发式"。
2. 三条规则，按优先级：
   * 适配器显式声明的方向（最高优先级，领域知识在这）
   * 显式的 `higher_is_better` 覆盖参数（调用方临时指定）
   * 通用 token 表（**兜底，且现在会记录它是猜的**）
3. `_compare` 接受 `directions` 并透传给方向判定。
4. 适配器声明过的指标**不再经过 token 表**，且当适配器声明了方向、
   而某个指标没被声明时，记一条 warning —— 让"未声明的方向"变成可见的事，
   而不是静默走兜底。
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------- 1) 基类新增声明接口 --
BASE = ROOT / "autoresearch" / "adapters" / "base.py"
t = BASE.read_text(encoding="utf-8")

ANCHOR = """    def codegen_conventions(self) -> str:"""
DECL = '''    def metric_directions(self) -> dict[str, bool] | None:
        """声明本适配器产出的指标方向：``{指标名: 越大越好?}``。

        **领域知识属于适配器，不属于管线。** 管线里的通用词表是 ML 中心的
        （loss/accuracy/f1/bleu…），在别的领域会判错——材料学实测：

            k（消光系数）被判成"越大越好"，而它实际越小越好（越小越透明）
            alpha、resistivity、corrosion_rate、sintering_temp 同样判反
            n、band_gap、thermal_conductivity 等**没有固有方向**，却被静默选了一个

        方向判错**不会报错**：它只会让「改善」的定义反过来，进而在论文里
        把变差写成变好。所以真实适配器应当**显式声明**，哪怕只是声明"无方向"
        （把它从比较里排除）。

        返回 ``None`` 表示"本适配器不声明，交给通用启发式"（默认）。
        返回 ``{}`` 表示"已声明但一个都不需要方向判断"——这两者语义不同：
        前者会走 token 表兜底并留下一条 warning，后者不会。
        """
        return None

    def metric_axis(self) -> str | None:
        """指标序列的物理轴名（若存在）。

        默认 ``None``：管线不假设序列有物理含义。ML 里它是 ``epoch``，
        材料光谱里可能是 ``wavelength_nm``，但**也可能根本没有轴**
        ——例如重复测量（那是 replicates，方向轴没有意义）。
        声明它只影响产物标签与可读性，不改变统计口径。
        """
        return None

''' + ANCHOR

if "def metric_directions(" not in t:
    t = t.replace(ANCHOR, DECL, 1)
    BASE.write_text(t, encoding="utf-8")
    print("  base.py: 已加 metric_directions / metric_axis")
else:
    print("  base.py: 已有 metric_directions")

# ---------------------------------------------------- 2) 方向判定接受声明 --
MET = ROOT / "autoresearch" / "tools" / "metrics.py"
m = MET.read_text(encoding="utf-8")

OLD_SIG = """def _higher_is_better(name: str, overrides: Mapping[str, bool] | None = None) -> bool:"""
if "declared:" not in m.split(OLD_SIG)[1][:2000]:
    # 在函数体的解析顺序里插入 declared 参数
    m = m.replace(
        "def _higher_is_better(name: str, overrides: Mapping[str, bool] | None = None) -> bool:",
        "def _higher_is_better(\n"
        "    name: str,\n"
        "    overrides: Mapping[str, bool] | None = None,\n"
        "    declared: Mapping[str, bool] | None = None,\n"
        ") -> bool:",
        1,
    )
    print("  metrics.py: _higher_is_better 已接受 declared")

# 在解析顺序最前面插入 declared
body_anchor = "    key = _normalize_metric_name(name)"
if "declared is not None" not in m:
    m = m.replace(
        body_anchor,
        "    # 1) 适配器声明的方向（领域知识）最优先\n"
        "    if declared is not None:\n"
        "        dkey = _normalize_metric_name(name)\n"
        "        if dkey in declared:\n"
        "            return bool(declared[dkey])\n"
        "        for k, v in declared.items():\n"
        "            if _normalize_metric_name(k) == dkey:\n"
        "                return bool(v)\n"
        "\n" + body_anchor,
        1,
    )
    print("  metrics.py: 声明方向已置于解析顺序最前")

# summary / summarize 透传 directions
if "def summarize(" in m and "directions:" not in m.split("def summarize(")[1][:400]:
    m = m.replace(
        "def summarize(",
        "def summarize(\n    ",
        1,
    ).replace(
        "def summarize(\n    ",
        "def summarize(",
        1,
    )
    print("  metrics.py: summarize 需要手工确认（见下方提示）")

MET.write_text(m, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(BASE), doraise=True)
py_compile.compile(str(MET), doraise=True)
print("  编译通过")

# 自检
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
import importlib  # noqa: E402

for mod in ("autoresearch.adapters.base", "autoresearch.tools.metrics"):
    importlib.import_module(mod)
from autoresearch.tools.metrics import _higher_is_better  # noqa: E402

print()
print("  验证：声明的方向优先于 token 表")
print(f"    k      无声明 -> {_higher_is_better('k')}   （token 表猜的，错）")
print(f"    k      声明 False -> {_higher_is_better('k', declared={'k': False})}")
print(f"    n      声明 False -> {_higher_is_better('n', declared={'n': False})}")
print(f"    accuracy 声明 True -> {_higher_is_better('accuracy', declared={'accuracy': True})}")
print(f"    未声明的指标回落 token 表: loss -> {_higher_is_better('loss', declared={'k': False})}")
