"""纠正一个被我反复引用却没有核对的数字：13/19 → 10/18。

## 怎么错的

我在一次快速统计后记下了「19 个指标、5 个判反、6 个无方向、13/19 不可信」，
之后**再没核对过**，却把它写进了 README、CHANGELOG、两处模块注释和一处测试注释。

重新逐条数（18 个指标）：

    判反        5   k, alpha, resistivity, corrosion_rate, sintering_temp
    无固有方向  5   n, reflectance, band_gap, youngs_modulus, thermal_conductivity
    判对        8   transmittance, conductivity, hardness, yield_strength,
                    dielectric_loss, seeck, zt, d33
    ------------------------------------------------
    不可信     10/18

两处都错：总数写成 19（实际 18），无方向写成 6（实际 5）。

## 为什么必须单独修

这个错误恰好是本项目一整套机制要防的东西：**一个未核对的数字被重复引用，
直到它看起来像事实**。它已经进了 git 提交信息（不可改），所以必须在
CHANGELOG 里留下更正记录，否则以后有人按 13/19 去引用会再次传播。
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXES = [
    # (文件, 旧, 新)
    (
        "autoresearch/adapters/materials_optics.py",
        "最后一条正是通用 token 表会判错的地方。实测 19 个材料学指标里\n"
        "**5 个判反、6 个无方向却被静默选了一个**，合计 13/19 不可信。所以本适配器",
        "最后一条正是通用 token 表会判错的地方。实测 18 个材料学指标里\n"
        "**5 个判反、5 个无固有方向却被静默选了一个**，合计 10/18 不可信。所以本适配器",
    ),
    (
        "autoresearch/stages/s4_experiment.py",
        "显式声明——通用词表只是兜底，且对非 ML 领域并不可靠（材料学实测 13/19 不可信）。",
        "显式声明——通用词表只是兜底，且对非 ML 领域并不可靠（材料学实测 10/18 不可信）。",
    ),
    (
        "autoresearch/tests/test_adapters.py",
        "（loss/accuracy/f1/bleu…），在材料学实测 19 个指标里 **5 个判反、6 个无方向\n"
        "    却被静默选了一个**，合计 13/19 不可信。",
        "（loss/accuracy/f1/bleu…），在材料学实测 18 个指标里 **5 个判反、5 个无固有方向\n"
        "    却被静默选了一个**，合计 10/18 不可信。",
    ),
    (
        "README.md",
        "**材料学：19 个常见指标里 13 个方向不可信**",
        "**材料学：18 个常见指标里 10 个方向不可信**",
    ),
    (
        "README.md",
        "换到材料学实测 13/19 不可信。方向判错不报错——这是所有失败模式里最难发现的一种。",
        "换到材料学实测 10/18 不可信（5 个判反 + 5 个无固有方向却被静默选了一个）。\n方向判错不报错——这是所有失败模式里最难发现的一种。",
    ),
    (
        "CHANGELOG.md",
        "（loss/accuracy/f1/bleu…）。在 19 个材料学常见指标上实测：",
        "（loss/accuracy/f1/bleu…）。在 18 个材料学常见指标上实测：",
    ),
    (
        "CHANGELOG.md",
        "即 **13/19 不可信**。而方向判错**不会报错**——它只会让「改善」的定义反过来。",
        "即 **10/18 不可信**。而方向判错**不会报错**——它只会让「改善」的定义反过来。",
    ),
]

CORRECTION = '''
> **更正记录（一处被我错误引用的数字）。** 上文的"不可信比例"最初写成
> **13/19**（5 个判反 + 6 个无方向）。重新逐条核对后，正确数字是
> **10/18**：18 个指标里 5 个判反、5 个无固有方向、8 个判对。
> 两处都错了——总数写成 19（实际 18），无方向写成 6（实际 5）。
>
> 之所以单独记一笔：这个数字我**记下后就没有再核对过**，却写进了 README、
> CHANGELOG 和两处模块注释。**一个未核对的数字被重复引用到看起来像事实**，
> 恰好是本项目整套机制要防的那种错误。提交信息里改不掉了，
> 所以在正文留下更正，避免有人继续按 13/19 引用。

'''

applied = 0
for rel, old, new in FIXES:
    p = ROOT / rel
    if not p.is_file():
        print(f"  MISS 文件不存在: {rel}")
        continue
    t = p.read_text(encoding="utf-8")
    if old in t:
        p.write_text(t.replace(old, new, 1), encoding="utf-8")
        applied += 1
        print(f"  OK   {rel}: 已修正")
    elif new in t:
        print(f"  SKIP {rel}: 已是新值")
    else:
        print(f"  MISS {rel}: 锚点未匹配")

# CHANGELOG 里加更正记录
cl = ROOT / "CHANGELOG.md"
ct = cl.read_text(encoding="utf-8")
if "更正记录（一处被我错误引用的数字）" not in ct:
    anchor = "即 **10/18 不可信**"
    idx = ct.find(anchor)
    if idx >= 0:
        end = ct.find("\n\n", idx)
        ct = ct[: end + 1] + CORRECTION + ct[end + 1 :]
        cl.write_text(ct, encoding="utf-8")
        print("  CHANGELOG: 已加更正记录")
    else:
        print("  MISS CHANGELOG: 找不到插入点")

# 全仓复查
import subprocess  # noqa: E402

leftover = []
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix not in (".md", ".py"):
        continue
    if any(part in str(p) for part in (".autoresearch", "__pycache__", "scratch")):
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    if "13/19" in text or "19 个材料学" in text or "19 个常见指标" in text:
        leftover.append(p.relative_to(ROOT).as_posix())
print()
print(f"  已修正 {applied} 处；残留 13/19 : {leftover if leftover else '（无）'}")
