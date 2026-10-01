"""PDF 编译闭环 —— 用**正确方式**验证（先替换占位符，再编译）。

## 上一次探针错在哪

我直接编译了裸模板 `templates/paper/main.tex`，得到：

    error: main.tex:66: Missing $ inserted

第 66 行是空行。真正的原因是 `\\maketitle` 在处理 **未替换的 `__TITLE__`**——
LaTeX 里 `_` 是数学模式专用字符，`__TITLE__` 必然触发 "Missing $ inserted"。

模板本来就不该裸编译：s6 会先替换 5 个占位符。这顺带**确认了占位符自检的价值**——
未替换时报的错误完全不指向真正的原因（指向一个空行）。

## 本脚本验证

  1. detect() 发现 vendored tectonic
  2. 填充占位符后 compile() 产出**真实 PDF**
  3. 闭环：注入一个 LaTeX 错误 → 抽出错误 → 修复 → 重新产出 PDF
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.tools.latex import LatexCompiler  # noqa: E402

#: 模板声明的 5 个占位符与填充值（模拟 s6 的产物）
FILL = {
    "__TITLE__": "Auditable Experiment Execution for Computational Materials",
    "__AUTHORS__": "A. Author\\and B. Author",
    "__DATE__": "October 2026",
    "__ABSTRACT__": (
        "We present a nine-stage pipeline that keeps every number in the paper "
        "traceable to an artifact on disk, and reports its own failures instead "
        "of emitting a plausible-looking empty result."
    ),
    "__KEYWORDS__": "reproducibility, experiment automation, auditability",
}


class Cfg:
    engine = ""
    tectonic_version = "0.15.0"
    latex_timeout = 900
    max_compile_attempts = 2


def fill_placeholders(paper: Path) -> list[str]:
    """替换全部占位符——模拟 s6 组装论文的那一步。"""
    unfilled: list[str] = []
    for tex in sorted(paper.rglob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        for key, value in FILL.items():
            text = text.replace(key, value)
        tex.write_text(text, encoding="utf-8")
        for match in re.finditer(r"__[A-Z][A-Z0-9_]*__", text):
            unfilled.append(f"{tex.name}:{match.group(0)}")
    return unfilled


def main() -> int:
    work = ROOT / ".autoresearch" / "_pdfprobe2"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    paper = work / "paper"
    shutil.copytree(ROOT / "autoresearch" / "templates" / "paper", paper)

    print("=" * 78)
    print("PDF 编译闭环（正确方式：先替换占位符）")
    print("=" * 78)

    left = fill_placeholders(paper)
    print(f"\n  占位符替换完成；残留未填: {left if left else '（无）'}")

    logs: list[tuple[str, dict]] = []

    class Recorder:
        def log(self, event: str, **fields) -> None:
            logs.append((event, fields))

    comp = LatexCompiler(Cfg(), work, Recorder())

    print("\n1) detect()")
    engine = comp.detect()
    print(f"   引擎: {engine!r}")
    if not engine:
        print("   FAIL 未发现引擎")
        return 1

    print("\n2) compile()")
    res = comp.compile(paper / "main.tex")
    print(f"   ok={res.ok}  engine={res.engine!r}")
    if not res.ok:
        print(f"   errors: {[e[:120] for e in res.errors[:4]]}")
        return 1
    pdf = Path(res.pdf)
    head = pdf.read_bytes()[:8]
    size = pdf.stat().st_size
    print(f"   PDF: {pdf.name}  {size} 字节  头 {head!r}")
    if not (size > 1000 and head.startswith(b"%PDF")):
        print("   FAIL 产物不是有效 PDF")
        return 1
    print("   >>> 真实 PDF 产出")

    print("\n3) 闭环：注入 LaTeX 错误 → 抽错 → 修复 → 重编译")
    target = paper / "sections" / "results.tex"
    original = target.read_text(encoding="utf-8")
    # 注入一个**典型**错误：未闭合的数学模式
    target.write_text(original + "\n我们的方法在 $x = 1 上更优。\n", encoding="utf-8")
    bad = comp.compile(paper / "main.tex")
    print(f"   注入后 ok={bad.ok}  抽出 {len(bad.errors)} 条错误")
    for e in bad.errors[:3]:
        print(f"     · {e[:130]}")
    # 注意：注入错误后 **ok 仍可能为 True** —— tectonic 产出了 PDF。
    # 这正是 s7 原本静默放行带错误论文的原因，所以判据必须看 errors。
    if not bad.errors:
        print("   FAIL 注入的错误没被抽出")
        target.write_text(original, encoding="utf-8")
        return 1
    print(f"   >>> 错误被抽出（可供 LLM 修复）；注意 ok={bad.ok} —— "
          "ok 只表示产出了 PDF，不代表没有 LaTeX 错误")

    target.write_text(original, encoding="utf-8")
    fixed = LatexCompiler(Cfg(), work, Recorder()).compile(paper / "main.tex")
    print(f"   修复后 ok={fixed.ok}  pdf={bool(fixed.pdf)}")
    if not (fixed.ok and fixed.pdf):
        print(f"   FAIL errors={[e[:110] for e in fixed.errors[:3]]}")
        return 1
    print("   >>> **闭环收敛**：错误 → 抽出 → 修复 → 重新产出 PDF")

    print("\n=== PDF 编译闭环验证通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
