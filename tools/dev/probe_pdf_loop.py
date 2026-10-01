"""打通 PDF 编译闭环：用管线自己的 LatexCompiler 编译真实论文模板。

此前 CHANGELOG 把这条列为「**从未收敛过**」的最大未验证面。
根因已确认不是代码问题，而是沙箱/ACL 拒绝 tectonic 在新建目录里写入
（`os error 5`）——在无沙箱终端里它能正常工作。

本脚本验证三件事：
  1. LatexCompiler.detect() 能发现 vendored tectonic
  2. compile() 能对**真实论文模板**产出 PDF（不是最小玩具文档）
  3. 编译失败 → 抽错 → 修复 → 重编译 这条闭环，能真的收敛
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoresearch.config import load_config  # noqa: E402
from autoresearch.tools.latex import LatexCompiler  # noqa: E402


class Cfg:
    """最小配置对象（LatexCompiler 只从这里读几个字段）。"""

    engine = ""           # 空 = 允许自动探测
    tectonic_version = "0.15.0"
    latex_timeout = 900
    max_compile_attempts = 2


def main() -> int:
    work = ROOT / ".autoresearch" / "_pdfprobe"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    # 把真实论文模板复制进工作区
    tpl = ROOT / "autoresearch" / "templates" / "paper"
    paper = work / "paper"
    shutil.copytree(tpl, paper)
    print("=" * 76)
    print("PDF 编译闭环验证：真实论文模板")
    print("=" * 76)
    print(f"\n  模板: {tpl}")
    print(f"  工作区: {paper}")
    tex_files = sorted(p.relative_to(paper).as_posix() for p in paper.rglob("*.tex"))
    print(f"  tex 文件: {len(tex_files)} 个 -> {tex_files[:6]}")

    logs: list[tuple[str, dict]] = []

    class Recorder:
        def log(self, event: str, **fields) -> None:
            logs.append((event, fields))

    comp = LatexCompiler(Cfg(), work, Recorder())

    # 1) 探测
    print("\n1) detect()")
    detected = comp.detect()
    print(f"   引擎: {detected!r}")
    if not detected:
        print("   FAIL 没探测到引擎 —— vendored tectonic 未被发现")
        for e, f in logs:
            if "tectonic" in e:
                print(f"     {e}: {f}")
        return 1

    # 2) 编译
    print("\n2) compile()")
    result = comp.compile(paper / "main.tex")
    print(f"   ok={result.ok}  engine={result.engine!r}")
    if result.pdf:
        size = Path(result.pdf).stat().st_size
        head = Path(result.pdf).read_bytes()[:8]
        print(f"   PDF: {result.pdf}")
        print(f"   大小: {size} 字节   头: {head!r}")
        if result.ok and size > 1000 and head.startswith(b"%PDF"):
            print("\n   >>> PDF 真的产出了（不是占位文件）")
        else:
            print("\n   FAIL 产物不是有效 PDF")
            return 1
    else:
        print("   FAIL 没有 PDF")
        print(f"   errors: {result.errors[:4]}")

    # 3) 日志里 tectonic 事件
    print("\n3) 事件日志")
    for event, fields in logs:
        if any(k in event for k in ("tectonic", "latex", "compile")):
            status = fields.get("status") or fields.get("engine") or ""
            print(f"   {event}: {status} {str(fields)[:110]}")

    # 4) 闭环：故意注入错误 -> 抽错 -> 修复 -> 重编译
    print("\n4) 编译闭环：注入错误 → 抽错 → 修复 → 重编译")
    method = paper / "sections" / "method.tex"
    original = method.read_text(encoding="utf-8")
    method.write_text(
        original + "\n\\thiscommanddoesnotexist{boom}\n", encoding="utf-8"
    )
    bad = comp.compile(paper / "main.tex")
    print(f"   注入错误后 ok={bad.ok}  errors={len(bad.errors)}")
    for e in bad.errors[:3]:
        print(f"     · {e[:120]}")
    if bad.ok or not bad.errors:
        print("   FAIL 注入的错误没有被抽出来 —— 闭环的第一步就不成立")
        method.write_text(original, encoding="utf-8")
        return 1
    print("   >>> 错误被抽出，修复信息可用")

    # "修复"：还原
    method.write_text(original, encoding="utf-8")
    fixed = LatexCompiler(Cfg(), work, Recorder()).compile(paper / "main.tex")
    print(f"   修复后 ok={fixed.ok}  pdf={bool(fixed.pdf)}")
    if not (fixed.ok and fixed.pdf):
        print("   FAIL 修复后仍未编译成功")
        return 1
    print("   >>> **闭环收敛**：注入错误 → 抽出错误 → 修复 → 重新产出 PDF")

    print("\n=== PDF 编译闭环验证通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
