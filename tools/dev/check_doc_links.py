"""检查全部 Markdown 里的相对链接与代码引用是否真实存在。

文档重写最容易留下悬空引用（链接到已改名的文件、提到已删除的函数）。
这个脚本把它们一次性列出来。
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
MD = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "docs" / "adapters.md",
    ROOT / "autoresearch" / "README.md",
    ROOT / "autoresearch" / "CONTRACTS.md",
    ROOT / "autoresearch" / "prompts" / "README.md",
    ROOT / "autoresearch" / "templates" / "experiment" / "README.md",
    ROOT / "autoresearch" / "templates" / "paper" / "compile.md",
    ROOT / "autoresearch" / "templates" / "paper" / "protocol.md",
]

LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def main() -> int:
    broken_links: list[str] = []
    broken_anchors: list[str] = []
    checked = 0

    for doc in MD:
        if not doc.is_file():
            continue
        text = doc.read_text(encoding="utf-8")
        for _label, target in LINK.findall(text):
            target = target.strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            path_part = target.split("#")[0]
            if not path_part:
                continue
            resolved = (doc.parent / path_part).resolve()
            if not resolved.exists():
                broken_links.append(f"{doc.relative_to(ROOT)} -> {target}")

    # 代码引用：反引号里的 Python 路径与符号
    py_blob = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in (ROOT / "autoresearch").rglob("*.py")
        if "__pycache__" not in p.parts
    )
    md_blob = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in (ROOT / "autoresearch").rglob("*.md")
    ) + "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in MD if p.is_file()
    )
    everything = py_blob + "\n" + md_blob

    missing_symbols: list[str] = []
    for doc in MD:
        if not doc.is_file():
            continue
        for token in set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]{3,60})`", doc.read_text(encoding="utf-8"))):
            if token.startswith("--") or "/" in token or token.endswith((".py", ".md", ".toml", ".tex")):
                continue
            leaf = token.split(".")[-1]
            if len(leaf) < 4:
                continue
            if leaf not in everything:
                missing_symbols.append(f"{doc.relative_to(ROOT)}: {token}")

    print(f"检查了 {checked} 个相对链接")
    print(f"  断裂链接: {broken_links if broken_links else '（无）'}")
    print(f"  无法核实的代码符号: {sorted(set(missing_symbols)) if missing_symbols else '（无）'}")
    print(f"  （锚点检查: {'未发现' if not broken_anchors else broken_anchors}）")
    return 1 if (broken_links or missing_symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
