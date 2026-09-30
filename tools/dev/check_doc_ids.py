"""一次性校验脚本：确认文档里引用的每个标识符都真实存在于代码中。

不是测试套件的一部分——它服务于文档评审，用完即可删除。
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [ROOT / "docs" / "adapters.md", ROOT / "CONTRIBUTING.md", ROOT / "autoresearch" / "README.md"]


def main() -> int:
    code_paths = [
        p for p in (ROOT / "autoresearch").rglob("*.py") if "__pycache__" not in p.parts
    ]
    blob = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in code_paths)
    md_blob = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in (ROOT / "autoresearch").rglob("*.md")
    )
    everything = blob + "\n" + md_blob

    for doc in DOCS:
        if not doc.is_file():
            print(f"{doc.relative_to(ROOT)}: 不存在")
            continue
        text = doc.read_text(encoding="utf-8")
        print(f"{doc.relative_to(ROOT)}: {len(text.splitlines())} 行")

    ticked: set[str] = set()
    for doc in DOCS:
        if not doc.is_file():
            continue
        for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_.]{2,60})`", doc.read_text(encoding="utf-8")):
            token = m.group(1)
            if token.startswith("--") or "/" in token or token.endswith(".py"):
                continue
            ticked.add(token)

    missing = []
    for token in sorted(ticked):
        leaf = token.split(".")[-1]
        if len(leaf) < 3:
            continue
        if leaf not in everything:
            missing.append(token)

    print(f"\n反引号标识符 {len(ticked)} 个")
    print("代码/文档中找不到的:", missing if missing else "（无）")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
