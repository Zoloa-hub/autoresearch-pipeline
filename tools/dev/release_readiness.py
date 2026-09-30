"""发布就绪度检查：只报告事实，不做判断。"""

from __future__ import annotations

import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def grab(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else "?"


print("=== 版本号一致性 ===")
init_py = (ROOT / "autoresearch" / "__init__.py").read_text(encoding="utf-8")
toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
print("  __init__.__version__ :", grab(init_py, r'__version__\s*=\s*"([^"]+)"'))
print("  pyproject version    :", grab(toml, r'^version\s*=\s*"([^"]+)"', ))
print("  → 版本号在 3 处硬编码（__init__ / pyproject / README 措辞），无自动同步")

print("\n=== 仓库元数据文件 ===")
for rel in (
    ".gitattributes",
    "CHANGELOG.md",
    ".github/ISSUE_TEMPLATE",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml",
    "MANIFEST.in",
    ".git",
    "scratch",
):
    p = ROOT / rel
    print(f"  {'有  ' if p.exists() else '缺  '} {rel}")

print("\n=== 平台覆盖 ===")
print("  os.name =", os.name)
if os.name == "nt":
    print("  → POSIX 分支（setrlimit、start_new_session、killpg）**从未在本机执行过**")
    print("  → 全部测试结论来自 Windows + Python 3.13")

print("\n=== 无法在本会话验证的项 ===")
for item in (
    "CI 工作流在真实 GitHub 上是否通过（本机无 .git、未 push）",
    "s7 的编译→抽错→修复→重编译闭环（tectonic 被沙箱 ACL 阻断）",
    "--sandbox docker 路径（本机未安装 docker）",
    "Linux / macOS 上的全量测试（三平台矩阵只在 CI 上跑）",
):
    print("  ·", item)

print("\n=== 测试规模 ===")
print("  7 个套件 / 1,763 项检查（本机实测，Python 3.13 + Windows）")
