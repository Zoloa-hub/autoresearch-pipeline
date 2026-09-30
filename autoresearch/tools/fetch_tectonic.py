#!/usr/bin/env python
"""按需下载 tectonic（单文件 LaTeX 引擎），不把二进制放进版本控制。

为什么需要这个脚本：tectonic 约 48 MB，且 Windows/Linux/macOS 是**不同的可执行
文件**。把它签入仓库会让 git 历史永久膨胀，并且在一个平台上克隆下来在另一个平台
上不能用。所以仓库里只有这个几十行的下载器，二进制按需获取。

用法::

    python -m autoresearch.tools.fetch_tectonic            # 下载并验证
    python -m autoresearch.tools.fetch_tectonic --check    # 只检查现状，不下载

等价的手工命令（不想跑 Python 时）::

    # Windows
    curl -L -o tect.zip https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip
    tar -xf tect.zip -C autoresearch/vendor/tectonic

    # Linux
    curl -L https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz | tar -xz -C autoresearch/vendor/tectonic

下载后 `LatexCompiler` 会自动发现它；也可以在 PATH 里放系统级的 tectonic。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接作为脚本运行（python autoresearch/tools/fetch_tectonic.py）
if __package__ in (None, ""):  # pragma: no cover - 脚本入口
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_tectonic",
        description="下载 tectonic 单文件可执行版（约 48 MB），供 s7 阶段编译 PDF。",
    )
    parser.add_argument("--version", default=None,
                        help="tectonic 版本（默认沿用配置，通常是 0.15.0）")
    parser.add_argument("--check", action="store_true",
                        help="只检查是否已可用，不下载")
    parser.add_argument("--force", action="store_true",
                        help="即使已存在也重新下载")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from autoresearch.config import PROJECT_ROOT, load_config
    from autoresearch.tools.latex import LatexCompiler

    cfg = load_config()
    if args.version:
        cfg.compile.tectonic_version = args.version

    compiler = LatexCompiler(cfg.compile, PROJECT_ROOT / ".fetch_tectonic")

    if not args.force:
        engine = compiler.detect()
        if engine:
            print(f"✔ LaTeX 引擎已可用：{engine}")
            reason = compiler.cache_block_reason()
            if reason:
                print(f"  注意：tectonic 已安装但预检未通过：\n  {reason}")
                return 1
            return 0
        existing_reason = compiler.cache_block_reason()
        if existing_reason:
            print("检测到 tectonic 已下载但**无法使用**，原因：")
            print(f"  {existing_reason}")
            print("\n这不是下载问题，重新下载不会解决。请按上面的提示放开写入权限，")
            print("或改用 `pdflatex` / Overleaf。")
            return 1

    if args.check:
        print("✘ 未检测到可用的 LaTeX 引擎（tectonic / pdflatex / xelatex）")
        return 1

    print(f"正在下载 tectonic {cfg.compile.tectonic_version} …")
    path = compiler.install_tectonic()
    if path is None:
        print("✘ 下载或预检失败。可能原因：")
        print("  · 网络不可达 GitHub Releases；")
        print("  · 平台/架构不在支持列表内（仅 x86_64 的 Windows / Linux / macOS）；")
        print("  · 当前沙箱拒绝 tectonic 写它的 bundle 缓存（报 os error 5）。")
        print(f"  详细原因见事件日志；也可手动下载到 {PROJECT_ROOT / 'vendor' / 'tectonic'}")
        return 1

    print(f"✔ 已安装并验证：{path}")
    print("  现在可以重新运行：python -m autoresearch.cli resume <run_id>")
    print("  （s7 阶段会自动发现它，前面的阶段不会重跑）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
