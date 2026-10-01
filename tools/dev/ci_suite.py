#!/usr/bin/env python3
"""在 CI 里跑一个测试套件，失败时把**具体失败项**写成 GitHub annotation。

## 为什么需要它

公开仓库的 **job 日志正文需要 admin 权限**才能通过 API 读取：

    GET /repos/{owner}/{repo}/actions/runs/{id}/logs
    -> 403 "Must have admin rights to Repository."

但 **annotation 是匿名可读的**：

    GET /repos/{owner}/{repo}/check-runs/{id}/annotations
    -> 200

于是出现过一个很别扭的处境：CI 红了，而自动生成的 annotation 只有一句
`Process completed with exit code 1.`——**谁都看不出失败在哪**，包括作者。
排查时必须靠人手点开网页复制日志，这在自动化流程里是个断点。

这个脚本把套件输出里的失败项主动转成 `::error::` annotation，
于是**失败原因本身**变成匿名可读的。上限 10 条（GitHub 的限制），
所以按重要性截断：先报失败计数，再报具体的 FAIL 行。

## 用法

    python tools/dev/ci_suite.py test_sandbox_latex

退出码与套件一致，所以 CI 的成败判定不受影响。
"""

from __future__ import annotations

import os
import subprocess
import sys


def _harden_stdio() -> None:
    """让本脚本的标准输出**永不因非 ASCII 而崩**。

    这个包装器会把套件的完整输出（含大量中文标签与 SKIP 说明）打印出来。
    在非 UTF-8 控制台下（Windows 的 cp1252/cp936、某些 CI 容器的 POSIX locale），
    `print()` 一个中文串会抛 `UnicodeEncodeError`，**脚本当场死掉**——于是
    一个本来只是"套件失败"的情况，变成"包装器崩溃、连 annotation 都没输出"，
    而 CI 上看到的是多个套件同时失败，极难归因到真正的源头。

    这是**同一个坑的第二次**：第一次在 autoresearch/__init__.py 的包入口
    （那里的日志与报告大量使用中文）。区别是这个脚本在包外面，
    享受不到包入口的加固，所以必须自带一份。

    `backslashreplace` 而不是 `replace`：保留可读的 ``\u4e2d``，
    信息量更大，也不会让编码问题掩盖真正的失败。
    """
    import sys as _sys

    for stream, errors in ((_sys.stdout, "backslashreplace"), (_sys.stderr, "replace")):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors=errors)
        except (ValueError, OSError, AttributeError):
            try:
                stream.reconfigure(errors=errors)
            except Exception:  # noqa: BLE001 - 加固本身绝不能成为失败源
                pass


_harden_stdio()



#: GitHub 每个 step 最多接受 10 条 error annotation
MAX_ANNOTATIONS = 8
#: 单条 annotation 的长度上限（GitHub 约 64KB，这里刻意保守）
MAX_CHARS = 900


def _failure_lines(text: str) -> list[str]:
    """从套件输出里挑出最值得上报的行。"""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        # 各套件的失败格式：`  - <label>` 或 `FAIL: <label>`
        if line.startswith("- ") or line.startswith("FAIL:"):
            out.append(line)
        elif line.startswith("FAILED ") and "checks" in line:
            out.insert(0, line)  # 计数放最前面
    # 去重保序
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("用法: python tools/dev/ci_suite.py <module> [args...]")
        return 2
    module = args[0]
    full = f"autoresearch.tests.{module}" if not module.startswith("autoresearch.") else module

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", full, *args[1:]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    print(text)

    if proc.returncode == 0:
        # 即使成功也把跳过情况报成 notice，方便一眼看出"哪些没真的测到"
        skips = [ln.strip() for ln in text.splitlines() if "SKIP" in ln]
        for line in skips[:4]:
            print("::notice::" + line[:MAX_CHARS])
        return 0

    failures = _failure_lines(text)
    print(f"::error::{module} 失败；共捕获 {len(failures)} 条问题（下面最多展示 {MAX_ANNOTATIONS} 条）")
    if not failures:
        # 没有可识别的 FAIL 行 —— 说明进程是中途崩的，把尾部原文报上去
        print(f"::error::{module} 输出里没有可识别的失败行，可能是中途崩溃；尾部输出：")
        for line in text.strip().splitlines()[-6:]:
            print("::error::" + line.strip()[:MAX_CHARS])
    else:
        for item in failures[:MAX_ANNOTATIONS]:
            print("::error::" + item[:MAX_CHARS])
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
