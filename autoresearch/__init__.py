"""Auto-Research: 自动化科研管线（离线可跑）.

包入口只做最小暴露：配置层 API + 版本号。
其余模块（llm / tools / stages ...）按需导入，保证 `import autoresearch`
在零第三方依赖环境下也能成功。

同时在这里做一次**标准输出编码加固**：见下面 `_harden_stdio()`。
"""

from __future__ import annotations

import sys as _sys

from .config import DEFAULT_RUNS_DIR, PROJECT_ROOT, WORKSPACE_ROOT
from .config import AutoResearchConfig, load_config

__version__ = "0.1.0"


def _harden_stdio() -> None:
    """确保标准输出/错误**永不因非 ASCII 字符而崩**。

    为什么必须在包入口做：本包的日志与报告大量使用中文（「致命」「降级」
    「未溯源」…）以及少量 emoji。在非 UTF-8 控制台下（Windows 的 cp1252/cp936、
    某些 CI 容器的 POSIX locale），`print()` 一个中文串会抛 `UnicodeEncodeError`。

    这个失败的形态**极难诊断**：进程当场死掉，标准输出缓冲区随之丢失，
    于是看到的是「输出停在中途 + 退出码 1」，看不到任何 traceback。
    在 GitHub Windows runner 上实测就是这样——套件跑到某个 section 就断了。

    处理方式对齐 CPython 自己的选择（PEP 540 / pytest 的做法）：
      * 可以重配就重配成 UTF-8；重配不成，退而把错误处理设为不抛异常；
      * stdout 用 ``backslashreplace``——保留可读的 ``\\u4e2d`` 而不是 ``?``，
        信息量更大，也不会因为编码问题掩盖真正的失败；
      * stderr 用 ``replace``（诊断信息宁可有损也不要二次崩）。
    """
    for stream, errors in ((_sys.stdout, "backslashreplace"), (_sys.stderr, "replace")):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # 被重定向到非 TextIOWrapper（例如 pytest 的捕获对象）
        try:
            reconfigure(encoding="utf-8", errors=errors)
        except (ValueError, OSError, AttributeError):
            # 已经写入了内容、或流不支持重配：至少把错误处理改成不抛
            try:
                stream.reconfigure(errors=errors)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - 加固本身绝不能成为新的失败源
                pass


_harden_stdio()

__all__ = [
    "__version__",
    "load_config",
    "AutoResearchConfig",
    "PROJECT_ROOT",
    "WORKSPACE_ROOT",
    "DEFAULT_RUNS_DIR",
]
