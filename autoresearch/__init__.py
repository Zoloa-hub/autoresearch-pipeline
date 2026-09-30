"""Auto-Research: 自动化科研管线（离线可跑）.

包入口只做最小暴露：配置层 API + 版本号。
其余模块（llm / tools / stages ...）按需导入，保证 `import autoresearch`
在零第三方依赖环境下也能成功。
"""

from __future__ import annotations

from .config import DEFAULT_RUNS_DIR, PROJECT_ROOT, WORKSPACE_ROOT
from .config import AutoResearchConfig, load_config

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_config",
    "AutoResearchConfig",
    "PROJECT_ROOT",
    "WORKSPACE_ROOT",
    "DEFAULT_RUNS_DIR",
]
