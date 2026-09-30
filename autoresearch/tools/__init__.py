"""``autoresearch.tools`` 公共出口。

``retrieve`` 与 ``pdfx`` **急切**导入；``sandbox`` / ``latex`` / ``figures`` /
``metrics`` 走 :pep:`562` 的惰性 ``__getattr__``。

惰性不是为了"某些模块还没写完"——四个模块都已实现并有测试覆盖。理由是导入开销：
``import autoresearch.tools`` 不该顺带把 matplotlib 拉进来，而 CLI 的 ``stages`` /
``doctor`` 这类命令只需要知道工具存在。惰性模块出问题时抛 ``AttributeError``
而不是 ``ImportError``，否则 ``hasattr(tools, "Sandbox")`` 会抛异常，
破坏一切探测式代码。
"""

from __future__ import annotations

import importlib
from typing import Any

from .retrieve import (
    CACHE_TTL,
    DEFAULT_SOURCES,
    LiteratureSearch,
    Paper,
    RetrievalError,
    relevance_score,
)
from .pdfx import (
    download_pdf,
    extract_metadata,
    extract_references,
    extract_sections,
    extract_text,
    pdf_to_markdown,
)

__all__ = [
    # retrieve
    "Paper",
    "RetrievalError",
    "LiteratureSearch",
    "relevance_score",
    "CACHE_TTL",
    "DEFAULT_SOURCES",
    # pdfx
    "extract_text",
    "extract_sections",
    "extract_references",
    "extract_metadata",
    "download_pdf",
    "pdf_to_markdown",
    # 以下来自兄弟模块，惰性可用（见 _LAZY_EXPORTS）
    "ExecResult",
    "Sandbox",
    "SandboxError",
    "SecurityViolation",
    "make_sandbox",
    "scan_code",
    "CompileResult",
    "LatexCompiler",
    "LatexError",
    "setup_style",
    "make_all_figures",
    "parse_metrics",
    "summarize",
    "compare_runs",
    "to_latex_table",
]

# 惰性导出：公共名 -> 所属兄弟模块
_LAZY_EXPORTS: dict[str, str] = {
    "ExecResult": "sandbox",
    "Sandbox": "sandbox",
    "SandboxError": "sandbox",
    "SecurityViolation": "sandbox",
    "make_sandbox": "sandbox",
    "scan_code": "sandbox",
    "CompileResult": "latex",
    "LatexCompiler": "latex",
    "LatexError": "latex",
    "setup_style": "figures",
    "make_all_figures": "figures",
    "plot_learning_curves": "figures",
    "plot_bar_comparison": "figures",
    "plot_ablation": "figures",
    "plot_boxplot": "figures",
    "plot_metric_grid": "figures",
    "parse_metrics": "metrics",
    "summarize": "metrics",
    "compare_runs": "metrics",
    "to_latex_table": "metrics",
}


def __getattr__(name: str) -> Any:
    """惰性加载兄弟模块的公共名；不可用时抛 AttributeError（不抛 ImportError）。"""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except Exception as exc:  # ImportError 或兄弟模块自身错误
        raise AttributeError(
            f"{name!r} unavailable: {__name__}.{module_name} is not importable ({exc})"
        ) from None
    try:
        value = getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"{name!r} unavailable: {__name__}.{module_name} does not define it"
        ) from None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
