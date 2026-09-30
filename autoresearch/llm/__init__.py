"""LLM 层：客户端 + 异常族 + 离线 Mock 后端。

对外只暴露契约要求的名字，调用方写
``from autoresearch.llm import LLMClient, LLMConfig`` 即可。
"""

from __future__ import annotations

from ..config import LLMConfig
from .client import (
    BackendError,
    BudgetExceeded,
    LLMBackend,
    LLMClient,
    LLMError,
    LLMResponse,
    OpenAICompatBackend,
    ParseError,
    Usage,
    extract_json_text,
    parse_json_loose,
)
from .mock import MockBackend

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "Usage",
    "LLMError",
    "BackendError",
    "ParseError",
    "BudgetExceeded",
    "LLMBackend",
    "OpenAICompatBackend",
    "MockBackend",
    "build_backend",
    "extract_json_text",
    "parse_json_loose",
]


def build_backend(cfg: LLMConfig, **kw):
    """按 provider 直接构造一个后端实例（不走 LLMClient 缓存）。

    便于 CLI `doctor` 自检与单测：``mock`` -> MockBackend，
    其余 OpenAI 兼容 provider -> OpenAICompatBackend。
    """
    provider = str(getattr(cfg, "provider", "openai") or "openai").strip().lower()
    if provider == "mock":
        return MockBackend(temperature=getattr(cfg, "temperature", 0.3), **kw)
    if provider in ("openai", "deepseek", "ollama", "openai-compatible", "compatible", "custom"):
        return OpenAICompatBackend(
            model=getattr(cfg, "model", ""),
            base_url=getattr(cfg, "base_url", None),
            api_key=getattr(cfg, "api_key", None),
            timeout=float(getattr(cfg, "timeout", 120.0)),
            **kw,
        )
    raise LLMError(
        f"未知的 LLM provider: {provider!r}；可选值：openai, deepseek, ollama, mock"
    )
