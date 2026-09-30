"""LLM 客户端层（CONTRACTS §3，已冻结）.

零强制第三方依赖：``openai`` 只在 ``OpenAICompatBackend.__init__`` 内部延迟
导入，因此没有装 openai 的环境也能正常 ``import autoresearch.llm.client``。

职责：
* ``Usage`` / ``LLMResponse`` 数据形状
* ``LLMError`` 异常族（BackendError 可重试 / ParseError / BudgetExceeded）
* ``LLMBackend`` Protocol + ``OpenAICompatBackend``
* ``LLMClient``：重试 + 退避 + 磁盘缓存 + JSON 自修复 + 事件登记
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

__all__ = [
    "Usage",
    "LLMResponse",
    "LLMError",
    "BackendError",
    "ParseError",
    "BudgetExceeded",
    "LLMBackend",
    "OpenAICompatBackend",
    "LLMClient",
    "extract_json_text",
]

_log = logging.getLogger("autoresearch.llm")

#: 网络/限流类错误关键词：命中即可重试
_RETRYABLE_HINTS = (
    "timeout",
    "timed out",
    "connection",
    "connect",
    "network",
    "temporarily",
    "overloaded",
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "server error",
    "internal error",
    "unavailable",
    "reset by peer",
    "read timed out",
)

#: 认证/参数类错误关键词：绝不重试
_FATAL_HINTS = (
    "api key",
    "api_key",
    "unauthorized",
    "authentication",
    "invalid_api_key",
    "permission",
    "forbidden",
    "401",
    "403",
    "400",
    "404",
    "422",
    "invalid request",
    "model_not_found",
    "does not exist",
    "bad request",
)

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_OPEN_TO_CLOSE = {"{": "}", "[": "]"}


# --------------------------------------------------------------------------- #
# 数据形状
# --------------------------------------------------------------------------- #


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        if other is None or (isinstance(other, int) and other == 0):
            # 支持 sum([...]) 的 0 起始值
            return self
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other: Any) -> "Usage":
        if other is None or (isinstance(other, int) and other == 0):
            return self
        if isinstance(other, Usage):
            return other + self
        return NotImplemented

    def __iadd__(self, other: "Usage") -> "Usage":
        return self + other

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@dataclass
class LLMResponse:
    text: str
    raw: dict | None = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    cached: bool = False
    #: 服务端给出的结束原因（``stop`` / ``length`` / ...）。
    #: ``"length"`` 表示输出被 max_tokens 截断——JSON 一定不完整，必须区别对待。
    finish_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "cached": self.cached,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if isinstance(self.usage, Usage) else {},
        }


class LLMError(RuntimeError):
    """LLM 层所有错误的基类。"""


class BackendError(LLMError):
    """网络/服务端错误，可重试。"""


class ParseError(LLMError):
    """JSON 解析失败。"""


class BudgetExceeded(LLMError):
    """mock/预算保护触发。"""


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class LLMBackend(Protocol):
    """后端统一接口（CONTRACTS §3）。"""

    name: str

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tag: str | None = None,
        **kw: Any,
    ) -> LLMResponse:  # pragma: no cover - 协议声明
        ...


# --------------------------------------------------------------------------- #
# OpenAI 兼容后端
# --------------------------------------------------------------------------- #


class OpenAICompatBackend:
    """OpenAI / DeepSeek / Ollama 等 OpenAI 兼容 HTTP 后端。

    ``openai`` 包在此类的 ``__init__`` 中延迟导入：没装也不影响包导入。
    """

    name = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 8192,
        event_logger: Any = None,
        **_: Any,
    ) -> None:
        try:
            from openai import OpenAI  # 延迟导入（零强制依赖）
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise LLMError(
                "provider 需要 `openai` 包，但导入失败："
                f"{exc}。请 `pip install openai`，或改用 --llm-provider mock。"
            ) from exc

        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.default_max_tokens = int(max_tokens or 8192)
        self.events = event_logger
        # OpenAI SDK 不接受结尾斜杠以外的奇怪形态；顺手规范化
        kwargs: dict[str, Any] = {"api_key": api_key or "EMPTY", "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        try:
            self.client = OpenAI(**kwargs)
        except Exception as exc:
            raise LLMError(f"初始化 OpenAI 客户端失败 (base_url={base_url!r}): {exc}") from exc

    # -- 内部 ------------------------------------------------------------ #

    def _log_event(self, event: str, **fields: Any) -> None:
        fn = getattr(self.events, "log", None)
        if callable(fn):
            try:
                fn(event, **fields)
            except Exception:  # pragma: no cover - 日志失败绝不影响调用
                pass

    @staticmethod
    def _build_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call(self, messages: list[dict[str, str]], temperature: float, max_tokens: int,
              json_mode: bool) -> Any:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**params)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tag: str | None = None,
        **kw: Any,
    ) -> LLMResponse:
        temp = 0.3 if temperature is None else float(temperature)
        max_tok = int(getattr(self, "default_max_tokens", 8192)) if max_tokens is None else int(max_tokens)
        messages = self._build_messages(prompt, system)

        try:
            resp = self._call(messages, temp, max_tok, json_mode)
        except Exception as exc:
            if json_mode and _mentions_response_format(exc):
                _log.warning(
                    "服务端不支持 response_format(json_object)，去掉该参数重试一次: %s", exc
                )
                try:
                    resp = self._call(messages, temp, max_tok, False)
                except Exception as exc2:
                    raise _classify(exc2) from exc2
            else:
                raise _classify(exc) from exc

        text, model, raw = _read_response(resp, self.model)
        finish = _read_finish_reason(resp)
        # 被 token 上限截断的输出永远无法解析成完整 JSON。以前这种情况会白白走完
        # 「解析失败 → 自修复 → 再失败」的整条路径（实测两次调用都精确停在
        # completion_tokens=4096，浪费约 9k tokens）。现在检测到截断就
        # **立刻用更大的预算重试一次**：这是唯一真正能解决问题的动作。
        if finish == "length":
            bumped = min(max(max_tok * 2, max_tok + 2048), 32768)
            _log.warning(
                "输出被 max_tokens=%d 截断(completion_tokens=%s)，用 max_tokens=%d 重试一次",
                max_tok,
                getattr(_read_usage(resp, prompt, text), "completion_tokens", "?"),
                bumped,
            )
            self._log_event(
                "llm_truncated",
                tag=tag,
                attempted_max_tokens=max_tok,
                retry_max_tokens=bumped,
                completion_chars=len(text),
            )
            try:
                resp2 = self._call(messages, temp, bumped, json_mode)
                text2, model2, raw2 = _read_response(resp2, self.model)
                if text2.strip():
                    usage2 = _read_usage(resp2, prompt, text2)
                    self._log_event(
                        "llm_call",
                        model=model2,
                        prompt_chars=len(prompt),
                        completion_chars=len(text2),
                        prompt_tokens=usage2.prompt_tokens,
                        completion_tokens=usage2.completion_tokens,
                        latency=0.0,
                        cached=False,
                        ok=True,
                        tag=tag,
                        attempt="retry_after_truncation",
                    )
                    return LLMResponse(
                        text=text2, raw=raw2, usage=usage2, model=model2, cached=False
                    )
            except Exception as exc:  # 重试失败就沿用被截断的结果，交给上层处理
                _log.warning("截断重试失败，沿用截断输出: %s", exc)

        usage = _read_usage(resp, prompt, text)
        return LLMResponse(
            text=text, raw=raw, usage=usage, model=model, cached=False,
            finish_reason=finish,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }


def _mentions_response_format(exc: BaseException) -> bool:
    return "response_format" in str(exc).lower()


def _read_response(resp: Any, default_model: str) -> tuple[str, str, dict | None]:
    """从 OpenAI 风格响应里稳妥取出 text / model / raw。"""
    text = ""
    model = default_model
    raw: dict | None = None
    try:
        model = getattr(resp, "model", None) or default_model
    except Exception:
        pass
    try:
        choices = getattr(resp, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                text = content or ""
            if not text:
                text = getattr(choices[0], "text", "") or ""
        if not text:
            text = getattr(resp, "output_text", "") or ""
    except Exception as exc:  # pragma: no cover
        _log.warning("解析 LLM 响应正文失败: %s", exc)
    try:
        if hasattr(resp, "model_dump"):
            dumped = resp.model_dump()
            if isinstance(dumped, dict):
                raw = dumped
    except Exception:
        raw = None
    return text, model, raw


def _read_finish_reason(resp: Any) -> str:
    """读取 ``choices[0].finish_reason``；拿不到时返回空串。

    ``"length"`` 是唯一需要特殊处理的取值：它意味着输出被 token 上限砍断，
    任何结构化解析都必然失败。
    """
    try:
        choices = getattr(resp, "choices", None)
        if choices is None and isinstance(resp, dict):
            choices = resp.get("choices")
        if not choices:
            return ""
        first = choices[0]
        reason = _get_field(first, "finish_reason", None)
        return str(reason or "").strip().lower()
    except Exception:
        return ""


def _read_usage(resp: Any, prompt: str, text: str) -> Usage:
    """防御式读取 token 用量；缺失时用字符数粗估。"""
    pt = ct = tt = 0
    try:
        usage = getattr(resp, "usage", None)
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")
        if usage is not None:
            pt = int(_get_field(usage, "prompt_tokens", 0) or 0)
            ct = int(_get_field(usage, "completion_tokens", 0) or 0)
            tt = int(_get_field(usage, "total_tokens", 0) or 0)
    except Exception:
        pt = ct = tt = 0
    if pt == 0 and ct == 0 and tt == 0:
        pt = max(1, len(prompt) // 4)
        ct = max(1, len(text) // 4)
        tt = pt + ct
    elif tt == 0:
        tt = pt + ct
    return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, calls=1)


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
    return None


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试。4xx（除 408/429）一律不重试。"""
    if isinstance(exc, BackendError):
        return True
    if isinstance(exc, (ParseError, BudgetExceeded)):
        return False
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        return False

    status = _status_of(exc)
    if status is not None:
        if status in (408, 409, 425, 429) or status >= 500:
            return True
        if 400 <= status < 500:
            return False

    msg = str(exc).lower()
    cls = type(exc).__name__.lower()
    if "authenticationerror" in cls or "permissiondenied" in cls or "badrequest" in cls:
        return False
    if "ratelimiterror" in cls or "apiconnectionerror" in cls or "apitimeouterror" in cls:
        return True
    if any(hint in msg for hint in _FATAL_HINTS):
        return False
    if any(hint in msg for hint in _RETRYABLE_HINTS):
        return True
    # 未知错误：保守起见按"服务端/网络"处理，由调用方 max_retries 限制次数
    return True


def _classify(exc: BaseException) -> LLMError:
    """把任意后端异常翻译成 LLM 异常族，并给出清晰提示。"""
    if isinstance(exc, LLMError):
        return exc
    status = _status_of(exc)
    msg = str(exc) or type(exc).__name__
    if status is not None and 400 <= status < 500 and status not in (408, 409, 425, 429):
        hint = ""
        if status in (401, 403):
            hint = "（认证/权限失败：请检查 API key 与 base_url）"
        elif status == 404:
            hint = "（模型或路径不存在：请检查 --model 与 base_url）"
        elif status == 400:
            hint = "（请求被拒：通常是参数不受支持，例如 response_format）"
        return LLMError(f"LLM 请求被拒 [HTTP {status}] {msg} {hint}".strip())
    if is_retryable(exc):
        return BackendError(f"LLM 后端错误（可重试）: {msg}")
    return LLMError(f"LLM 调用失败: {msg}")


# --------------------------------------------------------------------------- #
# JSON 提取
# --------------------------------------------------------------------------- #


def _balanced_scan(text: str, start: int) -> tuple[str, int] | None:
    """从 ``text[start]`` 的括号开始找第一个平衡块。返回 (块, 结束下标)。"""
    open_ch = text[start]
    close_ch = _OPEN_TO_CLOSE[open_ch]
    depth = 0
    in_str = False
    quote = ""
    escaped = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                quote = ch
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1], i + 1
        i += 1
    return None


def extract_json_text(raw: str) -> str:
    """从模型输出里抽取 JSON 文本。

    1. 去掉 ``` 围栏（含 ```json）；
    2. 找第一个平衡的 ``{...}`` / ``[...]`` 块；
    3. 容忍尾随解释文本。

    找不到平衡块时返回去围栏后的文本，交给 ``json.loads`` 报错。
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return text

    # 1) 去围栏：优先带 json 标记的，其次任意围栏
    fenced = None
    for m in _FENCE_RE.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        if lang in ("json", "json5", "jsonc", ""):
            fenced = m.group(2)
            break
    if fenced is not None:
        text = fenced.strip()

    # 2) 平衡块
    for i, ch in enumerate(text):
        if ch in _OPEN_TO_CLOSE:
            found = _balanced_scan(text, i)
            if found is not None:
                return found[0].strip()
            break
    return text.strip()


def _iter_json_candidates(raw: str) -> Iterator[str]:
    """依次给出候选 JSON 文本，供解析器逐个尝试。"""
    seen: set[str] = set()
    primary = extract_json_text(raw)
    if primary:
        seen.add(primary)
        yield primary
    stripped = str(raw or "").strip()
    if stripped and stripped not in seen:
        seen.add(stripped)
        yield stripped
    # 退一步：全文里扫描所有平衡块
    idx = 0
    text = stripped
    while idx < len(text):
        ch = text[idx]
        if ch in _OPEN_TO_CLOSE:
            found = _balanced_scan(text, idx)
            if found is not None:
                block = found[0].strip()
                if block and block not in seen:
                    seen.add(block)
                    yield block
                idx = found[1]
                continue
        idx += 1


def parse_json_loose(raw: str) -> Any:
    """解析 JSON；失败抛 ``ParseError``（带片段）。"""
    if raw is None or not str(raw).strip():
        raise ParseError("LLM 返回为空，无法解析 JSON")
    last_exc: Exception | None = None
    for candidate in _iter_json_candidates(str(raw)):
        try:
            return json.loads(candidate)
        except Exception as exc:
            last_exc = exc
            continue
    snippet = _snippet(str(raw))
    raise ParseError(f"JSON 解析失败: {last_exc}; 原文片段: {snippet}")


def _snippet(text: str, n: int = 200) -> str:
    text = (text or "").replace("\n", "\\n")
    return text[:n] + ("..." if len(text) > n else "")


# --------------------------------------------------------------------------- #
# LLMClient
# --------------------------------------------------------------------------- #


class LLMClient:
    """高层客户端：懒加载后端、重试退避、磁盘缓存、JSON 自修复、事件登记。"""

    def __init__(
        self,
        cfg: Any,
        cache_dir: Path | None = None,
        event_logger: Any = None,
        cache_enabled: bool = True,
    ) -> None:
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.event_logger = event_logger
        self.cache_enabled = bool(cache_enabled)
        self._backend: LLMBackend | None = None
        self._usage = Usage()
        self._hits = 0
        self._misses = 0
        self._cache_writes = 0

    # -- 后端 ------------------------------------------------------------ #

    def backend(self) -> LLMBackend:
        """懒加载并缓存后端实例。

        注意：这是**唯一**构造真实 HTTP 后端的路径，而 mock 走另一个分支。
        因此任何写在这里的拼写错误（属性名、参数名）在 mock 端到端测试里
        完全不可见——本项目真的踩过一次：``self.events`` 实际叫
        ``self.event_logger``，于是接上 DeepSeek 后**所有**阶段都在
        ``AttributeError`` 上失败，而 mock 全绿。改动此方法后必须跑
        ``tests/test_pipeline.py::test_real_backend_construction``。
        """
        if self._backend is not None:
            return self._backend

        provider = str(getattr(self.cfg, "provider", "openai") or "openai").strip().lower()
        if provider == "mock":
            from .mock import MockBackend  # 局部导入：避免 mock <-> client 循环

            self._backend = MockBackend(temperature=getattr(self.cfg, "temperature", 0.3))
        elif provider in ("openai", "deepseek", "ollama", "openai-compatible", "compatible", "custom"):
            self._backend = OpenAICompatBackend(
                model=getattr(self.cfg, "model", ""),
                base_url=getattr(self.cfg, "base_url", None),
                api_key=getattr(self.cfg, "api_key", None),
                timeout=float(getattr(self.cfg, "timeout", 120.0)),
                max_tokens=int(getattr(self.cfg, "max_tokens", 8192) or 8192),
                event_logger=self.event_logger,
            )
        else:
            valid = "openai, deepseek, ollama, mock"
            raise LLMError(f"未知的 LLM provider: {provider!r}；可选值：{valid}")
        return self._backend

    # -- 缓存 ------------------------------------------------------------ #

    def _cache_key(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        json_mode: bool,
        model: str,
    ) -> str:
        payload = f"{model}|{temperature}|{system}|{prompt}|{json_mode}"
        return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_enabled or self.cache_dir is None:
            return None
        return Path(self.cache_dir) / f"{key}.json"

    def _cache_read(self, path: Path | None) -> LLMResponse | None:
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log.debug("缓存读取失败 %s: %s", path, exc)
            return None
        if not isinstance(data, dict) or "text" not in data:
            return None
        return LLMResponse(
            text=str(data.get("text", "")),
            raw={"cache_key": path.stem},
            usage=Usage(),
            model=str(data.get("model", "")),
            cached=True,
        )

    def _cache_write(self, path: Path | None, resp: LLMResponse) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {"text": resp.text, "model": resp.model, "cached": False},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
            self._cache_writes += 1
        except Exception as exc:
            _log.debug("缓存写入失败 %s: %s", path, exc)

    def cache_stats(self) -> dict[str, int]:
        entries = 0
        if self.cache_enabled and self.cache_dir is not None and Path(self.cache_dir).is_dir():
            try:
                entries = sum(1 for p in Path(self.cache_dir).glob("*.json") if p.is_file())
            except Exception:
                entries = 0
        return {"hits": int(self._hits), "misses": int(self._misses), "entries": int(entries)}

    # -- 事件 / 用量 ------------------------------------------------------ #

    def _emit(self, event: str, **fields: Any) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger.log(event, **fields)
        except Exception:  # 事件失败不影响主流程
            pass

    def usage(self) -> Usage:
        return self._usage

    def reset_usage(self) -> None:
        self._usage = Usage()

    # -- 完成 ------------------------------------------------------------ #

    def complete(self, prompt: str, system: str | None = None, tag: str | None = None, **kw: Any) -> str:
        """完成一次调用，返回文本。带缓存 + 重试退避。"""
        json_mode = bool(kw.pop("json_mode", False))
        temperature = kw.pop("temperature", None)
        max_tokens = kw.pop("max_tokens", None)
        if kw:
            _log.debug("complete() 忽略未知参数: %s", sorted(kw))
        temp = float(getattr(self.cfg, "temperature", 0.3) if temperature is None else temperature)
        model = str(getattr(self.cfg, "model", "") or "")

        cache_path = self._cache_path(
            self._cache_key(prompt, system, temp, json_mode, model)
        )
        cached_resp = self._cache_read(cache_path)
        if cached_resp is not None:
            self._hits += 1
            self._emit(
                "llm_call",
                model=cached_resp.model or model,
                prompt_chars=len(prompt or ""),
                completion_chars=len(cached_resp.text or ""),
                prompt_tokens=0,
                completion_tokens=0,
                latency=0.0,
                cached=True,
                ok=True,
                tag=tag,
            )
            return cached_resp.text
        self._misses += 1

        backend = self.backend()
        attempts = max(1, int(getattr(self.cfg, "max_retries", 3) or 1))
        last_exc: BaseException | None = None

        for attempt in range(1, attempts + 1):
            started = time.time()
            try:
                resp = backend.complete(
                    prompt,
                    system=system,
                    temperature=temp,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    tag=tag,
                )
            except Exception as exc:
                last_exc = exc
                retryable = is_retryable(exc)
                self._emit(
                    "llm_call",
                    model=model,
                    prompt_chars=len(prompt or ""),
                    completion_chars=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency=round(time.time() - started, 4),
                    cached=False,
                    ok=False,
                    tag=tag,
                    attempt=attempt,
                    error=str(exc)[:400],
                    retryable=retryable,
                )
                if not retryable:
                    raise _classify(exc) if not isinstance(exc, LLMError) else exc
                if attempt >= attempts:
                    break
                delay = min(30.0, 0.5 * (2 ** (attempt - 1))) * (0.7 + 0.6 * random.random())
                _log.warning(
                    "LLM 调用失败（第 %d/%d 次，%.1fs 后重试）: %s",
                    attempt,
                    attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue

            text, resp_model, _raw = resp.text, resp.model, resp.raw
            usage = resp.usage if isinstance(resp.usage, Usage) else Usage()
            self._usage = self._usage + usage
            self._cache_write(cache_path, resp)
            self._emit(
                "llm_call",
                model=resp_model or model,
                prompt_chars=len(prompt or ""),
                completion_chars=len(text or ""),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency=round(time.time() - started, 4),
                cached=bool(getattr(resp, "cached", False)),
                ok=True,
                tag=tag,
                attempt=attempt,
            )
            return text

        raise _classify(last_exc) if not isinstance(last_exc, LLMError) else last_exc

    # -- JSON 完成 -------------------------------------------------------- #

    def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        schema_hint: dict | None = None,
        repair_attempts: int | None = None,
        default: Any = None,
        tag: str | None = None,
        **kw: Any,
    ) -> Any:
        """让模型产出 JSON 并解析；失败则自修复重试；最终回退 default。"""
        attempts = (
            int(getattr(self.cfg, "json_repair_attempts", 2))
            if repair_attempts is None
            else int(repair_attempts)
        )
        attempts = max(0, attempts)

        raw = self.complete(prompt, system=system, json_mode=True, tag=tag, **kw)
        try:
            return parse_json_loose(raw)
        except ParseError as exc:
            self._emit(
                "llm_parse_failed",
                tag=tag,
                attempt=0,
                error=str(exc)[:400],
                phase="initial",
                raw_chars=len(raw or ""),
            )

        last_error = "unknown"
        for i in range(1, attempts + 1):
            repair_prompt = _build_repair_prompt(prompt, raw, last_error, schema_hint)
            raw2 = self.complete(repair_prompt, system=system, json_mode=True, tag=tag, **kw)
            try:
                return parse_json_loose(raw2)
            except ParseError as exc:
                last_error = str(exc)
                self._emit(
                    "llm_parse_failed",
                    tag=tag,
                    attempt=i,
                    error=last_error[:400],
                    phase="repair",
                    raw_chars=len(raw2 or ""),
                )
                raw = raw2

        if default is not None:
            return default
        raise ParseError(
            f"JSON 解析在 {attempts} 次修复后仍失败: {last_error}"
        )


# --------------------------------------------------------------------------- #
# 修复提示
# --------------------------------------------------------------------------- #


def _build_repair_prompt(
    prompt: str,
    bad_output: str,
    parse_error: str,
    schema_hint: dict | None,
) -> str:
    hint_block = ""
    if schema_hint:
        try:
            hint_block = json.dumps(schema_hint, ensure_ascii=False, indent=2)
        except Exception:
            hint_block = str(schema_hint)
    return (
        "你上一条回复不是合法 JSON，解析器报错如下：\n"
        f"<parse_error>{parse_error}</parse_error>\n\n"
        "上一条回复的原文：\n"
        f"<bad_output>\n{_snippet(bad_output, 4000)}\n</bad_output>\n\n"
        "期望的 JSON 结构（仅作形状指引，字段值请用真实内容）：\n"
        f"<schema_hint>\n{hint_block or '(未提供)'}\n</schema_hint>\n\n"
        "原始任务上下文（供你保持内容一致）：\n"
        f"<original_task>\n{_snippet(prompt, 4000)}\n</original_task>\n\n"
        "请只返回修正后的合法 JSON 对象本身：不要 markdown 围栏、"
        "不要解释文字、不要尾随逗号、不要注释。"
    )
