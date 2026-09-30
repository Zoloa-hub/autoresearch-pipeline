"""阶段基类与共用工具。

每个阶段是一个纯函数式的转换：``state -> StageResult``。阶段**不直接**读写
全局变量、不依赖执行顺序以外的隐式状态、不吞异常（异常由引擎统一处理并决定
重试/降级）。这样做的收益是：任何阶段都能单独跑测试、能单独重跑、能在
LangGraph 下原样复用。
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..graph.state import Artifact


@dataclass
class StageResult:
    """阶段返回值。

    Attributes:
        ok: 阶段是否成功。``False`` 会触发引擎的重试/降级逻辑。
        state_updates: 要合并回全局状态的键值（必须 JSON 可序列化）。
        artifacts: 本阶段落盘的产物登记。
        detail: 一句话摘要，写进 trace。
        fatal: 为 ``True`` 时引擎立即终止（用于「没有 LLM 就无法继续」这类情况）。
        retry: 显式请求重试（例如 LLM 返回了不可解析的结构，值得换一次采样）。
    """

    ok: bool
    state_updates: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    detail: str = ""
    fatal: bool = False
    retry: bool = False

    # -- 便捷构造 ------------------------------------------------------- #
    @classmethod
    def success(
        cls,
        detail: str = "",
        updates: dict[str, Any] | None = None,
        artifacts: list[Artifact] | None = None,
    ) -> StageResult:
        return cls(True, updates or {}, artifacts or [], detail)

    @classmethod
    def failure(
        cls,
        detail: str,
        updates: dict[str, Any] | None = None,
        artifacts: list[Artifact] | None = None,
        retry: bool = False,
        fatal: bool = False,
    ) -> StageResult:
        return cls(False, updates or {}, artifacts or [], detail, fatal, retry)

    @classmethod
    def skip(cls, detail: str) -> StageResult:
        """阶段无事可做但**不算失败**（如无数据可画图）。"""
        return cls(True, {}, [], f"skipped: {detail}")


class Stage(abc.ABC):
    """所有阶段的基类。"""

    name: str = ""
    title: str = ""
    #: 依赖的状态键；缺失时只记 warning，不阻塞（阶段自己决定降级策略）。
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    max_attempts: int = 2
    optional: bool = False
    #: ``state -> next_stage_name | None``；``None`` 表示按声明顺序继续。
    router: Any = None

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.log = getattr(ctx, "log", None)

    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def run(self, state: dict[str, Any]) -> StageResult:
        """执行本阶段。"""

    # ------------------------------------------------------------------ #
    # 共用工具
    # ------------------------------------------------------------------ #
    def check_requires(self, state: dict[str, Any]) -> list[str]:
        """返回缺失的依赖键名列表（同时记 warning）。"""
        missing = []
        for key in self.requires:
            value = state.get(key)
            if value in (None, "", [], {}):
                missing.append(key)
        if missing:
            self.ctx.log_event(
                "stage_missing_input", stage=self.name, missing=missing
            )
            self._warn(f"{self.name}: missing inputs {missing}; degrading")
        return missing

    def _warn(self, msg: str) -> None:
        if self.log is not None:
            try:
                self.log.warning(msg)
            except Exception:  # pragma: no cover
                pass

    def _info(self, msg: str) -> None:
        if self.log is not None:
            try:
                self.log.info(msg)
            except Exception:  # pragma: no cover
                pass

    # -- LLM 便捷封装 --------------------------------------------------- #
    #: 按提示词给出输出预算。**默认 8192 对好几个阶段是不够的**：
    #: 真实 DeepSeek 上 ``s1_survey``（要 600-900 词综述 + 主题 + 缺口 + 方法版图）
    #: 与 ``s5_analysis``（claim 分级 + 发现 + 局限 + 效度威胁）都会精确撞到上限，
    #: 输出被截断 → JSON 不完整 → 解析失败 → 整段工作白做。
    #: 这类失败在 mock 下完全不可见（mock 输出很短），只有接真实模型才暴露。
    OUTPUT_BUDGETS: dict[str, int] = {
        "s1_queries": 2048,
        "s1_survey": 16384,
        "s2_ideas": 12288,
        "s2_novelty": 3072,
        "s3_plan": 6144,
        "s4_codegen": 16384,
        "s4_debug": 12288,
        "s5_analysis": 16384,
        "s6_section": 8192,
        "s6_abstract": 3072,
        "s6_revision": 16384,
        "s7_compile_fix": 4096,
        "s8_review": 6144,
        "s9_report": 6144,
    }

    def _budget_for(self, prompt_name: str) -> int:
        default = int(getattr(getattr(self.ctx, "cfg", None), "llm", None)
                      and getattr(self.ctx.cfg.llm, "max_tokens", 8192) or 8192)
        return int(self.OUTPUT_BUDGETS.get(prompt_name, default) or default)

    def llm_json(
        self,
        prompt_name: str,
        default: Any = None,
        schema_hint: dict | None = None,
        max_tokens: int | None = None,
        **vars: Any,
    ) -> Any:
        """渲染提示词 → 调 LLM → 解析 JSON。失败返回 ``default``。

        这是阶段里唯一被允许的 LLM 入口，好处是：每次调用都自动带上
        ``tag=f"{stage}:{prompt_name}"``，事件日志里能直接归因到阶段；
        输出预算按提示词名自动选取（见 ``OUTPUT_BUDGETS``）。

        **语言指令走 system 而不是 user prompt**：如果塞进 user prompt，
        模型（尤其是弱模型）会把它当成正文内容回显，污染标题与摘要——
        这个坑在离线 mock 上会立刻暴露成「标题里出现指令原文」。
        """
        prompt = self.ctx.prompts.render(prompt_name, **vars)
        return self.ctx.llm.complete_json(
            prompt,
            system=self._system_prompt(),
            schema_hint=schema_hint,
            default=default,
            tag=f"{self.name}:{prompt_name}",
            max_tokens=max_tokens or self._budget_for(prompt_name),
        )

    def llm_text(
        self, prompt_name: str, fallback: str = "", max_tokens: int | None = None, **vars: Any
    ) -> str:
        prompt = self.ctx.prompts.render(prompt_name, **vars)
        try:
            return self.ctx.llm.complete(
                prompt,
                system=self._system_prompt(),
                tag=f"{self.name}:{prompt_name}",
                max_tokens=max_tokens or self._budget_for(prompt_name),
            )
        except Exception as exc:
            self._warn(f"{self.name}: llm_text({prompt_name}) failed: {exc}")
            return fallback

    def _lang_directive(self) -> str:
        lang = getattr(self.ctx.cfg, "language", "zh")
        if lang == "zh":
            return (
                "所有自然语言内容用中文撰写；学术术语、方法名、数据集名、指标名"
                "保留英文原词。JSON 的键名保持英文不动。"
            )
        return "Write all natural-language content in English. Keep JSON keys exactly as specified."

    def _system_prompt(self) -> str:
        return (
            "You are a rigorous, honest research assistant operating inside an "
            "automated research pipeline. Never fabricate numbers, citations, or "
            "experimental results: every quantitative claim must be traceable to "
            "evidence supplied in the prompt. When evidence is missing, say so "
            "explicitly rather than inventing it. Prefer precise, falsifiable "
            "statements over hedging. When asked for JSON, reply with a single "
            "valid JSON object and nothing else.\n\n"
            "OUTPUT LANGUAGE: " + self._lang_directive()
        )

    # -- 状态读取助手 --------------------------------------------------- #
    def state_list(self, state: dict[str, Any], key: str) -> list[Any]:
        value = state.get(key)
        return list(value) if isinstance(value, (list, tuple)) else []

    def state_dict(self, state: dict[str, Any], key: str) -> dict[str, Any]:
        value = state.get(key)
        return dict(value) if isinstance(value, dict) else {}

    def selected_idea(self, state: dict[str, Any]) -> dict[str, Any]:
        idea = state.get("selected_idea") or {}
        if not idea and state.get("ideas"):
            ideas = state["ideas"]
            idea = max(ideas, key=lambda d: float(d.get("rank") or 0.0))
        return dict(idea) if isinstance(idea, dict) else {}


# --------------------------------------------------------------------------- #
# 文本/结构化工具（阶段共用，纯函数，可单测）
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """压掉多余空白，保留段落结构。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


#: 提示词注入的「语言/格式指令」可能被弱模型当正文回显。它们**不是内容**，
#: 出现在标题或摘要里会直接毁掉一篇论文，所以在所有 LLM 文本出口处统一剥离。
_DIRECTIVE_ECHO_RE = re.compile(
    r"(?:"
    r"输出语言\s*[:：][^。\n；;]*[。；;]?"
    r"|写作语言\s*[:：][^。\n；;]*[。；;]?"
    r"|语言\s*[:：]\s*(?:简体中文|繁體中文|中文|英文|Chinese|English)[^。\n；;]*[。；;]?"
    r"|(?:所有)?自然语言内容用中文撰写[^。\n；;]*[。；;]?"
    r"|(?:请)?用(?:简体)?中文(?:撰写|写作|回答)[^。\n；;]*[。；;]?"
    r"|JSON\s*的\s*key\s*必须保持英文[^。\n；;]*[。；;]?"
    r"|Output language\s*[:：][^\n.]*\.?"
    r"|Write (?:all )?(?:natural[- ]language )?content in English[^\n.]*\.?"
    r"|Respond in English[^\n.]*\.?"
    r"|Keep JSON keys (?:exactly )?as specified[^\n.]*\.?"
    r")",
    re.IGNORECASE,
)


def strip_directive_echo(text: str) -> str:
    """剥掉被回显的语言/格式指令，并清理因此产生的悬挂标点。

    这是「输出卫生」的最后一道闸：即使提示词工程出问题，标题与正文也不会
    把操作指令写进论文。
    """
    if not text:
        return ""
    cleaned = _DIRECTIVE_ECHO_RE.sub("", text)
    # 清理残留：连续空白、行首悬挂标点、空行
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"^\s*[：:；;，,。]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def title_from_text(text: str, fallback: str = "Untitled", max_len: int = 120) -> str:
    """把一段可能带杂质的文本压成可用标题。

    句号只在「后面跟空格或结尾」时才当作边界，避免把 ``3.5``、``e.g.``、
    ``ResNet-50.`` 里的点误判成句子结束。
    """
    cleaned = strip_directive_echo(clean_text(text or ""))
    cleaned = re.split(r"[。\n]|(?<=[.!?])\s+", cleaned)[0]
    cleaned = cleaned.strip(" ：:，,；;.-–—*#`\"'")
    # 去掉 markdown 强调与 LaTeX 命令残渣
    cleaned = re.sub(r"\*{1,3}", "", cleaned)
    cleaned = re.sub(r"\\[a-zA-Z]+\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ：:，,；;.-–—")
    if len(cleaned) < 6:
        cleaned = strip_directive_echo(clean_text(fallback or "")) or "Untitled"
        cleaned = re.split(r"[。\n]|(?<=[.!?])\s+", cleaned)[0].strip(" ：:，,；;.-")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(" ：:，,；;.-")
    return cleaned or "Untitled"


def clamp(text: str, limit: int) -> str:
    """按字符截断并标注省略量——给 LLM 的输入必须可预期地有界。"""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def coerce_list(value: Any) -> list[Any]:
    """LLM 经常把列表写成字符串或 ``{"items": [...]}``，这里统一成 list。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        for key in ("items", "list", "values", "results", "data"):
            if isinstance(value.get(key), list):
                return list(value[key])
        return [value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return coerce_list(parsed)
        # 退化：按行/分号切分
        parts = [p.strip(" -•\t") for p in re.split(r"[\n;]+", stripped)]
        return [p for p in parts if p]
    return [value]


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return default
    return default


def truncate_words(text: str, limit: int) -> str:
    return clamp(text, limit)


__all__ = [
    "Stage",
    "StageResult",
    "as_float",
    "clamp",
    "clean_text",
    "coerce_list",
    "truncate_words",
]
