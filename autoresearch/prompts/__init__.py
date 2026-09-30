"""Prompt library for the Auto-Research pipeline (CONTRACTS §10).

Design constraints (frozen):
  * Prompts are ``.md`` files under ``prompts_dir``.
  * ``render`` performs a *minimal*, self-contained template pass.  Only
    ``{{ ... }}`` and ``{% ... %}`` are syntax.  Every other brace -- and LaTeX
    prompt bodies are full of them (``\\begin{tabular}{ll}``, ``\\frac{a}{b}``,
    ``\\cite{k}``, JSON examples) -- is passed through **byte for byte**.
    Jinja2 is deliberately *not* used: its grammar would rewrite or reject
    constructs we must preserve.
  * Supported syntax::

        {{ var }}                 variable substitution (whitespace tolerant)
        {% if var %} ... {% endif %}
        {% if var %} ... {% else %} ... {% endif %}
        {% for x in xs %} ... {% endfor %}

    where ``xs`` may be a list of scalars or a list of mappings and ``x`` an
    optional dotted path (``{% for p in papers %}`` then ``{{ p.title }}``).
  * Unknown variables are replaced by the empty string and reported through
    :mod:`logging` (module logger ``autoresearch.prompts``).

Localization: ``language="zh"`` prefers ``<name>.zh.md`` over ``<name>.md``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = ["PromptLibrary", "TemplateError", "SUPPORTED_LANGUAGES"]

logger = logging.getLogger("autoresearch.prompts")

SUPPORTED_LANGUAGES: tuple[str, ...] = ("zh", "en")

#: Path separators that may prefix a prompt name (``sub/dir`` or ``sub\\dir``).
_SEP_RE = re.compile(r"[\\/]+")

#: Guard against a stray ``{%`` swallowing the rest of a template.
_MAX_PASSES = 50

_LANGUAGE_DIRECTIVES: dict[str, str] = {
    "zh": (
        "输出语言：简体中文（学术书面语）。JSON 的 key 必须保持英文原文，"
        "所有面向读者的字符串值一律使用中文；专业术语首次出现时用“中文（English）”格式。"
    ),
    "en": (
        "Output language: English (formal academic register). Keep every JSON key "
        "exactly as specified; use English for all human-readable string values."
    ),
}


class TemplateError(ValueError):
    """Raised for malformed template syntax inside a prompt file."""


def _language_directive(language: str) -> str:
    return _LANGUAGE_DIRECTIVES.get(language, _LANGUAGE_DIRECTIVES["en"])


def _as_text(value: Any) -> str:
    """Stringify a template value without surprising Python reprs."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_as_text(v) for v in value)
    return str(value)


def _iter_vars(value: Any) -> Iterator[Any]:
    """Iterate a loop target, tolerating ``None`` and non-sequences."""
    if value is None:
        return iter(())
    if isinstance(value, Mapping):
        return iter(value.items())
    if isinstance(value, str):
        return iter((value,))
    try:
        return iter(value)
    except TypeError:
        return iter((value,))


def _resolve(expr: str, scope: Mapping[str, Any]) -> tuple[Any, bool]:
    """Resolve ``a`` / ``a.b`` / ``a.0`` against *scope*.

    Returns ``(value, found)``; a missing root or attribute yields
    ``(None, False)`` so callers can warn instead of exploding.
    """
    expr = expr.strip()
    if not expr:
        return None, False
    parts = expr.split(".")
    root = parts[0]
    if root not in scope:
        return None, False
    current: Any = scope[root]
    for part in parts[1:]:
        if isinstance(current, Mapping):
            if part in current:
                current = current[part]
                continue
            return None, False
        if isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None, False
            continue
        current = getattr(current, part, None)
        if current is None:
            return None, False
    return current, True


def _is_truthy(value: Any, found: bool) -> bool:
    if not found:
        return False
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value != 0
    return True


class _Renderer:
    """Single-render state: variable scope, warnings and knobs."""

    def __init__(
        self,
        variables: Mapping[str, Any],
        *,
        warn_unknown: bool,
        empty_unknown: bool,
        label: str,
    ) -> None:
        self.vars: dict[str, Any] = dict(variables)
        self.warn_unknown = warn_unknown
        self.empty_unknown = empty_unknown
        self.label = label
        self.unknown: list[str] = []

    # -- logging helpers -------------------------------------------------
    def _warn(self, message: str, *args: Any) -> None:
        if self.warn_unknown:
            logger.warning("[%s] " + message, self.label, *args)

    # -- variables -------------------------------------------------------
    def sub_vars(self, body: str) -> str:
        def repl(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            value, found = _resolve(expr, self.vars)
            if found:
                return _as_text(value)
            if expr not in self.unknown:
                self.unknown.append(expr)
                self._warn("unknown variable {{ %s }} -> replaced with ''", expr)
            return "" if self.empty_unknown else match.group(0)

        return re.sub(r"\{\{(.*?)\}\}", repl, body, flags=re.DOTALL)

    # -- {% if %} --------------------------------------------------------
    def resolve_if(self, body: str) -> str:
        pattern = re.compile(
            r"\{%\s*if\s+([^{}%]+?)\s*%\}(.*?)\{%\s*endif\s*%\}",
            re.DOTALL,
        )
        while True:
            match = pattern.search(body)
            if match is None:
                return body
            expr = match.group(1).strip()
            block = match.group(2)
            cond_expr, else_expr = self._split_else(block)
            # NOTE: resolve ``expr`` (the condition), not ``cond_expr`` (which is
            # the already-split true-branch body).
            value, found = _resolve(expr, self.vars)
            chosen = cond_expr if _is_truthy(value, found) else else_expr
            body = body[: match.start()] + self.sub_vars(chosen) + body[match.end() :]

    def _split_else(self, block: str) -> tuple[str, str]:
        """Split on the ``{% else %}`` that belongs to *this* ``{% if %}``."""
        else_re = re.compile(r"\{%\s*else\s*%\}")
        depth = 0
        pos = 0
        while True:
            nxt_if = re.compile(r"\{%\s*if\s+").search(block, pos)
            nxt_else = else_re.search(block, pos)
            if nxt_else is None:
                return block, ""
            if nxt_if is not None and nxt_if.start() < nxt_else.start():
                depth += 1
                pos = nxt_if.end()
                continue
            if depth > 0:
                depth -= 1
                pos = nxt_else.end()
                continue
            return block[: nxt_else.start()], block[nxt_else.end() :]

    # -- {% for %} -------------------------------------------------------
    def resolve_for(self, body: str) -> str:
        pattern = re.compile(
            r"\{%\s*for\s+([A-Za-z_][\w.]*)\s+in\s+([^{}%]+?)\s*%\}"
            r"(.*?)\{%\s*endfor\s*%\}",
            re.DOTALL,
        )
        while True:
            match = pattern.search(body)
            if match is None:
                return body
            var_name = match.group(1).strip()
            seq_expr = match.group(2).strip()
            inner = match.group(3)
            seq, found = _resolve(seq_expr, self.vars)
            if not found:
                self._warn(
                    "unknown loop sequence {{ %s }} -> zero iterations", seq_expr
                )
            chunks: list[str] = []
            for index, item in enumerate(_iter_vars(seq)):
                child = _Renderer(
                    self.vars,
                    warn_unknown=self.warn_unknown,
                    empty_unknown=self.empty_unknown,
                    label=self.label,
                )
                child.vars.update(self._loop_scope(var_name, item))
                child.vars["loop"] = {
                    "index": index,
                    "index1": index + 1,
                    "first": index == 0,
                }
                chunks.append(child.render_fragment(inner))
                self.unknown.extend(u for u in child.unknown if u not in self.unknown)
            body = body[: match.start()] + "".join(chunks) + body[match.end() :]

    @staticmethod
    def _loop_scope(var_name: str, item: Any) -> dict[str, Any]:
        """Bind ``x`` (and, for mappings, ``x.key``) inside a loop body."""
        parts = var_name.split(".")
        scope: dict[str, Any] = {var_name: item}
        if len(parts) > 1:
            scope[parts[0]] = item
            if isinstance(item, Mapping):
                for key, value in item.items():
                    scope[f"{parts[0]}.{key}"] = value
        return scope

    # -- fragment driver -------------------------------------------------
    def render_fragment(self, body: str) -> str:
        """Resolve ``{% %}`` blocks, then substitute ``{{ }}``, until stable."""
        text = body
        for _ in range(_MAX_PASSES):
            new = self.resolve_for(text)
            new = self.resolve_if(new)
            new = self.sub_vars(new)
            if new == text:
                return new
            text = new
        raise TemplateError(
            f"template for '{self.label}' did not converge after {_MAX_PASSES} "
            "passes (unbalanced {% %} blocks?)"
        )


class PromptLibrary:
    """Loads and renders prompt templates from *prompts_dir*."""

    def __init__(self, prompts_dir: Path, language: str = "zh") -> None:
        self.prompts_dir = Path(prompts_dir).resolve()
        self.language = (language or "zh").strip().lower()
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"unsupported language {language!r}; expected one of "
                f"{SUPPORTED_LANGUAGES}"
            )

    # -- introspection ---------------------------------------------------
    def list_prompts(self) -> list[str]:
        """Sorted base stems, deduplicated across ``.zh`` / ``.en`` variants.

        ``README.md`` and ``_``-prefixed files are fixtures/documentation, not
        stage prompts, and are excluded.
        """
        if not self.prompts_dir.is_dir():
            return []
        stems: set[str] = set()
        for path in self.prompts_dir.glob("*.md"):
            if not path.is_file():
                continue
            if path.name.lower() == "readme.md" or path.name.startswith("_"):
                continue
            stems.add(path.stem.split(".")[0])
        return sorted(stems)

    def available_files(self) -> list[str]:
        if not self.prompts_dir.is_dir():
            return []
        return sorted(p.name for p in self.prompts_dir.glob("*.md") if p.is_file())

    # -- file resolution -------------------------------------------------
    def resolve(self, name: str) -> Path:
        """Resolve *name* to an existing prompt file, honouring the language.

        The resolved path is confined to ``prompts_dir``; a traversal-shaped
        name such as ``../secrets`` can never escape it.
        """
        stem = _SEP_RE.sub("/", str(name).strip())
        if stem.endswith(".md"):
            stem = stem[:-3]
        candidates: list[Path] = []
        if self.language:
            candidates.append(self.prompts_dir / f"{stem}.{self.language}.md")
        candidates.append(self.prompts_dir / f"{stem}.md")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved != self.prompts_dir and self.prompts_dir not in resolved.parents:
                logger.warning(
                    "[%s] prompt path escapes the prompts directory: %s",
                    name,
                    resolved,
                )
                continue
            return resolved
        attempted = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"prompt {name!r} not found. Resolved path(s): {attempted}. "
            f"Available prompts: {self.list_prompts()}"
        )

    def raw(self, name: str) -> str:
        """Return the prompt file text unchanged (localized variant preferred)."""
        return self.resolve(name).read_text(encoding="utf-8")

    # -- rendering -------------------------------------------------------
    def render(
        self,
        name: str,
        *,
        _strict: bool = False,
        _warn_unknown: bool = True,
        _empty_unknown: bool = True,
        **vars: Any,
    ) -> str:
        """Render prompt *name* with *vars*.

        Unknown ``{{ var }}`` placeholders become the empty string and emit a
        warning.  Literal braces outside ``{{ }}`` / ``{% %}`` are preserved.
        """
        text = self.raw(name)
        scope: dict[str, Any] = dict(vars)
        scope.setdefault("language", self.language)
        scope.setdefault("language_directive", _language_directive(self.language))
        renderer = _Renderer(
            scope,
            warn_unknown=_warn_unknown,
            empty_unknown=_empty_unknown,
            label=str(name),
        )
        try:
            rendered = renderer.render_fragment(text)
        except TemplateError:
            if _strict:
                raise
            logger.warning(
                "[%s] template error; falling back to raw text", name, exc_info=True
            )
            return text
        if _strict and renderer.unknown:
            raise TemplateError(
                f"prompt {name!r} referenced unknown variables: {renderer.unknown}"
            )
        leftovers = re.findall(r"\{\{[^{}]*\}\}", rendered)
        if leftovers:
            logger.warning("[%s] leftover placeholders: %s", name, leftovers[:8])
        return rendered

    # -- convenience -----------------------------------------------------
    def system_prompt(self, language: str | None = None) -> str:
        """A compact system message shared by every stage."""
        return (
            "You are the reasoning engine of an automated research pipeline. "
            "Follow the requested output contract exactly, never invent citations "
            "or numbers, and prefer falsifiable, concrete statements.\n"
            + _language_directive(language or self.language)
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PromptLibrary(prompts_dir={str(self.prompts_dir)!r}, "
            f"language={self.language!r}, prompts={self.list_prompts()})"
        )
