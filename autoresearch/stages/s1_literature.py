"""阶段①：选题构思与文献综述（检索侧）。

职责边界刻意划清：**本阶段只做检索与结构化，不做价值判断**。
「这些文献说明了什么研究缺口」是 ② 的活儿——因为缺口判断需要看到全部检索
结果后再做一次语义扩散（语义树扩散假设），塞进同一阶段会让提示词过长、
也让「检索失败」污染「构思失败」的归因。
"""

from __future__ import annotations

import re
from typing import Any

from ..graph.state import Artifact
from .base import Stage, StageResult, clamp, clean_text, coerce_list

#: 极端情况下关键词抠不出来时用的兜底检索式模板。
_FALLBACK_TEMPLATES = (
    "{direction}",
    "{direction} state of the art",
    "{direction} benchmark",
    "{direction} survey",
    "{direction} limitations",
    "{direction} improvement",
)

#: 抽关键词时要滤掉的停用词/虚词。
_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "using",
    "based", "via", "toward", "towards", "from", "by", "is", "are", "be", "as",
    "研究", "方法", "基于", "面向", "一种", "的", "与", "和", "及", "在", "对",
}


class LiteratureStage(Stage):
    name = "s1_literature"
    title = "选题构思与文献综述"
    requires = ("direction",)
    produces = ("queries", "papers", "lit_review", "themes", "gaps", "bib_entries")
    max_attempts = 2

    # ------------------------------------------------------------------ #
    def run(self, state: dict[str, Any]) -> StageResult:
        direction = str(state.get("direction") or "").strip()
        if not direction:
            return StageResult.failure("no research direction provided", fatal=True)

        artifacts: list[Artifact] = []
        warnings: list[str] = []

        # --- 1. 生成检索式 -------------------------------------------- #
        queries = self._generate_queries(direction)
        if not queries:
            queries = [t.format(direction=direction) for t in _FALLBACK_TEMPLATES]
            warnings.append("query generation fell back to templates")
        queries = _dedupe_preserve(queries)[:8]

        # --- 2. 检索 --------------------------------------------------- #
        per_query = int(getattr(self.ctx.cfg.retrieve, "max_results_per_query", 8) or 8)
        try:
            papers = self.ctx.search.multi_search(queries, per_query=per_query)
        except Exception as exc:  # 检索层已保证不抛，这里是最后一道防线
            self._warn(f"s1: multi_search raised: {exc}")
            self.ctx.log_event("retrieve_fail", stage=self.name, error=str(exc))
            papers = []

        if not papers:
            warnings.append("no papers retrieved (offline or all sources failed)")
            self.ctx.log_event("retrieve_empty", stage=self.name, queries=len(queries))

        # 只把最相关的一批喂给 LLM，控制提示词长度
        top = papers[:20]

        # --- 3. 综述与缺口 --------------------------------------------- #
        papers_block = _format_papers_block(top, max_chars=16000)
        survey = self.llm_json(
            "s1_survey",
            default=None,
            schema_hint=_SURVEY_SCHEMA,
            direction=direction,
            papers_block=papers_block or "（无检索结果——请仅基于领域常识给出保守的主题聚类，并明确标注证据不足）",
            n_gaps=4,
        )

        themes: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        llm_summary = ""
        method_landscape: list[dict[str, Any]] = []
        if isinstance(survey, dict):
            themes = coerce_list(survey.get("themes"))
            gaps = coerce_list(survey.get("gaps"))
            llm_summary = clean_text(str(survey.get("summary") or ""))
            method_landscape = coerce_list(survey.get("method_landscape"))
        else:
            warnings.append("LLM survey synthesis unavailable; emitting retrieval-only review")

        # 缺口统一成字符串列表（③ 的输入），同时保留结构化版本供审计
        gap_strings: list[str] = []
        for g in gaps:
            if isinstance(g, dict):
                text = str(g.get("gap") or g.get("title") or "").strip()
                why = str(g.get("why_unsolved") or "").strip()
                if text:
                    gap_strings.append(f"{text}（未解决原因：{why}）" if why else text)
            elif isinstance(g, str):
                gap_strings.append(g.strip())

        # --- 4. 组装综述正文 ------------------------------------------- #
        try:
            deterministic_review = self.ctx.search.format_review(top, max_papers=20)
        except Exception as exc:
            self._warn(f"s1: format_review failed: {exc}")
            deterministic_review = ""

        lit_review = _compose_review(
            direction=direction,
            queries=queries,
            llm_summary=llm_summary,
            themes=themes,
            gap_strings=gap_strings,
            method_landscape=method_landscape,
            deterministic_review=deterministic_review,
            paper_count=len(papers),
            warnings=warnings,
        )

        # --- 5. 落盘 --------------------------------------------------- #
        artifacts.append(
            self.ctx.save_json(
                "literature/papers.json",
                {"queries": queries, "count": len(papers), "papers": [p.to_dict() for p in papers]},
                stage=self.name,
                kind="json",
            )
        )
        artifacts.append(
            self.ctx.save_text("literature/review.md", lit_review, stage=self.name, kind="md")
        )
        if themes or gaps:
            artifacts.append(
                self.ctx.save_json(
                    "literature/themes_gaps.json",
                    {"themes": themes, "gaps": gaps, "method_landscape": method_landscape},
                    stage=self.name,
                )
            )

        # --- 6. 预生成参考文献 ----------------------------------------- #
        bib_entries: list[dict[str, Any]] = []
        try:
            bibtex = self.ctx.search.to_bibtex(top)
        except Exception as exc:
            self._warn(f"s1: to_bibtex failed: {exc}")
            bibtex = ""
        if bibtex.strip():
            artifacts.append(
                self.ctx.save_text("paper/references.bib", bibtex, stage=self.name, kind="bib")
            )
        for p in top:
            entry = {
                "id": p.id,
                "bibkey": _bibkey(p),
                "title": p.title,
                "authors": list(p.authors[:6]),
                "year": p.year,
                "venue": p.venue,
                "url": p.url or p.pdf_url,
                "citation_count": p.citation_count,
                "source": p.source,
            }
            if entry["title"]:
                bib_entries.append(entry)
        artifacts.append(
            self.ctx.save_json("literature/bib_entries.json", bib_entries, stage=self.name)
        )

        detail = (
            f"{len(queries)} queries → {len(papers)} papers, "
            f"{len(themes)} themes, {len(gap_strings)} gaps"
        )
        if warnings:
            detail += f" ({len(warnings)} warnings)"

        return StageResult.success(
            detail=detail,
            updates={
                "queries": queries,
                "papers": [p.to_dict() for p in papers],
                "lit_review": lit_review,
                "themes": themes,
                "gaps": gap_strings,
                "bib_entries": bib_entries,
                "warnings": list(state.get("warnings") or []) + warnings,
            },
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------ #
    def _generate_queries(self, direction: str) -> list[str]:
        n = 6
        try:
            result = self.llm_json(
                "s1_queries",
                default=None,
                schema_hint=_QUERIES_SCHEMA,
                direction=direction,
                n_queries=n,
            )
        except Exception as exc:
            self._warn(f"s1: query generation failed: {exc}")
            return []
        raw: list[Any] = []
        if isinstance(result, dict):
            raw = coerce_list(result.get("queries"))
        elif isinstance(result, list):
            raw = result
        cleaned = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("query") or item.get("text") or ""
            text = clean_text(str(item))
            text = re.sub(r"^[\d\.\-\)\s]+", "", text).strip().strip('"').strip("'")
            if 3 <= len(text) <= 200:
                cleaned.append(text)
        return cleaned


# --------------------------------------------------------------------------- #
# 模块级纯函数
# --------------------------------------------------------------------------- #

_QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["queries"],
}

_SURVEY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "paper_ids": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gap": {"type": "string"},
                    "why_unsolved": {"type": "string"},
                    "opportunity": {"type": "string"},
                    "supporting_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["gap"],
            },
        },
        "method_landscape": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["summary"],
}


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _format_papers_block(papers: list[Any], max_chars: int = 16000) -> str:
    """把 Paper 列表压成 LLM 友好的紧凑块（标题 + 年 + 引用 + 摘要片段）。"""
    lines: list[str] = []
    used = 0
    for i, p in enumerate(papers, 1):
        authors = ", ".join((p.authors or [])[:3]) or "unknown"
        if len(p.authors or []) > 3:
            authors += " et al."
        abstract = clean_text(p.abstract or p.tldr or "")
        snippet = abstract[:600] + ("…" if len(abstract) > 600 else "")
        block = (
            f"[{i}] id={p.id}\n"
            f"    title: {clean_text(p.title)}\n"
            f"    authors: {authors} | year: {p.year or 'n/a'} | venue: {p.venue or 'n/a'} "
            f"| citations: {p.citation_count} | source: {p.source}\n"
            f"    abstract: {snippet or '(none)'}\n"
        )
        if used + len(block) > max_chars:
            lines.append(f"... 其余 {len(papers) - i + 1} 篇因长度限制省略")
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)


def _compose_review(
    direction: str,
    queries: list[str],
    llm_summary: str,
    themes: list[Any],
    gap_strings: list[str],
    method_landscape: list[Any],
    deterministic_review: str,
    paper_count: int,
    warnings: list[str],
) -> str:
    """拼装最终综述文档：LLM 负责「意义」，代码负责「事实与结构」。"""
    parts: list[str] = [
        f"# 文献综述：{direction}",
        "",
        f"> 由自动科研管线生成（阶段 s1）。检索式 {len(queries)} 条，"
        f"命中文献 {paper_count} 篇，去重后进入综述的为 "
        f"{sum(1 for _ in themes) if themes else 0} 个主题。",
        "",
        "## 1. 检索策略",
        "",
    ]
    parts += [f"- `{q}`" for q in queries] or ["- （无检索式）"]

    parts += ["", "## 2. 领域脉络综述", ""]
    if llm_summary:
        parts.append(llm_summary)
    else:
        parts.append(
            "_LLM 综述合成不可用（离线模式或调用失败）。以下仅呈现检索侧的结构化结果，"
            "不代表完整综述。_"
        )

    parts += ["", "## 3. 主题聚类", ""]
    if themes:
        for i, theme in enumerate(themes, 1):
            if isinstance(theme, dict):
                name = theme.get("name") or f"主题 {i}"
                summary = clean_text(str(theme.get("summary") or ""))
                ids = theme.get("paper_ids") or []
                parts.append(f"### 3.{i} {name}")
                parts.append("")
                if summary:
                    parts.append(summary)
                    parts.append("")
                if ids:
                    parts.append("代表工作：" + ", ".join(f"`{x}`" for x in coerce_list(ids)[:12]))
                    parts.append("")
            else:
                parts.append(f"### 3.{i} {theme}")
                parts.append("")
    else:
        parts.append("_未能形成主题聚类。_")

    if method_landscape:
        parts += ["", "## 4. 方法版图", ""]
        parts.append("| 方法族 | 代表工作 | 局限 |")
        parts.append("|---|---|---|")
        for row in method_landscape:
            if isinstance(row, dict):
                approach = clean_text(str(row.get("approach") or ""))
                reps = ", ".join(str(x) for x in coerce_list(row.get("representative_ids"))[:6])
                limit = clean_text(str(row.get("limitation") or ""))
                parts.append(f"| {approach} | {reps} | {limit} |")
        parts.append("")

    parts += ["", "## 5. 研究缺口", ""]
    if gap_strings:
        for i, gap in enumerate(gap_strings, 1):
            parts.append(f"{i}. {gap}")
    else:
        parts.append("_未能识别出结构化缺口，后续构思阶段需做更宽的检索。_")
    parts.append("")

    if deterministic_review:
        parts += ["", "## 6. 检索结果清单", "", deterministic_review, ""]

    if warnings:
        parts += ["", "## 附录：本阶段告警", ""]
        parts += [f"- {w}" for w in warnings]
        parts.append("")

    return "\n".join(parts)


def _bibkey(paper: Any) -> str:
    """与 ``LiteratureSearch.to_bibtex`` 的 key 规则保持一致（首作者姓+年+首词）。"""
    authors = paper.authors or []
    first = authors[0] if authors else "anon"
    if "," in first:
        surname = first.split(",")[0]
    else:
        surname = first.split()[-1] if first.split() else "anon"
    surname = re.sub(r"[^a-z]", "", surname.lower()) or "anon"
    year = paper.year or "n"
    words = [w for w in re.findall(r"[A-Za-z]+", paper.title or "") if w.lower() not in _STOPWORDS]
    word = (words[0].lower() if words else "paper")[:12]
    return f"{surname}{year}{word}"


__all__ = ["LiteratureStage"]
