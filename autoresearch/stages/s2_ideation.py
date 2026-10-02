"""阶段②：假设生成与新颖性校验（语义树扩散 → 查新淘汰 → 排序）。

三步流水线，每一步都可独立失败而不拖垮整条：
1. **扩散**：以 ① 的缺口为种子，让 LLM 生成 ``max_ideas`` 个可证伪假设；
2. **查新**：对每个候选检索近期相似文献，让 LLM 对照判定
   ``novel / incremental / duplicate``，并与检索侧信号做交叉校验；
3. **排序**：新颖性、可行性、证据支撑加权，选出一个 ``selected_idea``。

查新一步有**确定性旁证**：候选idea的关键词与检索到的近期文献做词面重叠度计算。
LLM 说 novel 但重叠度极高时降级为 ``incremental`` 并记事件——这是防止
「LLM 自评新颖性」系统性乐观的廉价护栏。
"""

from __future__ import annotations

import re
from typing import Any

from ..graph.state import Artifact
from .base import Stage, StageResult, as_float, clean_text, coerce_list
from .s1_literature import _dedupe_preserve, _format_papers_block

#: 排序权重。新颖性优先，其次可行性——可行性不足的 idea 在 ④ 会崩。
_WEIGHTS = {"novelty": 0.45, "feasibility": 0.30, "evidence": 0.25}

_VERDICT_SCORE = {"novel": 1.0, "incremental": 0.55, "duplicate": 0.0, "unknown": 0.5}

_IDEAS_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "motivation": {"type": "string"},
                    "method_sketch": {"type": "string"},
                    "novelty_claim": {"type": "string"},
                    "feasibility": {"type": "string"},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "expected_metrics": {"type": "array", "items": {"type": "string"}},
                    "minimal_experiment": {"type": "string"},
                    "required_resources": {"type": "string"},
                },
                "required": ["title", "hypothesis"],
            },
        }
    },
    "required": ["ideas"],
}

_NOVELTY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["novel", "incremental", "duplicate", "unknown"]},
        "score": {"type": "number"},
        "rationale": {"type": "string"},
        "closest": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "id": {"type": "string"},
                    "year": {"type": "integer"},
                    "why": {"type": "string"},
                },
            },
        },
        "differentiators": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict"],
}


class IdeationStage(Stage):
    name = "s2_ideation"
    title = "假设生成与新颖性校验"
    requires = ("direction",)
    produces = ("ideas", "selected_idea", "novelty_summary")
    max_attempts = 2

    # ------------------------------------------------------------------ #
    def run(self, state: dict[str, Any]) -> StageResult:
        direction = str(state.get("direction") or "").strip()
        if not direction:
            return StageResult.failure("no research direction", fatal=True)

        artifacts: list[Artifact] = []
        warnings: list[str] = []

        gaps = [str(g) for g in (state.get("gaps") or [])]
        papers = list(state.get("papers") or [])

        # --- 1. 语义扩散：生成候选假设 -------------------------------- #
        max_ideas = int(getattr(self.ctx.cfg, "max_ideas", 6) or 6)
        ideas_raw = self._generate_ideas(direction, gaps, papers, max_ideas)

        if not ideas_raw:
            warnings.append("LLM returned no parseable ideas; using gap-derived placeholders")
            ideas_raw = _placeholder_ideas(direction, gaps, max_ideas)

        ideas: list[dict[str, Any]] = []
        for i, raw in enumerate(ideas_raw, 1):
            idea = _normalize_idea(raw, index=i)
            if idea:
                ideas.append(idea)
                self.ctx.log_event("idea_generated", stage=self.name, idea_id=idea["id"],
                                   title=idea["title"][:120])
        if not ideas:
            return StageResult.failure("no usable ideas could be constructed", retry=True)

        # --- 2. 新颖性校验 -------------------------------------------- #
        recent = _recent_papers(papers, years=3)
        for idea in ideas:
            idea["novelty"] = self._check_novelty(idea, recent, direction, warnings)
            self.ctx.log_event(
                "novelty_check",
                stage=self.name,
                idea_id=idea["id"],
                verdict=idea["novelty"].get("verdict"),
                score=idea["novelty"].get("score"),
                closest=len(idea["novelty"].get("closest") or []),
            )

        # --- 3. 淘汰 + 排序 ------------------------------------------- #
        survivors = [i for i in ideas if i["novelty"].get("verdict") != "duplicate"]
        eliminated = [i for i in ideas if i["novelty"].get("verdict") == "duplicate"]
        if not survivors:
            warnings.append("all candidates judged duplicate; keeping the best-scoring one anyway")
            survivors = sorted(ideas, key=lambda d: -as_float(d["novelty"].get("score")))[:1]
            eliminated = [i for i in ideas if i not in survivors]

        for idea in survivors:
            idea["rank"] = round(_rank(idea), 4)
        survivors.sort(key=lambda d: -as_float(d.get("rank")))
        for idea in eliminated:
            idea["rank"] = round(_rank(idea), 4)
            idea["eliminated"] = True

        selected = survivors[0]
        selected["selected"] = True
        keep = int(getattr(self.ctx.cfg, "keep_top_ideas", 3) or 3)
        for idea in survivors[keep:]:
            idea["selected"] = False

        novelty_summary = {
            "total": len(ideas),
            "eliminated": len(eliminated),
            "verdicts": _count_verdicts(ideas),
            "recent_pool": len(recent),
            "selected_id": selected["id"],
            "selected_rank": selected["rank"],
        }

        # --- 4. 落盘 --------------------------------------------------- #
        artifacts.append(
            self.ctx.save_json(
                "ideas/ideas.json",
                {"direction": direction, "count": len(ideas), "ideas": ideas,
                 "novelty_summary": novelty_summary},
                stage=self.name,
            )
        )
        artifacts.append(
            self.ctx.save_text(
                "ideas/IDEATION_REPORT.md",
                _render_ideation_report(direction, ideas, survivors, eliminated, warnings),
                stage=self.name,
            )
        )

        detail = (
            f"{len(ideas)} ideas → {len(survivors)} survived "
            f"({len(eliminated)} duplicates) → selected {selected['id']} "
            f"rank={selected['rank']}"
        )
        return StageResult.success(
            detail=detail,
            updates={
                "ideas": ideas,
                "selected_idea": selected,
                "novelty_summary": novelty_summary,
                "warnings": list(state.get("warnings") or []) + warnings,
            },
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------ #
    def _generate_ideas(
        self, direction: str, gaps: list[str], papers: list[dict[str, Any]], max_ideas: int
    ) -> list[dict[str, Any]]:
        gaps_block = "\n".join(f"- {g}" for g in gaps[:8]) or "（未识别出结构化缺口）"
        papers_block = _format_papers_block_from_dicts(papers[:14], max_chars=9000)
        result = self.llm_json(
            "s2_ideas",
            default=None,
            schema_hint=_IDEAS_SCHEMA,
            direction=direction,
            gaps_block=gaps_block,
            papers_block=papers_block,
            max_ideas=max_ideas,
            constraints=(
                f"必须在单机 CPU、{int(getattr(self.ctx.cfg.sandbox, 'timeout', 900))} 秒/次实验的"
                "预算内可验证；不得依赖私有数据集；不得依赖大规模预训练。"
            ),
        )
        if isinstance(result, dict):
            return [x for x in coerce_list(result.get("ideas")) if isinstance(x, dict)]
        if isinstance(result, list):
            return [x for x in result if isinstance(x, dict)]
        return []

    def _check_novelty(
        self,
        idea: dict[str, Any],
        recent: list[dict[str, Any]],
        direction: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        """LLM 判定 + 确定性词面重叠旁证。"""
        query = _idea_query(idea, direction)
        candidates: list[dict[str, Any]] = []
        try:
            found = self.ctx.search.search(query, max_results=6)
            if not found:
                # **零结果必须可见。**
                # 查新在空候选集上会给出"novel"，而那正是它最该避免的结论
                # （提示词自己写着"默认失败模式是高估新颖性"）。
                # 实测：检索式是词袋 + 含 schema 碎片时，arXiv 稳定返回 0 条。
                warnings.append(
                    f"查新检索零结果（idea={idea.get('id')}，query={query!r}）——"
                    "该 idea 的新颖性缺少候选文献支撑，结论应视为 unknown 而非 novel"
                )
            candidates = [p.to_dict() for p in found]
        except Exception as exc:
            self._warn(f"s2: novelty retrieval failed for {idea['id']}: {exc}")
            self.ctx.log_event("retrieve_fail", stage=self.name, idea_id=idea["id"], error=str(exc))

        pool = _merge_papers(candidates, recent)[:8]
        result = self.llm_json(
            "s2_novelty",
            default=None,
            schema_hint=_NOVELTY_SCHEMA,
            idea_block=_format_idea_block(idea),
            candidate_papers_block=_format_papers_block_from_dicts(pool, max_chars=9000)
            or "（未检索到候选文献——请判定为 unknown 并说明检索证据不足）",
        )

        if not isinstance(result, dict):
            warnings.append(f"{idea['id']}: novelty LLM unavailable")
            result = {"verdict": "unknown", "score": 0.5,
                      "rationale": "新颖性判定失败（LLM 或检索不可用）", "closest": []}

        verdict = str(result.get("verdict") or "unknown").lower().strip()
        if verdict not in _VERDICT_SCORE:
            verdict = "unknown"
        score = as_float(result.get("score"), _VERDICT_SCORE[verdict])
        score = max(0.0, min(1.0, score))

        # --- 确定性旁证：与候选题目的词面重叠 ---
        overlap, closest_titles = _max_title_overlap(idea, pool)
        if verdict == "novel" and overlap >= 0.55:
            warnings.append(
                f"{idea['id']}: LLM said novel but title overlap={overlap:.2f} with "
                f"'{closest_titles[:1]}'; downgraded to incremental"
            )
            verdict = "incremental"
            score = min(score, 0.55)
        if verdict == "duplicate" and overlap < 0.25 and not result.get("closest"):
            # 反向保护：LLM 判重但拿不出近邻，证据不足 → 降为 incremental
            warnings.append(f"{idea['id']}: duplicate verdict without supporting near-neighbour")
            verdict = "incremental"

        closest = [c for c in coerce_list(result.get("closest")) if isinstance(c, dict)]
        return {
            "verdict": verdict,
            "score": round(score, 3),
            "rationale": clean_text(str(result.get("rationale") or "")),
            "closest": closest[:3],
            "differentiators": [clean_text(str(x)) for x in coerce_list(result.get("differentiators"))][:5],
            "evidence": {
                "retrieved_candidates": len(candidates),
                "pool_size": len(pool),
                "max_title_overlap": round(overlap, 3),
                "near_titles": closest_titles[:3],
            },
        }


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def _normalize_idea(raw: dict[str, Any], index: int) -> dict[str, Any] | None:
    title = clean_text(str(raw.get("title") or "")).strip()
    hypothesis = clean_text(str(raw.get("hypothesis") or "")).strip()
    if not title and not hypothesis:
        return None
    if not title:
        title = hypothesis[:80]
    return {
        "id": str(raw.get("id") or f"I{index}").strip() or f"I{index}",
        "title": title,
        "hypothesis": hypothesis or title,
        "motivation": clean_text(str(raw.get("motivation") or "")),
        "method_sketch": clean_text(str(raw.get("method_sketch") or "")),
        "novelty_claim": clean_text(str(raw.get("novelty_claim") or "")),
        "feasibility": clean_text(str(raw.get("feasibility") or "")),
        "risks": [clean_text(str(x)) for x in coerce_list(raw.get("risks"))][:6],
        "expected_metrics": [clean_text(str(x)) for x in coerce_list(raw.get("expected_metrics"))][:8],
        "minimal_experiment": clean_text(str(raw.get("minimal_experiment") or "")),
        "required_resources": clean_text(str(raw.get("required_resources") or "")),
        "novelty": {},
        "pilot": {"ok": False, "cmd": [], "result": "not run", "metrics": {}},
        "rank": 0.0,
        "selected": False,
    }


def _placeholder_ideas(direction: str, gaps: list[str], max_ideas: int) -> list[dict[str, Any]]:
    """LLM 完全不可用时的保底：从缺口直接构造可验证假设。

    这些 idea 的 ``novelty.verdict`` 会是 ``unknown``，排序分也低，
    但足以让管线继续跑完并给出一份诚实的（低分的）报告，而不是崩在半路。
    """
    seeds = gaps[: max(1, min(max_ideas, 3))] or [f"{direction} 的现有方法存在未量化的失效模式"]
    out: list[dict[str, Any]] = []
    for i, gap in enumerate(seeds, 1):
        out.append(
            {
                "id": f"I{i}",
                "title": f"针对「{gap[:60]}」的受控消融研究",
                "hypothesis": f"若 {gap[:120]}，则在受控设置下应可观测到可量化的性能差异。",
                "motivation": "由检索缺口直接构造的保守假设（LLM 不可用时的占位方案）。",
                "method_sketch": "固定数据与种子，仅改变目标因子，比较均值±标准差。",
                "novelty_claim": "未评估（LLM 不可用）。",
                "feasibility": "高：单机 CPU 可完成。",
                "risks": ["可能只是复现已有结论", "效应量可能低于噪声"],
                "expected_metrics": ["accuracy", "f1"],
            }
        )
    return out


def _rank(idea: dict[str, Any]) -> float:
    novelty = idea.get("novelty") or {}
    verdict = str(novelty.get("verdict") or "unknown")
    novelty_score = max(as_float(novelty.get("score"), 0.0), _VERDICT_SCORE.get(verdict, 0.5))
    feasibility = _feasibility_score(idea)
    evidence = _evidence_score(idea, novelty)
    return (
        _WEIGHTS["novelty"] * novelty_score
        + _WEIGHTS["feasibility"] * feasibility
        + _WEIGHTS["evidence"] * evidence
    )


def _feasibility_score(idea: dict[str, Any]) -> float:
    """可行性启发式：文本信号 + 明确度。

    这是刻意的粗糙代理——它的作用是**排序**而非判断，且它比「让 LLM 自己
    打个可行性分」更不容易被乐观措辞操纵。
    """
    text = " ".join(
        str(idea.get(k) or "")
        for k in ("feasibility", "method_sketch", "minimal_experiment", "required_resources")
    ).lower()
    score = 0.5
    positives = {
        "cpu": 0.15, "single machine": 0.15, "单机": 0.15, "分钟": 0.1, "minutes": 0.1,
        "公开": 0.1, "public": 0.1, "合成": 0.1, "synthetic": 0.1, "小规模": 0.1,
        "toy": 0.05, "ablation": 0.1, "消融": 0.1, "fixed seed": 0.1, "固定种子": 0.1,
    }
    negatives = {
        "大规模预训练": -0.25, "pretrain": -0.25, "gpu cluster": -0.3, "多机": -0.25,
        "私有数据": -0.2, "proprietary": -0.2, "human evaluation": -0.15, "人工标注": -0.15,
        "billion": -0.2, "十亿": -0.2, "unbounded": -0.15,
    }
    for token, delta in {**positives, **negatives}.items():
        if token in text:
            score += delta
    if not (idea.get("method_sketch") or "").strip():
        score -= 0.15
    if (idea.get("minimal_experiment") or "").strip():
        score += 0.1
    return max(0.0, min(1.0, score))


def _evidence_score(idea: dict[str, Any], novelty: dict[str, Any]) -> float:
    """证据支撑度：被检索文献证成的程度，以及是否有可比的近邻。

    有明确近邻其实是**好事**（说明该方向有可复现的 baseline），
    完全没有近邻则意味着风险高——这里给中等分而不是 0，避免系统性惩罚新方向。
    """
    evidence = novelty.get("evidence") or {}
    pool = int(evidence.get("pool_size") or 0)
    overlap = as_float(evidence.get("max_title_overlap"), 0.0)
    if pool == 0:
        return 0.3
    base = min(1.0, 0.35 + 0.08 * pool)
    # 适度重叠是最好的：既有 baseline 可比，又不同质
    if 0.15 <= overlap <= 0.5:
        base += 0.15
    elif overlap > 0.7:
        base -= 0.2
    if idea.get("expected_metrics"):
        base += 0.05
    return max(0.0, min(1.0, base))


def _recent_papers(papers: list[dict[str, Any]], years: int = 3) -> list[dict[str, Any]]:
    years_seen = [int(p.get("year") or 0) for p in papers if p.get("year")]
    if not years_seen:
        return papers[:10]
    cutoff = max(years_seen) - years
    return [p for p in papers if int(p.get("year") or 0) >= cutoff][:12]


def _merge_papers(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for p in list(a) + list(b):
        key = re.sub(r"[^a-z0-9]", "", str(p.get("title") or "").lower()) or str(p.get("id") or "")
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


#: 构造检索式时要跳过的词。
#:
#: 分三类，第二类是本项目实测踩到的坑：
#:   1. 通用虚词与泛化名词
#:   2. **schema 字段名与引用格式碎片** —— 实测泄漏进检索式的有
#:      ``venue``（输出契约的字段名）、``cite``/``doi``/``iii``（引用格式碎片）。
#:      它们与主题无关，却会被当成关键词发出去。
#:   3. 论文写作的元词
_QUERY_STOPWORDS = frozenset({
    # 1) 通用
    "the", "and", "for", "with", "that", "this", "are", "was", "were", "can", "not",
    "which", "when", "from", "into", "using", "based", "via", "its", "our", "their",
    "than", "then", "also", "such", "more", "most", "less", "very", "both", "each",
    "method", "model", "models", "approach", "approaches", "results", "result",
    "performance", "study", "paper", "work", "works", "novel", "new", "propose",
    "proposed", "proposes", "show", "shows", "shown", "find", "finds", "found",
    # 2) schema 字段名 / 引用格式碎片（实测泄漏）
    "venue", "cite", "cited", "cites", "citation", "doi", "arxiv", "et", "al",
    "string", "bool", "boolean", "float", "int", "integer", "array", "object",
    "json", "schema", "field", "fields", "type", "types", "value", "values",
    "title", "abstract", "hypothesis", "claim", "claims", "score", "scores",
    "verdict", "rationale", "overlap", "risks", "differentiators", "closest",
    "i", "ii", "iii", "iv", "v", "vi", "regime", "regimes",
    # 3) 写作元词
    "however", "therefore", "moreover", "furthermore", "thus", "hence",
    "figure", "figures", "table", "tables", "section", "sections", "eq", "equation",
})

#: 一个 token 至少要有这么多字母才算"词"（滤掉 "s"、"e.g" 这类碎片）
_MIN_TOKEN_ALNUM = 3


def _query_tokens(text: str) -> list[str]:
    r"""从文本里取出可用的检索关键词。

    **不再用 ASCII-only 正则切词。** 早期实现用 ``[A-Za-z][A-Za-z\-]{2,}``，
    于是 ``Körmer`` 被剥掉 ``ö`` 只剩 ``rmer``——作者名被砍成无意义片段，
    检索必然失败。实测在 s2 的检索式里出现过 ``rmer``。

    现在：按 Unicode 字母取词，拉丁字母 + 组合变音符一并保留；
    若一个词去掉非 ASCII 后长度不足，则**整词丢弃**（宁可少一个词，
    也不要发出一个被腰斩的假词）。
    """
    tokens: list[str] = []
    for raw in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", text, flags=re.UNICODE):
        word = raw.strip("-'’")
        if not word:
            continue
        # 纯 ASCII 长度（用于判断"去掉非 ASCII 后还剩多少"）
        ascii_len = sum(1 for ch in word if ch.isascii() and ch.isalpha())
        if ascii_len < _MIN_TOKEN_ALNUM:
            # 形如 "rmer"（原词是 Körmer）或 "ller"（原词是 Müller）——
            # 无法判断它原本是什么，丢弃比发出去安全
            continue
        if word.lower() in _QUERY_STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def _idea_query(idea: dict[str, Any], direction: str) -> str:
    """构造查新检索式。

    **以 title 为主**，而不是把四个字段拼成散文再取前 8 个词。

    原因：四个字段混在一起后，按文档顺序取词得到的是**无序词袋**
    ——实测产出 ``'Hamilton PINN Jacobi Pareto Earth Moon Sun MLP'``
    这种把标题词、方法词、天体名混在一起的东西，arXiv 返回 0 是必然的。
    title 本身最接近一个检索式，也最能代表 idea 的主题。

    有效关键词少于 2 个时落回 ``direction``——宁可检索得宽一点，
    也不要发出一个注定 0 结果的查询（那会让查新在空候选集上做判断）。
    """
    title = str(idea.get("title") or "")
    keywords = _dedupe_preserve(_query_tokens(title))[:8]

    if len(keywords) < 2:
        # title 太短或全被过滤 —— 用 hypothesis 补充
        extra = _dedupe_preserve(
            _query_tokens(str(idea.get("hypothesis") or ""))
        )
        for word in extra:
            if word not in keywords:
                keywords.append(word)
            if len(keywords) >= 6:
                break

    if len(keywords) >= 2:
        return " ".join(keywords[:8])
    return direction


def _max_title_overlap(idea: dict[str, Any], pool: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """候选 idea 的实词集合与候选文献标题的 Jaccard 重叠，取最大者。"""
    idea_text = " ".join(
        str(idea.get(k) or "") for k in ("title", "hypothesis", "method_sketch", "novelty_claim")
    )
    idea_tokens = _content_tokens(idea_text)
    if not idea_tokens:
        return 0.0, []
    best = 0.0
    titles: list[str] = []
    scored: list[tuple[float, str]] = []
    for p in pool:
        title = str(p.get("title") or "")
        tokens = _content_tokens(title)
        if not tokens:
            continue
        inter = len(idea_tokens & tokens)
        union = len(idea_tokens | tokens)
        ratio = inter / union if union else 0.0
        scored.append((ratio, title))
        best = max(best, ratio)
    scored.sort(key=lambda t: -t[0])
    titles = [t for _, t in scored[:3]]
    return best, titles


def _content_tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "using",
        "based", "via", "toward", "towards", "from", "by", "is", "are", "be", "as",
        "we", "our", "this", "that", "it", "its", "can", "not", "new", "novel",
        "method", "model", "approach", "study", "analysis", "paper",
    }
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or "")
        if w.lower() not in stop
    }


def _format_idea_block(idea: dict[str, Any]) -> str:
    lines = [
        f"id: {idea.get('id')}",
        f"title: {idea.get('title')}",
        f"hypothesis: {idea.get('hypothesis')}",
        f"motivation: {idea.get('motivation')}",
        f"method_sketch: {idea.get('method_sketch')}",
        f"novelty_claim (作者自述，可能有偏): {idea.get('novelty_claim')}",
        f"feasibility: {idea.get('feasibility')}",
    ]
    risks = idea.get("risks") or []
    if risks:
        lines.append("risks: " + "; ".join(str(r) for r in risks))
    metrics = idea.get("expected_metrics") or []
    if metrics:
        lines.append("expected_metrics: " + ", ".join(str(m) for m in metrics))
    return "\n".join(lines)


def _format_papers_block_from_dicts(papers: list[dict[str, Any]], max_chars: int = 9000) -> str:
    """``state["papers"]`` 是 dict 列表，这里复用 s1 的格式化逻辑。

    为了不依赖 ``Paper`` 对象的构造（dict 可能缺字段），用一个轻量 shim。
    """
    from types import SimpleNamespace

    shims = []
    for p in papers:
        shims.append(
            SimpleNamespace(
                id=p.get("id", ""),
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                tldr=p.get("tldr", ""),
                authors=p.get("authors", []) or [],
                year=p.get("year"),
                venue=p.get("venue", ""),
                citation_count=p.get("citation_count", 0),
                source=p.get("source", ""),
            )
        )
    return _format_papers_block(shims, max_chars=max_chars)


def _count_verdicts(ideas: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idea in ideas:
        verdict = str((idea.get("novelty") or {}).get("verdict") or "unknown")
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def _render_ideation_report(
    direction: str,
    ideas: list[dict[str, Any]],
    survivors: list[dict[str, Any]],
    eliminated: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    parts = [
        f"# 构思报告：{direction}",
        "",
        f"候选假设 {len(ideas)} 个，淘汰 {len(eliminated)} 个（判定为重复），"
        f"保留 {len(survivors)} 个。排序权重："
        + ", ".join(f"{k}={v}" for k, v in _WEIGHTS.items())
        + "。",
        "",
        "## 排序后的候选方向",
        "",
    ]
    for idea in survivors:
        novelty = idea.get("novelty") or {}
        parts += [
            f"### {idea['id']}. {idea['title']}  —  rank {idea.get('rank')}",
            "",
            f"- **假设**：{idea['hypothesis']}",
            f"- **动机**：{idea.get('motivation') or '—'}",
            f"- **方法草图**：{idea.get('method_sketch') or '—'}",
            f"- **新颖性**：`{novelty.get('verdict')}` (score={novelty.get('score')})"
            f" — {novelty.get('rationale') or '—'}",
            f"- **可行性**：{idea.get('feasibility') or '—'}",
        ]
        risks = idea.get("risks") or []
        if risks:
            parts.append("- **风险**：" + "；".join(str(r) for r in risks))
        closest = novelty.get("closest") or []
        if closest:
            parts.append("- **最接近的工作**：")
            for c in closest:
                parts.append(
                    f"    - {c.get('title', '?')} ({c.get('year', 'n/a')}, `{c.get('id', '')}`)"
                    f" — {c.get('why', '')}"
                )
        evidence = novelty.get("evidence") or {}
        if evidence:
            parts.append(
                f"- **查新旁证**：检索候选 {evidence.get('retrieved_candidates')} 篇，"
                f"近邻池 {evidence.get('pool_size')} 篇，最大标题重叠 "
                f"{evidence.get('max_title_overlap')}"
            )
        parts.append("")

    if eliminated:
        parts += ["## 已淘汰（与现有工作重复）", ""]
        for idea in eliminated:
            novelty = idea.get("novelty") or {}
            parts.append(
                f"- **{idea['id']}. {idea['title']}** — {novelty.get('rationale') or '判定为重复'}"
            )
        parts.append("")

    if warnings:
        parts += ["## 告警", ""] + [f"- {w}" for w in warnings] + [""]

    return "\n".join(parts)


__all__ = ["IdeationStage"]
