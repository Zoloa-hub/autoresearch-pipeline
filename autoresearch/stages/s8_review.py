"""阶段⑧：自动同行评审与迭代修改。

评审不是「打个分」，而是**带可核查修复项的门禁**：

* 每条 weakness 必须带 ``min_fix``（具体、可执行）与 ``location``（改哪里）——
  没有修复建议的弱点无法驱动修改，只会让循环空转。
* 分数必须写进 ``review_score``，verdict 写进 ``review_verdict``，
  并由 ``route()`` 决定「继续改」还是「进入交付」。
* 停止条件有三重保护，防止无限循环烧钱：
  (a) 达到 ``max_review_rounds``；
  (b) 分数达标且 verdict ∈ {ready, almost}；
  (c) **收益递减**——连续两轮主分提升 < 0.2 就停，继续改只会重新排列措辞。

修改阶段严格禁止造数据：如果某条 major weakness 无法用现有证据解决，
必须写进 ``unaddressed`` 并给出理由，最终会出现在论文的 Limitations 与
交付报告的「未解决问题」清单里。**宁可在报告里承认不足，也不能编造实验。**
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..graph.state import Artifact
from .base import Stage, StageResult, clamp, clean_text, coerce_list

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "verdict": {"type": "string", "enum": ["ready", "almost", "revise", "reject"]},
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "object"}},
        "weaknesses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "severity": {"type": "string", "enum": ["major", "minor"]},
                    "evidence": {"type": "string"},
                    "min_fix": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["point"],
            },
        },
        "questions": {"type": "array", "items": {"type": "string"}},
        "per_criterion": {"type": "object"},
        "confidence": {"type": "number"},
        "recommendation": {"type": "string"},
    },
    "required": ["score", "verdict"],
}

_REVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "revisions": {"type": "array", "items": {"type": "object"}},
        "unaddressed": {"type": "array", "items": {"type": "object"}},
        "sections": {"type": "object"},
        "response_letter": {"type": "string"},
    },
    "required": ["revisions"],
}

#: 触发「进入交付」的分数门槛与判定集合。
ACCEPT_SCORE = 6.0
ACCEPT_VERDICTS = {"ready", "almost"}
#: 收益递减阈值：连续两轮提升小于它就不再继续迭代。
PLATEAU_DELTA = 0.2

#: 章节名 → 模板文件名（与 s6 的 SECTION_PLAN 对齐）。
_SECTION_FILES = {
    "method": "method.tex",
    "experiments": "experiments.tex",
    "results": "results.tex",
    "related_work": "related_work.tex",
    "introduction": "introduction.tex",
    "discussion": "discussion.tex",
    "limitations": "limitations.tex",
    "conclusion": "conclusion.tex",
    "abstract": "abstract.tex",
}


class ReviewStage(Stage):
    name = "s8_review"
    title = "自动同行评审与迭代修改"
    requires = ("paper_sections",)
    produces = ("reviews", "review_score", "review_verdict", "review_round", "revision_history")
    max_attempts = 2

    def run(self, state: dict[str, Any]) -> StageResult:
        warnings: list[str] = []
        artifacts: list[Artifact] = []
        max_rounds = max(1, int(getattr(self.ctx.cfg, "max_review_rounds", 3) or 3))
        round_no = int(state.get("review_round") or 0) + 1
        prior_reviews = list(state.get("reviews") or [])
        revision_history = list(state.get("revision_history") or [])

        if round_no > max_rounds:
            warnings.append(f"review round budget exhausted ({max_rounds})")
            return StageResult.success(
                detail=f"review budget exhausted at round {max_rounds}",
                updates={"warnings": list(state.get("warnings") or []) + warnings},
                artifacts=artifacts,
            )

        # --- 1. 评审 ---------------------------------------------------- #
        review = self._review(state, round_no, prior_reviews, warnings)
        score = _as_score(review.get("score"), prior_reviews)
        verdict = _normalize_verdict(review.get("verdict"), score)
        review["score"] = score
        review["verdict"] = verdict
        review["round"] = round_no
        prior_reviews.append(review)

        scores = [float(r.get("score") or 0.0) for r in prior_reviews if isinstance(r, dict)]
        delta = scores[-1] - scores[-2] if len(scores) >= 2 else None
        plateau = bool(delta is not None and abs(delta) < PLATEAU_DELTA and round_no >= 2)
        accepted = score >= ACCEPT_SCORE and verdict in ACCEPT_VERDICTS
        budget_left = round_no < max_rounds

        self.ctx.log_event(
            "review_round",
            stage=self.name,
            round=round_no,
            score=score,
            verdict=verdict,
            accepted=accepted,
            delta=delta,
            plateau=plateau,
            majors=sum(
                1 for w in coerce_list(review.get("weaknesses"))
                if isinstance(w, dict) and w.get("severity") == "major"
            ),
        )

        artifacts.append(
            self.ctx.save_json(
                f"review/round_{round_no}.json", review, stage=self.name
            )
        )
        artifacts.append(
            self.ctx.save_text(
                f"review/round_{round_no}.md",
                _render_review(review, round_no, max_rounds),
                stage=self.name,
            )
        )

        # --- 2. 决定是否修改 ------------------------------------------- #
        should_revise = budget_left and not accepted and not plateau
        if should_revise:
            revision = self._revise(state, review, round_no, max_rounds, warnings)
            if revision:
                revision_history.append(revision)
                self._apply_section_updates(state, revision, warnings)
                artifacts.append(
                    self.ctx.save_json(
                        f"review/revision_{round_no}.json", revision, stage=self.name
                    )
                )
                artifacts.append(
                    self.ctx.save_text(
                        f"review/revision_{round_no}.md",
                        _render_revision(revision, round_no),
                        stage=self.name,
                    )
                )

        # --- 3. 汇总评审历史 ------------------------------------------- #
        artifacts.append(
            self.ctx.save_text(
                "review/AUTO_REVIEW.md",
                _render_review_history(prior_reviews, revision_history, max_rounds, accepted, plateau),
                stage=self.name,
            )
        )

        detail = (
            f"round {round_no}/{max_rounds}: score {score:.1f}, verdict {verdict}"
            + (f", Δ={delta:+.2f}" if delta is not None else "")
            + (" → accepted" if accepted else (" → plateau stop" if plateau else " → revise"))
        )
        return StageResult.success(
            detail=detail,
            updates={
                "reviews": prior_reviews,
                "review_score": score,
                "review_verdict": verdict,
                "review_round": round_no,
                "revision_history": revision_history,
                "warnings": list(state.get("warnings") or []) + warnings,
            },
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------ #
    # 路由：是否需要再改一轮
    # ------------------------------------------------------------------ #
    def route(self, state: dict[str, Any]) -> str | None:
        """返回 ``s6_writing`` 继续迭代；``None`` 表示按声明顺序进入 s9。"""
        score = float(state.get("review_score") or 0.0)
        verdict = str(state.get("review_verdict") or "")
        round_no = int(state.get("review_round") or 0)
        max_rounds = max(1, int(getattr(self.ctx.cfg, "max_review_rounds", 3) or 3))

        reviews = [r for r in (state.get("reviews") or []) if isinstance(r, dict)]
        scores = [float(r.get("score") or 0.0) for r in reviews]
        plateau = len(scores) >= 2 and abs(scores[-1] - scores[-2]) < PLATEAU_DELTA

        if score >= ACCEPT_SCORE and verdict in ACCEPT_VERDICTS:
            self.ctx.log_event("review_stop", stage=self.name, reason="accepted",
                               score=score, verdict=verdict)
            return None
        if round_no >= max_rounds:
            self.ctx.log_event("review_stop", stage=self.name, reason="max_rounds",
                               round=round_no, score=score)
            return None
        if plateau:
            self.ctx.log_event("review_stop", stage=self.name, reason="plateau",
                               scores=scores[-2:])
            return None
        self.ctx.log_event("review_loop", stage=self.name, to="s6_writing",
                           round=round_no, score=score, verdict=verdict)
        return "s6_writing"

    # ------------------------------------------------------------------ #
    def _review(
        self,
        state: dict[str, Any],
        round_no: int,
        prior: list[dict[str, Any]],
        warnings: list[str],
    ) -> dict[str, Any]:
        paper_text = self._paper_text(state)
        inventory = _inventory_block(state)
        prior_weaknesses = _prior_weaknesses_block(prior)
        venue = str(getattr(self.ctx.cfg, "venue", "NeurIPS"))

        result = self.llm_json(
            "s8_review",
            default=None,
            schema_hint=_REVIEW_SCHEMA,
            venue=venue,
            paper_text=clamp(paper_text, 26000),
            figure_table_inventory=inventory,
            round=round_no,
            prior_weaknesses_block=prior_weaknesses,
        )
        if isinstance(result, dict):
            return result

        warnings.append("review LLM unavailable; using the deterministic self-check reviewer")
        return _selfcheck_review(state, round_no, venue)

    def _revise(
        self,
        state: dict[str, Any],
        review: dict[str, Any],
        round_no: int,
        max_rounds: int,
        warnings: list[str],
    ) -> dict[str, Any] | None:
        paper_text = self._paper_text(state)
        result = self.llm_json(
            "s6_revision",
            default=None,
            schema_hint=_REVISION_SCHEMA,
            paper_text=clamp(paper_text, 24000),
            review_block=clamp(_render_review(review, round_no, max_rounds), 8000),
            round=round_no,
            max_rounds=max_rounds,
        )
        if isinstance(result, dict):
            return result
        warnings.append(f"round {round_no}: revision LLM unavailable; no changes applied")
        return None

    def _apply_section_updates(
        self, state: dict[str, Any], revision: dict[str, Any], warnings: list[str]
    ) -> None:
        """把修改后的章节写回 tex 文件，并同步进状态。"""
        sections = state.get("paper_sections")
        if not isinstance(sections, dict):
            sections = {}
            state["paper_sections"] = sections
        paper_dir = self.ctx.path("paper")

        new_sections = revision.get("sections")
        if not isinstance(new_sections, dict):
            return
        for raw_name, body in new_sections.items():
            key = str(raw_name).strip().lower().replace(" ", "_")
            body = str(body or "").strip()
            if not body:
                continue
            filename = _SECTION_FILES.get(key)
            if filename is None:
                # 允许 “Experimental Setup” 这类别名
                for candidate, fname in _SECTION_FILES.items():
                    if candidate in key or key in candidate:
                        key, filename = candidate, fname
                        break
            if filename is None:
                warnings.append(f"revision returned an unknown section '{raw_name}'; ignored")
                continue
            body = _strip_fences(body)
            sections[key] = body
            display = "Abstract" if key == "abstract" else key.replace("_", " ").title()
            try:
                target = paper_dir / "sections" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                stripped = re.sub(r"\\section\*?\{[^}]*\}", "", body, count=1).strip()
                label = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
                if key == "abstract":
                    target.write_text(body + "\n", encoding="utf-8")
                else:
                    target.write_text(
                        f"\\section{{{display}}}\n\\label{{sec:{label}}}\n\n{stripped}\n",
                        encoding="utf-8",
                    )
            except OSError as exc:
                warnings.append(f"could not write revised section {key}: {exc}")
        if sections.get("abstract"):
            state["paper_abstract"] = sections["abstract"]
        self.ctx.log_event(
            "revision_applied",
            stage=self.name,
            sections=sorted(str(k) for k in new_sections),
            unaddressed=len(coerce_list(revision.get("unaddressed"))),
        )

    def _paper_text(self, state: dict[str, Any]) -> str:
        """评审输入：优先读盘上的 tex（可捕捉手改），退化到状态里的章节。"""
        paper_dir = self.ctx.path("paper")
        chunks: list[str] = []
        abstract = state.get("paper_abstract")
        title = state.get("paper_title")
        if title:
            chunks.append(f"# {title}")
        if abstract:
            chunks.append(f"## Abstract\n{abstract}")
        sections = self.state_dict(state, "paper_sections")
        order = ["introduction", "related_work", "method", "experiments", "results",
                 "discussion", "limitations", "conclusion"]
        for key in order:
            body = sections.get(key)
            if body:
                chunks.append(f"## {key.replace('_', ' ').title()}\n{body}")
        if len(chunks) <= 1:
            for path in sorted((paper_dir / "sections").glob("*.tex")):
                try:
                    chunks.append(f"## {path.stem}\n{path.read_text(encoding='utf-8', errors='replace')}")
                except OSError:
                    continue
        compile_result = self.state_dict(state, "compile_result")
        if compile_result:
            chunks.append(
                f"## Build status\nengine={compile_result.get('engine')} "
                f"ok={compile_result.get('ok')} errors={len(compile_result.get('errors') or [])}"
            )
        return "\n\n".join(chunks)

    #: 引擎通过**类属性** ``router`` 识别条件边
    #: （``linear_nodes`` 里是 ``getattr(impl, "router", None)``）。
    #: 必须在类体末尾显式把 ``route`` 绑上去：只定义 ``route()`` 的话，
    #: ``getattr`` 会命中基类的 ``router = None``，条件边永远不会装配，
    #: **评审循环会静默失效**——分数再低也不会触发第二轮修改。
    #: 这个 bug 只有跑真实 LLM 才看得出来（mock 下没人注意轮次）。
    router = route


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def _as_score(value: Any, prior: list[dict[str, Any]]) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    if score <= 0:
        # 解析失败时保守给分：不高于上一轮
        prev = float(prior[-1].get("score") or 0.0) if prior else 4.0
        score = prev if prev else 4.0
    return max(1.0, min(10.0, round(score, 2)))


def _normalize_verdict(value: Any, score: float) -> str:
    verdict = str(value or "").strip().lower()
    if verdict not in ("ready", "almost", "revise", "reject"):
        if score >= 7.5:
            verdict = "ready"
        elif score >= 6.0:
            verdict = "almost"
        elif score >= 4.0:
            verdict = "revise"
        else:
            verdict = "reject"
    # 一致性修正：分数与判定不能自相矛盾
    if verdict in ("ready", "almost") and score < ACCEPT_SCORE:
        verdict = "revise"
    if verdict == "reject" and score >= ACCEPT_SCORE:
        verdict = "revise"
    return verdict


def _selfcheck_review(
    state: dict[str, Any], round_no: int, venue: str
) -> dict[str, Any]:
    """LLM 不可用时的确定性自检评审。

    不是「假装评审」，而是把**代码能查的事实**查出来：引用是否落库、编译是否通过、
    指标是否存在、消融是否做过、局限是否写了。分数由这些硬事实推出来，
    因此它偏低但可解释。
    """
    weaknesses: list[dict[str, Any]] = []
    strengths: list[dict[str, Any]] = []

    sections = state.get("paper_sections") or {}
    citations = set(state.get("citations_used") or [])
    bib_keys = {
        str(e.get("bibkey"))
        for e in (state.get("bib_entries") or [])
        if isinstance(e, dict) and e.get("bibkey")
    }
    compile_result = state.get("compile_result") or {}
    comparison = (state.get("experiment/results") or {}).get("comparison") \
        if isinstance(state.get("experiment/results"), dict) else None
    review = state.get("analysis") or {}

    if not compile_result.get("ok"):
        weaknesses.append(
            {
                "point": "论文未能编译成 PDF。",
                "severity": "major",
                "evidence": f"engine={compile_result.get('engine')}",
                "min_fix": "安装 tectonic（或本地 TeX）后重新编译，确保零 `!` 级错误。",
                "location": "paper/main.tex",
            }
        )
    else:
        strengths.append({"point": "论文可编译为 PDF。", "evidence": "compile ok"})

    unknown = citations - bib_keys if bib_keys else set()
    if unknown:
        weaknesses.append(
            {
                "point": f"存在 {len(unknown)} 个未落库的引用。",
                "severity": "major",
                "evidence": ", ".join(sorted(unknown)[:5]),
                "min_fix": "补齐 .bib 条目或删除这些 \\cite。",
                "location": "paper/references.bib",
            }
        )

    if isinstance(comparison, dict) and comparison.get("available"):
        if comparison.get("regressed"):
            weaknesses.append(
                {
                    "point": "存在指标回退的维度，削弱了主 claim。",
                    "severity": "major",
                    "evidence": ", ".join(comparison["regressed"][:5]),
                    "min_fix": "补充解释或在正文中限定 claim 的适用范围。",
                    "location": "sections/results.tex",
                }
            )
        if len(comparison.get("improved") or []) >= 1:
            strengths.append(
                {"point": "主指标相对基线有可测提升。",
                 "evidence": str(comparison.get("summary_line"))}
            )
    else:
        weaknesses.append(
            {
                "point": "缺少可对照的 baseline/method 定量结果。",
                "severity": "major",
                "evidence": "comparison unavailable",
                "min_fix": "确保两个变体都产出 metrics.csv 并重新运行实验。",
                "location": "experiment/",
            }
        )

    if not (state.get("experiment_plan") or {}).get("ablation_matrix"):
        weaknesses.append(
            {
                "point": "未做消融实验。",
                "severity": "major",
                "evidence": "ablation_matrix empty",
                "min_fix": "至少完成种子稳健性与容量对照两组消融。",
                "location": "sections/experiments.tex",
            }
        )

    if not str(sections.get("limitations") or "").strip():
        weaknesses.append(
            {
                "point": "缺少 Limitations 章节。",
                "severity": "minor",
                "evidence": "empty section",
                "min_fix": "补充局限性讨论。",
                "location": "sections/limitations.tex",
            }
        )

    for key in ("method", "results", "introduction"):
        if len(str(sections.get(key) or "")) < 400:
            weaknesses.append(
                {
                    "point": f"`{key}` 章节过短，内容不充分。",
                    "severity": "minor" if key != "results" else "major",
                    "evidence": f"{len(str(sections.get(key) or ''))} chars",
                    "min_fix": "补充方法与结果的具体描述。",
                    "location": f"sections/{_SECTION_FILES.get(key, key + '.tex')}",
                }
            )

    majors = sum(1 for w in weaknesses if w["severity"] == "major")
    score = max(1.0, 8.0 - 1.2 * majors - 0.3 * (len(weaknesses) - majors))
    verdict = "reject" if score < 3 else "revise" if score < ACCEPT_SCORE else "almost"
    return {
        "score": round(score, 2),
        "verdict": verdict,
        "summary": (
            f"确定性自检评审（LLM 不可用）。共发现 {len(weaknesses)} 项问题，"
            f"其中 {majors} 项为 major。评分基于可机检的硬事实（编译、引用落库、"
            f"指标对照、消融完备性、章节长度），因此偏低但可复现。"
        ),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "questions": [],
        "per_criterion": {
            "novelty": 5.0,
            "rigor": round(min(10.0, 4.0 + (1.0 if comparison else 0.0)), 2),
            "clarity": 5.0,
            "experiments": round(min(10.0, 3.0 + (2.0 if comparison else 0.0)), 2),
            "reproducibility": 6.0 if state.get("code_files") else 3.0,
        },
        "confidence": 0.4,
        "recommendation": "确定性自检，仅供内部质量门禁使用，不替代真实同行评审。",
        "_generated_by": "deterministic-selfcheck",
        "_venue": venue,
    }


def _inventory_block(state: dict[str, Any]) -> str:
    lines: list[str] = []
    figured = state.get("figured") or {}
    if figured:
        lines.append("figures:")
        for name, paths in figured.items():
            lines.append(f"  - {name}: {len(paths)} file(s)")
    tables = state.get("latex_tables") or {}
    if tables:
        lines.append("tables: " + ", ".join(tables))
    runs = state.get("runs_executed") or []
    if runs:
        lines.append("runs:")
        for r in runs:
            if isinstance(r, dict):
                lines.append(
                    f"  - {r.get('variant')}: ok={r.get('ok')} rounds={r.get('rounds')} "
                    f"metrics={r.get('metrics_found')}"
                )
    debug = state.get("debug_history") or []
    flagged = [d for d in debug if isinstance(d, dict) and d.get("validity_flag")]
    if flagged:
        lines.append(f"validity_flags: {len(flagged)} debug round(s) flagged for fidelity")
    return "\n".join(lines) or "（无图表与运行记录）"


def _prior_weaknesses_block(prior: list[dict[str, Any]]) -> str:
    if not prior:
        return "（本轮为首轮评审）"
    lines: list[str] = []
    for r in prior:
        if not isinstance(r, dict):
            continue
        lines.append(f"round {r.get('round')} (score={r.get('score')}, verdict={r.get('verdict')}):")
        for w in coerce_list(r.get("weaknesses")):
            if isinstance(w, dict):
                lines.append(
                    f"  - [{w.get('severity', 'minor')}] {w.get('point')} "
                    f"(fix: {w.get('min_fix') or '未给出'})"
                )
    return "\n".join(lines) or "（无历史弱点）"


def _strip_fences(text: str) -> str:
    m = re.match(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", (text or "").strip(), re.S)
    return m.group(1).strip() if m else (text or "").strip()


def _render_review(review: dict[str, Any], round_no: int, max_rounds: int) -> str:
    parts = [
        f"# 评审报告 — 第 {round_no}/{max_rounds} 轮",
        "",
        f"**分数**：{review.get('score')}/10　**判定**：`{review.get('verdict')}`"
        + (f"　**置信度**：{review.get('confidence')}" if review.get("confidence") else ""),
        "",
        f"**总评**：{review.get('summary') or '—'}",
        "",
    ]
    per = review.get("per_criterion")
    if isinstance(per, dict) and per:
        parts += ["## 分项评分", "", "| 维度 | 分数 |", "|---|---|"]
        for k, v in per.items():
            try:
                parts.append(f"| {k} | {float(v):.1f} |")
            except (TypeError, ValueError):
                parts.append(f"| {k} | {v} |")
        parts.append("")

    strengths = coerce_list(review.get("strengths"))
    if strengths:
        parts += ["## 优点", ""]
        for s in strengths:
            if isinstance(s, dict):
                parts.append(f"- **{s.get('point', '')}** — {s.get('evidence', '')}")
            else:
                parts.append(f"- {s}")
        parts.append("")

    weaknesses = coerce_list(review.get("weaknesses"))
    if weaknesses:
        parts += ["## 弱点与最小修复", ""]
        for i, w in enumerate(weaknesses, 1):
            if not isinstance(w, dict):
                parts.append(f"{i}. {w}")
                continue
            parts.append(
                f"{i}. **[{w.get('severity', 'minor')}] {w.get('point', '')}**  \n"
                f"   - 证据：{w.get('evidence', '—')}  \n"
                f"   - 最小修复：{w.get('min_fix', '—')}  \n"
                f"   - 位置：`{w.get('location', '—')}`"
            )
        parts.append("")

    questions = coerce_list(review.get("questions"))
    if questions:
        parts += ["## 待澄清问题", ""] + [f"- {q}" for q in questions] + [""]

    if review.get("recommendation"):
        parts += ["## 建议", "", str(review["recommendation"]), ""]
    return "\n".join(parts)


def _render_revision(revision: dict[str, Any], round_no: int) -> str:
    parts = [f"# 第 {round_no} 轮修改记录", ""]
    revisions = coerce_list(revision.get("revisions"))
    if revisions:
        parts += ["## 已做修改", "", "| 位置 | 问题 | 修改 | 影响 |", "|---|---|---|---|"]
        for r in revisions:
            if isinstance(r, dict):
                parts.append(
                    f"| {r.get('location', '—')} | {r.get('issue', '—')} | "
                    f"{r.get('fix', '—')} | {r.get('change_summary', '—')} |"
                )
        parts.append("")

    unaddressed = coerce_list(revision.get("unaddressed"))
    if unaddressed:
        parts += ["## 未解决问题（诚实记录）", ""]
        for u in unaddressed:
            if isinstance(u, dict):
                parts.append(f"- **{u.get('issue', '')}** — 原因：{u.get('reason', '')}")
            else:
                parts.append(f"- {u}")
        parts.append("")

    if revision.get("response_letter"):
        parts += ["## 回复审稿人", "", str(revision["response_letter"]), ""]
    return "\n".join(parts)


def _render_review_history(
    reviews: list[dict[str, Any]],
    revisions: list[dict[str, Any]],
    max_rounds: int,
    accepted: bool,
    plateau: bool,
) -> str:
    parts = [
        "# 自动评审历史",
        "",
        f"共 {len(reviews)}/{max_rounds} 轮。"
        + ("已达标进入交付。" if accepted else ("收益递减，提前停止。" if plateau else "预算用尽。")),
        "",
        "| 轮次 | 分数 | 判定 | major 数 | 已做修改 |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(reviews):
        majors = sum(
            1 for w in coerce_list(r.get("weaknesses"))
            if isinstance(w, dict) and w.get("severity") == "major"
        )
        rev = revisions[i] if i < len(revisions) else {}
        n_rev = len(coerce_list((rev or {}).get("revisions")))
        parts.append(
            f"| {r.get('round', i + 1)} | {r.get('score')} | `{r.get('verdict')}` | "
            f"{majors} | {n_rev} |"
        )
    parts.append("")

    if len(reviews) >= 2:
        scores = [float(r.get("score") or 0) for r in reviews]
        parts += [
            "## 分数轨迹",
            "",
            " → ".join(f"{s:.1f}" for s in scores),
            "",
        ]

    for r in reviews:
        parts += [_render_review(r, int(r.get("round") or 0), max_rounds), ""]

    for i, rev in enumerate(revisions):
        parts += [_render_revision(rev, i + 1), ""]

    return "\n".join(parts)


__all__ = ["ReviewStage", "ACCEPT_SCORE", "ACCEPT_VERDICTS", "PLATEAU_DELTA"]
