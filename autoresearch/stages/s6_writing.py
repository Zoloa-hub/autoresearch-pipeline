"""阶段⑥：论文分章撰写。

写作顺序是**自底向上**的，而且这个顺序本身是有道理的，不只是偏好：

1. ``method`` / ``experiments`` —— 事实与数据支撑最充分，先写能锁定「论文到底做了什么」；
2. ``results`` —— 基于 ⑤ 已分级好的证据写讨论，避免先把 claim 说满再去凑数据；
3. ``related_work`` / ``introduction`` —— 此时才清楚该把贡献摆在哪个坐标系里；
4. ``abstract`` / ``conclusion`` —— 最后提炼，此时全部内容已成文，不可能吹过头；
5. ``limitations`` / ``discussion`` —— 由 ⑤ 的效度威胁清单直接生成，不靠 LLM 自觉。

两条硬约束贯穿全部章节：
* **引用闭集**：只能 ``\\cite`` 参考文献库里已存在的 key。模型每用一次都记账，
  不在库里的 key 被丢弃并记事件——杜绝「编造引用」这类最致命的问题。
* **数字溯源**：所有数值必须逐字来自 ⑤ 的指标块，写作提示词里明确禁止改写。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..graph.state import Artifact
from .base import (
    Stage,
    StageResult,
    clamp,
    clean_text,
    coerce_list,
    strip_directive_echo,
    title_from_text,
)

_SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "latex": {"type": "string"},
        "citations_used": {"type": "array", "items": {"type": "string"}},
        "word_count": {"type": "integer"},
    },
    "required": ["latex"],
}

_ABSTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "title_candidates": {"type": "array", "items": {"type": "string"}},
        "abstract": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "contributions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["abstract"],
}

#: 写作顺序与模板文件名。值 = (模板文件名, 章节显示名, 目标字数)
SECTION_PLAN: tuple[tuple[str, str, str, int], ...] = (
    ("method", "method.tex", "Method", 700),
    ("experiments", "experiments.tex", "Experimental Setup", 500),
    ("results", "results.tex", "Results and Analysis", 800),
    ("related_work", "related_work.tex", "Related Work", 600),
    ("introduction", "introduction.tex", "Introduction", 800),
    ("discussion", "discussion.tex", "Discussion", 500),
    ("limitations", "limitations.tex", "Limitations", 300),
    ("conclusion", "conclusion.tex", "Conclusion", 300),
)

_SECTION_INSTRUCTIONS: dict[str, str] = {
    "method": (
        "精确描述所提方法：问题形式化、符号定义、算法步骤、与基线的差异。"
        "必须与实验代码实际实现的行为一致；不要描述代码里没有的机制。"
    ),
    "experiments": (
        "描述实验设置：数据、划分、基线、评价指标、超参与运行次数、随机种子策略。"
        "只写实际做过的设置。"
    ),
    "results": (
        "基于给定的指标块展开讨论。每个定量陈述后紧跟具体数字（含均值和标准差）。"
        "必须如实报告负面或不显著的结果，并给出可能的解释。"
        "讨论图/表时使用给定的 label。"
    ),
    "related_work": (
        "按主题组织，而非逐篇罗列。每个主题说明代表性工作与本文的差异。"
        "只能引用参考文献库中已有的 key。"
    ),
    "introduction": (
        "问题背景 → 现有方法的不足 → 本文思路 → 贡献列表 → 结果概览。"
        "贡献列表要具体可核查，不要写「我们提出了一个新颖的框架」这类空话。"
    ),
    "discussion": (
        "解释结果背后的机制，讨论与文献中已有结论的一致或冲突之处，"
        "指出结果的适用边界。"
    ),
    "limitations": (
        "诚实列出局限：数据规模、评价指标的局限、未做的实验、"
        "结论可能失效的条件。不要写成变相的贡献声明。"
    ),
    "conclusion": "总结贡献与证据，给出 2-3 条具体的未来工作方向（要可执行）。",
}

_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")


class WritingStage(Stage):
    name = "s6_writing"
    title = "论文分章撰写"
    requires = ("experiment_plan",)
    produces = ("paper_sections", "paper_title", "paper_abstract", "paper_tex", "citations_used")
    max_attempts = 2

    # ------------------------------------------------------------------ #
    def run(self, state: dict[str, Any]) -> StageResult:
        warnings: list[str] = []
        artifacts: list[Artifact] = []

        # --- 0. 迭代路径：已有论文时只应用评审意见，不重新生成 ---------- #
        # 评审循环会把控制权交回本阶段。若此时无条件重跑全部章节，会：
        #   ① 白花一整轮 LLM 成本（8 章节 ≈ 6 万 tokens / 约 1 分钟）；
        #   ② **丢失上一轮针对评审意见做的修改**，让迭代变成原地打转。
        # 正确行为是：s8 已经把修改后的章节写进状态与磁盘，这里只做规范化落地。
        existing = self.state_dict(state, "paper_sections")
        if existing and int(state.get("review_round") or 0) > 0:
            return self._apply_iteration(state, existing, warnings)

        missing = self.check_requires(state)
        if missing:
            warnings.append(f"writing degraded: missing {missing}")

        paper_dir = self.ctx.path("paper")
        paper_dir.mkdir(parents=True, exist_ok=True)
        bib_keys = self._bib_keys(state)
        evidence = self._evidence_block(state)

        # --- 1. 自底向上逐章生成 --------------------------------------- #
        sections: dict[str, str] = {}
        citations_used: list[str] = []
        section_meta: dict[str, Any] = {}

        for key, filename, display, words in SECTION_PLAN:
            body, used = self._write_section(
                key, display, words, state, evidence, bib_keys, warnings
            )
            if not body:
                body = _placeholder_section(display, key, state)
                warnings.append(f"section '{key}' fell back to a placeholder")
            body = self._sanitize_citations(body, bib_keys, warnings, key)
            sections[key] = body
            citations_used.extend(used)
            section_meta[key] = {
                "display": display,
                "file": filename,
                "words": word_count(body),
                "chars": len(body),
            }
            # 立即落盘：即使后续章节失败，已写好的部分也能被编译/评审
            written = self._write_section_file(paper_dir, filename, display, body, state)
            artifacts.append(self.ctx.artifact(written, kind="tex", stage=self.name))

        # --- 2. 摘要与标题 --------------------------------------------- #
        title, abstract, keywords = self._write_abstract(state, sections, warnings)
        abstract_path = self._write_section_file(
            paper_dir, "abstract.tex", "Abstract", abstract, state
        )
        artifacts.append(self.ctx.artifact(abstract_path, kind="tex", stage=self.name))

        # --- 3. 装配 main.tex ------------------------------------------ #
        main_tex = self._assemble_main(paper_dir, title, abstract, warnings, keywords)
        artifacts.append(self.ctx.artifact(main_tex, kind="tex", stage=self.name))

        # --- 4. 写参考文献库 ------------------------------------------- #
        bib_path = self._write_bib(paper_dir, state, citations_used, warnings)
        if bib_path is not None:
            artifacts.append(self.ctx.artifact(bib_path, kind="bib", stage=self.name))

        # --- 5. 元数据落盘 --------------------------------------------- #
        artifacts.append(
            self.ctx.save_json(
                "paper/paper_meta.json",
                {
                    "title": title,
                    "abstract": abstract,
                    # 关键词必须落盘。评审回环会重跑本阶段，若只存在内存里，
                    # 回环路径就会用兜底串覆盖 LLM 生成的关键词——不报错、
                    # 不记事件，恰好是"安静地把错的东西印进 PDF"。
                    "keywords": keywords,
                    "sections": list(sections),
                    "section_meta": section_meta,
                    "citations_used": sorted(set(citations_used)),
                    "bib_keys_available": sorted(bib_keys),
                    "bib_keys_unused": sorted(bib_keys - set(citations_used)),
                },
                stage=self.name,
            )
        )

        detail = (
            f"{len(sections)} sections, {sum(m['words'] for m in section_meta.values())} words, "
            f"{len(set(citations_used))}/{len(bib_keys)} refs cited, title='{title[:60]}'"
        )
        return StageResult.success(
            detail=detail,
            updates={
                "paper_sections": sections,
                "paper_title": title,
                "paper_abstract": abstract,
                "paper_tex": self.ctx.rel(main_tex),
                "citations_used": sorted(set(citations_used)),
                "warnings": list(state.get("warnings") or []) + warnings,
            },
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------ #
    # 迭代落地（评审循环回环时走这条路径）
    # ------------------------------------------------------------------ #
    def _apply_iteration(
        self, state: dict[str, Any], existing: dict[str, Any], warnings: list[str]
    ) -> StageResult:
        """把 s8 修订后的章节规范化到磁盘，**不重新生成**。

        两种情况都要处理，且都不能把已有论文搞坏：

        * **修订产出了新章节** → 重新写 ``sections/*.tex`` 并重新装配 ``main.tex``；
        * **修订没产出章节**（LLM 只给了 revisions 列表、或干脆不可用）
          → 保留原稿，只重写与补全文件，然后**如实上报「本轮无实质修改」**，
          让评审循环据此判定收益递减并收敛，而不是靠重生成来假装做了工作。
        """
        paper_dir = self.ctx.path("paper")
        paper_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[Artifact] = []
        round_no = int(state.get("review_round") or 0)

        revisions = [r for r in (state.get("revision_history") or []) if isinstance(r, dict)]
        latest = revisions[-1] if revisions else {}
        touched = sorted(
            str(k).lower() for k in ((latest.get("sections") or {}) if isinstance(latest, dict) else {})
        )
        unaddressed = coerce_list(latest.get("unaddressed")) if isinstance(latest, dict) else []

        bib_keys = self._bib_keys(state)
        citations: list[str] = []
        for key, filename, display, _words in SECTION_PLAN:
            body = str(existing.get(key) or "")
            if not body:
                continue
            body = self._sanitize_citations(body, bib_keys, warnings, key)
            existing[key] = body
            citations.extend(_cites_in(body))
            written = self._write_section_file(paper_dir, filename, display, body, state)
            artifacts.append(self.ctx.artifact(written, kind="tex", stage=self.name))

        abstract = str(state.get("paper_abstract") or existing.get("abstract") or "")
        if abstract:
            written = self._write_section_file(
                paper_dir, "abstract.tex", "Abstract", abstract, state
            )
            artifacts.append(self.ctx.artifact(written, kind="tex", stage=self.name))

        title = title_from_text(
            str(state.get("paper_title") or ""),
            fallback=str((state.get("selected_idea") or {}).get("title") or "Untitled"),
        )
        # 关键词从首轮落盘的元数据里读回，而不是重新兜底——否则每次评审回环
        # 都会把 LLM 生成的关键词换成硬编码串。
        prior_meta = self.ctx.load_json("paper/paper_meta.json", default=None)
        keywords = []
        if isinstance(prior_meta, dict):
            keywords = [
                str(k) for k in coerce_list(prior_meta.get("keywords")) if str(k).strip()
            ]
        if not keywords:
            keywords = _fallback_keywords(state, self.state_dict(state, "experiment_plan"))
            warnings.append(
                "iteration: keywords not found in paper_meta.json; used deterministic fallback"
            )
        main_tex = self._assemble_main(paper_dir, title, abstract, warnings, keywords)
        artifacts.append(self.ctx.artifact(main_tex, kind="tex", stage=self.name))
        bib_path = self._write_bib(paper_dir, state, citations, warnings)
        if bib_path is not None:
            artifacts.append(self.ctx.artifact(bib_path, kind="bib", stage=self.name))

        state["paper_sections"] = existing
        detail = (
            f"iteration {round_no}: applied reviewer revisions "
            f"({len(touched)} section(s): {', '.join(touched) or 'none'}), "
            f"{len(unaddressed)} unaddressed item(s)"
        )
        if not touched:
            warnings.append(
                f"iteration {round_no}: reviewer produced no section edits; "
                "manuscript left unchanged (the review loop will see no improvement)"
            )
            self.ctx.log_event(
                "writing_iteration_noop", stage=self.name, round=round_no,
                unaddressed=len(unaddressed),
            )

        artifacts.append(
            self.ctx.save_json(
                "paper/paper_meta.json",
                {
                    "title": title,
                    "abstract": abstract,
                    "sections": list(existing),
                    "citations_used": sorted(set(citations)),
                    "iteration_round": round_no,
                    "sections_touched": touched,
                    "unaddressed": unaddressed,
                    "keywords": keywords,
                },
                stage=self.name,
            )
        )
        return StageResult.success(
            detail=detail,
            updates={
                "paper_sections": existing,
                "paper_title": title,
                "paper_abstract": abstract,
                "paper_tex": self.ctx.rel(main_tex),
                "citations_used": sorted(set(citations)),
                "warnings": list(state.get("warnings") or []) + warnings,
            },
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------ #
    # 单章生成
    # ------------------------------------------------------------------ #
    def _write_section(
        self,
        key: str,
        display: str,
        words: int,
        state: dict[str, Any],
        evidence: str,
        bib_keys: set[str],
        warnings: list[str],
    ) -> tuple[str, list[str]]:
        plan = self.state_dict(state, "experiment_plan")
        result = self.llm_json(
            "s6_section",
            default=None,
            schema_hint=_SECTION_SCHEMA,
            section_name=display,
            section_instructions=_SECTION_INSTRUCTIONS.get(key, ""),
            outline_block=_outline_block(state, plan),
            evidence_block=evidence,
            bib_keys_block=", ".join(sorted(bib_keys)) or "（无可用引用，请不要使用 \\cite）",
            venue=str(getattr(self.ctx.cfg, "venue", "NeurIPS")),
            word_target=words,
        )
        if not isinstance(result, dict):
            warnings.append(f"section '{key}': LLM unavailable")
            return "", []
        body = _extract_section_body(result)
        if not body:
            warnings.append(f"section '{key}': LLM returned no usable body text")
            return "", []
        used = [str(k).strip() for k in coerce_list(result.get("citations_used")) if str(k).strip()]
        # 正文里出现的 cite 也要记账（模型未必在 citations_used 里报全）
        used.extend(_cites_in(body))
        return body, used

    def _write_abstract(
        self, state: dict[str, Any], sections: dict[str, str], warnings: list[str]
    ) -> tuple[str, str, list[str]]:
        """返回 ``(标题, 摘要, 关键词)``。

        关键词是模板需要的第五个占位符（``__KEYWORDS__``）。以前这里只返回两项，
        于是 ``\\textbf{Keywords:} __KEYWORDS__`` 会原样进入 PDF——不报错、不被
        s7 的编译修复闭环发现，只是安静地把编译期 token 印在论文上。
        """
        plan = self.state_dict(state, "experiment_plan")
        idea = self.selected_idea(state)
        contributions = _contributions(state, plan, sections)
        results_block = _results_digest(state)
        result = self.llm_json(
            "s6_abstract",
            default=None,
            schema_hint=_ABSTRACT_SCHEMA,
            title_candidates_block="\n".join(
                f"- {t}" for t in [idea.get("title", ""), plan.get("core_claim", "")] if t
            )
            or "（无候选标题）",
            contributions_block="\n".join(f"- {c}" for c in contributions) or "（无）",
            results_block=results_block,
            venue=str(getattr(self.ctx.cfg, "venue", "NeurIPS")),
        )
        fallback_title = clean_text(
            str(idea.get("title") or plan.get("objective") or "Untitled")
        )
        if isinstance(result, dict) and str(result.get("abstract") or "").strip():
            raw_title = title_from_text(str(result.get("title") or ""), fallback=fallback_title)
            title = self._sanitize_title(
                raw_title, fallback_title, str(getattr(self.ctx.cfg, "venue", "")), warnings
            )
            abstract = strip_directive_echo(_extract_latex(str(result.get("abstract") or "")))
            keywords = [
                clean_text(str(k))
                for k in coerce_list(result.get("keywords"))
                if str(k).strip()
            ][:8]
            return title, abstract, keywords

        warnings.append("abstract: LLM unavailable; assembled deterministically")
        title = title_from_text(fallback_title)
        abstract = _fallback_abstract(state, plan, contributions, results_block)
        return title, abstract, _fallback_keywords(state, plan)

    def _sanitize_title(
        self, title: str, fallback: str, venue: str, warnings: list[str]
    ) -> str:
        """拦掉把「目标会议名」之类配置写进标题的模板噪声。

        ``"Controlled Budget Comparison for NeurIPS"`` 这种标题是小模型把提示词里的
        ``Venue: NeurIPS`` 当成研究对象抄进标题的典型症状。标题里出现会议名几乎
        必然是噪声（除非研究本身确实在做会议相关研究），因此直接退回 fallback。
        """
        cleaned = clean_text(title).strip()
        if not cleaned or len(cleaned) < 8:
            return title_from_text(fallback)

        venue_norm = (venue or "").strip()
        if venue_norm and venue_norm.lower() in cleaned.lower():
            # 仅当 fallback 里没有该会议名时才认为是噪声
            if venue_norm.lower() not in (fallback or "").lower():
                warnings.append(
                    f"title looked like template noise (contained venue '{venue_norm}'); "
                    f"replaced with the selected idea's title"
                )
                self.ctx.log_event(
                    "title_sanitized", stage=self.name, rejected=cleaned[:120], venue=venue_norm
                )
                return title_from_text(fallback)
        return cleaned

    # ------------------------------------------------------------------ #
    # 引用与参考文献
    # ------------------------------------------------------------------ #
    def _bib_keys(self, state: dict[str, Any]) -> set[str]:
        keys: set[str] = set()
        for entry in state.get("bib_entries") or []:
            if isinstance(entry, dict):
                key = str(entry.get("bibkey") or "").strip()
                if key:
                    keys.add(key)
        # 模板自带的示例条目也应视为合法（保证骨架本身能编译）
        template_bib = self._template_bib_keys()
        return keys | template_bib

    def _template_bib_keys(self) -> set[str]:
        try:
            from ..config import PROJECT_ROOT

            path = Path(PROJECT_ROOT) / "templates" / "paper" / "references.bib"
            if not path.exists():
                return set()
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return set()
        return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))

    def _sanitize_citations(
        self, body: str, bib_keys: set[str], warnings: list[str], section: str
    ) -> str:
        """把引用了不存在 key 的 ``\\cite`` 摘掉，而不是留一个会编译失败的命令。

        LaTeX 对未知 citation 只警告不报错，但一个指向空条目的引用会生成 ``[?]``，
        在评审阶段是硬伤。这里直接删除未知 key；若整条 cite 都无效，连命令一起删。
        """
        dropped: list[str] = []

        def repl(match: re.Match[str]) -> str:
            raw = match.group(0)
            keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
            keep = [k for k in keys if k in bib_keys]
            dropped.extend([k for k in keys if k not in bib_keys])
            if not keep:
                return ""
            return f"\\cite{{{', '.join(keep)}}}"

        cleaned = _CITE_RE.sub(repl, body)
        if dropped:
            self.ctx.log_event(
                "citations_dropped", stage=self.name, section=section,
                dropped=sorted(set(dropped)),
            )
            warnings.append(
                f"section '{section}': dropped {len(set(dropped))} citation key(s) "
                f"not present in the bibliography"
            )
        return cleaned

    def _write_bib(
        self,
        paper_dir: Path,
        state: dict[str, Any],
        citations_used: list[str],
        warnings: list[str],
    ) -> Path | None:
        """合并模板示例库与管线检索到的条目，产出最终 ``references.bib``。"""
        entries: dict[str, str] = {}

        try:
            from ..config import PROJECT_ROOT

            template_path = Path(PROJECT_ROOT) / "templates" / "paper" / "references.bib"
            if template_path.exists():
                text = template_path.read_text(encoding="utf-8", errors="replace")
                for key, entry in _split_bib_entries(text).items():
                    entries[key] = entry
        except Exception as exc:
            warnings.append(f"could not load template bibliography: {exc}")

        fetched = 0
        for entry in state.get("bib_entries") or []:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("bibkey") or "").strip()
            if not key or key in entries:
                continue
            entries[key] = _make_bib_entry(entry)
            fetched += 1

        cited = set(citations_used)
        # 策略：保留模板条目（保证骨架可编译），再加上所有被引用的检索条目
        keep = {k: v for k, v in entries.items() if k in cited or k in self._template_bib_keys()}
        if not keep:
            keep = entries

        try:
            paper_dir.mkdir(parents=True, exist_ok=True)
            target = paper_dir / "references.bib"
            header = (
                "% Auto-generated by the Auto-Research pipeline. Do not edit by hand;\n"
                "% regenerate via `python -m autoresearch.cli run`.\n"
                f"% entries: {len(keep)} (retrieved: {fetched}, cited: {len(cited)})\n\n"
            )
            target.write_text(header + "\n\n".join(keep.values()) + "\n", encoding="utf-8")
            return target
        except OSError as exc:
            warnings.append(f"could not write references.bib: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # 模板装配
    # ------------------------------------------------------------------ #
    def _write_section_file(
        self, paper_dir: Path, filename: str, display: str, body: str, state: dict[str, Any]
    ) -> Path:
        """写入 ``sections/<file>``。

        骨架文件里已有 ``\\section{...}``，但 chapter 体可能自带标题（LLM 常见），
        所以这里先剥掉重复的 ``\\section``，再统一补一个，保证不出现双标题。
        """
        section_dir = paper_dir / "sections"
        section_dir.mkdir(parents=True, exist_ok=True)
        stripped = re.sub(r"\\section\*?\{[^}]*\}", "", body or "", count=1).strip()
        label = _slug(filename)
        text = f"\\section{{{display}}}\n\\label{{sec:{label}}}\n\n{stripped}\n"
        path = section_dir / filename
        path.write_text(text, encoding="utf-8")
        return path

    def _assemble_main(
        self,
        paper_dir: Path,
        title: str,
        abstract: str,
        warnings: list[str],
        keywords: list[str] | None = None,
    ) -> Path:
        """从模板生成 ``main.tex``，替换**全部**占位符并写回。

        模板声明了五个占位符；早期实现只替换三个，于是 ``\\date{__DATE__}`` 与
        ``\\textbf{Keywords:} __KEYWORDS__`` 会**原样出现在编译后的论文里**。
        这类残留不会让编译失败，所以不会被 s7 的错误修复闭环发现——它只是安静地
        把编译期 token 印到 PDF 上。因此这里按模板声明的集合逐个替换，而不是
        只处理"我记得的那几个"。
        """
        template_text = ""
        try:
            from ..config import PROJECT_ROOT

            template = Path(PROJECT_ROOT) / "templates" / "paper" / "main.tex"
            if template.exists():
                template_text = template.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            warnings.append(f"could not read paper template: {exc}")

        if not template_text.strip():
            template_text = _FALLBACK_MAIN_TEX
            warnings.append("paper template missing; wrote the built-in minimal main.tex")

        authors = "Auto-Research Pipeline"
        keyword_text = ", ".join(str(k) for k in (keywords or []) if str(k).strip())
        if not keyword_text:
            keyword_text = "automated research, reproducibility, controlled comparison"
        date_text = time.strftime("%Y-%m-%d")

        rendered = (
            template_text.replace("__TITLE__", _latex_escape(title))
            .replace("__AUTHORS__", _latex_escape(authors))
            .replace("__ABSTRACT__", abstract.strip())
            .replace("__KEYWORDS__", _latex_escape(keyword_text))
            .replace("__DATE__", date_text)
        )
        path = paper_dir / "main.tex"
        path.write_text(rendered, encoding="utf-8")

        # 自检：模板里不该再有任何 __XXX__ 形式的上游 token 残留。
        # 这一步把"编译后才发现"降级为"当场就知道"。
        leftover = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]{2,}__", _strip_comments(rendered))))
        if leftover:
            warnings.append(
                f"main.tex still contains unfilled placeholders after assembly: {leftover}"
            )
            self.ctx.log_event("template_placeholders_unfilled", stage=self.name,
                               tokens=leftover)

        missing_inputs = [
            inc for inc in re.findall(r"\\input\{([^}]+)\}", rendered)
            if not (paper_dir / (inc if inc.endswith(".tex") else inc + ".tex")).exists()
        ]
        if missing_inputs:
            warnings.append(f"main.tex references missing inputs: {missing_inputs}")
        self.ctx.log_event(
            "paper_assembled", stage=self.name, inputs=len(
                re.findall(r"\\input\{([^}]+)\}", rendered)
            ),
            missing=missing_inputs,
            placeholders_left=leftover,
        )
        return path

    # ------------------------------------------------------------------ #
    def _evidence_block(self, state: dict[str, Any]) -> str:
        """给写作阶段的证据块：指标 + 对照 + 图 + 表 + 文献脉络。"""
        parts: list[str] = []

        parts.append("### 指标（唯一数字来源，逐字引用）")
        try:
            from .s5_analysis import _format_summary_block

            parts.append(_format_summary_block(self.state_dict(state, "metrics_summary")))
        except Exception:
            parts.append(clamp(str(self.state_dict(state, "metrics_summary")), 4000))

        # 对照信息优先用统一解析器（它知道多个来源），退回 ④ 的原始对照。
        # 标题与两侧标签都用真实名字：多臂运行下 treatment 可能是消融格点，
        # 写死 "method" 会让模型把消融结果写成"与基准方法对比"。
        try:
            from .s9_finalize import _resolve_comparison

            comparison = _resolve_comparison(state)
        except Exception:
            exp_results = self.state_dict(state, "experiment/results")
            comparison = exp_results.get("comparison") if isinstance(exp_results, dict) else None
        if isinstance(comparison, dict) and comparison.get("available"):
            base_label = str(comparison.get("baseline") or "baseline")
            treat_label = str(comparison.get("treatment") or "treatment")
            parts.append(f"\n### {base_label} vs {treat_label} 对照")
            parts.append(str(comparison.get("summary_line") or ""))
            for name, v in (comparison.get("per_metric") or {}).items():
                parts.append(
                    f"- {name}: {base_label}={v.get('baseline_best')}, "
                    f"{treat_label}={v.get('method_best')}, delta={v.get('delta')}, "
                    f"direction={v.get('direction')}"
                )

        figured = self.state_dict(state, "figured")
        if figured:
            parts.append("\n### 图（引用时使用这些 label）")
            for name, paths in figured.items():
                parts.append(f"- fig:{_slug(name)} → {', '.join(str(p) for p in paths)}")

        tables = self.state_dict(state, "latex_tables")
        if tables:
            parts.append("\n### 表（label 已在 LaTeX 中给出）")
            for name in tables:
                parts.append(f"- tab:{_slug(name)}")

        analysis = self.state_dict(state, "analysis")
        claim_evidence = coerce_list(analysis.get("claim_evidence"))
        if claim_evidence:
            parts.append("\n### 证据分级（严格遵守：不可把 partially/not supported 写成 supported）")
            for c in claim_evidence:
                if isinstance(c, dict):
                    parts.append(
                        f"- [{c.get('verdict')}] {c.get('claim')} — {c.get('evidence')}"
                    )
        limitations = coerce_list(analysis.get("limitations"))
        if limitations:
            parts.append("\n### 局限（写 limitations 章节时直接采用）")
            parts.extend(f"- {x}" for x in limitations)
        threats = coerce_list(analysis.get("threats_to_validity"))
        if threats:
            parts.append("\n### 效度威胁")
            parts.extend(f"- {x}" for x in threats)

        lit = str(state.get("lit_review") or "")
        if lit:
            parts.append("\n### 文献综述要点（related work 的素材）")
            parts.append(clamp(lit, 4000))

        return clamp("\n".join(parts), 20000)


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def _cites_in(body: str) -> list[str]:
    keys: list[str] = []
    for match in _CITE_RE.finditer(body or ""):
        keys.extend(k.strip() for k in match.group(1).split(",") if k.strip())
    return keys


def _extract_latex(text: str) -> str:
    """剥掉 LLM 可能包上的 ```latex / ```tex 围栏。"""
    if not text:
        return ""
    m = re.match(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", text.strip(), re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


#: LLM 对「章节正文」这个字段的命名并不统一。全部接受，按可靠度排序。
_SECTION_BODY_KEYS = (
    "latex", "section", "body", "content", "text", "markdown",
    "section_latex", "section_body", "tex",
)


def _extract_section_body(payload: dict[str, Any]) -> str:
    """从 LLM 返回的任意合理结构里取出章节正文。

    只认 ``latex`` 一个键是脆的：模型换个字段名就让整章退化成占位符，
    而失败信号（占位文本）看起来又很像「模型没干活」，很容易被误判成环境问题。
    这里按优先级尝试多个键名，最后再退化到「整个 payload 里最长的字符串」。
    """
    for key in _SECTION_BODY_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _extract_latex(value)
    # 退化：取所有字符串里最长的那个（正文必然是最大的文本块）
    candidates = [
        v for v in payload.values() if isinstance(v, str) and len(v.strip()) > 80
    ]
    if candidates:
        return _extract_latex(max(candidates, key=len))
    return ""


def _strip_comments(tex: str) -> str:
    """去掉 LaTeX 注释行，避免把模板注释里的占位符说明误判为未替换的 token。"""
    return "\n".join(
        line for line in (tex or "").splitlines() if not line.lstrip().startswith("%")
    )


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "x"


def word_count(text: str) -> int:
    """混排文本的词数：拉丁词按空格切，CJK 按字计。

    直接用 ``len(text.split())`` 会把一整段中文算成 1 个词，
    让「章节是否充实」这类判据完全失效——评审阶段会用这个数字。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff")
    latin = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-_.]*", text))
    return int(cjk + latin)


def _latex_escape(text: str) -> str:
    """只转义会破坏标题/作者行的字符——不碰数学模式（标题里不该有公式）。"""
    out = text or ""
    for char in ("&", "%", "$", "#"):
        out = out.replace(char, "\\" + char)
    out = out.replace("_", "\\_")
    return out


def _outline_block(state: dict[str, Any], plan: dict[str, Any]) -> str:
    idea = state.get("selected_idea") or {}
    lines = [
        f"research_direction: {state.get('direction', '')}",
        f"selected_idea_title: {idea.get('title', '')}",
        f"hypothesis: {idea.get('hypothesis', '')}",
        f"method_sketch: {idea.get('method_sketch', '')}",
        f"objective: {plan.get('objective', '')}",
        f"core_claim: {plan.get('core_claim', '')}",
    ]
    for m in plan.get("milestones") or []:
        lines.append(f"milestone {m.get('id')}: {m.get('name')} — {m.get('success_criterion')}")
    return "\n".join(lines)


def _contributions(state: dict[str, Any], plan: dict[str, Any], sections: dict[str, str]) -> list[str]:
    """从已写好的章节里回收贡献点——比让 LLM 重新生成更忠实。"""
    out: list[str] = []
    intro = sections.get("introduction", "")
    for line in re.findall(r"\\item\s+(.+)", intro)[:5]:
        clean = clean_text(re.sub(r"\\[a-zA-Z]+\{?|[{}]", "", line))
        if 10 < len(clean) < 300:
            out.append(clean)
    if not out and plan.get("core_claim"):
        out.append(clean_text(str(plan["core_claim"])))
    return out[:5]


def _results_digest(state: dict[str, Any]) -> str:
    """给摘要用的结果摘要。

    ``metrics_summary`` 的跨种子口径优先于 ④ 的裸指标对照——多种子运行时后者
    可能因为指标名带 ``@seed=N`` 后缀而匹配不上，误报「无结果」。
    """
    try:
        from .s9_finalize import _resolve_comparison

        comparison = _resolve_comparison(state)
        if comparison.get("available"):
            basis = comparison.get("basis") or "metrics"
            return clamp(f"{comparison.get('summary_line')}（口径：{basis}）", 1200)
    except Exception:
        pass
    summary = state.get("metrics_summary") or {}
    return clamp(str(summary), 1200) or "（无可用结果）"


def _fallback_keywords(state: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """LLM 不可用时从计划里拼一组关键词，避免模板占位符留空。"""
    out: list[str] = []
    for metric in (plan.get("metrics") or [])[:3]:
        if isinstance(metric, dict) and metric.get("name"):
            out.append(str(metric["name"]))
    idea = state.get("selected_idea") or {}
    if idea.get("title"):
        out.append(clean_text(str(idea["title"]))[:40])
    out.append("controlled comparison")
    out.append("reproducibility")
    seen: list[str] = []
    for item in out:
        if item and item not in seen:
            seen.append(item)
    return seen[:6]


def _fallback_abstract(
    state: dict[str, Any], plan: dict[str, Any], contributions: list[str], results_block: str
) -> str:
    direction = state.get("direction", "")
    lines = [
        f"本文研究{direction}。",
        f"我们提出并检验了如下假设：{plan.get('core_claim') or plan.get('objective') or '—'}。",
    ]
    if contributions:
        lines.append("主要工作包括：" + "；".join(contributions[:3]) + "。")
    if results_block:
        lines.append(f"实验结果显示：{results_block}。")
    lines.append(
        "本文同时报告了实验设置、定量对照与局限。所有结果均由管线内可复现的"
        "受控实验产生，指标原始文件随论文一并交付。"
    )
    return "\n\n".join(lines)


def _placeholder_section(display: str, key: str, state: dict[str, Any]) -> str:
    """LLM 不可用时的占位正文：诚实标注，不伪造内容。"""
    return (
        f"本章节由自动管线生成，但因语言模型不可用而未能完成实质撰写。\n\n"
        f"可用的确定性材料：研究大方向为「{state.get('direction', '')}」；"
        f"实验计划目标为「{(state.get('experiment_plan') or {}).get('objective', '—')}」。"
        f"完整的实验记录、指标与图表见随附的交付目录。\n\n"
        f"\\textbf{{标注}}：本章节内容不完整，不应作为最终投稿版本。"
    )


def _split_bib_entries(text: str) -> dict[str, str]:
    """把 .bib 文本切成 ``{key: entry_text}``。用花括号配平而不是正则——BibTeX 里
    字段值可以嵌套花括号，正则切不干净。"""
    entries: dict[str, str] = {}
    i = 0
    n = len(text)
    while i < n:
        at = text.find("@", i)
        if at < 0:
            break
        brace = text.find("{", at)
        if brace < 0:
            break
        depth = 0
        j = brace
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            break
        block = text[at : j + 1]
        comma = block.find(",")
        key = block[brace - at + 1 : comma].strip() if comma > 0 else ""
        if key:
            entries[key] = block.strip()
        i = j + 1
    return entries


def _bib_escape(value: str) -> str:
    out = str(value or "")
    for char in ("&", "%", "$", "#"):
        out = out.replace(char, "\\" + char)
    return out.replace("_", "\\_").replace("~", "\\textasciitilde{}")


def _make_bib_entry(entry: dict[str, Any]) -> str:
    key = str(entry.get("bibkey") or "anon")
    authors = entry.get("authors") or []
    author_field = " and ".join(str(a) for a in authors) or "Unknown"
    year = entry.get("year") or "n.d."
    venue = str(entry.get("venue") or "").strip()
    url = str(entry.get("url") or "").strip()
    kind = "article"
    source = str(entry.get("source") or "")
    fields = [
        f"  title = {{{_bib_escape(str(entry.get('title') or 'Untitled'))}}}",
        f"  author = {{{_bib_escape(author_field)}}}",
        f"  year = {{{year}}}",
    ]
    if "arxiv" in source.lower() or "arxiv" in url.lower():
        kind = "misc"
        eprint = url.rstrip("/").split("/")[-1] if url else ""
        if eprint:
            fields.append(f"  eprint = {{{_bib_escape(eprint)}}}")
        fields.append("  archivePrefix = {arXiv}")
    elif venue:
        fields.append(f"  journal = {{{_bib_escape(venue)}}}")
    if url:
        fields.append(f"  url = {{{_bib_escape(url)}}}")
    if entry.get("citation_count"):
        fields.append(f"  note = {{cited by {int(entry['citation_count'])}}}")
    body = ",\n".join(fields)
    return f"@{kind}{{{key},\n{body}\n}}"


_FALLBACK_MAIN_TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\usepackage{natbib}
\bibliographystyle{plainnat}

\title{__TITLE__}
\author{__AUTHORS__}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
__ABSTRACT__
\end{abstract}

\input{sections/introduction}
\input{sections/related_work}
\input{sections/method}
\input{sections/experiments}
\input{sections/results}
\input{sections/discussion}
\input{sections/limitations}
\input{sections/conclusion}

\bibliography{references}
\end{document}
"""


__all__ = ["WritingStage", "SECTION_PLAN"]
