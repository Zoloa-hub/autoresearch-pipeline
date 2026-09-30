"""阶段⑨：交付打包与最终报告。

交付物分三类，缺一不可：

1. **论文**：``paper/main.tex`` + ``sections/`` + ``references.bib`` + PDF（若引擎可用）；
2. **证据**：原始指标文件、图、表、事件日志——让任何人能重算论文里的每个数字；
3. **诚实记录**：未解决的评审问题、保真性告警、失败阶段、编译阻断原因。

第 3 类最容易被「自动化」掉，但它恰恰是自动科研能不能被信任的关键。
本阶段把它作为一等交付物写进最终报告，并在 ``deliverables`` 里列明。
退出码语义在 runner 里决定：有 PDF 或完整论文 → 0；仅部分完成 → 1；全崩 → 2。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..graph.state import Artifact, STATUS_FAILED, STATUS_SKIPPED
from .base import Stage, StageResult, clamp, clean_text, coerce_list

_REPORT_SCHEMA = {"type": "object"}

#: 交付包布局：run_dir/06_deliverables/<子目录>/...
_BUNDLE = "06_deliverables"
_BUNDLE_MAP: tuple[tuple[str, str], ...] = (
    ("paper", "paper"),
    ("figures", "figures"),
    ("metrics", "metrics"),
    ("analysis", "analysis"),
    ("experiment", "experiment"),
    ("review", "review"),
    ("literature", "literature"),
    ("ideas", "ideas"),
    ("plan", "plan"),
)


class FinalizeStage(Stage):
    name = "s9_finalize"
    title = "交付打包与最终报告"
    requires = ()
    produces = ("final_report", "deliverables")
    max_attempts = 2

    def run(self, state: dict[str, Any]) -> StageResult:
        warnings: list[str] = []
        artifacts: list[Artifact] = []

        # --- 1. 抽取「未解决问题」的全量清单 --------------------------- #
        open_issues = _collect_open_issues(state)

        # --- 2. 打包交付目录（只复制，不移动） ------------------------- #
        bundle_dir = self.ctx.path(_BUNDLE)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_tree(self.ctx, bundle_dir)
        deliverables = sorted(copied)

        # --- 3. 渲染事实性报告（确定性） ------------------------------- #
        facts = _render_final_report(state, open_issues, deliverables, self.ctx)
        artifacts.append(
            self.ctx.save_text("report/FINAL_REPORT.md", facts, stage=self.name)
        )

        # --- 4. LLM 叙述性总结（可选，叠加而非替代） ------------------- #
        narrative = self._narrative(state, open_issues, deliverables)
        if narrative:
            artifacts.append(
                self.ctx.save_text("report/RUN_SUMMARY.md", narrative, stage=self.name)
            )

        # --- 5. MANIFEST + 交付清单 ------------------------------------ #
        manifest = self._manifest(state, copied)
        artifacts.append(
            self.ctx.save_text(
                "report/MANIFEST.md", _render_manifest(manifest, open_issues), stage=self.name
            )
        )
        artifacts.append(
            self.ctx.save_json("report/manifest.json", manifest, stage=self.name)
        )
        artifacts.append(
            self.ctx.save_json("report/open_issues.json", open_issues, stage=self.name)
        )

        # --- 6. 可复现性说明 ------------------------------------------- #
        artifacts.append(
            self.ctx.save_text(
                "report/REPRODUCE.md", _render_reproduce(state, self.ctx), stage=self.name
            )
        )

        detail = (
            f"{len(copied)} files bundled, {len(open_issues['major'])} major open issues, "
            f"pdf={'yes' if state.get('final_pdf') else 'no'}"
        )
        return StageResult.success(
            detail=detail,
            updates={
                "final_report": "report/FINAL_REPORT.md",
                "deliverables": deliverables,
                "open_issues": open_issues,
                "warnings": list(state.get("warnings") or []) + warnings,
            },
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------ #
    def _narrative(
        self, state: dict[str, Any], open_issues: dict[str, Any], deliverables: list[str]
    ) -> str:
        run_summary = self._run_summary_block(state, open_issues)
        text = self.llm_text(
            "s9_report",
            fallback="",
            run_summary_block=run_summary,
            artifacts_block=clamp("\n".join(deliverables), 6000),
            review_block=clamp(
                _render_review_digest(state.get("reviews") or []), 4000
            ),
        )
        return clamp(text, 40000) if text.strip() else ""

    def _run_summary_block(self, state: dict[str, Any], open_issues: dict[str, Any]) -> str:
        idea = state.get("selected_idea") or {}
        plan = state.get("experiment_plan") or {}
        lines = [
            f"研究大方向：{state.get('direction', '')}",
            f"选定假设：{idea.get('title', '—')}",
            f"假设内容：{idea.get('hypothesis', '—')}",
            f"新颖性判定：{(idea.get('novelty') or {}).get('verdict', 'unknown')}",
            f"核心 claim：{plan.get('core_claim', '—')}",
            f"检索文献数：{len(state.get('papers') or [])}",
            f"生成候选假设数：{len(state.get('ideas') or [])}",
            f"执行 run 数：{len(state.get('runs_executed') or [])}",
            f"调试轮数：{len(state.get('debug_history') or [])}",
            f"评审分数：{state.get('review_score')}/10（{state.get('review_verdict')}）",
            f"编译结果：{(state.get('compile_result') or {}).get('ok')}",
            f"major 未解决问题：{len(open_issues.get('major') or [])}",
        ]
        comparison = (state.get("experiment/results") or {}).get("comparison") \
            if isinstance(state.get("experiment/results"), dict) else None
        if isinstance(comparison, dict) and comparison.get("available"):
            lines.append(f"指标对照：{comparison.get('summary_line')}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def _manifest(self, state: dict[str, Any], copied: list[str]) -> dict[str, Any]:
        from ..config import PROJECT_ROOT  # 延迟导入，避免 graph 层反向依赖

        deliverables = []
        for rel in sorted(copied):
            path = self.ctx.run_dir / rel
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            deliverables.append({"path": rel, "bytes": size})
        return {
            "run_id": self.ctx.cfg.run_id,
            "direction": state.get("direction", ""),
            "venue": state.get("venue", ""),
            "pipeline_version": _version(),
            "created_at": state.get("started_at", ""),
            "project_root": str(PROJECT_ROOT),
            "run_dir": self.ctx.run_dir.as_posix(),
            "artifacts": state.get("artifacts") or [],
            "deliverables": deliverables,
            "final_pdf": state.get("final_pdf", ""),
            "stage_status": state.get("stage_status") or {},
            "review": {
                "score": state.get("review_score", 0.0),
                "verdict": state.get("review_verdict", ""),
                "rounds": len(state.get("reviews") or []),
            },
            "llm_usage": _usage_dict(self.ctx),
        }


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def _version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "0.1.0"


def _resolve_comparison(state: dict[str, Any]) -> dict[str, Any]:
    """取得 baseline vs method 的定量对照，兼容三种来源。

    优先级：
    1. ``experiment/results.comparison`` —— 阶段 ④ 从指标文件直接算的，最原始；
    2. ``metrics_summary.cross_seed`` —— 阶段 ⑤ 的**跨种子**口径（学术上最该用的）；
    3. ``metrics_summary.summary`` —— 逐 run 统计。

    为什么要三级：④ 的对照按「裸指标名」比较，而多种子运行下指标名带
    ``@seed=N`` 后缀，于是 ④ 可能给出「无可对照指标」；而 ⑤ 的跨种子统计恰恰是
    真正该进论文的数字。若不回退，最终报告会在一份其实有结果的运行里写
    「未获得可对照的定量结果」——这是会自动科研最尴尬的一类错误：
    **数据在，报告说没有**。
    """
    exp = state.get("experiment/results")
    if isinstance(exp, dict):
        comparison = exp.get("comparison")
        if isinstance(comparison, dict) and comparison.get("available"):
            return comparison

    summary = state.get("metrics_summary") or {}
    cross = summary.get("cross_seed") if isinstance(summary, dict) else None
    rows: dict[str, dict[str, float]] = {}
    if isinstance(cross, dict) and "baseline" in cross and "method" in cross:
        for variant in ("baseline", "method"):
            stats = cross.get(variant) or {}
            flat: dict[str, float] = {}
            for key, entry in stats.items():
                if key.startswith("_") or not isinstance(entry, dict):
                    continue
                metric = key[: -len("__best")] if key.endswith("__best") else key
                if metric.endswith("__final"):
                    continue
                flat[metric] = _num(entry.get("mean"))
            rows[variant] = flat
    if not rows:
        inner = summary.get("summary") if isinstance(summary, dict) else None
        if isinstance(inner, dict):
            for variant in ("baseline", "method"):
                stats = inner.get(variant)
                if not isinstance(stats, dict):
                    continue
                flat = {}
                for metric, entry in stats.items():
                    if isinstance(entry, dict):
                        flat[metric] = _num(entry.get("best", entry.get("mean")))
                    elif isinstance(entry, (int, float)):
                        flat[metric] = float(entry)
                rows[variant] = flat
    if not rows.get("baseline") or not rows.get("method"):
        return {"available": False,
                "summary_line": "缺少 baseline 或 method 指标，无法对照"}

    baseline, method = rows["baseline"], rows["method"]
    shared = sorted(set(baseline) & set(method))
    if not shared:
        return {"available": False,
                "summary_line": "baseline 与 method 没有共同的指标名，无法对照"}

    per_metric: dict[str, Any] = {}
    bits: list[str] = []
    improved: list[str] = []
    regressed: list[str] = []
    primary = ""
    best_rel = 0.0
    for name in shared:
        b, m = baseline[name], method[name]
        delta = m - b
        rel = (delta / abs(b)) if b else 0.0
        direction = "lower" if _lower_is_better(name) else "higher"
        per_metric[name] = {
            "baseline_best": round(b, 6),
            "method_best": round(m, 6),
            "delta": round(delta, 6),
            "relative": round(rel, 6),
            "direction": direction,
            "baseline_final": round(b, 6),
            "method_final": round(m, 6),
        }
        bits.append(f"{name} {b:.4g}→{m:.4g} ({rel:+.1%})")
        if abs(rel) > abs(best_rel):
            best_rel, primary = rel, name
        better = (delta > 0 and direction == "higher") or (delta < 0 and direction == "lower")
        worse = (delta < 0 and direction == "higher") or (delta > 0 and direction == "lower")
        if better:
            improved.append(name)
        elif worse:
            regressed.append(name)

    return {
        "available": True,
        # 显式记录对照双方。只给指标数字而不说「跟谁比」的对照是不可解读的——
        # 读者必须自己猜 method 到底指哪个变体。
        "baseline": "baseline",
        "treatment": "method" if "method" in rows else sorted(rows)[-1],
        "primary_metric": primary,
        "per_metric": per_metric,
        "improved": improved,
        "regressed": regressed,
        "supports_claim": bool(improved) and not regressed,
        "summary_line": "; ".join(bits[:4]),
        "basis": "cross-seed mean" if (isinstance(cross, dict) and cross) else "per-run best",
        "source": "s5 cross-seed summary" if (isinstance(cross, dict) and cross)
                  else "s5 per-run summary",
    }


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _lower_is_better(name: str) -> bool:
    """指标方向：转调共享实现（tools.metrics 是唯一规范表）。"""
    from .s4_experiment import _higher_is_better

    return not _higher_is_better(name)

def _usage_dict(ctx: Any) -> dict[str, Any]:
    try:
        usage = ctx.llm.usage()
        return {
            "calls": getattr(usage, "calls", 0),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    except Exception:  # pragma: no cover
        return {}


def _copy_tree(ctx: Any, bundle_dir: Path) -> list[str]:
    """把关键产物复制进交付目录。失败静默跳过——打包不该让整条管线失败。"""
    copied: list[str] = []
    for sub, dest_name in _BUNDLE_MAP:
        src = ctx.run_dir / sub
        if not src.is_dir():
            continue
        dest = bundle_dir / dest_name
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for item in src.rglob("*"):
                if not item.is_file():
                    continue
                rel = item.relative_to(src)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                copied.append(target.relative_to(ctx.run_dir).as_posix())
        except (OSError, shutil.Error):
            continue

    # 顶层的一份「掌上入口」：最终报告 + 编译阻断说明
    for rel in ("report/FINAL_REPORT.md", "report/COMPILE_BLOCKED.md", "report/REPRODUCE.md"):
        src = ctx.run_dir / rel
        if src.exists():
            try:
                target = bundle_dir / Path(rel).name
                shutil.copy2(src, target)
                copied.append(target.relative_to(ctx.run_dir).as_posix())
            except (OSError, shutil.Error):
                continue
    return copied


def _collect_open_issues(state: dict[str, Any]) -> dict[str, Any]:
    """汇总所有「没解决的问题」，按严重度分组。这是交付物的一部分，不是附录。"""
    major: list[dict[str, Any]] = []
    minor: list[dict[str, Any]] = []
    notes: list[str] = []

    reviews = [r for r in (state.get("reviews") or []) if isinstance(r, dict)]
    final = reviews[-1] if reviews else {}
    for w in coerce_list(final.get("weaknesses")):
        if not isinstance(w, dict):
            continue
        entry = {
            "source": "review",
            "point": clean_text(str(w.get("point") or "")),
            "evidence": clean_text(str(w.get("evidence") or "")),
            "min_fix": clean_text(str(w.get("min_fix") or "")),
            "location": clean_text(str(w.get("location") or "")),
        }
        (major if str(w.get("severity")) == "major" else minor).append(entry)

    for rev in state.get("revision_history") or []:
        if not isinstance(rev, dict):
            continue
        for u in coerce_list(rev.get("unaddressed")):
            if isinstance(u, dict):
                major.append(
                    {
                        "source": "revision",
                        "point": clean_text(str(u.get("issue") or "")),
                        "evidence": "撰写智能体声明无法用现有证据解决",
                        "min_fix": clean_text(str(u.get("reason") or "")),
                        "location": "—",
                    }
                )
            elif u:
                major.append({"source": "revision", "point": str(u), "evidence": "",
                              "min_fix": "", "location": "—"})

    analysis = state.get("analysis") or {}
    for lim in coerce_list(analysis.get("limitations")):
        minor.append(
            {
                "source": "analysis",
                "point": clean_text(str(lim)),
                "evidence": "",
                "min_fix": "已在 Limitations 中说明",
                "location": "sections/limitations.tex",
            }
        )
    for threat in coerce_list(analysis.get("threats_to_validity")):
        minor.append(
            {
                "source": "validity",
                "point": clean_text(str(threat)),
                "evidence": "",
                "min_fix": "需补充实验或限定结论范围",
                "location": "sections/limitations.tex",
            }
        )

    for d in state.get("debug_history") or []:
        if isinstance(d, dict) and d.get("validity_flag"):
            major.append(
                {
                    "source": "experiment-fidelity",
                    "point": f"{d.get('variant')} 第 {d.get('round')} 轮补丁可能降低实验保真度",
                    "evidence": str(d.get("validity_flag")),
                    "min_fix": "人工复核该轮补丁，必要时回滚并重跑",
                    "location": "experiment/debug_history.json",
                }
            )

    compile_result = state.get("compile_result") or {}
    if not compile_result.get("ok"):
        major.append(
            {
                "source": "build",
                "point": "论文未产出 PDF。",
                "evidence": "; ".join(coerce_list(compile_result.get("errors"))[:3])
                or f"engine={compile_result.get('engine')}",
                "min_fix": "安装 LaTeX 引擎后重新编译（见 report/COMPILE_BLOCKED.md）",
                "location": "paper/main.tex",
            }
        )

    failed = [s for s, v in (state.get("stage_status") or {}).items() if v == STATUS_FAILED]
    for stage in failed:
        errors = (state.get("stage_errors") or {}).get(stage) or []
        major.append(
            {
                "source": "pipeline",
                "point": f"阶段 `{stage}` 失败。",
                "evidence": clean_text(str(errors[-1]))[:300] if errors else "",
                "min_fix": "查看事件日志并进行人工干预后 resume",
                "location": f"events.jsonl（{stage}）",
            }
        )

    skipped = [s for s, v in (state.get("stage_status") or {}).items() if v == STATUS_SKIPPED]
    if skipped:
        notes.append("跳过的阶段：" + ", ".join(skipped))

    if not state.get("final_pdf"):
        notes.append("本次运行未产出 PDF。")

    warnings = state.get("warnings") or []
    if warnings:
        notes.append(f"运行期间共 {len(warnings)} 条告警，详见 FINAL_REPORT.md 附录。")

    return {
        "major": major,
        "minor": minor,
        "notes": notes,
        "counts": {"major": len(major), "minor": len(minor)},
    }


def _render_review_digest(reviews: list[dict[str, Any]]) -> str:
    if not reviews:
        return "（无评审记录）"
    lines = []
    for r in reviews:
        lines.append(
            f"round {r.get('round')}: score={r.get('score')} verdict={r.get('verdict')}"
        )
        for w in coerce_list(r.get("weaknesses"))[:6]:
            if isinstance(w, dict):
                lines.append(f"  - [{w.get('severity')}] {w.get('point')}")
    return "\n".join(lines)


def _render_final_report(
    state: dict[str, Any],
    open_issues: dict[str, Any],
    deliverables: list[str],
    ctx: Any,
) -> str:
    idea = state.get("selected_idea") or {}
    plan = state.get("experiment_plan") or {}
    comparison = _resolve_comparison(state)
    status = state.get("stage_status") or {}
    trace = state.get("trace") or []

    parts: list[str] = [
        "# Auto-Research 运行报告",
        "",
        f"- **运行 ID**：`{state.get('run_id', '')}`",
        f"- **研究大方向**：{state.get('direction', '')}",
        f"- **目标会议**：{state.get('venue', '')}",
        f"- **论文标题**：{state.get('paper_title') or '—'}",
        f"- **最终 PDF**：{state.get('final_pdf') or '未产出（见 report/COMPILE_BLOCKED.md）'}",
        f"- **评审**：{state.get('review_score', 0)}/10 `{state.get('review_verdict', '')}`"
        f"（{len(state.get('reviews') or [])} 轮）",
        f"- **运行目录**：`{ctx.run_dir.as_posix()}`",
        "",
        "## 1. 阶段执行情况",
        "",
        "| 阶段 | 状态 | 尝试 | 耗时(s) | 说明 |",
        "|---|---|---|---|---|",
    ]
    by_stage = {t.get("stage"): t for t in trace if isinstance(t, dict)}
    seen_stages: set[str] = set()
    for stage, st in status.items():
        rec = by_stage.get(stage) or {}
        # 本阶段正在运行（报告由它自己写出），把自己标成 done 而不是 running——
        # 一份写在磁盘上、说"当前阶段正在运行"的报告会让人以为运行卡住了。
        shown = "done" if (stage == "s9_finalize" and st == "running") else st
        # 被评审循环重复执行的阶段要标出次数，否则读者会以为它只跑了一次，
        # 误判「迭代没生效」。
        runs = sum(1 for t in trace if isinstance(t, dict) and t.get("stage") == stage)
        times = f" ×{runs}" if runs > 1 else ""
        seen_stages.add(stage)
        parts.append(
            f"| {stage}{times} | `{shown}` | {rec.get('attempt', '—')} | "
            f"{rec.get('elapsed', '—')} | {clean_text(str(rec.get('detail') or ''))[:80]} |"
        )
    parts.append("")
    # 循环会让总步数超过阶段数，这里把两个数都写清楚。
    iterations = len(trace) - len(seen_stages)
    parts.append(
        f"> 图执行共 {len(trace)} 步，覆盖 {len(seen_stages)} 个阶段"
        + (f"（评审迭代额外执行了 {iterations} 步）" if iterations > 0 else "")
        + "。"
    )
    parts.append("")

    parts += [
        "## 2. 科研内容摘要",
        "",
        f"### 2.1 选定假设（{idea.get('id', '—')}）",
        "",
        f"**{idea.get('title', '—')}**",
        "",
        f"- 假设：{idea.get('hypothesis', '—')}",
        f"- 新颖性：`{(idea.get('novelty') or {}).get('verdict', 'unknown')}`"
        f"（score={(idea.get('novelty') or {}).get('score', '—')}）"
        f" — {(idea.get('novelty') or {}).get('rationale', '')}",
        f"- 排序分：{idea.get('rank', '—')}",
        "",
        "### 2.2 文献基础",
        "",
        f"- 检索式 {len(state.get('queries') or [])} 条，命中文献 {len(state.get('papers') or [])} 篇",
        f"- 识别研究缺口 {len(state.get('gaps') or [])} 个",
        f"- 综述文档：`literature/review.md`",
        "",
        "### 2.3 实验与结果",
        "",
        f"- 计划：{len(plan.get('milestones') or [])} 个里程碑，"
        f"{plan.get('total_runs', 0)} 次 run，预算 {plan.get('compute_budget_hours', '—')}h",
        f"- 实际执行：{len(state.get('runs_executed') or [])} 次，"
        f"自纠错 {len(state.get('debug_history') or [])} 轮",
    ]

    # 实验后端的自我声明必须出现在**最顶层报告**里，而不只是躺在
    # experiment/EXPERIMENT_RESULTS.md 里。理由：读者看 FINAL_REPORT 第 2 节
    # 时正是要判断"这组数字能支撑什么结论"，而合成数据与真实基准的差别
    # 恰恰决定了这个判断。让它只出现在嵌套文件里，等于把最重要的边界信息藏起来。
    adapter = state.get("adapter") or {}
    if isinstance(adapter, dict) and adapter.get("name"):
        parts += [
            "",
            f"- 实验后端：`{adapter.get('name')}`"
            f"（自带代码：{'是' if adapter.get('owns_code') else '否'}）",
        ]
        if adapter.get("description"):
            parts.append(f"- 后端说明：{clean_text(str(adapter['description']))[:200]}")
        if adapter.get("quality_note"):
            parts += [
                "",
                f"> **结果适用范围**：{clean_text(str(adapter['quality_note']))}",
            ]

    if isinstance(comparison, dict) and comparison.get("available"):
        base_label = str(comparison.get("baseline") or "baseline")
        treat_label = str(comparison.get("treatment") or "method")
        parts += [
            "",
            f"对照：`{base_label}`（基准） vs `{treat_label}`（治疗组）"
            + (f"，口径：{comparison['basis']}" if comparison.get("basis") else ""),
            "",
            f"| 指标 | {base_label}(best) | {treat_label}(best) | Δ | 相对 |",
            "|---|---|---|---|---|",
        ]
        for name, v in comparison["per_metric"].items():
            parts.append(
                f"| {name} | {v['baseline_best']:.6g} | {v['method_best']:.6g} | "
                f"{v['delta']:+.6g} | {v['relative']:+.2%} |"
            )
        parts += [
            "",
            f"**支持核心 claim**：{'是' if comparison.get('supports_claim') else '否'}",
        ]
    else:
        parts += ["", "_未获得可对照的定量结果。_"]

    parts += ["", "## 3. 未解决问题（诚实清单）", ""]
    major = open_issues.get("major") or []
    minor = open_issues.get("minor") or []
    if major:
        parts += ["### 3.1 严重（major）", ""]
        for i, issue in enumerate(major, 1):
            parts.append(
                f"{i}. **{issue['point']}**（来源：{issue['source']}）  \n"
                f"   - 证据：{issue['evidence'] or '—'}  \n"
                f"   - 建议处置：{issue['min_fix'] or '—'}  \n"
                f"   - 位置：`{issue['location']}`"
            )
        parts.append("")
    else:
        parts += ["_无 major 级未解决问题。_", ""]

    if minor:
        parts += ["### 3.2 次要（minor）", ""]
        for issue in minor[:20]:
            parts.append(f"- {issue['point']}")
        parts.append("")

    if open_issues.get("notes"):
        parts += ["### 3.3 备注", ""] + [f"- {n}" for n in open_issues["notes"]] + [""]

    parts += ["", "## 4. 交付物", ""]
    grouped: dict[str, list[str]] = {}
    for rel in deliverables:
        parts_of = Path(rel).parts
        key = parts_of[1] if len(parts_of) > 1 and parts_of[0] == _BUNDLE else parts_of[0]
        grouped.setdefault(key, []).append(rel)
    for key in sorted(grouped):
        parts.append(f"### {key}")
        parts.append("")
        for rel in grouped[key][:40]:
            parts.append(f"- `{rel}`")
        if len(grouped[key]) > 40:
            parts.append(f"- … 另有 {len(grouped[key]) - 40} 个文件")
        parts.append("")

    usage = _usage_dict(ctx)
    parts += [
        "## 5. 成本与资源",
        "",
        f"- LLM 调用 {usage.get('calls', 0)} 次，"
        f"总计 {usage.get('total_tokens', 0)} tokens",
        f"- 编译引擎：{(state.get('compile_result') or {}).get('engine', '—')}",
        f"- 沙箱后端：{getattr(getattr(ctx, 'sandbox', None), 'name', '—')}",
        "",
    ]

    warnings = state.get("warnings") or []
    errors = state.get("errors") or []
    if warnings or errors:
        parts += ["## 6. 告警与错误", ""]
        if errors:
            parts += ["### 错误", ""] + [f"- {clean_text(str(e))[:300]}" for e in errors[:30]] + [""]
        if warnings:
            parts += ["### 告警", ""] + [f"- {clean_text(str(w))[:300]}" for w in warnings[:40]] + [""]

    parts += [
        "---",
        "",
        "> 本报告由自动管线生成。第 3 节的未解决问题清单是**必读项**：",
        "> 任何引用本论文结论的下游使用，都应先核对该清单。",
        "",
    ]
    return "\n".join(parts)


def _render_manifest(manifest: dict[str, Any], open_issues: dict[str, Any]) -> str:
    parts = [
        "# 交付清单 (MANIFEST)",
        "",
        f"- run_id: `{manifest.get('run_id')}`",
        f"- venue: {manifest.get('venue')}",
        f"- pipeline: v{manifest.get('pipeline_version')}",
        f"- created_at: {manifest.get('created_at')}",
        f"- final_pdf: {manifest.get('final_pdf') or '(none)'}",
        f"- review: {manifest.get('review', {}).get('score')}/10 "
        f"{manifest.get('review', {}).get('verdict')}",
        "",
        f"未解决问题：major {open_issues['counts']['major']} / "
        f"minor {open_issues['counts']['minor']}",
        "",
        "## 文件",
        "",
        "| 路径 | 字节 |",
        "|---|---|",
    ]
    for item in manifest.get("deliverables", []):
        parts.append(f"| `{item['path']}` | {item['bytes']} |")
    usage = manifest.get("llm_usage") or {}
    if usage:
        parts += [
            "",
            "## LLM 用量",
            "",
            f"- calls: {usage.get('calls')}",
            f"- prompt tokens: {usage.get('prompt_tokens')}",
            f"- completion tokens: {usage.get('completion_tokens')}",
            f"- total tokens: {usage.get('total_tokens')}",
        ]
    return "\n".join(parts)


def _render_reproduce(state: dict[str, Any], ctx: Any) -> str:
    cfg = ctx.cfg
    seed = getattr(cfg, "seed", 0)
    parts = [
        "# 复现指南",
        "",
        "本文件描述如何从零重跑本次运行，以及如何独立验证论文中的数字。",
        "",
        "## 1. 重跑整条管线",
        "",
        "```powershell",
        f"python -m autoresearch.cli run --direction \"{state.get('direction', '')}\" `",
        f"    --venue {state.get('venue', 'NeurIPS')} --seed {seed}",
        "```",
        "",
        "## 2. 断点续跑",
        "",
        "```powershell",
        f"python -m autoresearch.cli resume {state.get('run_id', '<run_id>')}",
        "```",
        "",
        "运行目录下的 `state.json` 记录了每个阶段的状态与产物指纹；",
        "`checkpoints/` 保存了逐步快照，`events.jsonl` 是完整的结构化事件日志。",
        "",
        "## 3. 独立验证论文数字",
        "",
        "```powershell",
        f"python -m autoresearch.cli verify {state.get('run_id', '<run_id>')}",
        "```",
        "",
        "或手工执行：",
        "",
        "1. 打开 `metrics/baseline/metrics.csv` 与 `metrics/method/metrics.csv`；",
        "2. 用 `analysis/metrics_summary.json` 里的 mean/std/best 与论文表格逐项对照；",
        "3. `analysis/figure_inventory.json` 列出每张图对应的源数据文件。",
        "",
        "## 4. 手工编译论文",
        "",
        "见 `report/COMPILE_BLOCKED.md`（或直接 `tectonic -X compile paper/main.tex`）。",
        "",
        "## 5. 环境",
        "",
        f"- 目标会议：{getattr(cfg, 'venue', '')}",
        f"- 语言：{getattr(cfg, 'language', '')}",
        f"- 随机种子：{seed}",
        f"- 沙箱后端：{getattr(getattr(ctx, 'sandbox', None), 'name', '—')}",
        f"- 编译引擎：{getattr(getattr(cfg, 'compile', None), 'engine', '—')}",
        f"- LLM provider/model：{getattr(getattr(cfg, 'llm', None), 'provider', '—')}"
        f" / {getattr(getattr(cfg, 'llm', None), 'model', '—')}",
        "",
        "> 注意：实验脚本是确定性种子化的，同一 `--seed` 应复现同一组指标。",
        "> 若不一致，优先检查 Python 版本与 numpy 版本（浮点与哈希顺序可能不同）。",
        "",
    ]
    return "\n".join(parts)


__all__ = ["FinalizeStage"]

