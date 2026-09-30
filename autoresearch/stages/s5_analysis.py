"""阶段⑤：指标汇总、自动制图与定量表格。

原则：**图表是确定性代码的产物，不是 LLM 的产物。**
LLM 只被允许「解释」已经算好的数字（见输出里的 ``analysis``），
绝不允许它生成或修改任何数值。这样即使 LLM 胡言，图与表仍然可信，
而审计者可以拿图去对 CSV。

本阶段全程「失败不致命」：没有图就继续，没有表就继续，
甚至没有任何指标也能继续——管线要能产出一份诚实的「实验未产出可用指标」的论文，
而不是崩在画图这一步。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..graph.state import Artifact
from .base import Stage, StageResult, as_float, clamp, clean_text, coerce_list

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "partially_supported", "not_supported", "inconclusive"],
                    },
                    "evidence": {"type": "string"},
                    "numbers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "verdict"],
            },
        },
        "findings": {"type": "array", "items": {"type": "object"}},
        "negative_results": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "threats_to_validity": {"type": "array", "items": {"type": "string"}},
        "figure_discussion": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["claim_evidence"],
}

_METRIC_FILES = ("metrics.csv", "metrics.jsonl", "metrics.json", "metrics.log")


class AnalysisStage(Stage):
    name = "s5_analysis"
    title = "指标汇总、制图与定量表格"
    requires = ("experiment_plan",)
    produces = ("metrics", "metrics_summary", "figured", "latex_tables", "analysis")
    max_attempts = 2

    def run(self, state: dict[str, Any]) -> StageResult:
        warnings: list[str] = []
        artifacts: list[Artifact] = []
        plan = self.state_dict(state, "experiment_plan")

        # --- 1. 汇总（确定性） ----------------------------------------- #
        summary = self._summarize(state, warnings)
        metrics_dir = self.ctx.path("metrics")
        fig_dir = self.ctx.path("figures")

        # --- 2. 制图 + 表格（确定性） ---------------------------------- #
        figures: dict[str, list[str]] = {}
        table_tex = ""
        try:
            from ..tools.figures import make_all_figures

            produced = make_all_figures(metrics_dir, fig_dir, formats=("pdf", "png")) or {}
        except Exception as exc:
            warnings.append(f"figure generation unavailable: {exc}")
            self._warn(f"s5: make_all_figures failed: {exc}")
            produced = {}

        tables: dict[str, str] = {}
        # 注意：``tools.figures.make_all_figures`` 也会写一个
        # ``analysis/metrics_summary.json``，但那是**扁平聚合**（所有 run 混在一起），
        # 用途是画图，不是报告。这里绝不能把它读回来当成本阶段的 summary——
        # 那会用「混合统计」覆盖掉逐 run / 跨种子的正确口径。
        # 图表生成器返回的产出只取图与表，统计一律以本阶段自己算的为准。
        summary_json = produced.pop("__summary_json__", []) if isinstance(produced, dict) else []
        if isinstance(summary_json, list) and summary_json:
            try:
                # 归档为独立文件，避免与 analysis/metrics_summary.json 撞名
                flat = self.ctx.load_json(self.ctx.rel(Path(summary_json[0])), default=None)
                if isinstance(flat, dict) and flat:
                    artifacts.append(
                        self.ctx.save_json(
                            "analysis/aggregate_stats.json", flat, stage=self.name
                        )
                    )
            except Exception as exc:
                warnings.append(f"could not archive aggregate stats: {exc}")

        table_paths = produced.pop("__table_tex__", []) if isinstance(produced, dict) else []
        if isinstance(table_paths, list) and table_paths:
            try:
                table_tex = Path(table_paths[0]).read_text(encoding="utf-8", errors="replace")
                tables["evaluation"] = table_tex
            except OSError as exc:
                warnings.append(f"could not read evaluation table: {exc}")

        for name, paths in (produced or {}).items():
            if isinstance(paths, (list, tuple)):
                figures[str(name)] = [self.ctx.rel(Path(p)) for p in paths]

        if not figures:
            warnings.append("no figures produced (missing metrics or plotting failure)")
            figures = self._placeholder_figures_table(summary)
            if figures:
                table_tex = table_tex or self._fallback_table(summary)

        # --- 3. 表格：本阶段的表格是**权威版本**，覆盖 figures 层的扁平表 ------- #
        # figures 层的 evaluation_table.tex 把所有 run 混在一张表里（含每种子的中间行），
        # 不适合直接进论文。这里用跨种子口径重建，并以其为准。
        if summary:
            try:
                rows = _summary_to_rows(summary)
                if rows:
                    from ..tools.metrics import to_latex_table

                    tables["evaluation"] = to_latex_table(
                        rows,
                        caption=(
                            "主结果：各变体在多个随机种子上的均值 ± 标准差"
                            "（best 为每种子最优值后再取跨种子统计）。"
                        ),
                        label="tab:main",
                        bold_best=True,
                    )
            except Exception as exc:
                warnings.append(f"table construction failed: {exc}")
        if not tables and summary:
            fallback = self._fallback_table(summary)
            if fallback:
                tables["evaluation"] = fallback

        comparison = self.state_dict(state, "baseline_results"), self.state_dict(
            state, "method_results"
        )

        # --- 4. LLM 证据分级分析（只解释，不产数字） ------------------- #
        analysis = self._analyze(state, summary, tables, figures, plan, warnings)

        # --- 5. 落盘 --------------------------------------------------- #
        artifacts.append(
            self.ctx.save_json("analysis/metrics_summary.json", summary, stage=self.name)
        )
        if figures:
            artifacts.append(
                self.ctx.save_json("analysis/figure_inventory.json", figures, stage=self.name)
            )
        if tables:
            artifacts.append(
                self.ctx.save_json("analysis/tables.json", tables, stage=self.name)
            )
        artifacts.append(
            self.ctx.save_json("analysis/analysis.json", analysis, stage=self.name)
        )
        artifacts.append(
            self.ctx.save_text(
                "analysis/RESULTS_ANALYSIS.md",
                _render_analysis(state, summary, tables, figures, analysis, warnings),
                stage=self.name,
            )
        )

        n_metrics = len(summary)
        detail = (
            f"{n_metrics} metric groups, {sum(len(v) for v in figures.values())} figures, "
            f"{len(tables)} tables, "
            f"{len(coerce_list(analysis.get('claim_evidence')))} claims graded"
        )
        return StageResult.success(
            detail=detail,
            updates={
                "metrics": {
                    "baseline": comparison[0],
                    "method": comparison[1],
                },
                "metrics_summary": summary,
                "figured": figures,
                "latex_tables": tables,
                "analysis": analysis,
                "warnings": list(state.get("warnings") or []) + warnings,
            },
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------ #
    def _summarize(self, state: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        """从磁盘上的指标文件重建统计——**不信任**状态里的缓存。

        两级汇总，因为自动实验天然是多运行级别的：

        * ``runs``：``{run_path: {metric: [逐 epoch 值]}}`` —— 用来画学习曲线，
          横轴是 epoch，每个 run 一条线；
        * ``summary``：论文表格用的统计量。当同一个变体有多个种子时，
          **先算每个种子的 best/final，再在种子维度上算 mean/std**，
          而不是把所有 epoch 的点混在一起算——后者会把「收敛过程」当成
          「结果方差」，把误差棒算大，是实验报告里常见的隐性错误。
        """
        metrics_dir = self.ctx.path("metrics")
        series_map: dict[str, dict[str, list[float]]] = {}
        try:
            from ..tools.metrics import parse_metrics

            for path in sorted(metrics_dir.rglob("*")):
                if not path.is_file() or path.name not in _METRIC_FILES:
                    continue
                run_name = self.ctx.rel(path.parent if path.parent != metrics_dir
                                        else path.with_suffix(""))
                parsed = parse_metrics(path) or {}
                clean = {k: v for k, v in parsed.items() if isinstance(v, list) and v}
                if not clean:
                    continue
                # 同名指标在多个文件里（csv + jsonl）时，用更长的序列覆盖短序列，
                # 避免 jsonl 的额外字段把 csv 的 epoch 序列截断。
                bucket = series_map.setdefault(run_name, {})
                for k, v in clean.items():
                    if len(v) >= len(bucket.get(k) or []):
                        bucket[k] = v
        except Exception as exc:
            warnings.append(f"metric parsing unavailable: {exc}")
            self._warn(f"s5: parse_metrics failed: {exc}")

        if not series_map:
            # 退化：直接用状态里的序列（④ 已解析过一次）
            for key, run_name in (("baseline_results", "baseline"), ("method_results", "method")):
                block = self.state_dict(state, key)
                if block:
                    series_map[run_name] = {
                        k: [float(x) for x in v if isinstance(x, (int, float))]
                        for k, v in block.items()
                        if isinstance(v, list) and v
                    }
            if series_map:
                warnings.append("metrics rebuilt from state (no metric files on disk)")

        if not series_map:
            warnings.append("no metrics available at all")
            return {}

        summary = _stat_summary(series_map)

        cross_seed = _cross_seed_summary(series_map)
        if cross_seed:
            for variant, stats in cross_seed.items():
                summary[f"{variant} (multi-seed)"] = stats
            n_seeds = max((len(v) for v in cross_seed.values()), default=0)
            warnings.append(f"cross-seed statistics computed (up to {n_seeds} seeds per variant)")

        # 把原始序列一并带上，供下游（写作/评审）引用真实数字
        return {
            "runs": series_map,
            "summary": summary,
            "cross_seed": cross_seed,
        }

    def _fallback_table(self, summary: dict[str, Any]) -> str:
        try:
            from ..tools.metrics import to_latex_table

            rows = _summary_to_rows(summary)
            if rows:
                return to_latex_table(
                    rows, caption="Main results", label="tab:main", bold_best=True
                )
        except Exception as exc:
            self._warn(f"s5: to_latex_table failed: {exc}")
        return ""

    def _analyze(
        self,
        state: dict[str, Any],
        summary: dict[str, Any],
        tables: dict[str, str],
        figures: dict[str, list[str]],
        plan: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        metrics_summary_block = _format_summary_block(summary)
        result = self.llm_json(
            "s5_analysis",
            default=None,
            schema_hint=_ANALYSIS_SCHEMA,
            metrics_summary_block=metrics_summary_block,
            tables_block=clamp("\n\n".join(tables.values()), 6000) or "（无表格）",
            figure_inventory=clamp(
                "\n".join(f"- {k}: {', '.join(v)}" for k, v in figures.items()), 3000
            )
            or "（无图）",
            plan_block=_format_plan_block(plan),
            core_claim=str(plan.get("core_claim") or ""),
        )
        if isinstance(result, dict):
            return result
        warnings.append("LLM analysis unavailable; emitting deterministic claim grading")
        return _deterministic_analysis(state, summary)

    # ------------------------------------------------------------------ #
    def _placeholder_figures_table(self, summary: dict[str, Any]) -> dict[str, list[str]]:
        """没有可画的图时返回空清单——**不造假图**。

        宁可交付一份「无图」的结果，也不要一张用伪造数据画出来的曲线：
        后者是自动科研里最难被发现的错误，因为它看起来最正常。
        """
        return {}


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def _summary_to_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """把内部 summary 拍成论文表格的行。

    优先使用 **跨种子统计**（``cross_seed``）：论文的主表应该报「每个变体在多个种子
    上的 mean±std」，而不是「某个种子某次运行的 best」。只有当没有多种子数据时，
    才退回逐 run 的统计（并在行名上标明是单次运行，避免读者误读为方差）。
    """
    if not isinstance(summary, dict):
        return []

    cross = summary.get("cross_seed")
    if isinstance(cross, dict) and cross:
        rows: list[dict[str, Any]] = []
        for variant in sorted(cross):
            stats = cross[variant]
            if not isinstance(stats, dict):
                continue
            row: dict[str, Any] = {"name": variant}
            n_seeds = 0
            for key, st in stats.items():
                if not isinstance(st, dict):
                    continue
                # 两种命名都可能出现：``metric`` 与 ``metric__best``
                if key.endswith("__best"):
                    metric = key[: -len("__best")]
                elif key.endswith("__final"):
                    continue
                else:
                    metric = key
                if not metric or metric.startswith("_"):
                    continue
                row[metric] = round(as_float(st.get("mean")), 6)
                row[f"{metric}_std"] = round(as_float(st.get("std")), 6)
                n_seeds = max(n_seeds, int(as_float(st.get("count"))))
            if len(row) > 1:
                row["name"] = f"{variant} ({n_seeds} seeds)" if n_seeds > 1 else variant
                rows.append(row)
        if rows:
            return rows

    inner = summary.get("summary") if isinstance(summary, dict) else None
    if not isinstance(inner, dict) or not inner:
        return []
    rows = []
    for run_name, metrics in inner.items():
        if not isinstance(metrics, dict) or run_name.startswith("_"):
            continue
        row: dict[str, Any] = {"name": run_name}
        for metric, stats in metrics.items():
            if metric.startswith("_") or not isinstance(stats, dict):
                continue
            best = stats.get("best", stats.get("mean"))
            std = stats.get("std")
            if best is None:
                continue
            row[metric] = round(as_float(best), 6)
            if std is not None:
                row[f"{metric}_std"] = round(as_float(std), 6)
        if len(row) > 1:
            rows.append(row)
    return rows


def _build_comparison_table(state: dict[str, Any], summary: dict[str, Any]) -> str:
    rows = _summary_to_rows(summary)
    if not rows:
        return ""
    try:
        from ..tools.metrics import to_latex_table

        return to_latex_table(
            rows,
            caption=f"Main results ({state.get('direction', '')})",
            label="tab:main",
            bold_best=True,
        )
    except Exception:
        return ""


def _stat_summary(series_map: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    """逐 run 统计；优先用 ``tools.metrics.summarize``，失败则内联实现。"""
    try:
        from ..tools.metrics import summarize

        result = summarize(series_map) or {}
        if result:
            return result
    except Exception:
        pass
    return _inline_summarize(series_map)


#: 从 ``runs/method/seed_3`` 这样的目录名里抠出变体名与种子号。
_VARIANT_RE = re.compile(r"(?:^|/)([A-Za-z][\w\-]*)")
_SEED_RE = re.compile(r"seed[_\-=]?(\d+)", re.IGNORECASE)


def _cross_seed_summary(
    series_map: dict[str, dict[str, list[float]]]
) -> dict[str, dict[str, Any]]:
    """把 ``variant/seed_N`` 归并成变体级的 mean±std。

    这是实验报告的正确口径：**先在每个种子上取一个标量（best / final），
    再在种子维度上算均值与标准差**。若把所有 epoch 的点混在一起统计，
    得到的是「收敛轨迹的散布」而不是「结果的不确定性」，误差棒会系统性偏大。

    返回 ``{variant: {metric: {count, mean, std, min, max, best_per_seed, seeds}}}``。
    """
    grouped: dict[str, dict[str, list[float]]] = {}
    seeds_seen: dict[str, set[str]] = {}

    for run_path, metrics in series_map.items():
        seed_match = _SEED_RE.search(run_path)
        if not seed_match:
            # 顶层 ``metrics/<variant>/`` 是各种子文件的副本（供人直接查看），
            # 若把它也计入就会与 ``seed_N`` 重复计数。跨种子统计只认带 seed 的目录。
            continue
        seed_tag = seed_match.group(1)
        # 变体名 = 路径里 seed 之前的那一段
        head = run_path.split("/seed")[0].split("\\seed")[0]
        parts = [p for p in re.split(r"[/\\]", head) if p and p != "metrics"]
        variant = parts[-1] if parts else run_path
        if not variant:
            continue

        for name, values in metrics.items():
            grouped.setdefault(variant, {}).setdefault(f"{name}__best", []).append(
                _scalar(values, name, "best")
            )
            grouped.setdefault(variant, {}).setdefault(f"{name}__final", []).append(
                _scalar(values, name, "final")
            )
        seeds_seen.setdefault(variant, set()).add(seed_tag)

    out: dict[str, dict[str, Any]] = {}
    for variant, metrics in grouped.items():
        stats: dict[str, Any] = {}
        for key, values in metrics.items():
            nums = [float(v) for v in values if isinstance(v, (int, float))]
            if not nums:
                continue
            n = len(nums)
            mean = sum(nums) / n
            var = sum((x - mean) ** 2 for x in nums) / n if n > 1 else 0.0
            stats[key] = {
                "count": float(n),
                "mean": mean,
                "std": var ** 0.5,
                "min": min(nums),
                "max": max(nums),
                "first": nums[0],
                "final": nums[-1],
                "best": (max(nums) if not _is_lower_better(key) else min(nums)),
            }
        if stats:
            stats["_seeds"] = sorted(seeds_seen.get(variant, set()))
            out[variant] = stats
    return out


def _scalar(values: list[float], name: str, mode: str) -> float:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return float("nan")
    if mode == "final":
        return nums[-1]
    return min(nums) if _is_lower_better(name) else max(nums)


def _is_lower_better(name: str) -> bool:
    """指标方向：转调共享实现（tools.metrics 是唯一规范表）。"""
    try:
        from ..tools.metrics import _higher_is_better as _shared

        return not bool(_shared(name))
    except Exception:  # pragma: no cover - tools 层不可用时的保守回退
        return any(
            t in (name or "").lower()
            for t in ("loss", "error", "rmse", "mae", "mse", "perplexity", "ppl", "nll")
        )

def _inline_summarize(series_map: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    """零依赖的 summarize 兜底实现。"""
    out: dict[str, Any] = {}
    for run, metrics in series_map.items():
        out[run] = {}
        for name, values in metrics.items():
            nums = [float(v) for v in values if isinstance(v, (int, float))]
            if not nums:
                continue
            n = len(nums)
            mean = sum(nums) / n
            var = sum((x - mean) ** 2 for x in nums) / n if n else 0.0
            lower_better = any(
                t in name.lower()
                for t in ("loss", "error", "rmse", "mae", "mse", "perplexity", "ppl", "nll")
            )
            out[run][name] = {
                "count": n,
                "mean": mean,
                "std": var ** 0.5,
                "min": min(nums),
                "max": max(nums),
                "first": nums[0],
                "final": nums[-1],
                "best": min(nums) if lower_better else max(nums),
            }
    return out


def _format_summary_block(summary: dict[str, Any], max_runs: int = 6) -> str:
    """给 LLM 的指标块：明确标注「这些是真实测量值，只能引用不能改写」。"""
    if not summary:
        return "（没有任何可用指标。请在结论中明确说明实验未产出可用指标。）"
    inner = summary.get("summary") if isinstance(summary, dict) else None
    lines = [
        "以下数据由实验脚本直接写出，是唯一的数字来源。引用时**必须逐字一致**，"
        "不得四舍五入成不同的值，不得外推，不得补齐缺失项。",
        "",
    ]
    if isinstance(inner, dict) and inner:
        for run, metrics in list(inner.items())[:max_runs]:
            lines.append(f"### run: {run}")
            for name, stats in (metrics or {}).items():
                if not isinstance(stats, dict):
                    continue
                lines.append(
                    f"- {name}: mean={as_float(stats.get('mean')):.6g}, "
                    f"std={as_float(stats.get('std')):.6g}, "
                    f"min={as_float(stats.get('min')):.6g}, "
                    f"max={as_float(stats.get('max')):.6g}, "
                    f"first={as_float(stats.get('first')):.6g}, "
                    f"final={as_float(stats.get('final')):.6g}, "
                    f"best={as_float(stats.get('best')):.6g}, "
                    f"n={int(as_float(stats.get('count'), 0))}"
                )
            lines.append("")
    raw = summary.get("runs") if isinstance(summary, dict) else None
    if isinstance(raw, dict):
        lines.append("#### 逐点序列（截断）")
        for run, metrics in list(raw.items())[:max_runs]:
            for name, values in list((metrics or {}).items())[:6]:
                preview = ", ".join(f"{as_float(v):.6g}" for v in list(values)[:20])
                lines.append(f"- {run}.{name} = [{preview}{', …' if len(values) > 20 else ''}]")
    return "\n".join(lines)


def _format_plan_block(plan: dict[str, Any]) -> str:
    if not plan:
        return "（无实验计划）"
    lines = [
        f"objective: {plan.get('objective')}",
        f"core_claim: {plan.get('core_claim')}",
        f"baseline: {plan.get('baseline')}",
    ]
    for m in plan.get("milestones") or []:
        lines.append(f"- {m.get('id')} {m.get('name')}: success={m.get('success_criterion')}")
    return "\n".join(lines)


def _deterministic_analysis(state: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """LLM 不可用时的确定性「证据分级」：只看数字的方向与量级。"""
    inner = (summary or {}).get("summary") or {}
    baseline = inner.get("baseline") or {}
    method = inner.get("method") or {}
    claim_evidence: list[dict[str, Any]] = []
    for metric in sorted(set(baseline) & set(method)):
        b = as_float((baseline.get(metric) or {}).get("best"))
        m = as_float((method.get(metric) or {}).get("best"))
        b_std = as_float((baseline.get(metric) or {}).get("std"))
        m_std = as_float((method.get(metric) or {}).get("std"))
        lower_better = any(
            t in metric.lower()
            for t in ("loss", "error", "rmse", "mae", "mse", "perplexity", "nll")
        )
        delta = (b - m) if lower_better else (m - b)
        noise = max(b_std, m_std)
        if delta > noise and delta != 0:
            verdict = "supported"
        elif delta > 0:
            verdict = "partially_supported"
        elif delta == 0:
            verdict = "inconclusive"
        else:
            verdict = "not_supported"
        claim_evidence.append(
            {
                "claim": f"所提方法在 {metric} 上优于基线",
                "verdict": verdict,
                "evidence": (
                    f"baseline best={b:.6g} (std={b_std:.3g})，"
                    f"method best={m:.6g} (std={m_std:.3g})，Δ={delta:+.6g}"
                ),
                "numbers": [f"{b:.6g}", f"{m:.6g}"],
            }
        )
    return {
        "claim_evidence": claim_evidence,
        "findings": [],
        "negative_results": [
            c["evidence"] for c in claim_evidence if c["verdict"] == "not_supported"
        ],
        "limitations": [
            "指标来自单次运行的受控设置，跨数据集泛化性未验证。",
            "无人工评估，结论仅限自动指标。",
        ],
        "threats_to_validity": [
            "对照仅覆盖同一代码路径下的两个变体，未包含文献中的强基线。",
        ],
        "figure_discussion": [],
        "_generated_by": "deterministic",
    }


def _render_analysis(
    state: dict[str, Any],
    summary: dict[str, Any],
    tables: dict[str, str],
    figures: dict[str, list[str]],
    analysis: dict[str, Any],
    warnings: list[str],
) -> str:
    plan = state.get("experiment_plan") or {}
    parts = [
        "# 结果与分析",
        "",
        f"**目标**：{plan.get('objective', '—')}",
        "",
        f"**核心 claim**：{plan.get('core_claim', '—')}",
        "",
        "## 1. 定量结果",
        "",
    ]
    rows = _summary_to_rows(summary)
    if rows:
        metrics = sorted({k for r in rows for k in r if k != "name" and not k.endswith("_std")})
        parts.append("| run | " + " | ".join(metrics) + " |")
        parts.append("|" + "---|" * (len(metrics) + 1))
        for r in rows:
            cells = []
            for m in metrics:
                value = r.get(m)
                std = r.get(f"{m}_std")
                if value is None:
                    cells.append("--")
                elif std is None:
                    cells.append(f"{value:.6g}")
                else:
                    cells.append(f"{value:.6g} ± {std:.3g}")
            parts.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    else:
        parts.append("_没有可汇总的指标。_")

    parts += ["", "## 2. 图与表的清单", ""]
    if figures:
        for name, paths in figures.items():
            parts.append(f"- **{name}**：{', '.join(f'`{p}`' for p in paths)}")
    else:
        parts.append("_本阶段未生成图表（无可用指标）。_")
    for name in tables:
        parts.append(f"- **表 {name}**：见 `analysis/tables.json`")

    parts += ["", "## 3. 证据分级", ""]
    claim_evidence = coerce_list(analysis.get("claim_evidence"))
    if claim_evidence:
        parts.append("| claim | 判定 | 证据 |")
        parts.append("|---|---|---|")
        for c in claim_evidence:
            if not isinstance(c, dict):
                continue
            parts.append(
                f"| {clean_text(str(c.get('claim') or ''))} | "
                f"`{c.get('verdict')}` | {clean_text(str(c.get('evidence') or ''))} |"
            )
    else:
        parts.append("_未能进行证据分级。_")

    for title, key in (
        ("研究发现", "findings"),
        ("负面结果（诚实呈现）", "negative_results"),
        ("局限", "limitations"),
        ("效度威胁", "threats_to_validity"),
    ):
        items = coerce_list(analysis.get(key))
        if not items:
            continue
        parts += ["", f"## 4. {title}", ""]
        for item in items:
            if isinstance(item, dict):
                parts.append(
                    f"- **{clean_text(str(item.get('finding') or item.get('risk') or item.get('issue') or ''))}**"
                    f" — {clean_text(str(item.get('evidence') or item.get('significance') or item.get('mitigation') or ''))}"
                )
            else:
                parts.append(f"- {clean_text(str(item))}")

    if warnings:
        parts += ["", "## 附录：本阶段告警", ""] + [f"- {w}" for w in warnings] + [""]

    return "\n".join(parts)


__all__ = ["AnalysisStage"]
