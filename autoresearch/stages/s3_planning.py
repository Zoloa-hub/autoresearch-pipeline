"""阶段③：实验规划。

输入是 ② 选中的假设 + ① 的文献证据，输出是一份**claim 驱动**的实验路线图：
先写清「要证什么」，再倒推需要哪些 run、成功判据是什么、预算多少。

这里刻意不调用任何实验代码——规划与执行分离，使得「计划评审」可以在不烧
GPU/CPU 时间的前提下发生，也让 ④ 的失败不会被误归因为「计划本身有问题」。
"""

from __future__ import annotations

from typing import Any

from ..graph.state import Artifact
from .base import Stage, StageResult, as_float, clean_text, coerce_list

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "core_claim": {"type": "string"},
        "dataset": {"type": "object"},
        "baseline": {"type": "object"},
        "milestones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "runs": {"type": "integer"},
                    "est_minutes": {"type": "number"},
                    "success_criterion": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "metrics": {"type": "array", "items": {"type": "object"}},
        "ablation_matrix": {"type": "array", "items": {"type": "object"}},
        "compute_budget_hours": {"type": "number"},
        "risks": {"type": "array", "items": {"type": "object"}},
        "code_plan": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["objective", "milestones"],
}

#: 默认的消融矩阵。LLM 给出的矩阵若为空，用它兜底——没有消融的实验不足以支撑
#: 顶会级别的 claim，这条底线不能靠 LLM 自觉。
_DEFAULT_ABLATIONS = (
    {"name": "种子稳健性", "variants": ["seed=0", "seed=1", "seed=2"],
     "hypothesis": "结论不依赖单一种子（跨种子标准差应小于方法相对提升幅度）"},
    {"name": "容量/规模对照", "variants": ["small", "base", "large"],
     "hypothesis": "收益并非单纯来自参数量增加"},
    {"name": "训练预算对照", "variants": ["short", "base", "long"],
     "hypothesis": "收益在同等训练预算下依然存在"},
)


class PlanningStage(Stage):
    name = "s3_planning"
    title = "实验规划"
    requires = ("selected_idea",)
    produces = ("experiment_plan",)
    max_attempts = 2

    def run(self, state: dict[str, Any]) -> StageResult:
        missing = self.check_requires(state)
        idea = self.selected_idea(state)
        if not idea:
            return StageResult.failure("no selected idea to plan for")

        direction = str(state.get("direction") or "")
        venue = str(getattr(self.ctx.cfg, "venue", "NeurIPS"))
        warnings: list[str] = []
        if missing:
            warnings.append(f"planning degraded: missing {missing}")

        budget_hours = _estimate_budget(self.ctx, idea)
        plan = self.llm_json(
            "s3_plan",
            default=None,
            schema_hint=_PLAN_SCHEMA,
            idea_block=_format_idea_block(idea),
            venue=venue,
            compute_budget_hours=budget_hours,
            existing_code_block=_existing_code_block(self.ctx),
        )

        if not isinstance(plan, dict):
            warnings.append("LLM planning unavailable; built a deterministic minimal plan")
            plan = _fallback_plan(idea, budget_hours)

        plan = _normalize_plan(plan, idea, budget_hours, warnings)

        artifacts = [
            self.ctx.save_json("plan/experiment_plan.json", plan, stage=self.name),
            self.ctx.save_text(
                "plan/EXPERIMENT_PLAN.md", _render_plan(plan, idea, direction), stage=self.name
            ),
        ]

        detail = (
            f"{len(plan['milestones'])} milestones, "
            f"{plan['total_runs']} runs, ~{plan['compute_budget_hours']}h budget, "
            f"{len(plan['ablation_matrix'])} ablations"
        )
        return StageResult.success(
            detail=detail,
            updates={
                "experiment_plan": plan,
                "warnings": list(state.get("warnings") or []) + warnings,
            },
            artifacts=artifacts,
        )


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def _estimate_budget(ctx: Any, idea: dict[str, Any]) -> float:
    """按沙箱配置估算可用预算（小时）。"""

    timeout = as_float(getattr(getattr(ctx, "cfg", None), "sandbox", None) and
                       getattr(ctx.cfg.sandbox, "timeout", 900), 900.0)
    # 约定：实验阶段最多消耗 6 次单次超时长度的时间，取整到 0.5h
    hours = max(0.25, (timeout * 6) / 3600.0)
    return round(hours * 2) / 2


def _existing_code_block(ctx: Any) -> str:
    """告诉规划器「工作区里已经有什么可复用」，避免它凭空假设一个代码库。"""
    try:
        ws = getattr(ctx, "run_dir", None)
        if ws is None:
            return "（无既有代码库，需从零生成可运行的最小实验）"
        return (
            "（无外部代码库；管线会从零生成最小可运行实验。"
            "生成代码必须：仅用标准库/常见科学计算库、固定随机种子、"
            "CPU 可跑、不下载数据、把指标写进 metrics.csv 与 metrics.jsonl。）"
        )
    except Exception:  # pragma: no cover
        return "（未知）"


def _format_idea_block(idea: dict[str, Any]) -> str:
    novelty = idea.get("novelty") or {}
    lines = [
        f"id: {idea.get('id')}",
        f"title: {idea.get('title')}",
        f"hypothesis: {idea.get('hypothesis')}",
        f"motivation: {idea.get('motivation')}",
        f"method_sketch: {idea.get('method_sketch')}",
        f"novelty: {novelty.get('verdict')} (score={novelty.get('score')})",
        f"novelty_rationale: {novelty.get('rationale')}",
    ]
    closest = novelty.get("closest") or []
    if closest:
        lines.append("closest_prior_work:")
        for c in closest[:3]:
            lines.append(f"  - {c.get('title')} ({c.get('year')}): {c.get('why')}")
    metrics = idea.get("expected_metrics") or []
    if metrics:
        lines.append("expected_metrics: " + ", ".join(str(m) for m in metrics))
    risks = idea.get("risks") or []
    if risks:
        lines.append("risks: " + "; ".join(str(r) for r in risks))
    return "\n".join(lines)


def _fallback_plan(idea: dict[str, Any], budget_hours: float) -> dict[str, Any]:
    return {
        "objective": f"验证假设：{idea.get('hypothesis') or idea.get('title')}",
        "core_claim": idea.get("novelty_claim") or idea.get("title") or "所提方法优于基线",
        "dataset": {"name": "synthetic-controlled", "source": "in-pipeline generator",
                    "size": "small", "split": "train/val"},
        "baseline": {"name": "baseline", "description": "与所提方法同架构同预算的对照",
                     "expected_metrics": ["accuracy", "f1"]},
        "milestones": [
            {"id": "M1", "name": "跑通基线", "description": "在受控设置下复现基线指标",
             "runs": 3, "est_minutes": 5, "success_criterion": "基线指标稳定且无明显发散"},
            {"id": "M2", "name": "方法对照", "description": "在同一设置下运行所提方法",
             "runs": 3, "est_minutes": 5, "success_criterion": "方法指标均值高于基线且差异大于跨种子标准差"},
        ],
        "metrics": [
            {"name": "accuracy", "direction": "higher", "primary": True},
            {"name": "f1", "direction": "higher", "primary": True},
            {"name": "loss", "direction": "lower", "primary": False},
        ],
        "ablation_matrix": list(_DEFAULT_ABLATIONS),
        "compute_budget_hours": budget_hours,
        "risks": [{"risk": "效应量低于噪声", "mitigation": "多种子 + 报告标准差"}],
        "code_plan": [{"file": "train.py", "purpose": "单文件可运行实验，输出 metrics.csv/jsonl"}],
    }


def _normalize_plan(
    plan: dict[str, Any], idea: dict[str, Any], budget_hours: float, warnings: list[str]
) -> dict[str, Any]:
    objective = clean_text(str(plan.get("objective") or "")).strip()
    if not objective:
        objective = f"验证假设：{idea.get('hypothesis') or idea.get('title')}"

    milestones_raw = coerce_list(plan.get("milestones"))
    milestones: list[dict[str, Any]] = []
    for i, m in enumerate(milestones_raw, 1):
        if not isinstance(m, dict):
            m = {"name": str(m)}
        name = clean_text(str(m.get("name") or "")).strip()
        if not name:
            continue
        milestones.append(
            {
                "id": str(m.get("id") or f"M{i}"),
                "name": name,
                "description": clean_text(str(m.get("description") or "")),
                "runs": max(1, int(as_float(m.get("runs"), 1))),
                "est_minutes": max(0.5, as_float(m.get("est_minutes"), 5.0)),
                "success_criterion": clean_text(str(m.get("success_criterion") or "")),
                "depends_on": [str(x) for x in coerce_list(m.get("depends_on"))],
            }
        )
    if not milestones:
        warnings.append("no milestones parsed; using the deterministic minimal plan")
        milestones = _fallback_plan(idea, budget_hours)["milestones"]

    metrics_raw = coerce_list(plan.get("metrics"))
    metrics: list[dict[str, Any]] = []
    for m in metrics_raw:
        if isinstance(m, str):
            metrics.append({"name": m, "direction": _direction(m), "primary": not metrics})
            continue
        if not isinstance(m, dict):
            continue
        name = clean_text(str(m.get("name") or "")).strip()
        if not name:
            continue
        direction = str(m.get("direction") or "").lower()
        if direction not in ("higher", "lower"):
            direction = _direction(name)
        metrics.append({"name": name, "direction": direction, "primary": bool(m.get("primary"))})
    if not metrics:
        metrics = _fallback_plan(idea, budget_hours)["metrics"]
        warnings.append("no metrics parsed; defaulted to accuracy/f1/loss")
    if not any(m["primary"] for m in metrics):
        metrics[0]["primary"] = True

    ablations = [a for a in coerce_list(plan.get("ablation_matrix")) if isinstance(a, dict)]
    normalized_ablations = []
    for a in ablations:
        variants = [str(v) for v in coerce_list(a.get("variants")) if str(v).strip()]
        if len(variants) < 2:
            continue
        normalized_ablations.append(
            {
                "name": clean_text(str(a.get("name") or "ablation")),
                "variants": variants,
                "hypothesis": clean_text(str(a.get("hypothesis") or "")),
            }
        )
    if not normalized_ablations:
        normalized_ablations = list(_DEFAULT_ABLATIONS)
        warnings.append("ablation matrix missing; injected the default ablation battery")

    total_runs = sum(m["runs"] for m in milestones)
    est_minutes = sum(m["est_minutes"] * m["runs"] for m in milestones)

    baseline = plan.get("baseline") if isinstance(plan.get("baseline"), dict) else {}
    dataset = plan.get("dataset") if isinstance(plan.get("dataset"), dict) else {}

    return {
        "objective": objective,
        "core_claim": clean_text(str(plan.get("core_claim") or idea.get("novelty_claim") or idea.get("title") or "")),
        "dataset": dataset or {"name": "synthetic-controlled", "source": "in-pipeline generator"},
        "baseline": baseline or {"name": "baseline", "description": "同架构同预算对照"},
        "milestones": milestones,
        "metrics": metrics,
        "ablation_matrix": normalized_ablations,
        "compute_budget_hours": as_float(plan.get("compute_budget_hours"), budget_hours) or budget_hours,
        "risks": [r for r in coerce_list(plan.get("risks")) if isinstance(r, dict)],
        "code_plan": [c for c in coerce_list(plan.get("code_plan")) if isinstance(c, dict)],
        "total_runs": total_runs,
        "estimated_minutes": round(est_minutes, 1),
        "idea_id": idea.get("id", ""),
    }


def _direction(name: str) -> str:
    """指标方向字符串（``"higher"``/``"lower"``）：转调共享实现。"""
    from ..tools.metrics import _higher_is_better as _shared

    return "higher" if _shared(name) else "lower"

def _render_plan(plan: dict[str, Any], idea: dict[str, Any], direction: str) -> str:
    parts = [
        f"# 实验计划：{idea.get('title') or direction}",
        "",
        f"**目标**：{plan['objective']}",
        "",
        f"**核心 claim**：{plan['core_claim']}",
        "",
        f"**预算**：{plan['total_runs']} 次 run，约 {plan['estimated_minutes']} 分钟，"
        f"上限 {plan['compute_budget_hours']} 小时",
        "",
        "## 数据与基线",
        "",
        f"- 数据集：`{plan['dataset'].get('name', 'n/a')}` "
        f"（来源：{plan['dataset'].get('source', 'n/a')}）",
        f"- 基线：`{plan['baseline'].get('name', 'baseline')}` — "
        f"{plan['baseline'].get('description', '')}",
        "",
        "## 里程碑",
        "",
        "| ID | 里程碑 | runs | 预计分钟 | 成功判据 |",
        "|---|---|---|---|---|",
    ]
    for m in plan["milestones"]:
        parts.append(
            f"| {m['id']} | {m['name']} | {m['runs']} | {m['est_minutes']:g} | "
            f"{m['success_criterion'] or '—'} |"
        )
    parts += ["", "## 指标", "", "| 指标 | 方向 | 主指标 |", "|---|---|---|"]
    for m in plan["metrics"]:
        parts.append(f"| {m['name']} | {m['direction']} | {'✅' if m['primary'] else ''} |")

    parts += ["", "## 消融矩阵", ""]
    for a in plan["ablation_matrix"]:
        parts.append(f"### {a['name']}")
        parts.append("")
        parts.append(f"变体：{', '.join(a['variants'])}")
        if a.get("hypothesis"):
            parts.append("")
            parts.append(f"假设：{a['hypothesis']}")
        parts.append("")

    if plan.get("risks"):
        parts += ["## 风险与缓解", ""]
        for r in plan["risks"]:
            parts.append(f"- **{r.get('risk', '?')}** → {r.get('mitigation', '—')}")
        parts.append("")

    if plan.get("code_plan"):
        parts += ["## 代码计划", ""]
        for c in plan["code_plan"]:
            parts.append(f"- `{c.get('file', '?')}` — {c.get('purpose', '')}")
        parts.append("")

    return "\n".join(parts)


__all__ = ["PlanningStage"]
