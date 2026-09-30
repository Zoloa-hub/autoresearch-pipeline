"""Auto-Research 管线状态 schema。

契约见 ``autoresearch/CONTRACTS.md`` §9。状态是一个**扁平、JSON 可序列化**的
``dict``，这是刻意选择：

* 断点续跑只需 ``json.dump``；
* 自研引擎与 LangGraph 的 ``TypedDict`` 状态可一一对应，迁移成本为零；
* 阶段之间通过显式键名耦合，避免隐式全局状态。

任何阶段都**不得**把不可序列化对象（模型、句柄、``Path``、``numpy`` 数组）写回
状态——需要落盘的东西先 ``ctx.save_json``，再把相对路径写进状态。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# 阶段定义
# --------------------------------------------------------------------------- #

STAGES: tuple[str, ...] = (
    "s1_literature",
    "s2_ideation",
    "s3_planning",
    "s4_experiment",
    "s5_analysis",
    "s6_writing",
    "s7_compile",
    "s8_review",
    "s9_finalize",
)

STAGE_TITLES: dict[str, str] = {
    "s1_literature": "选题构思与文献综述",
    "s2_ideation": "假设生成与新颖性校验",
    "s3_planning": "实验规划",
    "s4_experiment": "沙箱内代码执行与自纠错",
    "s5_analysis": "指标汇总、制图与定量表格",
    "s6_writing": "论文分章撰写",
    "s7_compile": "LaTeX 编译与语法修复",
    "s8_review": "自动同行评审与迭代修改",
    "s9_finalize": "交付打包与最终报告",
}

#: 阶段之间的**声明顺序**，引擎默认沿此顺序推进，``router`` 可改变流向。
STAGE_ORDER: dict[str, int] = {name: i for i, name in enumerate(STAGES)}

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

TERMINAL_STATUSES = (STATUS_DONE, STATUS_SKIPPED)


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #


@dataclass
class Artifact:
    """一份落盘产物的元数据。``path`` 一律是相对 ``run_dir`` 的 POSIX 路径。"""

    path: str
    kind: str  # json|md|csv|py|tex|pdf|png|bib|log
    stage: str
    sha256: str = ""
    bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Artifact:
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return f"{self.kind}:{self.path}"


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """流式 SHA256，用于登记产物指纹。文件不存在返回空串。"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #

#: 每个键的默认值。``new_state`` 以此为底，保证所有阶段看到同一套键。
STATE_DEFAULTS: dict[str, Any] = {
    # --- 运行元信息（runner 写入，之后只读） ---
    "run_id": "",
    "direction": "",
    "venue": "NeurIPS",
    "language": "zh",
    "seed": 0,
    "started_at": "",
    # --- 引擎簿记 ---
    "stage_status": {},
    "stage_attempts": {},
    "stage_errors": {},
    "trace": [],
    "current_stage": "",
    "steps": 0,
    # --- 产物登记 ---
    "artifacts": [],
    # --- s1 文献 ---
    "queries": [],
    "papers": [],
    "lit_review": "",
    "themes": [],
    "gaps": [],
    "bib_entries": [],
    # --- s2 构思 ---
    "ideas": [],
    "selected_idea": {},
    "novelty_summary": {},
    # --- s3 规划 ---
    "experiment_plan": {},
    # --- s4 实验 ---
    "code_files": [],
    "workspace": "",
    "baseline_results": {},
    "method_results": {},
    "debug_history": [],
    "runs_executed": [],
    # --- s5 分析 ---
    "metrics": {},
    "metrics_summary": {},
    "figured": {},
    "latex_tables": {},
    "analysis": {},
    # --- s6 写作 ---
    "paper_sections": {},
    "paper_title": "",
    "paper_abstract": "",
    "paper_tex": "",
    "citations_used": [],
    # --- s7 编译 ---
    "compile_result": {},
    "compile_fix_history": [],
    # --- s8 评审 ---
    "reviews": [],
    "review_score": 0.0,
    "review_verdict": "",
    "review_round": 0,
    "revision_history": [],
    # --- s9 交付 ---
    "final_pdf": "",
    "final_report": "",
    "deliverables": [],
    # --- 全局 ---
    "errors": [],
    "warnings": [],
    "notes": {},
}


def new_state(
    run_id: str = "",
    direction: str = "",
    venue: str = "NeurIPS",
    language: str = "zh",
    seed: int = 0,
) -> dict[str, Any]:
    """构造一份全新状态。深拷贝默认值，避免可变默认值串味。"""
    state: dict[str, Any] = json.loads(json.dumps(STATE_DEFAULTS))
    state.update(
        run_id=run_id,
        direction=direction,
        venue=venue,
        language=language,
        seed=seed,
    )
    state["stage_status"] = {s: STATUS_PENDING for s in STAGES}
    state["stage_attempts"] = {s: 0 for s in STAGES}
    state["stage_errors"] = {s: [] for s in STAGES}
    return state


def state_schema() -> dict[str, Any]:
    """供 ``cli stages`` / 文档使用的可读 schema。"""
    return {
        "stages": list(STAGES),
        "stage_titles": dict(STAGE_TITLES),
        "keys": {k: type(v).__name__ for k, v in STATE_DEFAULTS.items()},
    }


def to_jsonable(obj: Any) -> Any:
    """把状态里可能混进来的非 JSON 对象降级成字符串。

    引擎在 checkpoint 之前调用它，宁可丢精度也不能让整份状态写不出去。
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    for attr in ("to_dict", "as_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return to_jsonable(fn())
            except Exception:  # pragma: no cover - 降级路径
                break
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


def record_artifact(state: dict[str, Any], artifact: Artifact) -> None:
    """把产物登记进状态，按 ``path`` 去重（后者覆盖前者，保留最新哈希）。"""
    entries: list[dict[str, Any]] = state.setdefault("artifacts", [])
    payload = artifact.to_dict()
    for i, existing in enumerate(entries):
        if existing.get("path") == payload["path"]:
            entries[i] = payload
            return
    entries.append(payload)


def artifacts_of(state: dict[str, Any], stage: str | None = None) -> list[Artifact]:
    out = []
    for d in state.get("artifacts", []):
        if stage and d.get("stage") != stage:
            continue
        try:
            out.append(Artifact.from_dict(d))
        except TypeError:  # pragma: no cover - 畸形条目直接跳过
            continue
    return out


def mark_stage(
    state: dict[str, Any], stage: str, status: str, error: str | None = None
) -> None:
    state.setdefault("stage_status", {})[stage] = status
    if error:
        state.setdefault("stage_errors", {}).setdefault(stage, []).append(error)


def completed_stages(state: dict[str, Any]) -> list[str]:
    status = state.get("stage_status", {})
    return [s for s in STAGES if status.get(s) in TERMINAL_STATUSES]


def first_incomplete_stage(state: dict[str, Any]) -> str | None:
    status = state.get("stage_status", {})
    for s in STAGES:
        if status.get(s) not in TERMINAL_STATUSES:
            return s
    return None


def summarize_state(state: dict[str, Any]) -> str:
    """一行摘要，供日志与 CLI ``status`` 使用。"""
    done = len(completed_stages(state))
    artifacts = len(state.get("artifacts", []))
    failed = [s for s, st in state.get("stage_status", {}).items() if st == STATUS_FAILED]
    parts = [f"{done}/{len(STAGES)} stages", f"{artifacts} artifacts"]
    if state.get("selected_idea"):
        parts.append(f"idea={state['selected_idea'].get('id', '?')}")
    if state.get("review_score"):
        parts.append(f"score={state['review_score']}")
    if failed:
        parts.append("failed=" + ",".join(failed))
    return " | ".join(parts)
