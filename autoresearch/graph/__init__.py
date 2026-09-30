"""图执行层。"""

from __future__ import annotations

from .checkpoint import (
    Checkpointer,
    CheckpointMeta,
    find_run_dir,
    list_runs,
    load_run_state,
    merge_resume,
    resume_plan,
)
from .engine import END, GraphEngine, GraphError, Node, linear_nodes
from .state import (
    STAGES,
    STAGE_ORDER,
    STAGE_TITLES,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    Artifact,
    artifacts_of,
    completed_stages,
    first_incomplete_stage,
    hash_file,
    mark_stage,
    new_state,
    record_artifact,
    state_schema,
    summarize_state,
    to_jsonable,
)

__all__ = [
    "STAGES",
    "STAGE_ORDER",
    "STAGE_TITLES",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_SKIPPED",
    "Artifact",
    "Checkpointer",
    "CheckpointMeta",
    "END",
    "GraphEngine",
    "GraphError",
    "Node",
    "artifacts_of",
    "completed_stages",
    "find_run_dir",
    "first_incomplete_stage",
    "hash_file",
    "linear_nodes",
    "list_runs",
    "load_run_state",
    "mark_stage",
    "merge_resume",
    "new_state",
    "record_artifact",
    "resume_plan",
    "state_schema",
    "summarize_state",
    "to_jsonable",
]


def __getattr__(name: str):  # pragma: no cover - 可选依赖延迟暴露
    if name in {"compile_langgraph", "langgraph_available", "describe_parity"}:
        from . import langgraph_adapter

        return getattr(langgraph_adapter, name)
    raise AttributeError(name)
