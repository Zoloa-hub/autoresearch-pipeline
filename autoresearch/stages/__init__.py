"""阶段注册表。

``STAGE_CLASSES`` 的顺序即默认执行顺序（与 ``graph.state.STAGES`` 一致）。
新增阶段只需在这里追加并同步 ``STAGES``/``STATE_DEFAULTS``。
"""

from __future__ import annotations

from .base import Stage, StageResult
from .s1_literature import LiteratureStage
from .s2_ideation import IdeationStage
from .s3_planning import PlanningStage
from .s4_experiment import ExperimentStage
from .s5_analysis import AnalysisStage
from .s6_writing import SECTION_PLAN, WritingStage
from .s7_compile import CompileStage
from .s8_review import ACCEPT_SCORE, ACCEPT_VERDICTS, PLATEAU_DELTA, ReviewStage
from .s9_finalize import FinalizeStage

STAGE_CLASSES: tuple[type[Stage], ...] = (
    LiteratureStage,
    IdeationStage,
    PlanningStage,
    ExperimentStage,
    AnalysisStage,
    WritingStage,
    CompileStage,
    ReviewStage,
    FinalizeStage,
)

STAGE_REGISTRY: dict[str, type[Stage]] = {cls.name: cls for cls in STAGE_CLASSES}

#: 允许失败但仍继续（可选）的阶段。实验与编译失败时管线仍应产出一份诚实的报告。
OPTIONAL_STAGES: frozenset[str] = frozenset({"s5_analysis", "s7_compile"})

__all__ = [
    "ACCEPT_SCORE",
    "ACCEPT_VERDICTS",
    "OPTIONAL_STAGES",
    "PLATEAU_DELTA",
    "SECTION_PLAN",
    "STAGE_CLASSES",
    "STAGE_REGISTRY",
    "AnalysisStage",
    "CompileStage",
    "ExperimentStage",
    "FinalizeStage",
    "IdeationStage",
    "LiteratureStage",
    "PlanningStage",
    "ReviewStage",
    "Stage",
    "StageResult",
    "WritingStage",
]
