"""多参数扫描：把"一条参数轴"变成"多参数实验设计"，并做效应分析。

## 为什么需要它

材料学最自然的实验形态就是参数扫描：温度 × 成分 × 退火时间。此前变体是
"一组固定参数"（``baseline`` / ``method`` / ``abl-<n>``），无法表达
"沿某条轴走一遍"或"在多参数网格上找最优格点"。

## 两种设计，语义不同，不能混用

* **OFAT**（one-factor-at-a-time）：从参考点出发，一次只动一条轴。
  运行数 ``1 + Σ(kᵢ − 1)``。便宜，但**原理上无法发现交互效应**——
  若"高温 + 高成分"才有效、单独调任一轴都看不出来，OFAT 会得出"两者都不重要"
  的**错误结论**。
* **网格**（full factorial）：所有轴取值全组合。运行数 ``Π kᵢ``。
  能估主效应**与交互效应**，但组合爆炸：5×4×3 = 60 组。

所以本模块**要求调用方显式选**，并在预算不足时如实报告——
而不是默默挑一种或悄悄截断。

## 截断网格会破坏正交性（重要）

如果网格超出运行预算而被截断，**剩下的格子不再是正交设计**：
主效应会与交互效应混淆，估出来的"某轴重要"可能只是采样不均造成的。
这不会有任何报错，只会给出看起来正常的错误结论。

因此 :class:`SweepPlan` 在截断时置 ``orthogonal=False``，
:func:`axis_effects` 会把这一点带进结果（``confounded=True``），
调用方可以据此拒绝出结论。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: 默认运行预算。超过即触发截断报告，而不是静默跑爆。
DEFAULT_MAX_RUNS = 24

#: 支持的设计模式
MODES = ("ofat", "grid")


@dataclass(frozen=True)
class SweepAxis:
    """一条可扫描的参数轴：名字 + 取值序列（+ 可选单位）。"""

    name: str
    values: tuple[Any, ...]
    unit: str = ""
    #: 该参数在**被驱动模块**里的绑定方式。空 = 与 ``name`` 同名的模块属性。
    #:
    #: 格式：``"N_VIS"``（普通属性，直接 setattr）或
    #: ``"eps_lorentz:g"``（该参数是函数 ``eps_lorentz`` 的默认参数——
    #: 这种绑定在函数**定义时**就固定了，setattr 模块常量无效，必须重建函数）。
    #:
    #: 之所以必须显式：绑定方式只有适配器知道，而**猜错的失败是静默的**——
    #: 所有格点算出同一组数值，扫描退化成重复实验，效应全为 0，
    #: 然后被解读成「这些参数都不重要」。
    target: str = ""

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("SweepAxis.name 不能为空")
        if len(self.values) < 2:
            raise ValueError(
                f"轴 {self.name!r} 至少有 2 个取值才有扫描意义（当前 {len(self.values)}）"
            )

    @property
    def levels(self) -> int:
        return len(self.values)

    def label(self, value: Any) -> str:
        """人类可读的取值标签（带单位）。"""
        return f"{value}{self.unit}" if self.unit else str(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": list(self.values),
            "unit": self.unit,
            "levels": self.levels,
            "target": self.target or self.name,
        }


@dataclass
class SweepPlan:
    """扫描方案：要跑哪些格点，以及**设计是否完整**。"""

    mode: str
    axes: tuple[SweepAxis, ...]
    cells: list[dict[str, Any]] = field(default_factory=list)
    total_possible: int = 0
    reference: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    dropped: int = 0
    #: 截断过的网格**不是**正交设计——主效应会与交互效应混淆。
    orthogonal: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.cells)

    def describe(self) -> str:
        axes = ", ".join(f"{a.name}({a.levels})" for a in self.axes)
        head = f"{self.mode} 设计：{axes}"
        if self.mode == "grid":
            head += f" → {math.prod(a.levels for a in self.axes)} 组"
        else:
            head += f" → {self.total_possible} 组"
        head += f"；实际 {self.n_runs} 组"
        if self.truncated:
            head += f"（截断 {self.dropped} 组，非正交）"
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "axes": [a.to_dict() for a in self.axes],
            "cells": [dict(c) for c in self.cells],
            "n_runs": self.n_runs,
            "total_possible": self.total_possible,
            "reference": dict(self.reference),
            "truncated": self.truncated,
            "dropped": self.dropped,
            "orthogonal": self.orthogonal,
            "notes": list(self.notes),
            "summary": self.describe(),
        }


# --------------------------------------------------------------------------- #
# 展开
# --------------------------------------------------------------------------- #
def expand_sweep(
    axes: Sequence[SweepAxis],
    mode: str = "ofat",
    max_runs: int | None = DEFAULT_MAX_RUNS,
    reference: Mapping[str, Any] | None = None,
) -> SweepPlan:
    """把参数轴展开成要跑的格点。

    ``mode="ofat"``：从参考点（默认每条轴的**第一个**取值）出发，一次只动一条轴。
    ``mode="grid"``：全组合。

    ``max_runs``：``None`` = 不限；整数 = 运行数上限。**超限时截断并如实标记**
    （``truncated=True`` 且 ``orthogonal=False``），而不是静默跑爆或静默少跑。

    截断顺序：网格模式按字典序，因此**低取值组合优先**。只要发生截断，
    结果就不再正交，必须把 ``orthogonal=False`` 一路带到结论里。
    """
    axes = tuple(axes)
    if not axes:
        raise ValueError("至少需要一条参数轴")
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，得到 {mode!r}")
    names = [a.name for a in axes]
    if len(set(names)) != len(names):
        raise ValueError(f"轴名重复: {names}")

    ref: dict[str, Any] = {}
    for axis in axes:
        if reference and axis.name in reference:
            value = reference[axis.name]
            if value not in axis.values:
                raise ValueError(
                    f"参考点 {axis.name}={value!r} 不在该轴取值 {list(axis.values)} 内"
                )
            ref[axis.name] = value
        else:
            ref[axis.name] = axis.values[0]

    notes: list[str] = []
    if mode == "grid":
        total = math.prod(a.levels for a in axes)
        cells = [
            dict(zip(names, combo))
            for combo in itertools.product(*(a.values for a in axes))
        ]
    else:
        # 参考点本身 + 每条轴各自偏离参考点的其余取值
        cells = [dict(ref)]
        for axis in axes:
            for value in axis.values:
                if value == ref[axis.name]:
                    continue
                cell = dict(ref)
                cell[axis.name] = value
                cells.append(cell)
        total = len(cells)

    truncated = False
    dropped = 0
    orthogonal = True
    if max_runs is not None and len(cells) > max_runs:
        dropped = len(cells) - max_runs
        cells = cells[:max_runs]
        truncated = True
        orthogonal = False
        notes.append(
            f"运行数 {total} 超出预算 {max_runs}，已截断 {dropped} 组。"
            "**截断后的网格不再正交**：主效应会与交互效应混淆，"
            "不能据此断定「哪条轴重要」——那可能只是采样不均。"
        )
    if mode == "ofat":
        notes.append(
            "OFAT 设计**原理上无法发现交互效应**。若两条轴只在特定组合下才有效，"
            "OFAT 会得出「两条轴都不重要」的错误结论。要检测交互请用 mode='grid'。"
        )
    return SweepPlan(
        mode=mode,
        axes=axes,
        cells=cells,
        total_possible=total,
        reference=ref,
        truncated=truncated,
        dropped=dropped,
        orthogonal=orthogonal,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# 效应分析
# --------------------------------------------------------------------------- #
@dataclass
class AxisEffect:
    """一条轴的主效应。"""

    axis: str
    unit: str
    #: 每个取值上的均值
    level_means: dict[str, float]
    #: 效应幅度 = 各取值均值的极差
    span: float
    #: 均值最优的取值
    best_level: Any
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "unit": self.unit,
            "level_means": dict(self.level_means),
            "span": self.span,
            "best_level": self.best_level,
            "n_obs": self.n_obs,
        }


@dataclass
class InteractionEffect:
    """两轴之间的交互强度。

    定义：在 B 的每个取值上分别算 A 的效应，这些效应之间的极差就是交互强度。
    若接近 0，说明 A 的作用不依赖于 B（可加性成立）。
    """

    axis_a: str
    axis_b: str
    #: 交互强度 = A 的效应随 B 取值的极差
    strength: float
    #: 每个 B 取值下 A 的效应
    effect_of_a_by_b: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_a": self.axis_a,
            "axis_b": self.axis_b,
            "strength": self.strength,
            "effect_of_a_by_b": dict(self.effect_of_a_by_b),
        }


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not _isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _isnan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _key(value: Any) -> str:
    """把取值规范成 dict 键（可以是数字或字符串）。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def axis_effects(
    observations: Sequence[tuple[Mapping[str, Any], float]],
    axes: Sequence[SweepAxis],
    *,
    orthogonal: bool = True,
) -> list[AxisEffect]:
    """算每条轴的主效应。

    ``observations``：``[(该次运行的参数, 指标值), ...]``。
    缺失某条轴取值的观测会被该轴忽略（各轴独立统计）。

    ``orthogonal=False`` 时**不阻止**计算，但调用方必须知道这些主效应
    可能是混淆的——返回值里没有标记，请把 ``SweepPlan.orthogonal`` 一起传下去。
    """
    out: list[AxisEffect] = []
    for axis in axes:
        buckets: dict[str, list[float]] = {}
        level_of_key: dict[str, Any] = {}
        for params, value in observations:
            if axis.name not in params:
                continue
            k = _key(params[axis.name])
            level_of_key[k] = params[axis.name]
            buckets.setdefault(k, []).append(float(value))
        if len(buckets) < 2:
            continue
        means = {k: _mean(v) for k, v in buckets.items()}
        finite = {k: v for k, v in means.items() if not _isnan(v)}
        if len(finite) < 2:
            continue
        best_key = max(finite, key=lambda k: finite[k])
        out.append(
            AxisEffect(
                axis=axis.name,
                unit=axis.unit,
                level_means={k: round(v, 6) for k, v in means.items()},
                span=round(max(finite.values()) - min(finite.values()), 6),
                best_level=level_of_key[best_key],
                n_obs=sum(len(v) for v in buckets.values()),
            )
        )
    return out


def interaction_effects(
    observations: Sequence[tuple[Mapping[str, Any], float]],
    axes: Sequence[SweepAxis],
    *,
    max_pairs: int = 6,
) -> list[InteractionEffect]:
    """算两两轴之间的交互强度（需要网格设计才有意义）。"""
    pairs = list(itertools.combinations(axes, 2))
    out: list[InteractionEffect] = []
    for axis_a, axis_b in pairs[:max_pairs]:
        by_b: dict[str, dict[str, list[float]]] = {}
        level_of: dict[str, Any] = {}
        for params, value in observations:
            if axis_a.name not in params or axis_b.name not in params:
                continue
            kb = _key(params[axis_b.name])
            level_of[kb] = params[axis_b.name]
            by_b.setdefault(kb, {}).setdefault(_key(params[axis_a.name]), []).append(
                float(value)
            )
        if len(by_b) < 2:
            continue
        effect_by_b: dict[str, float] = {}
        for kb, buckets in by_b.items():
            means = [_mean(v) for v in buckets.values()]
            finite = [m for m in means if not _isnan(m)]
            if len(finite) >= 2:
                effect_by_b[kb] = round(max(finite) - min(finite), 6)
        if len(effect_by_b) < 2:
            continue
        strength = round(max(effect_by_b.values()) - min(effect_by_b.values()), 6)
        out.append(
            InteractionEffect(
                axis_a=axis_a.name,
                axis_b=axis_b.name,
                strength=strength,
                effect_of_a_by_b=effect_by_b,
            )
        )
    out.sort(key=lambda e: e.strength, reverse=True)
    return out


def rank_axes(effects: Sequence[AxisEffect]) -> list[AxisEffect]:
    """按效应幅度排序 —— "哪条轴最关键"的直接依据。"""
    return sorted(effects, key=lambda e: e.span, reverse=True)


def effect_ratio(effects: Sequence[AxisEffect]) -> dict[str, float]:
    """各轴效应幅度相对最强轴的比例，便于写"X 的作用是 Y 的 N 倍"。"""
    if not effects:
        return {}
    top = max(e.span for e in effects)
    if top <= 0:
        return {e.axis: 0.0 for e in effects}
    return {e.axis: round(e.span / top, 4) for e in effects}
