# -*- coding: utf-8 -*-
"""Publication figures for the Auto-Research pipeline.

Implements ``CONTRACTS.md`` section 8 (FROZEN interface):

    setup_style(venue="NeurIPS") -> None
    plot_learning_curves(series, out_dir, name, formats) -> list[Path]
    plot_bar_comparison(summary, metric, out_dir, name, formats) -> list[Path]
    plot_ablation(df, metric, out_dir, name, formats) -> list[Path]
    plot_boxplot(runs, metric, out_dir, name, formats) -> list[Path]
    plot_metric_grid(series_map, out_dir, name, formats) -> list[Path]
    make_all_figures(metrics_dir, out_dir, formats) -> dict[str, list[Path]]

Rules honoured here
-------------------
* ``matplotlib.use("Agg")`` is called *before* importing ``pyplot`` and
  ``plt.show()`` is never called.
* Figures are always saved with ``bbox_inches="tight"`` and closed in a
  ``finally`` block, so no figure leaks between calls.
* CJK fonts: :data:`CJK_AVAILABLE` is resolved at import time from
  ``Microsoft YaHei / SimHei / Noto Sans CJK SC / Source Han Sans SC /
  Arial Unicode MS`` (plus a few well known fallbacks).  When a CJK font is
  found it is registered and used; otherwise :data:`CJK_AVAILABLE` stays
  ``False`` so callers can emit English-only labels instead of tofu boxes.
* Plotting problems are logged and skipped, never raised, except for genuine
  programmer errors (e.g. a non-dict ``series``) which raise ``TypeError``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # must happen before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

LOG = logging.getLogger(__name__)

__all__ = [
    "setup_style",
    "figsize",
    "plot_learning_curves",
    "plot_bar_comparison",
    "plot_ablation",
    "plot_boxplot",
    "plot_metric_grid",
    "make_all_figures",
    "PALETTE",
    "COLUMN_WIDTH",
    "TEXT_WIDTH",
    "CJK_AVAILABLE",
    "CJK_FONT",
]

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

#: Okabe-Ito colourblind safe palette.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
)

#: single column width (inches) -- NeurIPS / ICLR / ICML / CVPR / ICCV / ACL
COLUMN_WIDTH: float = 3.25
#: full text width (inches)
TEXT_WIDTH: float = 6.75

#: per-venue single column width (all listed venues use the same 3.25in column)
VENUE_COLUMN_WIDTH: dict[str, float] = {
    "neurips": 3.25,
    "nips": 3.25,
    "iclr": 3.25,
    "icml": 3.25,
    "cvpr": 3.25,
    "iccv": 3.25,
    "eccv": 3.25,
    "wacv": 3.25,
    "acl": 3.25,
    "emnlp": 3.25,
    "naacl": 3.25,
    "aaai": 3.25,
    "ijcai": 3.25,
}

DEFAULT_FORMATS: tuple[str, ...] = ("pdf", "png")
DEFAULT_DPI: int = 300
PREVIEW_DPI: int = 150

#: candidate CJK fonts, most preferred first
CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
    # widely available fallbacks (harmless when absent)
    "Microsoft YaHei UI",
    "Microsoft JhengHei",
    "Noto Sans SC",
    "Noto Sans CJK JP",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "Hiragino Sans GB",
    "SimSun",
    "NSimSun",
    "FangSong",
    "KaiTi",
)

#: x-axis keys: these are axes, not metrics (they are not drawn as panels)
X_KEYS: tuple[str, ...] = ("epoch", "epochs", "step", "steps", "iter", "iters", "iteration", "global_step")

_VENUE_FONTSIZE = {
    "neurips": 10.0,
    "nips": 10.0,
    "iclr": 10.0,
    "icml": 10.0,
    "cvpr": 8.0,
    "iccv": 8.0,
    "eccv": 8.0,
    "wacv": 8.0,
    "acl": 9.0,
    "emnlp": 9.0,
    "naacl": 9.0,
    "aaai": 9.0,
    "ijcai": 9.0,
}

_GRID_MAX_ROWS = 6
_GRID_MAX_COLS = 6
_BAR_METRIC_LIMIT = 6

# --------------------------------------------------------------------------- #
# CJK font detection
# --------------------------------------------------------------------------- #

CJK_FONT: str | None = None
#: ``True`` once a CJK capable font has been found and registered.
CJK_AVAILABLE: bool = False
_STYLE_APPLIED: bool = False


def _detect_cjk_font() -> str | None:
    """Return the name of an installed CJK font, or ``None``.

    The font is registered with matplotlib's font manager so that it can be
    selected by name.  Never raises.
    """
    global CJK_FONT, CJK_AVAILABLE
    try:
        installed = {entry.name for entry in font_manager.fontManager.ttflist}
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("font scan failed: %s", exc)
        installed = set()
        CJK_FONT, CJK_AVAILABLE = None, False
        return None

    for candidate in CJK_FONT_CANDIDATES:
        if candidate in installed:
            CJK_FONT, CJK_AVAILABLE = candidate, True
            return candidate

    # last resort: match on a normalised name (case / spacing differences)
    wanted = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in CJK_FONT_CANDIDATES}
    for entry in font_manager.fontManager.ttflist:
        key = re.sub(r"[^a-z0-9]", "", str(entry.name).lower())
        if key in wanted:
            try:
                font_manager.fontManager.addfont(entry.fname)
            except Exception:  # pragma: no cover - defensive
                pass
            CJK_FONT, CJK_AVAILABLE = entry.name, True
            return entry.name

    CJK_FONT, CJK_AVAILABLE = None, False
    return None


def _register_cjk_font(font_name: str) -> None:
    for entry in font_manager.fontManager.ttflist:
        if entry.name == font_name:
            try:
                font_manager.fontManager.addfont(entry.fname)
            except Exception:  # pragma: no cover - defensive
                pass
            return


_detect_cjk_font()
if not CJK_AVAILABLE:
    LOG.info("no CJK font found; figures fall back to English-only labels")


def setup_style(venue: str = "NeurIPS") -> None:
    """Configure matplotlib rcParams for a paper figure at *venue*.

    Idempotent and safe to call repeatedly.  When no CJK font is available
    :data:`CJK_AVAILABLE` is ``False`` and callers should avoid Chinese labels.
    """
    global _STYLE_APPLIED

    _detect_cjk_font()
    key = str(venue or "").strip().lower()
    font_size = _VENUE_FONTSIZE.get(key, 9.0)
    column_width = VENUE_COLUMN_WIDTH.get(key, COLUMN_WIDTH)

    rc = matplotlib.rcParams
    rc["figure.dpi"] = PREVIEW_DPI
    rc["savefig.dpi"] = DEFAULT_DPI
    rc["savefig.bbox"] = "tight"
    rc["figure.autolayout"] = False
    rc["figure.figsize"] = (column_width, 2.4)
    rc["figure.titlesize"] = font_size + 1
    rc["font.size"] = font_size
    rc["axes.titlesize"] = font_size
    rc["axes.labelsize"] = font_size
    rc["axes.grid"] = True
    rc["axes.grid.axis"] = "both"
    rc["grid.alpha"] = 0.3
    rc["grid.linestyle"] = ":"
    rc["grid.linewidth"] = 0.6
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.linewidth"] = 0.8
    rc["axes.prop_cycle"] = matplotlib.cycler(color=list(PALETTE))
    rc["lines.linewidth"] = 1.5
    rc["lines.markersize"] = 3.5
    rc["legend.frameon"] = False
    rc["legend.fontsize"] = max(6.0, font_size - 1.0)
    rc["xtick.labelsize"] = max(6.0, font_size - 1.0)
    rc["ytick.labelsize"] = max(6.0, font_size - 1.0)
    rc["xtick.direction"] = "out"
    rc["ytick.direction"] = "out"
    rc["savefig.pad_inches"] = 0.02
    rc["pdf.fonttype"] = 42
    rc["ps.fonttype"] = 42

    if CJK_AVAILABLE and CJK_FONT:
        _register_cjk_font(CJK_FONT)
        rc["font.family"] = "sans-serif"
        rc["font.sans-serif"] = [CJK_FONT, "DejaVu Sans", "Arial", "sans-serif"]
        rc["axes.unicode_minus"] = False
    else:
        rc["font.family"] = "sans-serif"
        rc["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
        rc["axes.unicode_minus"] = True

    _STYLE_APPLIED = True


def figsize(width: str | float = "single", height: float = 2.4) -> tuple[float, float]:
    """``figsize("single"|"double", height)``; a number is taken as inches."""
    if isinstance(width, (int, float)) and not isinstance(width, bool):
        return (float(width), float(height))
    key = str(width or "single").strip().lower()
    if key in ("double", "full", "text", "2", "two"):
        return (TEXT_WIDTH, float(height))
    return (COLUMN_WIDTH, float(height))


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #


def _ensure_style() -> None:
    if not _STYLE_APPLIED:
        setup_style()


def _normalize_formats(formats: Sequence[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for fmt in formats or ():
        text = str(fmt).strip().lower().lstrip(".")
        if text:
            out.append(text)
    return tuple(out) or DEFAULT_FORMATS


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_floats(values: Any) -> list[float]:
    """Coerce to a list of floats; non numeric entries become ``nan``."""
    if isinstance(values, Mapping):
        for key in ("mean", "value", "y", "values"):
            if key in values:
                return _as_floats(values[key])
        return []
    if isinstance(values, (str, bytes)):
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        num = _to_float(values)
        return [] if num is None else [num]
    out: list[float] = []
    for item in values:
        num = _to_float(item)
        out.append(float("nan") if num is None else num)
    return out


def _finite(values: Sequence[float]) -> list[float]:
    return [v for v in values if not math.isnan(v) and not math.isinf(v)]


def _is_std_mapping(value: Any) -> bool:
    """``{"mean": [...], "std": [...]}`` series shape."""
    return (
        isinstance(value, Mapping)
        and "mean" in value
        and not isinstance(value["mean"], Mapping)
    )


def _smooth(values: Sequence[float], alpha: float) -> list[float]:
    """Exponential moving average; ``alpha <= 0`` disables smoothing."""
    if alpha <= 0:
        return list(values)
    alpha = min(1.0, float(alpha))
    out: list[float] = []
    last: float | None = None
    for value in values:
        if last is None or math.isnan(last):
            last = value
        else:
            last = alpha * value + (1.0 - alpha) * last
        out.append(last)
    return out


def _safe_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._\-]+", "_", str(name)).strip("_")
    return text or "figure"


def _save_figure(fig, out_dir, name: str, formats: Sequence[str]) -> list[Path]:
    """Save *fig* in every requested format; returns the written paths."""
    paths: list[Path] = []
    try:
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOG.warning("cannot create figure directory %s: %s", out_dir, exc)
        return paths
    for fmt in _normalize_formats(formats):
        path = target / f"{_safe_name(name)}.{fmt}"
        try:
            fig.savefig(path, format=fmt, dpi=DEFAULT_DPI, bbox_inches="tight")
        except Exception as exc:
            LOG.warning("savefig failed for %s (%s): %s", path, fmt, exc)
            continue
        if path.exists() and path.stat().st_size > 0:
            paths.append(path)
        else:
            LOG.warning("figure %s was not written", path)
    return paths


def _normalize_run(metrics: Mapping[str, Any]) -> tuple[dict[str, tuple[list[float], list[float] | None]], list[float] | None, str | None]:
    """Split a run's metric dict into panels plus the shared x axis."""
    x_key = None
    for candidate in X_KEYS:
        if candidate in metrics:
            x_key = candidate
            break
    xs = _as_floats(metrics[x_key]) if x_key else None
    if xs is not None and not _finite(xs):
        xs = None

    panels: dict[str, tuple[list[float], list[float] | None]] = {}
    for metric, value in metrics.items():
        if metric == x_key:
            continue
        if _is_std_mapping(value):
            ys = _as_floats(value.get("mean"))
            std = _as_floats(value.get("std"))
            std = std if std and len(std) == len(ys) and _finite(std) else None
        else:
            ys = _as_floats(value)
            std = None
        if ys and _finite(ys):
            panels[str(metric)] = (ys, std)
    return panels, xs, x_key


# --------------------------------------------------------------------------- #
# learning curves
# --------------------------------------------------------------------------- #


def plot_learning_curves(
    series: dict[str, dict[str, list[float]]],
    out_dir,
    name: str = "learning_curves",
    formats=("pdf", "png"),
    *,
    smooth: float = 0.0,
) -> list[Path]:
    """Multi-panel learning curves (one panel per metric, at most 2 columns).

    ``series`` is ``dict[run][metric] -> list[float]``.  The
    ``dict[run][metric] -> {"mean": [...], "std": [...]}`` shape is detected at
    runtime and drawn with a shaded ``+/- std`` band.  ``epoch`` / ``step`` /
    ``iter`` keys are used as the x axis when present, otherwise ``1..N``.
    ``smooth`` is an EMA factor (0.0 = raw curves, the default).
    """
    if not isinstance(series, Mapping):
        raise TypeError(
            "plot_learning_curves(series): expected dict[run_name, dict[metric, list[float]]]"
        )
    _ensure_style()

    normalized: dict[str, tuple[dict[str, tuple[list[float], list[float] | None]], list[float] | None, str | None]] = {}
    for run, metrics in series.items():
        if not isinstance(metrics, Mapping):
            LOG.warning("plot_learning_curves: skipping run %r (not a mapping)", run)
            continue
        panels, xs, x_key = _normalize_run(metrics)
        if panels:
            normalized[str(run)] = (panels, xs, x_key)
    if not normalized:
        LOG.warning("plot_learning_curves: no plottable series")
        return []

    metrics: list[str] = []
    for panels, _xs, _xk in normalized.values():
        for metric in panels:
            if metric not in metrics:
                metrics.append(metric)
    if not metrics:
        LOG.warning("plot_learning_curves: no metrics besides the x axis")
        return []

    x_label = "Epoch"
    for _panels, _xs, x_key in normalized.values():
        if x_key:
            x_label = x_key.replace("_", " ").capitalize()
            break

    runs = list(normalized)
    colors = {run: PALETTE[i % len(PALETTE)] for i, run in enumerate(runs)}
    ncols = 2 if len(metrics) > 1 else 1
    nrows = int(math.ceil(len(metrics) / ncols))
    width = TEXT_WIDTH if ncols == 2 else COLUMN_WIDTH
    height = 1.75 * nrows + 0.55

    fig = None
    try:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(width, height),
            sharex=True,
            squeeze=False,
        )
        flat = axes.ravel()
        for index, metric in enumerate(metrics):
            ax = flat[index]
            for run in runs:
                panels, xs, _xk = normalized[run]
                if metric not in panels:
                    continue
                ys, std = panels[metric]
                color = colors[run]
                x_values = xs if xs is not None and len(xs) == len(ys) else list(range(1, len(ys) + 1))
                line_y = _smooth(ys, smooth)
                ax.plot(x_values, line_y, color=color, label=run, linewidth=1.5)
                if std is not None:
                    lower = np.asarray(line_y, dtype=float) - np.asarray(std, dtype=float)
                    upper = np.asarray(line_y, dtype=float) + np.asarray(std, dtype=float)
                    ax.fill_between(x_values, lower, upper, color=color, alpha=0.18, linewidth=0)
            values = _finite([v for run in runs if metric in normalized[run][0] for v in normalized[run][0][metric][0]])
            positive = [abs(v) for v in values if v != 0]
            if positive and max(positive) / max(min(positive), 1e-300) > 1e3:
                try:
                    ax.set_yscale("log")
                except Exception as exc:  # pragma: no cover - defensive
                    LOG.debug("log scale failed for %s: %s", metric, exc)
            ax.set_title(metric, fontsize=matplotlib.rcParams["axes.titlesize"])
            ax.set_ylabel(metric)
            if index >= len(metrics) - ncols:
                ax.set_xlabel(x_label)
        for index in range(len(metrics), len(flat)):
            flat[index].axis("off")

        handles, labels = flat[0].get_legend_handles_labels()
        if labels:
            if len(metrics) == 1:
                flat[0].legend(loc="best")
                fig.subplots_adjust(top=0.86, bottom=0.20)
            else:
                # reserve a band under the x labels for a figure level legend
                fig.subplots_adjust(hspace=0.45, wspace=0.28, top=0.88, bottom=0.30)
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.17),
                    ncol=min(len(labels), 4),
                )
        else:
            fig.subplots_adjust(top=0.88)
        fig.suptitle(name.replace("_", " "), fontsize=matplotlib.rcParams["figure.titlesize"], y=0.99)
        return _save_figure(fig, out_dir, name, formats)
    except Exception as exc:
        LOG.warning("plot_learning_curves failed: %s", exc)
        return []
    finally:
        if fig is not None:
            plt.close(fig)


# --------------------------------------------------------------------------- #
# bar comparison
# --------------------------------------------------------------------------- #


def _looks_like_stats(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        key in value for key in ("mean", "best", "final", "count", "min", "max")
    )


def _metric_bars(summary: Mapping[str, Any], metric: str):
    """Return ``(labels, means, stds)`` for *metric* across runs, or ``None``.

    Accepts both summary shapes:

    * run major -- ``summary[run][metric] = {"mean": .., "std": ..}`` (what
      :func:`autoresearch.tools.metrics.summarize` returns per run, and what
      ``make_all_figures`` passes);
    * metric major -- ``summary[metric] = {"mean": .., ...}`` (the plain
      ``summarize`` output, rendered as a single bar);
    * metric major with nested runs -- ``summary[metric][run] = {"mean": ..}``.
    """
    if not isinstance(summary, Mapping):
        return None
    entry = summary.get(metric)
    if isinstance(entry, Mapping):
        if _looks_like_stats(entry):
            return [metric], [_to_float(entry.get("mean")) or 0.0], [_to_float(entry.get("std")) or 0.0]
        nested = {str(k): v for k, v in entry.items() if _looks_like_stats(v)}
        if nested:
            labels = list(nested)
            means = [float(_to_float(nested[k].get("mean")) or 0.0) for k in labels]
            stds = [float(_to_float(nested[k].get("std")) or 0.0) for k in labels]
            return labels, means, stds

    labels, means, stds = [], [], []
    for run, metrics in summary.items():
        if not isinstance(metrics, Mapping):
            continue
        stats = metrics.get(metric)
        if not _looks_like_stats(stats):
            continue
        mean = _to_float(stats.get("mean"))
        if mean is None or math.isnan(mean):
            continue
        labels.append(str(run))
        means.append(float(mean))
        stds.append(float(_to_float(stats.get("std")) or 0.0))
    if not labels:
        return None
    return labels, means, stds


def plot_bar_comparison(
    summary: dict[str, dict[str, float]],
    metric: str,
    out_dir,
    name: str = "comparison",
    formats=("pdf", "png"),
) -> list[Path]:
    """Grouped bars for one *metric* across runs, with ``std`` error bars."""
    if not isinstance(summary, Mapping):
        raise TypeError("plot_bar_comparison(summary): expected a mapping of run/metric -> stats")
    _ensure_style()

    bars = _metric_bars(summary, metric)
    if bars is None:
        LOG.warning("plot_bar_comparison: metric %r not found in summary", metric)
        return []
    labels, means, stds = bars
    if not labels:
        return []

    best_index = None
    try:
        from .metrics import _higher_is_better  # shared direction heuristic

        best_index = (
            int(np.argmax(means)) if _higher_is_better(metric) else int(np.argmin(means))
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug("best bar detection failed: %s", exc)
        best_index = int(np.argmax(means))

    width = TEXT_WIDTH if len(labels) > 4 else COLUMN_WIDTH
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(width, 2.6))
        positions = np.arange(len(labels))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
        bars_artist = ax.bar(
            positions,
            means,
            yerr=stds if any(s > 0 for s in stds) else None,
            capsize=2.5,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 0.8, "ecolor": "black"},
        )
        for index, (patch, value) in enumerate(zip(bars_artist, means)):
            if index == best_index:
                patch.set_hatch("//")
                patch.set_linewidth(1.1)
            span = max(abs(v) for v in means) if means else 1.0
            offset = 0.02 * (span or 1.0)
            ax.annotate(
                "%.3g" % value,
                (patch.get_x() + patch.get_width() / 2.0, value + (offset if value >= 0 else -offset)),
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=matplotlib.rcParams["xtick.labelsize"],
            )
        ax.set_xticks(positions)
        rotation = 30 if len(labels) > 5 else 0
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} comparison")
        if min(means) >= 0:
            ax.set_ylim(0, max(means) * 1.20 if max(means) > 0 else 1.0)
        fig.subplots_adjust(bottom=0.32 if rotation else 0.24)
        return _save_figure(fig, out_dir, name, formats)
    except Exception as exc:
        LOG.warning("plot_bar_comparison failed: %s", exc)
        return []
    finally:
        if fig is not None:
            plt.close(fig)


# --------------------------------------------------------------------------- #
# ablation
# --------------------------------------------------------------------------- #


def plot_ablation(
    df,
    metric: str,
    out_dir,
    name: str = "ablation",
    formats=("pdf", "png"),
) -> list[Path]:
    """Horizontal bar chart of *metric* sorted by value with deltas vs baseline."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("plot_ablation(df): expected a pandas.DataFrame")
    _ensure_style()

    from .metrics import flatten_columns

    frame = flatten_columns(df)
    column = metric if metric in frame.columns else (f"{metric}_mean" if f"{metric}_mean" in frame.columns else None)
    if column is None:
        LOG.warning("plot_ablation: column %r not found", metric)
        return []

    values = [_to_float(v) for v in frame[column].tolist()]
    labels = [str(i) for i in frame.index]
    pairs = [(label, value) for label, value in zip(labels, values) if value is not None and not math.isnan(value)]
    if not pairs:
        LOG.warning("plot_ablation: no numeric values for %r", metric)
        return []

    baseline_label, baseline_value = pairs[0]
    ordered = sorted(pairs, key=lambda item: item[1])
    names = [item[0] for item in ordered]
    numbers = [item[1] for item in ordered]

    fig = None
    try:
        height = max(2.0, 0.32 * len(names) + 1.0)
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH + 0.6, height))
        positions = np.arange(len(names))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        ax.barh(positions, numbers, color=colors, edgecolor="black", linewidth=0.6)
        span = max(abs(v) for v in numbers) or 1.0
        for index, (label, value) in enumerate(zip(names, numbers)):
            delta = value - baseline_value
            text = "baseline" if label == baseline_label else "%+.3g" % delta
            ax.annotate(
                text,
                (value, index),
                xytext=(4 if value >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=matplotlib.rcParams["xtick.labelsize"],
            )
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.set_xlabel(metric)
        ax.set_title(f"{metric} ablation (baseline: {baseline_label})")
        if min(numbers) >= 0:
            ax.set_xlim(0, max(numbers) * 1.25 if max(numbers) > 0 else 1.0)
        else:
            ax.set_xlim(min(numbers) * 1.25, max(span, max(numbers) * 1.25))
        fig.subplots_adjust(left=0.32)
        return _save_figure(fig, out_dir, name, formats)
    except Exception as exc:
        LOG.warning("plot_ablation failed: %s", exc)
        return []
    finally:
        if fig is not None:
            plt.close(fig)


# --------------------------------------------------------------------------- #
# boxplot
# --------------------------------------------------------------------------- #


def plot_boxplot(
    runs: dict[str, list[float]],
    metric: str,
    out_dir,
    name: str = "boxplot",
    formats=("pdf", "png"),
) -> list[Path]:
    """Boxplot of *metric* across runs with deterministic jittered points."""
    if not isinstance(runs, Mapping):
        raise TypeError("plot_boxplot(runs): expected dict[run_name, list[float]]")
    _ensure_style()

    labels: list[str] = []
    data: list[list[float]] = []
    for run, values in runs.items():
        clean = _finite(_as_floats(values))
        if not clean:
            continue
        labels.append(str(run))
        data.append(clean)
    if not data:
        LOG.warning("plot_boxplot: no non-empty series for %r", metric)
        return []

    seed = int(hashlib.sha1(str(metric).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    width = TEXT_WIDTH if len(labels) > 4 else COLUMN_WIDTH
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(width, 2.7))
        artists = ax.boxplot(
            data,
            notch=False,
            patch_artist=True,
            showmeans=True,
            widths=0.55,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#D55E00",
                "markeredgecolor": "black",
                "markersize": 4,
            },
            medianprops={"color": "black", "linewidth": 1.0},
            boxprops={"linewidth": 0.7},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
            flierprops={"markersize": 2.5, "markerfacecolor": "none", "markeredgecolor": "0.4"},
        )
        for index, patch in enumerate(artists["boxes"]):
            patch.set_facecolor(PALETTE[index % len(PALETTE)])
            patch.set_alpha(0.45)
        for index, values in enumerate(data, start=1):
            jitter = rng.uniform(-0.14, 0.14, size=len(values))
            ax.scatter(
                index + jitter,
                values,
                s=4,
                color="black",
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )
        ax.set_xticks(range(1, len(labels) + 1))
        rotation = 30 if len(labels) > 5 else 0
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} distribution")
        fig.subplots_adjust(bottom=0.32 if rotation else 0.24)
        return _save_figure(fig, out_dir, name, formats)
    except Exception as exc:
        LOG.warning("plot_boxplot failed: %s", exc)
        return []
    finally:
        if fig is not None:
            plt.close(fig)


# --------------------------------------------------------------------------- #
# metric grid
# --------------------------------------------------------------------------- #


def plot_metric_grid(
    series_map: dict[str, dict[str, list[float]]],
    out_dir,
    name: str = "metric_grid",
    formats=("pdf", "png"),
) -> list[Path]:
    """Small multiples: rows = metrics, columns = runs."""
    if not isinstance(series_map, Mapping):
        raise TypeError("plot_metric_grid(series_map): expected dict[run_name, dict[metric, values]]")
    _ensure_style()

    runs: list[str] = []
    panels: dict[str, dict[str, list[float]]] = {}
    metrics: list[str] = []
    for run, values in series_map.items():
        if not isinstance(values, Mapping):
            continue
        usable: dict[str, list[float]] = {}
        for metric, raw in values.items():
            if str(metric).lower() in X_KEYS:
                continue  # x axes are not metrics
            series_values = _as_floats(raw.get("mean") if _is_std_mapping(raw) else raw)
            if _finite(series_values):
                usable[str(metric)] = series_values
                if str(metric) not in metrics:
                    metrics.append(str(metric))
        if usable:
            runs.append(str(run))
            panels[str(run)] = usable
    if not runs or not metrics:
        LOG.warning("plot_metric_grid: no data (empty series_map=%s)", not series_map)
        if not series_map:
            return []

    metrics = metrics[:_GRID_MAX_ROWS]
    runs = runs[:_GRID_MAX_COLS]
    if not runs or not metrics:
        runs, metrics = ["(none)"], ["(none)"]

    fig = None
    try:
        nrows, ncols = len(metrics), len(runs)
        width = max(COLUMN_WIDTH, min(TEXT_WIDTH, 1.9 * ncols))
        height = max(1.8, 1.4 * nrows + 0.6)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(width, height), squeeze=False, sharex="col"
        )
        for row, metric in enumerate(metrics):
            for col, run in enumerate(runs):
                ax = axes[row][col]
                values = panels.get(run, {}).get(metric)
                if not values:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "n/a",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="0.5",
                        fontsize=matplotlib.rcParams["axes.labelsize"] - 1,
                    )
                    if row == 0:
                        ax.set_title(run, fontsize=matplotlib.rcParams["axes.titlesize"] - 1)
                    continue
                ax.plot(range(1, len(values) + 1), values, color=PALETTE[col % len(PALETTE)])
                if row == 0:
                    ax.set_title(run, fontsize=matplotlib.rcParams["axes.titlesize"] - 1)
                if col == 0:
                    ax.set_ylabel(metric, fontsize=matplotlib.rcParams["axes.labelsize"] - 1)
                if row == nrows - 1:
                    ax.set_xlabel("step", fontsize=matplotlib.rcParams["axes.labelsize"] - 1)
                ax.tick_params(labelsize=max(5.0, matplotlib.rcParams["xtick.labelsize"] - 1.5))
        fig.subplots_adjust(hspace=0.55, wspace=0.4)
        return _save_figure(fig, out_dir, name, formats)
    except Exception as exc:
        LOG.warning("plot_metric_grid failed: %s", exc)
        return []
    finally:
        if fig is not None:
            plt.close(fig)


# --------------------------------------------------------------------------- #
# pipeline entry point
# --------------------------------------------------------------------------- #

_DISCOVER_SUFFIXES = {".csv", ".jsonl", ".ndjson", ".json", ".log", ".tsv"}


def _discover_run_files(metrics_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    try:
        candidates = sorted(
            path
            for path in metrics_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _DISCOVER_SUFFIXES
        )
    except OSError as exc:
        LOG.warning("make_all_figures: cannot scan %s: %s", metrics_dir, exc)
        return {}
    for path in candidates:
        run = path.stem if path.parent == metrics_dir else path.parent.name
        grouped.setdefault(run, []).append(path)
    return grouped


def make_all_figures(
    metrics_dir,
    out_dir,
    formats=("pdf", "png"),
) -> dict[str, list[Path]]:
    """Discover metrics, build every figure, and dump the summary artifacts.

    Never raises: every failure is logged and skipped.  When at least one metric
    is found, ``metrics_summary.json`` (the ``summarize`` output),
    ``runs_summary.json`` (per-run ``summarize`` output) and
    ``evaluation_table.tex`` (via ``to_latex_table``) are written to *out_dir*
    and reported under ``"__summary_json__"``, ``"__runs_summary_json__"`` and
    ``"__table_tex__"``.
    """
    from .metrics import compare_runs_flat, parse_metrics, summarize, to_latex_table

    figures: dict[str, list[Path]] = {}
    try:
        source = Path(metrics_dir)
    except TypeError as exc:
        LOG.warning("make_all_figures: bad metrics_dir %r: %s", metrics_dir, exc)
        return figures

    try:
        if not source.exists() or not source.is_dir():
            LOG.warning("make_all_figures: metrics dir not found: %s", source)
            return figures

        grouped = _discover_run_files(source)
        series_map: dict[str, dict[str, list[float]]] = {}
        for run, files in grouped.items():
            merged: dict[str, list[float]] = {}
            for path in files:
                try:
                    parsed = parse_metrics(path)
                except Exception as exc:  # parse_metrics should not raise, be safe
                    LOG.warning("make_all_figures: %s failed: %s", path, exc)
                    continue
                for metric, values in parsed.items():
                    if metric in merged:
                        LOG.debug(
                            "make_all_figures: duplicate metric %r in run %r, keeping first (%s)",
                            metric,
                            run,
                            path.name,
                        )
                        continue
                    merged[metric] = values
            if merged:
                series_map[run] = merged

        if not series_map:
            LOG.warning("make_all_figures: no metrics found under %s", source)
            return figures

        # per-run summary: dict[run][metric][stat]
        run_summary = {run: summarize(metrics) for run, metrics in series_map.items()}
        run_summary = {run: stats for run, stats in run_summary.items() if stats}

        # flat summary over all runs: summarize(metric -> concatenated values)
        pooled: dict[str, list[float]] = {}
        for metrics in series_map.values():
            for metric, values in metrics.items():
                pooled.setdefault(metric, []).extend(values)
        flat_summary = summarize(pooled)

        try:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOG.warning("make_all_figures: cannot create %s: %s", out_dir, exc)
            out_path = Path(out_dir)

        if flat_summary:
            try:
                summary_path = out_path / "metrics_summary.json"
                summary_path.write_text(
                    json.dumps(flat_summary, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8",
                )
                figures["__summary_json__"] = [summary_path]
            except Exception as exc:
                LOG.warning("make_all_figures: cannot write metrics_summary.json: %s", exc)
            if run_summary:
                try:
                    runs_path = out_path / "runs_summary.json"
                    runs_path.write_text(
                        json.dumps(run_summary, indent=2, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8",
                    )
                    figures["__runs_summary_json__"] = [runs_path]
                except Exception as exc:
                    LOG.warning("make_all_figures: cannot write runs_summary.json: %s", exc)

        comparison = None
        try:
            comparison = compare_runs_flat(series_map)
        except Exception as exc:
            LOG.warning("make_all_figures: compare_runs failed: %s", exc)

        if flat_summary:
            try:
                # mean +/- std per (run, metric) -> folded into one cell by to_latex_table
                rows: list[dict] = []
                if comparison is not None and not comparison.empty:
                    for run in comparison.index:
                        row: dict[str, Any] = {"name": str(run)}
                        for column in comparison.columns:
                            name = str(column)
                            base = re.sub(r"_(mean|std)$", "", name)
                            if base.lower() in X_KEYS:
                                continue  # x axes are not metrics
                            if name.endswith(("_mean", "_std")):
                                value = comparison.loc[run, column]
                                row[name] = float(value) if value is not None else float("nan")
                        rows.append(row)
                table = to_latex_table(
                    rows,
                    caption="Evaluation results across runs (mean $\\pm$ std).",
                    label="evaluation",
                )
                table_path = out_path / "evaluation_table.tex"
                table_path.write_text(table, encoding="utf-8")
                figures["__table_tex__"] = [table_path]
            except Exception as exc:
                LOG.warning("make_all_figures: cannot write evaluation_table.tex: %s", exc)

        def _attempt(key: str, func, *args, **kwargs) -> None:
            try:
                produced = func(*args, **kwargs)
            except Exception as exc:
                LOG.warning("make_all_figures: %s failed: %s", key, exc)
                return
            if produced:
                figures[key] = [Path(p) for p in produced]
                LOG.info("figure_made %s -> %s", key, produced)

        _attempt("learning_curves", plot_learning_curves, series_map, out_path, "learning_curves", formats)

        # x-axis keys (epoch/step/...) are axes, not comparable metrics
        metric_names = [m for m in flat_summary if str(m).lower() not in X_KEYS] or list(flat_summary)
        for metric in metric_names[:_BAR_METRIC_LIMIT]:
            safe = _safe_name(metric)
            _attempt(
                f"comparison_{safe}",
                plot_bar_comparison,
                run_summary,
                metric,
                out_path,
                f"comparison_{safe}",
                formats,
            )
            if comparison is not None and not comparison.empty:
                _attempt(
                    f"ablation_{safe}",
                    plot_ablation,
                    comparison,
                    metric,
                    out_path,
                    f"ablation_{safe}",
                    formats,
                )
            box_runs = {run: metrics.get(metric, []) for run, metrics in series_map.items()}
            _attempt(
                f"boxplot_{safe}",
                plot_boxplot,
                box_runs,
                metric,
                out_path,
                f"boxplot_{safe}",
                formats,
            )

        _attempt("metric_grid", plot_metric_grid, series_map, out_path, "metric_grid", formats)
        return figures
    except Exception as exc:  # absolute safety net: a plotting bug must not kill the stage
        LOG.warning("make_all_figures failed: %s", exc, exc_info=True)
        return figures
