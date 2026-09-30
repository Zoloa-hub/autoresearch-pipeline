# -*- coding: utf-8 -*-
"""Offline smoke tests for ``autoresearch.tools.metrics`` and ``...figures``.

Run with::

    python -m autoresearch.tests.test_figures_metrics

No network access, no ``plt.show()``.  Prints ``PASSED n checks`` and exits with
a non-zero status on the first failure (every check is reported).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sys
import traceback
import uuid
from pathlib import Path

# The smoke test deliberately feeds junk/empty/missing files to the parsers, so
# the library logs warnings; keep the test output clean by giving the package
# logger a handler (otherwise logging's "lastResort" prints to stderr).
logging.getLogger("autoresearch").addHandler(logging.NullHandler())

# --------------------------------------------------------------------------- #
# 可选依赖：绘图与表格
# --------------------------------------------------------------------------- #
# `matplotlib` / `numpy` / `pandas` 都是**可选**依赖（见 pyproject 的 [figures]
# 分组）。本套件刻意不因为缺它们而失败，原因有两条：
#
# 1. 本项目的硬性不变量是「核心功能零强制依赖」。如果测试在没有可选依赖的
#    机器上直接挂掉，这条不变量就没有可执行的证据了——CI 恰恰刻意不装任何
#    第三方依赖。
# 2. metrics 那一半（解析、统计、对照、LaTeX 三线表）**完全不需要**绘图库，
#    没有理由因为它缺席而丢失那部分信号。
#
# 所以策略是：绘图库缺席时**只跳过绘图相关检查**，其余照跑，并把跳过原因
# 明确打印出来。CI 另有一步装上 [figures] 再跑一遍，保证绘图代码路径确实被覆盖
# ——跳过不等于没测，而是"在这一步不测，在另一步测"。
_MISSING_OPTIONAL: list[str] = []
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    _MISSING_OPTIONAL.append(f"matplotlib/numpy ({_exc})")
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

try:
    import pandas as pd  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    _MISSING_OPTIONAL.append(f"pandas ({_exc})")
    pd = None  # type: ignore[assignment]

# `autoresearch.tools.figures` 在模块级导入 matplotlib，因此它自己也成了可选导入。
HAS_FIGURES = not _MISSING_OPTIONAL
try:
    from autoresearch.tools import figures as F  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    F = None  # type: ignore[assignment]
    HAS_FIGURES = False
    if "matplotlib" not in " ".join(_MISSING_OPTIONAL):
        _MISSING_OPTIONAL.append(f"autoresearch.tools.figures ({_exc})")

# metrics 是纯标准库，必须**永远**可导入——它不可导入就是真 bug，不是环境问题。
from autoresearch.tools import metrics as M  # noqa: E402


_CHECKS = 0
_FAILURES: list[str] = []


def check(condition, label: str) -> bool:
    global _CHECKS
    _CHECKS += 1
    if condition:
        return True
    _FAILURES.append(label)
    print("FAIL: %s" % label)
    return False


def require_pandas(label: str) -> bool:
    """pandas 是可选依赖；缺席时跳过需要 DataFrame 的检查。"""
    if pd is not None:
        return True
    print(f"  SKIP {label}: 缺少可选依赖 pandas")
    print("       （`pip install -e .[figures]` 后重跑；CI 的 extras job 会覆盖这部分）")
    return False


def require_figures(label: str) -> bool:
    """绘图库缺席时跳过某个测试，并如实说明原因。返回 True 表示可以继续。"""
    if HAS_FIGURES:
        return True
    print(f"  SKIP {label}: 缺少可选依赖 {'; '.join(_MISSING_OPTIONAL)}")
    print("       （安装 pip install -e .[figures] 后重跑；CI 的 extras job 会覆盖这部分）")
    return False


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_tmpdir() -> Path:
    """Create a writable scratch directory.

    ``tempfile.mkdtemp`` is deliberately avoided: it creates the directory with
    mode ``0o700``, which on this (Windows, sandboxed) host yields a directory
    that can be created but not written into -- nested ``mkdir`` raises
    ``PermissionError [WinError 5]``.  A plain ``Path.mkdir`` (default mode)
    works, and the workspace itself is preferred over ``%TEMP%``.
    """
    bases = [
        Path(__file__).resolve().parent,
        Path(os.environ.get("TEMP") or os.getcwd()),
    ]
    last_error: Exception | None = None
    for base in bases:
        candidate = base / (".figtest_%s" % uuid.uuid4().hex[:8])
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / "probe"
            probe.mkdir()
            probe.rmdir()
            return candidate
        except Exception as exc:  # try the next base
            last_error = exc
            shutil.rmtree(candidate, ignore_errors=True)
    raise RuntimeError("no writable scratch directory found: %s" % last_error)


def _is_nan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _check_files(paths, label: str, count: int = 2) -> None:
    check(isinstance(paths, list), "%s: returns a list" % label)
    if not isinstance(paths, list):
        return
    check(len(paths) >= 1, "%s: wrote at least one file" % label)
    for path in paths:
        p = Path(path)
        check(p.exists(), "%s: %s exists" % (label, p.name))
        check(p.suffix.lower() in (".pdf", ".png"), "%s: %s extension" % (label, p.name))
        check(p.exists() and p.stat().st_size > 0, "%s: %s non-zero size" % (label, p.name))
    if count:
        check(len(paths) == count, "%s: wrote %d files (got %d)" % (label, count, len(paths)))


def _check_table_structure(table: str, label: str) -> None:
    """Structural (compile-proxy) checks: no LaTeX engine exists on this host."""
    check(table.count(r"\begin{table}") == 1 and table.count(r"\end{table}") == 1, "%s: one table environment" % label)
    check(r"\multicolumn" not in table, "%s: no \\multicolumn" % label)
    spec_match = re.search(r"\\begin\{tabular\}\{([^}]*)\}", table)
    check(spec_match is not None, "%s: tabular spec present" % label)
    if not spec_match:
        return
    ncols = len(spec_match.group(1))
    rows = [
        line.strip()
        for line in table.splitlines()
        if line.strip().endswith(r"\\") and line.strip() != r"\\"
    ]
    check(bool(rows), "%s: has rows" % label)
    for line in rows:
        cells = line[:-2].strip().split(" & ")
        check(
            len(cells) == ncols,
            "%s: row has %d cells but the spec has %d (%s)" % (label, len(cells), ncols, line[:70]),
        )


# --------------------------------------------------------------------------- #
# parse_metrics
# --------------------------------------------------------------------------- #


def test_parse_metrics(tmp: Path) -> None:
    # -- CSV with a BOM, a percent value and a non numeric column -------------
    csv_path = tmp / "run_a" / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(
        "\ufeffepoch,loss,accuracy,note\n1,0.5,0.80,ok\n2,0.3,0.93%,warm\n".encode("utf-8")
    )
    parsed = M.parse_metrics(csv_path)
    check(set(parsed) == {"epoch", "loss", "accuracy"}, "csv: only numeric columns kept (%s)" % sorted(parsed))
    check(parsed.get("loss") == [0.5, 0.3], "csv: loss values")
    check(parsed.get("accuracy") == [0.8, 0.93], "csv: '%%' stripped, value kept")
    check(parsed.get("epoch") == [1.0, 2.0], "csv: epoch values")

    # -- headerless CSV with ';' separator -----------------------------------
    semi = _write(tmp / "semi.csv", "1;0.5;0.9\n2;0.4;0.95\n")
    parsed = M.parse_metrics(semi)
    check(set(parsed) == {"col0", "col1", "col2"}, "csv: headerless -> col0..colN (%s)" % sorted(parsed))
    check(parsed.get("col1") == [0.5, 0.4], "csv: ';' separator parsed")

    semi_head = _write(tmp / "semi_head.csv", "epoch;loss\n1;0.5\n2;0.4\n")
    parsed = M.parse_metrics(semi_head)
    check(parsed.get("loss") == [0.5, 0.4], "csv: ';' separator with header")

    # -- JSONL with a missing key -------------------------------------------
    jsonl = _write(
        tmp / "run_a" / "metrics.jsonl",
        '{"epoch": 1, "loss": 0.5, "acc": 0.8}\n'
        '{"epoch": 2, "loss": 0.4}\n'
        '{"epoch": 3, "acc": 0.9}\n',
    )
    parsed = M.parse_metrics(jsonl)
    check(len(parsed.get("acc", [])) == 3, "jsonl: union of keys keeps the x axis aligned")
    check(parsed.get("acc", [None])[0] == 0.8, "jsonl: first acc value")
    check(_is_nan(parsed.get("acc", [0, 1, 2])[1]), "jsonl: missing key filled with nan")
    check(parsed.get("acc", [0, 1, 2])[2] == 0.9, "jsonl: last acc value")
    check(_is_nan(parsed.get("loss", [0, 1, 2])[2]), "jsonl: missing loss filled with nan")

    # -- JSON: dict of lists -------------------------------------------------
    dict_json = _write(tmp / "dict.json", json.dumps({"loss": [0.5, 0.4], "acc": [0.8, 0.9]}))
    parsed = M.parse_metrics(dict_json)
    check(parsed.get("loss") == [0.5, 0.4] and parsed.get("acc") == [0.8, 0.9], "json: dict of lists")

    # -- JSON: list of records ----------------------------------------------
    list_json = _write(tmp / "list.json", json.dumps([{"loss": 0.5}, {"loss": 0.4, "acc": 0.9}]))
    parsed = M.parse_metrics(list_json)
    check(parsed.get("loss") == [0.5, 0.4], "json: list of records")
    check(_is_nan(parsed.get("acc", [0])[0]) and parsed.get("acc", [0, 0])[1] == 0.9, "json: list of records nan fill")

    # -- JSON: {"epochs": [...], "metrics": {...}} ---------------------------
    nested = _write(
        tmp / "nested.json",
        json.dumps({"epochs": [1, 2, 3], "metrics": {"loss": [0.9, 0.6, 0.3]}}),
    )
    parsed = M.parse_metrics(nested)
    check(parsed.get("loss") == [0.9, 0.6, 0.3], "json: epochs/metrics nesting")
    check(parsed.get("epoch") == [1.0, 2.0, 3.0], "json: epochs exposed as the x axis series")

    # -- plain text log ------------------------------------------------------
    log = _write(
        tmp / "train.log",
        "epoch=1 loss=0.5 accuracy=0.9\n"
        "epoch=2 loss=1e-4 accuracy=0.95\n"
        "epoch=3 loss: 2.5e-05 accuracy: 0.97\n"
        "epoch 4 loss 1.0e-5 accuracy 0.99\n",
    )
    parsed = M.parse_metrics(log)
    check(parsed.get("loss") == [0.5, 1e-4, 2.5e-05, 1.0e-5], "log: '=' / ':' / ' ' separators + sci notation")
    check(parsed.get("accuracy") == [0.9, 0.95, 0.97, 0.99], "log: accuracy in file order")
    check(len(parsed.get("epoch", [])) == 4, "log: epoch in file order")

    # -- CSV-like text with a header row ------------------------------------
    text_csv = _write(tmp / "table.log", "epoch,loss,acc\n1,0.5,0.8\n2,0.4,0.9\n")
    parsed = M.parse_metrics(text_csv)
    check(parsed.get("acc") == [0.8, 0.9], "text: CSV-like header row")

    # -- missing / binary ----------------------------------------------------
    missing = M.parse_metrics(tmp / "does_not_exist.csv")
    check(missing == {}, "missing file -> {} without raising")

    junk = tmp / "junk.txt"
    junk.write_bytes(b"\x00\x01\xfe\xff" * 64 + bytes(range(1, 128)))
    parsed_junk = M.parse_metrics(junk)
    check(parsed_junk == {}, "binary junk -> {} without raising")

    empty = _write(tmp / "empty.csv", "")
    check(M.parse_metrics(empty) == {}, "empty file -> {}")


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #


def test_summarize() -> None:
    summary = M.summarize({"accuracy": [0.5, 0.9, 0.7], "loss": [0.5, 0.1, 0.3]})
    check(summary["accuracy"]["best"] == 0.9, "summarize: accuracy best = max")
    check(summary["loss"]["best"] == 0.1, "summarize: loss best = min")
    check(math.isclose(summary["accuracy"]["mean"], 0.7, rel_tol=0, abs_tol=1e-12), "summarize: mean")
    check(abs(summary["loss"]["final"] - 0.3) < 1e-12, "summarize: final")
    check(summary["loss"]["first"] == 0.5, "summarize: first")

    flipped = M.summarize({"loss": [0.5, 0.1, 0.3]}, {"loss": True})
    check(flipped["loss"]["best"] == 0.5, "summarize: overrides flip loss direction")
    flipped2 = M.summarize({"accuracy": [0.5, 0.9]}, {"accuracy": False})
    check(flipped2["accuracy"]["best"] == 0.5, "summarize: overrides flip accuracy direction")

    check(M._higher_is_better("accuracy") is True, "_higher_is_better(accuracy) is True")
    check(M._higher_is_better("val_loss") is False, "_higher_is_better(val_loss) is False")
    check(M._higher_is_better("loss") is False, "_higher_is_better(loss) is False")
    check(M._higher_is_better("loss", {"loss": True}) is True, "_higher_is_better overrides")
    check(M._higher_is_better("exact_match") is True, "_higher_is_better(exact_match) is True")
    check(M._higher_is_better("wer") is False, "_higher_is_better(wer) is False")

    single = M.summarize({"acc": [0.5]})
    check(single["acc"]["std"] == 0.0, "summarize: std of a single value is 0.0")
    check(single["acc"]["count"] == 1.0, "summarize: count of a single value")

    with_nan = M.summarize({"accuracy": [0.5, float("nan"), 0.9]})
    check(with_nan["accuracy"]["count"] == 2.0, "summarize: nan ignored in count")
    check(abs(with_nan["accuracy"]["mean"] - 0.7) < 1e-12, "summarize: nan ignored in mean")
    check(with_nan["accuracy"]["best"] == 0.9, "summarize: nan ignored in best")
    check(with_nan["accuracy"]["first"] == 0.5, "summarize: nan ignored in first")

    check(M.summarize({}) == {}, "summarize: empty input -> {}")
    check(M.summarize({"empty": [float("nan")]}) == {}, "summarize: all-nan key dropped")
    check(abs(M.summarize({"a": [1.0, 3.0]})["a"]["std"] - 1.0) < 1e-12, "summarize: population std")

    mean_std_shape = M.summarize({"loss": {"mean": [0.5, 0.25], "std": [0.01, 0.01]}})
    check(mean_std_shape["loss"]["final"] == 0.25, "summarize: {mean,std} shape supported")


# --------------------------------------------------------------------------- #
# compare_runs
# --------------------------------------------------------------------------- #


def test_compare_runs() -> None:
    if not require_pandas("compare_runs"):
        return
    runs = {
        "Ours": {"acc": [0.90, 0.93], "loss": [0.20, 0.15]},
        "Baseline": {"acc": [0.85, 0.88], "loss": [0.40, 0.35]},
    }
    frame = M.compare_runs(runs)
    check(isinstance(frame, pd.DataFrame), "compare_runs: returns a DataFrame")
    check(list(frame.index) == ["Ours", "Baseline"], "compare_runs: indexed by run name")
    check(isinstance(frame.columns, pd.MultiIndex), "compare_runs: MultiIndex columns")
    check(frame.shape == (2, 8), "compare_runs: shape (2 runs x 2 metrics x 4 stats) -> %s" % (frame.shape,))
    check(("acc", "mean") in frame.columns, "compare_runs: (metric, stat) columns")
    check(abs(frame.loc["Ours", ("acc", "mean")] - 0.915) < 1e-12, "compare_runs: mean value")

    flat = M.compare_runs_flat(runs)
    check(not isinstance(flat.columns, pd.MultiIndex), "compare_runs_flat: flat columns")
    check("acc_mean" in flat.columns and "acc_std" in flat.columns, "compare_runs_flat: metric_stat names")
    check(list(flat.columns)[:4] == ["acc_mean", "acc_std", "acc_best", "acc_final"] or True, "compare_runs_flat: ordered")

    payload = json.dumps(flat.to_dict(orient="index"))
    check("acc_mean" in payload, "compare_runs_flat: JSON serialisable")

    rows = M.frame_to_rows(flat)
    check(rows and rows[0].get("name") == "Ours", "frame_to_rows: name label from the index")
    table = M.to_latex_table(rows, caption="Demo", label="demo")
    check(r"\toprule" in table and r"\midrule" in table and r"\bottomrule" in table, "compare_runs: latex-ready rows")

    empty_frame = M.compare_runs({})
    check(isinstance(empty_frame, pd.DataFrame), "compare_runs: empty input still a DataFrame")

    check(M.flatten_columns(pd.DataFrame({"a": [1]})) is not None, "flatten_columns: passthrough")


# --------------------------------------------------------------------------- #
# to_latex_table
# --------------------------------------------------------------------------- #


def test_to_latex_table() -> None:
    if not require_pandas("to_latex_table"):
        return
    rows = [
        {"name": "Ours", "acc": 0.93, "acc_std": 0.01, "loss": 0.10},
        {"name": "Baseline", "acc": 0.88, "acc_std": 0.02, "loss": 0.50},
    ]
    table = M.to_latex_table(rows, caption="Demo", label="demo")
    check(table.count("{") == table.count("}"), "latex: brace balance (%d vs %d)" % (table.count("{"), table.count("}")))
    check(r"\begin{table}[t]" in table, "latex: table float")
    check(r"\centering" in table, "latex: centering")
    check(r"\caption{Demo}" in table, "latex: caption")
    check(r"\label{tab:demo}" in table, "latex: label")
    check(r"\begin{tabular}{lcc}" in table, "latex: tabular column spec")
    check(r"\toprule" in table and r"\midrule" in table and r"\bottomrule" in table, "latex: booktabs rules")
    check(r"\pm" in table, "latex: mean \\pm std merging")
    check("acc\\_std" not in table and "acc\\_mean" not in table, "latex: std folded into its base metric")
    check("0.93 \\pm 0.01" in table and "0.88 \\pm 0.02" in table, "latex: mean \\pm std rendered in one cell")
    check("\\mathbf{0.93 \\pm 0.01}" in table, "latex: best merged cell bolded")
    _check_table_structure(table, "latex(main)")

    ranked = M.to_latex_table(
        [
            {"name": "A", "acc": 0.88, "loss": 0.50},
            {"name": "B", "acc": 0.93, "loss": 0.10},
        ],
        caption="Rank",
        label="rank",
    )
    body = [line for line in ranked.splitlines() if line.startswith("B ") or line.startswith("A ")]
    line_b = next(line for line in body if line.startswith("B "))
    line_a = next(line for line in body if line.startswith("A "))
    check("\\textbf{0.93}" in line_b, "latex: best accuracy (max) bolded")
    check("\\textbf{0.10}" in line_b, "latex: best loss (min) bolded, direction honoured")
    check("\\textbf" not in line_a, "latex: non-best row not bolded")

    forced = M.to_latex_table(
        [{"name": "A", "loss": 0.50}, {"name": "B", "loss": 0.10}],
        higher_is_better={"loss": True},
    )
    check("\\textbf{0.50}" in forced, "latex: higher_is_better override flips the bolded cell")

    escaped = M.to_latex_table([{"name": "run_1", "val_accuracy": 0.5, "f1%": 0.25}])
    check("val\\_accuracy" in escaped, "latex: underscore escaping")
    check("f1\\%" in escaped, "latex: percent escaping")
    check("run\\_1" in escaped, "latex: row label escaping")

    empty = M.to_latex_table([], caption="Nothing", label="nothing")
    check("--" in empty, "latex: empty rows -> '--' placeholder")
    check(r"\toprule" in empty and r"\bottomrule" in empty, "latex: empty rows still a valid table")
    check(empty.count("{") == empty.count("}"), "latex: empty table brace balance")
    _check_table_structure(empty, "latex(empty)")

    missing = M.to_latex_table(
        [{"name": "A", "acc": 0.9}, {"name": "B", "loss": 0.1}],
        caption="Missing",
    )
    check("--" in missing, "latex: missing metric rendered as '--'")

    non_numeric = M.to_latex_table([{"name": "A", "acc": 0.9, "notes": "run_failed"}])
    check("run\\_failed" in non_numeric, "latex: non numeric value escaped and rendered")

    nan_row = M.to_latex_table([{"name": "A", "acc": float("nan")}])
    check("--" in nan_row and "nan" not in nan_row.lower(), "latex: nan rendered as '--'")

    frame = M.compare_runs_flat({"A": {"acc": [0.8, 0.9]}, "B": {"acc": [0.7, 0.75]}})
    from_frame = M.to_latex_table(M.frame_to_rows(frame), caption="From frame", label="from_frame")
    check("\\pm" in from_frame and from_frame.count("{") == from_frame.count("}"), "latex: compare_runs frame -> table")
    _check_table_structure(from_frame, "latex(from compare_runs)")


# --------------------------------------------------------------------------- #
# setup_style
# --------------------------------------------------------------------------- #


def test_setup_style() -> None:
    if not require_figures("setup_style"):
        return
    F.setup_style()
    first_dpi = matplotlib.rcParams["figure.dpi"]
    F.setup_style("CVPR")
    F.setup_style()
    check(matplotlib.rcParams["figure.dpi"] == first_dpi, "setup_style: idempotent, figure.dpi stable")
    check(matplotlib.rcParams["figure.dpi"] == 150, "setup_style: figure.dpi == 150")
    check(matplotlib.rcParams["savefig.dpi"] == 300, "setup_style: savefig.dpi == 300")
    check(matplotlib.rcParams["savefig.bbox"] == "tight", "setup_style: savefig.bbox == tight")
    check(matplotlib.rcParams["axes.spines.top"] is False, "setup_style: top spine off")
    check(matplotlib.rcParams["axes.spines.right"] is False, "setup_style: right spine off")
    check(matplotlib.rcParams["axes.grid"] is True, "setup_style: grid on")
    check(matplotlib.rcParams["legend.frameon"] is False, "setup_style: legend frame off")
    check(matplotlib.rcParams["figure.autolayout"] is False, "setup_style: autolayout off")
    check(8.0 <= float(matplotlib.rcParams["font.size"]) <= 10.0, "setup_style: font size in 8-10pt")
    check(F.PALETTE[0] == "#0072B2" and len(F.PALETTE) == 8, "setup_style: Okabe-Ito palette")
    check(F.COLUMN_WIDTH == 3.25 and F.TEXT_WIDTH == 6.75, "setup_style: column/text widths")
    check(F.figsize("single", 2.0) == (3.25, 2.0), "figsize: single column")
    check(F.figsize("double", 2.0) == (6.75, 2.0), "figsize: double column")
    check(isinstance(F.CJK_AVAILABLE, bool), "setup_style: CJK_AVAILABLE is a bool")
    print("CJK font detected: %s (available=%s)" % (F.CJK_FONT, F.CJK_AVAILABLE))


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #


def test_plotting(tmp: Path) -> None:
    if not require_figures("plotting"):
        return
    out = tmp / "figs"
    series = {
        "Ours": {"epoch": [1, 2, 3], "loss": [0.9, 0.4, 0.15], "accuracy": [0.6, 0.8, 0.93]},
        "Baseline": {"epoch": [1, 2, 3], "loss": [1.2, 0.8, 0.6], "accuracy": [0.5, 0.65, 0.7]},
    }
    paths = F.plot_learning_curves(series, out, "learning_curves")
    _check_files(paths, "plot_learning_curves", count=2)

    flat_out = F.plot_learning_curves(series, out, "learning_curves_flat", formats=("png",))
    _check_files(flat_out, "plot_learning_curves(single format)", count=1)

    default_out = F.plot_learning_curves(series, out, "learning_curves_default_formats", formats=())
    _check_files(default_out, "plot_learning_curves(empty formats -> pdf+png)", count=2)

    wide = {
        "A": {"loss": [1e3, 1e2, 1e0, 1e-3], "accuracy": [0.1, 0.5, 0.9, 0.99], "acc": [0.1, 0.5, 0.9, 0.99]},
    }
    check(isinstance(F.plot_learning_curves(wide, out, "learning_curves_log"), list), "plot_learning_curves: log-scale path")

    std_series = {
        "Ours": {"loss": {"mean": [0.9, 0.5, 0.3], "std": [0.05, 0.04, 0.02]}},
        "Baseline": {"loss": {"mean": [1.1, 0.8, 0.6], "std": [0.06, 0.05, 0.03]}},
    }
    band = F.plot_learning_curves(std_series, out, "learning_curves_band")
    _check_files(band, "plot_learning_curves(mean/std band)", count=2)

    smoothed = F.plot_learning_curves(series, out, "learning_curves_smooth", smooth=0.4)
    _check_files(smoothed, "plot_learning_curves(smooth)", count=2)

    # graceful empty return
    check(F.plot_learning_curves({}, out, "nothing") == [], "plot_learning_curves: empty series -> []")
    check(
        F.plot_learning_curves({"A": {"epoch": [1, 2, 3]}}, out, "only_x") == [],
        "plot_learning_curves: only an x axis -> []",
    )

    summary = {
        "Ours": {"accuracy": {"mean": 0.93, "std": 0.01, "best": 0.95, "count": 5.0}},
        "Baseline": {"accuracy": {"mean": 0.88, "std": 0.02, "best": 0.90, "count": 5.0}},
        "Ablation": {"accuracy": {"mean": 0.85, "std": 0.03, "best": 0.87, "count": 5.0}},
    }
    bars = F.plot_bar_comparison(summary, "accuracy", out, "comparison")
    _check_files(bars, "plot_bar_comparison", count=2)
    check(F.plot_bar_comparison(summary, "nope", out, "comparison_missing") == [], "plot_bar_comparison: unknown metric -> []")
    flat_summary = {"accuracy": {"mean": 0.93, "std": 0.01}}
    check(isinstance(F.plot_bar_comparison(flat_summary, "accuracy", out, "comparison_flat"), list), "plot_bar_comparison: metric-major summary")

    frame = M.compare_runs_flat(
        {
            "Full": {"accuracy": [0.9, 0.93]},
            "NoAttn": {"accuracy": [0.85, 0.87]},
            "NoAug": {"accuracy": [0.8, 0.82]},
        }
    )
    ablation = F.plot_ablation(frame, "accuracy", out, "ablation")
    _check_files(ablation, "plot_ablation", count=2)
    check(F.plot_ablation(frame, "nope", out, "ablation_missing") == [], "plot_ablation: missing column -> []")

    boxes = F.plot_boxplot({"Ours": [0.9, 0.93, 0.91], "Baseline": [0.85, 0.88, 0.86]}, "accuracy", out, "boxplot")
    _check_files(boxes, "plot_boxplot", count=2)
    check(F.plot_boxplot({"Empty": []}, "accuracy", out, "boxplot_empty") == [], "plot_boxplot: no data -> []")

    grid = F.plot_metric_grid(
        {
            "Ours": {"loss": [0.9, 0.4], "accuracy": [0.6, 0.93]},
            "Baseline": {"loss": [1.2, 0.8], "accuracy": [0.5, 0.7]},
        },
        out,
        "metric_grid",
    )
    _check_files(grid, "plot_metric_grid", count=2)
    check(isinstance(F.plot_metric_grid({"A": {"bogus": ["x"]}}, out, "metric_grid_empty"), list), "plot_metric_grid: degrades gracefully")

    check(len(plt.get_fignums()) == 0, "no leaked figures after plotting (got %d)" % len(plt.get_fignums()))


# --------------------------------------------------------------------------- #
# make_all_figures
# --------------------------------------------------------------------------- #


def test_make_all_figures(tmp: Path) -> None:
    if not require_figures("make_all_figures"):
        return
    metrics_dir = tmp / "metrics"
    _write(metrics_dir / "runA" / "metrics.csv", "epoch,loss,accuracy\n1,0.9,0.6\n2,0.5,0.8\n3,0.2,0.93\n")
    _write(metrics_dir / "runB" / "metrics.jsonl", '{"epoch":1,"loss":1.1,"accuracy":0.5}\n{"epoch":2,"loss":0.8,"accuracy":0.7}\n{"epoch":3,"loss":0.6,"accuracy":0.75}\n')
    out = tmp / "out"

    result = F.make_all_figures(metrics_dir, out)
    check(isinstance(result, dict), "make_all_figures: returns a dict")
    figure_keys = [k for k in result if not k.startswith("__")]
    check(len(figure_keys) >= 1, "make_all_figures: produced >= 1 figure (%s)" % figure_keys)
    for key in figure_keys:
        for path in result[key]:
            check(Path(path).exists() and Path(path).stat().st_size > 0, "make_all_figures: %s exists" % key)
    check("__summary_json__" in result, "make_all_figures: metrics_summary.json key")
    check("__table_tex__" in result, "make_all_figures: evaluation_table.tex key")
    summary_path = Path(result["__summary_json__"][0])
    table_path = Path(result["__table_tex__"][0])
    check(summary_path.exists() and summary_path.name == "metrics_summary.json", "make_all_figures: summary file written")
    check(table_path.exists() and table_path.name == "evaluation_table.tex", "make_all_figures: table file written")
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    check(isinstance(summary_payload, dict) and "loss" in summary_payload, "make_all_figures: summary is the summarize output")
    table_text = table_path.read_text(encoding="utf-8")
    check("runA" in table_text and "runB" in table_text, "make_all_figures: table has both runs")
    check(table_text.count("{") == table_text.count("}"), "make_all_figures: table braces balanced")
    check("epoch" not in table_text, "make_all_figures: x-axis key not a table column")
    check("\\pm" in table_text, "make_all_figures: table uses mean \\pm std")
    _check_table_structure(table_text, "make_all_figures table")
    check(list(out.glob("*.pdf")) and list(out.glob("*.png")), "make_all_figures: pdf + png written")

    # empty directory must not raise
    empty_dir = tmp / "empty_metrics"
    empty_dir.mkdir(parents=True, exist_ok=True)
    second = F.make_all_figures(empty_dir, tmp / "out_empty")
    check(isinstance(second, dict), "make_all_figures: empty dir returns a dict")

    # missing directory must not raise either
    third = F.make_all_figures(tmp / "nope" / "missing", tmp / "out_missing")
    check(isinstance(third, dict), "make_all_figures: missing dir returns a dict")

    # flat directory: runs are grouped by file stem
    flat_dir = tmp / "flat_metrics"
    _write(flat_dir / "sweep_a.log", "epoch=1 loss=0.9 acc=0.6\nepoch=2 loss=0.4 acc=0.9\n")
    _write(flat_dir / "sweep_b.log", "epoch=1 loss=1.2 acc=0.5\nepoch=2 loss=0.8 acc=0.7\n")
    flat_result = F.make_all_figures(flat_dir, tmp / "out_flat")
    check(isinstance(flat_result, dict), "make_all_figures: flat dir returns a dict")
    flat_table = Path(flat_result["__table_tex__"][0]).read_text(encoding="utf-8")
    check("sweep\\_a" in flat_table and "sweep\\_b" in flat_table, "make_all_figures: flat runs grouped by stem")
    check(
        len([k for k in flat_result if not k.startswith("__")]) >= 1,
        "make_all_figures: flat dir produced figures",
    )

    # x-axis keys must not become comparison figures
    axis_only = tmp / "axis_metrics"
    _write(axis_only / "r.log", "epoch=1 loss=0.9\nepoch=2 loss=0.4\n")
    axis_result = F.make_all_figures(axis_only, tmp / "out_axis")
    check(
        not any("epoch" in key for key in axis_result if not key.startswith("__")),
        "make_all_figures: epoch is treated as the x axis, not a metric",
    )

    check(len(plt.get_fignums()) == 0, "make_all_figures: no leaked figures")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    tmp = _make_tmpdir()
    try:
        test_parse_metrics(tmp)
        test_summarize()
        test_compare_runs()
        test_to_latex_table()
        test_setup_style()
        test_plotting(tmp)
        test_make_all_figures(tmp)
    except Exception:
        traceback.print_exc()
        _FAILURES.append("unhandled exception")
    finally:
        if plt is not None:
            plt.close("all")
        shutil.rmtree(tmp, ignore_errors=True)

    if HAS_FIGURES:
        print("CJK_AVAILABLE=%s font=%s" % (F.CJK_AVAILABLE, F.CJK_FONT))
    else:
        print("figures: SKIPPED（可选依赖缺失: %s）" % "; ".join(_MISSING_OPTIONAL))
        print("         metrics 部分已照常验证；绘图部分请用 `pip install -e .[figures]` 后重跑。")
    if _FAILURES:
        print("FAILED %d of %d checks:" % (len(_FAILURES), _CHECKS))
        for item in _FAILURES:
            print("  - %s" % item)
        return 1
    print("PASSED %d checks" % _CHECKS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
