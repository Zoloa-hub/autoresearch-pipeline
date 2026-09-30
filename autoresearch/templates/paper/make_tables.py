#!/usr/bin/env python
"""Minimal booktabs table generator for the paper skeleton (s5_analysis helper).

Usage:
    python make_tables.py results_table_csv --caption "Main results" \
        --label tab:main_results --out tables/main_results.tex

Reads a CSV whose first column is the row label and whose remaining columns are
numeric metrics, and writes a three-line (booktabs) LaTeX table. This is the
in-tree reference implementation; the pipeline's own generator lives in
``autoresearch/tools/metrics.py`` (``to_latex_table``).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_table(rows, header, caption, label, decimals=2, bold_best=True,
                higher_is_better=True):
    """Return a booktabs table as a LaTeX string."""
    ncols = len(header)
    best_idx = -1
    if bold_best and ncols > 1:
        values = []
        for row in rows:
            try:
                values.append(float(row[-1]))
            except (TypeError, ValueError):
                values.append(float("-inf") if higher_is_better else float("inf"))
        if values:
            best_idx = max(range(len(values)), key=lambda i: values[i]) \
                if higher_is_better else min(range(len(values)), key=lambda i: values[i])

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{" + "l" + "c" * (ncols - 1) + "}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    for index, row in enumerate(rows):
        cells = [str(cell) for cell in row]
        if bold_best and index == best_idx and len(cells) > 1:
            cells[-1] = r"\textbf{" + cells[-1] + "}"
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--caption", default="Results")
    parser.add_argument("--label", default="tab:results")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args(argv)

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]
    if not rows:
        raise SystemExit(f"no rows in {args.csv_path}")
    header, body = rows[0], [r for r in rows[1:]]
    body = [
        [cell if i == 0 else f"{float(cell):.{args.decimals}f}"
         for i, cell in enumerate(row)]
        for row in body
    ]
    table = build_table(body, header, args.caption, args.label, args.decimals)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
