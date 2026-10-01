# -*- coding: utf-8 -*-
"""Metric parsing, summarization and booktabs (three-line) LaTeX tables.

Implements ``CONTRACTS.md`` section 8 (FROZEN interface):

    parse_metrics(path) -> dict[str, list[float]]
    summarize(series) -> dict[str, dict[str, float]]
    compare_runs(runs) -> pandas.DataFrame
    to_latex_table(rows, caption, label, decimals, bold_best, higher_is_better) -> str

Extra helpers (additive, the frozen signatures above are untouched):

    compare_runs_flat(runs)   -> flat "metric_stat" columns, JSON friendly
    flatten_columns(df)       -> flatten a MultiIndex column frame
    frame_to_rows(df)         -> list[dict] with a "name" row label

Design notes / resolved ambiguities
-----------------------------------
* Only ``dict[str, list[float]]`` is ever returned by :func:`parse_metrics`; the
  return type stays clean (non numeric columns are simply dropped).
* ``PERCENT_AS_FRACTION`` documents the ``"0.93%"`` policy.  The contract says
  "strip the ``%`` and note it", so by default the bare number is kept
  (``0.93``) and a debug record is logged.  Flip the constant to ``True`` to
  divide by 100 instead.
* A key that has no numeric value anywhere (e.g. a JSONL "note" field, a CSV
  label column) is dropped from the result.
* Missing cells inside an otherwise numeric series become ``float('nan')`` so
  that x axes stay aligned.
* A key whose every value is ``nan`` is dropped by :func:`summarize` (keeps the
  emitted JSON strictly free of ``NaN`` literals).
* ``_higher_is_better`` splits the metric name into ``_`` tokens, so
  ``val_loss`` / ``train/accuracy`` resolve correctly, and it accepts an
  ``overrides`` mapping (exact, case/separator insensitive).
* ``_mean`` / ``_avg`` suffixes are recognised as the central value so that the
  flattened ``compare_runs`` frame (``acc_mean`` + ``acc_std``) folds back into a
  single ``mean \\pm std`` cell.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

LOG = logging.getLogger(__name__)

__all__ = [
    "OptionalDependencyError",
    "parse_metrics",
    "summarize",
    "compare_runs",
    "compare_runs_flat",
    "flatten_columns",
    "frame_to_rows",
    "to_latex_table",
    "PERCENT_AS_FRACTION",
]


class OptionalDependencyError(ImportError):
    """某个**可选**依赖缺席（而非代码缺陷）。

    单独一个类型是为了让调用方能区分两件事：环境没装 vs 功能坏了。
    前者应当优雅降级/跳过并打印安装提示，后者应当失败。
    `parse_metrics` / `summarize` 这类核心功能**不会**抛它——
    只有绘图与 DataFrame 形态的输出才会。
    """


# --------------------------------------------------------------------------- #
# module level policy
# --------------------------------------------------------------------------- #

#: ``"0.93%"`` -> ``0.93`` when False (contract: "strip the % and note it"),
#: ``0.0093`` when True.
PERCENT_AS_FRACTION = False

_STATS = ("mean", "std", "best", "final")

_NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
_KV_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_\-\./]*)"
    r"\s*(?P<sep>=|:|\s)\s*"
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*(?P<pct>%)?"
)

_CSV_SUFFIXES = {".csv", ".tsv"}
_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
_JSON_SUFFIXES = {".json"}

_COMMENT_PREFIXES = ("#", "//", ";")

# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #


def _to_float(value: Any) -> float | None:
    """Coerce *value* to ``float``; return ``None`` when it is not numeric.

    ``nan`` passes through unchanged (callers decide how to treat it).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        pct = s.endswith("%")
        if pct:
            s = s[:-1].strip()
        if not s:
            return None
        try:
            num = float(s.replace(",", "") if ("," in s and "." not in s) else s)
        except ValueError:
            return None
        if pct and PERCENT_AS_FRACTION:
            num = num / 100.0
        return num
    # numpy scalars / anything with __float__
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_scalar(value: Any) -> float | None:
    """Parse-time coercion: ``None`` for non numeric *and* for ``nan``."""
    num = _to_float(value)
    if num is None or math.isnan(num):
        return None
    return num


def _is_number_text(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("%") and len(s) > 1:
            s = s[:-1].strip()
        return bool(_NUMBER_RE.match(s))
    return False


def _clean_key(name: Any) -> str:
    return str(name).strip()


# --------------------------------------------------------------------------- #
# series extraction (direction of "good")
# --------------------------------------------------------------------------- #

#: metrics where larger is better (multi-word entries are matched as a whole)
_HIGHER_WORDS = frozenset(
    """
    accuracy acc f1 precision recall auc auroc bleu rouge score reward iou dice
    psnr ssim top1 top5 map mrr ndcg corr
    """.split()
)
_HIGHER_PHRASES = frozenset(
    ["exact_match", "success_rate", "win_rate", "f1_score", "pass_at_1", "pass@1"]
)

#: metrics where smaller is better.
#: 这是**全项目唯一的规范表**。此前 s3/s4/s5/s9 与 adapters 各自维护了一份副本，
#: 且 token 集合已经漂移（有的有 fid/fdr，有的有 flops/params），导致同一个指标名
#: 在不同阶段被判定为不同方向——这类不一致不会报错，只会让「改善」的定义在不同
#: 章节里悄悄改变。现在所有地方都从这里取。
_LOWER_WORDS = frozenset(
    """
    loss error rmse mae mse perplexity ppl nll latency time cost flops params
    wer cer memory fid fdr
    """.split()
)
_LOWER_PHRASES = frozenset(["val_loss", "train_loss", "test_loss"])


def _normalize_metric_name(name: str) -> str:
    key = str(name).strip().lower()
    key = re.sub(r"[\s/\-\.\:@%]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def _higher_is_better(name: str, overrides: Mapping[str, bool] | None = None) -> bool:
    """Return ``True`` when a larger value of *name* is better.

    Resolution order:
      1. ``overrides`` (compared after normalisation, exact key wins);
      2. the multi-word phrase tables;
      3. the ``_`` token tables (so ``val_loss`` and ``train/accuracy`` work);
      4. substring fallback for tokens of >= 4 characters;
      5. default ``True`` (unknown metrics are assumed higher-is-better).
    """
    key = _normalize_metric_name(name)
    if overrides:
        for cand, flag in overrides.items():
            if _normalize_metric_name(cand) == key:
                return bool(flag)
    if not key:
        return True

    for phrase in _LOWER_PHRASES:
        if key == phrase or key.endswith("_" + phrase) or key.startswith(phrase + "_"):
            return False
    tokens = [t for t in key.split("_") if t]
    for token in tokens:
        if token in _LOWER_WORDS:
            return False
    for phrase in _HIGHER_PHRASES:
        if key == phrase or key.endswith("_" + phrase) or key.startswith(phrase + "_"):
            return True
    for token in tokens:
        if token in _HIGHER_WORDS:
            return True
    # substring fallback: "eval_loss_mean" -> loss, "runtime" -> time
    for word in _LOWER_WORDS:
        if len(word) >= 4 and word in key:
            return False
    for word in _HIGHER_WORDS:
        if len(word) >= 4 and word in key:
            return True
    return True


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def _read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        LOG.warning("parse_metrics: cannot read %s: %s", path, exc)
        return None
    if not raw.strip():
        LOG.warning("parse_metrics: %s is empty", path)
        return None
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError as exc:
            LOG.warning("parse_metrics: %s looks binary (%s)", path, exc)
            return None
    if b"\x00" in raw[:8192]:
        LOG.warning("parse_metrics: %s looks binary (NUL byte)", path)
        return None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return None
    if text.count("\ufffd") > max(8, len(text) // 20):
        LOG.warning("parse_metrics: %s is not valid text", path)
        return None
    return text


def _read_delimited_stdlib(text: str) -> list[list[str]] | None:
    """纯标准库的「无表头分隔文本」读取回退（pandas 缺席时使用）。

    为什么必须有这个回退：解析 `metrics.csv` 是**核心功能**，而 pandas 是
    **可选**依赖（pyproject 的 [figures] 分组）。如果解析 CSV 需要 pandas，那么
    「核心零强制依赖」这条不变量在真正用到指标解析时就不成立了——而指标解析恰好
    是适配器契约里最基础的一环（`read_standard_metrics` 直接依赖它）。

    尝试顺序与 pandas 版本一致：自动嗅探（用 csv.Sniffer）、逗号、分号、制表符、
    空白。返回「列数最多」的候选，与 pandas 版本挑 `shape[1] > 1` 的意图相同。
    """
    import csv as _csv

    text = text.strip("\ufeff")
    if not text.strip():
        return None

    candidates: list[list[list[str]]] = []

    # 1) 嗅探分隔符
    try:
        sample = "\n".join(text.splitlines()[:20])
        dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t| ")
        candidates.append([row for row in _csv.reader(io.StringIO(text), dialect)])
    except Exception as exc:  # noqa: BLE001
        LOG.debug("parse_metrics: csv.Sniffer failed: %s", exc)

    # 2) 显式分隔符
    for sep in (",", ";", "\t"):
        try:
            candidates.append([row for row in _csv.reader(io.StringIO(text), delimiter=sep)])
        except Exception as exc:  # noqa: BLE001
            LOG.debug("parse_metrics: csv.reader(delimiter=%r) failed: %s", sep, exc)

    # 3) 空白分隔（最后手段：metrics 文件有时用空格对齐）
    candidates.append([line.split() for line in text.splitlines()])

    best: list[list[str]] | None = None
    best_cols = 0
    for rows in candidates:
        rows = [r for r in rows if any(str(c).strip() for c in r)]
        if not rows:
            continue
        cols = max(len(r) for r in rows)
        if cols > best_cols:
            best, best_cols = rows, cols
        if cols > 1:
            return rows
    return best


def _rows_to_series(rows: list[list[str]] | None) -> dict[str, list[float]]:
    """把「无表头的行列表」转成 `{指标名: [值...]}`。

    表头探测规则与 pandas 版本保持一致：首行若全为非空字符串且不含数字，就当作表头。
    """
    if not rows:
        return {}

    # 去掉全空列与全空行（对应 pandas 版的 dropna）
    width = max(len(r) for r in rows)
    rows = [list(r) + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(str(r[i]).strip() for r in rows)]
    rows = [[r[i] for i in keep] for r in rows if any(str(c).strip() for c in r)]
    if not rows:
        return {}
    width = len(rows[0])

    first = rows[0]
    non_empty = [c for c in first if str(c).strip()]
    header: list[str] | None = None
    if non_empty and all(not _is_number_text(c) for c in non_empty):
        header = [_clean_key(c) for c in first]
        rows = rows[1:]
    if not rows:
        return {}

    names: list[str] = []
    seen: dict[str, int] = {}
    for i in range(width):
        if header is not None and i < len(header):
            name = header[i] or f"col{i}"
        else:
            name = f"col{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)

    series: dict[str, list[float]] = {}
    for i, name in enumerate(names):
        values: list[float | None] = []
        for row in rows:
            cell = row[i] if i < len(row) else ""
            values.append(_coerce_scalar(cell))
        if not any(v is not None for v in values):
            continue
        series[name] = [float("nan") if v is None else float(v) for v in values]
    return series


def _read_delimited(text: str):
    """Read *text* as a header-less CSV, trying several separators.

    pandas 在时用 pandas（生态兼容、行为经过验证）；不在时回退到标准库实现。
    两条路径的**契约相同**：返回可交给 `_frame_to_series` / `_rows_to_series` 的
    中间形态。
    """
    try:
        import pandas as pd
    except ImportError:
        LOG.debug("parse_metrics: pandas 缺席，改用标准库解析分隔文本")
        return _read_delimited_stdlib(text)

    candidates = (
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";", "engine": "python"},
        {"sep": "\t", "engine": "python"},
        {"sep": r"\s+", "engine": "python"},
    )
    best = None
    for kw in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = pd.read_csv(io.StringIO(text), header=None, skip_blank_lines=True, **kw)
        except Exception as exc:  # parser errors are expected while probing
            LOG.debug("parse_metrics: read_csv%r failed: %s", kw, exc)
            continue
        if frame.shape[1] > 1:
            return frame
        if best is None:
            best = frame
    return best


def _frame_to_series(frame) -> dict[str, list[float]]:
    # 标准库回退路径传进来的是 list[list[str]]，不是 DataFrame
    if isinstance(frame, (list, tuple)) or frame is None:
        return _rows_to_series(frame)

    import pandas as pd

    if frame is None or frame.empty:
        return {}
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if frame.empty:
        return {}

    first = list(frame.iloc[0])
    header = None
    non_empty = [v for v in first if not (isinstance(v, float) and math.isnan(v))]
    if non_empty and all(isinstance(v, str) and v.strip() for v in non_empty) and not any(
        _is_number_text(v) for v in non_empty
    ):
        header = [_clean_key(v) for v in first]
        frame = frame.iloc[1:].reset_index(drop=True)

    if header is None:
        names: list[str] = []
        for i in range(frame.shape[1]):
            names.append(f"col{i}")
    else:
        names = []
        seen: dict[str, int] = {}
        for i in range(frame.shape[1]):
            raw = header[i] if i < len(header) else f"col{i}"
            name = raw or f"col{i}"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            names.append(name)

    series: dict[str, list[float]] = {}
    pct_seen = 0
    for i, name in enumerate(names):
        if i >= frame.shape[1]:
            break
        column = list(frame.iloc[:, i])
        values: list[float | None] = []
        for cell in column:
            if isinstance(cell, str) and cell.strip().endswith("%"):
                pct_seen += 1
            values.append(_coerce_scalar(cell))
        if not any(v is not None for v in values):
            continue  # non numeric column -> dropped (keeps the return type clean)
        series[name] = [float("nan") if v is None else float(v) for v in values]
    if pct_seen:
        LOG.debug(
            "parse_metrics: stripped '%%' from %d value(s); PERCENT_AS_FRACTION=%s",
            pct_seen,
            PERCENT_AS_FRACTION,
        )
    return series


def _parse_csv(text: str) -> dict[str, list[float]]:
    return _frame_to_series(_read_delimited(text))


def _records_to_series(records: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    keys: list[str] = []
    for record in records:
        for key in record:
            if str(key) not in keys:
                keys.append(str(key))
    series: dict[str, list[float]] = {}
    for key in keys:
        values: list[float] = []
        for record in records:
            if key in record:
                num = _coerce_scalar(record[key])
                values.append(float("nan") if num is None else num)
            else:
                values.append(float("nan"))  # keep the x axis aligned
        if any(not math.isnan(v) for v in values):
            series[key] = values
    return series


def _list_to_series(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for item in values:
        num = _coerce_scalar(item)
        out.append(float("nan") if num is None else num)
    return out


def _parse_jsonl(text: str) -> dict[str, list[float]]:
    records: list[Mapping[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(_COMMENT_PREFIXES):
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            LOG.debug("parse_metrics: JSONL line %d skipped: %s", lineno, exc)
            continue
        if isinstance(obj, Mapping):
            records.append(obj)
        elif isinstance(obj, list):
            records.extend(x for x in obj if isinstance(x, Mapping))
    return _records_to_series(records) if records else {}


def _parse_json_object(obj: Any) -> dict[str, list[float]]:
    if isinstance(obj, list):
        if obj and all(isinstance(item, Mapping) for item in obj):
            return _records_to_series([dict(item) for item in obj])
        if obj and all(not isinstance(item, (Mapping, list, tuple)) for item in obj):
            values = _list_to_series(obj)
            return {"value": values} if any(not math.isnan(v) for v in values) else {}
        return {}

    if not isinstance(obj, Mapping):
        return {}

    nested = obj.get("metrics")
    if isinstance(nested, Mapping):
        series = _parse_json_object(nested)
        for axis_key in ("epochs", "epoch", "steps", "step", "iters"):
            axis = obj.get(axis_key)
            if not isinstance(axis, (list, tuple)):
                continue
            target = "epoch" if axis_key.startswith("epoch") else "step"
            if target in series:
                continue
            values = _list_to_series(axis)
            if any(not math.isnan(v) for v in values):
                series[target] = values
        return series

    series: dict[str, list[float]] = {}
    for key, value in obj.items():
        name = str(key)
        if isinstance(value, (list, tuple)):
            values = _list_to_series(value)
            if any(not math.isnan(v) for v in values):
                series[name] = values
        elif not isinstance(value, Mapping):
            num = _coerce_scalar(value)
            if num is not None:
                series[name] = [num]
    return series


def _parse_json(text: str) -> dict[str, list[float]]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        LOG.debug("parse_metrics: not JSON: %s", exc)
        return {}
    return _parse_json_object(obj)


def _split_row(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,\t;|]", line)]


def _parse_text(text: str) -> dict[str, list[float]]:
    """Parse ``key=value`` / ``key: value`` / ``key value`` logs and CSV-like text."""
    lines = text.splitlines()
    first_idx = None
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue
        first_idx = i
        break
    if first_idx is None:
        return {}

    parts = _split_row(lines[first_idx])
    if len(parts) >= 2 and all(p for p in parts) and not any(_is_number_text(p) for p in parts):
        # header row followed by rows of numbers
        header = [p or f"col{i}" for i, p in enumerate(parts)]
        columns: dict[str, list[float]] = {name: [] for name in header}
        for raw in lines[first_idx + 1:]:
            stripped = raw.strip()
            if not stripped or stripped.startswith(_COMMENT_PREFIXES):
                continue
            row = _split_row(stripped)
            if len(row) != len(header):
                for key, value in _kv_pairs(stripped):
                    columns.setdefault(key, []).append(value)
                continue
            for name, cell in zip(header, row):
                num = _coerce_scalar(cell)
                columns[name].append(float("nan") if num is None else num)
        series = {k: v for k, v in columns.items() if v and any(not math.isnan(x) for x in v)}
        if series:
            return series

    series: dict[str, list[float]] = {}
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue
        for key, value in _kv_pairs(stripped):
            series.setdefault(key, []).append(value)
    return series


def _kv_pairs(line: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for match in _KV_RE.finditer(line):
        key = match.group("key")
        num = _coerce_scalar(match.group("value"))
        if num is not None:
            pairs.append((key, num))
    return pairs


def parse_metrics(path) -> dict[str, list[float]]:
    """Parse a metrics file into ``{metric_name: [float, ...]}``.

    Supports CSV/TSV, JSONL, JSON (dict-of-lists, list-of-records and
    ``{"epochs": [...], "metrics": {...}}``) and plain text logs.  Unknown,
    empty, binary or unreadable files yield ``{}`` -- this function never raises.
    """
    try:
        file_path = Path(path)
    except TypeError as exc:
        LOG.warning("parse_metrics: bad path %r: %s", path, exc)
        return {}

    try:
        if not file_path.exists() or not file_path.is_file():
            LOG.warning("parse_metrics: file not found: %s", file_path)
            return {}
    except OSError as exc:
        LOG.warning("parse_metrics: cannot stat %s: %s", file_path, exc)
        return {}

    text = _read_text(file_path)
    if text is None:
        return {}

    suffix = file_path.suffix.lower()
    if suffix in _CSV_SUFFIXES:
        parsers = (_parse_csv, _parse_json, _parse_jsonl, _parse_text)
    elif suffix in _JSONL_SUFFIXES:
        parsers = (_parse_jsonl, _parse_json, _parse_csv, _parse_text)
    elif suffix in _JSON_SUFFIXES:
        parsers = (_parse_json, _parse_jsonl, _parse_csv, _parse_text)
    else:
        parsers = (_parse_jsonl, _parse_json, _parse_text, _parse_csv)

    for parser in parsers:
        try:
            result = parser(text)
        except Exception as exc:  # never raise out of parse_metrics
            LOG.debug("parse_metrics: %s failed on %s: %s", parser.__name__, file_path, exc)
            continue
        if result:
            LOG.debug("parse_metrics: %s parsed by %s (%d series)", file_path, parser.__name__, len(result))
            return result
    LOG.warning("parse_metrics: no metrics found in %s", file_path)
    return {}


# --------------------------------------------------------------------------- #
# summarization
# --------------------------------------------------------------------------- #


def _as_float_list(value: Any) -> list[float]:
    """Accept ``list[float]``, scalars, or the ``{"mean": [...], "std": [...]}`` shape."""
    if isinstance(value, Mapping):
        for key in ("mean", "value", "y", "values"):
            if key in value:
                return _as_float_list(value[key])
        return []
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            num = _to_float(item)
            out.append(float("nan") if num is None else num)
        return out
    num = _to_float(value)
    return [] if num is None else [num]


def summarize(
    series: Mapping[str, Any],
    higher_is_better: Mapping[str, bool] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-metric statistics (``count/mean/std/min/max/first/final/best``).

    ``nan`` values are ignored in every statistic; ``std`` is the *population*
    standard deviation and is ``0.0`` for a single observation.  ``best`` is the
    max or the min depending on :func:`_higher_is_better` (and on the optional
    ``higher_is_better`` overrides).  Keys with no numeric value are dropped.
    """
    if not series:
        return {}
    if not isinstance(series, Mapping):
        raise TypeError("summarize(series): expected a mapping of metric -> values")

    summary: dict[str, dict[str, float]] = {}
    for name, raw_values in series.items():
        values = [v for v in _as_float_list(raw_values) if not math.isnan(v)]
        if not values:
            LOG.debug("summarize: dropping %r (no numeric values)", name)
            continue
        count = len(values)
        mean = math.fsum(values) / count
        if count > 1:
            variance = math.fsum((v - mean) ** 2 for v in values) / count
            std = math.sqrt(variance) if variance > 0 else 0.0
        else:
            std = 0.0
        best = max(values) if _higher_is_better(str(name), higher_is_better) else min(values)
        summary[str(name)] = {
            "count": float(count),
            "mean": float(mean),
            "std": float(std),
            "min": float(min(values)),
            "max": float(max(values)),
            "first": float(values[0]),
            "final": float(values[-1]),
            "best": float(best),
        }
    return summary


# --------------------------------------------------------------------------- #
# run comparison
# --------------------------------------------------------------------------- #


def compare_runs(runs: Mapping[str, Mapping[str, Any]]):
    """Compare runs; returns a DataFrame indexed by run name.

    Columns are a :class:`pandas.MultiIndex` of ``(metric, stat)`` with
    ``stat in {mean, std, best, final}``.  Use :func:`compare_runs_flat` (or
    :func:`flatten_columns`) for a JSON / ``to_latex_table`` friendly frame.

    **需要 pandas（可选依赖）。** 缺席时抛 :class:`OptionalDependencyError` 而不是
    ``ImportError``：调用方据此可以区分「环境缺依赖」与「代码坏了」，并给出
    「装 .[figures]」这种可操作提示。指标解析本身（:func:`parse_metrics`）不需要
    pandas，因此核心路径不受影响。
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise OptionalDependencyError(
            "compare_runs() 需要 pandas（可选依赖）。"
            "安装：pip install -e .[figures]。"
            "注意 parse_metrics/summarize 不需要 pandas，核心指标解析不受影响。"
        ) from exc

    if not isinstance(runs, Mapping):
        raise TypeError("compare_runs(runs): expected dict[run_name, dict[metric, list[float]]]")

    per_run = {str(name): summarize(metrics) for name, metrics in runs.items()}
    metrics: list[str] = []
    for stats in per_run.values():
        for metric in stats:
            if metric not in metrics:
                metrics.append(metric)

    data: dict[str, dict[tuple[str, str], float]] = {}
    for run, stats in per_run.items():
        row: dict[tuple[str, str], float] = {}
        for metric in metrics:
            values = stats.get(metric, {})
            for stat in _STATS:
                row[(metric, stat)] = float(values.get(stat, float("nan")))
        data[run] = row

    frame = pd.DataFrame.from_dict(data, orient="index") if data else pd.DataFrame()
    if frame.shape[1]:
        frame.columns = pd.MultiIndex.from_tuples(list(frame.columns))
    frame.index.name = "run"
    if not metrics:
        frame = pd.DataFrame(index=list(per_run))
        frame.index.name = "run"
    return frame


def flatten_columns(frame):
    """Flatten a MultiIndex column frame to ``metric_stat`` string columns."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise OptionalDependencyError(
            "to_latex_table() 处理 DataFrame 时需要 pandas（可选依赖）。"
            "安装：pip install -e .[figures]。可先用 frame_to_rows() 拿到无依赖的 dict 列表。"
        ) from exc

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("flatten_columns(frame): expected a pandas.DataFrame")
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy()
    out = frame.copy()
    out.columns = [
        "_".join(str(part) for part in tup if part is not None and str(part) != "")
        for tup in frame.columns
    ]
    return out


def compare_runs_flat(runs: Mapping[str, Mapping[str, Any]]):
    """Flat variant of :func:`compare_runs` (``acc_mean``, ``acc_std``, ...)."""
    return flatten_columns(compare_runs(runs))


def frame_to_rows(frame) -> list[dict]:
    """Convert a DataFrame to ``list[dict]`` rows with a ``"name"`` label key."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame_to_rows(frame): expected a pandas.DataFrame")
    flat = flatten_columns(frame)
    flat = flat.reset_index()
    index_column = flat.columns[0]
    rows: list[dict] = []
    for _, record in flat.iterrows():
        label = record.iloc[0]
        row: dict[str, Any] = {"name": "" if pd.isna(label) else str(label)}
        for column in flat.columns[1:]:
            row[str(column)] = record[column]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #

_NAME_KEYS = ("name", "run", "model", "method", "approach", "config", "variant", "setting", "label")
_STD_SUFFIXES = ("_std", "_stddev", "_sd")
_MEAN_SUFFIXES = ("_mean", "_avg")

_HEADER_LABELS = {
    "name": "Run",
    "run": "Run",
    "model": "Model",
    "method": "Method",
    "approach": "Method",
    "config": "Config",
    "variant": "Variant",
    "setting": "Setting",
    "label": "Setting",
}

_LATEX_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_RE = re.compile("|".join(re.escape(k) for k in _LATEX_MAP))


def _escape(value: Any) -> str:
    return _LATEX_RE.sub(lambda m: _LATEX_MAP[m.group(0)], str(value))


def _classify_key(key: str) -> tuple[str, str]:
    """Return ``(base_metric, kind)`` where kind is ``"value"`` or ``"std"``."""
    lower = key.lower()
    for suffix in _STD_SUFFIXES:
        if lower.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], "std"
    for suffix in _MEAN_SUFFIXES:
        if lower.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], "value"
    return key, "value"


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))


def _bold(cell: str) -> str:
    if cell.startswith("$") and cell.endswith("$") and len(cell) >= 2:
        return "$\\mathbf{" + cell[1:-1] + "}$"
    return "\\textbf{" + cell + "}"


def _row_label(row: Mapping[str, Any], name_key: str) -> str:
    for key in (name_key,) + _NAME_KEYS:
        if key in row:
            return str(row[key])
    return ""


def _cell_parts(row: Mapping[str, Any], base: str) -> tuple[Any, Any]:
    central: Any = None
    for key in (base,) + tuple(base + s for s in _MEAN_SUFFIXES):
        if key in row:
            central = row[key]
            break
    uncertainty: Any = None
    for key in tuple(base + s for s in _STD_SUFFIXES):
        if key in row:
            uncertainty = row[key]
            break
    return central, uncertainty


def _empty_table(caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            "\\caption{%s}" % caption,
            "\\label{tab:%s}" % label,
            r"\begin{tabular}{l}",
            r"\toprule",
            r"Run \\",
            r"\midrule",
            r"-- \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _looks_like_dataframe(obj: Any) -> bool:
    """鸭子类型识别 DataFrame（pandas 缺席时用）。

    判据刻意保守：类名含 ``DataFrame`` 且具备 ``to_dict``/``columns``。
    单纯有 ``to_dict`` 的普通对象不该被当成 DataFrame——那会把它错误地
    展开成多行。
    """
    if obj is None or isinstance(obj, (list, tuple, Mapping)):
        return False
    cls = type(obj)
    if "DataFrame" not in cls.__name__:
        return False
    return hasattr(obj, "to_dict") and hasattr(obj, "columns")


def to_latex_table(
    rows: list[dict],
    caption: str = "",
    label: str = "",
    decimals: int = 2,
    bold_best: bool = True,
    higher_is_better: dict[str, bool] | None = None,
) -> str:
    """Render ``rows`` as a booktabs three-line table.

    * each row dict needs a ``name`` / ``run`` / ``model`` key used as row label;
    * remaining keys become columns; ``_std`` / ``_stddev`` / ``_sd`` keys are
      folded into their base metric and rendered as ``mean \\pm std``;
    * ``nan`` / missing values render as ``--``, non numeric values are escaped;
    * ``bold_best=True`` wraps the best value of each metric in ``\\textbf{}``
      (or ``\\mathbf{}`` inside the ``\\pm`` math cell), using
      :func:`_higher_is_better` unless ``higher_is_better`` overrides it.

    **不需要 pandas。** 唯一的 pandas 用法是判断入参是不是 DataFrame；渲染本身
    完全是纯 Python。早期实现无条件 ``import pandas``，于是缺 pandas 时
    ``s5_analysis`` 里那个 try/except 会把它吞成一条 warning——
    **论文主表就这样静默消失了**（实验数据完好，但交付的论文里没有结果表）。
    pandas 缺席时改为按鸭子类型识别 DataFrame（有 ``to_dict`` 且名字是
    DataFrame 的对象），其余路径原样工作。
    """
    frame_module = None
    try:
        import pandas as pd

        frame_module = pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    if frame_module is not None and isinstance(rows, frame_module.DataFrame):
        rows = frame_to_rows(rows)
    elif frame_module is None and _looks_like_dataframe(rows):
        # 没有 pandas 也可能收到 DataFrame 对象（调用方装了 pandas）；
        # 用鸭子类型处理，避免把一整个 DataFrame 当成一个 dict 行。
        rows = frame_to_rows(rows)
    if rows is None:
        rows = []
    if isinstance(rows, Mapping):
        rows = [dict(rows)]
    if not isinstance(rows, (list, tuple)):
        raise TypeError("to_latex_table(rows): expected list[dict], got %s" % type(rows).__name__)

    norm_rows: list[dict] = []
    for item in rows:
        if not isinstance(item, Mapping):
            raise TypeError(
                "to_latex_table(rows): every row must be a dict-like mapping, got %s"
                % type(item).__name__
            )
        norm_rows.append(dict(item))

    if not norm_rows:
        return _empty_table(caption, label)

    name_key = "name"
    for key in _NAME_KEYS:
        if any(key in row for row in norm_rows):
            name_key = key
            break

    bases: list[str] = []
    for row in norm_rows:
        for key in row:
            if key in _NAME_KEYS or str(key).startswith("__"):
                continue
            base, _kind = _classify_key(str(key))
            if base and base not in bases:
                bases.append(base)

    body: list[list[tuple[str, float | None]]] = []
    for row in norm_rows:
        cells: list[tuple[str, float | None]] = []
        for base in bases:
            central, uncertainty = _cell_parts(row, base)
            raw_number = _to_float(central)
            if raw_number is not None and math.isnan(raw_number):
                # nan means "no measurement", not a label
                central = None
            number = None if raw_number is None or math.isnan(raw_number) else raw_number
            if number is None:
                if central is None:
                    cells.append(("--", None))
                else:
                    cells.append((_escape(central), None))
                continue
            std = _to_float(uncertainty)
            if std is not None and math.isnan(std):
                std = None
            if std is None:
                cells.append(("%.*f" % (decimals, number), number))
            else:
                cells.append(
                    ("$%.*f \\pm %.*f$" % (decimals, number, decimals, std), number)
                )
        body.append(cells)

    if bold_best:
        for index, base in enumerate(bases):
            numeric = [
                (row_index, cells[index][1])
                for row_index, cells in enumerate(body)
                if cells[index][1] is not None
            ]
            if not numeric:
                continue
            best = (
                max(value for _i, value in numeric)
                if _higher_is_better(base, higher_is_better)
                else min(value for _i, value in numeric)
            )
            for row_index, value in numeric:
                if _close(float(value), float(best)):
                    text, val = body[row_index][index]
                    body[row_index][index] = (_bold(text), val)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        "\\caption{%s}" % caption,
        "\\label{tab:%s}" % label,
        r"\begin{tabular}{" + "l" + "c" * len(bases) + "}",
        r"\toprule",
        " & ".join([_escape(_HEADER_LABELS.get(name_key, name_key))] + [_escape(b) for b in bases]) + r" \\",
        r"\midrule",
    ]
    for row, cells in zip(norm_rows, body):
        lines.append(" & ".join([_escape(_row_label(row, name_key))] + [c[0] for c in cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)
