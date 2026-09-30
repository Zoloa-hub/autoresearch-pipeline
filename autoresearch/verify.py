"""论文数字溯源校验（``cli verify``）。

自动科研最容易出现的、也最难被肉眼发现的故障是：**论文里的数字与实验产出不符**。
LLM 在改写、四舍五入、"顺手修饰" 的过程中都可能引入偏差，而这些偏差在 PDF 里
看起来完全正常。

本模块做一件很窄但很硬的事：

1. 从 ``metrics/`` 重算每个 run 的指标统计（mean/std/min/max/first/final/best）；
2. 从 ``paper/sections/*.tex`` 与 ``analysis/*.json`` 里抽出所有数值 token；
3. 判定每个数值是否能在证据集合里找到（容许 ``rel_tol`` 内的匹配）；
4. 输出报告：**未溯源的数字**（论文里有、证据里没有）+ **未被引用的指标**
   （有证据但论文没提）。

它**不是**完整的形式化验证：数值可能被合法地重新组合（差值、百分比）。
因此报告分三级：``exact``（直接命中）、``derived``（可由两个证据值通过四则运算
得到）、``unmatched``（找不到来源）。只有第三级需要人工复核——这个筛子把
"需要人看的东西" 从上千个数字降到个位数。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: 匹配数值 token：含科学计数法、千分位、百分号、负号。
NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])"      # 1,234.5
    r"|(?<![\w.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?(?![\w])"          # 12.34 / 1.2e-3
    r"|(?<![\w.])[-+]?\d+(?:[eE][-+]?\d+)?(?![\w.])"              # 12 / 1e5
)

#: 排除这些上下文里的数字——它们不是实验结果。
_IGNORE_CONTEXT = (
    "usepackage", "documentclass", "begin{", "end{", "includegraphics",
    "linewidth", "textwidth", "columnwidth", "vspace", "hspace", "baselineskip",
    "label{", "ref{", "cite", "geometry", "fontsize", "scriptsize", "small",
    "\\rule", "cmidrule", "toprule", "midrule", "bottomrule", "tabular",
    "section", "subsection", "section*", "item", "date", "author", "title",
)

#: 允许的浮点误差（相对）。
DEFAULT_REL_TOL = 1e-3


@dataclass
class NumberFinding:
    value: float
    raw: str
    source: str          # 出现位置（相对路径:行号）
    context: str         # 上下文片段
    status: str = "unmatched"     # exact | derived | unmatched
    matched_metric: str = ""
    matched_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "raw": self.raw,
            "source": self.source,
            "context": self.context,
            "status": self.status,
            "matched_metric": self.matched_metric,
            "matched_value": self.matched_value,
        }


@dataclass
class VerifyReport:
    run_dir: str
    evidence_values: list[dict[str, Any]] = field(default_factory=list)
    exact: list[NumberFinding] = field(default_factory=list)
    derived: list[NumberFinding] = field(default_factory=list)
    unmatched: list[NumberFinding] = field(default_factory=list)
    unused_metrics: list[str] = field(default_factory=list)
    files_scanned: int = 0
    numbers_found: int = 0

    @property
    def ok(self) -> bool:
        return not self.unmatched

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "ok": self.ok,
            "files_scanned": self.files_scanned,
            "numbers_found": self.numbers_found,
            "evidence_count": len(self.evidence_values),
            "exact": len(self.exact),
            "derived": len(self.derived),
            "unmatched": [f.to_dict() for f in self.unmatched],
            "unused_metrics": self.unused_metrics,
        }

    def render(self, max_unmatched: int = 40, max_unused: int = 40) -> str:
        lines = [
            "# 数字溯源校验报告",
            "",
            f"- 运行目录：`{self.run_dir}`",
            f"- 扫描文件：{self.files_scanned}",
            f"- 论文/分析中的数值 token：{self.numbers_found}",
            f"- 证据库数值：{len(self.evidence_values)}",
            f"- 直接命中：{len(self.exact)}　派生可解释：{len(self.derived)}　"
            f"**未溯源：{len(self.unmatched)}**",
            "",
        ]
        if self.ok:
            lines += ["✅ **论文中的每个数字都能在实验证据中找到来源。**", ""]

        if self.unmatched:
            lines += [
                "## ⚠️ 未溯源数字（需人工复核）",
                "",
                "这些数字出现在论文或分析文档中，但按相对误差 "
                f"{DEFAULT_REL_TOL:g} 无法在指标证据里找到匹配。",
                "可能原因：① 正确的派生量（如差值、百分比）但未被自动识别；"
                "② 四舍五入到证据精度之外；③ **幻觉数字**。",
                "",
                "| 原始 token | 位置 | 上下文 |",
                "|---|---|---|",
            ]
            for f in self.unmatched[:max_unmatched]:
                ctx = f.context.replace("|", "\\|")[:110]
                lines.append(f"| `{f.raw}` | `{f.source}` | {ctx} |")
            if len(self.unmatched) > max_unmatched:
                lines.append(f"| … | | 另有 {len(self.unmatched) - max_unmatched} 项 |")
            lines.append("")

        if self.unused_metrics:
            lines += [
                "## 有证据但论文未引用",
                "",
                "不是错误，但值得确认是否有意省略（评审常问「为什么不报 X」）。",
                "",
            ]
            lines += [f"- `{m}`" for m in self.unused_metrics[:max_unused]]
            if len(self.unused_metrics) > max_unused:
                lines.append(f"- … 另有 {len(self.unused_metrics) - max_unused} 项")
            lines.append("")

        lines += [
            "## 方法说明",
            "",
            "本校验是**范围检查**而非形式化证明：它保证论文数字能落到证据网格上，",
            "但不保证语义正确（例如把 accuracy 写成 f1 仍会通过）。",
            "语义层的校验由 s8 评审与人工复核承担。",
            "",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 证据提取
# --------------------------------------------------------------------------- #


def collect_evidence(run_dir: Path) -> tuple[list[float], dict[str, float]]:
    """从 metrics/ 与 analysis/ 收集全部可信数值。返回 (值列表, 名字→值)。"""
    values: list[float] = []
    named: dict[str, float] = {}
    run_dir = Path(run_dir)

    metrics_dir = run_dir / "metrics"
    try:
        from .tools.metrics import parse_metrics, summarize
    except Exception:  # pragma: no cover
        parse_metrics = None  # type: ignore
        summarize = None  # type: ignore

    series_map: dict[str, dict[str, list[float]]] = {}
    for path in _iter_metric_files(metrics_dir):
        run_name = path.parent.name if path.parent != metrics_dir else path.stem
        parsed: dict[str, list[float]] = {}
        if parse_metrics is not None:
            try:
                parsed = {k: v for k, v in (parse_metrics(path) or {}).items()
                          if isinstance(v, list) and v}
            except Exception:
                parsed = {}
        if not parsed:
            parsed = _jsonl_fallback(path)
        for metric, vals in parsed.items():
            series_map.setdefault(run_name, {})[metric] = [float(v) for v in vals]
            for v in vals:
                values.append(float(v))

    # 优先使用阶段 ⑤ 落盘的**权威**统计（含跨种子口径），它才是论文表格的来源；
    # 否则退回自己从原始指标重算。
    analysis_summary = run_dir / "analysis" / "metrics_summary.json"
    if analysis_summary.exists():
        try:
            loaded = json.loads(analysis_summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            _absorb_stats(loaded, values, named)

    summary: dict[str, Any] = {}
    if summarize is not None and series_map:
        try:
            summary = summarize(series_map) or {}
        except Exception:
            summary = {}
    if not summary:
        summary = _inline_summary(series_map)

    _absorb_stats(summary, values, named)

    # analysis/*.json 里的数字若已由上面覆盖则无所谓，这里额外收集图表名映射
    return values, named


def _absorb_stats(stats: Any, values: list[float], named: dict[str, float]) -> None:
    """把任意层级的 ``{名字: {统计量: 数值}}`` 结构摊平成 ``名字.统计量 → 数值``。

    容忍三种形态：单层（``metric → stat → float``）、双层（``variant → metric → stat``）、
    以及混杂形态（既含 metric 子字典又含 ``count``/``_seeds`` 这类标量）。
    摊平是为了让「论文里的一个数字」能在证据表里被命名地找到。
    """

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if str(key).startswith("_"):
                    continue
                full = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, bool):
                    continue
                if isinstance(child, (int, float)):
                    value = float(child)
                    values.append(value)
                    named[full] = value
                elif isinstance(child, dict):
                    walk(child, full)
                elif isinstance(child, list) and all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in child
                ):
                    for i, x in enumerate(child):
                        named[f"{full}[{i}]"] = float(x)
                        values.append(float(x))

    walk(stats, "")


def _iter_metric_files(metrics_dir: Path) -> Iterable[Path]:
    if not metrics_dir.is_dir():
        return []
    return [
        p for p in sorted(metrics_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in (".csv", ".jsonl", ".json", ".log", ".txt")
    ]


def _jsonl_fallback(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return out
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                for k, v in obj.items():
                    try:
                        out.setdefault(str(k), []).append(float(v))
                    except (TypeError, ValueError):
                        continue
        return out
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return out
    header = [h.strip() for h in re.split(r"[,\t;]", lines[0])]
    for line in lines[1:]:
        cells = [c.strip() for c in re.split(r"[,\t;]", line)]
        for key, cell in zip(header, cells):
            try:
                out.setdefault(key, []).append(float(cell))
            except (TypeError, ValueError):
                continue
    return out


def _inline_summary(series_map: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for run, metrics in series_map.items():
        summary[run] = {}
        for name, vals in metrics.items():
            nums = [float(v) for v in vals]
            if not nums:
                continue
            n = len(nums)
            mean = sum(nums) / n
            var = sum((x - mean) ** 2 for x in nums) / n
            lower = any(t in name.lower() for t in ("loss", "error", "rmse", "mae", "mse"))
            summary[run][name] = {
                "count": n, "mean": mean, "std": var ** 0.5,
                "min": min(nums), "max": max(nums), "first": nums[0], "final": nums[-1],
                "best": min(nums) if lower else max(nums),
            }
    return summary


# --------------------------------------------------------------------------- #
# 论文侧数字提取
# --------------------------------------------------------------------------- #


def extract_numbers_from_text(text: str, source: str) -> list[NumberFinding]:
    """抽出文本里的数值 token。

    刻意跳过两类噪声，否则报告会被无关数字淹没、失去可用性：

    * **代码/路径/标识符**（行内反引号里的内容，如 ``metrics.csv``、``seed_0``）；
    * **明确标注为告警/说明的附录区块**（``## 附录`` 之后的排版性文字）。
    """
    findings: list[NumberFinding] = []
    in_appendix = False
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.startswith("#") and ("附录" in stripped or "appendix" in stripped.lower()):
            in_appendix = True
            continue
        if in_appendix:
            continue
        # 去掉行内代码，避免把文件名/字段名里的数字当成实验结果
        scannable = re.sub(r"`[^`]*`", " ", line)
        lowered = scannable.lower()
        if any(tok in lowered for tok in _IGNORE_CONTEXT):
            if not any(ch.isdigit() for ch in scannable):
                continue
        for match in NUMBER_RE.finditer(scannable):
            raw = match.group(0)
            cleaned = raw.replace(",", "")
            try:
                value = float(cleaned)
            except ValueError:
                continue
            context = _context(scannable, match.start(), match.end())
            findings.append(
                NumberFinding(value=value, raw=raw, source=f"{source}:{line_no}", context=context)
            )
    return findings


def _context(line: str, start: int, end: int, width: int = 60) -> str:
    a = max(0, start - width)
    b = min(len(line), end + width)
    snippet = line[a:b].strip()
    return re.sub(r"\s+", " ", snippet)


def scan_paper(run_dir: Path) -> tuple[list[NumberFinding], list[Path]]:
    run_dir = Path(run_dir)
    findings: list[NumberFinding] = []
    files: list[Path] = []

    paper_dir = run_dir / "paper"
    for path in sorted(paper_dir.rglob("*.tex")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files.append(path)
        findings.extend(extract_numbers_from_text(text, _rel(path, run_dir)))

    # 分析文档也纳入（它们是论文数字的上游）
    analysis_dir = run_dir / "analysis"
    for path in sorted(analysis_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files.append(path)
        findings.extend(extract_numbers_from_text(text, _rel(path, run_dir)))
    return findings, files


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- #
# 匹配
# --------------------------------------------------------------------------- #


def _close(a: float, b: float, rel_tol: float) -> bool:
    if a == b:
        return True
    if abs(b) < 1e-12:
        return abs(a) < rel_tol
    return abs(a - b) <= rel_tol * max(abs(a), abs(b))


def _decimals(raw: str) -> int:
    """原始 token 里小数点后的位数。``"0.94"`` → 2，``"94"`` → 0。"""
    text = (raw or "").replace(",", "")
    if "e" in text.lower():
        return 6
    return len(text.split(".")[1]) if "." in text else 0


def _matches_token(token_value: float, decimals: int, evidence_value: float, rel_tol: float) -> bool:
    """判定证据值能否解释论文里的一个数值 token。

    论文里的数字是**四舍五入后**的，所以不能要求逐位相等。这里接受两种解释：

    1. **舍入一致**：证据值按 token 的精度四舍五入后恰好等于 token
       （``0.9404`` → ``0.94``）；这是论文表格与摘要里绝大多数数字的来源；
    2. **相对误差**：在 ``rel_tol`` 内接近（覆盖把 ``0.94`` 写成 ``0.940`` 之类的写法）。

    这条规则刻意宽松——校验器的价值在于把「上千个数字」缩到「几个需要人看的东西」，
    一个只会报警的校验器和没有校验器没区别。
    """
    if _close(token_value, evidence_value, rel_tol):
        return True
    try:
        if round(evidence_value, decimals) == token_value:
            return True
    except (OverflowError, ValueError):
        pass
    return False


def match_findings(
    findings: list[NumberFinding],
    evidence: list[float],
    named: dict[str, float],
    rel_tol: float = DEFAULT_REL_TOL,
) -> tuple[list[NumberFinding], list[NumberFinding], list[NumberFinding], list[str]]:
    exact: list[NumberFinding] = []
    derived: list[NumberFinding] = []
    unmatched: list[NumberFinding] = []
    # 记录被引用到的**证据键**（不是被引用到的数值），用于「有证据但论文未提」判定
    used_keys: set[str] = set()

    pairs = _evidence_pairs(named)
    sorted_ev = sorted(set(evidence))

    for f in findings:
        hit_name = ""
        matched = 0.0
        decimals = _decimals(f.raw)
        for value, name in pairs:
            if _matches_token(f.value, decimals, value, rel_tol):
                hit_name, matched = name, value
                used_keys.add(name)
                break
        if not hit_name:
            for value in sorted_ev:
                if _matches_token(f.value, decimals, value, rel_tol):
                    hit_name, matched = f"(unnamed value {value:.6g})", value
                    break
        if hit_name:
            f.status, f.matched_metric, f.matched_value = "exact", hit_name, matched
            exact.append(f)
            continue

        # 派生检测：两证据之和/差（覆盖 Δ、增量、相对提升的分子）
        deriv = _find_derivation(f.value, sorted_ev, rel_tol)
        if deriv:
            f.status, f.matched_metric, f.matched_value = "derived", deriv[1], deriv[0]
            derived.append(f)
            continue

        unmatched.append(f)

    used_metrics = {_metric_of(k) for k in used_keys}
    unused = sorted(
        {k for k, _ in pairs if _metric_of(k) not in used_metrics}
    )
    return exact, derived, unmatched, unused


def _evidence_pairs(named: dict[str, float]) -> list[tuple[float, str]]:
    return [(float(v), k) for k, v in named.items() if isinstance(v, (int, float))]


def _metric_of(name: Any) -> str:
    """把 ``run.metric.stat`` 归并到 ``run.metric``，用于「未被引用」判定。

    对非字符串输入直接返回其字符串形式——校验器是诊断工具，
    不该因为一个意外的类型而崩掉整天的工作。
    """
    if not isinstance(name, str):
        return str(name)
    parts = name.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else name


def _find_derivation(value: float, evidence: list[float], rel_tol: float) -> tuple[float, str] | None:
    """检查 value 是否**严格**等于两个证据值之差/和。

    这里必须用严格容差（默认 1e-9），不能用匹配主证据时的宽松容差。
    原因：证据表有上千个数值，两两组合是百万量级；在 ``1e-3`` 的容差下几乎
    任何随机数字都能被某个组合「解释」，校验器会退化成永远说 OK 的橡皮图章
    ——这比没有校验器更糟，因为它给出虚假的安心感。

    真正由证据派生的数字（差值、提升量）在 IEEE-754 下要么精确相等，
    要么只差最后几个 ulp；``1e-9`` 足以覆盖后者而不放过前者。
    """
    strict = min(abs(rel_tol), 1e-9)
    sample = evidence[:60]
    for i, a in enumerate(sample):
        for b in sample[i + 1:]:
            for candidate, label in ((a - b, "a-b"), (a + b, "a+b"), (b - a, "b-a")):
                if _close(value, candidate, strict):
                    return candidate, f"derived({label}: {a:.6g}, {b:.6g})"
    return None


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def verify_run(run_dir: Path, rel_tol: float = DEFAULT_REL_TOL) -> VerifyReport:
    run_dir = Path(run_dir)
    evidence, named = collect_evidence(run_dir)
    findings, files = scan_paper(run_dir)
    exact, derived, unmatched, unused = match_findings(findings, evidence, named, rel_tol)
    return VerifyReport(
        run_dir=run_dir.as_posix(),
        evidence_values=[{"name": k, "value": v} for k, v in sorted(named.items())],
        exact=exact,
        derived=derived,
        unmatched=unmatched,
        unused_metrics=unused,
        files_scanned=len(files),
        numbers_found=len(findings),
    )


__all__ = [
    "DEFAULT_REL_TOL",
    "NumberFinding",
    "VerifyReport",
    "collect_evidence",
    "extract_numbers_from_text",
    "match_findings",
    "scan_paper",
    "verify_run",
]
