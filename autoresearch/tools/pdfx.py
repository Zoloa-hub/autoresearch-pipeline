"""PDF 解析工具（CONTRACTS §5）。

* 首选 ``fitz`` (PyMuPDF)；``ImportError`` 时回退 ``pdftotext``；两者都不可用时返回空串
  并记 warning（**不 raise**）。
* 唯一允许抛出的异常是文件不存在时的 :class:`FileNotFoundError`。
* 所有启发式切分都是 best-effort：结果给 LLM 读，可读优先于完美。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import statistics
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

try:  # 兄弟模块（并行开发中）不存在时优雅降级
    from ..logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover - 依赖兄弟模块
    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)

log = get_logger("autoresearch.tools.pdfx")

__all__ = [
    "extract_text",
    "extract_sections",
    "extract_references",
    "extract_metadata",
    "download_pdf",
    "pdf_to_markdown",
]

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36 AutoResearch/0.1"
)
_MAX_PDF_BYTES = 40 * 1024 * 1024
_MAX_REFS = 200
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_FITZ: Any = None
_FITZ_TRIED = False


# --------------------------------------------------------------------------- #
# 后端探测
# --------------------------------------------------------------------------- #

def _fitz():
    """惰性导入 PyMuPDF；不可用时返回 None（模块级缓存）。"""
    global _FITZ, _FITZ_TRIED
    if not _FITZ_TRIED:
        _FITZ_TRIED = True
        try:
            import fitz  # type: ignore

            _FITZ = fitz
        except Exception as exc:  # pragma: no cover - 取决于环境
            log.warning("PyMuPDF (fitz) unavailable: %s", exc)
            _FITZ = None
    return _FITZ


def _pdftotext() -> str | None:
    return shutil.which("pdftotext")


def _clean_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _require_file(pdf_path: Any) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return path


def _pdftotext_extract(path: Path, max_pages: int | None = None) -> str:
    exe = _pdftotext()
    if not exe:
        return ""
    cmd = [exe]
    if max_pages:
        cmd += ["-l", str(int(max_pages))]
    cmd += [str(path), "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception as exc:
        log.warning("pdftotext failed for %s: %s", path, exc)
        return ""
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace")[:300]
        log.warning("pdftotext rc=%s for %s: %s", proc.returncode, path, err)
        return ""
    return (proc.stdout or b"").decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# 正文抽取
# --------------------------------------------------------------------------- #

def extract_text(pdf_path: Path, max_pages: int | None = None) -> str:
    """按页抽取纯文本，页间以 ``\\n`` 连接。缺文件抛 FileNotFoundError。"""
    path = _require_file(pdf_path)
    fitz = _fitz()
    if fitz is not None:
        try:
            with fitz.open(str(path)) as doc:
                parts: list[str] = []
                for i, page in enumerate(doc):
                    if max_pages is not None and i >= int(max_pages):
                        break
                    parts.append(page.get_text("text") or "")
                return "\n".join(parts)
        except Exception as exc:
            log.warning("fitz extract_text failed for %s: %s", path, exc)
    text = _pdftotext_extract(path, max_pages)
    if text:
        return text
    if _pdftotext() is None:
        log.warning("no PDF text backend available (fitz/pdftotext missing); returning '' for %s", path)
    return ""


# --------------------------------------------------------------------------- #
# 章节切分
# --------------------------------------------------------------------------- #

# 规范名 -> 标题正则（顺序即匹配优先级）
_SECTION_SPECS: tuple[tuple[str, str], ...] = (
    ("abstract", r"abstract"),
    ("introduction", r"introductions?|intro"),
    ("related_work", r"related\s+works?|literature\s+review|prior\s+work|previous\s+work"),
    ("background", r"background|preliminar(?:y|ies)|notation"),
    ("method", r"methods?|methodolog(?:y|ies)|approach(?:es)?|our\s+approach|"
               r"proposed\s+(?:method|approach|framework|model)|models?|framework|"
               r"architecture|system\s+overview"),
    ("experiments", r"experiments?|experimental\s+(?:setup|settings|protocol)|setup|"
                    r"implementation\s+details|training\s+details"),
    ("evaluation", r"evaluations?|empirical\s+(?:study|evaluation|results)"),
    ("results", r"results?|findings|main\s+results"),
    ("discussion", r"discussions?"),
    ("ablation", r"ablations?(?:\s+stud(?:y|ies))?"),
    ("conclusion", r"conclusions?|concluding\s+remarks|conclusion\s+and\s+future\s+work"),
    ("limitations", r"limitations?|threats?\s+to\s+validity"),
    ("references", r"references?|bibliograph(?:y|ies)"),
    ("appendix", r"appendi(?:x|ces)|supplementar(?:y|ies)|supplementary\s+material"),
)

# 编号前缀：`1.` / `1.1.` / `II.` / `A.`(附录) / `§2` 等（数字分支允许不带分隔符，如 "2 Method"）
_NUM_PAT = r"(?:§\s*\d+(?:\.\d+)*|\d+(?:\.\d+)*[.)]?|[IVXLC]{1,5}[.)]|[A-Z][.)])"
_HEADING_RES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        canon,
        re.compile(
            r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*"
            r"(?:" + _NUM_PAT + r"\s*)?"
            r"(?:\*\*|__)?\s*(?:" + pat + r")\s*"
            r"(?:\*\*|__)?\s*:?\s*\.?\s*$",
            re.I,
        ),
    )
    for canon, pat in _SECTION_SPECS
)

_REFERENCES_NAMES = {"references"}


def _match_heading(line: str) -> str | None:
    text = (line or "").strip()
    if not text or len(text) > 90:
        return None
    for canon, rx in _HEADING_RES:
        if rx.match(text):
            return canon
    return None


def _flush_sections(sections: dict[str, list[str]], current: str, buf: list[str]) -> None:
    body = "\n".join(buf).strip("\n")
    if body.strip():
        sections.setdefault(current, []).append(body)


def extract_sections(pdf_path: Path) -> dict[str, str]:
    """按常见标题启发式切分正文；首个标题之前的内容放在 ``__preamble__``。

    命中 ``References`` 后停止识别新标题（避免把参考文献吞进 Conclusion）。
    """
    text = extract_text(pdf_path)
    if not text.strip():
        return {}
    sections: dict[str, list[str]] = {}
    current = "__preamble__"
    buf: list[str] = []
    stopped = False
    for line in text.splitlines():
        canon = None if stopped else _match_heading(line)
        if canon is None:
            buf.append(line)
            continue
        _flush_sections(sections, current, buf)
        buf = []
        current = canon
        if canon in _REFERENCES_NAMES:
            stopped = True
    _flush_sections(sections, current, buf)
    return {name: "\n\n".join(chunks).strip() for name, chunks in sections.items()}


# --------------------------------------------------------------------------- #
# 参考文献
# --------------------------------------------------------------------------- #

_NUMBERED_SPLIT = re.compile(r"(?m)(?=^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}[.)])\s+\S)")
_YEAR_SPLIT = re.compile(r"(?m)(?=^[A-Z][^\n]{0,140}?\((?:19|20)\d{2}[a-z]?\)\s*[.,;)])")


def _references_tail(text: str) -> str | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _match_heading(line) == "references":
            tail = "\n".join(lines[i + 1:])
            if tail.strip():
                return tail
    return None


def _split_reference_entries(tail: str) -> list[str]:
    numbered = [c for c in _NUMBERED_SPLIT.split(tail) if c.strip()]
    if len(numbered) >= 2:
        return numbered
    by_year = [c for c in _YEAR_SPLIT.split(tail) if c.strip()]
    if len(by_year) >= 2:
        return by_year
    paras = [c for c in re.split(r"\n\s*\n", tail) if c.strip()]
    if len(paras) >= 2:
        return paras
    return [tail] if tail.strip() else []


def extract_references(pdf_path: Path) -> list[str]:
    """抽取 References/Bibliography 之后的条目（清洗、去短碎片、上限 200）。"""
    text = extract_text(pdf_path)
    if not text.strip():
        return []
    tail = _references_tail(text)
    if tail is None:
        # 没有 References 标题时，仅当正文里存在编号条目才退化为全文切分
        if _NUMBERED_SPLIT.search(text):
            tail = text
        else:
            return []
    out: list[str] = []
    for raw in _split_reference_entries(tail):
        entry = _clean_ws(raw).strip(" .;,、")
        if len(entry) < 20:
            continue
        if re.fullmatch(r"[\d\W_]+", entry):
            continue
        if entry.lower() in {"references", "bibliography"}:
            continue
        out.append(entry)
        if len(out) >= _MAX_REFS:
            break
    return out


# --------------------------------------------------------------------------- #
# 元数据
# --------------------------------------------------------------------------- #

_TITLE_SKIP_RE = re.compile(
    r"^(?:arxiv:|doi:|https?://|www\.|©|copyright|keywords?\b|abstract\b|"
    r"proceedings\b|journal\b|volume\b|vol\.|pp\.|preprint\b|under review\b)",
    re.I,
)


def _split_authors(raw: Any) -> list[str]:
    text = _clean_ws(raw)
    if not text:
        return []
    if ";" in text:
        parts = text.split(";")
    elif re.search(r"\band\b", text, re.I):
        parts = re.split(r"\s+and\s+", text, flags=re.I)
    else:
        parts = text.split(",")
        if any(len(p.split()) > 4 for p in parts):
            parts = [text]
    return [_clean_ws(p).strip(",") for p in parts if _clean_ws(p).strip(",")]


def _year_from_pdf_date(value: Any) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(m.group(0)) if m else None


def _guess_year(text: str) -> int | None:
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(text or "")]
    years = [y for y in years if 1900 <= y <= 2100]
    if not years:
        return None
    counts = Counter(years)
    return sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]


def _guess_title(first_page_text: str) -> str:
    lines = [ln.strip() for ln in (first_page_text or "").splitlines()]
    cands: list[str] = []
    for ln in lines[:25]:
        if len(ln) < 12 or len(ln) > 250:
            continue
        if not re.search(r"[A-Za-z]", ln):
            continue
        if "@" in ln or _TITLE_SKIP_RE.match(ln):
            continue
        if re.fullmatch(r"[\W\d_]+", ln):
            continue
        cands.append(ln)
        if len(cands) >= 2:
            break
    return _clean_ws(" ".join(cands))


def extract_metadata(pdf_path: Path) -> dict:
    """尽力而为地抽取 title/authors/year/doi/pages（``source`` 标明后端）。"""
    path = _require_file(pdf_path)
    meta: dict[str, Any] = {
        "title": "", "authors": [], "year": None, "doi": "", "pages": 0, "source": "",
    }
    first_page_text = ""
    full_text = ""
    fitz = _fitz()
    if fitz is not None:
        try:
            with fitz.open(str(path)) as doc:
                meta["pages"] = int(getattr(doc, "page_count", 0) or 0)
                md = doc.metadata or {}
                meta["title"] = _clean_ws(md.get("title") or "")
                meta["authors"] = _split_authors(md.get("author") or "")
                meta["year"] = _year_from_pdf_date(md.get("creationDate") or md.get("modDate"))
                if meta["pages"]:
                    first_page_text = doc[0].get_text("text") or ""
                if not meta["year"]:
                    full_text = "\n".join((doc[i].get_text("text") or "") for i in range(meta["pages"]))
                meta["source"] = "fitz"
        except Exception as exc:
            log.warning("fitz extract_metadata failed for %s: %s", path, exc)
            meta["source"] = ""

    if not meta["source"]:
        text = _pdftotext_extract(path)
        if text:
            pages = [p for p in text.split("\f") if p.strip()]
            meta["pages"] = meta["pages"] or len(pages)
            first_page_text = first_page_text or (pages[0] if pages else text)
            full_text = full_text or text
            meta["source"] = "pdftotext"

    if not first_page_text and not full_text:
        full_text = extract_text(path)
        first_page_text = full_text

    if not meta["doi"]:
        m = _DOI_RE.search(first_page_text or "")
        if not m and full_text:
            m = _DOI_RE.search(full_text)
        if m:
            meta["doi"] = m.group(0).rstrip(".,;)")
    if not meta["year"]:
        meta["year"] = _guess_year(full_text or first_page_text)
    if not meta["title"]:
        meta["title"] = _guess_title(first_page_text)
    return meta


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #

def download_pdf(url: str, dest: Path, timeout: float = 60.0) -> Path | None:
    """下载 PDF：浏览器 UA、跟随跳转、校验 ``%PDF`` 魔数、原子落盘；失败返回 None。"""
    if not url or not str(url).strip():
        return None
    dest_path = Path(dest)
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("download_pdf: cannot create parent dir for %s: %s", dest_path, exc)
        return None
    tmp = dest_path.with_name(dest_path.name + ".part")
    req = urllib.request.Request(
        str(url),
        headers={"User-Agent": _BROWSER_UA, "Accept": "application/pdf,*/*"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout or 60.0)) as resp:  # noqa: S310
            status = getattr(resp, "status", 200) or 200
            if int(status) >= 400:
                log.warning("download_pdf HTTP %s for %s", status, url)
                return None
            head = resp.read(1024)
            if b"%PDF" not in head[:64]:
                log.warning("download_pdf: response is not a PDF (%s)", url)
                return None
            total = len(head)
            with open(tmp, "wb") as fh:
                fh.write(head)
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_PDF_BYTES:
                        raise ValueError(f"PDF exceeds {_MAX_PDF_BYTES} bytes cap")
                    fh.write(chunk)
        os.replace(tmp, dest_path)
        return dest_path
    except Exception as exc:
        log.warning("download_pdf failed for %s: %s", url, exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return None


# --------------------------------------------------------------------------- #
# Markdown（给 LLM 读）
# --------------------------------------------------------------------------- #

_BULLET_RE = re.compile(r"^\s*[•·▪◦‣∙\-\*\u2022]\s+")
_GAP_SPLIT_RE = re.compile(r"\s{3,}")


def _rough_table_or_list(text: str, nspans: int) -> str:
    if _BULLET_RE.match(text):
        return "- " + _BULLET_RE.sub("", text, count=1).strip()
    cells = [c.strip() for c in _GAP_SPLIT_RE.split(text) if c.strip()]
    if nspans >= 3 and len(cells) >= 3:
        return "| " + " | ".join(cells) + " |"
    return text


def _page_markdown(page: Any) -> str:
    try:
        data = page.get_text("dict") or {}
    except Exception:
        try:
            return page.get_text("text") or ""
        except Exception:
            return ""
    rows: list[tuple[str, float, int]] = []
    sizes: list[float] = []
    for block in data.get("blocks", []) or []:
        if int(block.get("type", 0) or 0) != 0:
            continue
        for line in block.get("lines", []) or []:
            spans = [s for s in (line.get("spans") or []) if str(s.get("text") or "").strip()]
            if not spans:
                continue
            text = "".join(str(s.get("text") or "") for s in (line.get("spans") or []))
            if not text.strip():
                continue
            try:
                size = max(float(s.get("size") or 0.0) for s in spans)
            except Exception:
                size = 0.0
            sizes.append(size)
            rows.append((text.rstrip(), size, len(spans)))
    if not rows:
        return ""
    try:
        median = statistics.median(sizes) if sizes else 0.0
    except Exception:
        median = 0.0
    out: list[str] = []
    for text, size, nspans in rows:
        stripped = text.strip()
        if not stripped:
            continue
        if median and size >= median * 1.25 and 3 <= len(stripped) <= 120 and not stripped.endswith("."):
            out.append(f"## {stripped}")
            continue
        out.append(_rough_table_or_list(stripped, nspans))
    return "\n".join(out)


def pdf_to_markdown(pdf_path: Path, max_pages: int | None = None) -> str:
    """按页转 markdown，页间用 ``<!-- page N -->`` 分隔。"""
    path = _require_file(pdf_path)
    limit = int(max_pages) if max_pages else None
    fitz = _fitz()
    if fitz is not None:
        try:
            parts: list[str] = []
            with fitz.open(str(path)) as doc:
                for i in range(int(getattr(doc, "page_count", 0) or 0)):
                    if limit is not None and i >= limit:
                        break
                    parts.append(f"<!-- page {i + 1} -->\n\n{_page_markdown(doc[i])}")
            return "\n\n".join(parts)
        except Exception as exc:
            log.warning("fitz pdf_to_markdown failed for %s: %s", path, exc)
    text = extract_text(path, max_pages=limit)
    return f"<!-- page 1 -->\n\n{text}" if text else ""
