"""文献检索工具（CONTRACTS §4）。

设计约束（与契约一致）：

* 主 HTTP 路径是标准库 ``urllib.request`` + ``xml.etree.ElementTree``，零第三方依赖；
  ``requests`` 仅作为 urllib 失败时的可选回退（未安装也能正常工作）。
* :meth:`LiteratureSearch.search` / ``multi_search`` / ``similar_to`` **绝不抛异常**
  到管线顶层：任何源失败都记 ``retrieve_fail`` 事件并返回 ``[]``。
  :class:`RetrievalError` 仅用于 *误用*（例如未知 source 名）。
* 磁盘缓存：``cache_dir/<source>_<sha1(query|max_results|sort|year)>.json``，
  内容 ``{"ts": float, "papers": [...]}``，TTL 7 天；缓存目录不可创建时自动降级为无缓存。
* 相关度评分（documented relevance score）::

      text_score = sum_t w(t) / (3 * |T|)
          w(t) = 3   若 t 出现在标题
                 1   若 t 出现在摘要 / tldr
                 0   其它
      score = text_score * (1 + 0.2 * log1p(citation_count))
              + 0.15 * clamp((year - 2014) / 10, 0, 1)     # 时效性奖励
      排序键 = (-round(score, 10), paper.key(), paper.id)     # 完全确定

  即：正文/标题相关度是主项，引用量是次线性加权，年份给最多 0.15 的加成。
"""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import logging
import math
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields as _dc_fields
from pathlib import Path
from typing import Any, Callable, Iterable

try:  # 兄弟模块（并行开发中）不存在时优雅降级
    from ..logging_utils import get_logger  # type: ignore
except Exception:  # pragma: no cover - 依赖兄弟模块
    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)

log = get_logger("autoresearch.tools.retrieve")

__all__ = [
    "Paper",
    "RetrievalError",
    "LiteratureSearch",
    "relevance_score",
    "CACHE_TTL",
    "DEFAULT_SOURCES",
    "S2_FIELDS",
]

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

CACHE_TTL: float = 7 * 24 * 3600.0
DEFAULT_SOURCES: tuple[str, ...] = ("arxiv", "s2", "openalex", "crossref")
S2_FIELDS: str = (
    "title,abstract,year,venue,citationCount,authors,externalIds,url,openAccessPdf,tldr"
)
# recommendations 端点不支持 tldr（会返回 HTTP 400: Unrecognized fields: [tldr]）
S2_REC_FIELDS: str = S2_FIELDS.replace(",tldr", "")

ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"

_UA_BASE = "AutoResearch/0.1 (literature retrieval; +https://example.org/autoresearch)"
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36 AutoResearch/0.1"
)
_MAX_HTTP_BYTES = 8 * 1024 * 1024
_ARXIV_SORT_KEYS = ("relevance", "submittedDate", "lastUpdatedDate")

# arXiv 限速是**进程级**的（多个 LiteratureSearch 实例共享），符合官方 3s 建议
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST: list[float] = [0.0]

_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "also", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its",
    "just", "may", "me", "might", "more", "most", "much", "must", "my", "no", "nor",
    "not", "now", "of", "off", "on", "once", "one", "only", "or", "other", "our", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "two", "under", "until", "up", "use", "used", "using", "very", "via", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "why", "will",
    "with", "would", "you", "your",
}
_TITLE_STOPWORDS = _STOPWORDS | {"towards", "toward", "new", "novel", "study", "analysis"}

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_JATS_RE = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _clean_ws(text: Any) -> str:
    """折叠所有空白（含换行）为单个空格并 strip。"""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _json_safe(obj: Any) -> Any:
    """把任意对象递归转换为 JSON 可序列化结构（未知类型转 str）。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        m = re.search(r"(?:19|20)\d{2}", text)
        if m:
            return int(m.group(0))
    return None


def _slug_alnum(text: Any) -> str:
    """小写 + 去掉所有非字母数字字符（保留 Unicode 字母，如中文）。"""
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _normalize_title_for_compare(title: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", str(title or "").lower())).strip()


def _titles_near_dup(a: Any, b: Any) -> bool:
    """归一化前缀相等，或 difflib 相似度 > 0.92。"""
    na, nb = _normalize_title_for_compare(a), _normalize_title_for_compare(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if min(len(na), len(nb)) >= 12 and (na.startswith(nb) or nb.startswith(na)):
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() > 0.92


def _terms(text: Any) -> list[str]:
    """分词（小写词 + 单个 CJK 字符），去掉停用词与单字母。"""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(str(text or "").lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) == 1 and not ("\u4e00" <= tok <= "\u9fff"):
            continue
        out.append(tok)
    return out


def _dedup_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _keywords_from_text(text: Any, limit: int = 8) -> list[str]:
    counts = Counter(t for t in _terms(text) if len(t) >= 3 or "\u4e00" <= t <= "\u9fff")
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:limit]]


def _retry_after_seconds(headers: dict | None) -> float | None:
    if not headers:
        return None
    raw = headers.get("retry-after")
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        return max(0.0, min(float(raw), 60.0))
    except Exception:
        pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            import datetime as _dt
            now = _dt.datetime.now(_dt.timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            return max(0.0, min((dt - now).total_seconds(), 60.0))
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Paper
# --------------------------------------------------------------------------- #

_PAPER_FIELDS: tuple[str, ...] = (
    "id", "title", "abstract", "authors", "year", "venue", "url", "pdf_url",
    "citation_count", "source", "tldr", "keywords", "extra",
)


@dataclass
class Paper:
    """一条文献记录（字段与 CONTRACTS §4 完全一致）。"""

    id: str = ""
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    url: str = ""
    pdf_url: str = ""
    citation_count: int = 0
    source: str = ""
    tldr: str = ""
    keywords: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _clean_ws(self.id)
        self.title = _clean_ws(self.title)
        self.abstract = _clean_ws(self.abstract)
        self.tldr = _clean_ws(self.tldr)
        self.venue = _clean_ws(self.venue)
        self.url = "" if self.url is None else str(self.url).strip()
        self.pdf_url = "" if self.pdf_url is None else str(self.pdf_url).strip()
        self.source = _clean_ws(self.source)
        if isinstance(self.authors, str):
            self.authors = [a.strip() for a in re.split(r"[;\n]| and ", self.authors) if a.strip()]
        else:
            self.authors = [_clean_ws(a) for a in (self.authors or []) if _clean_ws(a)]
        if isinstance(self.keywords, str):
            self.keywords = [k.strip() for k in re.split(r"[;,]", self.keywords) if k.strip()]
        else:
            self.keywords = [_clean_ws(k) for k in (self.keywords or []) if _clean_ws(k)]
        self.year = _as_int(self.year)
        self.citation_count = int(_as_int(self.citation_count) or 0)
        self.extra = dict(self.extra) if isinstance(self.extra, dict) else {}

    # -- 序列化 ---------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "citation_count": int(self.citation_count),
            "source": self.source,
            "tldr": self.tldr,
            "keywords": list(self.keywords),
            "extra": _json_safe(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        if isinstance(d, cls):
            return d
        data = d if isinstance(d, dict) else {}
        known = {f.name for f in _dc_fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    # -- 去重键 ---------------------------------------------------------- #
    def key(self) -> str:
        """归一化去重键：标题小写去标点（+ 年份），标题为空时退回 id。"""
        slug = _slug_alnum(self.title)
        if not slug:
            return _slug_alnum(self.id) or "untitled"
        return f"{slug}-{self.year}" if self.year else slug

    # -- 便捷 ------------------------------------------------------------ #
    def doi(self) -> str:
        """尽力而为地取出 DOI（Paper 无 doi 字段，故从 id/extra 推导）。"""
        for cand in (self.extra.get("doi"), self.extra.get("DOI")):
            if cand:
                return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(cand)).strip()
        if self.id.lower().startswith("doi:"):
            return self.id[4:].strip()
        return ""

    def arxiv_id(self) -> str:
        if self.extra.get("arxiv_id"):
            return str(self.extra["arxiv_id"]).strip()
        if self.id.lower().startswith("arxiv:"):
            return self.id[6:].strip()
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", self.url or "", re.I)
        return m.group(1).strip() if m else ""


class RetrievalError(RuntimeError):
    """检索误用（例如未知 source 名）。网络/解析失败不允许抛这个。"""


# --------------------------------------------------------------------------- #
# HTTP（stdlib 为主，requests 可选回退；永不抛异常）
# --------------------------------------------------------------------------- #

def _urllib_get(url: str, timeout: float, headers: dict) -> tuple[int, str, dict, str]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(_MAX_HTTP_BYTES)
            text = raw.decode("utf-8", "replace")
            hdrs = {str(k).lower(): str(v) for k, v in (resp.headers or {}).items()}
            status = getattr(resp, "status", 200) or 200
            return int(status), text, hdrs, ""
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            pass
        hdrs = {str(k).lower(): str(v) for k, v in (exc.headers or {}).items()}
        return int(exc.code or 0), body, hdrs, f"HTTP {exc.code}"
    except Exception as exc:  # URLError / timeout / ssl ...
        return 0, "", {}, f"{type(exc).__name__}: {exc}"


def _requests_get(url: str, timeout: float, headers: dict):
    """可选加速/回退路径；requests 未安装时返回 None。"""
    try:
        import requests  # type: ignore
    except Exception:
        return None
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        hdrs = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        return int(resp.status_code), resp.text, hdrs, "" if resp.ok else f"HTTP {resp.status_code}"
    except Exception as exc:
        return 0, "", {}, f"requests.{type(exc).__name__}: {exc}"


def _http_get(url: str, timeout: float = 20.0, headers: dict | None = None,
              retries_429: int = 0, sleep: Callable[[float], None] = time.sleep
              ) -> tuple[int, str, dict, str]:
    """GET，返回 ``(status, text, headers, error)``；从不抛异常。

    ``status == 0`` 表示网络层失败。遇到 429 时按 ``Retry-After``（或指数退避）
    最多重试 ``retries_429`` 次。
    """
    merged = {"User-Agent": _UA_BASE, "Accept": "*/*"}
    if headers:
        merged.update({k: v for k, v in headers.items() if v is not None})
    attempt = 0
    result: tuple[int, str, dict, str] = (0, "", {}, "no attempt")
    while True:
        status, text, hdrs, err = _urllib_get(url, timeout, merged)
        if status == 0:
            fallback = _requests_get(url, timeout, merged)
            if fallback is not None and fallback[0] != 0:
                status, text, hdrs, err = fallback
            elif fallback is not None:
                err = f"{err}; {fallback[3]}"
        result = (status, text, hdrs, err)
        if status == 429 and attempt < retries_429:
            attempt += 1
            wait = _retry_after_seconds(hdrs)
            if wait is None:
                wait = min(2.0 ** attempt, 8.0)
            log.info("HTTP 429 -> backoff %.1fs (attempt %d/%d)", wait, attempt, retries_429)
            try:
                sleep(wait)
            except Exception:
                pass
            continue
        return result


def _get_json(url: str, timeout: float = 20.0, headers: dict | None = None,
              retries_429: int = 0, sleep: Callable[[float], None] = time.sleep
              ) -> tuple[Any, str, int]:
    """返回 ``(data, error, status)``；``data is None`` 表示失败。"""
    status, text, _hdrs, err = _http_get(url, timeout, headers, retries_429=retries_429, sleep=sleep)
    if status == 0:
        return None, err or "network error", 0
    if status >= 400:
        return None, err or f"HTTP {status}", status
    try:
        return json.loads(text), "", status
    except Exception as exc:
        return None, f"JSONDecodeError: {exc}", status


# --------------------------------------------------------------------------- #
# 去重 / 合并
# --------------------------------------------------------------------------- #

_RICH_FIELDS = ("title", "abstract", "authors", "year", "venue", "url", "pdf_url",
                "citation_count", "tldr", "keywords", "extra")


def _richness(p: Paper) -> int:
    return sum(1 for f in _RICH_FIELDS if getattr(p, f, None))


def _pick_primary(a: Paper, b: Paper) -> tuple[Paper, Paper]:
    """返回 ``(primary, secondary)``：字段更全者优先，其次引用量高、摘要更长。"""
    ra, rb = _richness(a), _richness(b)
    if ra != rb:
        return (a, b) if ra > rb else (b, a)
    if a.citation_count != b.citation_count:
        return (a, b) if a.citation_count > b.citation_count else (b, a)
    if len(a.abstract) != len(b.abstract):
        return (a, b) if len(a.abstract) > len(b.abstract) else (b, a)
    return (a, b) if a.id <= b.id else (b, a)


def _join_sources(a: Paper, b: Paper) -> str:
    parts: list[str] = []
    for src in (a.source, b.source):
        for piece in str(src or "").split("+"):
            piece = piece.strip()
            if piece and piece not in parts:
                parts.append(piece)
    return "+".join(parts)


def _merge_pair(primary: Paper, secondary: Paper) -> Paper:
    """合并两条重复记录：primary 的非空字段优先，其余从 secondary 补齐。"""
    title = primary.title or secondary.title
    if (secondary.title and len(secondary.title) > len(primary.title)
            and _titles_near_dup(primary.title, secondary.title)):
        title = secondary.title  # 同一篇论文时取更完整的标题拼写
    merged = Paper(
        id=primary.id or secondary.id,
        title=title,
        abstract=primary.abstract or secondary.abstract,
        authors=primary.authors or secondary.authors,
        year=primary.year or secondary.year,
        venue=primary.venue or secondary.venue,
        url=primary.url or secondary.url,
        pdf_url=primary.pdf_url or secondary.pdf_url,
        citation_count=max(int(primary.citation_count or 0), int(secondary.citation_count or 0)),
        source=_join_sources(primary, secondary),
        tldr=primary.tldr or secondary.tldr,
        keywords=_dedup_preserve(list(primary.keywords) + list(secondary.keywords)),
        extra={**_json_safe(secondary.extra), **_json_safe(primary.extra)},
    )
    if not merged.abstract:
        merged.abstract = primary.abstract or secondary.abstract
    return merged


def _dedupe(papers: Iterable[Paper]) -> list[Paper]:
    """按 :meth:`Paper.key` **与** 近似标题合并去重。"""
    groups: list[Paper] = []
    for p in papers:
        if not isinstance(p, Paper):
            continue
        placed = False
        for i, g in enumerate(groups):
            if p.key() == g.key() or _titles_near_dup(p.title, g.title):
                primary, secondary = _pick_primary(g, p)
                groups[i] = _merge_pair(primary, secondary)
                placed = True
                break
        if not placed:
            groups.append(p)
    return groups


# --------------------------------------------------------------------------- #
# 相关度
# --------------------------------------------------------------------------- #

def relevance_score(paper: Paper, query: str) -> float:
    """Documented relevance score（见模块 docstring）。"""
    q_terms = _dedup_preserve(_terms(query))
    if not q_terms or not isinstance(paper, Paper):
        return 0.0
    title_tokens = set(_terms(paper.title))
    body_tokens = set(_terms(f"{paper.abstract} {paper.tldr} {paper.venue}"))
    total = 0.0
    for t in q_terms:
        if t in title_tokens:
            total += 3.0
        elif t in body_tokens:
            total += 1.0
    text_score = total / (3.0 * len(q_terms))
    cite_weight = 1.0 + 0.2 * math.log1p(max(0, int(paper.citation_count or 0)))
    recency = 0.0
    if paper.year:
        recency = 0.15 * min(1.0, max(0.0, (int(paper.year) - 2014) / 10.0))
    return text_score * cite_weight + recency


def _rank(papers: Iterable[Paper], query: str) -> list[Paper]:
    """确定性排序：(-score, key, id)。"""
    scored = []
    for p in papers:
        if not isinstance(p, Paper):
            continue
        scored.append((-round(relevance_score(p, query), 10), p.key(), p.id, p))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in scored]


# --------------------------------------------------------------------------- #
# arXiv 查询净化 & Atom 解析
# --------------------------------------------------------------------------- #

_ARXIV_FIELD_RE = re.compile(r"^\s*(all|ti|abs|au|cat|co|jr|rn|id|doi)\s*:\s*(.+)$", re.I | re.S)
_ARXIV_BAD_RE = re.compile(r'[+!()\[\]{}^~*?:\\/|&"\'<>]')


def _sanitize_arxiv_query(query: str) -> str:
    """把裸查询包成 ``all:"..."``，并去掉 arXiv 语法会拒绝的字符。"""
    raw = " ".join(str(query or "").split())
    if not raw:
        return ""
    field_prefix = "all"
    body = raw
    m = _ARXIV_FIELD_RE.match(raw)
    if m:
        field_prefix = m.group(1).lower()
        body = m.group(2)
    body = _ARXIV_BAD_RE.sub(" ", body)
    body = " ".join(body.split())
    if not body:
        return ""
    return f'{field_prefix}:"{body}"'


def _bare_arxiv_id(raw_id: str) -> str:
    text = str(raw_id or "").strip()
    if not text:
        return ""
    m = re.search(r"(?:abs|pdf)/([^\s?#]+)", text)
    if m:
        text = m.group(1)
    else:
        text = text.rstrip("/").rsplit("/", 1)[-1]
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"v\d+$", "", text)
    return text


def _year_from_date(value: Any) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(m.group(0)) if m else None


def _text(node: Any) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()) if hasattr(node, "itertext") else str(node)


def _parse_arxiv_atom(xml_text: str) -> list[Paper]:
    if not xml_text or "<entry" not in xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        log.warning("arxiv atom parse failed: %s", exc)
        return []
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        title = _clean_ws(_text(entry.find(f"{ATOM}title")))
        summary = _clean_ws(_text(entry.find(f"{ATOM}summary")))
        raw_id = _clean_ws(_text(entry.find(f"{ATOM}id")))
        bare = _bare_arxiv_id(raw_id)
        authors = [
            _clean_ws(_text(a.find(f"{ATOM}name")))
            for a in entry.findall(f"{ATOM}author")
        ]
        authors = [a for a in authors if a]
        published = _text(entry.find(f"{ATOM}published")) or _text(entry.find(f"{ATOM}updated"))
        pdf_url = ""
        for link in entry.findall(f"{ATOM}link"):
            if str(link.get("title") or "").lower() == "pdf":
                pdf_url = str(link.get("href") or "").strip()
                break
        if not pdf_url and bare:
            pdf_url = f"https://arxiv.org/pdf/{bare}"
        extra: dict[str, Any] = {}
        if bare:
            extra["arxiv_id"] = bare
        if raw_id:
            extra["arxiv_url"] = raw_id
        primary = entry.find(f"{ARX}primary_category")
        category = str(primary.get("term") or "").strip() if primary is not None else ""
        if category:
            extra["primary_category"] = category
        comment = _clean_ws(_text(entry.find(f"{ARX}comment")))
        if comment:
            extra["comment"] = comment
        journal_ref = _clean_ws(_text(entry.find(f"{ARX}journal_ref")))
        if journal_ref:
            extra["journal_ref"] = journal_ref
        venue = journal_ref or category
        papers.append(Paper(
            id=f"arxiv:{bare}" if bare else f"arxiv:{raw_id}",
            title=title,
            abstract=summary,
            authors=authors,
            year=_year_from_date(published),
            venue=venue,
            url=f"https://arxiv.org/abs/{bare}" if bare else raw_id,
            pdf_url=pdf_url,
            citation_count=0,
            source="arxiv",
            tldr="",
            keywords=_keywords_from_text(f"{title} {summary}"),
            extra=extra,
        ))
    return papers


# --------------------------------------------------------------------------- #
# OpenAlex 反演摘要
# --------------------------------------------------------------------------- #

def _deinvert_abstract(inverted: Any) -> str:
    """``abstract_inverted_index`` -> 正常语序摘要（按位置排序重建）。"""
    if not inverted or not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        if not isinstance(word, str):
            continue
        if isinstance(idxs, (int, float)):
            idxs = [idxs]
        if isinstance(idxs, (list, tuple, set)):
            for i in idxs:
                try:
                    positions.append((int(i), word))
                except Exception:
                    continue
        else:
            positions.append((len(positions), word))
    if not positions:
        return ""
    positions.sort(key=lambda t: (t[0], t[1]))
    return _clean_ws(" ".join(word for _, word in positions))


def _strip_jats(abstract: Any) -> str:
    # 先反转义：Crossref 的 JATS/HTML 常以 &lt;jats:p&gt; 形式出现，顺序反了会残留标签
    text = html.unescape(str(abstract or ""))
    text = re.sub(r"</?jats:[^>]*>", " ", text)
    text = _JATS_RE.sub(" ", text)
    return _clean_ws(text)


# --------------------------------------------------------------------------- #
# 各源记录 -> Paper
# --------------------------------------------------------------------------- #

def _paper_from_s2(item: dict) -> Paper:
    item = item or {}
    ext = item.get("externalIds") or {}
    doi = str(ext.get("DOI") or "").strip()
    arx = str(ext.get("ArXiv") or "").strip()
    s2_id = str(item.get("paperId") or "").strip()
    if arx:
        pid = f"arxiv:{arx}"
    elif doi:
        pid = f"doi:{doi}"
    else:
        pid = f"s2:{s2_id}"
    oa = item.get("openAccessPdf") or {}
    authors = [str((a or {}).get("name") or "").strip() for a in (item.get("authors") or [])]
    tldr_obj = item.get("tldr") or {}
    tldr = str(tldr_obj.get("text") or "") if isinstance(tldr_obj, dict) else str(tldr_obj or "")
    url = str(item.get("url") or "").strip() or (f"https://doi.org/{doi}" if doi else "")
    extra: dict[str, Any] = {}
    if s2_id:
        extra["s2_id"] = s2_id
    if ext:
        extra["external_ids"] = {str(k): str(v) for k, v in ext.items()}
    if doi:
        extra["doi"] = doi
    if arx:
        extra["arxiv_id"] = arx
    title = _clean_ws(item.get("title"))
    abstract = _clean_ws(item.get("abstract"))
    return Paper(
        id=pid,
        title=title,
        abstract=abstract,
        authors=[a for a in authors if a],
        year=_as_int(item.get("year")),
        venue=_clean_ws(item.get("venue")),
        url=url,
        pdf_url=str(oa.get("url") or "").strip() if isinstance(oa, dict) else "",
        citation_count=int(_as_int(item.get("citationCount")) or 0),
        source="s2",
        tldr=_clean_ws(tldr),
        keywords=_keywords_from_text(f"{title} {abstract} {tldr}"),
        extra=extra,
    )


def _openalex_venue(item: dict) -> str:
    host = item.get("host_venue") or {}
    if isinstance(host, dict) and host.get("display_name"):
        return _clean_ws(host.get("display_name"))
    loc = item.get("primary_location") or {}
    src = (loc.get("source") or {}) if isinstance(loc, dict) else {}
    return _clean_ws((src or {}).get("display_name"))


def _paper_from_openalex(item: dict) -> Paper:
    item = item or {}
    title = _clean_ws(item.get("title") or item.get("display_name"))
    abstract = _deinvert_abstract(item.get("abstract_inverted_index"))
    doi_raw = str(item.get("doi") or "").strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi_raw)
    oa_id = str(item.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
    pid = f"doi:{doi}" if doi else (f"openalex:{oa_id}" if oa_id else "")
    loc = item.get("primary_location") or {}
    best = item.get("best_oa_location") or {}
    pdf_url = ""
    if isinstance(loc, dict):
        pdf_url = str(loc.get("pdf_url") or "").strip()
    if not pdf_url and isinstance(best, dict):
        pdf_url = str(best.get("pdf_url") or "").strip()
    authors = [
        _clean_ws(((a or {}).get("author") or {}).get("display_name"))
        for a in (item.get("authorships") or [])
    ]
    kw_objs = item.get("keywords") or []
    keywords = [_clean_ws((k or {}).get("display_name")) for k in kw_objs if isinstance(k, dict)]
    keywords = [k for k in keywords if k] or _keywords_from_text(f"{title} {abstract}")
    extra: dict[str, Any] = {}
    if oa_id:
        extra["openalex_id"] = oa_id
    if doi:
        extra["doi"] = doi
    if item.get("type"):
        extra["type"] = item.get("type")
    return Paper(
        id=pid or f"openalex:{_slug_alnum(title)[:40]}",
        title=title,
        abstract=abstract,
        authors=[a for a in authors if a],
        year=_as_int(item.get("publication_year")),
        venue=_openalex_venue(item),
        url=doi_raw or str(item.get("id") or ""),
        pdf_url=pdf_url,
        citation_count=int(_as_int(item.get("cited_by_count")) or 0),
        source="openalex",
        tldr="",
        keywords=keywords[:8],
        extra=extra,
    )


def _crossref_authors(item: dict) -> list[str]:
    out: list[str] = []
    for a in item.get("author") or []:
        if not isinstance(a, dict):
            continue
        given = _clean_ws(a.get("given"))
        family = _clean_ws(a.get("family"))
        name = _clean_ws(f"{given} {family}") or _clean_ws(a.get("name"))
        if name:
            out.append(name)
    return out


def _paper_from_crossref(item: dict) -> Paper:
    item = item or {}
    titles = item.get("title") or []
    title = _clean_ws(titles[0]) if titles else ""
    abstract = _strip_jats(item.get("abstract"))
    issued = item.get("issued") or {}
    parts = issued.get("date-parts") or []
    year = None
    if parts and isinstance(parts[0], (list, tuple)) and parts[0]:
        year = _as_int(parts[0][0])
    containers = item.get("container-title") or []
    venue = _clean_ws(containers[0]) if containers else ""
    doi = str(item.get("DOI") or "").strip()
    url = str(item.get("URL") or "").strip() or (f"https://doi.org/{doi}" if doi else "")
    extra: dict[str, Any] = {}
    if doi:
        extra["doi"] = doi
    if item.get("type"):
        extra["type"] = item.get("type")
    if item.get("publisher"):
        extra["publisher"] = _clean_ws(item.get("publisher"))
    return Paper(
        id=f"doi:{doi}" if doi else f"crossref:{_slug_alnum(title)[:40]}",
        title=title,
        abstract=abstract,
        authors=_crossref_authors(item),
        year=year,
        venue=venue,
        url=url,
        pdf_url="",
        citation_count=int(_as_int(item.get("is-referenced-by-count")) or 0),
        source="crossref",
        tldr="",
        keywords=_keywords_from_text(f"{title} {abstract} {venue}"),
        extra=extra,
    )


def _s2_paper_id(paper_id: str) -> str:
    """把内部 id 转成 S2 recommendations 端点接受的 paperId。"""
    pid = str(paper_id or "").strip()
    if not pid:
        return ""
    low = pid.lower()
    if low.startswith("arxiv:"):
        return f"ARXIV:{pid[6:]}"
    if low.startswith("doi:"):
        return f"DOI:{pid[4:]}"
    if low.startswith("s2:"):
        return pid[3:]
    return pid


# --------------------------------------------------------------------------- #
# 默认缓存目录
# --------------------------------------------------------------------------- #

def default_cache_dir() -> Path:
    """``<runs_dir_parent>/cache/retrieve``（即 WORKSPACE_ROOT/.autoresearch/cache/retrieve）。"""
    try:
        from ..config import DEFAULT_RUNS_DIR  # type: ignore
        return Path(DEFAULT_RUNS_DIR).parent / "cache" / "retrieve"
    except Exception:
        # tools/retrieve.py -> tools -> autoresearch -> WORKSPACE_ROOT
        return Path(__file__).resolve().parents[2] / ".autoresearch" / "cache" / "retrieve"


# --------------------------------------------------------------------------- #
# LiteratureSearch
# --------------------------------------------------------------------------- #

class LiteratureSearch:
    """多源文献检索器（CONTRACTS §4）。"""

    def __init__(self, cfg: Any, cache_dir: Path | None = None, event_logger: Any = None) -> None:
        self.cfg = cfg
        self.events = event_logger
        self.language: str = str(getattr(cfg, "language", None) or "zh")
        self.timeout: float = float(getattr(cfg, "timeout", 20.0) or 20.0)
        self.mailto: str = str(getattr(cfg, "mailto", "") or "autoresearch@example.org")
        self.offline: bool = bool(getattr(cfg, "offline", False))
        self.min_interval: float = 3.0  # arXiv 官方要求的请求间隔（秒，进程级生效）
        self._cache_warned = False

        self.cache_dir: Path | None = None
        self._cache_enabled = False
        target: Path | None = None
        if cache_dir is not None:
            target = Path(cache_dir)
        elif getattr(cfg, "cache_dir", None):
            target = Path(getattr(cfg, "cache_dir"))
        else:
            target = default_cache_dir()
        if target is not None:
            self.cache_dir = Path(target)
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache_enabled = True
            except Exception as exc:
                log.warning("retrieve cache disabled (%s): %s", self.cache_dir, exc)
                self._cache_enabled = False

    # -- 事件 / 日志 ------------------------------------------------------ #
    def _event(self, event: str, **fields: Any) -> None:
        ev = self.events
        if ev is None:
            return
        try:
            fn = getattr(ev, "log", None)
            if callable(fn):
                fn(event, **fields)
            elif callable(ev):
                ev(event, **fields)
        except Exception:
            pass

    def _ua(self) -> str:
        return f"{_UA_BASE} mailto:{self.mailto}" if self.mailto else _UA_BASE

    # -- 缓存 ------------------------------------------------------------ #
    def _cache_file(self, source: str, query: str, max_results: int,
                    sort: str = "", year: str = "") -> Path:
        payload = f"{query}|{int(max_results)}|{sort}|{year}"
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        base = self.cache_dir or default_cache_dir()
        return Path(base) / f"{source}_{digest}.json"

    def _cache_load(self, source: str, query: str, max_results: int,
                    sort: str = "", year: str = "") -> list[Paper] | None:
        if not self._cache_enabled or self.cache_dir is None:
            return None
        path = self._cache_file(source, query, max_results, sort, year)
        try:
            raw = json.loads(path.read_text("utf-8"))
        except FileNotFoundError:
            return None
        except Exception as exc:
            log.debug("cache read failed %s: %s", path, exc)
            return None
        try:
            ts = float(raw.get("ts", 0.0))
            papers = [Paper.from_dict(d) for d in (raw.get("papers") or [])]
        except Exception as exc:
            log.debug("cache decode failed %s: %s", path, exc)
            return None
        if time.time() - ts > CACHE_TTL:
            log.debug("cache expired: %s", path)
            return None
        self._event("retrieve_cache_hit", source=source, query=query, count=len(papers),
                    path=str(path))
        log.info("retrieve_cache_hit %s (%d papers)", source, len(papers))
        return papers

    def _cache_store(self, source: str, query: str, max_results: int, papers: list[Paper],
                     sort: str = "", year: str = "") -> None:
        if not self._cache_enabled or self.cache_dir is None or not papers:
            return
        path = self._cache_file(source, query, max_results, sort, year)
        tmp = path.with_name(path.name + ".tmp")
        payload = {"ts": time.time(), "papers": [p.to_dict() for p in papers]}
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            log.warning("cache write failed %s: %s", path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # -- arXiv 限速 ------------------------------------------------------ #
    def _arxiv_wait(self) -> None:
        """进程级限速：任意两个 arXiv 调用之间至少间隔 ``min_interval`` 秒。"""
        with _ARXIV_LOCK:
            if _ARXIV_LAST[0] > 0:
                wait = float(self.min_interval) - (time.time() - _ARXIV_LAST[0])
                if wait > 0:
                    time.sleep(wait)
            _ARXIV_LAST[0] = time.time()

    # -- 源实现 ---------------------------------------------------------- #
    def _arxiv_impl(self, query: str, max_results: int, sort_by: str) -> list[Paper]:
        search_query = _sanitize_arxiv_query(query)
        if not search_query:
            self._event("retrieve_fail", source="arxiv", query=query, error="empty query")
            return []
        self._arxiv_wait()
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": int(max_results),
            "sortBy": sort_by if sort_by in _ARXIV_SORT_KEYS else "relevance",
            "sortOrder": "descending",
        }
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        status, text, _hdrs, err = _http_get(url, self.timeout, {"User-Agent": self._ua(),
                                                                "Accept": "application/atom+xml"})
        if status == 0 or status >= 400 or not text:
            self._event("retrieve_fail", source="arxiv", query=query,
                        error=err or f"HTTP {status}")
            log.warning("arxiv fetch failed (%s): %s", status, err)
            return []
        papers = _parse_arxiv_atom(text)
        if not papers:
            log.info("arxiv returned 0 entries for %r", query)
        return papers

    def _s2_impl(self, query: str, max_results: int, year: str) -> list[Paper]:
        params = {"query": query, "limit": max(1, min(int(max_results), 100)), "fields": S2_FIELDS}
        if year:
            params["year"] = year
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
        data, err, status = _get_json(url, self.timeout, {"User-Agent": self._ua()}, retries_429=3)
        if data is None:
            self._event("retrieve_fail", source="s2", query=query, error=err)
            log.warning("s2 search failed (%s): %s", status, err)
            return []
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            self._event("retrieve_fail", source="s2", query=query, error="malformed payload")
            return []
        out: list[Paper] = []
        for item in items:
            try:
                out.append(_paper_from_s2(item))
            except Exception as exc:
                log.debug("s2 item skipped: %s", exc)
        return out

    def _openalex_impl(self, query: str, max_results: int) -> list[Paper]:
        params = {"search": query, "per-page": max(1, min(int(max_results), 200)),
                  "mailto": self.mailto}
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        data, err, status = _get_json(url, self.timeout, {"User-Agent": self._ua()}, retries_429=2)
        if data is None:
            self._event("retrieve_fail", source="openalex", query=query, error=err)
            log.warning("openalex search failed (%s): %s", status, err)
            return []
        items = data.get("results") if isinstance(data, dict) else None
        if not isinstance(items, list):
            self._event("retrieve_fail", source="openalex", query=query, error="malformed payload")
            return []
        out: list[Paper] = []
        for item in items:
            try:
                out.append(_paper_from_openalex(item))
            except Exception as exc:
                log.debug("openalex item skipped: %s", exc)
        return out

    def _crossref_impl(self, query: str, max_results: int) -> list[Paper]:
        params = {"query.bibliographic": query, "rows": max(1, min(int(max_results), 200)),
                  "mailto": self.mailto}
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        headers = {"User-Agent": self._ua(), "Accept": "application/json"}
        data, err, status = _get_json(url, self.timeout, headers, retries_429=2)
        if data is None:
            self._event("retrieve_fail", source="crossref", query=query, error=err)
            log.warning("crossref search failed (%s): %s", status, err)
            return []
        items = ((data.get("message") or {}) if isinstance(data, dict) else {}).get("items")
        if not isinstance(items, list):
            self._event("retrieve_fail", source="crossref", query=query, error="malformed payload")
            return []
        out: list[Paper] = []
        for item in items:
            try:
                out.append(_paper_from_crossref(item))
            except Exception as exc:
                log.debug("crossref item skipped: %s", exc)
        return out

    # -- 公开源方法（含缓存，永不抛异常） -------------------------------- #
    def search_arxiv(self, query: str, max_results: int = 8,
                     sort_by: str = "relevance") -> list[Paper]:
        q = _clean_ws(query)
        try:
            n = int(max_results or 0)
        except Exception:
            n = 0
        if not q or n <= 0 or self.offline:
            if self.offline:
                self._event("retrieve_fail", source="arxiv", query=q, error="offline")
            return []
        cached = self._cache_load("arxiv", q, n, sort_by)
        if cached is not None:
            return cached
        try:
            papers = self._arxiv_impl(q, n, sort_by)
        except Exception as exc:  # 双保险
            self._event("retrieve_fail", source="arxiv", query=q,
                        error=f"{type(exc).__name__}: {exc}")
            return []
        self._cache_store("arxiv", q, n, papers, sort_by)
        return papers

    def search_semantic_scholar(self, query: str, max_results: int = 8,
                               year: str = "") -> list[Paper]:
        q = _clean_ws(query)
        try:
            n = int(max_results or 0)
        except Exception:
            n = 0
        if not q or n <= 0 or self.offline:
            if self.offline:
                self._event("retrieve_fail", source="s2", query=q, error="offline")
            return []
        cached = self._cache_load("s2", q, n, "", str(year or ""))
        if cached is not None:
            return cached
        try:
            papers = self._s2_impl(q, n, str(year or ""))
        except Exception as exc:
            self._event("retrieve_fail", source="s2", query=q,
                        error=f"{type(exc).__name__}: {exc}")
            return []
        self._cache_store("s2", q, n, papers, "", str(year or ""))
        return papers

    def search_openalex(self, query: str, max_results: int = 8) -> list[Paper]:
        q = _clean_ws(query)
        try:
            n = int(max_results or 0)
        except Exception:
            n = 0
        if not q or n <= 0 or self.offline:
            if self.offline:
                self._event("retrieve_fail", source="openalex", query=q, error="offline")
            return []
        cached = self._cache_load("openalex", q, n)
        if cached is not None:
            return cached
        try:
            papers = self._openalex_impl(q, n)
        except Exception as exc:
            self._event("retrieve_fail", source="openalex", query=q,
                        error=f"{type(exc).__name__}: {exc}")
            return []
        self._cache_store("openalex", q, n, papers)
        return papers

    def search_crossref(self, query: str, max_results: int = 8) -> list[Paper]:
        q = _clean_ws(query)
        try:
            n = int(max_results or 0)
        except Exception:
            n = 0
        if not q or n <= 0 or self.offline:
            if self.offline:
                self._event("retrieve_fail", source="crossref", query=q, error="offline")
            return []
        cached = self._cache_load("crossref", q, n)
        if cached is not None:
            return cached
        try:
            papers = self._crossref_impl(q, n)
        except Exception as exc:
            self._event("retrieve_fail", source="crossref", query=q,
                        error=f"{type(exc).__name__}: {exc}")
            return []
        self._cache_store("crossref", q, n, papers)
        return papers

    # -- source 解析（RetrievalError 仅用于误用） ------------------------ #
    def _source_func(self, name: str) -> Callable[..., list[Paper]] | None:
        return {
            "arxiv": self.search_arxiv,
            "s2": self.search_semantic_scholar,
            "semantic_scholar": self.search_semantic_scholar,
            "semanticscholar": self.search_semantic_scholar,
            "openalex": self.search_openalex,
            "crossref": self.search_crossref,
        }.get((name or "").strip().lower())

    def _resolve_source(self, name: str) -> Callable[..., list[Paper]]:
        fn = self._source_func(name)
        if fn is None:
            raise RetrievalError(
                f"unknown source: {name!r}; known: arxiv, s2, openalex, crossref"
            )
        return fn

    def _configured_sources(self) -> list[str]:
        srcs = getattr(self.cfg, "sources", None) or list(DEFAULT_SOURCES)
        return [str(s).strip().lower() for s in srcs if str(s).strip()]

    # -- 聚合检索 -------------------------------------------------------- #
    def search(self, query: str, max_results: int | None = None,
               sources: list[str] | None = None) -> list[Paper]:
        """多源聚合 + 去重 + 相关度排序。**绝不抛异常**。"""
        try:
            return self._search_impl(query, max_results, sources)
        except Exception as exc:
            self._event("retrieve_fail", source="all", query=str(query),
                        error=f"{type(exc).__name__}: {exc}")
            log.warning("search failed: %s", exc)
            return []

    def _search_impl(self, query: str, max_results: int | None,
                     sources: list[str] | None) -> list[Paper]:
        q = _clean_ws(query)
        try:
            n = int(max_results if max_results is not None
                    else getattr(self.cfg, "max_results_per_query", 8) or 8)
        except Exception:
            n = 8
        if not q or n <= 0:
            self._event("retrieve_fail", source="all", query=q, error="empty query or max_results<=0")
            return []
        if self.offline:
            self._event("retrieve_fail", source="offline", query=q,
                        error="offline mode: retrieval skipped")
            log.info("retrieve offline: skip %r", q)
            return []
        if n > 200:
            n = 200
        names = ([str(s).strip().lower() for s in sources] if sources is not None
                 else self._configured_sources())
        collected: list[Paper] = []
        for name in names:
            try:
                fn = self._resolve_source(name)
            except RetrievalError as exc:
                self._event("retrieve_fail", source=name, query=q, error=str(exc))
                log.warning("%s", exc)
                continue
            self._event("retrieve_call", source=name, query=q, max_results=n)
            try:
                collected.extend(fn(q, n) or [])
            except Exception as exc:  # 源方法本身已兜底，这里再兜一层
                self._event("retrieve_fail", source=name, query=q,
                            error=f"{type(exc).__name__}: {exc}")
                log.warning("source %s failed: %s", name, exc)
        return _rank(_dedupe(collected), q)[:n]

    def _safe_search(self, query: str, n: int) -> list[Paper]:
        try:
            return self.search(query, max_results=n) or []
        except Exception as exc:  # pragma: no cover - search 自身已兜底
            self._event("retrieve_fail", source="worker", query=query,
                        error=f"{type(exc).__name__}: {exc}")
            return []

    def multi_search(self, queries: list[str], per_query: int = 6) -> list[Paper]:
        """多查询并行（4 线程）+ 跨查询去重 + 末端确定性排序。**绝不抛异常**。"""
        try:
            qs = [_clean_ws(q) for q in (queries or [])]
            qs = [q for q in qs if q]
            if not qs:
                return []
            try:
                n = max(1, int(per_query or 6))
            except Exception:
                n = 6
            groups: list[list[Paper]] = []
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {pool.submit(self._safe_search, q, n): q for q in qs}
                    for fut in as_completed(futures):
                        try:
                            groups.append(list(fut.result() or []))
                        except Exception as exc:  # 线程内异常绝不外泄
                            q = futures.get(fut, "")
                            self._event("retrieve_fail", source="worker", query=q,
                                        error=f"{type(exc).__name__}: {exc}")
                            groups.append([])
            except Exception as exc:
                self._event("retrieve_fail", source="threadpool", query=" ".join(qs),
                            error=f"{type(exc).__name__}: {exc}")
                groups = [self._safe_search(q, n) for q in qs]
            flat = [p for group in groups for p in group]
            return _rank(_dedupe(flat), " ".join(qs))
        except Exception as exc:
            self._event("retrieve_fail", source="multi_search", query=str(queries),
                        error=f"{type(exc).__name__}: {exc}")
            return []

    # -- S2 recommendations ---------------------------------------------- #
    def get_recommendations(self, paper_id: str, max_results: int = 8,
                            fields: str = S2_REC_FIELDS) -> list[Paper]:
        """Semantic Scholar recommendations（失败返回 []）。

        注意：recommendations 端点不接受 ``tldr`` 字段（HTTP 400），因此这里会把它剔除。
        """
        pid = _s2_paper_id(paper_id)
        if not pid or self.offline:
            if self.offline and pid:
                self._event("retrieve_fail", source="s2", query=pid, error="offline")
            return []
        try:
            n = max(1, min(int(max_results or 8), 100))
        except Exception:
            n = 8
        safe_fields = ",".join(
            f.strip() for f in str(fields or S2_REC_FIELDS).split(",")
            if f.strip() and f.strip() != "tldr"
        ) or S2_REC_FIELDS
        url = ("https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"
               + urllib.parse.quote(pid, safe="")
               + "?" + urllib.parse.urlencode({"limit": n, "fields": safe_fields}))
        data, err, status = _get_json(url, self.timeout, {"User-Agent": self._ua()}, retries_429=3)
        if data is None:
            self._event("retrieve_fail", source="s2", query=pid, error=err)
            log.warning("s2 recommendations failed (%s): %s", status, err)
            return []
        items = None
        if isinstance(data, dict):
            items = data.get("recommendedPapers") or data.get("data")
        if not isinstance(items, list):
            self._event("retrieve_fail", source="s2", query=pid, error="malformed payload")
            return []
        out: list[Paper] = []
        for item in items:
            try:
                out.append(_paper_from_s2(item))
            except Exception as exc:
                log.debug("s2 rec item skipped: %s", exc)
        return out[:n]

    def similar_to(self, paper: Paper, max_results: int = 8) -> list[Paper]:
        """相似文献：S2 recommendations 优先，失败回退关键词检索。**绝不抛异常**。"""
        try:
            if not isinstance(paper, Paper):
                return []
            try:
                n = max(1, int(max_results or 8))
            except Exception:
                n = 8
            if self.offline:
                self._event("retrieve_fail", source="offline", query=paper.title,
                            error="offline mode: similar_to skipped")
                return []
            cands: list[Paper] = []
            if paper.id:
                try:
                    cands = self.get_recommendations(paper.id, max_results=n + 2)
                except Exception as exc:
                    self._event("retrieve_fail", source="s2", query=paper.id,
                                error=f"{type(exc).__name__}: {exc}")
                    cands = []
            if not cands and paper.title:
                non_arxiv = [s for s in self._configured_sources() if s != "arxiv"]
                try:
                    cands = self._search_impl(paper.title, max(n * 3, 10), non_arxiv or ["s2"])
                except Exception:
                    cands = []
                if not cands:
                    try:
                        cands = self._search_impl(paper.title, max(n * 3, 10), ["arxiv"])
                    except Exception:
                        cands = []
            cands = [c for c in cands if c.id != paper.id and c.key() != paper.key()]
            ranked = _rank(_dedupe(cands), paper.title)
            return ranked[:n]
        except Exception as exc:
            self._event("retrieve_fail", source="similar_to", query=getattr(paper, "title", ""),
                        error=f"{type(exc).__name__}: {exc}")
            return []

    # -- BibTeX ---------------------------------------------------------- #
    def to_bibtex(self, papers: list[Paper]) -> str:
        """生成标准 ``@article{}`` 条目；无标题条目被跳过。"""
        entries: list[str] = []
        used: Counter[str] = Counter()
        for p in papers or []:
            if not isinstance(p, Paper) or not p.title:
                continue
            key = self._bib_key(p, used)
            fields: list[tuple[str, str]] = [("title", p.title)]
            if p.authors:
                fields.append(("author", " and ".join(p.authors)))
            if p.year:
                fields.append(("year", str(p.year)))
            if p.venue:
                fields.append(("booktitle" if _looks_like_conference(p.venue) else "journal", p.venue))
            doi = p.doi()
            if doi:
                fields.append(("doi", doi))
            if p.url:
                fields.append(("url", p.url))
            arx = p.arxiv_id()
            if arx:
                fields.append(("eprint", arx))
                fields.append(("archivePrefix", "arXiv"))
                primary = _clean_ws(p.extra.get("primary_category"))
                if primary:
                    fields.append(("primaryClass", primary))
            body = ",\n".join(f"  {name} = {{{_bib_escape(value)}}}" for name, value in fields)
            entries.append(f"@article{{{key},\n{body}\n}}")
        return "\n\n".join(entries) + ("\n" if entries else "")

    def _bib_key(self, p: Paper, used: Counter[str]) -> str:
        surname = ""
        if p.authors:
            first = p.authors[0]
            surname = first.split(",")[0] if "," in first else first.split()[-1]
        surname = _ascii_fold(surname)
        surname = re.sub(r"[^a-z0-9]", "", surname.lower()) or "anon"
        year = str(p.year) if p.year else "n.d."
        word = "untitled"
        for tok in _terms(p.title):
            if tok in _TITLE_STOPWORDS:
                continue
            if len(tok) < 3 and not ("\u4e00" <= tok <= "\u9fff"):
                continue
            word = _ascii_fold(tok)
            word = re.sub(r"[^a-z0-9]", "", word.lower()) or "untitled"
            break
        base = f"{surname}{year}{word}"
        used[base] += 1
        return base if used[base] == 1 else f"{base}{chr(ord('a') + used[base] - 2)}"

    # -- 综述 ------------------------------------------------------------ #
    def format_review(self, papers: list[Paper], max_papers: int = 12) -> str:
        """markdown 文献综述脚手架（含主题聚类与 LLM 待填缺口）。"""
        lang = "en" if str(getattr(self, "language", "zh")).lower().startswith("en") else "zh"
        L = _REVIEW_LABELS[lang]
        items = [p for p in (papers or []) if isinstance(p, Paper)]
        unique = _dedupe(items)
        shown = unique[: max(0, int(max_papers or 0))] if max_papers else unique

        lines: list[str] = []
        lines.append(f"## {L['overview']}")
        lines.append("")
        lines.append(L["total"].format(n=len(items), u=len(unique)))
        counts = Counter()
        for p in items:
            for src in str(p.source or "unknown").split("+"):
                if src.strip():
                    counts[src.strip()] += 1
        breakdown = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        lines.append(L["sources"].format(breakdown=breakdown or L["none"]))
        years = sorted({int(p.year) for p in items if p.year})
        span = f"{years[0]}–{years[-1]}" if years else L["none"]
        lines.append(L["years"].format(span=span))
        lines.append("")

        themes = _cluster_themes(unique, lang=lang)
        lines.append(f"## {L['themes']}")
        lines.append("")
        if themes:
            for idx, (name, members) in enumerate(themes, 1):
                lines.append(f"**{L['theme']} {idx}** — {name} ({len(members)})")
                lines.append("")
                for p in members:
                    lines.append(f"- {p.title} ({p.year or 'n.d.'})")
                lines.append("")
        else:
            lines.append(f"_{L['no_themes']}_")
            lines.append("")

        lines.append(f"## {L['representative']}")
        lines.append("")
        if shown:
            for p in shown:
                venue = p.venue or L["unknown_venue"]
                url = p.url or p.pdf_url or ""
                head = L["bullet"].format(
                    title=p.title,
                    year=p.year or "n.d.",
                    venue=venue,
                    cites=int(p.citation_count or 0),
                )
                lines.append(f"- **{head}**" + (f" — {url}" if url else ""))
                lines.append(f"  {_digest(p)}")
        else:
            lines.append(f"- _{L['none']}_")
        lines.append("")

        lines.append(f"## {L['gaps']}")
        lines.append("")
        lines.append(L["gaps_note"])
        lines.append("")
        for i in range(1, 4):
            lines.append(f"- [ ] TODO(LLM): {L['gap_item'].format(i=i)}")
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# BibTeX 辅助
# --------------------------------------------------------------------------- #

_CONFERENCE_HINTS = (
    "conference", "proceedings", "workshop", "symposium", "meeting", "colloquium",
    "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv", "acl", "emnlp",
    "naacl", "coling", "kdd", "www", "sigir", "aaai", "ijcai", "acl", "interspeech",
    "icra", "iros", "emnlp", "tmlr", "acl", "usenix", "osdi", "sosp", "nsdi",
)


def _looks_like_conference(venue: str) -> bool:
    low = str(venue or "").lower()
    return any(h in low for h in _CONFERENCE_HINTS)


def _ascii_fold(text: Any) -> str:
    try:
        return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    except Exception:
        return re.sub(r"[^A-Za-z0-9]", "", str(text or ""))


_BIB_ESCAPE_MAP: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def _bib_escape(value: Any) -> str:
    """转义 ``& % $ # _ { } ~ ^ \\``（单遍映射，避免二次转义；花括号保持平衡）。"""
    return "".join(_BIB_ESCAPE_MAP.get(ch, ch) for ch in str(value if value is not None else ""))


# --------------------------------------------------------------------------- #
# 综述辅助（主题聚类 / 摘要消化）
# --------------------------------------------------------------------------- #

_REVIEW_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "overview": "检索概览",
        "total": "- 命中总数 {n} 条（去重后 {u} 条）",
        "sources": "- 来源分布：{breakdown}",
        "years": "- 年份跨度：{span}",
        "themes": "主题聚类",
        "theme": "主题",
        "no_themes": "文献不足，无法聚类。",
        "representative": "代表性工作",
        "bullet": "{title}（{year}，{venue}，被引 {cites}）",
        "unknown_venue": "未知来源",
        "gaps": "研究缺口",
        "gaps_note": "> ⚠️ 以下条目为**占位符**，须由 LLM 在 `s1_literature` 阶段结合上述文献填写（TODO(LLM)）。",
        "gap_item": "待补：现有方法在何种设定下的何种失效模式尚未被系统研究（缺口 {i}）。",
        "none": "无",
    },
    "en": {
        "overview": "Search Overview",
        "total": "- Hits: {n} (unique {u})",
        "sources": "- Sources: {breakdown}",
        "years": "- Year span: {span}",
        "themes": "Topic Clusters",
        "theme": "Theme",
        "no_themes": "Not enough papers to cluster.",
        "representative": "Representative Work",
        "bullet": "{title} ({year}, {venue}, {cites} citations)",
        "unknown_venue": "unknown venue",
        "gaps": "Research Gaps",
        "gaps_note": "> ⚠️ The items below are **placeholders** to be filled by the LLM in the `s1_literature` stage (TODO(LLM)).",
        "gap_item": "TODO: which failure mode of current methods under which setting is still unstudied (gap {i}).",
        "none": "none",
    },
}


def _digest(paper: Paper, max_sentences: int = 3, limit: int = 360) -> str:
    """2–3 句摘要（优先 abstract，其次 tldr），超长截断。"""
    text = paper.abstract or paper.tldr or ""
    text = _clean_ws(text)
    if not text:
        return "（无摘要）"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
    picked: list[str] = []
    size = 0
    for s in sentences[:max_sentences]:
        if size + len(s) > limit and picked:
            break
        picked.append(s)
        size += len(s) + 1
    out = " ".join(picked) if picked else text
    if len(out) > limit:
        out = out[: limit - 1].rstrip() + "…"
    return out


def _cluster_themes(papers: list[Paper], lang: str = "zh", min_themes: int = 2,
                    max_themes: int = 5) -> list[tuple[str, list[Paper]]]:
    """简单关键词共现聚类；输出 2–5 个主题（确定性）。"""
    L = _REVIEW_LABELS[lang if lang in _REVIEW_LABELS else "zh"]
    if not papers:
        return []
    per_paper: list[set[str]] = []
    counts: Counter[str] = Counter()
    for p in papers:
        toks = _dedup_preserve(
            t for t in _terms(f"{p.title} {p.abstract[:400]} {p.tldr[:200]}")
            if len(t) >= 3 or "\u4e00" <= t <= "\u9fff"
        )
        term_set = set(toks[:24])
        per_paper.append(term_set)
        counts.update(term_set)
    top_terms = [t for t, _c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:12]

    # 共现并查集
    parent: dict[str, str] = {t: t for t in top_terms}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for t1 in top_terms:
        for t2 in top_terms:
            if t1 >= t2:
                continue
            shared = sum(1 for s in per_paper if t1 in s and t2 in s)
            if shared >= 2:
                union(t1, t2)

    buckets: dict[str, list[str]] = {}
    for t in top_terms:
        buckets.setdefault(find(t), []).append(t)
    themes_terms = [sorted(v, key=lambda t: (-counts[t], t)) for v in buckets.values()]

    # 补齐到 min_themes
    if len(themes_terms) < min_themes:
        flat = [t for terms in themes_terms for t in terms]
        for t in top_terms:
            if len(themes_terms) >= min_themes:
                break
            if t not in flat:
                themes_terms.append([t])
                flat.append(t)
        while len(themes_terms) < min_themes:
            themes_terms.append([])

    def label(terms: list[str]) -> str:
        return " · ".join(terms[:3]) if terms else (L["none"] if lang == "en" else "综合")

    # 软分配：论文与主题词集有交集即归入该主题（关键词主题轴，允许一篇论文跨主题）
    themes: list[tuple[str, list[Paper]]] = []
    for terms in themes_terms:
        tset = set(terms)
        members = [p for p, pt in zip(papers, per_paper) if pt & tset]
        if members:
            themes.append((label(terms), members))
    if not themes:
        themes = [(label(themes_terms[0]) if themes_terms else L["none"], list(papers))]

    if len(themes) > max_themes:
        themes.sort(key=lambda kv: -len(kv[1]))  # 稳定排序 -> 确定性
        kept = themes[:max_themes]
        covered = {id(p) for _n, m in kept for p in m}
        rest = [p for p in papers if id(p) not in covered]
        if rest:
            kept = kept[: max_themes - 1] + [(L["none"] if lang == "en" else "其他", rest)]
        themes = kept

    # 退化输入（例如整篇只有一个关键词）也要给出 2 个主题
    while len(themes) < min_themes:
        themes.append((L["none"] if lang == "en" else "综合", list(papers)))
    return themes[:max_themes]
