"""离线冒烟测试：``python -m autoresearch.tests.test_retrieve``。

**不需要任何网络**：所有网络路径用 ``offline=True`` 短路或直接用纯函数/本地文件覆盖。
末尾打印 ``PASSED n checks``；任一检查失败则 exit code = 1。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --- 允许 `python autoresearch/tests/test_retrieve.py` 直接运行 --------------- #
WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from autoresearch.tools.pdfx import (  # noqa: E402
    download_pdf,
    extract_metadata,
    extract_references,
    extract_sections,
    extract_text,
    pdf_to_markdown,
)
from autoresearch.tools.retrieve import (  # noqa: E402
    CACHE_TTL,
    LiteratureSearch,
    Paper,
    _bare_arxiv_id,
    _deinvert_abstract,
    _dedupe,
    _http_get as _ORIGINAL_HTTP_GET,
    _merge_pair,
    _parse_arxiv_atom,
    _rank,
    _sanitize_arxiv_query,
    relevance_score,
)

try:  # 兄弟模块并行开发中：缺失时用契约同形的本地 shim
    from autoresearch.config import RetrieveConfig  # type: ignore
    _CFG_SOURCE = "autoresearch.config"
except Exception:  # pragma: no cover
    @dataclass
    class RetrieveConfig:  # type: ignore[no-redef]
        sources: list = field(default_factory=lambda: ["arxiv", "s2", "openalex", "crossref"])
        max_results_per_query: int = 8
        cache_dir: Path | None = None
        offline: bool = False
        timeout: float = 20.0
        mailto: str = "autoresearch@example.org"

    _CFG_SOURCE = "local fallback shim"

# --------------------------------------------------------------------------- #
# 检查框架
# --------------------------------------------------------------------------- #

_CHECKS = 0
_FAILURES: list[str] = []


def check(cond: bool, msg: str) -> bool:
    global _CHECKS
    _CHECKS += 1
    ok = bool(cond)
    if ok:
        print(f"  ok   {_CHECKS:02d} {msg}")
    else:
        print(f"  FAIL {_CHECKS:02d} {msg}")
        _FAILURES.append(msg)
    return ok


class Events:
    """EventLogger 的轻量替身（只需 .log）。"""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def log(self, event: str, **fields) -> None:
        self.items.append({"event": event, **fields})

    def names(self) -> list[str]:
        return [i["event"] for i in self.items]

    def find(self, event: str) -> list[dict]:
        return [i for i in self.items if i["event"] == event]


def _sample_papers() -> list[Paper]:
    return [
        Paper(
            id="arxiv:2401.00001", title="Retrieval Augmented Generation for Long Documents",
            abstract="We study retrieval augmented generation for long documents. Our retriever "
                     "uses hierarchical chunking and improves faithfulness. Experiments cover "
                     "three benchmarks.",
            authors=["Jane Doe", "John Smith"], year=2023, venue="arXiv cs.CL",
            url="https://arxiv.org/abs/2401.00001", pdf_url="https://arxiv.org/pdf/2401.00001",
            citation_count=12, source="arxiv", tldr="Hierarchical RAG for long documents.",
            keywords=["retrieval", "generation"], extra={"primary_category": "cs.CL"},
        ),
        Paper(
            id="s2:abc123", title="Sparse and Dense Retrieval Fusion for Question Answering",
            abstract="We combine sparse and dense retrieval for open-domain question answering "
                     "and analyse robustness.",
            authors=["Alice Roe"], year=2022, venue="ACL", url="https://example.org/x",
            citation_count=340, source="s2", tldr="Hybrid retrieval improves QA.",
        ),
        Paper(
            id="doi:10.9/graph", title="Graph Neural Networks for Citation Recommendation",
            abstract="A graph neural network recommends citations in scholarly graphs.",
            authors=["Bob Lee"], year=2019, venue="Journal of Informetrics",
            url="https://doi.org/10.9/graph", citation_count=88, source="openalex",
        ),
    ]


# --------------------------------------------------------------------------- #
# 各检查组
# --------------------------------------------------------------------------- #

def test_paper_dataclass() -> None:
    print("[Paper]")
    p = Paper(
        id="arxiv:2401.00001", title="Attention Is All You Need!", abstract="We propose a model.",
        authors=["Ashish Vaswani", "Noam Shazeer"], year=2017, venue="NeurIPS",
        url="https://arxiv.org/abs/1706.03762", pdf_url="https://arxiv.org/pdf/1706.03762",
        citation_count=100000, source="arxiv", tldr="Transformers.", keywords=["transformer"],
        extra={"primary_category": "cs.CL"},
    )
    d = p.to_dict()
    check(Paper.from_dict(d) == p, "to_dict/from_dict 往返一致")
    check(json.dumps(d, ensure_ascii=False) is not None, "to_dict 可 JSON 序列化")
    check(Paper.from_dict({"unknown": 1, "title": "T"}).title == "T", "from_dict 忽略未知键")
    check(Paper.from_dict(p.to_dict()).to_dict() == d, "to_dict 幂等")

    a = Paper(id="x1", title="Attention Is All You Need", year=2017)
    b = Paper(id="x2", title="attention is all you need!!", year=2017)
    check(a.key() == b.key(), f"同一标题两种拼写 key 相同 ({a.key()})")
    check(a.key() != Paper(id="x3", title="Attention Is All You Need", year=2018).key(),
          "同年份不同则 key 不同")
    check(Paper(title="", id="arXiv:2401.00001").key() == "arxiv240100001",
          "标题为空时 key 退回 id")
    check(Paper(title="注意力机制", year=2020).key().startswith("注意力机制"),
          "中文标题保留（Unicode 字母数字）")


def test_deinvert_abstract() -> None:
    print("[OpenAlex abstract]")
    idx = {"Retrieval": [0], "augmented": [1], "generation": [2], "improves": [3],
           "faithfulness": [4]}
    check(_deinvert_abstract(idx) == "Retrieval augmented generation improves faithfulness",
          "_deinvert_abstract 重建正常语序")
    shuffled = {"generation": [2], "Retrieval": [0], "augmented": [1]}
    check(_deinvert_abstract(shuffled) == "Retrieval augmented generation",
          "_deinvert_abstract 与字典插入顺序无关")
    check(_deinvert_abstract(None) == "" and _deinvert_abstract({}) == "",
          "空/None 反演索引返回空串")
    check(_deinvert_abstract({"a": 0, "b": 1}) == "a b", "单整数位置亦可处理")


def test_arxiv_helpers() -> None:
    print("[arXiv helpers]")
    check(_sanitize_arxiv_query('retrieval "augmented" generation?') == 'all:"retrieval augmented generation"',
          "裸查询被包成 all:\"...\" 且剔除非法字符")
    check(_sanitize_arxiv_query("ti:retrieval augmented") == 'ti:"retrieval augmented"',
          "已带字段前缀时保留 ti:")
    check(_sanitize_arxiv_query('all:"RAG & LLM"') == 'all:"RAG LLM"',
          "引号/& 等非法字符被清理")
    check(_bare_arxiv_id("http://arxiv.org/abs/2401.00001v2") == "2401.00001",
          "arXiv id 去掉版本号")

    atom = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v3</id>
    <published>2024-01-01T00:00:00Z</published>
    <title>Retrieval
  Augmented Generation</title>
    <summary> We study   RAG. </summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <link href="http://arxiv.org/abs/2401.00001v3" rel="alternate"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v3"/>
    <arxiv:primary_category term="cs.CL"/>
    <arxiv:comment>18 pages</arxiv:comment>
  </entry>
</feed>"""
    papers = _parse_arxiv_atom(atom)
    check(len(papers) == 1, "Atom 解析返回 1 条")
    if papers:
        ap = papers[0]
        check(ap.id == "arxiv:2401.00001", "entry/id -> arxiv:<bare id>")
        check(ap.title == "Retrieval Augmented Generation", "entry/title 折叠换行")
        check(ap.abstract == "We study RAG.", "entry/summary 折叠空白")
        check(ap.authors == ["Jane Doe", "John Smith"], "entry/author/name")
        check(ap.year == 2024, "entry/published -> year")
        check(ap.pdf_url == "http://arxiv.org/pdf/2401.00001v3", "link[@title='pdf'] -> pdf_url")
        check(ap.venue == "cs.CL", "primary_category -> venue-ish")
        check(ap.extra.get("comment") == "18 pages", "comment -> extra")
        check(ap.source == "arxiv", "source 标记为 arxiv")
    check(_parse_arxiv_atom("<feed></feed>") == [], "无 entry 的 XML 返回 []")
    check(_parse_arxiv_atom("not xml at all") == [], "畸形 XML 返回 [] 而非抛异常")


def test_bibtex() -> None:
    print("[BibTeX]")
    searcher = LiteratureSearch(RetrieveConfig(offline=True), cache_dir=_TMP / "cache_bib")
    p1 = Paper(
        id="doi:10.1/x", title="RAG & the 100% solution_ with #hashes $math$ ~tilde^caret\\slash",
        authors=["Jane Q. Doe", "John Smith"], year=2021, venue="Journal of Retrieval",
        url="https://doi.org/10.1/x", source="crossref", extra={"doi": "10.1/x"},
    )
    p2 = Paper(id="openalex:W1", title="Retrieval Without Authors", year=2020,
               venue="Proceedings of ACL", source="openalex")
    p3 = Paper(id="s2:none", title="", abstract="no title here", source="s2")
    p4 = Paper(
        id="arxiv:2401.00001", title="Hierarchical Retrieval for Long Documents",
        authors=["Mei Chen"], year=2024, venue="cs.CL", source="arxiv",
        url="https://arxiv.org/abs/2401.00001",
        extra={"arxiv_id": "2401.00001", "primary_category": "cs.CL"},
    )
    bib = searcher.to_bibtex([p1, p2, p3, p4])
    entries = re.findall(r"@article\{([^,]+),\n(.*?)\n\}", bib, re.S)
    check(len(entries) == 3, f"无标题条目被跳过（得到 {len(entries)} 条）")
    check(bib.count("{") == bib.count("}"), "花括号平衡")
    check(r"\&" in bib and r"\%" in bib and r"\$" in bib and r"\#" in bib and r"\_" in bib,
          "& % $ # _ 全部转义")
    check(r"\textasciitilde{}" in bib and r"\textasciicircum{}" in bib
          and r"\textbackslash{}" in bib, "~ ^ \\ 也按 LaTeX 文本命令转义")
    check("author = {Jane Q. Doe and John Smith}" in bib, "多作者用 ' and ' 连接")

    by_key = {k: body for k, body in entries}
    no_author = [b for k, b in entries if "author" not in b and "Retrieval Without Authors" in b]
    check(len(no_author) == 1, "无作者条目不含 author 字段")
    arx = [b for k, b in entries if "eprint = {2401.00001}" in b]
    check(len(arx) == 1, "arXiv 条目含 eprint")
    check("archivePrefix = {arXiv}" in arx[0] and "primaryClass = {cs.CL}" in arx[0],
          "arXiv 条目含 archivePrefix/primaryClass")
    check("booktitle = {Proceedings of ACL}" in bib, "会议 venue 用 booktitle")
    check("journal = {Journal of Retrieval}" in bib, "期刊 venue 用 journal")
    check("doi = {10.1\\_x}" not in bib and "doi = {10.1/x}" in bib, "doi 原样写入（/ 不转义）")
    keys_ok = all(re.match(r"^[a-z0-9]+$", k, re.I) for k, _ in entries)
    check(keys_ok, "bibtex key 仅含字母数字")
    check(searcher.to_bibtex([]) == "\n" or searcher.to_bibtex([]) == "", "空输入返回空串")


def test_merge_and_dedupe() -> None:
    print("[Dedupe / merge]")
    dups = [
        Paper(id="arxiv:1", title="Retrieval Augmented Generation for Long Documents",
              abstract="We study long-document RAG.", authors=["A B"], year=2023,
              venue="arXiv cs.CL", source="arxiv", citation_count=5),
        Paper(id="s2:2", title="Retrieval-Augmented Generation for Long Documents!",
              abstract="", authors=[], year=2023, venue="ACL", source="s2",
              citation_count=42, pdf_url="https://example.org/p.pdf"),
        Paper(id="doi:3", title="Retrieval Augmented Generation for Long Document",
              abstract="Another abstract.", authors=["C D"], year=2023, source="crossref",
              citation_count=1),
    ]
    merged = _dedupe(dups)
    check(len(merged) == 1, f"3 条近似重复合并为 1 条（得到 {len(merged)}）")
    if merged:
        m = merged[0]
        check(m.citation_count == 42, "合并后 citation_count 取最大值")
        check(bool(m.abstract), "合并后保留非空 abstract")
        check(bool(m.authors), "合并后保留非空 authors")
        check(bool(m.pdf_url), "合并后补齐 pdf_url")
        check(set(m.source.split("+")) == {"arxiv", "s2", "crossref"},
              f"source 以 + 连接多源（{m.source}）")
        check(bool(m.venue), "合并后保留 venue")

    a = Paper(id="i1", title="A Study of Retrieval", abstract="short", year=2020, source="arxiv",
              citation_count=1)
    b = Paper(id="i2", title="A Study of Retrieval", abstract="a much longer abstract here",
              year=2020, source="s2", citation_count=9, pdf_url="https://p.pdf")
    pair = _merge_pair(*(
        (a, b) if len(a.abstract) >= len(b.abstract) else (b, a)
    ))
    check(pair.citation_count == 9 and pair.pdf_url == "https://p.pdf",
          "_merge_pair 直接调用也能补齐字段")
    check(_dedupe([Paper(id="u1", title="Completely Different Topic One", year=2020),
                   Paper(id="u2", title="Another Unrelated Subject Two", year=2020)]).__len__() == 2,
          "不相关标题不会被误合并")
    check(len(_dedupe([Paper(id="e1", title=""), Paper(id="e2", title="")])) == 2,
          "空标题不参与近似去重")


def test_relevance_and_sort() -> None:
    print("[Relevance sort]")
    papers = _sample_papers()
    q = "retrieval augmented generation for long documents"
    r1 = [p.id for p in _rank(papers, q)]
    r2 = [p.id for p in _rank(list(reversed(papers)), q)]
    check(r1 == r2, "固定输入集合的排序完全确定（与输入顺序无关）")
    check(r1[0] == "arxiv:2401.00001", "与查询最相关的论文排第一")
    check(relevance_score(papers[0], q) > relevance_score(papers[2], q),
          "相关论文得分高于无关论文（即便后者引用更多）")
    check(relevance_score(papers[0], "") == 0.0, "空查询得分为 0")
    check(isinstance(relevance_score(papers[0], q), float), "得分是 float")


def test_cache() -> None:
    print("[Cache]")
    cache_dir = _TMP / "cache_ttl"
    ev = Events()
    s = LiteratureSearch(RetrieveConfig(), cache_dir=cache_dir, event_logger=ev)
    papers = _sample_papers()[:2]
    s._cache_store("arxiv", "rag for long docs", 5, papers, sort="relevance")
    path = s._cache_file("arxiv", "rag for long docs", 5, "relevance")
    check(path.name.startswith("arxiv_") and path.suffix == ".json", "缓存文件名 <source>_<hash>.json")
    check(len(path.stem.split("_")[1]) == 40, "缓存 hash 是 sha1（40 hex）")
    check(path.exists(), "缓存文件已落盘")
    check(json.loads(path.read_text("utf-8"))["papers"][0]["title"] == papers[0].title,
          "缓存内容为 {ts, papers}")

    ev2 = Events()
    s2 = LiteratureSearch(RetrieveConfig(), cache_dir=cache_dir, event_logger=ev2)
    loaded = s2._cache_load("arxiv", "rag for long docs", 5, "relevance")
    check(loaded is not None and [p.to_dict() for p in loaded] == [p.to_dict() for p in papers],
          "缓存命中返回相同的 papers")
    check("retrieve_cache_hit" in ev2.names(), "命中时记录 retrieve_cache_hit 事件")
    check(s2._cache_load("arxiv", "different query", 5, "relevance") is None,
          "不同 query 不命中")

    data = json.loads(path.read_text("utf-8"))
    data["ts"] = time.time() - (CACHE_TTL + 3600)
    path.write_text(json.dumps(data), "utf-8")
    check(s2._cache_load("arxiv", "rag for long docs", 5, "relevance") is None,
          "ttl 过期（ts 置为 8 天前）后视为 miss")

    # 不可创建的缓存目录 -> 无缓存但绝不抛异常（不触网）
    blocker = _TMP / "blocker_file"
    blocker.write_text("not a dir", "utf-8")
    try:
        s3 = LiteratureSearch(RetrieveConfig(), cache_dir=blocker / "sub", event_logger=Events())
        check(s3._cache_enabled is False, "缓存目录不可创建时自动降级（不抛异常）")
        check(s3._cache_load("arxiv", "q", 1) is None, "降级后缓存读取直接 miss")
        s3._cache_store("arxiv", "q", 1, _sample_papers()[:1])
        check(not (blocker / "sub").exists(), "降级后不会创建缓存目录")
    except Exception as exc:  # pragma: no cover
        check(False, f"缓存降级路径抛异常: {exc}")


def test_offline_and_review() -> None:
    print("[Offline + format_review]")
    ev = Events()
    s = LiteratureSearch(RetrieveConfig(offline=True), cache_dir=_TMP / "cache_off", event_logger=ev)
    check(s.search("anything") == [], "offline search 返回 []")
    fails = ev.find("retrieve_fail")
    check(len(fails) >= 1 and any("offline" in str(f.get("error", "")).lower() or
                                  str(f.get("source")) == "offline" for f in fails),
          "offline 时记录 retrieve_fail/offline 事件")
    check(s.search_arxiv("rag") == [], "offline search_arxiv 返回 []")
    check(s.search_semantic_scholar("rag") == [], "offline search_semantic_scholar 返回 []")
    check(s.search_openalex("rag") == [] and s.search_crossref("rag") == [],
          "offline openalex/crossref 返回 []")
    check(s.multi_search(["a", "b"], per_query=2) == [], "offline multi_search 返回 []")
    check(s.similar_to(Paper(id="arxiv:1", title="t")) == [], "offline similar_to 返回 []")
    check(s.get_recommendations("arxiv:1") == [], "offline get_recommendations 返回 []")

    papers = _sample_papers()
    md = s.format_review(papers, max_papers=2)
    check("## 检索概览" in md and "## 主题聚类" in md and "## 代表性工作" in md and "## 研究缺口" in md,
          "中文综述含四个固定小节")
    check("arxiv 1" in md or "arxiv" in md, "概览含来源分布")
    for p in papers:
        check(p.title in md, f"综述含标题：{p.title[:40]}")
    theme_count = len(re.findall(r"\*\*主题 \d+\*\*", md))
    check(2 <= theme_count <= 5, f"主题聚类给出 2–5 个主题（{theme_count}）")
    check("TODO(LLM)" in md, "研究缺口标记为 LLM 待填占位符")
    check(all(p.title in md for p in papers[:2]), "代表性工作遵循 max_papers 之外的标题仍在聚类中出现")

    single = s.format_review([papers[0]])
    check(len(re.findall(r"\*\*主题 \d+\*\*", single)) >= 2, "单篇文献也能给出 ≥2 个主题")
    check(s.format_review([]).count("##") == 4, "空输入仍产出四个小节而不崩溃")

    s.language = "en"
    md_en = s.format_review(papers)
    check("## Search Overview" in md_en and "## Research Gaps" in md_en,
          "language='en' 切换英文标题")
    check("检索概览" not in md_en, "英文模式下不出现中文小节名")


def test_pdfx() -> None:
    print("[pdfx]")
    try:
        import fitz  # type: ignore
    except Exception as exc:  # noqa: BLE001
        # PyMuPDF 是**可选**依赖（pyproject 的 [pdf] 分组）。缺了它应当**跳过**而不是
        # 判定失败：本项目的硬性不变量是「核心零强制依赖」，而 CI 刻意不装任何第三方
        # 依赖。若这里 check(False)，CI 第一次运行就会红，而红的原因是没装可选依赖
        # ——那样的红叉会训练人忽略 CI。
        # 真正的覆盖在 CI 的 extras job 里：它装上 [pdf,figures] 再跑一遍。
        print(f"  SKIP test_pdfx: 缺少可选依赖 PyMuPDF ({exc})")
        print("       （pip install -e .[pdf] 后重跑；CI 的 extras job 会覆盖这部分）")
        return

    pdf_path = _TMP / "fake_paper.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    lines = [
        "Retrieval Augmented Generation for Long Documents",
        "Jane Doe, John Smith",
        "Abstract",
        "We study retrieval augmented generation for long documents.",
        "Our method improves faithfulness by fusing dense and sparse retrieval.",
        "1. Introduction",
        "Retrieval augmented generation (RAG) grounds generation in evidence.",
        "Prior work studies short documents rather than long ones.",
        "2. Method",
        "We propose a hierarchical retriever with query decomposition.",
        "3. Experiments",
        "We evaluate on three long-document benchmarks.",
        "4. Conclusion",
        "Retrieval augmentation scales to long documents.",
        "References",
        "[1] Smith, J. (2020). A Study of Retrieval Augmentation. Journal of Retrieval, 1(1), 1-10.",
        "[2] Doe, J. (2019). Faithful Generation with Evidence. In Proceedings of ACL, pages 1-10.",
        "doi:10.1234/rag.2024.001",
    ]
    y = 72.0
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16.0
    doc.set_metadata({
        "title": "Retrieval Augmented Generation for Long Documents",
        "author": "Jane Doe; John Smith",
        "creationDate": "D:20240101120000Z",
    })
    doc.save(str(pdf_path))
    doc.close()
    check(pdf_path.exists() and pdf_path.stat().st_size > 0, "测试 PDF 已生成")

    text = extract_text(pdf_path)
    check(bool(text.strip()), "extract_text 非空")
    check("Abstract" in text and "Retrieval augmented generation" in text, "extract_text 含正文")
    check(extract_text(pdf_path, max_pages=1).strip() != "", "extract_text 支持 max_pages")
    try:
        extract_text(_TMP / "missing.pdf")
        check(False, "缺失文件应抛 FileNotFoundError")
    except FileNotFoundError:
        check(True, "缺失文件抛 FileNotFoundError")

    secs = extract_sections(pdf_path)
    check("abstract" in secs, f"extract_sections 找到 abstract（keys={sorted(secs)}）")
    check("introduction" in secs, "extract_sections 找到 introduction")
    check("method" in secs and "experiments" in secs and "conclusion" in secs,
          "extract_sections 找到 method/experiments/conclusion")
    check("references" in secs, "extract_sections 找到 references")
    check("__preamble__" in secs and "Retrieval Augmented Generation" in secs["__preamble__"],
          "首个标题之前的内容归入 __preamble__")
    check("hierarchical retriever" in secs.get("method", ""), "method 段落内容正确")
    check("hierarchical retriever" not in secs.get("references", ""),
          "References 之后的正文不会被吞进其它章节")

    refs = extract_references(pdf_path)
    check(len(refs) >= 1, f"extract_references 至少返回 1 条（{len(refs)}）")
    check(all(len(r) >= 20 for r in refs), "参考文献条目已清洗且过滤短碎片")
    check(any("Smith" in r for r in refs), "参考文献内容正确")

    md = pdf_to_markdown(pdf_path)
    check("<!-- page 1 -->" in md, "pdf_to_markdown 含页分隔符")
    check("Retrieval augmented generation" in md, "pdf_to_markdown 含正文")

    meta = extract_metadata(pdf_path)
    check(meta.get("pages") == 1, f"extract_metadata pages==1（{meta.get('pages')}）")
    check(meta.get("doi") == "10.1234/rag.2024.001", f"extract_metadata DOI（{meta.get('doi')}）")
    check(meta.get("year") == 2024, f"extract_metadata year（{meta.get('year')}）")
    check(meta.get("authors") == ["Jane Doe", "John Smith"], f"extract_metadata authors（{meta.get('authors')}）")
    check(meta.get("title", "").startswith("Retrieval Augmented Generation"), "extract_metadata title")
    check(meta.get("source") == "fitz", "extract_metadata source 标记后端")

    # download_pdf：本地 file:// 走通原子写入；非 PDF 与死链条返回 None
    dest = _TMP / "downloaded" / "p.pdf"
    got = download_pdf(pdf_path.as_uri(), dest, timeout=10)
    check(got is not None and dest.exists() and dest.read_bytes()[:4] == b"%PDF",
          "download_pdf 成功下载（file://）并校验 %PDF")
    check(not (dest.parent / (dest.name + ".part")).exists(), "download_pdf 不留 .part 临时文件")
    not_pdf = _TMP / "not_a_pdf.txt"
    not_pdf.write_text("definitely not a pdf", "utf-8")
    check(download_pdf(not_pdf.as_uri(), _TMP / "nope.pdf", timeout=10) is None,
          "download_pdf 拒绝非 PDF 内容")
    check(download_pdf("http://127.0.0.1:9/none.pdf", _TMP / "dead.pdf", timeout=2) is None,
          "download_pdf 网络失败返回 None（不抛异常）")


def test_source_parsers() -> None:
    """各源 JSON -> Paper 的解析（用手工构造的 payload，不需要网络）。"""
    print("[Source parsers]")
    from autoresearch.tools.retrieve import (
        _get_json,
        _paper_from_crossref,
        _paper_from_openalex,
        _paper_from_s2,
        _strip_jats,
    )

    cr = _paper_from_crossref({
        "title": ["Retrieval Augmented Generation: A Survey"],
        "abstract": "&lt;jats:p&gt;We survey &lt;strong&gt;RAG&lt;/strong&gt; methods.&lt;/jats:p&gt;",
        "issued": {"date-parts": [[2021, 5, 1]]},
        "container-title": ["Journal of Retrieval"],
        "is-referenced-by-count": 77,
        "author": [{"given": "Jane", "family": "Doe"}, {"family": "Smith"}],
        "DOI": "10.1/abc",
        "URL": "https://doi.org/10.1/abc",
    })
    check(cr.title == "Retrieval Augmented Generation: A Survey", "crossref title[0]")
    check(cr.abstract == "We survey RAG methods.", f"crossref 摘要去 JATS/HTML（{cr.abstract!r}）")
    check(cr.year == 2021, "crossref issued.date-parts -> year")
    check(cr.venue == "Journal of Retrieval", "crossref container-title -> venue")
    check(cr.citation_count == 77, "crossref is-referenced-by-count -> citations")
    check(cr.authors == ["Jane Doe", "Smith"], "crossref author given/family")
    check(cr.id == "doi:10.1/abc" and cr.doi() == "10.1/abc", "crossref DOI -> id/doi()")
    check(_strip_jats("<jats:p>plain &amp; simple</jats:p>") == "plain & simple", "_strip_jats")

    s2 = _paper_from_s2({
        "paperId": "abc123", "title": "Sparse Retrieval", "abstract": "abs",
        "year": 2020, "venue": "ACL", "citationCount": 12,
        "authors": [{"name": "A B"}], "externalIds": {"DOI": "10.2/xy", "ArXiv": "2001.00001"},
        "url": "https://s2/x", "openAccessPdf": {"url": "https://s2/x.pdf"},
        "tldr": {"text": "A short summary."},
    })
    check(s2.tldr == "A short summary.", "s2 tldr.text -> tldr")
    check(s2.pdf_url == "https://s2/x.pdf", "s2 openAccessPdf.url -> pdf_url")
    check(s2.id == "arxiv:2001.00001", "s2 有 arXiv id 时优先 arxiv: 前缀")
    check(s2.citation_count == 12 and s2.year == 2020 and s2.authors == ["A B"], "s2 基本字段")
    s2b = _paper_from_s2({"paperId": "z9", "title": "No ids"})
    check(s2b.id == "s2:z9" and s2b.citation_count == 0, "s2 无外部 id 时用 s2:<hash>")

    oa = _paper_from_openalex({
        "id": "https://openalex.org/W42", "title": "Long Document RAG",
        "abstract_inverted_index": {"Long": [0], "documents": [1], "matter": [2]},
        "publication_year": 2023, "host_venue": {"display_name": "NeurIPS"},
        "cited_by_count": 9, "doi": "https://doi.org/10.3/zz",
        "authorships": [{"author": {"display_name": "C D"}}],
        "primary_location": {"pdf_url": "https://oa/x.pdf"},
    })
    check(oa.abstract == "Long documents matter", "openalex 反演摘要 -> abstract")
    check(oa.venue == "NeurIPS" and oa.citation_count == 9, "openalex venue/citations")
    check(oa.id == "doi:10.3/zz" and oa.doi() == "10.3/zz", "openalex doi 去 URL 前缀")
    check(oa.pdf_url == "https://oa/x.pdf" and oa.authors == ["C D"], "openalex pdf/authors")
    check(oa.year == 2023 and oa.source == "openalex", "openalex year/source")

    # 429 backoff（本地回环 HTTP，不依赖外部网络）
    try:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        state = {"n": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                state["n"] += 1
                if self.path.startswith("/bad"):
                    body = b"{not json"
                    self.send_response(200)
                elif self.path.startswith("/gone"):
                    body = b"nope"
                    self.send_response(404)
                elif state["n"] == 1:
                    body = b"slow down"
                    self.send_response(429)
                    self.send_header("Retry-After", "0")
                else:
                    body = json.dumps({"data": [{"paperId": "x"}]}).encode()
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # 静音
                return

        srv = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        sleeps: list[float] = []
        data, err, status = _get_json(base + "/json", timeout=5, retries_429=3,
                                      sleep=lambda s: sleeps.append(s))
        check(data is not None and data["data"][0]["paperId"] == "x", "429 后重试成功拿到 JSON")
        check(state["n"] == 2, f"429 恰好重试 1 次（实际请求 {state['n']} 次）")
        check(sleeps == [0.0], f"honour Retry-After（sleeps={sleeps}）")
        bad, bad_err, _st = _get_json(base + "/bad", timeout=5)
        check(bad is None and "JSONDecodeError" in bad_err, "畸形 JSON -> (None, JSONDecodeError)")
        nf, nf_err, nf_status = _get_json(base + "/gone", timeout=5)
        check(nf is None and nf_status == 404 and "404" in nf_err, "404 -> (None, HTTP 404)")
        srv.shutdown()
        srv.server_close()
    except OSError as exc:  # 回环端口被限制：跳过而非误报失败
        print(f"  skip 429-backoff 检查（无法绑定回环端口: {exc}）")

    # 源级兜底：任何源失败都必须返回 [] 并记 retrieve_fail
    import autoresearch.tools.retrieve as R

    ev = Events()
    s = LiteratureSearch(RetrieveConfig(), cache_dir=_TMP / "cache_src", event_logger=ev)
    s.min_interval = 0.0
    original_json = R._get_json
    try:
        R._get_json = lambda *a, **k: (None, "HTTP 404", 404)
        check(s.search_semantic_scholar("q-404-xyz") == [], "S2 404 -> [] 且不抛异常")
        check(s.search_openalex("q-404-xyz") == [], "OpenAlex 错误 -> []")
        check(s.search_crossref("q-404-xyz") == [], "Crossref 错误 -> []")
        check(ev.find("retrieve_fail") and all(
            "error" in f for f in ev.find("retrieve_fail")), "源失败全部记 retrieve_fail（含 error）")
        R._get_json = lambda *a, **k: ({"data": "not-a-list"}, "", 200)
        check(s.search_semantic_scholar("q-mal-formed") == [], "S2 畸形 payload -> []")
        R._get_json = lambda *a, **k: (None, "HTTP 500", 500)
        R._http_get = lambda *a, **k: (0, "", {}, "URLError: blocked")
        check(s.search("q-all-fail", max_results=3, sources=["arxiv", "s2", "openalex", "crossref"]) == [],
              "search 在所有源失败时返回 []（不抛异常）")

        atom = ('<feed xmlns="http://www.w3.org/2005/Atom">'
                '<entry><id>http://arxiv.org/abs/2401.99999v1</id>'
                '<published>2024-02-02T00:00:00Z</published>'
                '<title>Offline Atom</title><summary>s</summary>'
                '<author><name>Zed</name></author></entry></feed>')
        R._http_get = lambda *a, **k: (200, atom, {}, "")
        got = s.search_arxiv("offline-atom-xyz")
        check(len(got) == 1 and got[0].id == "arxiv:2401.99999", "arXiv 成功路径（打桩 HTTP）解析出 Paper")
        R._http_get = lambda *a, **k: (0, "", {}, "URLError: blocked")
        check(s.search_arxiv("offline-atom-fail") == [], "arXiv 网络失败 -> [] 而非抛异常")

        # S2 recommendations：URL 形状 + tldr 字段必须剔除（该端点会 400）+ 解析
        captured: dict = {}

        def fake_json(url, *a, **k):
            captured["url"] = url
            return {"recommendedPapers": [dict(
                paperId="r1", title="Recommended Paper", year=2021, citationCount=3,
                authors=[{"name": "R E"}], externalIds={"DOI": "10.9/rec"},
                url="https://s2/r1", openAccessPdf={"url": "https://s2/r1.pdf"},
                tldr={"text": "rec tldr"},
            )]}, "", 200

        R._get_json = fake_json
        rec = s.get_recommendations("arxiv:2005.11401", max_results=2)
        check("forpaper/ARXIV%3A2005.11401" in captured.get("url", ""),
              f"recommendations URL 使用 S2 paperId（{captured.get('url', '')[:70]}）")
        check("tldr" not in captured.get("url", ""),
              "recommendations fields 剔除 tldr（否则 S2 返回 HTTP 400）")
        check(len(rec) == 1 and rec[0].title == "Recommended Paper" and rec[0].source == "s2",
              "recommendations 结果解析为 Paper")
        R._get_json = lambda *a, **k: (None, "HTTP 400", 400)
        check(s.get_recommendations("arxiv:2005.11401") == [], "recommendations 400 -> []")
        R._get_json = fake_json
        sim = s.similar_to(Paper(id="arxiv:2005.11401", title="Seed Paper"), max_results=1)
        check(len(sim) == 1 and sim[0].title == "Recommended Paper",
              "similar_to 优先使用 recommendations（不打桩网络回退）")
    finally:
        R._get_json = original_json
        R._http_get = _ORIGINAL_HTTP_GET


def test_heading_variants() -> None:
    """章节标题启发式（编号/加粗/大小写变体，以及不该误判的行）。"""
    print("[section headings]")
    from autoresearch.tools.pdfx import _match_heading

    cases = {
        "Abstract": "abstract", "**Abstract**": "abstract", "1. Introduction": "introduction",
        "1.1 Introduction": "introduction", "II. Method": "method", "2 Methods": "method",
        "§2 Related Work": "related_work", "3. Approach": "method", "4. Our Approach": "method",
        "Approach": "method", "5. Experiments": "experiments",
        "Experimental Setup": "experiments", "6. Evaluation": "evaluation",
        "7. Results": "results", "8. Discussion": "discussion",
        "9. Ablation Study": "ablation", "10. Conclusion": "conclusion",
        "Conclusion and Future Work": "conclusion", "11. Limitations": "limitations",
        "References": "references", "Bibliography": "references", "A. Appendix": "appendix",
        "Supplementary Material": "appendix", "Model": "method", "Background": "background",
        "Preliminaries": "background", "Notation": "background", "Intro": "introduction",
        "3.2 Training Details": "experiments",
        "Figure 2. Results": None,
        "We study retrieval augmented generation for long documents.": None,
        "Retrieval Augmented Generation for Long Documents": None,
        "A. Vaswani, N. Shazeer, et al. Attention is all you need.": None,
        "Table 1: Results on three benchmarks": None,
    }
    bad = [(t, want, _match_heading(t)) for t, want in cases.items() if _match_heading(t) != want]
    check(not bad, f"{len(cases)} 个标题变体全部正确（错误：{bad}）")


def test_tools_package() -> None:
    print("[tools package]")
    try:
        import autoresearch.tools as T
        check(hasattr(T, "LiteratureSearch") and hasattr(T, "Paper"), "tools 导出 retrieve 公共名")
        check(hasattr(T, "extract_text") and hasattr(T, "pdf_to_markdown"), "tools 导出 pdfx 公共名")
        try:
            T._definitely_not_a_symbol_  # noqa: B018
            check(False, "未知属性应抛 AttributeError")
        except AttributeError:
            check(True, "未知属性抛 AttributeError")
        sandbox_probe = getattr(T, "Sandbox", None)
        check(True, f"兄弟模块惰性访问安全（Sandbox {'可用' if sandbox_probe else '暂不可用'}）")
    except Exception as exc:
        check(False, f"import autoresearch.tools 失败: {exc}")
    print(f"  (RetrieveConfig 来源: {_CFG_SOURCE})")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main() -> int:
    global _TMP
    _TMP = WORKSPACE / "tmp" / f"test_retrieve_{os.getpid()}"
    if _TMP.exists():
        shutil.rmtree(_TMP, ignore_errors=True)
    _TMP.mkdir(parents=True, exist_ok=True)
    print(f"autoresearch tests (offline)   tmp={_TMP}")
    tests = [
        test_paper_dataclass,
        test_deinvert_abstract,
        test_arxiv_helpers,
        test_bibtex,
        test_merge_and_dedupe,
        test_relevance_and_sort,
        test_cache,
        test_offline_and_review,
        test_source_parsers,
        test_heading_variants,
        test_pdfx,
        test_tools_package,
    ]
    try:
        for fn in tests:
            fn()
    except Exception as exc:  # 任何未预期异常都算失败，但要把已有结果打出来
        import traceback

        traceback.print_exc()
        check(False, f"未预期异常: {type(exc).__name__}: {exc}")
    finally:
        if not os.environ.get("AUTORESEARCH_KEEP_TMP"):
            shutil.rmtree(_TMP, ignore_errors=True)

    if _FAILURES:
        print(f"\nFAILED {len(_FAILURES)}/{_CHECKS} checks")
        for msg in _FAILURES:
            print(f"  - {msg}")
        return 1
    print(f"\nPASSED {_CHECKS} checks")
    return 0


_TMP = WORKSPACE / "tmp" / "test_retrieve_bootstrap"

if __name__ == "__main__":
    sys.exit(main())
