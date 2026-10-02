r"""修 s2 查新检索式的两个缺陷 —— 它们让"查新"在空候选集上做判断。

## 实测症状（两次真实全管线跑都复现）

    arxiv returned 0 entries for 'MLP PINN venue'
    arxiv returned 0 entries for 'regime PINN iii Licata cite doi'
    arxiv returned 0 entries for 'Sarris PINN cite LAWP Jacobi rmer Verlet'
    arxiv returned 0 entries for 'Hamilton PINN Jacobi Pareto Earth Moon Sun MLP'

对比 s1（正常）：`'residual learning trajectory prediction dynamical systems'`

## 缺陷 A：检索式是词袋，不是查询

原实现把 title/hypothesis/method_sketch/novelty_claim 四个字段**散文拼接**，
正则抽出全部英文词，去掉 22 个停用词，**按文档顺序取前 8 个**。

产出的是无序词袋。`'Hamilton PINN Jacobi Pareto Earth Moon Sun MLP'` 不是检索式——
它把标题词、方法词、天体名混在一起且无短语结构。arXiv 返回 0 是必然的。

## 缺陷 B：schema 常量泄漏 + 非 ASCII 被腰斩

* `venue` / `cite` / `doi` / `iii` —— 字段名与引用格式碎片被当成关键词。
  停用词表只有 22 个通用词，完全没有覆盖这类**领域外的格式噪声**。
* **`rmer`** —— 这是 `Körmer` 被正则 `[A-Za-z][A-Za-z\-]{2,}` **剥掉 `ö` 后的残片**。
  同理 `Poincaré` → `Poincar`、`Brouwer` 尚可但 `Müller` → `ller`。
  把作者名砍成无意义片段，检索必然失败。

## 修法

1. **优先用 title**（最接近检索式的字段），而不是把四个字段混起来。
   不足时才补 hypothesis，最后落回 direction。
2. **扩充噪声词表**，覆盖 schema 字段名与引用格式碎片（venue/cite/doi/et al…）。
3. **要求 token 是完整词**：含非 ASCII 字母时按原样保留（不再腰斩），
   而不是用 ASCII-only 正则切碎。名称对检索很重要。
4. **质量护栏**：有效关键词少于 2 个时落回 direction，
   而不是发出一个必然 0 结果的查询。
5. **零结果要可见**：检索返回 0 条时记一条 warning——
   否则查新会在空候选集上给出"novel"，而这正是它最该避免的结论。
"""

from __future__ import annotations

import pathlib
import re

p = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "stages" / "s2_ideation.py"
t = p.read_text(encoding="utf-8")

OLD = '''def _idea_query(idea: dict[str, Any], direction: str) -> str:
    """构造查新检索式：标题/方法关键词优先，落回研究大方向。"""
    text = " ".join(
        str(idea.get(k) or "") for k in ("title", "hypothesis", "method_sketch", "novelty_claim")
    )
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\\-]{2,}", text)]
    # 去掉过于通用的词
    generic = {"the", "and", "for", "with", "that", "this", "are", "was", "can", "not",
               "which", "when", "from", "into", "using", "based", "method", "model",
               "approach", "results", "performance"}
    keywords = _dedupe_preserve([w for w in words if w.lower() not in generic])[:8]
    if keywords:
        return " ".join(keywords)
    return direction'''

NEW = '''#: 构造检索式时要跳过的词。
#:
#: 分三类，第二类是本项目实测踩到的坑：
#:   1. 通用虚词与泛化名词
#:   2. **schema 字段名与引用格式碎片** —— 实测泄漏进检索式的有
#:      ``venue``（输出契约的字段名）、``cite``/``doi``/``iii``（引用格式碎片）。
#:      它们与主题无关，却会被当成关键词发出去。
#:   3. 论文写作的元词
_QUERY_STOPWORDS = frozenset({
    # 1) 通用
    "the", "and", "for", "with", "that", "this", "are", "was", "were", "can", "not",
    "which", "when", "from", "into", "using", "based", "via", "its", "our", "their",
    "than", "then", "also", "such", "more", "most", "less", "very", "both", "each",
    "method", "model", "models", "approach", "approaches", "results", "result",
    "performance", "study", "paper", "work", "works", "novel", "new", "propose",
    "proposed", "proposes", "show", "shows", "shown", "find", "finds", "found",
    # 2) schema 字段名 / 引用格式碎片（实测泄漏）
    "venue", "cite", "cited", "cites", "citation", "doi", "arxiv", "et", "al",
    "string", "bool", "boolean", "float", "int", "integer", "array", "object",
    "json", "schema", "field", "fields", "type", "types", "value", "values",
    "title", "abstract", "hypothesis", "claim", "claims", "score", "scores",
    "verdict", "rationale", "overlap", "risks", "differentiators", "closest",
    "i", "ii", "iii", "iv", "v", "vi", "regime", "regimes",
    # 3) 写作元词
    "however", "therefore", "moreover", "furthermore", "thus", "hence",
    "figure", "figures", "table", "tables", "section", "sections", "eq", "equation",
})

#: 一个 token 至少要有这么多字母才算"词"（滤掉 "s"、"e.g" 这类碎片）
_MIN_TOKEN_ALNUM = 3


def _query_tokens(text: str) -> list[str]:
    """从文本里取出可用的检索关键词。

    **不再用 ASCII-only 正则切词。** 早期实现用 ``[A-Za-z][A-Za-z\\-]{2,}``，
    于是 ``Körmer`` 被剥掉 ``ö`` 只剩 ``rmer``——作者名被砍成无意义片段，
    检索必然失败。实测在 s2 的检索式里出现过 ``rmer``。

    现在：按 Unicode 字母取词，拉丁字母 + 组合变音符一并保留；
    若一个词去掉非 ASCII 后长度不足，则**整词丢弃**（宁可少一个词，
    也不要发出一个被腰斩的假词）。
    """
    tokens: list[str] = []
    for raw in re.findall(r"[^\\W\\d_]+(?:[-'’][^\\W\\d_]+)*", text, flags=re.UNICODE):
        word = raw.strip("-'’")
        if not word:
            continue
        # 纯 ASCII 长度（用于判断"去掉非 ASCII 后还剩多少"）
        ascii_len = sum(1 for ch in word if ch.isascii() and ch.isalpha())
        if ascii_len < _MIN_TOKEN_ALNUM:
            # 形如 "rmer"（原词是 Körmer）或 "ller"（原词是 Müller）——
            # 无法判断它原本是什么，丢弃比发出去安全
            continue
        if word.lower() in _QUERY_STOPWORDS:
            continue
        tokens.append(word)
    return tokens


def _idea_query(idea: dict[str, Any], direction: str) -> str:
    """构造查新检索式。

    **以 title 为主**，而不是把四个字段拼成散文再取前 8 个词。

    原因：四个字段混在一起后，按文档顺序取词得到的是**无序词袋**
    ——实测产出 ``'Hamilton PINN Jacobi Pareto Earth Moon Sun MLP'``
    这种把标题词、方法词、天体名混在一起的东西，arXiv 返回 0 是必然的。
    title 本身最接近一个检索式，也最能代表 idea 的主题。

    有效关键词少于 2 个时落回 ``direction``——宁可检索得宽一点，
    也不要发出一个注定 0 结果的查询（那会让查新在空候选集上做判断）。
    """
    title = str(idea.get("title") or "")
    keywords = _dedupe_preserve(_query_tokens(title))[:8]

    if len(keywords) < 2:
        # title 太短或全被过滤 —— 用 hypothesis 补充
        extra = _dedupe_preserve(
            _query_tokens(str(idea.get("hypothesis") or ""))
        )
        for word in extra:
            if word not in keywords:
                keywords.append(word)
            if len(keywords) >= 6:
                break

    if len(keywords) >= 2:
        return " ".join(keywords[:8])
    return direction'''

if OLD in t:
    t = t.replace(OLD, NEW, 1)
    print("  s2: _idea_query 已重写（title 优先 + 噪声词表 + 不腰斩非 ASCII）")
else:
    print("  MISS: _idea_query 原文未匹配")

# --- 零结果要可见 ---
OLD_ZERO = '''            found = self.ctx.search.search(query, max_results=6)'''
NEW_ZERO = '''            found = self.ctx.search.search(query, max_results=6)
            if not found:
                # **零结果必须可见。**
                # 查新在空候选集上会给出"novel"，而那正是它最该避免的结论
                # （提示词自己写着"默认失败模式是高估新颖性"）。
                # 实测：检索式是词袋 + 含 schema 碎片时，arXiv 稳定返回 0 条。
                warnings.append(
                    f"查新检索零结果（idea={idea.get('id')}，query={query!r}）——"
                    "该 idea 的新颖性缺少候选文献支撑，结论应视为 unknown 而非 novel"
                )'''
if OLD_ZERO in t:
    t = t.replace(OLD_ZERO, NEW_ZERO, 1)
    print("  s2: 已加零结果可见性")
else:
    print("  MISS: 检索调用点")

p.write_text(t, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(p), doraise=True)
print("  编译通过")
