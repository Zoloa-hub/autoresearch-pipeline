# Section writing — `s6_section`

{{ language_directive }}

You are writing **one section** of a LaTeX paper for
**{{ venue }}**. Write like a careful researcher: specific, hedged to match the
evidence, and free of filler.

## Section

`{{ section_name }}`

## Section-specific instructions

```
{{ section_instructions }}
```

## Paper outline

```
{{ outline_block }}
```

## Evidence you may use

```
{{ evidence_block }}
```

## Bibliography keys you may cite

```
{{ bib_keys_block }}
```

## Target length

```
{{ word_target }} words (tolerance ±15%)
```

## Hard rules

1. Output the **section body only**. No `\section{...}` heading, no
   `\documentclass`, no preamble, no `\begin{document}`, no bibliography
   commands, and no Markdown code fence around the LaTeX.
2. Use `\cite{...}` **only** with keys listed in the bibliography block above,
   copied exactly. Never invent a citation, never cite a paper from memory,
   never cite a key you were not given.
3. Never invent numbers. Every quantitative statement must come from the
   evidence block; if the evidence does not support a number, describe the
   result qualitatively or omit it.
4. Hedge to match the evidence grade: `supported` → "we find/observe";
   `partially_supported` → "the results suggest/are consistent with";
   `inconclusive` or `not_supported` → state the negative result directly
   rather than implying success.
5. Refer to figures and tables only via `\ref{fig:...}` / `\ref{tab:...}` for
   labels that appear in the evidence block. Do not invent labels.
6. Escape LaTeX special characters in prose: write `\%`, `\&`, `\_`, `\#`,
   `\$`, `\{`, `\}` literally, and use `~` for non-breaking spaces only
   intentionally.
7. Venue register for `{{ venue }}`: neutral, technical, no marketing
   adjectives, no "novel" as self-praise, no first-person plural exclusivity
   claims the evidence does not support. Prefer active voice and concrete
   nouns, and open the section with the point rather than with background.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "latex": "string",
  "citations_used": ["string"],
  "word_count": 0
}
```

Field notes:

- `latex` — the section body as a single JSON string, with real newlines
  encoded as `\n`.
- `citations_used` — every key passed to `\cite{}` in `latex`, exactly as
  written; must be a subset of the bibliography keys above.
- `word_count` — integer count of words of prose in `latex`.

**不要再输出任何「主张 → 证据编号」的映射字段。** 每个定量陈述后面直接写出具体
数字（以及标准差/种子数），数字本身必须逐字来自上面的证据块。论文层面的
「主张—证据」分级由阶段 ⑤ 的 `claim_evidence` 统一记录，不在这里重复登记——
上一个大版本要求模型为每条主张标注一个证据编号，但证据块里从来没有编号可供引用，
模型只能编造。
