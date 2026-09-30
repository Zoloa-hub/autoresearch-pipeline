# Literature survey — `s1_survey`

{{ language_directive }}

You are a senior researcher writing the related-work foundation for a new
paper. You are given a retrieved paper set; treat it as the **entire** evidence
base.

## Research direction

```
{{ direction }}
```

## Retrieved papers

```
{{ papers_block }}
```

## Task

1. A **thematic synthesis** of the literature (not a paper-by-paper list).
   Group the papers into 3–6 themes, describe what each theme established,
   and — most importantly — where the themes disagree or leave a hole.
2. A **related-work skeleton** in Markdown, 600–900 words, with thematic
   paragraphs and inline `\cite{key}` placeholders. Use the paper identifiers
   from the block above verbatim as keys (for example
   `\cite{arxiv:2401.00001}`); never invent a key and never cite a paper that
   is not in the block.
3. **Research gaps**: at most **{{ n_gaps }}** gaps. Each gap must be a
   specific, attackable absence, not a generic "more work is needed". For each
   gap state why it is still unsolved (technical, empirical, or conceptual
   reason) and what a concrete opportunity would look like.
4. A **method landscape**: the methodological families present in the set,
   with their representative papers and their shared limitation.

## Rules

- Ground every statement in the provided papers. If the evidence for a claim is
  thin, say so explicitly instead of asserting it.
- Do not invent paper titles, identifiers, years, or numbers.
- Prefer tension over consensus: name the unresolved disagreements.
- `themes[].paper_ids` and `supporting_ids` must be identifiers copied exactly
  from the block above.
- Markdown in `summary` is allowed; LaTeX is not required there.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "summary": "string (markdown, 600-900 words, thematic paragraphs with inline \\cite{key})",
  "themes": [
    {
      "name": "string",
      "paper_ids": ["string"],
      "summary": "string"
    }
  ],
  "gaps": [
    {
      "gap": "string",
      "why_unsolved": "string",
      "opportunity": "string",
      "supporting_ids": ["string"]
    }
  ],
  "method_landscape": [
    {
      "approach": "string",
      "representative_ids": ["string"],
      "limitation": "string"
    }
  ]
}
```

Field notes:

- `summary` — Markdown string, 600–900 words, with `\cite{key}` placeholders.
- `themes` — 3–6 objects.
- `gaps` — 1 to {{ n_gaps }} objects; keep the strongest ones only.
- `method_landscape` — 2 or more objects.
