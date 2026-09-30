# Title and abstract — `s6_abstract`

{{ language_directive }}

You are drafting the title and abstract for a paper targeting
**{{ venue }}**. This is the highest-leverage text in the paper: it must be
accurate first and compelling second.

## Candidate titles already proposed

```
{{ title_candidates_block }}
```

## Contributions

```
{{ contributions_block }}
```

## Results

```
{{ results_block }}
```

## Task

1. Choose the **title** and list the alternatives you considered. A good title
   names the problem and the contribution; it avoids hype, avoids a colon-led
   "X: Y" cliché unless the second half is genuinely informative, and is
   readable in one pass.
2. Write the **abstract**, 150–250 words, structured as: the problem and why it
   matters (1–2 sentences) → the gap in existing work (1 sentence) → what we
   do, concretely (2–3 sentences) → the key quantitative results (1–2
   sentences, numbers only from `results_block`) → the implication (1
   sentence).
3. List 4–6 **keywords**.
4. List the **contributions** as bullet-style strings, each independently
   checkable against the evidence.

## Hard rules

- **No citations** in the abstract — no `\cite{}`, no bracketed references, no
  author-year mentions.
- **No numbers that are not in `results_block`.** Do not round or extrapolate.
- If a promised result is missing from the evidence, do not claim it; describe
  the contribution qualitatively instead and keep it in the abstract terms.
- Avoid: "state-of-the-art" unless a direct comparison in the evidence supports
  it, "significantly" unless a significance test is reported, "novel" as
  self-description, and any claim about deployment or generality that was not
  tested.
- Keep the abstract self-contained: no undefined acronyms, no forward
  references to sections.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "title": "string",
  "title_candidates": ["string"],
  "abstract": "string",
  "keywords": ["string"],
  "contributions": ["string"]
}
```

Field notes:

- `title` — the chosen title; it must also appear in `title_candidates`.
- `title_candidates` — 2 to 5 strings, including the chosen title.
- `abstract` — 150–250 words, plain text (LaTeX math allowed, citations not).
- `keywords` — 4 to 6 strings.
- `contributions` — 2 to 5 strings.
