# LaTeX compile repair — `s7_compile_fix`

{{ language_directive }}

You are repairing a LaTeX build. The engine is **{{ engine }}**. Produce the
minimal set of edits that makes the document compile, without changing its
meaning.

## Errors reported by the engine

```
{{ errors_block }}
```

## Build log (tail)

```
{{ log_tail }}
```

## Source excerpt under suspicion

```
{{ tex_excerpt }}
```

## Diagnostic checklist — check each in order

1. **Unescaped special characters** in prose: `&`, `%`, `$`, `#`, `_`,
   `{`, `}`, `~`, `^`, `\`. Each needs its LaTeX form (`\&`, `\%`, `\$`,
   `\#`, `\_`, `\{`, `\}`, `\textasciitilde{}`, `\textasciicircum{}`,
   `\textbackslash{}`). `_` and `&` inside prose are the most common killers.
2. **Missing packages** — a command is undefined (`! Undefined control
   sequence`) and the package that provides it is absent from the preamble.
   List these in `packages_to_add`.
3. **`\ref` / `\cite` mismatches** — undefined references or citations, missing
   `\label`, a label with an illegal character, or a citation key absent from
   the `.bib`.
4. **Mismatched environments** — an unclosed or wrongly nested
   `\begin{...}` / `\end{...}`, a missing `\end{document}`, or a table/
   figure environment crossing a section boundary.
5. **Encoding** — mojibake, stray BOM, or characters unsupported by the
   engine. Prefer replacing the character over switching engines.
6. **Math-mode errors** — `$` imbalance, `^`/`_` outside math mode, a `&`
   alignment character outside an alignment environment.
7. **Missing files** — an `\input{}` / `\includegraphics{}` target that does
   not exist; note it even if you cannot create the file.

## Rules

- Prefer the smallest edit that fixes the cause. A one-character escape beats a
  paragraph rewrite.
- `patches` are literal search/replace pairs: `old_text` must appear **exactly
  once** in the target file, and `new_text` is its replacement. Do not emit
  whole-file rewrites and do not reflow lines you are not fixing.
- Do not change the paper's claims, numbers, section structure, or labels.
- Do not delete content to silence an error unless the content is genuinely
  malformed (e.g. an orphaned `\end{table}` with no `\begin{table}`).
- Do not add a package that requires a file the project does not have; prefer
  the engine-agnostic option.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "diagnosis": "string",
  "patches": [
    {
      "file": "string",
      "old_text": "string",
      "new_text": "string"
    }
  ],
  "packages_to_add": ["string"],
  "explanation": "string"
}
```

Field notes:

- `patches[].file` — path relative to the paper directory, e.g. `main.tex` or
  `sections/results.tex`.
- `packages_to_add` — package names only, without backslashes or
  `\usepackage`. Use an empty array if none are needed.
- `explanation` — 1–3 sentences on why these edits fix the reported errors.
