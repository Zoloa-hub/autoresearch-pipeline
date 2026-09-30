# Final run report — `s9_report`

{{ language_directive }}

You are writing the **human-facing run report** for an automated research run.
The reader is a researcher who wants to know, in two minutes, what was tried,
what happened, and what to do next.

## Run summary

```
{{ run_summary_block }}
```

## Artifacts produced

```
{{ artifacts_block }}
```

## Review outcome

```
{{ review_block }}
```

## Output format — read this carefully

**Write Markdown directly. Do NOT output JSON.** No JSON object, no JSON code
fence, no YAML front matter. The reply itself *is* the narrative document, and it
will be written to `report/RUN_SUMMARY.md` verbatim.

Note: `report/FINAL_REPORT.md` is a **different** file — it is generated
deterministically by code (stage tables, comparison numbers, the open-issues
list) and your output does not replace it. Do not claim to be writing it.

## Required structure

1. `# ` — a title naming the direction and the run.
2. **Executive summary** — 3–5 sentences: the direction, the selected idea, the
   headline result, and the verdict.
3. **Direction and literature** — the research question and the gaps that
   motivated the work.
4. **Selected idea** — the hypothesis and why it was chosen over the
   alternatives.
5. **Experiment plan** — milestones, baseline, and compute actually used.
6. **Results** — a Markdown table of the key numbers, plus a short reading of
   each. Every number must come from the blocks above.
7. **Figures and tables** — a list of the artifacts with their paths.
8. **Analysis** — what the evidence supports, what it does not, and any
   negative or surprising results.
9. **Peer review** — score, verdict, and the major weaknesses with their status.
10. **Artifacts index** — the files produced, with relative paths.
11. **Next steps** — 3–6 concrete, ordered actions; each must be something a
    person could start today.
12. **Limitations** — an honest list, including anything that failed or was
    skipped during the run.

## Rules

- Every factual statement must be traceable to the blocks above. Do not add
  results, numbers, or citations from memory; if something is missing, write
  `not available` rather than guessing.
- Report failures and skipped stages plainly — a run that hit a wall is a
  useful report.
- Use relative paths for artifacts.
- No emoji, no marketing language, no exclamation marks.

Reply with the Markdown report and nothing else.
