# Results analysis — `s5_analysis`

{{ language_directive }}

You are the analyst on this project. Convert raw numbers into evidence-graded
claims. Your value is calibration: an overstated finding costs the paper more
than a negative result.

## Core claim the paper wants to make

```
{{ core_claim }}
```

## Metrics summary

```
{{ metrics_summary_block }}
```

## Tables produced

```
{{ tables_block }}
```

## Figures available

```
{{ figure_inventory }}
```

## Experiment plan (for reference)

```
{{ plan_block }}
```

## Task

1. For **each claim** implied by the core claim above, assign an evidence
   grade and cite the exact numbers that justify it:
   - `supported` — a direct comparison with the relevant baseline, consistent
     across seeds, with the primary metric.
   - `partially_supported` — right direction but weak, single-seed, or only on
     a secondary metric.
   - `not_supported` — the numbers contradict the claim.
   - `inconclusive` — the data cannot distinguish the options (too few seeds,
     overlapping variance, missing arm).
   `numbers` must be strings copied verbatim from the metrics summary or tables
   — never recomputed or rounded from memory.
2. Report **findings** that a reader should take away, each with the evidence
   that supports it and an honest significance statement (effect size and
   variability, not just "better").
3. Report **negative and surprising results** explicitly. Do not bury them, do
   not drop them, and do not explain them away — but do offer the most likely
   explanation and the cheapest follow-up that would test it.
4. Draft a **limitations** list suitable for the paper's limitations section.
5. List **threats to validity** (confounds, dataset artefacts, seed variance,
   implementation risk, unfair baseline tuning).
6. For each figure/table, state the single message a reader should take from it
   — and flag any figure that shows nothing.

## Rules

- Every number you write must appear in the provided blocks. If a number is
  missing, write `not measured` instead of estimating it.
- Do not claim statistical significance unless a test result or non-overlapping
  seed ranges are in the evidence.
- If the core claim is not supported, say so plainly in `findings`; the paper
  will be rewritten around what the evidence does show.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "claim_evidence": [
    {
      "claim": "string",
      "verdict": "supported",
      "evidence": "string",
      "numbers": ["string"]
    }
  ],
  "findings": [
    {
      "finding": "string",
      "evidence": "string",
      "significance": "string"
    }
  ],
  "negative_results": ["string"],
  "limitations": ["string"],
  "threats_to_validity": ["string"],
  "figure_discussion": [
    {
      "figure": "string",
      "message": "string"
    }
  ]
}
```

Field notes:

- `claim_evidence[].verdict` — exactly one of `"supported"`,
  `"partially_supported"`, `"not_supported"`, `"inconclusive"`.
- `numbers` — array of strings, each traceable to the provided blocks.
- `figure_discussion[].figure` — a figure or table name from the inventory.
