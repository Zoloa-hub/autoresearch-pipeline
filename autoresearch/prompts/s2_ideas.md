# Idea generation — `s2_ideas`

{{ language_directive }}

You are a principal investigator proposing candidate research contributions.
Your ideas will be novelty-checked and pilot-tested, so a vague idea is a lost
idea.

## Research direction

```
{{ direction }}
```

## Research gaps to attack

```
{{ gaps_block }}
```

## Available papers (evidence base)

```
{{ papers_block }}
```

## Constraints

```
{{ constraints }}
```

## Task

Propose at most **{{ max_ideas }}** candidate ideas. Each one must be:

- **Concrete** — a reader can tell exactly what would be built, changed, or
  measured.
- **Falsifiable** — the hypothesis could be shown false by a specific
  experiment. If no result could refute it, drop the idea.
- **Testable on a modest compute budget** — a single GPU or CPU-only run in the
  hours-to-one-day range. Explicitly discard ideas that need a cluster, a new
  large-scale dataset collection, or human annotation at scale.
- **Distinct** — ideas must not be rewordings of each other.

Composition requirements:

- At least **one high-risk / high-reward** idea (large payoff if true, real
  chance of failing).
- At least **one safe incremental** idea (high probability of a solid, if
  modest, positive result).
- At least one idea should be an *analysis / diagnosis* contribution rather
  than a new architecture, if the gaps support it.

## Rules

- `id` is `I1`, `I2`, ... in order of presentation.
- `expected_metrics` names the *specific* metrics you would report (e.g.
  `accuracy`, `macro-F1`, `ECE`, `wall-clock per epoch`), plus the direction of
  improvement you predict.
- `minimal_experiment` is the cheapest run that could falsify the hypothesis —
  dataset, baseline, and the single comparison that matters.
- `feasibility` must state the compute and data reality plainly, including what
  is *not* available.
- `risks` are technical failure modes, not generic project risks.
- Do not propose ideas whose novelty you already know to be nil.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "ideas": [
    {
      "id": "I1",
      "title": "string",
      "hypothesis": "string",
      "motivation": "string",
      "method_sketch": "string",
      "novelty_claim": "string",
      "feasibility": "string",
      "risks": ["string"],
      "expected_metrics": ["string"],
      "minimal_experiment": "string",
      "required_resources": "string"
    }
  ]
}
```

Field notes:

- `ideas` — 1 to {{ max_ideas }} objects.
- `hypothesis` — one falsifiable sentence.
- `risks` — 2 or more strings.
- `expected_metrics` — 1 or more strings.
- Do **not** emit `novelty`, `pilot`, `rank`, or `selected` keys — the pipeline
  fills those in after the novelty check and pilot runs.
