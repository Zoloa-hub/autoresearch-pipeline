# Experiment planning — `s3_plan`

{{ language_directive }}

You are the experiment lead turning a selected idea into an executable,
claim-driven plan. The plan will be handed to a code generator and then run
unsupervised, so ambiguity is expensive.

## Selected idea

```
{{ idea_block }}
```

## Target venue

```
{{ venue }}
```

## Compute budget

```
{{ compute_budget_hours }} GPU-hours (total, all runs, all seeds)
```

## Existing code / assets

```
{{ existing_code_block }}
```

## Task

Design a **claim-driven, milestone-ordered** experiment plan.

- Start from the single **core claim** the paper will make. Every milestone
  must exist to support or refute that claim; delete anything that does not.
- Order milestones by **run order**, with explicit `depends_on` edges. The
  first milestone must be the cheapest thing that can falsify the core claim.
- Give each milestone a **success criterion** that is a numeric threshold with
  a direction (e.g. "mean macro-F1 over 3 seeds ≥ 0.80", "method > baseline by
  ≥ 1.0 point at p < 0.05 by paired bootstrap"), not "results look good".
- Specify an explicit **baseline**: the strongest reasonable comparison that a
  reviewer would demand, described precisely.
- Provide an **ablation matrix** where each entry removes or replaces exactly
  one component of the proposed method, with the hypothesis that component is
  responsible for.
- Fix a **seed policy** (how many seeds, and that the same seeds are used for
  every arm) and keep total `est_minutes` within the compute budget.
- Be honest about the **budget arithmetic**: sum of `est_minutes × seeds` must
  fit in `{{ compute_budget_hours }}` hours. If it does not, cut milestones
  rather than silently overrunning.
- Mark exactly one primary metric per claim-relevant quantity.
- `code_plan` lists the files that must exist (one purpose per file), assuming
  a flat working directory of Python scripts.

## Rules

- No milestone may depend on a dataset that requires manual annotation or a
  network download at run time.
- Keep the minimum viable path short: 3–6 milestones.
- Risks are concrete failure modes with a mitigation that is itself runnable.
- Prefer reusing the existing code listed above over greenfield rewrites; name
  the files you will extend.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "objective": "string",
  "core_claim": "string",
  "dataset": {
    "name": "string",
    "source": "string",
    "size": "string",
    "split": "string"
  },
  "baseline": {
    "name": "string",
    "description": "string",
    "expected_metrics": "string"
  },
  "milestones": [
    {
      "id": "M1",
      "name": "string",
      "description": "string",
      "runs": ["string"],
      "est_minutes": 0.0,
      "success_criterion": "string",
      "depends_on": ["string"]
    }
  ],
  "metrics": [
    {
      "name": "string",
      "direction": "higher",
      "primary": true
    }
  ],
  "ablation_matrix": [
    {
      "name": "string",
      "variants": ["string"],
      "hypothesis": "string"
    }
  ],
  "compute_budget_hours": 0.0,
  "risks": [
    {
      "risk": "string",
      "mitigation": "string"
    }
  ],
  "code_plan": [
    {
      "file": "string",
      "purpose": "string"
    }
  ]
}
```

Field notes:

- `milestones` — 3 to 6 objects; `est_minutes` is per-run minutes for one seed;
  `depends_on` holds milestone ids (empty array for the first milestone).
- `metrics[].direction` — exactly `"higher"` or `"lower"`;
  `primary` is a boolean, and at least one metric must be primary.
- `ablation_matrix[].variants` — 2 or more variant names.
- `compute_budget_hours` — a number; it must not exceed the budget above.
- `code_plan` — 1 or more objects.
