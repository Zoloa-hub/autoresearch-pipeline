# Experimental protocol template

Stage `s6_writing` prompt `s6_section` (section_name=experiments) fills this in.
The headings below are the ones the prompt expects to be able to talk about; add
detail under each rather than deleting a heading, so section generation stays
comparable across runs.

## 1. Research question and core claim

- **Core claim.** One sentence, falsifiable, with the metric and direction.
- **Falsifier.** The observation that would make us abandon the claim.

## 2. Datasets

| Dataset | Source (local path) | Size | Split | Notes |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | e.g. class balance, leakage checks |

Rules: no runtime downloads; every dataset is a local artifact recorded in the
run manifest; report the split protocol and whether the test set was ever used
for model selection.

## 3. Baselines

| Baseline | Why it is the right comparison | Reference or reimplementation |
|---|---|---|
| TODO | TODO | TODO |

Rules: the baseline must be the strongest reasonable comparison, tuned with the
same search budget as the method. State explicitly if a baseline was not tuned.

## 4. Metrics

| Metric | Direction | Primary? | Reported as |
|---|---|---|---|
| TODO | higher/lower | yes/no | mean $\pm$ std over seeds |

## 5. Protocol

- **Seeds.** TODO: `n` seeds, identical across all arms.
- **Selection.** TODO: what was selected on validation, and what was touched
  exactly once at the end.
- **Compute.** TODO: hardware, wall-clock per run, total GPU-hours.
- **Run order.** TODO: milestone order from the plan, with the cheapest
  falsifying run first.

## 6. Ablations

| Ablation | Variants | Hypothesis tested |
|---|---|---|
| TODO | TODO | TODO |

## 7. Success criteria

| Milestone | Numeric threshold | Decision if missed |
|---|---|---|
| TODO | TODO | TODO |

## 8. Reporting rules

- Report mean and standard deviation across seeds; never a single best run.
- Report the primary metric for every arm, including arms that lost.
- Report negative and inconclusive results; do not drop arms.
- No metric appears in the paper that does not appear in `metrics.csv`.
