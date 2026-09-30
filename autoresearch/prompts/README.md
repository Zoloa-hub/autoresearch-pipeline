# `autoresearch/prompts/` — prompt library

Templates for the Auto-Research pipeline. Loaded by
[`PromptLibrary`](__init__.py) (CONTRACTS §10).

```python
from autoresearch.prompts import PromptLibrary

lib = PromptLibrary(Path("autoresearch/prompts"), language="zh")
text = lib.render("s3_plan", idea_block=idea_md, venue="NeurIPS",
                  compute_budget_hours=8, existing_code_block="")
```

## Template syntax

| Construct | Meaning |
|---|---|
| `{{ var }}` / `{{var}}` | Substitute `var`. Whitespace inside the braces is ignored. Dotted paths work: `{{ plan.baseline.name }}`. |
| `{% if var %} … {% endif %}` | Include the block when `var` is non-empty / non-zero / truthy. |
| `{% if var %} … {% else %} … {% endif %}` | Two-branch form. |
| `{% for x in xs %} … {% endfor %}` | Repeat for each element. `{{ x }}` binds the element; for a list of mappings, `{{ x.title }}` also resolves, as does `{% for p in papers %}{{ p.title }}{% endfor %}`. |
| `{{ loop.index }}` / `{{ loop.index1 }}` / `{{ loop.first }}` | Available inside a `for` body. |

**Only** `{{ … }}` and `{% … %}` are syntax. Every other brace is passed through
byte for byte, so LaTeX (`\begin{tabular}{ll}`, `\frac{a}{b}`, `\cite{k}`) and
JSON examples inside the prompt bodies render unchanged. Jinja2 is intentionally
not used.

Two extra variables are injected automatically into every render:

- `{{ language }}` — the configured language code (`zh` or `en`).
- `{{ language_directive }}` — the localized output-language instruction, also
  available directly as `PromptLibrary.system_prompt()`.

Unknown variables are replaced with the empty string and logged through the
`autoresearch.prompts` logger (a typo therefore shows up in the run log, not as
a crash).

## Language variants

`language="zh"` resolves `<name>.zh.md` first, then falls back to `<name>.md`.
`list_prompts()` returns deduplicated base stems, so shipping both
`s3_plan.md` and `s3_plan.zh.md` yields a single entry `s3_plan`.

`README.md` and `_`-prefixed files are excluded from `list_prompts()`: they are
documentation and fixtures, not stage prompts. The one fixture here,
[`_template_demo.md`](_template_demo.md), is not model-facing — it exists so
`tests/test_prompts.py` can pin the engine's behaviour (`{% if %}` /
`{% for %}` / brace preservation) against a committed file. There are no
`.zh.md` variants shipped yet; the resolution order is covered by tests that
create one in a scratch directory.

## Booleans and value formatting

`{{ var }}` renders `True`/`False` as the lowercase JSON literals `true`/
`false`, `None` as the empty string, floats without a trailing `.0` when
integral, and lists/mappings as comma-joined text / pretty JSON respectively.
None of this affects literal braces in the surrounding prompt body.

## Prompts and their placeholders

| Prompt | Stage | Placeholders |
|---|---|---|
| [`s1_queries.md`](s1_queries.md) | s1_literature | `direction`, `n_queries` |
| [`s1_survey.md`](s1_survey.md) | s1_literature | `direction`, `papers_block`, `n_gaps` |
| [`s2_ideas.md`](s2_ideas.md) | s2_ideation | `direction`, `gaps_block`, `papers_block`, `max_ideas`, `constraints` |
| [`s2_novelty.md`](s2_novelty.md) | s2_ideation | `idea_block`, `candidate_papers_block` |
| [`s3_plan.md`](s3_plan.md) | s3_planning | `idea_block`, `venue`, `compute_budget_hours`, `existing_code_block` |
| [`s4_codegen.md`](s4_codegen.md) | s4_experiment | `plan_block`, `existing_code_block`, `data_info`, `variant`, `workspace_conventions` |
| [`s4_debug.md`](s4_debug.md) | s4_experiment | `attempt`, `run_command`, `returncode`, `stdout_tail`, `stderr_tail`, `current_files_block`, `plan_block` |
| [`s5_analysis.md`](s5_analysis.md) | s5_analysis | `metrics_summary_block`, `tables_block`, `figure_inventory`, `plan_block`, `core_claim` |
| [`s6_section.md`](s6_section.md) | s6_writing | `section_name`, `section_instructions`, `outline_block`, `evidence_block`, `bib_keys_block`, `venue`, `word_target` |
| [`s6_abstract.md`](s6_abstract.md) | s6_writing | `title_candidates_block`, `contributions_block`, `results_block`, `venue` |
| [`s6_revision.md`](s6_revision.md) | s6_writing | `paper_text`, `review_block`, `round`, `max_rounds` |
| [`s7_compile_fix.md`](s7_compile_fix.md) | s7_compile | `engine`, `errors_block`, `log_tail`, `tex_excerpt` |
| [`s8_review.md`](s8_review.md) | s8_review | `venue`, `paper_text`, `figure_table_inventory`, `round`, `prior_weaknesses_block` |
| [`s9_report.md`](s9_report.md) | s9_finalize | `run_summary_block`, `artifacts_block`, `review_block` |

All block variables (`*_block`, `*_inventory`) are pre-formatted Markdown
strings — the stages are responsible for the formatting, the prompts only
consume them.

## Output contracts

Every prompt except `s9_report.md` demands **a single JSON object and nothing
else**, and spells out the exact schema with the field names of CONTRACTS §9.
`s9_report.md` demands **Markdown directly** (no JSON).

Pipeline-filled `Idea` keys are explicitly excluded from `s2_ideas.md`: the
model returns `id, title, hypothesis, motivation, method_sketch, novelty_claim,
feasibility, risks, expected_metrics, minimal_experiment, required_resources`,
while `novelty` (from `s2_novelty`), `pilot`, `rank`, and `selected` are added
downstream, matching the `Idea` shape in CONTRACTS §9.

> **Documented ambiguity.** The task brief for `s2_ideas.md` stated that
> `novelty`/`pilot`/`rank` are pipeline-filled *and* that its field list matched
> CONTRACTS §9 exactly; §9 additionally lists `minimal_experiment` and
> `required_resources`. This library follows §9 (the frozen contract) and emits
> the two extra keys, which are harmless if a stage ignores them.
