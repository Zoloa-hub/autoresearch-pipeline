# Debugging — `s4_debug`

{{ language_directive }}

You are debugging the experiment harness. Attempt **{{ attempt }}** failed.
Produce a root-cause diagnosis and the **smallest** patch that fixes it.

## Command that was run

```
{{ run_command }}
```

Exit code: `{{ returncode }}`

## stdout (tail)

```
{{ stdout_tail }}
```

## stderr (tail)

```
{{ stderr_tail }}
```

## Current files

```
{{ current_files_block }}
```

## Experiment plan (the intent that must be preserved)

```
{{ plan_block }}
```

## Method

1. Read the traceback bottom-up; identify the **first** frame in our own code.
2. State a **root cause** — the specific wrong assumption, not the symptom. If
   the evidence is insufficient to distinguish two causes, say which one you
   are fixing and why.
3. Emit the **smallest possible patch**: touch only the lines responsible.
   Do not restructure, reformat, rename, or "improve" unrelated code.

## Hard prohibitions

- Do **not** rewrite unrelated parts of any file.
- Do **not** weaken the experiment to make it pass. Specifically forbidden:
  - faking, hard-coding, or interpolating metric values;
  - shrinking the test set, the number of epochs, or the evaluation into a
    no-op;
  - `try`/`except: pass` (or bare `except`) wrapping the real computation;
  - deleting an assertion, a metric, or a comparison instead of fixing it;
  - replacing the baseline/method with a stub.
- If the honest fix is not possible with the available evidence, say so in
  `validity_note` and lower `confidence` rather than emitting a
  validity-destroying patch.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "diagnosis": "string",
  "root_cause": "string",
  "files": [
    {
      "path": "string",
      "content": "string"
    }
  ],
  "commands_to_verify": ["string"],
  "confidence": 0.0,
  "validity_note": "string"
}
```

Field notes:

- `files` — only files you actually changed; `content` is the **complete** new
  file body (not a diff), with newlines encoded as `\n`. Use an empty array if
  no file change is needed (e.g. the fix is a command-line change).
- `commands_to_verify` — 1 or more commands that prove the fix worked.
- `confidence` — float in `0.0–1.0`, your confidence in the root cause.
- `validity_note` — one sentence confirming that the fix does not weaken the
  experiment, and naming anything it does change about the results.
