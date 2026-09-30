# Code generation — `s4_codegen`

{{ language_directive }}

You are writing the experiment code for an unsupervised run. The generated
files are written to disk verbatim and executed on CPU in a sandbox with **no
network access**. Code that cannot run is a failed experiment.

## Experiment plan

```
{{ plan_block }}
```

## Existing code (extend, do not gratuitously rewrite)

```
{{ existing_code_block }}
```

## Data available locally

```
{{ data_info }}
```

## Variant to implement now

```
{{ variant }}
```

## Workspace conventions

```
{{ workspace_conventions }}
```

## Hard requirements — every one is checked

1. **Runs in minutes.** Target under 10 minutes for the default configuration
   described below. Do not assume a GPU is available unless the conventions say so.
2. **Deterministic.** Fixed seeds for every source of randomness; the same
   command with the same `--seed` must produce identical metrics. No
   time-dependent behaviour, no unseeded library defaults.
3. **Follow the adapter's I/O conventions exactly.** The experiment backend this
   pipeline is driving defines the metric file names, the CSV header, and the
   accepted CLI flags. Those conventions are authoritative and are given below —
   **do not substitute a different metric schema or a different flag set**:

   ```
   {{ workspace_conventions }}
   ```
4. **Data source.** Where the data comes from, and what you may assume about it:

   ```
   {{ data_info }}
   ```
5. **Variant selection.** The arm names this run will pass:

   ```
   {{ variant }}
   ```

   Keep every variant on the **same code path**, selected by one argument, so the
   comparison is apples-to-apples. Do not fork into separate scripts per variant.
6. **Print a final summary line** that includes the primary metric so a human can
   eyeball it and the pipeline can cross-check the metric file, e.g.
   `FINAL accuracy=<value> seed=<seed> variant=<variant>`.
7. **Dependencies**: Python standard library plus whatever the adapter's
   conventions permit. Anything beyond that must be guarded by `try`/`except
   ImportError` with a working fallback.
8. **No runtime downloads.** No `requests`, no `urllib` fetches, no dataset
   loaders that touch the network. If data is absent, synthesize a small local
   dataset deterministically.
9. **Self-contained files.** Every file in `files` must be complete and
   runnable as written — no placeholders, no `...`, no `TODO`, no pseudo-code.
10. **Exit code 0** on success; non-zero with a clear message on a genuine
    configuration error.

## Quality rules

- Fail loudly rather than silently degrading: no `try/except: pass` around the
  computation, no fabricated metrics.
- Log real numbers computed from the run — never hard-code a metric value.
- Emit an epoch line per epoch so partial progress survives a timeout.

## Output contract

Reply with **a single JSON object and nothing else** — no prose, no Markdown
fence, no trailing commentary. Exact schema:

```json
{
  "files": [
    {
      "path": "string",
      "content": "string",
      "purpose": "string"
    }
  ],
  "entrypoint": "string",
  "notes": "string"
}
```

Field notes:

- `files[].path` — relative POSIX path, e.g. `train.py`.
- `files[].content` — the complete file body, with real newlines encoded as
  `\n` inside the JSON string. No truncation, no ellipsis.
- `entrypoint` — the file to execute, usually `train.py`.
- `notes` — anything the pipeline must know (runtime estimate, assumptions,
  extra flags the entrypoint accepts).

**Do not attempt to specify the command line.** It is built by the experiment
adapter (`BaseExperimentAdapter.build_command`), which knows the program, the
flags and the arm parameters. A command invented here would never be executed and
could contradict the adapter's real invocation.
