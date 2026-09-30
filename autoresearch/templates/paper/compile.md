# Compiling the paper skeleton (Windows, no Makefile)

`tectonic`, `pdflatex`, and `xelatex` are **not** assumed to be on `PATH`. The
pipeline resolves an engine at run time via
`tools/latex.py::LatexCompiler.detect()` and, when configured, downloads a
single-file `tectonic` binary into `autoresearch/vendor/tectonic/`
(`CompileConfig.auto_install_tectonic`). Nothing in this skeleton requires a
vendored conference style file (`neurips.sty` or similar).

## Commands the pipeline runs

Engine `tectonic` (preferred; resolves the bibliography itself, so a single
invocation is enough):

```powershell
tectonic --keep-logs --outdir . main.tex
```

Engine `pdflatex` (needs the bibliography pass; `LatexCompiler.compile` runs the
engine `runs` times, default 2):

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

Engine `xelatex` (same four-step shape):

```powershell
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

Manual equivalent from PowerShell inside this directory:

```powershell
Push-Location "D:\user\Documents\deepseekv4flash harness\autoresearch\templates\paper"
tectonic --keep-logs --outdir . main.tex
Pop-Location
```

## Environment detection

```powershell
Get-Command tectonic, pdflatex, xelatex -ErrorAction SilentlyContinue |
    Select-Object Name, Source
```

If the command returns nothing, no engine is available: the pipeline is
expected to emit `paper/main.tex` plus `report/COMPILE_BLOCKED.md` and exit
normally (CONTRACTS §14), not to fail the run.

## Placeholder substitution before compiling

`main.tex` contains literal tokens that the pipeline replaces with plain string
substitution (**not** a TeX-aware templating pass). Replacement text must
already be LaTeX-safe:

| Token | Replaced with | Notes |
|---|---|---|
| `__TITLE__` | paper title | do not wrap in `\texttt{}` |
| `__AUTHORS__` | author line | use `\\` for line breaks inside the value |
| `__ABSTRACT__` | abstract body | separate paragraphs with a blank line |
| `__KEYWORDS__` | comma-separated keywords | inserted after `\textbf{Keywords:}` |
| `__DATE__` | date string | any text TeX understands |

Every token is also documented in a header comment at the top of `main.tex`.

## What is checked without an engine

With no engine installed, `autoresearch/tests/test_prompts.py` performs a
structural validation instead:

1. every `\input{...}` target exists on disk;
2. every `\begin{env}` in every `.tex` file has a matching `\end{env}`, with
   empty bodies allowed and nesting verified;
3. the required placeholder tokens are present in `main.tex` and no other
   `__TOKEN__`-shaped placeholder is left unsubstituted in a required slot;
4. `references.bib` parses as BibTeX entries with balanced braces.

This structural check is **not** a compilation test; a real compile is only
attempted when `LatexCompiler.detect()` returns an engine name.
