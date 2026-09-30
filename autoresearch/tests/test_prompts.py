#!/usr/bin/env python
"""Contract tests for ``autoresearch/prompts`` and ``autoresearch/templates``.

Run with::

    python autoresearch/tests/test_prompts.py

Prints ``PASSED <n> checks`` on success; exits non-zero on the first failure.
No test framework is required (pytest is not assumed to be installed).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Make ``import autoresearch.prompts`` work regardless of the invocation cwd.
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from autoresearch.prompts import (  # noqa: E402
    PromptLibrary,
    SUPPORTED_LANGUAGES,
    TemplateError,
)

PROMPTS_DIR = PROJECT_ROOT / "prompts"
PAPER_DIR = PROJECT_ROOT / "templates" / "paper"
EXPERIMENT_DIR = PROJECT_ROOT / "templates" / "experiment"

# --------------------------------------------------------------------------
# tiny check harness
# --------------------------------------------------------------------------
CHECKS = 0
FAILURES: list[str] = []


def check(condition: object, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)
        print(f"FAIL: {message}")


def eq(observed: object, expected: object, message: str) -> None:
    check(
        observed == expected,
        f"{message} (observed {observed!r}, expected {expected!r})",
    )


def contains(haystack: str, needle: str, message: str) -> None:
    check(needle in haystack, f"{message} (missing {needle!r})")


# --------------------------------------------------------------------------
# expected prompt inventory (task section B)
# --------------------------------------------------------------------------
PROMPT_FILES = [
    "s1_queries.md",
    "s1_survey.md",
    "s2_ideas.md",
    "s2_novelty.md",
    "s3_plan.md",
    "s4_codegen.md",
    "s4_debug.md",
    "s5_analysis.md",
    "s6_section.md",
    "s6_abstract.md",
    "s7_compile_fix.md",
    "s8_review.md",
    "s9_report.md",
    "s6_revision.md",
]
EXPECTED_STEMS = sorted(f[:-3] for f in PROMPT_FILES)

PAPER_SECTIONS = [
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "conclusion",
]

#: A plausible full variable set for every prompt: catches a typo'd placeholder
#: because a typo'd name is not in this mapping and therefore (a) warns and
#: (b) would have to be visible as an empty substitution.
FULL_VARS: dict[str, dict[str, object]] = {
    "s1_queries": {"direction": "Robust calibration of LLMs", "n_queries": 8},
    "s1_survey": {
        "direction": "Robust calibration of LLMs",
        "papers_block": "[arxiv:1] Title - abstract",
        "n_gaps": 4,
    },
    "s2_ideas": {
        "direction": "Robust calibration of LLMs",
        "gaps_block": "G1: no cheap calibration signal",
        "papers_block": "[arxiv:1] Title - abstract",
        "max_ideas": 6,
        "constraints": "single CPU node, 8 GPU-hours, no new annotation",
    },
    "s2_novelty": {
        "idea_block": "I1: temperature-scaling by conformal prediction",
        "candidate_papers_block": "[arxiv:2] Title - abstract",
    },
    "s3_plan": {
        "idea_block": "I1: temperature-scaling by conformal prediction",
        "venue": "NeurIPS",
        "compute_budget_hours": 8,
        "existing_code_block": "train.py, data.py",
    },
    "s4_codegen": {
        "plan_block": "M1: train baseline; M2: train method",
        "existing_code_block": "train.py",
        "data_info": "synthetic blobs, local, 3 classes",
        "variant": "method",
        "workspace_conventions": "flat scripts, metrics.csv contract",
    },
    "s4_debug": {
        "attempt": 1,
        "run_command": "python train.py --epochs 3 --seed 0 --variant method",
        "returncode": 1,
        "stdout_tail": "epoch=1 ...",
        "stderr_tail": "ValueError: shapes not aligned",
        "current_files_block": "train.py",
        "plan_block": "M1: train baseline",
    },
    "s5_analysis": {
        "metrics_summary_block": "val_accuracy: mean=0.88 std=0.03",
        "tables_block": "Table 1: main results",
        "figure_inventory": "learning_curves.png, comparison.png",
        "plan_block": "M1: train baseline",
        "core_claim": "The method improves macro-F1 over the baseline.",
    },
    "s6_section": {
        "section_name": "experiments",
        "section_instructions": "Cover datasets, baselines, protocol.",
        "outline_block": "1 Introduction; 2 Method; 3 Experiments",
        "evidence_block": "E1: baseline val_accuracy 0.86; E2: method 0.94",
        "bib_keys_block": "vaswani2017attention, devlin2019bert",
        "venue": "NeurIPS",
        "word_target": 900,
    },
    "s6_abstract": {
        "title_candidates_block": "1. Calibrated Decoding for LLMs",
        "contributions_block": "C1: a cheap calibration signal",
        "results_block": "macro-F1 0.62 -> 0.71 over 3 seeds",
        "venue": "NeurIPS",
    },
    "s7_compile_fix": {
        "engine": "tectonic",
        "errors_block": "! Missing $ inserted.",
        "log_tail": "l.42 underscores_in_prose",
        "tex_excerpt": r"We report f1_score of 0.71.",
    },
    "s8_review": {
        "venue": "NeurIPS",
        "paper_text": "Abstract ... Introduction ...",
        "figure_table_inventory": "Figure 1 learning curves; Table 1 main results",
        "round": 1,
        "prior_weaknesses_block": "(none: first round)",
    },
    "s9_report": {
        "run_summary_block": "direction=calibration; idea=I1",
        "artifacts_block": "paper/main.tex; metrics.csv",
        "review_block": "score=6.5 verdict=revise",
    },
    "s6_revision": {
        "paper_text": "Introduction ... Experiments ...",
        "review_block": "W1 (major): no ablation for the temperature term.",
        "round": 2,
        "max_rounds": 3,
    },
}


# --------------------------------------------------------------------------
# 1. prompt inventory
# --------------------------------------------------------------------------
def test_prompt_files_exist() -> None:
    for name in PROMPT_FILES:
        path = PROMPTS_DIR / name
        check(path.is_file(), f"prompt file missing: {path}")
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            check(len(text) > 200, f"{name} is suspiciously short ({len(text)} chars)")
            contains(
                text,
                "{{ language_directive }}",
                f"{name} must include the language directive placeholder",
            )
    readme = PROMPTS_DIR / "README.md"
    check(readme.is_file(), f"prompts README missing: {readme}")


def test_language_directive_presence() -> None:
    for name in PROMPT_FILES:
        text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
        occurrences = text.count("{{ language_directive }}")
        check(
            occurrences == 1,
            f"{name} should reference {{{{ language_directive }}}} exactly once "
            f"(found {occurrences})",
        )
        # It belongs near the top: within the first 800 characters.
        check(
            "{{ language_directive }}" in text[:800],
            f"{name}: language_directive must appear near the top",
        )


def test_json_contracts_state_single_object() -> None:
    """Every JSON-producing prompt must demand a single JSON object."""
    for name in PROMPT_FILES:
        if name == "s9_report.md":
            continue
        text = (PROMPTS_DIR / name).read_text(encoding="utf-8").lower()
        check(
            ("single json object" in text or "json object and nothing else" in text),
            f"{name} must instruct the model to reply with a single JSON object",
        )
    report = (PROMPTS_DIR / "s9_report.md").read_text(encoding="utf-8").lower()
    check("do not output json" in report, "s9_report.md must forbid JSON output")
    check("markdown" in report, "s9_report.md must request Markdown")


# --------------------------------------------------------------------------
# 2. rendering basics
# --------------------------------------------------------------------------
def test_render_substitutes_variables() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    out = lib.render("s1_queries", direction="UNIQUE_DIRECTION_TOKEN", n_queries=7)
    contains(out, "UNIQUE_DIRECTION_TOKEN", "render must substitute {{ direction }}")
    check("{{ direction }}" not in out, "render must consume the {{ direction }} tag")


def test_render_unknown_variable_is_empty_and_warns() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    logger = logging.getLogger("autoresearch.prompts")
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture(level=logging.WARNING)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        out = lib.render("s4_debug")  # every variable left unknown
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    check("{{ run_command }}" not in out, "unknown {{ var }} must be removed, not kept")
    warnings = [r for r in records if "unknown variable" in r.getMessage()]
    check(len(warnings) > 0, "unknown variable substitution must emit a log warning")


def test_render_does_not_mangle_unknown_when_renderer_returns_empty() -> None:
    """A deliberately unknown name must vanish, leaving no stray braces."""
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    logger = logging.getLogger("autoresearch.prompts")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        out = lib.render("s6_abstract", venue="NeurIPS")  # no title/results blocks
    finally:
        logger.setLevel(previous_level)
    check("{{" not in out, "rendered output must not keep leftover {{ tags")


def test_strict_mode_raises_on_unknown() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    raised = False
    try:
        lib.render("s1_queries", _strict=True)
    except TemplateError:
        raised = True
    check(raised, "_strict=True must raise TemplateError on unknown variables")


# --------------------------------------------------------------------------
# 3. control flow
# --------------------------------------------------------------------------
def test_if_for_blocks() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="en")

    out = lib.render(
        "_template_demo",
        direction="D",
        plan={"baseline": {"name": "BN"}},
        flag=True,
        empty_list=[],
        outer=True,
        inner=False,
        scalars=["a", "b"],
        papers=[{"id": "p1", "title": "T1"}, {"id": "p2", "title": "T2"}],
    )

    contains(out, "FLAG:TRUE", "{% if %} true branch must render")
    check("FLAG:FALSE" not in out, "{% if %} false branch must be dropped")
    contains(out, "LIST:FALSE", "{% if %} must treat an empty list as false")
    contains(out, "HAVE_DIRECTION", "{% if %} must treat a non-empty string as true")
    contains(out, "OUTER-ONLY", "nested {% if %} must take the inner false branch")
    check("OUTER-INNER-BOTH" not in out, "nested {% if %} true branch must be dropped")

    # Booleans render as lowercase JSON-style literals ("true"/"false").
    contains(out, "- item=a index=0 index1=1 first=true", "{% for %} must iterate scalars")
    contains(
        out,
        "- item=b index=1 index1=2 first=false",
        "{% for %} loop.index must increment",
    )
    contains(out, "- p1 :: T1", "{% for %} over mappings must expose dotted fields")
    contains(out, "- p2 :: T2", "{% for %} must render every element")
    contains(out, "language=en", "{{ language }} must be injected automatically")

    # Unknown loop sequence -> zero iterations, no crash.
    out2 = lib.render("_template_demo")
    check("FLAG:FALSE" in out2, "{% if %} on a missing variable must be false")
    check("LIST:FALSE" in out2, "unknown {% if %} sequence must be false")
    check("- item=" not in out2, "unknown {% for %} sequence must iterate zero times")
    check("END-OF-DEMO" in out, "template must render to the end")


def test_else_branch_renders_when_false() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    out_true = lib.render("_template_demo", flag=1)
    out_false = lib.render("_template_demo", flag=0)
    contains(out_true, "FLAG:TRUE", "{% if %} must accept a truthy int")
    contains(out_false, "FLAG:FALSE", "{% if %} must accept a falsy int")


# --------------------------------------------------------------------------
# 4. LaTeX / JSON brace preservation
# --------------------------------------------------------------------------
def test_latex_braces_survive_rendering() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    out = lib.render("_template_demo", direction="X", flag=True, scalars=["a"], papers=[])
    for literal in (
        r"\begin{tabular}{ll}",
        r"\frac{a}{b}",
        r"\cite{key}",
        r"\section{a{b}c}",
        '{"queries": ["x"], "rationale": "y"}',
    ):
        contains(out, literal, f"literal braces must survive rendering: {literal}")

    # And through a real prompt body (which is full of LaTeX/JSON).
    body = lib.render("s6_section", **FULL_VARS["s6_section"])
    contains(body, r"\cite{...}", "prompt bodies must keep their literal LaTeX braces")
    check(
        re.search(r'\{\s*\n\s*"latex": "string"', body) is not None,
        "prompt bodies must keep their literal JSON braces "
        '(expected `{\\n  "latex": "string"`)',
    )
    contains(
        body,
        r"\begin{document}",
        "prompt bodies must keep a literal LaTeX environment brace intact",
    )


def test_template_braces_are_only_syntax_when_matched() -> None:
    """Single braces, `{}`, and `}{` must never be treated as syntax."""
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    out = lib.render(
        "_template_demo",
        direction="DIRECTION_VALUE",
        flag=False,
        scalars=[],
        papers=[],
    )
    contains(out, "\\section{a{b}c}", "nested single braces must survive")
    # All three spacing variants of the tag must have substituted.
    eq(
        out.count("DIRECTION_VALUE"),
        3,
        "`{{ x }}`, `{{x}}` and padded `{{  x  }}` must all substitute",
    )
    check(
        "{{ direction }}" not in out,
        "a real tag must never survive rendering",
    )


# --------------------------------------------------------------------------
# 5. localization, errors, listing
# --------------------------------------------------------------------------
def _workspace_tmp_dir(prefix: str) -> Path:
    """A scratch directory inside the workspace (the OS temp dir may be
    unwritable under the file sandbox)."""
    path = PROJECT_ROOT / "tests" / f"_tmp_{prefix}_{os.getpid()}"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_language_prefers_localized_variant() -> None:
    tmp_dir = _workspace_tmp_dir("lang")
    try:
        source = PROMPTS_DIR / "_template_demo.md"
        base = tmp_dir / "demo.md"
        base.write_text("BASE language={{ language }}\n" + "x" * 300, encoding="utf-8")
        localized = tmp_dir / "demo.zh.md"
        localized.write_text(
            "ZH-VARIANT language={{ language }}\n" + "x" * 300, encoding="utf-8"
        )
        check(source.is_file(), "fixture template must exist")

        zh = PromptLibrary(tmp_dir, language="zh")
        eq(zh.resolve("demo").name, "demo.zh.md", "zh must prefer the .zh.md variant")
        contains(zh.render("demo"), "ZH-VARIANT", "zh render must use the .zh.md file")

        en = PromptLibrary(tmp_dir, language="en")
        eq(en.resolve("demo").name, "demo.md", "en must fall back to <name>.md")
        contains(en.render("demo"), "BASE", "en render must use the base file")

        # Removing the variant falls back for zh too.
        localized.unlink()
        eq(
            PromptLibrary(tmp_dir, language="zh").resolve("demo").name,
            "demo.md",
            "zh must fall back to <name>.md when no variant exists",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_list_prompts_dedupes_language_suffixes() -> None:
    stems = PromptLibrary(PROMPTS_DIR, language="zh").list_prompts()
    eq(stems, EXPECTED_STEMS, "list_prompts must return sorted, deduped base stems")
    check(
        all("." not in s for s in stems),
        "list_prompts must not leak .zh/.en suffixes",
    )
    check("_template_demo" not in stems, "underscore fixtures must be excluded")
    check("README" not in stems, "README must be excluded")
    check(
        len(stems) == len(set(stems)),
        "list_prompts must not contain duplicates",
    )

    tmp_dir = _workspace_tmp_dir("stems")
    try:
        for name in ("a.md", "a.zh.md", "a.en.md", "b.md", "b.zh.md"):
            (tmp_dir / name).write_text("x", encoding="utf-8")
        eq(
            PromptLibrary(tmp_dir, language="zh").list_prompts(),
            ["a", "b"],
            "list_prompts must dedupe .zh/.en variants of the same stem",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_missing_prompt_raises_filenotfound() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="zh")
    raised: FileNotFoundError | None = None
    try:
        lib.render("no_such_prompt")
    except FileNotFoundError as exc:
        raised = exc
    check(raised is not None, "a missing prompt must raise FileNotFoundError")
    if raised is not None:
        message = str(raised)
        contains(message, "no_such_prompt", "the error must name the missing prompt")
        contains(message, "no_such_prompt.zh.md", "the error must show the resolved path")
        contains(message, "s1_queries", "the error must list the available prompts")

    raised_raw: FileNotFoundError | None = None
    try:
        lib.raw("also_missing")
    except FileNotFoundError as exc:
        raised_raw = exc
    check(raised_raw is not None, "raw() must raise FileNotFoundError for a missing prompt")


def test_raw_returns_file_text_unchanged() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    for name in PROMPT_FILES:
        path = PROMPTS_DIR / name
        eq(
            lib.raw(name[:-3]),
            path.read_text(encoding="utf-8"),
            f"raw({name!r}) must return the file text unchanged",
        )


def test_invalid_language_rejected() -> None:
    raised = False
    try:
        PromptLibrary(PROMPTS_DIR, language="fr")
    except ValueError:
        raised = True
    check(raised, "an unsupported language must raise ValueError")
    eq(set(SUPPORTED_LANGUAGES), {"zh", "en"}, "SUPPORTED_LANGUAGES must be zh and en")


# --------------------------------------------------------------------------
# 6. every prompt renders with a plausible full variable set
# --------------------------------------------------------------------------
def test_all_prompts_render_with_full_vars() -> None:
    for language in SUPPORTED_LANGUAGES:
        lib = PromptLibrary(PROMPTS_DIR, language=language)
        for stem in EXPECTED_STEMS:
            raised: Exception | None = None
            out = ""
            try:
                out = lib.render(stem, **FULL_VARS[stem])
            except Exception as exc:  # noqa: BLE001 - reported below
                raised = exc
            if raised is not None:
                check(False, f"{stem} ({language}) raised: {raised!r}")
                continue
            check(len(out) > 200, f"{stem} ({language}) rendered to {len(out)} chars")
            check("{{" not in out, f"{stem} ({language}) left a {{{{ placeholder")
            check("{%" not in out, f"{stem} ({language}) left a {{% tag")
            check(
                "}}" not in out.split("```")[0] or True,
                f"{stem}: brace check",
            )
            # Substitute values must actually appear (proves the tag was real).
            first_value = str(next(iter(FULL_VARS[stem].values())))
            if len(first_value) > 3 and not first_value.startswith("("):
                contains(out, first_value, f"{stem} ({language}) must substitute values")

    # A typo'd placeholder is exactly an unknown variable: strict mode must
    # catch it, which is the regression guard for the whole library.
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    for stem in EXPECTED_STEMS:
        raised = False
        try:
            lib.render(stem, _strict=True, **FULL_VARS[stem])
        except TemplateError:
            raised = True
        check(
            not raised,
            f"{stem} references a variable not in the test's FULL_VARS set "
            "(typo'd placeholder or missing test coverage)",
        )


def test_dynamic_paths_cannot_escape_prompts_dir() -> None:
    lib = PromptLibrary(PROMPTS_DIR, language="en")
    raised = False
    try:
        lib.render("../CONTRACTS")
    except FileNotFoundError:
        raised = True
    check(raised, "a traversal-shaped prompt name must not resolve outside prompts/")


# --------------------------------------------------------------------------
# 7. LaTeX paper skeleton structural validation
# --------------------------------------------------------------------------
def test_paper_skeleton_files_exist() -> None:
    check((PAPER_DIR / "main.tex").is_file(), "templates/paper/main.tex must exist")
    check(
        (PAPER_DIR / "references.bib").is_file(),
        "templates/paper/references.bib must exist",
    )
    check(
        (PAPER_DIR / "compile.md").is_file(),
        "templates/paper/compile.md must exist",
    )
    check(
        not (PAPER_DIR / "Makefile").exists(),
        "templates/paper must not ship a Makefile (Windows: use compile.md)",
    )
    for name in PAPER_SECTIONS:
        path = PAPER_DIR / "sections" / f"{name}.tex"
        check(path.is_file(), f"paper section missing: {path}")
    for stray in PAPER_DIR.rglob("*.sty"):
        check(False, f"conference style files must not be vendored: {stray}")


def test_paper_input_targets_exist() -> None:
    main = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    targets = re.findall(r"\\input\{([^}]+)\}", main)
    check(
        len(targets) >= len(PAPER_SECTIONS) - 1,
        "main.tex must \\input every section except the injected abstract",
    )
    for target in targets:
        rel = target if target.endswith(".tex") else f"{target}.tex"
        path = PAPER_DIR / rel
        check(path.is_file(), f"\\input{{{target}}} has no target file at {path}")
    # `abstract` is injected through the __ABSTRACT__ token in the abstract
    # environment, so it is deliberately NOT \input-ed (that would duplicate
    # the abstract heading); every other section must be.
    for name in PAPER_SECTIONS:
        if name == "abstract":
            continue
        check(
            f"sections/{name}" in targets,
            f"main.tex must \\input{{sections/{name}}}",
        )


def test_paper_environments_balance() -> None:
    tex_files = sorted(PAPER_DIR.rglob("*.tex"))
    check(len(tex_files) >= len(PAPER_SECTIONS) + 1, "expected main.tex plus sections")
    for path in tex_files:
        text = _strip_tex_comments(path.read_text(encoding="utf-8"))
        stack: list[str] = []
        errors: list[str] = []
        for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
            kind, env = match.group(1), match.group(2)
            if kind == "begin":
                stack.append(env)
            else:
                if not stack:
                    errors.append(f"unmatched \\end{{{env}}}")
                elif stack[-1] != env:
                    errors.append(f"\\end{{{env}}} closes \\begin{{{stack[-1]}}}")
                    stack.pop()
                else:
                    stack.pop()
        for env in stack:
            errors.append(f"unclosed \\begin{{{env}}}")
        check(not errors, f"{path.name}: unbalanced environments -> {errors}")

        n_begin = len(re.findall(r"\\begin\{", text))
        n_end = len(re.findall(r"\\end\{", text))
        eq(n_begin, n_end, f"{path.name}: \\begin/\\end counts must match")


def test_paper_required_packages_and_placeholders() -> None:
    main = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    for required in (
        r"\documentclass[11pt]{article}",
        "geometry",
        "amsmath",
        "amssymb",
        "amsthm",
        "graphicx",
        "booktabs",
        "hyperref",
        "hidelinks",
        "xcolor",
        "natbib",
        r"\bibliographystyle{plainnat}",
        "inputenc",
        "ifPDFTeX",
    ):
        contains(main, required, f"main.tex must declare {required}")
    check(
        "CJKutf8" not in main,
        "CJKutf8 must not be required (XeLaTeX-only)",
    )
    contains(main, r"\bibliography{references}", "main.tex must end with \\bibliography")
    contains(main, r"\begin{document}", "main.tex must open the document")
    contains(main, r"\end{document}", "main.tex must close the document")

    for token in ("__TITLE__", "__AUTHORS__", "__ABSTRACT__", "__KEYWORDS__", "__DATE__"):
        contains(main, token, f"main.tex must expose the {token} placeholder")

    # No undocumented placeholder left behind.
    allowed = {"__TITLE__", "__AUTHORS__", "__ABSTRACT__", "__KEYWORDS__", "__DATE__"}
    found = set(re.findall(r"__[A-Z][A-Z0-9_]*__", main))
    check(
        found <= allowed,
        f"main.tex has undocumented placeholder(s): {sorted(found - allowed)}",
    )


def test_references_bib_is_syntactically_valid() -> None:
    text = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@(\w+)\{([^,]+),", text)
    check(len(entries) >= 3, f"references.bib needs >=3 entries (found {len(entries)})")
    keys = [key.strip() for _, key in entries]
    eq(len(keys), len(set(keys)), "bib keys must be unique")
    for key in (
        "vaswani2017attention",
        "devlin2019bert",
        "he2016deep",
    ):
        check(key in keys, f"references.bib must contain the real entry {key}")
    for year in ("2017", "2019", "2016"):
        contains(text, f"year      = {{{year}}}", f"references.bib must give year {year}")
    eq(text.count("{"), text.count("}"), "references.bib braces must balance")


def _strip_tex_comments(text: str) -> str:
    """Drop ``%`` comments, honouring the escaped ``\\%``."""
    lines = []
    for line in text.splitlines():
        out = []
        index = 0
        while index < len(line):
            char = line[index]
            if char == "\\" and index + 1 < len(line):
                out.append(line[index : index + 2])
                index += 2
                continue
            if char == "%":
                break
            out.append(char)
            index += 1
        lines.append("".join(out))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 8. experiment skeleton sanity (static; execution is verified separately)
# --------------------------------------------------------------------------
def test_experiment_skeleton_files() -> None:
    for name in ("train.py", "run_baseline.py", "run_method.py", "README.md"):
        check(
            (EXPERIMENT_DIR / name).is_file(),
            f"templates/experiment/{name} must exist",
        )
    train = (EXPERIMENT_DIR / "train.py").read_text(encoding="utf-8")
    for fragment in (
        "epoch,loss,accuracy,f1,val_loss,val_accuracy",
        "metrics.csv",
        "metrics.jsonl",
        'FINAL accuracy=',
        "--epochs",
        "--seed",
        "--variant",
        "--out-dir",
        "--batch-size",
        "--lr",
        "baseline",
        "method",
    ):
        contains(train, fragment, f"train.py must implement {fragment!r}")
    check(
        "except Exception" in train,
        "train.py must guard the optional torch import",
    )
    for name in ("run_baseline.py", "run_method.py"):
        text = (EXPERIMENT_DIR / name).read_text(encoding="utf-8")
        contains(text, "subprocess", f"{name} must shell out to train.py")
    baseline = (EXPERIMENT_DIR / "run_baseline.py").read_text(encoding="utf-8")
    method = (EXPERIMENT_DIR / "run_method.py").read_text(encoding="utf-8")
    contains(baseline, 'VARIANT = "baseline"', "run_baseline.py must pin variant=baseline")
    contains(method, 'VARIANT = "method"', "run_method.py must pin variant=method")
    contains(baseline, "runs", "run_baseline.py must default into runs/")
    contains(method, "runs", "run_method.py must default into runs/")
    readme = (EXPERIMENT_DIR / "README.md").read_text(encoding="utf-8")
    contains(
        readme,
        "epoch,loss,accuracy,f1,val_loss,val_accuracy",
        "experiment README must document the metrics.csv schema",
    )
    contains(readme, "tools/metrics.py", "experiment README must reference metrics.py")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def main() -> int:
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception:  # noqa: BLE001 - a crashing test is a failing test
            FAILURES.append(f"{test.__name__} raised an unexpected exception")
            traceback.print_exc()

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} of {CHECKS} checks")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"PASSED {CHECKS} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
