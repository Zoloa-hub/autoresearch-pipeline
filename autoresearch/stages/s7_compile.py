"""阶段⑦：LaTeX 自动化编译与语法修复。

三个设计要点：

1. **引擎无关**：优先 ``tectonic``（单文件、自动拉宏包、无需完整 TeX 发行版），
   退到 ``pdflatex``/``xelatex``，都没有时**不假装成功**——写一份
   ``COMPILE_BLOCKED.md`` 说明缺什么、怎么装，然后管线照常收尾。
   一份写清阻断原因的交付物，比一个崩溃的管线有用得多。
2. **修复有边界**：编译错误由 LLM 出最小补丁，但补丁只允许改 ``paper/`` 下的
   ``.tex``/``.bib``，且每轮只改被报告的错误；超过 ``max_fix_rounds`` 就停下，
   绝不无限重编译烧钱。
3. **日志留证**：每轮编译的日志尾部、错误列表、应用的补丁全部归档，
   评审阶段与最终报告会引用它们。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..graph.state import Artifact
from .base import Stage, StageResult, clamp, clean_text, coerce_list

_FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["file", "old_text", "new_text"],
            },
        },
        "packages_to_add": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["diagnosis"],
}

#: 确定性修复规则：编译错误里最常见且最容易安全自动修的一批。
_DETERMINISTIC_FIXES: tuple[tuple[str, str, str], ...] = (
    (r"\\usepackage\[[^\]]*\]\{inputenc\}", "", "pdfLaTeX 下 UTF-8 已默认，去掉 inputenc 冲突"),
    (r"\\bibliographystyle\{plainnat\}", "\\bibliographystyle{plainnat}", "保持 natbib 风格"),
)

#: 允许被自动修改的扩展名——防止补丁越界改到实验代码或系统文件。
_ALLOWED_SUFFIXES = (".tex", ".bib", ".cls", ".sty")
_MAX_ROUNDS = 3


class CompileStage(Stage):
    name = "s7_compile"
    title = "LaTeX 编译与语法修复"
    requires = ("paper_tex",)
    produces = ("compile_result", "compile_fix_history", "final_pdf")
    max_attempts = 2

    def run(self, state: dict[str, Any]) -> StageResult:
        warnings: list[str] = []
        artifacts: list[Artifact] = []
        missing = self.check_requires(state)
        if missing:
            warnings.append(f"compile degraded: missing {missing}")

        paper_dir = self.ctx.path("paper")
        main_tex = paper_dir / "main.tex"
        if not main_tex.exists():
            msg = "paper/main.tex was never written (writing stage failed)"
            artifacts.append(
                self.ctx.save_text(
                    "report/COMPILE_BLOCKED.md",
                    f"# 编译被阻断\n\n{msg}\n\n"
                    "请检查 s6 阶段的事件日志与 `paper/` 目录内容。\n",
                    stage=self.name,
                )
            )
            return StageResult.failure(msg, artifacts=artifacts)

        engine = self._resolve_engine(warnings)
        fix_history: list[dict[str, Any]] = []
        result: Any = None

        if engine is None:
            blocked = self._write_blocked_report(paper_dir, main_tex, state)
            artifacts.append(blocked)
            warnings.append("no LaTeX engine available; PDF not produced")
            compile_result = {
                "ok": False,
                "pdf": "",
                "engine": "none",
                "errors": ["no LaTeX engine available (tectonic/pdflatex/xelatex)"],
                "warnings": [],
                "log_tail": "",
                "rounds": 0,
            }
            return StageResult.success(
                detail="no LaTeX engine; wrote COMPILE_BLOCKED.md instead of a PDF",
                updates={
                    "compile_result": compile_result,
                    "compile_fix_history": [],
                    "warnings": list(state.get("warnings") or []) + warnings,
                },
                artifacts=artifacts,
            )

        max_rounds = min(_MAX_ROUNDS, max(1, int(getattr(self.ctx.cfg, "max_review_rounds", 3) or 3)))
        for round_no in range(1, max_rounds + 1):
            self._info(f"s7: compiling with {engine} (round {round_no})")
            t0 = time.monotonic()
            try:
                result = self.ctx.compiler.compile(main_tex, runs=2, engine=engine)
            except Exception as exc:
                warnings.append(f"compiler raised: {exc}")
                result = None
            elapsed = time.monotonic() - t0
            self.ctx.log_event(
                "compile_run",
                stage=self.name,
                engine=engine,
                round=round_no,
                ok=bool(result and result.ok),
                duration=round(elapsed, 2),
                errors=len(getattr(result, "errors", []) or []),
            )

            if result is not None and result.ok:
                break

            errors = list(getattr(result, "errors", []) or []) if result is not None else ["compiler raised"]
            log_tail = clamp(str(getattr(result, "log", "") or ""), 6000) if result is not None else ""

            fix = self._propose_fix(state, engine, errors, log_tail, paper_dir)
            applied = self._apply_patches(paper_dir, fix, warnings)
            fix_history.append(
                {
                    "round": round_no,
                    "engine": engine,
                    "errors": errors[:10],
                    "diagnosis": clean_text(str((fix or {}).get("diagnosis") or "")),
                    "packages_to_add": coerce_list((fix or {}).get("packages_to_add")),
                    "patches_applied": applied,
                    "elapsed": round(elapsed, 2),
                }
            )
            if not applied:
                warnings.append(f"round {round_no}: no applicable fix for {len(errors)} error(s)")
                break
            if round_no == max_rounds:
                warnings.append("compile still failing after the maximum number of fix rounds")
                break

        pdf: Path | None = getattr(result, "pdf", None) if result is not None else None
        ok = bool(result is not None and result.ok and pdf and Path(pdf).exists())
        final_pdf = ""
        if ok and pdf is not None:
            final_pdf = self.ctx.rel(Path(pdf))
            try:
                artifacts.append(self.ctx.artifact(Path(pdf), kind="pdf", stage=self.name))
            except Exception as exc:  # pragma: no cover
                warnings.append(f"could not register PDF artifact: {exc}")
            # 同时把日志归档
            try:
                artifacts.append(
                    self.ctx.save_text(
                        "paper/compile.log",
                        clamp(str(getattr(result, "log", "") or ""), 200000),
                        stage=self.name,
                        kind="log",
                    )
                )
            except Exception as exc:
                warnings.append(f"could not archive compile log: {exc}")

        compile_result = {
            "ok": ok,
            "pdf": final_pdf,
            "engine": str(getattr(result, "engine", engine) or engine),
            "errors": list(getattr(result, "errors", []) or [])[:20] if result is not None else [],
            "warnings": list(getattr(result, "warnings", []) or [])[:20] if result is not None else [],
            "log_tail": clamp(str(getattr(result, "log", "") or ""), 4000) if result is not None else "",
            "rounds": len(fix_history) + 1,
        }
        artifacts.append(
            self.ctx.save_json("paper/compile_result.json", compile_result, stage=self.name)
        )
        artifacts.append(
            self.ctx.save_json("paper/compile_fix_history.json", fix_history, stage=self.name)
        )
        if not ok:
            artifacts.append(self._write_blocked_report(paper_dir, main_tex, state, compile_result))

        detail = (
            f"{compile_result['engine']}: {'PDF ok' if ok else 'failed'} "
            f"({len(compile_result['errors'])} errors, {len(fix_history)} fix rounds)"
        )
        updates: dict[str, Any] = {
            "compile_result": compile_result,
            "compile_fix_history": fix_history,
            "warnings": list(state.get("warnings") or []) + warnings,
        }
        if final_pdf:
            updates["final_pdf"] = final_pdf
        return StageResult.success(detail=detail, updates=updates, artifacts=artifacts)

    # ------------------------------------------------------------------ #
    def _resolve_engine(self, warnings: list[str]) -> str | None:
        """确定用哪个引擎；必要时尝试自动安装 tectonic。"""
        cfg_engine = str(getattr(self.ctx.cfg.compile, "engine", "tectonic") or "tectonic").lower()
        if cfg_engine in ("none", "off"):
            warnings.append("LaTeX compilation disabled by config (engine=none)")
            return None

        try:
            detected = self.ctx.compiler.detect()
        except Exception as exc:
            warnings.append(f"engine detection failed: {exc}")
            detected = None
        if detected:
            return detected

        if cfg_engine == "tectonic" and getattr(self.ctx.cfg.compile, "auto_install_tectonic", True):
            self._info("s7: tectonic not found; attempting automatic download")
            try:
                installed = self.ctx.compiler.install_tectonic()
            except Exception as exc:
                warnings.append(f"tectonic auto-install failed: {exc}")
                installed = None
            if installed:
                try:
                    return self.ctx.compiler.detect() or "tectonic"
                except Exception:
                    return "tectonic"
            # 「装上了但用不了」与「装不上」是两种完全不同的处境，必须区分报告：
            # 前者用户什么都不用做（二进制已在 vendor/），只需要放开一次写入权限。
            reason = ""
            try:
                reason = str(self.ctx.compiler.cache_block_reason() or "")
            except Exception:
                reason = ""
            if reason:
                warnings.append(f"tectonic installed but unusable: {reason}")
                self.ctx.log_event("compile_blocked", stage=self.name, reason=reason)
            else:
                warnings.append(
                    "tectonic auto-install unavailable (no network or unsupported platform)"
                )
        return None

    def _propose_fix(
        self,
        state: dict[str, Any],
        engine: str,
        errors: list[str],
        log_tail: str,
        paper_dir: Path,
    ) -> dict[str, Any] | None:
        tex_excerpt = clamp(_read_all_tex(paper_dir), 14000)
        payload = self.llm_json(
            "s7_compile_fix",
            default=None,
            schema_hint=_FIX_SCHEMA,
            engine=engine,
            errors_block="\n".join(errors[:15]) or "（未抽取到 `!` 开头的错误行）",
            log_tail=log_tail,
            tex_excerpt=tex_excerpt,
        )
        if isinstance(payload, dict):
            return payload
        return _deterministic_fix(errors, log_tail)

    def _apply_patches(
        self, paper_dir: Path, fix: dict[str, Any] | None, warnings: list[str]
    ) -> list[str]:
        if not isinstance(fix, dict):
            return []
        applied: list[str] = []
        root = paper_dir.resolve()
        for patch in coerce_list(fix.get("patches")):
            if not isinstance(patch, dict):
                continue
            rel = str(patch.get("file") or "").strip().replace("\\", "/")
            old = str(patch.get("old_text") or "")
            new = str(patch.get("new_text") or "")
            if not rel or not old:
                continue
            target = (paper_dir / rel.lstrip("./")).resolve()
            # 越界保护：只允许改 paper/ 下的 TeX 生态文件
            try:
                target.relative_to(root)
            except ValueError:
                warnings.append(f"patch rejected (outside paper/): {rel}")
                continue
            if target.suffix.lower() not in _ALLOWED_SUFFIXES:
                warnings.append(f"patch rejected (unsupported file type): {rel}")
                continue
            if not target.exists():
                warnings.append(f"patch skipped (missing file): {rel}")
                continue
            try:
                text = target.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                warnings.append(f"patch read failed for {rel}: {exc}")
                continue
            if old not in text:
                warnings.append(f"patch anchor not found in {rel}")
                continue
            target.write_text(text.replace(old, new, 1), encoding="utf-8")
            applied.append(rel)

        # 需要新增的宏包：插到 \begin{document} 之前
        packages = [str(p).strip() for p in coerce_list(fix.get("packages_to_add")) if str(p).strip()]
        if packages:
            main = paper_dir / "main.tex"
            if main.exists():
                try:
                    text = main.read_text(encoding="utf-8", errors="replace")
                    missing = [p for p in packages if f"{{{p}}}" not in text and f"{{{p}," not in text]
                    if missing and "\\begin{document}" in text:
                        inject = "".join(f"\\usepackage{{{p}}}\n" for p in missing)
                        text = text.replace("\\begin{document}", inject + "\n\\begin{document}", 1)
                        main.write_text(text, encoding="utf-8")
                        applied.append("main.tex (packages)")
                except OSError as exc:
                    warnings.append(f"package injection failed: {exc}")
        return applied

    def _write_blocked_report(
        self,
        paper_dir: Path,
        main_tex: Path,
        state: dict[str, Any],
        compile_result: dict[str, Any] | None = None,
    ) -> Artifact:
        errors = (compile_result or {}).get("errors") or []
        log_tail = (compile_result or {}).get("log_tail") or ""
        cache_reason = ""
        try:
            cache_reason = str(self.ctx.compiler.cache_block_reason() or "")
        except Exception:
            cache_reason = ""
        warnings = state.get("warnings") or []
        lines = [
            "# 编译被阻断",
            "",
            "本环境的 LaTeX 编译链不可用，因此本次运行**没有产出 PDF**。",
            "论文源码已完整落盘，可自行编译。",
            "",
        ]
        # 把「已知的具体原因」放在最前面：这是本文件存在的意义。
        if cache_reason:
            lines += [
                "## 根因（已自动诊断）",
                "",
                f"> {cache_reason}",
                "",
                "tectonic 的二进制**已经就位**（`autoresearch/vendor/tectonic/`），"
                "失败发生在写入环节而不是缓存位置上。可选处置：",
                "",
                "1. 在有完整文件权限的终端里直接编译（见下方命令），不需要重跑整条管线；",
                "2. 或换用系统 TeX 发行版（`pdflatex` / `xelatex`）——它不依赖 tectonic 的"
                "自建缓存与临时目录；",
                "3. 或改用 Overleaf（方式三），零本地依赖。",
                "",
            ]
        related = [w for w in warnings if "tectonic" in str(w).lower() or "engine" in str(w).lower()]
        if related:
            lines += ["## 相关诊断", ""] + [f"- {w}" for w in related[:8]] + [""]
        lines += [
            "## 可用的源码",
            "",
            f"- 主文件：`{self.ctx.rel(main_tex)}`",
            f"- 章节：`paper/sections/*.tex`",
            f"- 参考文献：`paper/references.bib`",
            "",
            "## 如何自行编译",
            "",
            "**方式一：tectonic（推荐，单文件、自动拉宏包）**",
            "",
            "```powershell",
            "# Windows",
            "curl -L -o tectonic.zip https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip",
            "Expand-Archive tectonic.zip -DestinationPath vendor\\tectonic -Force",
            "vendor\\tectonic\\tectonic.exe -X compile paper\\main.tex --keep-logs --outdir paper",
            "```",
            "",
            "**方式二：本地 TeX 发行版**",
            "",
            "```bash",
            "cd paper && pdflatex -interaction=nonstopmode main.tex",
            "bibtex main && pdflatex -interaction=nonstopmode main.tex",
            "pdflatex -interaction=nonstopmode main.tex",
            "```",
            "",
            "**方式三：上传到 Overleaf**",
            "",
            "把 `paper/` 整个目录压缩后上传即可（不依赖本地 TeX）。",
            "",
        ]
        if errors:
            lines += ["## 编译错误", ""] + [f"- `{e}`" for e in errors[:20]] + [""]
        if log_tail:
            lines += ["## 日志尾部", "", "```text", log_tail, "```", ""]
        return self.ctx.save_text(
            "report/COMPILE_BLOCKED.md", "\n".join(lines), stage=self.name
        )


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def _read_all_tex(paper_dir: Path, limit_chars: int = 14000) -> str:
    chunks: list[str] = []
    used = 0
    for path in sorted(paper_dir.rglob("*.tex")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(paper_dir).as_posix()
        block = f"% ===== {rel} =====\n{text}\n"
        if used + len(block) > limit_chars:
            chunks.append(f"% [{rel} 及后续省略]")
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)


def _deterministic_fix(errors: list[str], log_tail: str) -> dict[str, Any] | None:
    """无 LLM 时对最常见的几类错误做保守修复。

    只处理**能确定安全**的模式，宁可少修也不乱修。
    """
    blob = "\n".join(errors) + "\n" + log_tail
    patches: list[dict[str, str]] = []
    diagnosis_bits: list[str] = []
    packages: list[str] = []

    if "Undefined control sequence" in blob:
        for pkg in ("amsmath", "amssymb", "graphicx", "booktabs", "xcolor", "hyperref"):
            if re.search(rf"\\{pkg}", blob):
                packages.append(pkg)
        diagnosis_bits.append("疑似缺少宏包导致未定义命令")

    if "Missing $ inserted" in blob or "Missing $ inserted" in blob:
        diagnosis_bits.append("正文中含未转义的数学符号（_ ^ % & #）")

    if "File `" in blob and "not found" in blob:
        m = re.search(r"File `([^']+)' not found", blob)
        if m and m.group(1).endswith(".sty"):
            packages.append(m.group(1)[:-4])
            diagnosis_bits.append(f"缺少宏包 {m.group(1)}")

    if "Environment " in blob and "undefined" in blob:
        m = re.search(r"Environment (\w+) undefined", blob)
        if m:
            diagnosis_bits.append(f"未定义环境 {m.group(1)}（可能缺宏包或 \\begin/\\end 不配对）")

    # 通用安全修复：把正文中裸的下划线转义（只动疑似文本，不动数学模式很困难，
    # 因此这里只修 citations/url 以外的明显情形；保守起见不自动改文本）
    if not diagnosis_bits and not packages and not patches:
        return None
    return {
        "diagnosis": "；".join(diagnosis_bits) or "确定性修复：补齐缺失宏包",
        "patches": patches,
        "packages_to_add": sorted(set(packages)),
        "explanation": "本补丁由确定性规则生成（LLM 不可用时的降级路径）。",
    }


__all__ = ["CompileStage"]
