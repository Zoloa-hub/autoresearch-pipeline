"""阶段④：沙箱内代码执行、自纠错与迭代。

这是整条管线里唯一**真正执行代码**的阶段，因此三件事必须做对：

1. **隔离**：所有执行都经过 ``ctx.sandbox``，静态危险扫描（``scan_code``）先于
   任何进程启动；超时/资源上限/输出截断由沙箱统一处理，不外泄到宿主。
2. **自纠错**：捕获 stdout/stderr/Traceback → LLM 反思 → 打最小补丁 → 重跑，
   最多 ``max_debug_rounds`` 轮。每一轮都记进 ``debug_history``，可审计。
3. **不许作弊**：调试必须**保真**。若某一轮补丁删掉了指标输出、把真实计算换成
   常量、或把异常整个吞掉，本项目把它标记为 ``validity_flag`` 并记事件——
   指标可以难看，但不能是假的。这是自动科研最容易被 silently 破坏的一环。

**职责划分（阶段 B 重构的核心结论）**：本阶段不负责「实验怎么跑」——那是
:mod:`autoresearch.adapters` 的事。适配器提供命令构造与指标解析，本阶段负责
编排：多种子、执行、判断、反思、修补、保真度门禁。因此：

* 换实验后端（合成任务 / 用户自带脚本 / 真实 PyTorch 训练）只改适配器；
* 换模型、换调试策略、换种子策略只改本阶段；
* **任何适配器的产物都受同一套调试闭环与保真度检查约束**——这正是把代码生成
  留在主干、而不是交给适配器的原因。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..adapters import AdapterError, BaseExperimentAdapter, RunSpec, resolve_adapter
from ..graph.state import Artifact
from .base import Stage, StageResult, as_float, clamp, clean_text, coerce_list

_CODEGEN_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        "entrypoint": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["files"],
}

_DEBUG_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "root_cause": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
        "commands_to_verify": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "validity_note": {"type": "string"},
    },
    "required": ["diagnosis"],
}

#: 判定「补丁是否破坏了实验保真度」的信号。
_CHEAT_SIGNALS = (
    "metrics.csv",
    "metrics.jsonl",
    "accuracy=0.99",
    "return 0.99",
    "except: pass",
    "except Exception: pass",
    "except Exception:\n    pass",
)

_METRIC_FILE_NAMES = ("metrics.csv", "metrics.jsonl", "metrics.json")

#: 每个变体跑几个种子。**不是可调的性能参数，而是方法学要求**：
#: 单种子下 baseline 与方法可能恰好打平（模板实测 seed 0 即平局），
#: 此时任何「提升」都只是噪声。三种子是报告均值±标准差的最低要求。
_SEEDS_PER_VARIANT = 3


class ExperimentStage(Stage):
    name = "s4_experiment"
    title = "沙箱内代码执行与自纠错"
    requires = ("experiment_plan",)
    produces = ("code_files", "workspace", "baseline_results", "method_results", "debug_history")
    max_attempts = 2

    def run(self, state: dict[str, Any]) -> StageResult:
        missing = self.check_requires(state)
        plan = self.state_dict(state, "experiment_plan")
        if not plan:
            return StageResult.failure("no experiment plan to execute")

        warnings: list[str] = []
        if missing:
            warnings.append(f"experiment stage degraded: missing {missing}")

        artifacts: list[Artifact] = []
        ws = self.ctx.path("experiment")
        ws.mkdir(parents=True, exist_ok=True)

        # --- 0. 解析并预检实验适配器 ----------------------------------- #
        # 一切运行之前先把「能不能跑」问清楚：缺少 torch / CUDA / 数据路径这类问题
        # 应当**立刻**失败并说清原因，而不是让 6 次运行各超时一次（那要烧掉
        # timeout × 6 的墙上时间，还留下一堆难归因的 traceback）。
        try:
            adapter = resolve_adapter(
                getattr(self.ctx.cfg, "experiment_adapter", None),
                params=getattr(self.ctx.cfg, "adapter_params", None) or {},
            )
        except Exception as exc:
            return StageResult.failure(
                f"实验适配器解析失败：{type(exc).__name__}: {exc}", fatal=True
            )
        adapter.prepare(ws, plan)
        ok_env, env_reason = adapter.validate_environment()
        adapter_info = adapter.describe()
        self.ctx.log_event(
            "adapter_resolved",
            stage=self.name,
            adapter=adapter.name,
            owns_code=bool(adapter.owns_code),
            environment_ok=bool(ok_env),
        )
        if not ok_env:
            artifacts.append(
                self.ctx.save_json(
                    "experiment/adapter.json",
                    {"adapter": adapter_info, "environment": {"ok": False, "reason": env_reason}},
                    stage=self.name,
                )
            )
            return StageResult.failure(
                f"实验适配器 {adapter.name} 的环境预检未通过：{env_reason}", fatal=True
            )
        if env_reason:
            warnings.append(f"adapter environment note: {env_reason}")

        # --- 1. 获取实验代码 ------------------------------------------- #
        code_result = self._obtain_code(plan, state, ws, warnings, adapter)
        if code_result is None:
            return StageResult.failure("could not obtain runnable experiment code", retry=True)
        entrypoint, code_files, code_source = code_result

        artifacts.append(
            self.ctx.save_json(
                "experiment/code_manifest.json",
                {
                    "source": code_source,
                    "entrypoint": entrypoint,
                    "adapter": adapter_info,
                    "files": [
                        {
                            "path": p.relative_to(ws).as_posix(),
                            "bytes": p.stat().st_size if p.exists() else 0,
                        }
                        for p in code_files
                        if _is_within(p, ws)
                    ],
                },
                stage=self.name,
            )
        )

        # --- 2. 逐变体逐种子执行 --------------------------------------- #
        debug_history: list[dict[str, Any]] = []
        runs_executed: list[dict[str, Any]] = []
        results: dict[str, dict[str, Any]] = {}
        metrics_dir = self.ctx.path("metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        variants = _plan_variants(plan, max_variants=_variant_budget(self.ctx.cfg, adapter))
        # 过滤掉适配器明确说不支持的变体。理由见 BaseExperimentAdapter.supported_variants
        # ——让它们在 argparse 上撞 6 次墙，既浪费墙上时间，又把真实原因埋掉。
        supported = adapter.supported_variants()
        if supported is not None:
            skipped_variants = [v for v in variants if v not in supported]
            variants = [v for v in variants if v in supported]
            if skipped_variants:
                warnings.append(
                    f"适配器 {adapter.name} 不支持这些变体，已跳过："
                    f"{', '.join(skipped_variants)}（它支持：{', '.join(sorted(supported))}）"
                )
                self.ctx.log_event(
                    "adapter_variants_filtered", stage=self.name, adapter=adapter.name,
                    skipped=skipped_variants, supported=sorted(supported),
                )
        if not variants:
            return StageResult.failure(
                f"适配器 {adapter.name} 与实验计划没有共同变体：计划声明 "
                f"{_plan_variants(plan)}，适配器支持 {sorted(supported or [])}",
                fatal=True,
            )

        seed = int(getattr(self.ctx.cfg, "seed", 0) or 0)
        seeds = [seed + i for i in range(_SEEDS_PER_VARIANT)]
        max_debug = int(getattr(self.ctx.cfg, "max_debug_rounds", 4) or 4)
        base_timeout = max(60, int(as_float(getattr(self.ctx.cfg.sandbox, "timeout", 900), 900)))
        total_timeout = int(adapter.default_timeout or base_timeout)

        for variant in variants:
            outcome = self._run_variant_with_debug(
                ws=ws,
                adapter=adapter,
                variant=variant,
                seeds=seeds,
                max_debug=max_debug,
                timeout=total_timeout,
                plan=plan,
                warnings=warnings,
                debug_history=debug_history,
            )
            runs_executed.extend(outcome["run_records"])
            if outcome["metrics"]:
                results[variant] = outcome["metrics"]
                # 每个种子一份指标落到 metrics/<variant>/seed_<s>/，供 s5 统一发现并
                # 计算 mean±std；同时保留一份合并序列在 metrics/<variant>/ 顶层。
                self._publish_metrics(outcome["seed_dirs"], metrics_dir, variant)
            else:
                warnings.append(f"{variant}: run finished without parseable metrics")
                results[variant] = {}

        # --- 3. 汇总 --------------------------------------------------- #
        # 适配器声明的指标方向优先于通用 token 表 —— 领域知识在适配器里。
        # 材料学实测：token 表会把 k（消光系数）判成「越大越好」，而它越小越好。
        declared = adapter.metric_directions()
        comparison = _compare(results, directions=declared)
        if declared:
            # 声明了方向却没覆盖到的指标会静默回落 token 表 —— 那正是会判反的地方，
            # 所以如实记一条 warning，让它可见。
            declared_names = {str(k).split("@seed=")[0] for k in declared}
            seen_names = {
                str(k).split("@seed=")[0]
                for blk in results.values()
                for k in (blk or {})
            }
            uncovered = sorted(seen_names - declared_names)
            if uncovered:
                warnings.append(
                    f"适配器声明了 {len(declared)} 个指标方向，但以下指标未声明、"
                    f"将回落通用词表（可能判反）: {', '.join(uncovered[:8])}"
                )
        artifacts.append(
            self.ctx.save_json(
                "experiment/results.json",
                {
                    "metrics": results,
                    "comparison": comparison,
                    "runs": runs_executed,
                    "entrypoint": entrypoint,
                    "code_source": code_source,
                    "adapter": adapter_info,
                    "variants": variants,
                    "seeds": seeds,
                },
                stage=self.name,
            )
        )
        artifacts.append(
            self.ctx.save_json(
                "experiment/debug_history.json", debug_history, stage=self.name
            )
        )
        artifacts.append(
            self.ctx.save_text(
                "experiment/EXPERIMENT_RESULTS.md",
                _render_results(
                    results, comparison, runs_executed, debug_history, plan, warnings,
                    adapter_info, variants, seeds,
                ),
                stage=self.name,
            )
        )

        if not any(results.get(v) for v in variants):
            return StageResult.failure(
                "no variant produced parseable metrics", retry=True
            )

        detail = (
            f"[{adapter.name}] {len(variants)}v × {len(seeds)}s = {len(runs_executed)} runs, "
            f"{len(debug_history)} debug rounds; "
            + (comparison.get("summary_line") or "no comparable metrics")
        )
        updates: dict[str, Any] = {
            "code_files": [p.relative_to(ws).as_posix() for p in code_files if _is_within(p, ws)],
            "workspace": self.ctx.rel(ws),
            "baseline_results": results.get("baseline") or {},
            "method_results": results.get("method") or next(
                (results[v] for v in variants if v != "baseline" and results.get(v)), {}
            ),
            "debug_history": debug_history,
            "runs_executed": runs_executed,
            "adapter": adapter_info,
            "warnings": list(state.get("warnings") or []) + warnings,
        }
        return StageResult.success(detail=detail, updates=updates, artifacts=artifacts)

    # ------------------------------------------------------------------ #
    # 代码获取
    # ------------------------------------------------------------------ #
    def _obtain_code(
        self,
        plan: dict[str, Any],
        state: dict[str, Any],
        ws: Path,
        warnings: list[str],
        adapter: Any,
    ) -> tuple[str, list[Path], str] | None:
        """确定要跑的代码。

        优先级（适配器 ``owns_code=True`` 时完全不同）：

        * ``owns_code=True``：**尊重用户脚本，跳过 LLM 生成**。这是
          :class:`ScriptWrapperAdapter` 存在的意义——用户明确说了「别让模型改写我的
          训练脚本」，这个要求必须被满足。此时仍会保留调试闭环与保真度检查。
        * 否则：① 适配器提供的模板 → ② LLM 基于模板生成/适配 → ③ 内置保底脚本。

        三级降级都失败才返回 ``None``：没有实验代码意味着整条管线归零，
        值得用两层保底换「总能跑起来」。
        """
        if getattr(adapter, "owns_code", False):
            entry = str(getattr(adapter, "entrypoint", "") or "")
            try:
                provided = [p for p in adapter.code_files(ws) if p.is_file()]
            except Exception as exc:
                warnings.append(f"adapter {adapter.name}.code_files() failed: {exc}")
                provided = []
            if provided:
                return entry or provided[0].name, provided, f"adapter-provided ({adapter.name})"
            warnings.append(
                f"adapter {adapter.name} declares owns_code=True but exposes no readable "
                f"entrypoint ({entry!r}); falling back to the built-in generator"
            )

        # --- ① 适配器模板 --------------------------------------------- #
        seeded_paths: list[Path] = []
        try:
            templates = adapter.seed_code(ws, plan) or {}
        except Exception as exc:
            warnings.append(f"adapter {adapter.name}.seed_code() failed: {exc}")
            templates = {}
        for rel, content in templates.items():
            safe = _safe_relpath(str(rel))
            if not safe or not str(content or "").strip():
                continue
            try:
                out = ws / safe
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(str(content), encoding="utf-8")
                seeded_paths.append(out)
            except OSError as exc:
                warnings.append(f"adapter template write failed for {safe}: {exc}")

        # --- ② LLM 基于模板生成/适配 ---------------------------------- #
        payload = self.llm_json(
            "s4_codegen",
            default=None,
            schema_hint=_CODEGEN_SCHEMA,
            plan_block=_format_plan_block(plan),
            existing_code_block=(
                _read_files_block(ws, limit_chars=14000)
                if seeded_paths
                else "（无既有代码库，从零生成）"
            ),
            data_info=adapter.codegen_data_info(),
            variant=adapter.codegen_variants(),
            workspace_conventions=adapter.codegen_conventions(),
        )

        files: list[Path] = []
        entrypoint = ""
        if isinstance(payload, dict):
            entrypoint = str(payload.get("entrypoint") or "").strip()
            for item in coerce_list(payload.get("files")):
                if not isinstance(item, dict):
                    continue
                rel = _safe_relpath(str(item.get("path") or ""))
                content = _clean_code(str(item.get("content") or ""))
                if not rel or not content.strip():
                    continue
                target = ws / rel
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    files.append(target)
                except OSError as exc:
                    warnings.append(f"could not write {rel}: {exc}")

        if files:
            if not entrypoint or not (ws / _safe_relpath(entrypoint)).exists():
                entrypoint = _guess_entrypoint(files, ws)
            self.ctx.log_event(
                "code_generated", stage=self.name, source="llm", files=len(files),
                entrypoint=entrypoint,
            )
            return entrypoint, files, "llm"

        # --- 降级：适配器模板直接可用 -------------------------------- #
        # 走到这里说明 LLM 没能给出可用文件（离线、解析失败、返回空）。
        # 此时**仍然可以跑**：适配器提供的模板本身就是一份完整可运行的实现。
        # 这比"再写一个内置保底脚本"更好——保底脚本与适配器可能不兼容
        # （命令参数、指标路径都对不上），而适配器模板天然与它自己的
        # build_command / parse_results 配套。
        if seeded_paths:
            warnings.append(
                "LLM codegen unavailable; running the adapter's seeded template as-is"
            )
            entry = str(getattr(adapter, "entrypoint", "") or "")
            entry_path = ws / _safe_relpath(entry) if entry else None
            if entry_path is None or not entry_path.is_file():
                entry = _guess_entrypoint(seeded_paths, ws)
            self.ctx.log_event(
                "code_generated", stage=self.name, source="adapter-template",
                files=len(seeded_paths), entrypoint=entry,
            )
            return entry, seeded_paths, f"adapter-template ({adapter.name})"

        return None

    # ------------------------------------------------------------------ #
    # 单变体执行 + 调试闭环（多种子）
    # ------------------------------------------------------------------ #
    def _run_variant_with_debug(
        self,
        ws: Path,
        adapter: Any,
        variant: str,
        seeds: list[int],
        max_debug: int,
        timeout: int,
        plan: dict[str, Any],
        warnings: list[str],
        debug_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """跑完一个变体的全部种子。

        **多种子是刚性要求，不是可选项**：单一种子下 baseline 与方法可能恰好打平
        （模板实测 seed 0 就是平局），此时任何「提升」都是噪声。因此本阶段默认对
        每个变体跑 ``_SEEDS_PER_VARIANT`` 个种子，并把每个种子的指标分别落盘，
        让 ⑤ 能算出跨种子的 mean±std。

        调试闭环按种子独立进行：第一轮产生的补丁会被后续种子复用（只跑一次修复），
        因为「脚本能不能跑」与环境有关，而与种子无关。
        """
        seed_dirs: dict[int, Path] = {}
        run_records: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}
        already_fixed = False
        params = _variant_params(plan, variant)

        for seed in seeds:
            out_dir = f"runs/{variant}/seed_{seed}"
            run_dir = ws / out_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            spec = RunSpec(
                variant=variant, seed=seed, out_dir=out_dir, params=dict(params)
            )

            outcome = self._run_single(
                ws=ws,
                adapter=adapter,
                spec=spec,
                run_dir=run_dir,
                max_debug=max_debug,
                timeout=timeout,
                plan=plan,
                warnings=warnings,
                debug_history=debug_history,
                skip_debug=already_fixed,
            )
            run_records.append(outcome["run_record"])
            if outcome["metrics"]:
                seed_dirs[seed] = run_dir
                # 同一变体的跨种子序列合并：绘图用（每个种子一条曲线）
                for name, values in outcome["metrics"].items():
                    metrics.setdefault(f"{name}@seed={seed}", []).extend(
                        float(v) for v in values
                    )
            if outcome.get("patched"):
                already_fixed = True

        return {"run_records": run_records, "metrics": metrics, "seed_dirs": seed_dirs}

    def _run_single(
        self,
        ws: Path,
        adapter: Any,
        spec: RunSpec,
        run_dir: Path,
        max_debug: int,
        timeout: int,
        plan: dict[str, Any],
        warnings: list[str],
        debug_history: list[dict[str, Any]],
        skip_debug: bool = False,
    ) -> dict[str, Any]:
        """跑单个 (variant, seed)，必要时进入自纠错闭环。

        命令由**适配器**构造（``adapter.build_command(spec)``），指标也由适配器解析
        （``adapter.parse_results(run_dir)``）。本方法只管编排：执行、判断成功、
        反思、打补丁、保真度检查——这部分对所有适配器一视同仁。
        """
        variant, seed, out_dir = spec.variant, spec.seed, spec.out_dir
        result = None
        metrics: dict[str, Any] = {}
        round_no = 0
        patched = False
        try:
            primary_cmd = adapter.build_command(spec)
        except Exception as exc:
            raise AdapterError(
                f"适配器 {adapter.name}.build_command() 失败：{type(exc).__name__}: {exc}"
            ) from exc
        attempt_cmds = [primary_cmd, _loose_command(primary_cmd)]
        cmd_index = 0
        effective_debug_budget = 0 if skip_debug else max_debug

        while round_no <= effective_debug_budget:
            round_no += 1
            current_cmd = attempt_cmds[min(cmd_index, len(attempt_cmds) - 1)]
            self._info(
                f"s4: running {variant} seed={seed} (round {round_no}): {' '.join(current_cmd)}"
            )
            try:
                result = self.ctx.sandbox.run_command(
                    current_cmd, timeout=timeout, cwd=str(ws)
                )
            except Exception as exc:
                # 沙箱拒绝执行（argv 非法、含 shell 元字符等）也是可诊断的失败，
                # 不该让整条管线崩掉。
                from ..tools.sandbox import ExecResult

                result = ExecResult(
                    ok=False, returncode=-1, stdout="",
                    stderr=f"sandbox rejected the command: {type(exc).__name__}: {exc}",
                    duration=0.0, backend=getattr(self.ctx.sandbox, "name", "?"),
                )
            self.ctx.log_event(
                "sandbox_run",
                stage=self.name,
                adapter=adapter.name,
                variant=variant,
                seed=seed,
                round=round_no,
                ok=result.ok,
                returncode=result.returncode,
                timed_out=result.timed_out,
                duration=round(result.duration, 2),
            )

            try:
                metrics = adapter.parse_results(run_dir) or {}
            except Exception as exc:
                warnings.append(
                    f"{variant}/seed={seed}: adapter.parse_results() failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                metrics = {}
            if result.ok and metrics:
                break

            if round_no > effective_debug_budget:
                break

            # 先用宽松命令行再试一次（常见失败：脚本不支持某个参数）
            if cmd_index == 0 and not result.ok and metrics == {}:
                stderr_head = (result.stderr or "").lower()
                if any(tok in stderr_head for tok in ("unrecognized argument", "no such option",
                                                      "unrecognized arguments", "error: unrecognized",
                                                      "unknown option", "unexpected argument")):
                    cmd_index = 1
                    debug_history.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "round": round_no,
                            "strategy": "retry_with_loose_flags",
                            "error": _first_error_line(result.stderr),
                            "fix": "去掉脚本不支持的参数重跑",
                            "ok": False,
                        }
                    )
                    continue

            # LLM 反思与打补丁
            patch = self._reflect_and_patch(
                ws, str(getattr(adapter, "entrypoint", "") or ""), variant,
                current_cmd, result, plan, round_no,
            )
            validity = _validity_check(patch, None)
            applied = self._apply_patch(ws, patch, warnings) if patch else []
            if applied:
                patched = True
            debug_history.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "round": round_no,
                    "strategy": "llm_patch",
                    "error": _first_error_line(result.stderr) or _first_error_line(result.stdout),
                    "diagnosis": clean_text(str((patch or {}).get("diagnosis") or "")),
                    "root_cause": clean_text(str((patch or {}).get("root_cause") or "")),
                    "files_patched": applied,
                    "commands_to_verify": coerce_list((patch or {}).get("commands_to_verify")),
                    "validity_note": clean_text(str((patch or {}).get("validity_note") or "")),
                    "validity_flag": validity,
                    "ok": False,
                }
            )
            self.ctx.log_event(
                "debug_round", stage=self.name, variant=variant, seed=seed, round=round_no,
                patched=len(applied), validity_flag=bool(validity),
                error=_first_error_line(result.stderr)[:300],
            )
            if not applied:
                if cmd_index == 0:
                    cmd_index = 1
                    continue
                break

        if debug_history and debug_history[-1].get("variant") == variant \
                and debug_history[-1].get("seed") == seed:
            debug_history[-1]["ok"] = bool(result is not None and result.ok and metrics)

        run_record = {
            "variant": variant,
            "seed": seed,
            "cmd": attempt_cmds[min(cmd_index, len(attempt_cmds) - 1)],
            "ok": bool(result is not None and result.ok),
            "returncode": result.returncode if result is not None else -1,
            "duration": round(result.duration, 2) if result is not None else 0.0,
            "timed_out": bool(result is not None and result.timed_out),
            "rounds": round_no,
            "metrics_found": bool(metrics),
            "stdout_tail": clamp(result.stdout or "", 1500) if result is not None else "",
            "stderr_tail": clamp(result.stderr or "", 1500) if result is not None else "",
            "out_dir": out_dir,
            "patched": bool(patched),
        }
        return {"run_record": run_record, "metrics": metrics, "run_dir": run_dir, "patched": patched}

    def _reflect_and_patch(
        self,
        ws: Path,
        entrypoint: str,
        variant: str,
        cmd: list[str],
        result: Any,
        plan: dict[str, Any],
        round_no: int,
    ) -> dict[str, Any] | None:
        current_files = _read_files_block(ws, limit_chars=14000)
        payload = self.llm_json(
            "s4_debug",
            default=None,
            schema_hint=_DEBUG_SCHEMA,
            attempt=round_no,
            run_command=" ".join(cmd),
            returncode=getattr(result, "returncode", -1),
            stdout_tail=clamp(getattr(result, "stdout", "") or "", 3000),
            stderr_tail=clamp(getattr(result, "stderr", "") or "", 4000),
            current_files_block=current_files,
            plan_block=_format_plan_block(plan),
        )
        return payload if isinstance(payload, dict) else None

    def _apply_patch(
        self, ws: Path, patch: dict[str, Any], warnings: list[str]
    ) -> list[str]:
        applied: list[str] = []
        for item in coerce_list(patch.get("files")):
            if not isinstance(item, dict):
                continue
            rel = _safe_relpath(str(item.get("path") or ""))
            content = _clean_code(str(item.get("content") or ""))
            if not rel or not content.strip():
                continue
            target = ws / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                applied.append(rel)
            except OSError as exc:
                warnings.append(f"patch write failed for {rel}: {exc}")
        return applied

    # ------------------------------------------------------------------ #
    def _publish_metrics(
        self, seed_dirs: dict[int, Path], metrics_root: Path, variant: str
    ) -> None:
        """把每个种子的指标文件发布到 ``metrics/<variant>/`` 下。

        布局（两层，各有用途）：

        * ``metrics/<variant>/seed_<s>/metrics.csv`` —— 每个种子一份，
          供 ⑤ 计算**跨种子**均值±标准差（正确口径）；
        * ``metrics/<variant>/metrics.csv`` —— 第一个种子的副本，
          纯粹是让人打开目录时能直接看到东西，**不参与跨种子统计**。
        """
        dest_root = metrics_root / variant
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        for seed in sorted(seed_dirs):
            src = seed_dirs[seed]
            dest = dest_root / f"seed_{seed}"
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            for name in _METRIC_FILE_NAMES:
                candidate = src / name
                if candidate.exists():
                    try:
                        (dest / name).write_bytes(candidate.read_bytes())
                    except OSError:
                        continue

        # 顶层副本（便于肉眼查看）
        first = sorted(seed_dirs)[0] if seed_dirs else None
        if first is not None:
            src = seed_dirs[first]
            for name in _METRIC_FILE_NAMES:
                candidate = src / name
                if candidate.exists():
                    try:
                        (dest_root / name).write_bytes(candidate.read_bytes())
                    except OSError:
                        continue


# --------------------------------------------------------------------------- #
# 纯函数：路径、命令、指标、渲染
# --------------------------------------------------------------------------- #


def _safe_relpath(raw: str) -> str:
    """只允许工作区内的相对路径，挡掉 ``../`` 与绝对路径。"""
    raw = (raw or "").strip().replace("\\", "/").lstrip("./")
    if not raw:
        return ""
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return ""
    if re.match(r"^[A-Za-z]:", raw) or raw.startswith("/"):
        return ""
    rel = "/".join(parts)
    if not rel or len(rel) > 200:
        return ""
    return rel


_CODE_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?\s*```\s*$", re.S)


def _clean_code(text: str) -> str:
    """剥掉 LLM 习惯性包上的 ``` 围栏。"""
    if not text:
        return ""
    m = _CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).rstrip() + "\n"
    return text


def _guess_entrypoint(files: list[Path], ws: Path) -> str:
    """在生成的文件里挑一个入口：优先名字像入口的，其次是含 ``__main__`` 的。"""
    preferred = ("train.py", "main.py", "run.py", "experiment.py", "run_experiment.py")
    by_name = {p.name.lower(): p for p in files}
    for name in preferred:
        if name in by_name:
            return by_name[name].relative_to(ws).as_posix()
    for p in files:
        try:
            if "__main__" in p.read_text(encoding="utf-8", errors="replace"):
                return p.relative_to(ws).as_posix()
        except OSError:
            continue
    return files[0].relative_to(ws).as_posix() if files else "train.py"


def _variant_budget(cfg: Any, adapter: Any) -> int:
    """决定这次跑几个臂：配置上限与适配器自身上限取小。

    两侧都有存在的理由，缺一不可：

    * **配置上限**（``--max-variants``）是用户对这次运行的总预算约束；
    * **适配器上限**（``BaseExperimentAdapter.max_variants``）是后端能力的约束——
      合成任务可以轻松跑 4 臂，而真实训练脚本跑 6 臂可能是 6 小时。

    取小值意味着用户**不能**把一个只愿意跑 2 臂的适配器强行推到 6 臂，
    这对「真实训练」是必要的保护（一个跑不完的消融比没有消融更糟）。
    """
    configured = int(getattr(cfg, "max_variants", 0) or 0)
    if configured <= 0:
        configured = 6
    adapter_cap = getattr(adapter, "max_variants", None)
    if adapter_cap is None:
        return configured
    try:
        cap = int(adapter_cap)
    except (TypeError, ValueError):
        return configured
    return max(2, min(configured, cap)) if cap > 0 else configured


def _plan_variants(plan: dict[str, Any], max_variants: int | None = 2) -> list[str]:
    """从实验计划推导要跑哪些**变体**。

    两类变体，语义完全不同，不能混为一谈：

    * **主对照**：``baseline`` 与 ``method``。这是论文主表与「核心 claim 是否成立」
      的唯一依据，**永远排在最前**，任何预算都先保证它们。
    * **消融格点**：由 s3 的消融矩阵给出。注意矩阵的 ``variants`` 是**轴上的取值**
      （如 ``momentum`` 轴的 ``"0.0 (off)"`` / ``"0.9 (on)"``），**不是变体名**。
      早期实现直接把取值当变体名，于是产出 ``'0.9_on'`` 这种既无法归因到轴、
      也无法解析成超参的东西。正确做法是合成为 ``<轴名>=<取值>``。

    ``max_variants`` 的语义：``None`` = 不限；整数 = **总臂数上限**（主对照 + 消融）。
    注意早期实现把整数当成「只返回主对照」，于是 ``max_variants=4`` 与 ``=2``
    返回同样的东西——用户要求跑 4 个臂却只拿到 2 个，而且不会有任何提示。
    """
    primary: list[str] = ["baseline", "method"]
    if max_variants is not None and max_variants <= 2:
        return primary[: max(1, max_variants)]

    names = list(primary)
    for label, _axis, _value, _params in _ablation_cells(plan):
        if label not in names:
            names.append(label)
    if max_variants is None:
        return names
    return names[: max(2, int(max_variants))]


def _ablation_cells(plan: dict[str, Any]) -> list[tuple[str, str, str, dict[str, Any]]]:
    """把消融矩阵摊平成 ``[(变体名, 轴名, 取值, 该格的超参), ...]``。

    **变体名用序号形式** ``abl-<n>``，而不是 ``momentum=0.9 (on)``。原因有两条，
    第二条是硬约束：

    1. 变体名会进目录名与命令行，含空格/括号/等号的名字在 argv 里是单个 token，
       靠 shell 语义之外的方式传递极易出错；``abl-1`` 这类名字任何后端都能接受。
    2. **轴与取值通过 ``spec.params`` 传递，而不是编码进名字里**。适配器负责把
       ``{"momentum": 0.9}`` 变成自己认识的命令行参数——这才是「适配器知道怎么跑」
       的正确分工。把语义压进变体名，等于强迫所有适配器去解析我们的命名约定。

    轴名会规范成合法标识符（``noise level`` → ``noise_level``），因为多数训练脚本
    用 argparse，非法标识符会直接报错。
    """
    cells: list[tuple[str, str, str, dict[str, Any]]] = []
    index = 0
    for entry in coerce_list(plan.get("ablation_matrix")):
        if not isinstance(entry, dict):
            continue
        axis = _axis_identifier(str(entry.get("name") or "ablation"))
        for raw in coerce_list(entry.get("variants")):
            if isinstance(raw, dict):
                label_value = str(raw.get("name") or "").strip()
                extra = {str(k): v for k, v in raw.items() if k != "name"}
            else:
                label_value = str(raw).strip()
                extra = {}
            if not label_value:
                continue
            index += 1
            params: dict[str, Any] = {axis: _coerce_scalar(label_value)}
            params.update({k: v for k, v in extra.items() if k != "name"})
            cells.append((f"abl-{index}", axis, label_value, params))
    return cells


def _axis_identifier(raw: str) -> str:
    """把消融轴名规范成合法标识符（会进 argparse 的 ``--flag``）。"""
    name = re.sub(r"[^0-9A-Za-z_]+", "_", (raw or "ablation").strip()).strip("_")
    if not name:
        return "ablation"
    if name[0].isdigit():
        name = "ax_" + name
    return name[:32]


def _coerce_scalar(raw: str) -> Any:
    """把 ``"0.9 (on)"`` / ``"2e-4"`` / ``"0.70"`` 解析成数值（失败则保留字符串）。"""
    text = (raw or "").strip()
    head = text.split("(")[0].strip().split()[0] if text.split() else text
    for candidate in (head, text):
        try:
            return int(candidate)
        except (TypeError, ValueError):
            pass
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return text


def _safe_variant(raw: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_.=-]+", "_", (raw or "").strip()).strip("_")
    return name[:48]


def _variant_params(plan: dict[str, Any], variant: str) -> dict[str, Any]:
    """给某个变体取出该应用的超参。

    主对照（baseline/method）返回空 dict——它们必须用**同一套默认超参**，
    否则「提升」就归因不到方法本身。
    消融格点返回 ``{轴名: 取值}``，让适配器把它变成命令行参数。
    """
    for label, _axis, _value, params in _ablation_cells(plan):
        if _safe_variant(label) == variant or label == variant:
            return dict(params)
    return {}


def _loose_command(cmd: list[str]) -> list[str]:
    """退化成最小参数集再试一次。

    脚本不认某个参数时的报错五花八门（`unrecognized arguments`、
    `unknown option`…），与其逐个枚举，不如直接构造一个「只保留程序名 +
    位置参数 + --variant/--out-dir」的命令。位置参数指不以 `-` 开头的 token，
    以及紧跟 flag 的取值——这里保守处理：保留程序名、第一个位置参数（通常是
    脚本路径），以及 --variant/--out-dir 及其值。
    """
    if not cmd:
        return list(cmd)
    keep: list[str] = [cmd[0]]
    i = 1
    while i < len(cmd):
        token = cmd[i]
        if token in ("--variant", "--out-dir") and i + 1 < len(cmd):
            keep.extend([token, cmd[i + 1]])
            i += 2
            continue
        if not token.startswith("-") and len(keep) == 1:
            keep.append(token)
        i += 1
    return keep if len(keep) > 1 else list(cmd)


def _is_within(path: Path, root: Path) -> bool:
    """判断路径是否在 root 之内。

    适配器可能把脚本放在工作区外（`own_code=true` 的场景），这时
    `relative_to` 会抛 `ValueError`，而产物清单只是想记一笔相对路径——
    不该因此让整阶段失败。
    """
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def _first_error_line(text: str) -> str:
    """从 stdout/stderr 里挑最能说明问题的一行（优先 Traceback 尾行）。"""
    text = text or ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"^\w*(Error|Exception|Warning)\b", ln.strip()) or "Error:" in ln:
            return ln.strip()[:400]
    for ln in lines:
        if ln.strip().startswith("Traceback"):
            return ln.strip()[:400]
    return (lines[-1].strip()[:400] if lines else "")


def _read_files_block(ws: Path, limit_chars: int = 14000) -> str:
    """把工作区里的代码汇总成给 LLM 看的块（有界）。"""
    chunks: list[str] = []
    used = 0
    for path in sorted(ws.rglob("*.py")):
        if any(part.startswith(".") or part == "runs" for part in path.relative_to(ws).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(ws).as_posix()
        block = f"--- {rel} ---\n{text}\n"
        if used + len(block) > limit_chars:
            chunks.append(f"... [{rel} 及后续文件因长度限制省略]")
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)


def _validity_check(patch: dict[str, Any] | None, previous_entry: Any) -> str:
    """检测补丁是否在悄悄降低实验保真度。

    返回空串表示没发现问题，否则返回一段人类可读的告警文本。
    """
    if not isinstance(patch, dict):
        return ""
    notes: list[str] = []
    for item in coerce_list(patch.get("files")):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        if not content:
            continue
        lowered = content.lower()
        rel = str(item.get("path") or "?")
        if "metrics.csv" not in lowered and "metrics.jsonl" not in lowered:
            notes.append(f"{rel}: 补丁后不再写出 metrics.csv/jsonl（指标证据链断裂）")
        if re.search(r"except\s*(Exception)?\s*:\s*\n\s*pass", content):
            notes.append(f"{rel}: 出现 `except: pass`，可能吞掉了真实失败")
        if re.search(r"return\s+0?\.9\d+", content) or re.search(r"=\s*0?\.9\d+\s*#\s*fake", content):
            notes.append(f"{rel}: 出现疑似硬编码的高分返回值")
        if re.search(r"^\s*pass\s*$", content, re.M) and len(content.strip()) < 40:
            notes.append(f"{rel}: 补丁后文件几乎为空，疑似把计算变成了空操作")
    return "；".join(notes)


def _compare(
    results: dict[str, dict[str, Any]],
    directions: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """主对照：baseline vs 最佳非基线变体。

    为什么要有回退链：指标名在多种子运行下带 ``@seed=N`` 后缀，直接 ``取交集``
    可能为空；而一个运行确实有结果却报「无法对照」是会自动科研里最尴尬的错误
    ——数据在，报告说没有。

    对照对象的选择规则：在共同指标上相对提升最大的非 baseline 变体。
    显式记录 ``baseline`` / ``treatment`` 是谁，避免读者猜「method 到底指哪个」。
    """
    baseline_block = results.get("baseline") or {}
    if not baseline_block:
        return {
            "available": False,
            "summary_line": "缺少 baseline 指标，无法对照（对照缺失时任何提升都不可信）",
        }

    def _strip_seed(name: str) -> str:
        return name.split("@seed=")[0]

    def _flatten(block: dict[str, Any]) -> dict[str, float]:
        """把 ``{metric@seed=N: [...]}`` / ``{metric: [...]}`` 统一成 ``{metric: 代表值}``。"""
        grouped: dict[str, list[float]] = {}
        for key, values in (block or {}).items():
            if not isinstance(values, list) or not values:
                continue
            nums = [float(v) for v in values if isinstance(v, (int, float))]
            if not nums:
                continue
            metric = _strip_seed(str(key))
            grouped.setdefault(metric, []).append(_best_of(metric, nums, directions))
        return {m: (sum(vs) / len(vs)) for m, vs in grouped.items() if vs}

    baseline_flat = _flatten(baseline_block)
    candidates: list[tuple[float, str, dict[str, float]]] = []
    for variant, block in results.items():
        if variant == "baseline" or not block:
            continue
        flat = _flatten(block)
        shared = set(baseline_flat) & set(flat)
        if not shared:
            continue
        rels = []
        for metric in shared:
            b, m = baseline_flat[metric], flat[metric]
            if b:
                rel = (m - b) / abs(b)
                rels.append(rel if _higher_is_better(metric, directions) else -rel)
        if rels:
            candidates.append((sum(rels) / len(rels), variant, flat))

    if not candidates:
        return {
            "available": False,
            "summary_line": (
                "有 baseline 指标，但没有任何变体与它有共同指标名，无法对照"
            ),
        }

    candidates.sort(key=lambda item: -item[0])
    _score, treatment, treatment_flat = candidates[0]

    per_metric: dict[str, Any] = {}
    bits: list[str] = []
    improved: list[str] = []
    regressed: list[str] = []
    primary = ""
    best_rel = 0.0
    for name in sorted(set(baseline_flat) & set(treatment_flat)):
        b, m = baseline_flat[name], treatment_flat[name]
        delta = m - b
        rel = (delta / abs(b)) if b else 0.0
        direction = "higher" if _higher_is_better(name, directions) else "lower"
        per_metric[name] = {
            "baseline_best": round(b, 6),
            "method_best": round(m, 6),
            "delta": round(delta, 6),
            "relative": round(rel, 6),
            "direction": direction,
            "baseline_final": round(b, 6),
            "method_final": round(m, 6),
        }
        bits.append(f"{name} {b:.4g}→{m:.4g} ({rel:+.1%})")
        if abs(rel) > abs(best_rel):
            best_rel, primary = rel, name
        better = (delta > 0) == (direction == "higher") and delta != 0
        worse = (delta < 0) == (direction == "higher") and delta != 0
        if better:
            improved.append(name)
        elif worse:
            regressed.append(name)

    return {
        "available": True,
        "baseline": "baseline",
        "treatment": treatment,
        "other_variants": [name for _, name, _ in candidates[1:]],
        "primary_metric": primary,
        "per_metric": per_metric,
        "improved": improved,
        "regressed": regressed,
        "supports_claim": bool(improved) and not regressed,
        "summary_line": "; ".join(bits[:4]),
        "basis": "per-seed best, averaged across seeds",
    }


def _best_of(
    name: str, values: list[float], directions: dict[str, bool] | None = None
) -> float:
    """取"最好"的那个值：方向由适配器声明优先决定。

    模块级函数（不是 `_compare` 的闭包），所以 `directions` 必须显式传入——
    否则会落进"模块级函数里引用外层局部变量"的 NameError。
    """
    return min(values) if not _higher_is_better(name, directions) else max(values)


def _higher_is_better(name: str, directions: dict[str, bool] | None = None) -> bool:
    """指标方向：适配器声明优先，其次共享 token 表。

    此前 s3/s4/s5/s9 各有一份 token 表副本，且已经漂移。现在方向判定收敛到
    `tools.metrics` 一张表；而**领域差异**由适配器通过 `metric_directions()`
    显式声明——通用词表只是兜底，且对非 ML 领域并不可靠（材料学实测 13/19 不可信）。
    """
    from ..tools.metrics import _higher_is_better as _shared

    return bool(_shared(name, declared=directions))

def _format_plan_block(plan: dict[str, Any]) -> str:
    lines = [
        f"objective: {plan.get('objective')}",
        f"core_claim: {plan.get('core_claim')}",
        f"dataset: {plan.get('dataset')}",
        f"baseline: {plan.get('baseline')}",
        "metrics: " + ", ".join(
            f"{m.get('name')}({m.get('direction')})" for m in (plan.get("metrics") or [])
        ),
        "milestones:",
    ]
    for m in plan.get("milestones") or []:
        lines.append(
            f"  - {m.get('id')} {m.get('name')}: {m.get('description')} "
            f"[runs={m.get('runs')}, success={m.get('success_criterion')}]"
        )
    if plan.get("ablation_matrix"):
        lines.append("ablations:")
        for a in plan["ablation_matrix"]:
            lines.append(f"  - {a.get('name')}: {', '.join(a.get('variants') or [])}")
    return "\n".join(lines)


def _render_results(
    results: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    runs: list[dict[str, Any]],
    debug_history: list[dict[str, Any]],
    plan: dict[str, Any],
    warnings: list[str],
    adapter_info: dict[str, Any] | None = None,
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
) -> str:
    parts = [
        "# 实验结果",
        "",
        f"计划目标：{plan.get('objective', '—')}",
        "",
    ]

    # 适配器声明放最前面：读者必须先知道「这组实验是什么后端跑的、
    # 能支撑什么级别的结论」，再去看数字。这是诚实报告的基本顺序。
    if adapter_info:
        parts += [
            "## 实验后端",
            "",
            f"- 适配器：`{adapter_info.get('name', '?')}`",
            f"- 说明：{adapter_info.get('description') or '—'}",
            f"- 自带代码（跳过 LLM 生成）：{'是' if adapter_info.get('owns_code') else '否'}",
            f"- 变体：{', '.join(variants or [])}",
            f"- 种子：{', '.join(str(s) for s in (seeds or []))}",
            "",
            f"> **结果适用范围**：{adapter_info.get('quality_note') or '未声明'}",
            "",
        ]

    parts += [
        "## 运行记录",
        "",
        "| 变体 | 种子 | ok | 返回码 | 秒 | 轮次 | 拿到指标 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        parts.append(
            f"| {r.get('variant')} | {r.get('seed')} | {'✅' if r.get('ok') else '❌'} | "
            f"{r.get('returncode')} | {r.get('duration')} | {r.get('rounds')} | "
            f"{'✅' if r.get('metrics_found') else '❌'} |"
        )

    parts += ["", "## 指标对照", ""]
    if comparison.get("available"):
        # 列名必须用真实的 treatment 名，而不是写死 "method"：多臂运行下
        # treatment 可能是 `momentum=0.9 (on)` 这样的消融格点，标成 method 会让
        # 读者以为是在跟基准方法比。对照的双方必须一眼可读。
        base_label = str(comparison.get("baseline") or "baseline")
        treat_label = str(comparison.get("treatment") or "treatment")
        parts += [
            f"| 指标 | {base_label}(best) | {treat_label}(best) | Δ | 相对 | 方向 |",
            "|---|---|---|---|---|---|",
        ]
        for name, v in comparison["per_metric"].items():
            parts.append(
                f"| {name} | {v['baseline_best']:.6g} | {v['method_best']:.6g} | "
                f"{v['delta']:+.6g} | {v['relative']:+.2%} | {v['direction']} |"
            )
        others = comparison.get("other_variants") or []
        parts += [
            "",
            f"**对照**：`{base_label}`（基准） vs `{treat_label}`（治疗组，按共同指标上的"
            "相对提升最大者选出）",
            f"**主指标**：{comparison.get('primary_metric') or '—'}；"
            f"**支持 claim**：{'是' if comparison.get('supports_claim') else '否'}"
            f"（改善 {len(comparison.get('improved') or [])} 项，"
            f"回退 {len(comparison.get('regressed') or [])} 项）",
        ]
        if others:
            parts.append(f"**其余变体**（未作为对照）：{', '.join(f'`{o}`' for o in others)}")
    else:
        parts.append(f"_{comparison.get('summary_line', '不可对照')}_")

    parts += ["", "## 原始指标序列", ""]
    if results:
        for variant, series in results.items():
            parts.append(f"### {variant}")
            parts.append("")
            for name, values in series.items():
                preview = ", ".join(f"{v:.4g}" for v in values[:12])
                more = " …" if len(values) > 12 else ""
                parts.append(f"- `{name}` ({len(values)} 点)：{preview}{more}")
            parts.append("")
    else:
        parts.append("_没有解析到任何指标。_")

    if debug_history:
        parts += ["## 自纠错历史", ""]
        for d in debug_history:
            parts.append(
                f"### {d['variant']} round {d['round']} — {d.get('strategy')}"
            )
            parts.append("")
            if d.get("error"):
                parts.append(f"- 报错：`{d['error']}`")
            if d.get("diagnosis"):
                parts.append(f"- 诊断：{d['diagnosis']}")
            if d.get("root_cause"):
                parts.append(f"- 根因：{d['root_cause']}")
            if d.get("files_patched"):
                parts.append(f"- 修改文件：{', '.join(d['files_patched'])}")
            if d.get("validity_note"):
                parts.append(f"- 保真性说明：{d['validity_note']}")
            if d.get("validity_flag"):
                parts.append(f"- ⚠️ **保真性告警**：{d['validity_flag']}")
            parts.append(f"- 结果：{'通过' if d.get('ok') else '仍未通过'}")
            parts.append("")

    if warnings:
        parts += ["## 告警", ""] + [f"- {w}" for w in warnings] + [""]

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# 内置保底脚本：当 LLM 与模板都不可用时使用。零第三方依赖，确定性，CPU 秒级。
# --------------------------------------------------------------------------- #


__all__ = ["ExperimentStage"]
