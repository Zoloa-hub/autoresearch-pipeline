"""Auto-Research 命令行入口。

```
python -m autoresearch.cli run --direction "..." [选项]
python -m autoresearch.cli resume <RUN_ID>
python -m autoresearch.cli status [RUN_ID]
python -m autoresearch.cli verify <RUN_ID>
python -m autoresearch.cli doctor
python -m autoresearch.cli stages
python -m autoresearch.cli demo            # 离线端到端冒烟（等价于 run --mock --offline）
```

退出码：``0`` 完整成功 / ``1`` 部分完成（有产出但有失败阶段或未编译出 PDF）/ ``2`` 致命失败。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import AutoResearchConfig, DEFAULT_RUNS_DIR, PROJECT_ROOT, WORKSPACE_ROOT, load_config
from .graph.checkpoint import Checkpointer, find_run_dir, list_runs, merge_resume, resume_plan
from .graph.state import STAGES, STAGE_TITLES, state_schema, summarize_state
from .runner import EXIT_FATAL, EXIT_OK, EXIT_PARTIAL, build_context, run_pipeline

# --------------------------------------------------------------------------- #
# 输出助手
# --------------------------------------------------------------------------- #

_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def ok(text: str) -> str:
    return _c(text, "32")


def warn(text: str) -> str:
    return _c(text, "33")


def bad(text: str) -> str:
    return _c(text, "31")


def dim(text: str) -> str:
    return _c(text, "2")


def bold(text: str) -> str:
    return _c(text, "1")


def emit(payload: dict[str, Any], as_json: bool, human: str) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(human)


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoresearch",
        description="Auto-Research：选题→检索→构思→实验→分析→写作→编译→评审→交付 全自动管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m autoresearch.cli demo\n"
            "  python -m autoresearch.cli run --direction \"稀疏注意力在长序列上的效率-精度权衡\" "
            "--venue NeurIPS\n"
            "  python -m autoresearch.cli run --direction \"...\" --llm-provider deepseek "
            "--model deepseek-chat\n"
            "  python -m autoresearch.cli status\n"
            "  python -m autoresearch.cli verify 20260101-120000-my-topic\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"autoresearch {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # -- 共用选项 ------------------------------------------------------ #
    def add_common(p: argparse.ArgumentParser, with_direction: bool = True) -> None:
        if with_direction:
            p.add_argument("--direction", "-d", default="", help="研究大方向（领域/数据集/核心目标）")
            p.add_argument("--direction-file", default="", help="从文件读取研究大方向（适合长 brief）")
        p.add_argument("--venue", default=None, help="目标会议/期刊：NeurIPS/ICLR/ICML/CVPR/ACL/AAAI/ACM")
        p.add_argument("--language", choices=["zh", "en"], default=None, help="产出语言（默认 zh）")
        p.add_argument("--seed", type=int, default=None, help="随机种子")
        p.add_argument("--runs-dir", default=None, help="运行目录根（默认 <workspace>/.autoresearch/runs）")
        p.add_argument("--llm-provider", default=None,
                       choices=["openai", "deepseek", "ollama", "mock"],
                       help="LLM 后端；mock 为离线启发式（无需 API Key，可跑通全流程）")
        p.add_argument("--model", default=None, help="模型名")
        p.add_argument("--base-url", default=None, help="OpenAI 兼容端点")
        p.add_argument("--temperature", type=float, default=None)
        p.add_argument("--max-tokens", type=int, default=None)
        p.add_argument("--offline", action="store_true", default=None,
                       help="禁用一切网络检索（arXiv/S2/OpenAlex/Crossref）")
        p.add_argument("--sandbox", default=None, choices=["subprocess", "docker"],
                       help="代码执行后端")
        p.add_argument("--sandbox-timeout", type=int, default=None, help="单次执行超时（秒）")
        p.add_argument("--compile-engine", default=None,
                       choices=["tectonic", "pdflatex", "xelatex", "none"],
                       help="LaTeX 引擎；none 表示跳过编译")
        p.add_argument("--no-tectonic-install", action="store_true",
                       help="禁止自动下载 tectonic")
        p.add_argument("--max-review-rounds", type=int, default=None, help="评审迭代轮数上限")
        p.add_argument("--max-debug-rounds", type=int, default=None, help="实验自纠错轮数上限")
        p.add_argument("--max-ideas", type=int, default=None, help="候选假设数上限")
        p.add_argument("--experiment-adapter", default=None, metavar="SPEC",
                       help="实验后端适配器：内置名 (synthetic-toy/script-wrapper)、"
                            ".py 文件路径、module:Class，或已安装包的 entry point 名")
        p.add_argument("--adapter-arg", action="append", default=None, metavar="K=V",
                       help="传给适配器的参数，可重复。值支持 bool/int/float/逗号列表/"
                            "JSON 字面量（dict 也走 JSON），例如 "
                            "--adapter-arg script=train.py --adapter-arg program=torchrun "
                            "--adapter-arg 'metrics_map={\"val_acc\":\"val_accuracy\"}'")
        p.add_argument("--max-variants", type=int, default=None,
                       help="单次运行最多跑几个臂（主对照+消融格点）；实际取与适配器"
                            "上限的较小值。默认 6")
        p.add_argument("--quiet", "-q", action="store_true", help="减少日志输出")
        p.add_argument("--json", action="store_true", help="以 JSON 输出结果摘要")
        p.add_argument("--verbose", "-v", action="store_true", help="详细日志（DEBUG）")

    # -- run ----------------------------------------------------------- #
    p_run = sub.add_parser("run", help="跑一条完整管线")
    add_common(p_run)
    p_run.add_argument("--dry-run", action="store_true", help="不写检查点（只跑不落盘快照）")
    p_run.add_argument("--stop-on-error", action="store_true", help="任一阶段失败即终止")
    p_run.add_argument("--no-resume", action="store_true", help="即使发现同名 run 也从头跑")

    # -- resume -------------------------------------------------------- #
    p_resume = sub.add_parser("resume", help="从断点续跑")
    p_resume.add_argument("run_id", nargs="?", default="", help="运行 ID（支持唯一前缀）")
    p_resume.add_argument("--restart", action="store_true", help="忽略已完成阶段，从头重跑")
    add_common(p_resume, with_direction=False)

    # -- status -------------------------------------------------------- #
    p_status = sub.add_parser("status", help="查看运行状态")
    p_status.add_argument("run_id", nargs="?", default="", help="运行 ID（省略则列出全部）")
    p_status.add_argument("--runs-dir", default=None)
    p_status.add_argument("--json", action="store_true")
    p_status.add_argument("--artifacts", action="store_true", help="列出全部产物")

    # -- verify -------------------------------------------------------- #
    p_verify = sub.add_parser("verify", help="校验论文数字是否可溯源到实验证据")
    p_verify.add_argument("run_id", nargs="?", default="", help="运行 ID（省略则取最近一次）")
    p_verify.add_argument("--runs-dir", default=None)
    p_verify.add_argument("--rel-tol", type=float, default=1e-3, help="相对误差容限")
    p_verify.add_argument("--json", action="store_true")
    p_verify.add_argument("--strict", action="store_true", help="存在未溯源数字即返回退出码 1")

    # -- doctor -------------------------------------------------------- #
    p_doctor = sub.add_parser("doctor", help="环境自检")
    p_doctor.add_argument("--probe-network", action="store_true", help="实际发起一次网络探测")
    p_doctor.add_argument("--json", action="store_true")

    # -- stages -------------------------------------------------------- #
    p_stages = sub.add_parser("stages", help="列出阶段与状态键")
    p_stages.add_argument("--json", action="store_true")

    # -- demo ---------------------------------------------------------- #
    p_demo = sub.add_parser("demo", help="离线端到端冒烟（mock LLM + 无网络）")
    p_demo.add_argument("--direction", "-d", default="自动化科研管线的可复现性研究",
                        help="演示用的研究大方向")
    p_demo.add_argument("--runs-dir", default=None)
    p_demo.add_argument("--max-review-rounds", type=int, default=2)
    p_demo.add_argument("--json", action="store_true")

    return parser


# --------------------------------------------------------------------------- #
# 配置组装
# --------------------------------------------------------------------------- #


#: ``--adapter-arg`` 的值按这些类型转换；转换失败则保留字符串。
_ADAPTER_BOOL_TRUE = ("true", "yes", "on", "1")
_ADAPTER_BOOL_FALSE = ("false", "no", "off", "0")


def _parse_adapter_args(pairs: list[str] | None) -> dict[str, Any]:
    """把 ``--adapter-arg k=v`` 解析成字典，并做尽力而为的类型转换。

    转换规则：``true/false`` → bool，整数串 → int，含小数点的数字串 → float，
    ``a,b`` → list（适配器的 ``metrics_map``/``args_template`` 常用逗号分隔），
    其余保持字符串。类型搞错会让适配器直接抛错，所以这里只做**无歧义**的转换。
    """
    out: dict[str, Any] = {}
    for raw in pairs or []:
        text = str(raw)
        key, sep, value = text.partition("=")
        key = key.strip()
        if not sep or not key:
            print(warn(f"忽略无效的 --adapter-arg {text!r}（需要 K=V 形式）"), file=sys.stderr)
            continue
        value = value.strip()
        lowered = value.lower()
        if lowered in _ADAPTER_BOOL_TRUE:
            out[key] = True
        elif lowered in _ADAPTER_BOOL_FALSE:
            out[key] = False
        elif re.fullmatch(r"-?\d+", value):
            out[key] = int(value)
        elif re.fullmatch(r"-?\d+\.\d+", value):
            out[key] = float(value)
        elif value.startswith("{") or value.startswith("["):
            # JSON 形式的 dict / list：``--adapter-arg 'metrics_map={"val_acc":"val_accuracy"}'``
            # 没有这条，ScriptWrapperAdapter 的 metrics_map / assets 就只能靠写 .py 子类设置，
            # 而那正是脚本适配器想避免的门槛。
            try:
                out[key] = json.loads(value)
            except ValueError:
                out[key] = value
        elif "," in value:
            out[key] = [piece.strip() for piece in value.split(",") if piece.strip()]
        else:
            out[key] = value
    return out


def _read_direction(args: argparse.Namespace) -> str:
    direction = (getattr(args, "direction", "") or "").strip()
    path = (getattr(args, "direction_file", "") or "").strip()
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = WORKSPACE_ROOT / p
        try:
            from_file = p.read_text(encoding="utf-8").strip()
            return from_file or direction
        except OSError as exc:
            print(warn(f"无法读取 --direction-file {p}: {exc}"), file=sys.stderr)
    return direction


def config_from_args(args: argparse.Namespace, require_direction: bool = True) -> AutoResearchConfig:
    direction = _read_direction(args)
    if require_direction and not direction:
        raise SystemExit(
            "错误：必须提供 --direction（研究大方向），或用 --direction-file 指定文件。"
        )

    overrides: dict[str, Any] = {"direction": direction}
    if getattr(args, "venue", None):
        overrides["venue"] = args.venue
    if getattr(args, "language", None):
        overrides["language"] = args.language
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "runs_dir", None):
        overrides["runs_dir"] = args.runs_dir
    if getattr(args, "max_review_rounds", None) is not None:
        overrides["max_review_rounds"] = args.max_review_rounds
    if getattr(args, "max_debug_rounds", None) is not None:
        overrides["max_debug_rounds"] = args.max_debug_rounds
    if getattr(args, "max_ideas", None) is not None:
        overrides["max_ideas"] = args.max_ideas
    if getattr(args, "max_variants", None) is not None:
        # 死参数修复：此前 --max-variants 只在 argparse 里注册、从未进入 overrides，
        # 于是用户设它完全不生效（永远用 config 默认值）。一个"接受但忽略"的开关
        # 比没有这个开关更糟——它让人以为自己控制住了预算。
        overrides["max_variants"] = args.max_variants
    if getattr(args, "dry_run", False):
        overrides["dry_run"] = True
    if getattr(args, "experiment_adapter", None):
        overrides["experiment_adapter"] = args.experiment_adapter
    adapter_params = _parse_adapter_args(getattr(args, "adapter_arg", None))
    if adapter_params:
        overrides["adapter_params"] = adapter_params

    llm: dict[str, Any] = {}
    for attr, key in (("llm_provider", "provider"), ("model", "model"), ("base_url", "base_url"),
                      ("temperature", "temperature"), ("max_tokens", "max_tokens")):
        value = getattr(args, attr, None)
        if value is not None:
            llm[key] = value
    if llm:
        overrides["llm"] = llm

    sandbox: dict[str, Any] = {}
    if getattr(args, "sandbox", None):
        sandbox["backend"] = args.sandbox
    if getattr(args, "sandbox_timeout", None) is not None:
        sandbox["timeout"] = args.sandbox_timeout
    if sandbox:
        overrides["sandbox"] = sandbox

    if getattr(args, "offline", None):
        overrides["retrieve"] = {"offline": True}

    compile_over: dict[str, Any] = {}
    if getattr(args, "compile_engine", None):
        compile_over["engine"] = args.compile_engine
    if getattr(args, "no_tectonic_install", False):
        compile_over["auto_install_tectonic"] = False
    if compile_over:
        overrides["compile"] = compile_over

    cfg = load_config(**overrides)
    if getattr(args, "verbose", False):
        from .logging_utils import configure_logging

        configure_logging(level="DEBUG")
    return cfg


# --------------------------------------------------------------------------- #
# 子命令实现
# --------------------------------------------------------------------------- #


def _progress_printer(as_json: bool):
    if as_json:
        return None
    state = {"n": 0}

    def on_step(record: Any, s: dict[str, Any]) -> None:
        state["n"] += 1
        status = getattr(record, "status", "")
        mark = ok("✔") if getattr(record, "ok", False) else (
            warn("⚠") if status == "skipped" else bad("✘")
        )
        detail = str(getattr(record, "detail", "") or "")[:110]
        elapsed = float(getattr(record, "elapsed", 0.0) or 0.0)
        stage = getattr(record, "stage", "?")
        nxt = getattr(record, "next_stage", "")
        arrow = f" → {nxt}" if nxt and nxt != "__end__" else ""
        print(f"  {mark} [{state['n']:02d}] {stage:<16} {elapsed:6.1f}s  {dim(detail)}{arrow}")

    return on_step


def cmd_run(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    direction = cfg.direction

    if not args.json:
        print(bold(f"\nAuto-Research v{__version__}"))
        print(f"  方向   : {direction[:110]}")
        print(f"  会议   : {cfg.venue}   语言: {cfg.language}   种子: {cfg.seed}")
        print(f"  LLM    : {cfg.llm.provider}/{cfg.llm.model}"
              f"{'  (离线)' if cfg.retrieve.offline else ''}")
        print(f"  沙箱   : {cfg.sandbox.backend}   编译: {cfg.compile.engine}")
        print(f"  输出   : {cfg.runs_dir}")
        print("")

    state, ctx, engine = run_pipeline(
        cfg,
        quiet=bool(getattr(args, "quiet", False)),
        on_step=_progress_printer(bool(getattr(args, "json", False))),
    )
    return _report_run(state, ctx, bool(getattr(args, "json", False)))


def cmd_resume(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir) if args.runs_dir else DEFAULT_RUNS_DIR
    run_id = args.run_id or ""
    if not run_id:
        runs = list_runs(runs_dir)
        if not runs:
            print(bad(f"未在 {runs_dir} 找到任何运行。"))
            return EXIT_FATAL
        run_id = runs[0]["run_id"]
        print(dim(f"未指定 run_id，使用最近一次：{run_id}"))

    run_dir = find_run_dir(runs_dir, run_id)
    if run_dir is None:
        print(bad(f"找不到运行 '{run_id}'（在 {runs_dir}）"))
        return EXIT_FATAL

    saved = Checkpointer(run_dir).load()
    if saved is None:
        print(bad(f"{run_dir} 中没有 state.json，无法恢复。"))
        return EXIT_FATAL

    plan = resume_plan(saved)
    if not args.json:
        print(bold(f"恢复运行 {saved.get('run_id', run_id)}"))
        print(f"  已完成: {', '.join(plan['completed']) or '(无)'}")
        if plan["failed"]:
            print(f"  失败过: {warn(', '.join(plan['failed']))}")
        print(f"  将从此阶段继续: {bold(plan['next_stage'] or '(已全部完成)')}")
        print("")

    if plan["complete"] and not getattr(args, "restart", False):
        if not args.json:
            print(ok("该运行已全部完成。如需重跑请加 --restart。"))
        else:
            print(json.dumps({"run_id": run_id, "complete": True, "plan": plan},
                             ensure_ascii=False, indent=2))
        return EXIT_OK

    cfg = config_from_args(args, require_direction=False)
    cfg.direction = str(saved.get("direction") or cfg.direction or "")
    cfg.venue = str(saved.get("venue") or cfg.venue)
    cfg.language = str(saved.get("language") or cfg.language)
    cfg.seed = int(saved.get("seed") if saved.get("seed") is not None else cfg.seed)
    cfg.run_id = str(saved.get("run_id") or run_dir.name)

    state, ctx, engine = run_pipeline(
        cfg,
        resume=True,
        force_restart=bool(getattr(args, "restart", False)),
        quiet=bool(getattr(args, "quiet", False)),
        run_dir=run_dir,
        on_step=_progress_printer(bool(getattr(args, "json", False))),
    )
    return _report_run(state, ctx, bool(getattr(args, "json", False)))


def _report_run(state: dict[str, Any], ctx: Any, as_json: bool) -> int:
    exit_code = int(state.get("exit_code", EXIT_PARTIAL))
    status = state.get("stage_status") or {}
    failed = [s for s, v in status.items() if v == "failed"]
    skipped = [s for s, v in status.items() if v == "skipped"]
    # 对照结果有三级来源（④ 的原始对照 → ⑤ 的跨种子口径 → 逐 run 统计）。
    # 只读 ``experiment/results`` 是不够的：多种子运行时 ④ 的对照会因为指标名带
    # ``@seed=N`` 后缀而判定「无法对照」，而真正该进论文的是 ⑤ 的跨种子统计。
    # 复用统一的解析器，保证 CLI 摘要与最终报告不会给出两个不同答案。
    try:
        from .stages.s9_finalize import _resolve_comparison

        comparisons = _resolve_comparison(state)
    except Exception:  # pragma: no cover - 解析器不可用时降级
        comparisons = {"available": False, "summary_line": "（无法解析对照结果）"}

    payload = {
        "run_id": state.get("run_id"),
        "run_dir": ctx.run_dir.as_posix(),
        "exit_code": exit_code,
        "final_pdf": state.get("final_pdf") or None,
        "final_report": str(ctx.run_dir / "report" / "FINAL_REPORT.md"),
        "review_score": state.get("review_score"),
        "review_verdict": state.get("review_verdict"),
        "review_rounds": state.get("review_round"),
        "selected_idea": (state.get("selected_idea") or {}).get("title"),
        "papers": len(state.get("papers") or []),
        "ideas": len(state.get("ideas") or []),
        "runs_executed": len(state.get("runs_executed") or []),
        "figures": sum(len(v) for v in (state.get("figured") or {}).values()),
        "comparison": comparisons,
        "supports_claim": bool(comparisons.get("supports_claim")),
        "failed_stages": failed,
        "skipped_stages": skipped,
        "open_issues": ((state.get("open_issues") or {}).get("counts") or {}),
        "elapsed_seconds": state.get("elapsed_seconds"),
        "llm_calls": getattr(getattr(ctx.llm, "usage", lambda: None)(), "calls", 0)
        if hasattr(ctx.llm, "usage") else 0,
    }

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return exit_code

    print("")
    print(bold("─" * 66))
    print(f"  运行 ID   : {state.get('run_id')}")
    print(f"  运行目录  : {ctx.run_dir}")
    print(f"  阶段      : {summarize_state(state)}")
    if comparisons.get("available"):
        print(f"  指标对照  : {comparisons.get('summary_line')}")
        print(f"  支持 claim: {'是' if comparisons.get('supports_claim') else '否'}"
              + (f"（治疗组：{comparisons['treatment']}）"
                 if comparisons.get("treatment") else ""))
    else:
        print(f"  指标对照  : {warn(comparisons.get('summary_line') or '不可用')}")
    print(f"  评审      : {state.get('review_score')}/10 "
          f"({state.get('review_verdict')}, {state.get('review_round')} 轮)")
    pdf = state.get("final_pdf")
    if pdf:
        print(f"  论文 PDF  : {ok(str(ctx.run_dir / pdf))}")
    else:
        print(f"  论文 PDF  : {warn('未产出（见 report/COMPILE_BLOCKED.md）')}")
    print(f"  最终报告  : {ctx.run_dir / 'report' / 'FINAL_REPORT.md'}")
    if failed:
        print(f"  失败阶段  : {bad(', '.join(failed))}")
    if skipped:
        print(f"  跳过阶段  : {warn(', '.join(skipped))}")
    counts = (state.get("open_issues") or {}).get("counts") or {}
    if counts:
        print(f"  未解决问题: {counts.get('major', 0)} major / {counts.get('minor', 0)} minor")
    print(bold("─" * 66))

    if exit_code == EXIT_OK:
        print(ok("\n完成。"))
    elif exit_code == EXIT_PARTIAL:
        print(warn("\n部分完成：有阶段失败或未编译出 PDF，请查看最终报告第 3 节。"))
    else:
        print(bad("\n致命失败：未产出实质结果，请查看运行目录下的日志与 events.jsonl。"))
    return exit_code


def cmd_status(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir) if args.runs_dir else DEFAULT_RUNS_DIR
    run_id = args.run_id or ""

    if not run_id:
        runs = list_runs(runs_dir)
        if args.json:
            print(json.dumps({"runs_dir": runs_dir.as_posix(), "runs": runs},
                             ensure_ascii=False, indent=2))
            return EXIT_OK
        if not runs:
            print(f"{dim('（无运行记录）')}  {runs_dir}")
            return EXIT_OK
        print(bold(f"运行记录（{runs_dir}）\n"))
        print(f"  {'run_id':<44} {'阶段':>7} {'分数':>6}  下一阶段")
        print(f"  {'─' * 44} {'─' * 7} {'─' * 6}  {'─' * 18}")
        for r in runs:
            score = r.get("review_score") or 0
            print(
                f"  {r['run_id'][:44]:<44} {r['completed']:>3}/{len(STAGES):<3} "
                f"{score:>6.1f}  {r.get('next_stage') or ok('完成')}"
            )
        return EXIT_OK

    run_dir = find_run_dir(runs_dir, run_id)
    if run_dir is None:
        print(bad(f"找不到运行 '{run_id}'"))
        return EXIT_FATAL
    state = Checkpointer(run_dir).load()
    if state is None:
        print(bad(f"{run_dir} 没有 state.json"))
        return EXIT_FATAL

    plan = resume_plan(state)
    if args.json:
        payload = dict(plan)
        payload["dir"] = run_dir.as_posix()
        if args.artifacts:
            payload["artifacts"] = state.get("artifacts") or []
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    print(bold(f"运行 {state.get('run_id', run_id)}"))
    print(f"  目录     : {run_dir}")
    print(f"  方向     : {str(state.get('direction', ''))[:80]}")
    print(f"  论文标题 : {state.get('paper_title') or '—'}")
    print(f"  最终 PDF : {state.get('final_pdf') or warn('未产出')}")
    print("")
    print(bold("  阶段状态"))
    attempts = state.get("stage_attempts") or {}
    errors = state.get("stage_errors") or {}
    for stage in STAGES:
        st = (state.get("stage_status") or {}).get(stage, "pending")
        colorize = ok if st in ("done", "skipped") else (bad if st == "failed" else dim)
        note = ""
        if st == "failed" and errors.get(stage):
            note = dim(" — " + str(errors[stage][-1])[:70])
        print(f"    {stage:<16} {colorize(st):<20} attempt={attempts.get(stage, 0)}{note}")

    print("")
    print(f"  评审     : {state.get('review_score')}/10 ({state.get('review_verdict')})")
    print(f"  产物     : {len(state.get('artifacts') or [])} 个")
    if args.artifacts:
        print("")
        for a in state.get("artifacts") or []:
            print(f"    {a.get('kind', '?'):<5} {a.get('path')}")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    from .verify import verify_run

    runs_dir = Path(args.runs_dir) if args.runs_dir else DEFAULT_RUNS_DIR
    run_id = args.run_id or ""
    if not run_id:
        runs = list_runs(runs_dir)
        if not runs:
            print(bad(f"未在 {runs_dir} 找到任何运行。"))
            return EXIT_FATAL
        run_id = runs[0]["run_id"]
        if not args.json:
            print(dim(f"未指定 run_id，使用最近一次：{run_id}"))

    run_dir = find_run_dir(runs_dir, run_id)
    if run_dir is None:
        print(bad(f"找不到运行 '{run_id}'"))
        return EXIT_FATAL

    report = verify_run(run_dir, rel_tol=args.rel_tol)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        text = report.render()
        print(text)
        out = run_dir / "report" / "VERIFY_REPORT.md"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(dim(f"报告已写入 {out}"))
        except OSError as exc:
            print(warn(f"无法写入报告：{exc}"))

    if args.strict and not report.ok:
        return EXIT_PARTIAL
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, hint: str = "") -> None:
        checks.append({"name": name, "status": status, "detail": detail, "hint": hint})

    # Python
    add("python", "ok", f"{sys.version.split()[0]} ({sys.executable})")

    # 关键依赖
    for mod, required in (
        ("openai", True), ("jinja2", False), ("matplotlib", False),
        ("pandas", False), ("fitz", False), ("requests", False),
    ):
        try:
            __import__(mod)
            add(f"dep:{mod}", "ok", "已安装")
        except ImportError:
            add(
                f"dep:{mod}",
                "bad" if required else "warn",
                "未安装",
                "pip install openai" if required else "可选：影响图表/PDF 解析能力",
            )

    # 可选依赖
    try:
        import langgraph  # noqa: F401

        add("dep:langgraph", "ok",
            "已安装（可用 graph.compile_langgraph 编译同一份节点图）")
    except ImportError:
        add("dep:langgraph", "warn", "未安装", "可选：自研引擎已提供等价语义")

    # LLM 配置
    try:
        cfg = load_config()
        provider = cfg.llm.provider
        has_key = bool(cfg.llm.api_key)
        if provider == "mock":
            add("llm", "ok", "provider=mock（离线启发式，无需 Key）")
        elif has_key:
            add("llm", "ok", f"provider={provider} model={cfg.llm.model} key=已设置")
        else:
            add(
                "llm",
                "warn",
                f"provider={provider} model={cfg.llm.model} 未检测到 API Key",
                "设置 OPENAI_API_KEY / DEEPSEEK_API_KEY，或用 --llm-provider mock",
            )
    except Exception as exc:
        add("llm", "bad", f"配置加载失败：{exc}")

    # 检索
    try:
        from .tools.retrieve import LiteratureSearch

        search = LiteratureSearch(load_config().retrieve)
        add("retrieve", "ok", f"引擎就绪（{', '.join(load_config().retrieve.sources)}）")
        if getattr(args, "probe_network", False):
            t0 = time.monotonic()
            papers = search.search("retrieval augmented generation", max_results=2,
                                   sources=["arxiv"])
            dt = time.monotonic() - t0
            if papers:
                add("retrieve:network", "ok", f"arXiv 可达，{dt:.1f}s 返回 {len(papers)} 篇")
            else:
                add("retrieve:network", "warn",
                    f"arXiv 无返回（{dt:.1f}s）", "离线环境属正常；可用 --offline 跳过")
    except Exception as exc:
        add("retrieve", "bad", f"检索层不可用：{exc}")

    # 沙箱
    try:
        from .tools.sandbox import make_sandbox

        cfg = load_config()
        probe = PROJECT_ROOT / ".doctor_probe"
        sandbox = make_sandbox(cfg.sandbox, probe, event_logger=None)
        result = sandbox.run_python(code="print('sandbox-ok')", timeout=60)
        if result.ok and "sandbox-ok" in result.stdout:
            add("sandbox", "ok", f"backend={getattr(sandbox, 'name', '?')} 可执行 Python")
        else:
            add("sandbox", "bad", f"执行失败 rc={result.returncode}: {result.tail(300)}")
        try:
            import shutil

            shutil.rmtree(probe, ignore_errors=True)
        except Exception:
            pass
        # docker 可用性
        try:
            from .tools.sandbox import DockerSandbox

            docker = DockerSandbox(cfg.sandbox, probe)
            if docker.available():
                add("sandbox:docker", "ok", "docker 可用（更强的隔离）")
            else:
                add("sandbox:docker", "warn", "docker 不可用", "需要隔离时请安装 Docker Desktop")
        except Exception as exc:
            add("sandbox:docker", "warn", f"docker 探测失败：{exc}")
    except Exception as exc:
        add("sandbox", "bad", f"沙箱不可用：{exc}")

    # 编译器
    try:
        from .tools.latex import LatexCompiler

        cfg = load_config()
        compiler = LatexCompiler(cfg.compile, PROJECT_ROOT / ".doctor_probe")
        engine = compiler.detect()
        if engine:
            add("latex", "ok", f"引擎={engine}")
        else:
            cache_reason = ""
            try:
                cache_reason = str(compiler.cache_block_reason() or "")
            except Exception:
                cache_reason = ""
            if cache_reason:
                add("latex", "warn", "tectonic 已安装但无法使用（bundle 缓存写入被拒）",
                    cache_reason)
            else:
                add(
                    "latex",
                    "warn",
                    "未找到 tectonic/pdflatex/xelatex",
                    "s7 会自动尝试下载 tectonic；或手动放置二进制到 vendor/tectonic/",
                )
        add("latex:bibtex", "ok" if compiler.bibtex_available() else "warn",
            "bibtex 可用" if compiler.bibtex_available() else "bibtex 不可用（tectonic 不需要）")
    except Exception as exc:
        add("latex", "bad", f"编译器不可用：{exc}")

    # 提示词与模板
    prompts_dir = PROJECT_ROOT / "prompts"
    n_prompts = len(list(prompts_dir.glob("*.md"))) if prompts_dir.is_dir() else 0
    add("prompts", "ok" if n_prompts >= 10 else "warn", f"{n_prompts} 个提示词文件",
        "" if n_prompts >= 10 else "提示词缺失会导致阶段降级")
    tpl = PROJECT_ROOT / "templates" / "paper" / "main.tex"
    add("template:paper", "ok" if tpl.exists() else "warn",
        "LaTeX 骨架就绪" if tpl.exists() else "缺少 templates/paper/main.tex（将使用内置兜底）")
    tpl_exp = PROJECT_ROOT / "templates" / "experiment" / "train.py"
    add("template:experiment", "ok" if tpl_exp.exists() else "warn",
        "参考实验脚本存在（手动运行用）" if tpl_exp.exists()
        else "缺少 templates/experiment/train.py（仅影响手动运行参考，不影响管线："
             "管线用的是适配器自带的模板）")

    # 目录可写
    for name, path in (("runs_dir", DEFAULT_RUNS_DIR),):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe_file = path / ".write_test"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink()
            add(name, "ok", f"可写 {path}")
        except OSError as exc:
            add(name, "bad", f"不可写 {path}: {exc}", "检查权限或改用 --runs-dir")

    bads = [c for c in checks if c["status"] == "bad"]
    warns = [c for c in checks if c["status"] == "warn"]

    if args.json:
        print(json.dumps(
            {"checks": checks, "summary": {"ok": len(checks) - len(bads) - len(warns),
                                           "warn": len(warns), "bad": len(bads)}},
            ensure_ascii=False, indent=2,
        ))
    else:
        print(bold(f"\nAuto-Research 环境自检（v{__version__}）\n"))
        icon = {"ok": ok("✔"), "warn": warn("!"), "bad": bad("✘")}
        for c in checks:
            print(f"  {icon[c['status']]} {c['name']:<22} {c['detail']}")
            if c["hint"] and c["status"] != "ok":
                print(f"      {dim('→ ' + c['hint'])}")
        print("")
        print(f"  结果：{ok(str(len(checks) - len(bads) - len(warns)) + ' ok')}, "
              f"{warn(str(len(warns)) + ' warn')}, {bad(str(len(bads)) + ' bad')}")
        if bads:
            print(bad("\n存在致命问题，请先修复后再运行管线。"))
        elif warns:
            print(warn("\n存在警告：管线仍可运行，但相关能力会降级。"))
        else:
            print(ok("\n环境完备。"))
        print("")

    return EXIT_FATAL if bads else EXIT_OK


def cmd_stages(args: argparse.Namespace) -> int:
    from .stages import OPTIONAL_STAGES, STAGE_CLASSES

    schema = state_schema()
    rows = []
    for cls in STAGE_CLASSES:
        rows.append(
            {
                "stage": cls.name,
                "title": cls.title,
                "requires": list(cls.requires),
                "produces": list(cls.produces),
                "max_attempts": cls.max_attempts,
                "optional": cls.name in OPTIONAL_STAGES,
                "conditional": cls.router is not None,
            }
        )

    if args.json:
        print(json.dumps({"stages": rows, "state_keys": schema["keys"]},
                         ensure_ascii=False, indent=2))
        return EXIT_OK

    print(bold(f"\n管线阶段（{len(rows)} 个）\n"))
    print(f"  {'#':<3} {'阶段':<16} {'标题':<26} {'尝试':>4} {'可选':>4} {'条件':>4}")
    print(f"  {'─' * 3} {'─' * 16} {'─' * 26} {'─' * 4} {'─' * 4} {'─' * 4}")
    for i, r in enumerate(rows, 1):
        title = STAGE_TITLES.get(r["stage"], r["title"])
        print(f"  {i:<3} {r['stage']:<16} {title[:26]:<26} {r['max_attempts']:>4} "
              f"{'Y' if r['optional'] else '-':>4} {'Y' if r['conditional'] else '-':>4}")
    print("")
    print(bold("  依赖关系"))
    for r in rows:
        req = ", ".join(r["requires"]) or "—"
        prod = ", ".join(r["produces"]) or "—"
        print(f"    {r['stage']:<16} 需要: {req}")
        print(f"    {'':<16} 产出: {prod}")
    print("")
    print(bold("  状态键"))
    for key, typ in schema["keys"].items():
        print(f"    {key:<22} {typ}")
    print("")
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    """离线端到端冒烟：mock LLM + 无网络 + 自动降级编译。"""
    if not args.json:
        print(bold("\n离线端到端冒烟测试（mock LLM，无网络）"))
        print(dim("  目的：验证在没有任何外部依赖的环境里，管线仍能产出完整交付物。\n"))

    cfg = load_config(
        direction=args.direction,
        llm={"provider": "mock", "model": "mock-heuristic"},
        retrieve={"offline": True},
        compile={"engine": "none"},
        max_review_rounds=args.max_review_rounds,
        runs_dir=args.runs_dir or DEFAULT_RUNS_DIR,
    )
    state, ctx, engine = run_pipeline(
        cfg, quiet=True, on_step=_progress_printer(bool(args.json))
    )
    code = _report_run(state, ctx, bool(args.json))

    if not args.json:
        print(dim("\n  说明：mock 模式用于验证管线连通性与降级路径，"))
        print(dim("  其生成的内容不代表真实科研质量。接真实模型请去掉 --llm-provider mock。"))
    return code


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


_DISPATCH = {
    "run": cmd_run,
    "resume": cmd_resume,
    "status": cmd_status,
    "verify": cmd_verify,
    "doctor": cmd_doctor,
    "stages": cmd_stages,
    "demo": cmd_demo,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    handler = _DISPATCH.get(args.command)
    if handler is None:  # pragma: no cover - argparse 已保证
        parser.print_help()
        return EXIT_FATAL

    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print(warn("\n已中断。状态已保存，可用 resume 继续："))
        print(f"  python -m autoresearch.cli resume <run_id>")
        return EXIT_PARTIAL
    except SystemExit:
        raise
    except Exception as exc:
        print(bad(f"\n未捕获的异常：{type(exc).__name__}: {exc}"))
        import traceback

        traceback.print_exc()
        return EXIT_FATAL


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
