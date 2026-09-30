"""运行上下文与管线装配。

``RunContext`` 是阶段与外部世界之间**唯一**的接触面：LLM、检索、沙箱、编译器、
提示词库、日志、路径。阶段拿到 ctx 之后，除了 ``state`` 不再需要别的输入——
这让每个阶段都能用假的 ctx 单测，也让「换掉沙箱后端/换掉 LLM」不影响阶段代码。

``run_pipeline`` 负责：建目录 → 装 ctx → 建图 → （可选）合并断点 → 跑 → 收尾。
退出码语义：
* ``0``：全流程完成，且产出了论文（有 PDF，或至少 main.tex + 完整章节）；
* ``1``：部分完成（有阶段 failed，但仍产出报告）；
* ``2``：致命失败（没有任何实质产出，或初始化就失败）。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AutoResearchConfig, PROJECT_ROOT, load_config
from .graph.checkpoint import Checkpointer, find_run_dir, merge_resume
from .graph.engine import GraphEngine, Node, linear_nodes
from .graph.state import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    Artifact,
    hash_file,
    new_state,
    record_artifact,
    summarize_state,
    to_jsonable,
)
from .logging_utils import EventLogger, configure_logging, get_logger
from .stages import OPTIONAL_STAGES, STAGE_REGISTRY

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FATAL = 2

#: 运行目录下的产物子目录（与 stage 实现里的相对路径保持一致）。
RUN_SUBDIRS: tuple[str, ...] = (
    "literature", "ideas", "plan", "experiment", "metrics", "figures",
    "analysis", "paper", "review", "report", "logs",
)

_KIND_BY_SUFFIX = {
    ".json": "json", ".jsonl": "json", ".md": "md", ".txt": "md", ".log": "log",
    ".csv": "csv", ".py": "py", ".tex": "tex", ".bib": "bib", ".pdf": "pdf",
    ".png": "png", ".svg": "png", ".sty": "tex", ".cls": "tex",
}


# --------------------------------------------------------------------------- #
# 运行 ID
# --------------------------------------------------------------------------- #


def slugify(text: str, max_len: int = 40) -> str:
    """把研究方向转成文件名友好的短 slug。中文保留（NTFS/UTF-8 都没问题），
    但去掉路径分隔符与空白。"""
    slug = re.sub(r"[\\/:*?\"<>|\s]+", "-", (text or "").strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "run"


def make_run_id(direction: str, prefix: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{prefix + '-' if prefix else ''}{slugify(direction, 32)}-{stamp}"
    return base


# --------------------------------------------------------------------------- #
# RunContext
# --------------------------------------------------------------------------- #


@dataclass
class RunContext:
    """阶段可用的全部外部能力。"""

    cfg: AutoResearchConfig
    run_dir: Path
    llm: Any
    events: EventLogger
    log: Any
    sandbox: Any
    search: Any
    compiler: Any
    prompts: Any
    extra: dict[str, Any] = field(default_factory=dict)

    # -- 路径 ----------------------------------------------------------- #
    def path(self, *parts: str) -> Path:
        """``run_dir`` 下的路径，自动创建父目录。"""
        p = self.run_dir.joinpath(*[str(x) for x in parts if str(x)])
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return p

    def rel(self, p: Path | str) -> str:
        """转成相对 ``run_dir`` 的 POSIX 路径。"""
        path = Path(p)
        try:
            return path.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except (ValueError, OSError):
            return path.as_posix()

    def log_event(self, event: str, **fields: Any) -> None:
        try:
            self.events.log(event, **fields)
        except Exception:  # pragma: no cover
            pass

    # -- 产物 ----------------------------------------------------------- #
    def bind_state(self, state: dict[str, Any] | None) -> None:
        """把「当前状态」交给 ctx，让每次落盘自动登记产物。

        这解决了一类系统性的记账 bug：阶段各自往 ``state["artifacts"]`` 里追加，
        任何一个阶段忘了做，最终清单就缺项。改为由 ``save_*``/``artifact`` 统一登记，
        阶段代码不再需要关心这件事。
        """
        self.extra["state"] = state

    def artifact(self, p: Path, kind: str = "", stage: str = "") -> Artifact:
        path = Path(p)
        suffix = path.suffix.lower()
        art = Artifact(
            path=self.rel(path),
            kind=kind or _KIND_BY_SUFFIX.get(suffix, "bin"),
            stage=stage,
            sha256=hash_file(path),
            bytes=path.stat().st_size if path.exists() else 0,
        )
        live = self.extra.get("state")
        if isinstance(live, dict):
            try:
                record_artifact(live, art)
            except Exception:  # pragma: no cover
                pass
        return art

    def save_json(self, rel_path: str, obj: Any, stage: str = "", kind: str = "json") -> Artifact:
        target = self.path(rel_path)
        target.write_text(
            json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.artifact(target, kind=kind, stage=stage)

    def save_text(self, rel_path: str, text: str, stage: str = "", kind: str = "md") -> Artifact:
        target = self.path(rel_path)
        target.write_text(text if text is not None else "", encoding="utf-8")
        return self.artifact(target, kind=kind, stage=stage)

    def load_json(self, rel_path: str, default: Any = None) -> Any:
        target = self.run_dir / rel_path
        if not target.exists():
            return default
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    # -- 便捷 ----------------------------------------------------------- #
    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.cfg.run_id,
            "run_dir": self.run_dir.as_posix(),
            "direction": self.cfg.direction,
            "venue": self.cfg.venue,
            "language": self.cfg.language,
            "llm": {
                "provider": self.cfg.llm.provider,
                "model": self.cfg.llm.model,
                "base_url": self.cfg.llm.base_url,
            },
            "sandbox": getattr(self.sandbox, "name", "unknown"),
            "compile_engine": self.cfg.compile.engine,
            "retrieve_sources": list(self.cfg.retrieve.sources),
            "offline": bool(self.cfg.retrieve.offline),
        }


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #


def build_context(
    cfg: AutoResearchConfig,
    run_dir: Path | None = None,
    logger_name: str = "autoresearch",
    quiet: bool = False,
) -> RunContext:
    """装配一份可用的 RunContext。任何组件不可用都降级而不是报错。"""
    cfg.run_id = cfg.run_id or make_run_id(cfg.direction)
    run_dir = Path(run_dir) if run_dir else Path(cfg.runs_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in RUN_SUBDIRS:
        try:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    configure_logging(log_file=run_dir / "logs" / "run.log", quiet=quiet)
    log = get_logger(logger_name)
    events = EventLogger(run_dir)

    # --- LLM --- #
    from .llm.client import LLMClient

    llm = LLMClient(cfg.llm, cache_dir=run_dir / "logs" / "llm_cache", event_logger=events)

    # --- 检索 --- #
    from .tools.retrieve import LiteratureSearch

    cache_dir = cfg.retrieve.cache_dir or (Path(cfg.runs_dir).parent / "cache" / "retrieve")
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = None
    try:
        search = LiteratureSearch(cfg.retrieve, cache_dir=cache_dir, event_logger=events)
    except Exception as exc:  # pragma: no cover - 检索层构造失败也要能继续
        log.warning("LiteratureSearch init failed (%s); using offline stub", exc)
        search = _OfflineSearch(cfg.retrieve)

    # --- 沙箱 --- #
    from .tools.sandbox import make_sandbox

    try:
        sandbox = make_sandbox(cfg.sandbox, run_dir / "experiment", event_logger=events)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"could not create a sandbox: {exc}") from exc

    # --- 编译器 --- #
    from .tools.latex import LatexCompiler

    try:
        compiler = LatexCompiler(cfg.compile, run_dir / "paper", event_logger=events)
    except Exception as exc:  # pragma: no cover
        log.warning("LatexCompiler init failed (%s); compilation will be skipped", exc)
        compiler = _NullCompiler()

    # --- 提示词 --- #
    from .prompts import PromptLibrary

    prompts = PromptLibrary(PROJECT_ROOT / "prompts", language=cfg.language)

    return RunContext(
        cfg=cfg,
        run_dir=run_dir,
        llm=llm,
        events=events,
        log=log,
        sandbox=sandbox,
        search=search,
        compiler=compiler,
        prompts=prompts,
    )


def build_engine(
    ctx: RunContext,
    checkpointer: Checkpointer | None = None,
    stop_on_error: bool = False,
    on_step: Any = None,
) -> GraphEngine:
    """把 ``STAGE_REGISTRY`` 装配成一张可执行图。

    * 阶段类按注册顺序构造实例（实例持有 ctx）；
    * ``s8_review`` 自带 ``route``，会被 ``linear_nodes`` 识别为条件边；
    * ``OPTIONAL_STAGES`` 里的阶段失败后跳过而不是终止。
    """
    instances: dict[str, Any] = {
        name: cls(ctx) for name, cls in STAGE_REGISTRY.items()
    }
    nodes: list[Node] = linear_nodes(instances, optional=set(OPTIONAL_STAGES))
    return GraphEngine(
        nodes,
        checkpoint=checkpointer,
        stop_on_error=stop_on_error,
        event_logger=ctx.events,
        logger=ctx.log,
        on_step=on_step,
    )


# --------------------------------------------------------------------------- #
# 运行
# --------------------------------------------------------------------------- #


def run_pipeline(
    cfg: AutoResearchConfig,
    resume: bool = False,
    force_restart: bool = False,
    quiet: bool = False,
    run_dir: Path | None = None,
    on_step: Any = None,
) -> tuple[dict[str, Any], RunContext, GraphEngine]:
    """跑完整条管线。返回 ``(state, ctx, engine)``。"""
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    # --- 恢复模式：先定位运行目录 --- #
    resumed_state: dict[str, Any] | None = None
    if resume and cfg.run_id and run_dir is None:
        existing = find_run_dir(cfg.runs_dir, cfg.run_id)
        if existing is not None:
            run_dir = existing
            cfg.run_id = existing.name
            probe = Checkpointer(existing)
            resumed_state = probe.load()

    ctx = build_context(cfg, run_dir=run_dir, quiet=quiet)
    ctx.cfg.run_id = ctx.run_dir.name

    state = new_state(
        run_id=ctx.cfg.run_id,
        direction=ctx.cfg.direction,
        venue=ctx.cfg.venue,
        language=ctx.cfg.language,
        seed=ctx.cfg.seed,
    )
    state["started_at"] = started
    ctx.bind_state(state)

    start_at: str | None = None
    if resumed_state is not None:
        resumed_state.setdefault("run_id", ctx.cfg.run_id)
        state = merge_resume(state, resumed_state, force_restart=force_restart)
        ctx.log_event("resume_start", run_id=ctx.cfg.run_id, force_restart=force_restart)
        from .graph.state import first_incomplete_stage

        start_at = first_incomplete_stage(state)
        ctx.log.info("resuming run %s from stage %s", ctx.cfg.run_id, start_at or "(complete)")

    checkpointer = Checkpointer(
        ctx.run_dir,
        run_id=ctx.cfg.run_id,
        event_logger=ctx.events,
        enabled=not bool(getattr(cfg, "dry_run", False)),
    )
    checkpointer.save(state, stage="init")

    engine = build_engine(ctx, checkpointer=checkpointer, on_step=on_step)

    ctx.log_event(
        "run_start",
        run_id=ctx.cfg.run_id,
        direction=ctx.cfg.direction,
        venue=ctx.cfg.venue,
        provider=ctx.cfg.llm.provider,
        model=ctx.cfg.llm.model,
        offline=bool(ctx.cfg.retrieve.offline),
        sandbox=getattr(ctx.sandbox, "name", "unknown"),
        resume=bool(resumed_state is not None),
        start_at=start_at,
    )
    ctx.log.info("run_id=%s dir=%s", ctx.cfg.run_id, ctx.run_dir)
    ctx.log.info("llm=%s/%s offline=%s sandbox=%s",
                 ctx.cfg.llm.provider, ctx.cfg.llm.model,
                 ctx.cfg.retrieve.offline, getattr(ctx.sandbox, "name", "?"))

    t0 = time.monotonic()
    try:
        state = engine.run(state, start_at=start_at if resumed_state is not None else None)
    except KeyboardInterrupt:
        ctx.log_event("run_interrupted", run_id=ctx.cfg.run_id)
        checkpointer.save(state, stage="interrupted")
        ctx.log.warning("interrupted; state saved, resume with: resume %s", ctx.cfg.run_id)
        raise
    except Exception as exc:
        state.setdefault("errors", []).append(f"pipeline crashed: {type(exc).__name__}: {exc}")
        ctx.log_event("run_crash", run_id=ctx.cfg.run_id, error=str(exc))
        ctx.log.exception("pipeline crashed")
    elapsed = time.monotonic() - t0

    state["elapsed_seconds"] = round(elapsed, 2)
    exit_code = compute_exit_code(state)
    state["exit_code"] = exit_code
    checkpointer.save(state, stage="final")

    ctx.log_event(
        "run_end",
        run_id=ctx.cfg.run_id,
        exit_code=exit_code,
        elapsed=round(elapsed, 2),
        stages_done=sum(1 for v in (state.get("stage_status") or {}).values()
                        if v in (STATUS_DONE, STATUS_SKIPPED)),
        final_pdf=str(state.get("final_pdf") or ""),
        review_score=state.get("review_score", 0.0),
        llm_calls=_llm_calls(ctx),
    )
    ctx.log.info("run finished in %.1fs → exit=%d | %s", elapsed, exit_code, summarize_state(state))
    return state, ctx, engine


def compute_exit_code(state: dict[str, Any]) -> int:
    """0 完整 / 1 部分 / 2 致命。"""
    status = state.get("stage_status") or {}
    paper_tex = Path(state.get("paper_tex") or "")
    has_paper = bool(state.get("paper_sections")) or bool(state.get("final_pdf"))
    produced_something = bool(state.get("artifacts")) and has_paper

    fatal_stages = [
        s for s in ("s1_literature", "s2_ideation", "s3_planning", "s4_experiment")
        if status.get(s) == STATUS_FAILED
    ]
    if not produced_something:
        return EXIT_FATAL
    if fatal_stages or any(v == STATUS_FAILED for v in status.values()):
        return EXIT_PARTIAL
    if not state.get("final_pdf"):
        # 论文写出来了但没编译成 PDF：算部分完成（引擎缺失是可接受的环境限制）
        return EXIT_PARTIAL if not state.get("paper_sections") else EXIT_OK
    return EXIT_OK


def _llm_calls(ctx: RunContext) -> int:
    try:
        return int(getattr(ctx.llm.usage(), "calls", 0))
    except Exception:  # pragma: no cover
        return 0


# --------------------------------------------------------------------------- #
# 降级实现（依赖缺失时用，保证管线永不因环境问题崩在初始化）
# --------------------------------------------------------------------------- #


class _OfflineSearch:
    """检索层不可用时的空实现：永远返回空结果，但接口一致。"""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.language = getattr(cfg, "language", "zh")

    def search(self, query: str, max_results: int | None = None, sources: Any = None) -> list[Any]:
        return []

    def multi_search(self, queries: list[str], per_query: int = 6) -> list[Any]:
        return []

    def similar_to(self, paper: Any, max_results: int = 8) -> list[Any]:
        return []

    def to_bibtex(self, papers: list[Any]) -> str:
        return ""

    def format_review(self, papers: list[Any], max_papers: int = 12) -> str:
        return "_检索层不可用（构造失败）。_"


class _NullCompiler:
    """编译器不可用时的空实现：报告 blocked 而不是抛异常。"""

    def __init__(self) -> None:
        self.name = "null"

    def detect(self) -> str | None:
        return None

    def install_tectonic(self) -> None:
        return None

    def bibtex_available(self) -> bool:
        return False

    def extract_errors(self, log: str) -> list[str]:
        return []

    def compile(self, tex_file: Path, runs: int = 2, timeout: int = 300, engine: str | None = None):
        from .tools.latex import CompileResult

        return CompileResult(
            ok=False,
            pdf=None,
            log="LatexCompiler unavailable in this environment",
            errors=["LatexCompiler unavailable (import/构造失败)"],
            warnings=[],
            engine="none",
            duration=0.0,
        )


__all__ = [
    "EXIT_FATAL",
    "EXIT_OK",
    "EXIT_PARTIAL",
    "RUN_SUBDIRS",
    "RunContext",
    "build_context",
    "build_engine",
    "compute_exit_code",
    "make_run_id",
    "run_pipeline",
    "slugify",
]
