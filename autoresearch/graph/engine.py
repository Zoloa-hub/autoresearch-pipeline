"""自研轻量状态机引擎。

为什么不用现成框架
------------------
LangGraph 的能力集是「带条件的循环图 + checkpoint」。这套语义的核心其实很小：
**节点、边、条件路由、重试、检查点**。用标准库实现它大约 250 行，换来的是：

* 零依赖 —— 装不上 ``langgraph`` 的机器也能跑（本项目的硬性要求之一）；
* 状态是纯 ``dict`` —— 断点续跑只需 ``json.dump``；
* 与 LangGraph 一一对应 —— ``graph/langgraph_adapter.py`` 提供编译层，
  环境里有 ``langgraph`` 时可切换到它，语义不变。

执行语义（与 LangGraph 保持一致的子集）
--------------------------------------
1. 节点按 ``nodes`` 声明顺序推进；带 ``router`` 的节点返回下一步名称即可改变流向。
2. ``router`` 返回 ``None`` → 按声明顺序继续；``"__end__"`` → 立即结束。
3. 节点抛异常或返回 ``ok=False`` → 在 ``max_attempts`` 内重试。
4. 重试耗尽后：``optional=True`` 记 ``skipped`` 继续；否则记 ``failed``；
   若 ``stop_on_error=True`` 则终止，否则**继续后续节点**——
   宁可产出部分结果，也不要因为一个阶段崩掉而浪费前面几十分钟的 LLM 成本。
5. 每步之后落一次检查点。
6. 环路保护：单节点访问次数上限 + 全局步数上限，两者都超出即终止并记事件。
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .state import (
    STAGES,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    mark_stage,
    to_jsonable,
)

END = "__end__"

#: 单个节点在一次运行中允许被访问的最大次数（支撑「评审→改写→重评」这类环路）。
DEFAULT_MAX_VISITS = 4


@dataclass
class Node:
    """图中的一个阶段。

    ``fn(state) -> StageResult``；``StageResult`` 由 ``stages.base`` 定义，
    但引擎只依赖它的 ``ok / state_updates / detail / fatal / retry`` 四个属性，
    因此这里用鸭子类型，避免 ``graph`` 反向依赖 ``stages``。
    """

    name: str
    fn: Callable[[dict[str, Any]], Any]
    max_attempts: int = 2
    optional: bool = False
    router: Callable[[dict[str, Any]], str | None] | None = None
    title: str = ""


@dataclass
class StepRecord:
    stage: str
    ok: bool
    elapsed: float
    attempt: int
    detail: str = ""
    status: str = ""
    error: str = ""
    next_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "elapsed": round(self.elapsed, 3),
            "attempt": self.attempt,
            "detail": self.detail,
            "status": self.status,
            "error": self.error,
            "next_stage": self.next_stage,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


class GraphError(RuntimeError):
    pass


class GraphEngine:
    """顺序 + 条件路由的阶段执行器。"""

    def __init__(
        self,
        nodes: list[Node],
        checkpoint: Any = None,
        max_steps: int = 200,
        stop_on_error: bool = False,
        event_logger: Any = None,
        logger: Any = None,
        max_visits: int = DEFAULT_MAX_VISITS,
        on_step: Callable[[StepRecord, dict[str, Any]], None] | None = None,
    ) -> None:
        if not nodes:
            raise GraphError("GraphEngine requires at least one node")
        names = [n.name for n in nodes]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise GraphError(f"duplicate node names: {dupes}")
        self.nodes = list(nodes)
        self.node_map = {n.name: n for n in self.nodes}
        self.order = {n.name: i for i, n in enumerate(self.nodes)}
        self.checkpoint = checkpoint
        self.max_steps = int(max_steps)
        self.stop_on_error = bool(stop_on_error)
        self.events = event_logger
        self.log = logger
        self.max_visits = max(1, int(max_visits))
        self.on_step = on_step
        self._trace: list[StepRecord] = []
        self._visits: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    def _log(self, event: str, **fields: Any) -> None:
        fn = getattr(self.events, "log", None)
        if callable(fn):
            try:
                fn(event, **fields)
            except Exception:  # pragma: no cover
                pass

    def _info(self, msg: str, *args: Any) -> None:
        if self.log is not None:
            try:
                self.log.info(msg, *args)
            except Exception:  # pragma: no cover
                pass

    def _warning(self, msg: str, *args: Any) -> None:
        if self.log is not None:
            try:
                self.log.warning(msg, *args)
            except Exception:  # pragma: no cover
                pass

    def trace(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._trace]

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self, state: dict[str, Any], start_at: str | None = None) -> dict[str, Any]:
        """执行图。``start_at`` 用于断点续跑——从该节点开始，跳过之前的节点。"""
        if start_at and start_at not in self.node_map:
            available = ", ".join(self.node_map)
            raise GraphError(f"unknown start node {start_at!r}; available: {available}")

        idx = self.order[start_at] if start_at else 0
        steps = 0
        current: str | None = self.nodes[idx].name

        while current is not None and current != END:
            if steps >= self.max_steps:
                msg = f"max_steps={self.max_steps} reached; aborting graph"
                self._warning(msg)
                state.setdefault("errors", []).append(msg)
                self._log("graph_abort", reason="max_steps", steps=steps)
                break

            node = self.node_map.get(current)
            if node is None:
                msg = f"router pointed at unknown node {current!r}; aborting"
                self._warning(msg)
                state.setdefault("errors", []).append(msg)
                self._log("graph_abort", reason="unknown_node", node=current)
                break

            self._visits[current] = self._visits.get(current, 0) + 1
            if self._visits[current] > self.max_visits:
                msg = (
                    f"node {current!r} visited {self._visits[current] - 1} times "
                    f"(limit {self.max_visits}); treating as done to break the cycle"
                )
                self._warning(msg)
                state.setdefault("warnings", []).append(msg)
                state.setdefault("stage_status", {})[current] = STATUS_DONE
                self._log("graph_cycle_break", node=current, visits=self._visits[current] - 1)
                current = self._next_after(node, state, forced_next=None)
                steps += 1
                continue

            steps += 1
            state["steps"] = steps
            state["current_stage"] = current
            next_name = self._run_node(node, state)
            steps_snapshot = state.get("steps", steps)

            if self.checkpoint is not None:
                try:
                    self.checkpoint.save(state, stage=current)
                except Exception as exc:  # pragma: no cover - 检查点失败不致命
                    self._warning("checkpoint save failed: %s", exc)

            if next_name == END:
                current = END
            elif next_name is not None:
                current = next_name
            else:
                current = self._next_after(node, state, forced_next=None)

            # 记录下一步，便于审计
            if self._trace:
                self._trace[-1].next_stage = current or END
            state["steps"] = max(steps_snapshot, steps)

        state["current_stage"] = ""
        return state

    def _next_after(self, node: Node, state: dict[str, Any], forced_next: str | None) -> str | None:
        if forced_next is not None:
            return None if forced_next == END else forced_next
        i = self.order[node.name]
        if i + 1 >= len(self.nodes):
            return None
        return self.nodes[i + 1].name

    # ------------------------------------------------------------------ #
    # 单节点执行（含重试）
    # ------------------------------------------------------------------ #
    def _run_node(self, node: Node, state: dict[str, Any]) -> str | None:
        attempts = max(1, int(node.max_attempts))
        status_map = state.setdefault("stage_status", {})
        attempt_map = state.setdefault("stage_attempts", {})
        last_error = ""

        for attempt in range(1, attempts + 1):
            attempts_used = int(attempt_map.get(node.name, 0)) + 1
            attempt_map[node.name] = attempts_used
            status_map[node.name] = STATUS_RUNNING

            self._log("stage_start", stage=node.name, attempt=attempt, title=node.title or node.name)
            self._info("▶ %s (attempt %d/%d)", node.name, attempt, attempts)
            t0 = time.monotonic()
            result: Any = None
            error = ""

            try:
                result = node.fn(state)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc(limit=8)
                self._warning("stage %s raised: %s", node.name, error)
                state.setdefault("errors", []).append(f"[{node.name}] {error}")
                self._log("stage_exception", stage=node.name, error=error, traceback=tb[-2000:])

            elapsed = time.monotonic() - t0
            ok = bool(getattr(result, "ok", False)) if result is not None else False
            detail = str(getattr(result, "detail", "") or "")
            fatal = bool(getattr(result, "fatal", False))
            wants_retry = bool(getattr(result, "retry", False))

            # 合并状态更新（拿到 result 就先合并，好让 router 看到新状态）
            if result is not None:
                updates = getattr(result, "state_updates", None)
                if isinstance(updates, dict):
                    try:
                        for k, v in updates.items():
                            state[k] = to_jsonable(v)
                    except Exception as exc:  # pragma: no cover
                        self._warning("state merge failed for %s: %s", node.name, exc)

            record = StepRecord(
                stage=node.name,
                ok=ok,
                elapsed=elapsed,
                attempt=attempt,
                detail=detail,
                status=STATUS_DONE if ok else STATUS_FAILED,
                error=error,
            )
            self._trace.append(record)
            state.setdefault("trace", []).append(record.to_dict())

            if ok:
                status_map[node.name] = STATUS_DONE
                record.status = STATUS_DONE
                self._info("✔ %s in %.1fs %s", node.name, elapsed, detail[:120])
                self._log(
                    "stage_done", stage=node.name, elapsed=round(elapsed, 3),
                    detail=detail, attempt=attempt,
                )
                self._emit_step(record, state)
                return self._route(node, state)

            last_error = error or detail or "stage reported ok=False"
            mark_stage(state, node.name, STATUS_FAILED, last_error)

            retryable = (wants_retry or attempt < attempts) and not fatal
            if retryable:
                self._warning(
                    "stage %s failed (attempt %d/%d): %s — retrying",
                    node.name, attempt, attempts, last_error[:200],
                )
                self._log("stage_retry", stage=node.name, attempt=attempt, error=last_error[:500])
                if self.checkpoint is not None:
                    try:
                        self.checkpoint.save(state, stage=node.name)
                    except Exception:  # pragma: no cover
                        pass
                continue

            # 重试耗尽
            final_status = STATUS_SKIPPED if node.optional else STATUS_FAILED
            status_map[node.name] = final_status
            record.status = final_status
            event = "stage_skipped" if node.optional else "stage_failed"
            self._log(event, stage=node.name, error=last_error[:500], attempt=attempt)
            if node.optional:
                self._warning("⚠ %s skipped (optional): %s", node.name, last_error[:200])
                state.setdefault("warnings", []).append(f"[{node.name}] skipped: {last_error}")
                self._emit_step(record, state)
                return self._route(node, state)

            self._warning("✘ %s failed: %s", node.name, last_error[:200])
            self._emit_step(record, state)
            if self.stop_on_error or fatal:
                self._log("graph_abort", reason="stage_failed", node=node.name)
                return END
            return self._route(node, state)

        # 理论不可达
        return self._route(node, state)

    def _emit_step(self, record: StepRecord, state: dict[str, Any]) -> None:
        if self.on_step is None:
            return
        try:
            self.on_step(record, state)
        except Exception:  # pragma: no cover
            pass

    def _route(self, node: Node, state: dict[str, Any]) -> str | None:
        if node.router is None:
            return None
        try:
            target = node.router(state)
        except Exception as exc:
            self._warning("router for %s raised: %s — falling through", node.name, exc)
            self._log("router_error", stage=node.name, error=f"{type(exc).__name__}: {exc}")
            return None
        if target in (None, "", END):
            return END if target == END else None
        if target == node.name:
            # 自环：等价于「重来一次」，交给访问计数兜底
            return target
        if target not in self.node_map:
            self._warning("router for %s returned unknown target %r — ignoring", node.name, target)
            self._log("router_unknown_target", stage=node.name, target=str(target))
            return None
        self._log("route", stage=node.name, target=target)
        return target


# --------------------------------------------------------------------------- #
# 便捷构造：按 STAGES 顺序建一条默认线性链
# --------------------------------------------------------------------------- #


def linear_nodes(
    stage_impls: dict[str, Any],
    default_attempts: int = 2,
    optional: set[str] | None = None,
    routers: dict[str, Callable[[dict[str, Any]], str | None]] | None = None,
) -> list[Node]:
    """把 ``{stage_name: Stage}`` 装配成 ``list[Node]``。

    ``Stage`` 需暴露 ``name / title / max_attempts`` 与可调用的 ``run(state)``。
    """
    optional = optional or set()
    routers = routers or {}
    nodes: list[Node] = []
    for name in STAGES:
        impl = stage_impls.get(name)
        if impl is None:
            continue
        run_fn = getattr(impl, "run")
        title = getattr(impl, "title", "") or name
        nodes.append(
            Node(
                name=name,
                fn=run_fn,
                max_attempts=int(getattr(impl, "max_attempts", default_attempts) or default_attempts),
                optional=name in optional,
                router=routers.get(name) or getattr(impl, "router", None),
                title=title,
            )
        )
    return nodes
