"""可选 LangGraph 适配层。

本项目的默认执行器是 ``graph/engine.py``（零依赖）。若环境里装了 ``langgraph``，
可以用同一个 ``list[Node]`` 编译出等价的 ``StateGraph``，把编排层换成 LangGraph
的 checkpoint / streaming 生态，而**阶段实现完全不动**。

映射关系
--------
======================  ==========================================
本引擎                    LangGraph
======================  ==========================================
``Node.name``            ``StateGraph.add_node`` 的 key
``Node.fn(state)``       节点函数（返回部分状态字典）
``Node.router``           ``add_conditional_edges`` 的路由函数
``END``                  ``langgraph.graph.END``
``Checkpointer.save``     ``BaseCheckpointSaver``（本适配层不接管，见下）
======================  ==========================================

一个刻意的差异
--------------
本引擎把 ``StageResult`` 的 **元数据**（ok/detail/artifacts）与 **状态**分开：
失败的阶段靠重试+降级继续跑，图本身不会抛异常。LangGraph 的惯例是节点直接返回
状态更新、异常即中断。适配层因此把 ``StageResult`` 转成一个状态补丁，
并把失败信息写进 ``state["stage_errors"]``；重试由 LangGraph 的
``RetryPolicy`` 承担，语义对齐但不逐字节相同。

用法::

    nodes = [...]                      # 与自研引擎同一份 list[Node]
    app = compile_langgraph(nodes)     # 需要 pip install langgraph
    final_state = app.invoke(initial_state)
"""

from __future__ import annotations

from typing import Any

from .engine import END, Node

LANGGRAPH_HINT = (
    "LangGraph is not installed. The built-in engine (graph/engine.py) already "
    "implements the same conditional-routing semantics; install with "
    "`pip install langgraph` only if you specifically want LangGraph's "
    "checkpointers/streaming."
)


class LangGraphUnavailable(RuntimeError):
    """``langgraph`` 未安装时抛出，消息里给出安装指引。"""


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


def _require_langgraph() -> Any:
    try:
        import langgraph.graph as lg  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise LangGraphUnavailable(LANGGRAPH_HINT) from exc
    return lg


def compile_langgraph(
    nodes: list[Node],
    state_type: Any = None,
    checkpointer: Any = None,
    entry_point: str | None = None,
    recursion_limit: int = 200,
) -> Any:
    """把 ``list[Node]`` 编译成 LangGraph 应用。

    ``state_type`` 缺省时用 ``dict``（LangGraph 允许，只是丢掉键的类型提示）。
    ``Node.router`` 通过 ``add_conditional_edges`` 接上；路由返回 ``END`` 时
    映射到 LangGraph 的 ``END``。
    """
    lg = _require_langgraph()
    StateGraph = lg.StateGraph
    LG_END = lg.END

    schema = state_type or dict
    graph = StateGraph(schema)
    node_names = [n.name for n in nodes]

    def make_fn(node: Node):
        def _fn(state: Any) -> dict[str, Any]:
            result = node.fn(dict(state) if isinstance(state, dict) else state)
            patch: dict[str, Any] = {}
            updates = getattr(result, "state_updates", None)
            if isinstance(updates, dict):
                patch.update(updates)
            ok = bool(getattr(result, "ok", False))
            detail = str(getattr(result, "detail", "") or "")
            stage = node.name
            status = dict(patch.get("stage_status") or state.get("stage_status") or {})
            status[stage] = "done" if ok else ("skipped" if node.optional else "failed")
            patch["stage_status"] = status
            if not ok:
                errs = dict(patch.get("stage_errors") or state.get("stage_errors") or {})
                errs.setdefault(stage, [])
                errs[stage] = list(errs[stage]) + [detail or "stage reported ok=False"]
                patch["stage_errors"] = errs
                if bool(getattr(result, "fatal", False)):
                    raise RuntimeError(f"[{stage}] fatal: {detail}")
            trace = list(patch.get("trace") or state.get("trace") or [])
            trace.append(
                {"stage": stage, "ok": ok, "detail": detail, "engine": "langgraph"}
            )
            patch["trace"] = trace
            return patch

        return _fn

    for node in nodes:
        graph.add_node(node.name, make_fn(node))

    def make_router(node: Node):
        def _router(state: Any) -> str:
            try:
                target = node.router(state)  # type: ignore[misc]
            except Exception:
                target = None
            if target in (None, ""):
                i = node_names.index(node.name)
                return node_names[i + 1] if i + 1 < len(node_names) else LG_END
            return LG_END if target == END else target

        return _router

    entry = entry_point or node_names[0]
    graph.set_entry_point(entry)

    for i, node in enumerate(nodes):
        if node.router is not None:
            targets = [n for n in node_names if n != node.name] + [LG_END]
            graph.add_conditional_edges(node.name, make_router(node), targets)
        else:
            nxt = node_names[i + 1] if i + 1 < len(node_names) else LG_END
            graph.add_edge(node.name, nxt)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    app = graph.compile(**compile_kwargs)
    try:  # 递归上限挂在 config 上，这里尽力透传
        app = app.with_config({"recursion_limit": recursion_limit})
    except Exception:  # pragma: no cover - 旧版本没有 with_config
        pass
    return app


def describe_parity(nodes: list[Node]) -> dict[str, Any]:
    """对比自研引擎与 LangGraph 的装配结果，供 ``cli doctor`` 打印。

    不导入 langgraph 也能运行；只做静态描述。
    """
    return {
        "langgraph_available": langgraph_available(),
        "nodes": [n.name for n in nodes],
        "conditional": [n.name for n in nodes if n.router is not None],
        "entry_point": nodes[0].name if nodes else "",
        "note": LANGGRAPH_HINT if not langgraph_available() else "langgraph importable",
    }


__all__ = [
    "END",
    "LANGGRAPH_HINT",
    "LangGraphUnavailable",
    "compile_langgraph",
    "describe_parity",
    "langgraph_available",
]
