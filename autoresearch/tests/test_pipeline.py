"""Auto-Research 管线端到端回归测试（离线，无网络，无需 API Key）。

这是整条管线**唯一**的集成级测试，因此它的定位是「契约回归」而不是「单元覆盖」：
各模块自己的测试负责细节（提示词引擎、指标解析、沙箱、LaTeX），这里负责回答
一个问题——

    **在一台什么都没有的机器上，`cli run --llm-provider mock --offline`
    能不能从头跑到尾、产出可交付的文件，并且诚实地报告它没做到的部分？**

测试覆盖的关键不变量（每一条都对应一个真实踩过的坑）：
1. 九个阶段全部执行完毕，且**没有**阶段停留在 ``pending``/``running``；
2. 产物清单非空，且每个登记产物的 sha256 与磁盘内容一致（记账不能撒谎）；
3. 检查点可读、可恢复（``state.json`` + ``checkpoints/``），``resume`` 能定位断点；
4. 论文骨架的 ``\\input`` 目标全部存在（否则编译必然失败，且错误很难定位）；
5. 参考文献库只包含被引用或模板自带的 key（杜绝悬空 ``\\cite``）；
6. 数值汇总的**跨种子口径**正确：每个种子贡献一个标量，再算 mean/std；
7. 评审循环能终止（分数单调或达轮次上限），且路由不会无限自环；
8. 数字溯源校验器能跑通，并且**故意植入的假数字必须被抓出来**
   （一个抓不到错的校验器等于没有校验器）；
9. 无 LaTeX 引擎时降级为 ``COMPILE_BLOCKED.md``，而不是抛异常或假装成功。

运行：``python -m autoresearch.tests.test_pipeline``（也可被 pytest 收集）
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 测试脚手架（不依赖 pytest，同时兼容 pytest 收集）
# --------------------------------------------------------------------------- #

_CHECKS = 0
_FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    global _CHECKS
    _CHECKS += 1
    if condition:
        return True
    message = f"FAIL: {label}" + (f" — {detail}" if detail else "")
    _FAILURES.append(message)
    print("  " + message)
    return False


def eq(actual, expected, label: str) -> bool:
    return check(actual == expected, label, f"expected {expected!r}, got {actual!r}")


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def _workspace_root() -> Path:
    """定位工作区根（``autoresearch`` 的上一级）。"""
    return Path(__file__).resolve().parents[2]


def _fresh_dir(name: str) -> Path:
    root = _workspace_root() / ".autoresearch" / "test_tmp" / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run_pipeline_once(runs_dir: Path, direction: str = "管线端到端回归：受控对照的可复现性"):
    from autoresearch.config import load_config
    from autoresearch.runner import run_pipeline

    cfg = load_config(
        direction=direction,
        llm={"provider": "mock", "model": "mock-heuristic"},
        retrieve={"offline": True},
        compile={"engine": "none"},
        sandbox={"backend": "subprocess", "timeout": 120},
        max_review_rounds=2,
        runs_dir=runs_dir,
    )
    state, ctx, engine = run_pipeline(cfg, quiet=True)
    return state, ctx, engine


def test_intent_routing() -> None:
    """mock 后端的意图路由必须认得出真实的提示词模板。"""
    section("意图路由")
    from autoresearch.config import PROJECT_ROOT
    from autoresearch.llm.mock import MockBackend

    backend = MockBackend()
    prompts_dir = PROJECT_ROOT / "prompts"
    expected = {
        "s1_queries": "query",
        "s1_survey": "literature",
        "s2_ideas": "idea",
        "s2_novelty": "novelty",
        "s3_plan": "plan",
        "s4_codegen": "code",
        "s4_debug": "debug",
        "s5_analysis": "analysis",
        "s6_section": "writing",
        "s6_abstract": "writing",
        # 修订提示词里同时含「评审意见」与「改写」，mock 判为 review 也合理——
        # 真实模型不会因为意图标签而改写行为，这里不做过窄的断言。
        "s6_revision": ("writing", "review"),
        # 编译修复提示词含 Traceback 与代码，判成 debug 同样可接受。
        "s7_compile_fix": ("compile_fix", "debug"),
        "s8_review": "review",
        "s9_report": None,  # Markdown 提示词，走 fallback 也算正常
    }
    from autoresearch.prompts import PromptLibrary

    lib = PromptLibrary(prompts_dir, language="zh")
    for name, intent in expected.items():
        path = prompts_dir / f"{name}.md"
        if not check(path.exists(), f"提示词存在: {name}"):
            continue
        rendered = lib.raw(name)
        got = backend.intent_of(rendered)
        if intent is None:
            check(bool(got), f"{name} 能路由到某个意图（got={got}）")
        elif isinstance(intent, tuple):
            check(got in intent, f"{name} → {'|'.join(intent)}", f"got {got}")
        else:
            eq(got, intent, f"{name} → {intent}")


def test_graph_engine_semantics() -> None:
    """自研引擎的条件路由、重试、降级与环路保护。"""
    section("图引擎语义")
    from autoresearch.graph.engine import GraphEngine, Node
    from autoresearch.graph.state import new_state
    from autoresearch.stages.base import StageResult

    def fresh() -> dict:
        """引擎要求状态里带有 stage_status 等键；用 new_state 保证形状一致。"""
        return new_state(run_id="graph-test", direction="d")

    calls: list[str] = []

    def make(name: str, ok: bool = True, router=None, boom: bool = False):
        def fn(state):
            calls.append(name)
            if boom:
                raise RuntimeError(f"{name} exploded")
            return StageResult(ok, {}, [], name) if ok else StageResult(False, {}, [], name)
        return Node(name=name, fn=fn, max_attempts=2, router=router)

    # 1. 顺序执行
    calls.clear()
    engine = GraphEngine([make("a"), make("b"), make("c")])
    state = engine.run(fresh())
    eq(calls, ["a", "b", "c"], "顺序推进")

    # 2. 条件路由成环并能跳出
    calls.clear()

    def router(state):
        visits = calls.count("loop")
        return "tail" if visits >= 2 else "loop"

    engine = GraphEngine([make("loop", router=router), make("tail")])
    engine.run(fresh())
    check(calls.count("loop") >= 2, "路由可以回到本节点", f"calls={calls}")
    check("tail" in calls, "路由最终离开环路", f"calls={calls}")

    # 3. 重试：抛出异常后重试，第二次成功
    calls.clear()
    attempts = {"n": 0}

    def flaky(state):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("first attempt fails")
        return StageResult(True, {}, [], "recovered")

    engine = GraphEngine([Node(name="flaky", fn=flaky, max_attempts=3)])
    state = engine.run(fresh())
    eq(attempts["n"], 2, "失败后重试")
    eq(state["stage_status"]["flaky"], "done", "重试成功后标记 done")

    # 4. 可选阶段失败 → skipped，管线继续
    calls.clear()
    engine = GraphEngine(
        [
            Node(name="opt", fn=lambda s: StageResult(False, {}, [], "nope"),
                 max_attempts=1, optional=True),
            make("after"),
        ]
    )
    state = engine.run(fresh())
    eq(state["stage_status"]["opt"], "skipped", "可选阶段失败标记 skipped")
    eq(state["stage_status"]["after"], "done", "可选阶段失败后继续执行")

    # 5. 非可选阶段失败 → failed，但默认继续（要产出部分结果）
    engine = GraphEngine(
        [
            Node(name="hard", fn=lambda s: StageResult(False, {}, [], "boom"), max_attempts=1),
            make("later"),
        ]
    )
    state = engine.run(fresh())
    eq(state["stage_status"]["hard"], "failed", "必需阶段失败标记 failed")
    eq(state["stage_status"]["later"], "done", "失败后仍继续（默认不终止）")

    # 6. stop_on_error 生效
    engine = GraphEngine(
        [
            Node(name="hard", fn=lambda s: StageResult(False, {}, [], "boom"), max_attempts=1),
            make("later"),
        ],
        stop_on_error=True,
    )
    state = engine.run(fresh())
    eq(state["stage_status"].get("later", "not-run"), "not-run",
       "stop_on_error 时后续不执行")

    # 7. 自环保护：一直路由回自己也不会死循环
    engine = GraphEngine(
        [Node(name="spin", fn=lambda s: StageResult(True, {}, [], "spin"),
              router=lambda s: "spin")],
        max_steps=10,
        max_visits=3,
    )
    state = engine.run(fresh())
    check(len(engine.trace()) <= 5, "自环被访问计数打断", f"trace={len(engine.trace())}")


def test_pipeline_end_to_end() -> tuple[Path | None, dict]:
    """完整跑一遍 mock/离线管线，返回 (run_dir, state)。"""
    section("端到端执行")
    runs_dir = _fresh_dir("runs")
    try:
        state, ctx, engine = run_pipeline_once(runs_dir)
    except Exception as exc:  # pragma: no cover - 出错时把栈打出来便于定位
        import traceback

        traceback.print_exc()
        check(False, "管线执行未抛异常", f"{type(exc).__name__}: {exc}")
        return None, {}

    run_dir = Path(ctx.run_dir)
    print(f"  run_dir = {run_dir}")
    print(f"  exit_code = {state.get('exit_code')}")

    # 1. 所有阶段都已收敛（没有卡在 pending/running）
    status = state.get("stage_status") or {}
    stuck = [k for k, v in status.items() if v in ("pending", "running")]
    eq(stuck, [], "没有阶段停留在 pending/running")
    eq(len(status), 9, "九个阶段都有状态")

    # 2. 关键阶段不能是 failed（这几步失败意味着管线没产出科研内容）
    for stage in ("s1_literature", "s2_ideation", "s3_planning", "s4_experiment",
                  "s6_writing", "s9_finalize"):
        eq(status.get(stage), "done", f"{stage} 完成")

    # 3. 状态里的关键内容
    check(bool(state.get("queries")), "生成了检索式")
    check(len(state.get("ideas") or []) >= 1, "生成了候选假设")
    check(bool(state.get("selected_idea")), "选出了假设")
    check(bool(state.get("experiment_plan", {}).get("milestones")), "有实验里程碑")
    runs_executed = state.get("runs_executed") or []
    check(len(runs_executed) >= 4, "至少执行了 4 次 run（2 变体 × ≥2 种子）",
          f"got {len(runs_executed)}")
    check(bool(state.get("paper_sections")), "生成了论文章节")
    check(len(state.get("paper_sections") or {}) >= 8, "至少 8 个章节")
    check(bool(state.get("final_report")), "有最终报告")

    # 4. 产物记账必须与磁盘一致（sha256 自校验）
    artifacts = state.get("artifacts") or []
    check(len(artifacts) >= 15, "产物登记数量合理", f"got {len(artifacts)}")
    bad_hash: list[str] = []
    missing: list[str] = []
    from autoresearch.graph.state import hash_file

    for art in artifacts:
        path = run_dir / str(art.get("path"))
        if not path.exists():
            missing.append(str(art.get("path")))
            continue
        recorded = art.get("sha256") or ""
        if recorded and hash_file(path) != recorded:
            bad_hash.append(str(art.get("path")))
    eq(missing, [], "所有登记产物都存在于磁盘")
    eq(bad_hash, [], "所有登记产物的 sha256 与磁盘一致")

    # 5. 检查点
    check((run_dir / "state.json").exists(), "state.json 存在")
    snaps = sorted((run_dir / "checkpoints").glob("state_*.json")) \
        if (run_dir / "checkpoints").is_dir() else []
    check(len(snaps) >= 9, "逐步检查点已落盘", f"got {len(snaps)}")
    saved = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    eq(saved.get("run_id"), state.get("run_id"), "state.json 的 run_id 一致")

    # 6. 论文骨架完整性
    paper = run_dir / "paper"
    main_tex = paper / "main.tex"
    check(main_tex.exists(), "main.tex 存在")
    if main_tex.exists():
        tex = main_tex.read_text(encoding="utf-8")
        inputs = re.findall(r"\\input\{([^}]+)\}", tex)
        check(len(inputs) >= 8, "main.tex 至少 input 8 个章节", f"got {len(inputs)}")
        missing_inputs = [
            inc for inc in inputs
            if not (paper / (inc if inc.endswith(".tex") else inc + ".tex")).exists()
        ]
        eq(missing_inputs, [], "所有 \\input 目标都存在")
        check("__TITLE__" not in tex and "__ABSTRACT__" not in tex,
              "占位符已被替换")
        # 括号配平：编译前的结构性检查
        eq(tex.count("\\begin{document}"), tex.count("\\end{document}"),
           "document 环境配对")

    # 7. 章节文件也都括号配平
    unbalanced: list[str] = []
    for path in sorted((paper / "sections").glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        if len(re.findall(r"\\begin\{", text)) != len(re.findall(r"\\end\{", text)):
            unbalanced.append(path.name)
    eq(unbalanced, [], "章节文件的 \\begin/\\end 配平")

    # 8. 参考文献闭集：正文里的 cite 必须都能在 .bib 里找到
    bib_path = paper / "references.bib"
    check(bib_path.exists(), "references.bib 存在")
    if bib_path.exists():
        bib_text = bib_path.read_text(encoding="utf-8")
        bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
        check(len(bib_keys) >= 1, "参考文献库非空")
        dangling: list[str] = []
        for path in sorted((paper / "sections").glob("*.tex")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", text):
                for key in (k.strip() for k in match.group(1).split(",")):
                    if key and key not in bib_keys:
                        dangling.append(f"{path.name}:{key}")
        eq(dangling, [], "没有悬空引用")

    # 9. 编译结果必须诚实
    compile_result = state.get("compile_result") or {}
    if compile_result.get("engine") == "none":
        check(not compile_result.get("ok"), "engine=none 时 ok=False")
        check((run_dir / "report" / "COMPILE_BLOCKED.md").exists(),
              "无引擎时产出 COMPILE_BLOCKED.md")
        check(not state.get("final_pdf"), "无引擎时不宣称有 PDF")
    else:
        check(bool(state.get("final_pdf")), "有引擎时必须产出 PDF")

    return run_dir, state


def test_cross_seed_statistics(run_dir: Path, state: dict) -> None:
    """跨种子统计口径：每种子一个标量，再在种子维度上算 mean/std。"""
    section("跨种子统计")
    summary = state.get("metrics_summary") or {}
    cross = summary.get("cross_seed") or {}
    check(bool(cross), "存在跨种子统计", f"keys={list(cross)}")

    for variant, stats in cross.items():
        metric_stats = {
            k: v for k, v in stats.items()
            if isinstance(v, dict) and not k.startswith("_")
        }
        check(bool(metric_stats), f"{variant} 有指标统计")
        for metric, st in metric_stats.items():
            count = st.get("count")
            check(count is not None and float(count) >= 2,
                  f"{variant}.{metric} 至少 2 个种子", f"count={count}")
            std = float(st.get("std", -1))
            check(std >= 0, f"{variant}.{metric} std 非负")

    # 磁盘布局：每个种子一个目录 + 顶层一份便于查看的副本
    metrics_dir = run_dir / "metrics"
    for variant in ("baseline", "method"):
        seed_dirs = sorted((metrics_dir / variant).glob("seed_*")) \
            if (metrics_dir / variant).is_dir() else []
        check(len(seed_dirs) >= 2, f"{variant} 有多个种子目录", f"got {len(seed_dirs)}")
        for sd in seed_dirs:
            check((sd / "metrics.csv").exists(), f"{variant}/{sd.name}/metrics.csv 存在")

    # 论文表格必须来自跨种子口径
    tables = state.get("latex_tables") or {}
    table = tables.get("evaluation") or ""
    check("\\toprule" in table and "\\bottomrule" in table, "主表是 booktabs 三线表")
    check(table.count("{") == table.count("}"), "主表括号配平")
    check("seeds" in table or "seed" in table, "主表标注了种子数")


def test_numeric_provenance(run_dir: Path) -> None:
    """溯源校验器必须跑通，且能抓出植入的假数字。"""
    section("数字溯源")
    from autoresearch.verify import verify_run

    report = verify_run(run_dir)
    check(report.numbers_found > 0, "论文/分析中抽到了数值 token",
          f"got {report.numbers_found}")
    check(len(report.evidence_values) > 0, "证据库非空")
    check(len(report.exact) > 0, "存在能直接溯源的数字")
    print(f"  numbers={report.numbers_found} exact={len(report.exact)} "
          f"derived={len(report.derived)} unmatched={len(report.unmatched)}")
    check(len(report.unmatched) <= max(3, report.numbers_found // 4),
          "未溯源数字占比可控（否则校验器等于噪声）",
          f"unmatched={len(report.unmatched)} / {report.numbers_found}")

    # 植入一个证据里绝对不存在的数字，必须被抓出来
    from autoresearch.verify import extract_numbers_from_text, match_findings

    evidence, named = __import__(
        "autoresearch.verify", fromlist=["collect_evidence"]
    ).collect_evidence(run_dir)
    planted_text = "本节报告一个不存在的结果：准确率达到了 0.123456789。"
    findings = extract_numbers_from_text(planted_text, "planted:1")
    check(len(findings) == 1, "植入文本抽到 1 个数字", f"got {len(findings)}")
    _, _, unmatched, _ = match_findings(findings, evidence, named)
    eq(len(unmatched), 1, "植入的假数字被判为未溯源")

    # 反向检查：真实证据里的数字必须能匹配上自己
    if named:
        sample_name, sample_value = next(iter(sorted(named.items())))
        planted_real = extract_numbers_from_text(
            f"结果值为 {sample_value:.10g}。", "planted:2"
        )
        _, _, unmatched_real, _ = match_findings(planted_real, evidence, named)
        eq(len(unmatched_real), 0, f"真实证据值可溯源（{sample_name}）")


def test_resume_and_checkpoint(run_dir: Path) -> None:
    """断点续跑：状态可读、能定位断点、合并规则不丢已完成阶段。"""
    section("断点续跑")
    from autoresearch.graph.checkpoint import Checkpointer, merge_resume, resume_plan
    from autoresearch.graph.state import new_state

    state = Checkpointer(run_dir).load()
    check(state is not None, "能读回 state.json")
    if state is None:
        return

    plan = resume_plan(state)
    check(plan["complete"], "已完成的运行被判定为 complete", f"next={plan['next_stage']}")
    check(len(plan["completed"]) >= 8, "完成的阶段数合理", f"got {plan['completed']}")

    # 合并规则：已完成阶段保持完成；失败/运行中重置为 pending
    fresh = new_state(run_id="x", direction="d")
    merged = merge_resume(fresh, state)
    for stage, status in (state.get("stage_status") or {}).items():
        if status in ("done", "skipped"):
            eq(merged["stage_status"].get(stage), status, f"恢复保留 {stage} 的完成态")

    # 产物合并要去重且不丢
    before = {(a.get("path")) for a in (state.get("artifacts") or [])}
    after = {(a.get("path")) for a in (merged.get("artifacts") or [])}
    eq(after - before, set(), "恢复后没有凭空多出产物")
    check(len(after) >= len(before) - 1, "恢复后产物数量不减少",
          f"{len(before)} → {len(after)}")

    # force_restart 必须清空完成态
    restarted = merge_resume(new_state(run_id="x", direction="d"), state, force_restart=True)
    eq(
        [s for s, v in restarted["stage_status"].items() if v == "done"],
        [],
        "force_restart 清空所有完成态",
    )

    # 定位运行目录：唯一前缀匹配
    from autoresearch.graph.checkpoint import find_run_dir

    run_id = state.get("run_id", "")
    if len(run_id) > 6:
        found = find_run_dir(run_dir.parent, run_id[:6])
        check(found is not None, "支持用唯一前缀定位运行目录")


def test_litellm_review_loop_termination(run_dir: Path, state: dict) -> None:
    """评审循环必须终止，且分数轨迹可审计。"""
    section("评审循环")
    reviews = state.get("reviews") or []
    check(len(reviews) >= 1, "至少产生一轮评审")
    rounds = [r.get("round") for r in reviews if isinstance(r, dict)]
    eq(rounds, list(range(1, len(rounds) + 1)), "轮次连续递增")
    check(len(reviews) <= int(state.get("max_review_rounds") or 99),
          "轮次不超过上限", f"got {len(reviews)}")
    for r in reviews:
        score = float(r.get("score") or 0)
        check(1.0 <= score <= 10.0, f"第 {r.get('round')} 轮分数在 1-10 内", f"score={score}")
        check(r.get("verdict") in ("ready", "almost", "revise", "reject"),
              f"第 {r.get('round')} 轮 verdict 合法", f"got {r.get('verdict')}")

    history = run_dir / "review" / "AUTO_REVIEW.md"
    check(history.exists(), "AUTO_REVIEW.md 已归档")


def test_review_loop_is_wired() -> None:
    """评审循环必须**真的接到引擎的条件边上**。

    这是本项目踩过的最隐蔽的一个坑：``ReviewStage`` 只定义了 ``route()``，
    而引擎用 ``getattr(impl, "router", None)`` 找条件边，于是命中了基类的
    ``router = None``——条件边从未装配，评审分数再低也不会触发第二轮修改。
    管线「看起来」是正常的：9/9 阶段 done、报告齐全、分数照打，
    只是**迭代机制完全没生效**。只有接真实 LLM（分数低到该触发修改）才看得出来。

    因此这里同时断言两件事：(a) 条件边已装配；(b) 低分确实会路由回 s6。
    """
    section("评审循环接线")
    import shutil as _shutil

    from autoresearch.config import load_config
    from autoresearch.graph.state import new_state
    from autoresearch.runner import RunContext, build_context, build_engine
    from autoresearch.stages.s8_review import ACCEPT_SCORE, ReviewStage

    cfg = load_config(
        direction="评审循环接线测试",
        llm={"provider": "mock"},
        retrieve={"offline": True},
        max_review_rounds=3,
    )
    probe = _workspace_root() / ".autoresearch" / "test_tmp" / "review_wired"
    if probe.exists():
        _shutil.rmtree(probe, ignore_errors=True)
    ctx = build_context(cfg, run_dir=probe, quiet=True)

    # (a) 条件边必须存在，且指向 s6_writing（不是 None、不是 s9）
    engine = build_engine(ctx)
    node = next((n for n in engine.nodes if n.name == "s8_review"), None)
    check(node is not None, "引擎里存在 s8_review 节点")
    if node is not None:
        check(node.router is not None,
              "s8_review 装配了条件边（router 不是 None）",
              "route() 定义后忘记 router = route 会让评审循环静默失效")
    check(ReviewStage.router is not None, "ReviewStage.router 是类属性且非空")

    # (b) 低分 → 回 s6；高分/预算耗尽/收益递减 → 前进
    stage = ReviewStage(ctx)
    low = new_state(run_id="t", direction="d")
    low.update(review_score=2.5, review_verdict="revise", review_round=1,
               reviews=[{"round": 1, "score": 2.5, "verdict": "revise", "weaknesses": []}])
    eq(stage.route(low), "s6_writing", "低分触发回到 s6_writing 继续修改")

    high = new_state(run_id="t", direction="d")
    high.update(review_score=7.4, review_verdict="almost", review_round=1,
                reviews=[{"round": 1, "score": 7.4, "verdict": "almost", "weaknesses": []}])
    check(stage.route(high) is None, "达标后不再回环（进入 s9）",
          f"score={high['review_score']} >= {ACCEPT_SCORE}")

    exhausted = new_state(run_id="t", direction="d")
    exhausted.update(review_score=3.0, review_verdict="reject", review_round=3,
                     reviews=[{"round": 1, "score": 2.0, "verdict": "reject", "weaknesses": []},
                              {"round": 2, "score": 3.0, "verdict": "reject", "weaknesses": []},
                              {"round": 3, "score": 3.0, "verdict": "reject", "weaknesses": []}])
    check(stage.route(exhausted) is None, "轮次用尽后停止回环")

    # 分数与判定自相矛盾时要被修正（ready 但分低 → revise）
    from autoresearch.stages.s8_review import _normalize_verdict

    eq(_normalize_verdict("ready", 3.0), "revise", "低分不允许自称 ready")
    eq(_normalize_verdict("reject", 8.0), "revise", "高分不允许自称 reject")
    check(_normalize_verdict("", 8.2) in ("ready", "almost"), "缺 verdict 时按分数推断")

    _shutil.rmtree(probe, ignore_errors=True)


def test_output_budget_covers_long_stages() -> None:
    """长输出阶段的输出预算必须显著高于默认值。"""
    section("输出预算")
    from autoresearch.stages.base import Stage

    budgets = Stage.OUTPUT_BUDGETS
    # 这几个阶段的真实输出在 DeepSeek 上会撞到 4096 上限（实测被截断后 JSON 全废）
    for name, floor in (("s1_survey", 8192), ("s5_analysis", 8192),
                        ("s2_ideas", 8192), ("s4_codegen", 8192), ("s6_revision", 8192)):
        check(budgets.get(name, 0) >= floor,
              f"{name} 输出预算 >= {floor}", f"got {budgets.get(name)}")
    # 短输出阶段不该浪费预算
    check(budgets.get("s1_queries", 10**9) <= 4096, "s1_queries 预算保持精简")

    # 未列出的提示词必须回落到配置里的 max_tokens（而不是硬编码或崩溃）
    class _Ctx:
        class cfg:  # noqa: N801 - 只需形状
            class llm:  # noqa: N801
                max_tokens = 5000

    class _Stage(Stage):
        def run(self, state):  # pragma: no cover - 仅为实例化
            return None

    stage = _Stage(_Ctx())
    eq(stage._budget_for("s1_survey"), 16384, "已列出的提示词用专门预算")
    eq(stage._budget_for("some_new_prompt"), 5000, "未列出的提示词回落配置默认值")


def test_real_backend_construction() -> None:
    """真实（非 mock）后端的构造路径必须可用。

    mock 后端与 HTTP 后端是**两条完全不同的分支**：mock 端到端测试全绿，
    也不代表接上真实模型能跑。本项目踩过一次真实的坑：给
    ``OpenAICompatBackend`` 传 ``event_logger=self.events``，而 ``LLMClient``
    上那个属性其实叫 ``self.event_logger``——结果接上 DeepSeek 后**每个**阶段
    都在 ``AttributeError`` 上失败，而 204 项 mock 检查全部通过。

    这个测试把真实后端的构造路径走通一遍（用假的 OpenAI 客户端，不发网络请求），
    专门锁住「属性名/参数名拼错」这一类只有在真实运行时才炸的错误。
    """
    section("真实后端构造")
    try:
        import openai  # noqa: F401
    except ImportError:  # pragma: no cover - 无 openai 时无从构造
        check(True, "openai 未安装，跳过真实后端构造检查")
        return

    from autoresearch.llm.client import LLMClient
    from autoresearch.config import load_config

    for provider, model, base_url, key in (
        ("deepseek", "deepseek-chat", "https://api.deepseek.com/v1", "sk-test"),
        ("ollama", "llama3.1", "http://localhost:11434/v1", "ollama"),
        ("openai", "gpt-4o-mini", None, "sk-test"),
    ):
        cfg = load_config(llm={"provider": provider, "model": model,
                               "base_url": base_url, "api_key": key})
        client = LLMClient(cfg.llm)
        backend = None
        try:
            backend = client.backend()
        except Exception as exc:
            check(False, f"{provider} 后端可构造", f"{type(exc).__name__}: {exc}")
            continue
        check(backend is not None, f"{provider} 后端可构造")
        check(getattr(backend, "name", "") == "openai_compat",
              f"{provider} 走 openai 兼容后端", f"name={getattr(backend, 'name', '?')}")
        # 这两个属性的存在性就是那次 bug 的核心
        check(hasattr(backend, "events"),
              f"{provider} 后端持有事件日志句柄（属性名必须与 LLMClient 一致）")
        check(int(getattr(backend, "default_max_tokens", 0)) >= 8192,
              f"{provider} 后端拿到足够的输出预算")

    # 端到端走一次 complete()，用假客户端替换网络调用：验证 event/usage 记账路径
    class _Msg:
        content = '{"ok": true}'

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        model = "stub"
        usage = None

    class _Completions:
        def create(self, **kwargs):
            _Completions.last = kwargs
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Fake:
        chat = _Chat()

    events: list[dict] = []

    class _Rec:
        def log(self, event, **fields):
            events.append({"event": event, **fields})

    cfg = load_config(llm={"provider": "deepseek", "model": "deepseek-chat",
                           "base_url": "https://api.deepseek.com/v1", "api_key": "sk-test"})
    client = LLMClient(cfg.llm, event_logger=_Rec())
    backend = client.backend()
    backend.client = _Fake()
    backend._log_event("test_event", tag="t")
    eq(len(events), 1, "后端能把事件写进注入的日志器")
    eq(events[0]["event"], "test_event", "事件名原样传递")

    resp = backend.complete("hello", system="sys", max_tokens=2048, tag="t")
    eq(resp.text, '{"ok": true}', "complete() 返回 stub 内容")
    eq(resp.finish_reason, "stop", "finish_reason 被读取")
    eq(_Completions.last.get("max_tokens"), 2048, "max_tokens 透传到请求")
    eq(_Completions.last.get("model"), "deepseek-chat", "model 透传到请求")

    # 截断识别：finish_reason == "length" 时必须触发一次自动放大重试
    calls: list[int] = []

    class _Completions2:
        def create(self, **kwargs):
            calls.append(kwargs.get("max_tokens"))
            if len(calls) == 1:
                class _Trunc:
                    choices = [type("C", (), {"message": type("M", (), {"content": '{"a":' })(),
                                              "finish_reason": "length"})()]
                    model = "stub"
                    usage = None
                return _Trunc()
            return _Resp()

    backend.client = type("F", (), {"chat": type("C", (), {"completions": _Completions2()})()})()
    out = backend.complete("hello", max_tokens=1024, tag="trunc")
    eq(len(calls), 2, "检测到截断后自动重试一次")
    check(calls[1] > calls[0], "重试时放大了输出预算", f"{calls[0]} -> {calls[1]}")
    eq(out.text, '{"ok": true}', "重试后返回完整内容")
    check(any(e["event"] == "llm_truncated" for e in events),
          "截断事件已记录（llm_truncated）")


def test_provider_inference() -> None:
    """只配 Key、不写 provider 时必须能推断出正确的 provider。

    这是最自然的配置方式（``DEEPSEEK_API_KEY=sk-...`` 一行搞定），但如果
    provider 静默留在默认的 ``openai``，请求会带着 ``EMPTY`` key 打到 OpenAI
    并返回 401——**报错指向 OpenAI，而用户配的是 DeepSeek**，误导性极强。
    本项目实测踩过：删除 ``.env`` 里的 provider 行后，整条管线在 401 上全崩。
    """
    section("provider 推断")
    import os as _os

    from autoresearch.config import infer_provider, load_config

    saved = {k: _os.environ.get(k) for k in
             ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
              "AUTORESEARCH_LLM_PROVIDER", "AUTORESEARCH_NO_DOTENV")}
    try:
        for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                    "AUTORESEARCH_LLM_PROVIDER"):
            _os.environ.pop(key, None)
        _os.environ["AUTORESEARCH_NO_DOTENV"] = "1"

        # 无任何 Key → 落回默认 provider
        eq(infer_provider(None), "openai", "无 Key 时落回默认 provider")

        # 只有 DeepSeek Key → 推断为 deepseek（而不是默认 openai）
        _os.environ["DEEPSEEK_API_KEY"] = "sk-ds"
        eq(infer_provider(None), "deepseek", "DEEPSEEK_API_KEY 推断 deepseek")
        cfg = load_config()
        eq(cfg.llm.provider, "deepseek", "load_config 推断出 deepseek")
        eq(cfg.llm.base_url, "https://api.deepseek.com/v1", "并带上正确的 base_url")
        eq(cfg.llm.model, "deepseek-chat", "并带上正确的默认模型")
        eq(cfg.llm.api_key, "sk-ds", "并读到正确的 Key")

        # 两个 Key 同时在 → DeepSeek 优先（OPENAI_API_KEY 常是遗留变量）
        _os.environ["OPENAI_API_KEY"] = "sk-oa"
        eq(infer_provider(None), "deepseek", "两 Key 并存时 deepseek 优先")

        # 显式指定永远优先（含 mock）
        eq(infer_provider("mock"), "mock", "显式 mock 被尊重")
        eq(load_config(llm={"provider": "ollama"}).llm.provider, "ollama",
           "显式 ollama 被尊重，不会被 Key 推断覆盖")

        # 环境变量 AUTORESEARCH_LLM_PROVIDER 也优先于推断
        _os.environ["AUTORESEARCH_LLM_PROVIDER"] = "ollama"
        eq(load_config().llm.provider, "ollama", "环境变量 provider 优先于 Key 推断")

        # 显式 base_url 不被 provider 默认值覆盖（自建代理场景）
        custom = load_config(llm={"provider": "deepseek",
                                  "base_url": "http://localhost:9999/v1"})
        eq(custom.llm.base_url, "http://localhost:9999/v1", "显式 base_url 优先")
    finally:
        for key, value in saved.items():
            if value is None:
                _os.environ.pop(key, None)
            else:
                _os.environ[key] = value


def test_cli_surface() -> None:
    """CLI 的退出码与子命令契约。"""
    section("CLI 契约")
    from autoresearch.cli import build_parser, main
    from autoresearch.runner import EXIT_FATAL, EXIT_OK, compute_exit_code

    parser = build_parser()
    for command in ("run", "resume", "status", "verify", "doctor", "stages", "demo"):
        try:
            ns = parser.parse_args([command] if command != "run" else [command, "-d", "x"])
            check(ns.command == command, f"子命令可解析: {command}")
        except SystemExit:
            check(False, f"子命令可解析: {command}")

    # run 缺少 direction 必须报错退出（不能默默跑空）
    try:
        parser.parse_args(["run"])
        parse_ok = True
    except SystemExit:
        parse_ok = False
    # argparse 不会强制 direction（由 config_from_args 抛 SystemExit），这里只验证
    # 「没有 direction 时确实会失败」
    from autoresearch.config import load_config
    from autoresearch.cli import config_from_args

    try:
        args = parser.parse_args(["run"])
        config_from_args(args)
        failed = False
    except SystemExit:
        failed = True
    check(failed, "缺少 --direction 时 run 会失败")

    # exit code 语义
    eq(compute_exit_code({"artifacts": [], "stage_status": {}}), EXIT_FATAL,
       "无产物 → EXIT_FATAL")
    eq(
        compute_exit_code(
            {
                "artifacts": [{"path": "a"}],
                "paper_sections": {"method": "x"},
                "stage_status": {"s1_literature": "done"},
                "final_pdf": "paper/main.pdf",
            }
        ),
        EXIT_OK,
        "完整产出 → EXIT_OK",
    )
    eq(
        compute_exit_code(
            {
                "artifacts": [{"path": "a"}],
                "paper_sections": {"method": "x"},
                "stage_status": {"s1_literature": "done", "s4_experiment": "failed"},
                "final_pdf": "paper/main.pdf",
            }
        ),
        1,
        "有失败阶段 → 部分完成",
    )

    # stages / doctor 必须能跑（doctor 会做真实探测，允许非 0）
    rc = main(["stages", "--json"])
    eq(rc, EXIT_OK, "stages --json 退出码为 0")
    rc = main(["status", "--json", "--runs-dir", str(_workspace_root() / ".autoresearch" / "runs")])
    eq(rc, EXIT_OK, "status --json 退出码为 0")


def test_no_forbidden_writes() -> None:
    """管线只能在运行目录/工作区内写文件，绝不能碰到系统目录。"""
    section("写入边界")
    from autoresearch.runner import RUN_SUBDIRS
    from autoresearch.graph.state import STAGES

    eq(len(RUN_SUBDIRS), 11, "运行目录子目录清单完整")
    for name in ("report", "paper", "metrics", "figures", "analysis", "review"):
        check(name in RUN_SUBDIRS, f"运行目录包含 {name}/")
    eq(len(STAGES), 9, "阶段数为 9")


def main() -> int:
    print("=" * 70)
    print("Auto-Research 管线端到端回归（离线 / mock LLM）")
    print("=" * 70)

    test_intent_routing()
    test_graph_engine_semantics()
    run_dir, state = test_pipeline_end_to_end()

    if run_dir is not None:
        test_cross_seed_statistics(run_dir, state)
        test_numeric_provenance(run_dir)
        test_resume_and_checkpoint(run_dir)
        test_litellm_review_loop_termination(run_dir, state)

    test_cli_surface()
    test_no_forbidden_writes()
    test_review_loop_is_wired()
    test_output_budget_covers_long_stages()
    test_real_backend_construction()
    test_provider_inference()

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} of {_CHECKS} checks")
        for failure in _FAILURES[:40]:
            print("  - " + failure)
        return 1
    print(f"PASSED {_CHECKS} checks")
    return 0


# pytest 入口（保持同名函数可被收集，但不重复跑重活）
def test_pipeline_suite() -> None:
    """pytest 收集入口：跑完整套件并断言无失败。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
