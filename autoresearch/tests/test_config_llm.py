"""config / logging_utils / llm 层的冒烟测试（零 pytest 依赖）。

两种跑法：

    python -m autoresearch.tests.test_config_llm     # 纯脚本，末尾打印 PASSED n checks
    pytest autoresearch/tests/test_config_llm.py     # pytest 也可收集

退出码：0 = 全部通过；非 0 = 有断言失败。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from autoresearch.config import (
    DEFAULT_RUNS_DIR,
    PROJECT_ROOT,
    WORKSPACE_ROOT,
    AutoResearchConfig,
    coerce_value,
    ensure_dirs,
    load_config,
    load_dotenv,
)
from autoresearch.llm import (
    LLMClient,
    LLMConfig,
    MockBackend,
    build_backend,
    extract_json_text,
    parse_json_loose,
)
from autoresearch.llm.client import ParseError
from autoresearch.llm.mock import INTENT_NAMES, intent_of
from autoresearch.logging_utils import EventLogger

# --------------------------------------------------------------------------- #
# 临时目录
#
# 注意：某些受限环境下系统 %TEMP% 不可写（沙箱只放行工作区），
# 因此这里优先在工作区内建目录，失败才回退 tempfile。
# --------------------------------------------------------------------------- #

_TMP_BASE = WORKSPACE_ROOT / ".autoresearch" / "test_tmp"


class _WorkDir:
    """with 块内可用 Path 的临时目录；__exit__ 时清理。"""

    def __init__(self, name: str = "t") -> None:
        self.name = name
        self.path: Path | None = None

    def __enter__(self) -> Path:
        last_exc: Exception | None = None
        for base in (_TMP_BASE, Path(tempfile.gettempdir())):
            try:
                base.mkdir(parents=True, exist_ok=True)
                path = base / f"{self.name}-{os.getpid()}-{int(time.time() * 1000) % 1000000}"
                path.mkdir(parents=True, exist_ok=False)
                self.path = path
                return path
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"无法创建临时目录: {last_exc}")

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
        return False


def _workdir(name: str = "t") -> _WorkDir:
    return _WorkDir(name)

# --------------------------------------------------------------------------- #
# 极简断言工具（脚本模式下自己数数）
# --------------------------------------------------------------------------- #

CHECKS = {"n": 0}


def check(cond, msg: str = "assertion failed") -> None:
    CHECKS["n"] += 1
    if not cond:
        raise AssertionError(msg)


def eq(actual, expected, msg: str = "") -> None:
    check(
        actual == expected,
        f"{msg or 'value mismatch'}: 期望 {expected!r}，实际 {actual!r}",
    )


#: 测试期间禁用 .env 文件加载。开发机上的 .env 含真实密钥与真实 provider，
#: 会泄漏进测试进程，使「provider 默认 model」这类断言依赖本地配置而失败。
os.environ.setdefault("AUTORESEARCH_NO_DOTENV", "1")


def _clear_autoresearch_env() -> None:
    for key in list(os.environ):
        if key.startswith("AUTORESEARCH_"):
            os.environ.pop(key, None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ.pop("OPENAI_BASE_URL", None)


def _llm(provider: str = "mock", **kw) -> LLMClient:
    return LLMClient(LLMConfig(provider=provider, **kw))


# =========================================================================== #
# 1. config：dotenv
# =========================================================================== #


def test_dotenv_precedence() -> None:
    _clear_autoresearch_env()
    ws_env = WORKSPACE_ROOT / ".env"
    proj_env = PROJECT_ROOT / ".env"
    if ws_env.exists() or proj_env.exists():
        # 只在工作区真的存在 .env 时验证合并行为（无则跳过该断言）
        vals = load_dotenv()
        check(isinstance(vals, dict))
        check(bool(vals), "工作区存在 .env，解析结果不应为空")

    with _workdir() as td:
        p = Path(td) / ".env"
        p.write_text(
            "PLAIN=abc\n"
            "export EXPORTED=yes\n"
            'DQ="quoted value"\n'
            "SQ='sq value'\n"
            "HASH_COMMENT=value # trailing\n"
            "EXPANDED=${PLAIN}/tail\n"
            "EXPANDED_ENV=${SOME_ABSENT_VAR:-default}\n",
            encoding="utf-8",
        )
        os.environ["SOME_ABSENT_VAR"] = "from-env"
        try:
            d = load_dotenv(p)
        finally:
            os.environ.pop("SOME_ABSENT_VAR", None)

    eq(d.get("PLAIN"), "abc", "普通 KEY=VALUE")
    eq(d.get("EXPORTED"), "yes", "export 前缀应被剥离")
    eq(d.get("DQ"), "quoted value", "双引号应被剥离")
    eq(d.get("SQ"), "sq value", "单引号应被剥离")
    eq(d.get("HASH_COMMENT"), "value", "行尾注释应被剥离")
    eq(d.get("EXPANDED"), "abc/tail", "${VAR} 应展开为已解析值")
    check("EXPANDED_ENV" in d, "带 :- 的引用仍应产出键")
    check("PLAIN" in os.environ, "load_dotenv 应 setdefault 进 os.environ")
    os.environ.pop("PLAIN", None)
    os.environ.pop("EXPORTED", None)
    os.environ.pop("DQ", None)
    os.environ.pop("SQ", None)
    os.environ.pop("HASH_COMMENT", None)
    os.environ.pop("EXPANDED", None)
    os.environ.pop("EXPANDED_ENV", None)


def test_dotenv_explicit_path_only() -> None:
    with _workdir() as td:
        p = Path(td) / "custom.env"
        p.write_text("# comment only\n\n", encoding="utf-8")
        eq(load_dotenv(p), {}, "只有注释的文件应解析为空 dict")


# =========================================================================== #
# 2. config：环境变量与健壮转换
# =========================================================================== #


def test_env_overrides_and_coercion() -> None:
    _clear_autoresearch_env()
    base = load_config()
    eq(base.venue, "NeurIPS", "默认 venue")
    eq(base.max_review_rounds, 3, "默认 max_review_rounds")
    eq(base.retrieve.offline, False, "默认 offline")

    os.environ["AUTORESEARCH_VENUE"] = "ICLR"
    os.environ["AUTORESEARCH_SEED"] = "7"
    os.environ["AUTORESEARCH_TEMPERATURE"] = "0.9"
    os.environ["AUTORESEARCH_OFFLINE"] = "true"
    os.environ["AUTORESEARCH_MAX_TOKENS"] = "2048"
    os.environ["AUTORESEARCH_LLM_PROVIDER"] = "mock"
    try:
        cfg = load_config()
        eq(cfg.venue, "ICLR", "AUTORESEARCH_VENUE")
        eq(cfg.seed, 7, "AUTORESEARCH_SEED -> int")
        eq(cfg.llm.temperature, 0.9, "AUTORESEARCH_TEMPERATURE -> float")
        eq(cfg.llm.max_tokens, 2048, "AUTORESEARCH_MAX_TOKENS -> int")
        eq(cfg.retrieve.offline, True, "AUTORESEARCH_OFFLINE -> bool")
        eq(cfg.llm.provider, "mock", "AUTORESEARCH_LLM_PROVIDER")

        # 显式覆盖优先于环境变量（两者同时存在）
        eq(load_config(venue="CVPR").venue, "CVPR", "显式 overrides 优先级最高")
        eq(load_config(seed=99).seed, 99, "显式 int 覆盖环境变量")
        eq(load_config(llm={"temperature": 0.1}).llm.temperature, 0.1, "嵌套显式覆盖")

        # 坏值：warning + 回退默认值，绝不崩
        os.environ["AUTORESEARCH_SEED"] = "not-a-number"
        eq(load_config().seed, 0, "坏 int 回退默认值")
        os.environ["AUTORESEARCH_TEMPERATURE"] = "hot"
        eq(load_config().llm.temperature, 0.3, "坏 float 回退默认值")
        os.environ["AUTORESEARCH_OFFLINE"] = "maybe"
        eq(load_config().retrieve.offline, False, "坏 bool 回退默认值")
    finally:
        _clear_autoresearch_env()

    # 直接测 coerce_value
    eq(coerce_value("true", False), True, "true -> True")
    eq(coerce_value("1", False), True, "1 -> True")
    eq(coerce_value("yes", False), True, "yes -> True")
    eq(coerce_value("0", True), False, "0 -> False")
    eq(coerce_value("3.5", 1.0), 3.5, "float 字符串")
    eq(coerce_value("12", 5), 12, "int 字符串")
    eq(coerce_value("oops", 5), 5, "坏 int -> 默认")
    eq(coerce_value("oops", 1.5), 1.5, "坏 float -> 默认")
    eq(coerce_value("oops", False), False, "坏 bool -> 默认")
    eq(coerce_value(None, "x"), "x", "None -> 默认")


def test_provider_defaults_and_api_key_mapping() -> None:
    _clear_autoresearch_env()
    os.environ["DEEPSEEK_API_KEY"] = "sk-deepseek-test"
    os.environ["OPENAI_API_KEY"] = "sk-openai-test"
    try:
        d = load_config(llm={"provider": "deepseek"})
        eq(d.llm.base_url, "https://api.deepseek.com/v1", "deepseek base_url 默认")
        eq(d.llm.model, "deepseek-chat", "deepseek model 默认")
        eq(d.llm.api_key, "sk-deepseek-test", "deepseek 读 DEEPSEEK_API_KEY")

        o = load_config(llm={"provider": "openai"})
        eq(o.llm.api_key, "sk-openai-test", "openai 读 OPENAI_API_KEY")

        l = load_config(llm={"provider": "ollama"})
        eq(l.llm.base_url, "http://localhost:11434/v1", "ollama base_url 默认")
        eq(l.llm.model, "llama3.1", "ollama model 默认")
        eq(l.llm.api_key, "ollama", "ollama 无需真实 key")

        m = load_config(llm={"provider": "mock"})
        eq(m.llm.api_key, None, "mock 不需要 key")
        eq(m.llm.model, LLMConfig.model, "mock 保留默认 model")

        # api_key 不落盘
        eq(d.to_dict()["llm"]["api_key"], "sk-deepseek-test", "to_dict 保留真实 key")
        eq(d.redacted_dict()["llm"]["api_key"], "***", "redacted_dict 打码")
        eq(d.llm.api_key, "sk-deepseek-test", "redacted_dict 不改变原对象")
    finally:
        _clear_autoresearch_env()


def test_ensure_dirs_and_constants() -> None:
    eq(PROJECT_ROOT.name, "autoresearch", "PROJECT_ROOT 是包目录")
    eq(WORKSPACE_ROOT, PROJECT_ROOT.parent, "WORKSPACE_ROOT 是上一级")
    eq(DEFAULT_RUNS_DIR, WORKSPACE_ROOT / ".autoresearch" / "runs", "DEFAULT_RUNS_DIR")

    with _workdir() as td:
        cfg = load_config(runs_dir=Path(td) / "runs", run_id="r-001")
        eq(cfg.run_dir, Path(td) / "runs" / "r-001", "run_dir = runs_dir/run_id")
        ensure_dirs(cfg)
        check((Path(td) / "runs" / "r-001").is_dir(), "ensure_dirs 应创建 run_dir")

        cfg2 = load_config(runs_dir=Path(td) / "runs2")
        ensure_dirs(cfg2)
        check((Path(td) / "runs2").is_dir(), "ensure_dirs 应创建 runs_dir")
        eq(cfg2.run_dir, Path(td) / "runs2", "无 run_id 时 run_dir = runs_dir")


# =========================================================================== #
# 3. config：to_dict / from_dict / redacted
# =========================================================================== #


def test_to_from_dict_roundtrip() -> None:
    cfg = load_config(
        direction="受控比较协议",
        venue="ICML",
        seed=11,
        max_review_rounds=4,
        runs_dir=Path("D:/tmp/runs-xyz"),
        llm={"provider": "mock", "temperature": 0.55, "max_tokens": 1024},
        sandbox={"backend": "docker", "timeout": 30, "cpus": 4.0},
        retrieve={"sources": ["arxiv"], "offline": True, "cache_dir": "D:/tmp/cache-x"},
        compile={"engine": "none", "tectonic_version": "9.9.9"},
    )
    d = cfg.to_dict()
    json.dumps(d, ensure_ascii=False)  # 必须 JSON 可序列化
    eq(d["runs_dir"], str(Path("D:/tmp/runs-xyz")), "Path -> str")
    eq(d["llm"]["provider"], "mock", "嵌套 llm")
    eq(d["sandbox"]["cpus"], 4.0, "嵌套 sandbox")
    eq(d["retrieve"]["sources"], ["arxiv"], "嵌套 retrieve list")
    eq(d["compile"]["tectonic_version"], "9.9.9", "嵌套 compile")

    back = AutoResearchConfig.from_dict(d)
    eq(back.to_dict(), d, "to_dict/from_dict 往返一致")
    check(isinstance(back.runs_dir, Path), "from_dict 应还原 Path")
    check(isinstance(back.retrieve.cache_dir, Path), "from_dict 应还原嵌套 Path")
    eq(back.llm.temperature, 0.55, "嵌套 llm 字段还原")

    # 接受 str 形式的 Path
    d2 = dict(d)
    d2["runs_dir"] = "D:/tmp/runs-str"
    eq(AutoResearchConfig.from_dict(d2).runs_dir, Path("D:/tmp/runs-str"), "str -> Path")

    # 未知键被忽略而不是崩溃
    d3 = dict(d)
    d3["totally_unknown"] = 1
    AutoResearchConfig.from_dict(d3)


def test_redacted_dict() -> None:
    cfg = load_config(llm={"provider": "openai", "api_key": "sk-secret-value"})
    red = cfg.redacted_dict()
    eq(red["llm"]["api_key"], "***", "api_key 非空 -> ***")
    eq(cfg.llm.api_key, "sk-secret-value", "原对象不变")
    check("sk-secret-value" not in json.dumps(red, ensure_ascii=False), "打码后不应泄漏明文")

    empty = load_config(llm={"provider": "mock", "api_key": ""})
    eq(empty.redacted_dict()["llm"]["api_key"], None, "空 key 不打码（保持 None）")
    check("***" not in json.dumps(empty.redacted_dict(), ensure_ascii=False), "空 key 不出现 ***")


# =========================================================================== #
# 4. logging_utils
# =========================================================================== #


def test_event_logger_jsonl_and_tail() -> None:
    with _workdir() as td:
        run_dir = Path(td) / "run1"
        ev = EventLogger(run_dir)
        ev.log("run_start", run_id="r1", direction="方向A", n=1)
        ev.log("stage_start", stage="s1_literature")
        ev.log("llm_call", model="mock", cached=False, tokens={"p": 1})

        path = run_dir / "events.jsonl"
        check(path.is_file(), "events.jsonl 应被创建")
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        eq(len(lines), 3, "3 条事件 -> 3 行")
        for line in lines:
            obj = json.loads(line)  # 每行必须是合法 JSON
            check(isinstance(obj, dict), "每行是 JSON 对象")
            check("ts" in obj, "自动加 ts")
            check("event" in obj, "自动加 event")
            check(str(obj["ts"]).endswith("Z"), "ts 是 ISO8601 UTC")

        tail = ev.tail(2)
        eq(len(tail), 2, "tail(2) 返回 2 条")
        eq(tail[-1]["event"], "llm_call", "tail 顺序为文件顺序")
        eq(tail[-1]["tokens"], {"p": 1}, "嵌套字段原样落盘")
        eq(len(ev.tail(0)), 0, "tail(0) 为空")
        eq(len(ev.tail(99)), 3, "tail 超出总数返回全部")
        eq(ev.count(), 3, "count 正确")

        # 非 ASCII 不转义
        ev.log("note", text="中文不转义")
        last_line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
        check("中文不转义" in last_line, "ensure_ascii=False")

        # 追加而非覆盖
        ev2 = EventLogger(run_dir)
        ev2.log("event_after_reopen")
        eq(ev2.count(), 5, "重新打开后仍追加")

        # 写入失败不抛异常（把目标换成一个目录，制造写入错误）
        broken = EventLogger(run_dir)
        broken.path = run_dir  # 目录当文件写 -> OSError
        broken.log("should_not_raise")


def test_configure_logging_idempotent() -> None:
    import logging

    from autoresearch.logging_utils import configure_logging, get_logger

    with _workdir() as td:
        log_file = Path(td) / "logs" / "run.log"
        root = logging.getLogger()
        before = len(root.handlers)
        try:
            configure_logging("INFO", log_file=log_file, quiet=False)
            n1 = len(root.handlers)
            configure_logging("INFO", log_file=log_file, quiet=False)
            n2 = len(root.handlers)
            eq(n2, n1, "重复 configure_logging 不叠加 handler")
            configure_logging("DEBUG", log_file=log_file, quiet=True)
            check(len(root.handlers) <= 2, "handler 数量不超过 2")
            log = get_logger("autoresearch.test")
            log.warning("hello-file-handler")
            check(log_file.is_file(), "日志文件应被创建")
            check("hello-file-handler" in log_file.read_text(encoding="utf-8"), "日志内容落盘")
        finally:
            configure_logging("INFO", log_file=None, quiet=False)
            root.handlers = root.handlers[:before] if before else root.handlers


def test_progress_reporter_no_ansi() -> None:
    import io

    from autoresearch.logging_utils import ProgressReporter

    buf = io.StringIO()
    pr = ProgressReporter(stream=buf)
    pr.step("s1_literature", "done", "12 papers", elapsed=1.25)
    pr.step("s2_ideation", "failed", "boom", elapsed=61.0)
    pr.step("s3_planning", "running")
    # 兼容兄弟 CLI 可能的调用形状
    pr.step("s4_experiment", ok=True, detail="done via ok=")
    pr.step("s5_analysis", "done", duration=3.5)
    pr.step("s6_writing", "weird-status")  # 未知状态只 warning
    pr.summary(
        [
            {"stage": "s1_literature", "ok": True, "elapsed": 1.2, "detail": "12 papers"},
            {"stage": "s2_ideation", "ok": False, "elapsed": 2.5, "detail": "boom"},
        ]
    )
    out = buf.getvalue()
    check("\x1b[" not in out, "不得输出 ANSI 转义序列")
    check("s1_literature" in out and "s2_ideation" in out, "summary 含 stage 名")
    check("FAILED" in out.upper(), "summary 含失败状态")
    check("s5_analysis" in out and "3.5s" in out, "duration= 关键字被接受")


# =========================================================================== #
# 5. MockBackend 意图路由
# =========================================================================== #

#: 每个意图一条最典型的 prompt（中文/英文混合）
INTENT_PROMPTS: dict[str, str] = {
    "query": "请根据研究关键词生成 6 条检索式，用于 arXiv 检索。研究方向：参数高效微调。",
    "literature": "请写一段文献综述，并指出 related work 中的 research gap。",
    "idea": "请头脑风暴，构思 4 个研究假设（idea），要求可复现。",
    "novelty": "请对这个想法做新颖性检查（novelty check），判断是否已被做过。",
    "plan": "请给出实验计划：里程碑 milestone、消融矩阵与算力预算。",
    "code": "请生成实验代码，输出 train.py 脚本。",
    "debug": "运行脚本时出现 Traceback (most recent call last): SyntaxError: bad 报错，请 debug 修复。",
    "writing": "请撰写论文的 method 章节草稿（draft the section）。",
    "review": "你是审稿人，请评审这篇论文并打分（review round 1）。",
    "analysis": "请分析实验结果，给出主要结论与局限。",
    "fallback": "你好，请用一句话介绍一下你自己。",
}


def test_intent_routing_all() -> None:
    backend = MockBackend()
    for expected, prompt in INTENT_PROMPTS.items():
        got = backend.intent_of(prompt)
        eq(got, expected, f"intent_of 分类错误 [{prompt[:24]}...]")
        eq(intent_of(prompt), expected, "模块级 intent_of 与实例方法一致")
    eq(set(INTENT_PROMPTS), set(INTENT_NAMES), "测试覆盖了全部意图名")


#: 哥哥模块 prompts/*.md 的真实意图（模板标题决定任务类型）
TEMPLATE_INTENTS: dict[str, str] = {
    "s1_queries": "query",
    "s1_survey": "literature",
    "s2_ideas": "idea",
    "s2_novelty": "novelty",
    "s3_plan": "plan",
    "s4_codegen": "code",
    "s4_debug": "debug",
    "s5_analysis": "analysis",
    "s6_abstract": "writing",
    "s6_revision": "review",
    "s6_section": "writing",
    "s7_compile_fix": "debug",
    "s8_review": "review",
    "s9_report": "writing",
}


def test_intent_routing_on_real_prompt_templates() -> None:
    """回归：真实提示词模板必须路由到正确意图。

    模板不存在时跳过（pipeline 的 prompts/ 由兄弟模块负责）。
    """
    prompts_dir = PROJECT_ROOT / "prompts"
    if not prompts_dir.is_dir():
        return
    checked = 0
    for stem, expected in TEMPLATE_INTENTS.items():
        path = prompts_dir / f"{stem}.md"
        if not path.is_file():
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        eq(intent_of(text), expected, f"模板 {stem}.md 路由错误")
    check(checked > 0, "至少应检查到一个真实模板")


def test_mock_intent_payload_shapes() -> None:
    backend = MockBackend()
    code = json.loads(backend.complete(INTENT_PROMPTS["code"]).text)
    check(isinstance(code["files"], list) and code["files"], "code -> files 非空")
    eq(code["files"][0]["path"], "train.py", "code -> train.py")
    check("def train" in code["files"][0]["content"], "生成脚本应含 def train")

    ideas = json.loads(backend.complete(INTENT_PROMPTS["idea"]).text)["ideas"]
    eq(len(ideas), 4, "idea -> 默认 4 条")
    for idea in ideas:
        for key in (
            "id",
            "title",
            "hypothesis",
            "motivation",
            "method_sketch",
            "novelty_claim",
            "feasibility",
            "risks",
            "expected_metrics",
        ):
            check(key in idea, f"Idea 缺字段 {key}")
        check(isinstance(idea["risks"], list) and idea["risks"], "risks 非空 list")
        check(isinstance(idea["expected_metrics"], list), "expected_metrics 是 list")

    q = json.loads(backend.complete(INTENT_PROMPTS["query"]).text)
    eq(len(q["queries"]), 6, "query -> 6 条检索式")

    rev = json.loads(backend.complete(INTENT_PROMPTS["review"]).text)
    for key in ("score", "verdict", "summary", "strengths", "weaknesses", "questions",
                "min_fixes", "per_criterion"):
        check(key in rev, f"review 缺字段 {key}")
    eq(sorted(rev["per_criterion"]),
       ["clarity", "experiments", "novelty", "reproducibility", "rigor"],
       "per_criterion 五个维度")
    check(isinstance(rev["strengths"][0], dict) and "point" in rev["strengths"][0],
          "strengths 是 {point,evidence} 对象")
    check("severity" in rev["weaknesses"][0] and "min_fix" in rev["weaknesses"][0],
          "weaknesses 含 severity/min_fix")

    nov = json.loads(backend.complete(INTENT_PROMPTS["novelty"]).text)
    check(nov["verdict"] in ("novel", "incremental", "duplicate"), "novelty verdict 合法")
    check(0.0 <= float(nov["score"]) <= 1.0, "novelty score 在 [0,1]")

    plan = json.loads(backend.complete(INTENT_PROMPTS["plan"]).text)
    for key in ("objective", "milestones", "dataset", "baseline", "metrics",
                "ablation_matrix", "compute_budget_hours", "risks"):
        check(key in plan, f"plan 缺字段 {key}")
    check(plan["milestones"], "milestones 非空")

    dbg = json.loads(backend.complete(INTENT_PROMPTS["debug"]).text)
    for key in ("diagnosis", "root_cause", "files", "confidence", "commands_to_verify",
                "validity_note"):
        check(key in dbg, f"debug 缺字段 {key}")
    check(isinstance(dbg["files"], list), "debug.files 是 list")

    lit = json.loads(backend.complete(INTENT_PROMPTS["literature"]).text)
    for key in ("summary", "gaps", "themes", "method_landscape"):
        check(key in lit, f"literature 缺字段 {key}")
    for theme in lit["themes"]:
        check(isinstance(theme, dict) and "name" in theme and "paper_ids" in theme,
              "themes 是结构化对象")
    for gap in lit["gaps"]:
        check(isinstance(gap, dict) and "gap" in gap and "why_unsolved" in gap,
              "gaps 是结构化对象")

    wr = json.loads(backend.complete(INTENT_PROMPTS["writing"]).text)
    for key in ("section", "latex", "claims", "citations_used", "word_count"):
        check(key in wr, f"writing 缺字段 {key}")
    check(wr["word_count"] > 0, "word_count > 0")

    an = json.loads(backend.complete(INTENT_PROMPTS["analysis"]).text)
    for key in ("claim_evidence", "findings", "limitations", "negative_results",
                "threats_to_validity", "figure_discussion"):
        check(key in an, f"analysis 缺字段 {key}")

    fb = json.loads(backend.complete(INTENT_PROMPTS["fallback"]).text)
    check("text" in fb and fb.get("ok") is True, "fallback -> {text, ok:true}")


def test_mock_is_deterministic_and_logs_calls() -> None:
    b1, b2 = MockBackend(), MockBackend()
    p = INTENT_PROMPTS["idea"]
    t1 = b1.complete(p).text
    t2 = b2.complete(p).text
    t3 = b1.complete(p).text
    eq(t1, t2, "同 prompt 必须完全一致（跨实例）")
    eq(t1, t3, "重复调用一致")
    eq(len(b1.call_log), 2, "call_log 记录每次调用")
    check(all(len(entry) <= 120 for entry in b1.call_log), "call_log 是 120 字符前缀")
    eq(b1.call_log[0], p[:120], "call_log 存 prompt 前缀")
    eq(len(b1.intents), 2, "intents 同步记录")
    eq(b1.intents[0], "idea", "intent 被记录")
    eq(len(b2.call_log), 1, "b2 只调用一次")

    resp = b1.complete(p)
    eq(resp.model, "mock-heuristic", "model 名")
    check(resp.usage.prompt_tokens > 0 and resp.usage.completion_tokens > 0, "usage 非零")
    eq(resp.usage.calls, 1, "usage.calls")

    # 负向：不同 prompt -> 不同输出
    other = b1.complete("请检索：扩散模型 采样加速 检索式").text
    check(other != b1.complete(p).text, "不同 prompt 输出应不同")


def test_mock_idea_respects_max_ideas() -> None:
    b = MockBackend()
    text = b.complete('请构思研究假设，max_ideas=2，输出 JSON。').text
    eq(len(json.loads(text)["ideas"]), 2, "max_ideas=2 应被遵守")
    text6 = b.complete('请构思研究假设，最多生成 6 个 idea。').text
    eq(len(json.loads(text6)["ideas"]), 6, "中文“最多生成 N 个”应被遵守")


def test_mock_review_score_monotonic() -> None:
    b = MockBackend()
    scores = []
    for rnd in (1, 2, 3, 4):
        p = f"你是审稿人，请评审这篇论文并打分（review round {rnd}）。"
        scores.append(float(json.loads(b.complete(p).text)["score"]))
    check(all(b > a for a, b in zip(scores, scores[1:])), f"分数应单调递增: {scores}")
    check(5.5 <= scores[0] <= 6.5, f"round1 分数应在 6.0 附近: {scores[0]}")
    check(7.5 <= scores[2] <= 8.5, f"round3 分数应在 7.8 附近: {scores[2]}")
    check(8.0 <= scores[3] <= 9.4, f"round4 分数应更高: {scores[3]}")

    b2 = MockBackend()
    text = b2.complete("请评审论文并打分（第 2 轮）").text
    check(float(json.loads(text)["score"]) > 6.5, "中文轮次也被识别")


def test_mock_debug_repairs_pasted_script() -> None:
    b = MockBackend()
    pasted = (
        "运行 train.py 报错 Traceback: NameError\n```python\n"
        "import csv\n\n\ndef main():\n"
        "    with open('metrics.csv', 'w', newline='') as fh:\n"
        "        fh.write('epoch,loss\\n')\n\n\n"
        "if __name__ == '__main__':\n    main()\n```\n"
    )
    out = json.loads(b.complete(pasted).text)
    eq(len(out["files"]), 1, "应返回被修复的文件")
    eq(out["files"][0]["path"], "train.py", "路径沿用 train.py")
    check("import os" in out["files"][0]["content"], "修复应补 import os")
    check("makedirs" in out["files"][0]["content"], "修复应有目录保护")
    check(0.0 < float(out["confidence"]) <= 1.0, "confidence 在 (0,1]")


# =========================================================================== #
# 6. LLMClient
# =========================================================================== #


def test_build_backend_and_lazy_cache() -> None:
    from autoresearch.llm import Usage

    check(isinstance(build_backend(LLMConfig(provider="mock")), MockBackend), "mock 后端")
    eq(Usage(1, 2, 3, 1) + Usage(2, 3, 5, 1), Usage(3, 5, 8, 2), "Usage.__add__")
    eq(sum([Usage(1, 2, 3, 1), Usage(2, 3, 5, 1)]), Usage(3, 5, 8, 2), "Usage 支持 sum() 的 0 起始")
    try:
        build_backend(LLMConfig(provider="nope"))
        raise AssertionError("未知 provider 应抛 LLMError")
    except Exception as exc:
        check("nope" in str(exc), "错误信息应包含 provider 名")
        check("mock" in str(exc), "错误信息应列出可选 provider")

    client = _llm("mock")
    b1 = client.backend()
    b2 = client.backend()
    check(b1 is b2, "backend() 应缓存实例")
    check(isinstance(b1, MockBackend), "provider=mock -> MockBackend")


def test_complete_json_parses_mock_json() -> None:
    client = _llm("mock")
    data = client.complete_json(INTENT_PROMPTS["query"], schema_hint={"queries": ["str"]})
    check(isinstance(data, dict) and isinstance(data.get("queries"), list), "返回 dict")
    eq(len(data["queries"]), 6, "6 条检索式")
    check(all(isinstance(q, str) and q for q in data["queries"]), "每条检索式是非空 str")
    check(client.usage().calls >= 1, "usage 应累加")

    ideas = client.complete_json(INTENT_PROMPTS["idea"])
    check(isinstance(ideas.get("ideas"), list) and ideas["ideas"], "ideas 解析成功")

    client.reset_usage()
    eq(client.usage().calls, 0, "reset_usage 归零")


def test_extract_json_handles_fences_trailing_and_prose() -> None:
    eq(parse_json_loose('```json\n{"a": 1}\n```'), {"a": 1}, "```json 围栏")
    eq(parse_json_loose('```\n{"a": 1}\n```'), {"a": 1}, "裸围栏")
    eq(parse_json_loose('好的，结果如下：\n{"a": 1}\n希望有帮助。'), {"a": 1}, "前后有解释")
    eq(parse_json_loose('[1, 2, 3] 以上。'), [1, 2, 3], "数组 + 尾随文本")
    eq(parse_json_loose('{"s": "含 } 与 \\" 引号的字符串"}'),
       {"s": '含 } 与 " 引号的字符串'}, "字符串内的括号不干扰平衡扫描")
    eq(parse_json_loose('{"n": {"deep": [1, {"x": 2}]}}'),
       {"n": {"deep": [1, {"x": 2}]}}, "嵌套结构")
    eq(extract_json_text('```json\n{"a":1}\n```'), '{"a":1}', "extract_json_text 去围栏")
    try:
        parse_json_loose("not json at all")
        raise AssertionError("非法 JSON 应抛 ParseError")
    except ParseError:
        pass


def test_complete_json_repair_path_returns_default() -> None:
    client = _llm("mock")
    backend = client.backend()
    seen: list[str] = []

    def garbage(*args, **kw):
        seen.append(str(kw.get("system", "")) + str(args[0] if args else ""))
        resp = MockBackend().complete("x")
        resp.text = "not json"
        return resp

    backend.complete = garbage  # monkeypatch：强制返回垃圾
    sentinel = {"fallback": True}
    out = client.complete_json("请输出 JSON", repair_attempts=1, default=sentinel)
    check(out is sentinel, "修复失败后应返回 default")
    eq(len(seen), 2, "1 次原始 + 1 次修复 = 2 次调用")
    check("parse_error" in seen[1], "修复提示应包含解析错误")
    check("not json" in seen[1], "修复提示应包含原始坏输出")

    try:
        client.complete_json("请输出 JSON", repair_attempts=0)
        raise AssertionError("default=None 且解析失败应抛 ParseError")
    except ParseError as exc:
        check("JSON" in str(exc) or "json" in str(exc), "ParseError 信息可读")


def test_complete_json_repair_recovers() -> None:
    client = _llm("mock")
    backend = client.backend()
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += 1
        resp = MockBackend().complete("x")
        resp.text = '{"ok": true}' if calls["n"] > 1 else "garbage without json"
        return resp

    backend.complete = flaky
    out = client.complete_json("请输出 JSON", repair_attempts=2)
    eq(out, {"ok": True}, "第一次坏 -> 修复后成功")


def test_cache_hit_and_stats() -> None:
    with _workdir() as td:
        cache_dir = Path(td) / "cache"
        client = LLMClient(LLMConfig(provider="mock"), cache_dir=cache_dir)
        p = INTENT_PROMPTS["literature"]
        first = client.complete(p)
        mid = client.cache_stats()
        eq(mid["hits"], 0, "首次是 miss")
        eq(mid["misses"], 1, "misses=1")
        eq(mid["entries"], 1, "缓存文件 1 个")

        second = client.complete(p)
        eq(second, first, "缓存命中返回相同文本")
        stats = client.cache_stats()
        eq(stats["hits"], 1, "hits=1")
        eq(stats["misses"], 1, "misses 仍为 1")
        eq(stats["entries"], 1, "entries 仍为 1")
        check(len(list(cache_dir.glob("*.json"))) == 1, "磁盘上只有 1 个缓存文件")

        usage_after_hit = client.usage()
        client.complete(p)
        eq(client.usage().calls, usage_after_hit.calls, "缓存命中不计 usage")
        eq(client.cache_stats()["hits"], 2, "第二次命中")

        # 不同参数 -> 不同缓存键
        client.complete(p, system="另一个 system")
        eq(client.cache_stats()["misses"], 2, "不同 system -> 新 miss")
        client.complete(p, json_mode=True)
        eq(client.cache_stats()["misses"], 3, "json_mode 影响缓存键")

        # 关闭缓存
        nocache = LLMClient(LLMConfig(provider="mock"), cache_dir=Path(td) / "c2",
                            cache_enabled=False)
        nocache.complete(p)
        nocache.complete(p)
        eq(nocache.cache_stats(), {"hits": 0, "misses": 2, "entries": 0}, "关闭缓存时不落盘")
        check(not (Path(td) / "c2").exists(), "关闭缓存时不应创建目录")


def test_llm_call_event_registered() -> None:
    with _workdir() as td:
        run_dir = Path(td) / "run"
        ev = EventLogger(run_dir)
        client = LLMClient(
            LLMConfig(provider="mock"), cache_dir=Path(td) / "cache", event_logger=ev
        )
        client.complete(INTENT_PROMPTS["analysis"], tag="s5_analysis")
        client.complete(INTENT_PROMPTS["analysis"], tag="s5_analysis")  # 命中缓存

        events = [e for e in ev.tail(20) if e.get("event") == "llm_call"]
        eq(len(events), 2, "每次调用记一条 llm_call")
        first = events[0]
        for key in ("model", "prompt_chars", "completion_chars", "prompt_tokens",
                    "completion_tokens", "latency", "cached", "ok", "tag"):
            check(key in first, f"llm_call 缺字段 {key}")
        eq(first["ok"], True, "ok=True")
        eq(first["cached"], False, "首次未命中缓存")
        eq(first["tag"], "s5_analysis", "tag 被记录")
        eq(events[1]["cached"], True, "第二次命中缓存")
        check(first["prompt_chars"] > 0, "prompt_chars > 0")

        # tag 必须能作为关键字传给 complete / complete_json
        eq(isinstance(client.complete_json(INTENT_PROMPTS["analysis"], tag="s5"), dict),
           True, "complete_json 接受 tag")


def test_generated_train_py_actually_runs() -> None:
    """联网/离线都能跑：只依赖 stdlib，写 metrics.csv + metrics.jsonl。"""
    backend = MockBackend()
    content = json.loads(backend.complete(INTENT_PROMPTS["code"]).text)["files"][0]["content"]
    with _workdir() as td:
        script = Path(td) / "train.py"
        script.write_text(content, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script), "--epochs", "12", "--seed", "1"],
            cwd=td,
            capture_output=True,
            text=True,
            timeout=180,
        )
        eq(proc.returncode, 0, f"生成脚本应能运行；stderr={proc.stderr[-500:]}")
        check("FINAL accuracy=" in proc.stdout, "应打印 FINAL accuracy=")
        csv_path = Path(td) / "metrics.csv"
        jsonl_path = Path(td) / "metrics.jsonl"
        check(csv_path.is_file(), "应写 metrics.csv")
        check(jsonl_path.is_file(), "应写 metrics.jsonl")
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].strip()
        eq(header, "epoch,loss,accuracy,f1,val_loss,val_accuracy", "metrics.csv 表头")
        rows = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        check(len(rows) >= 5, "jsonl 应有多个 epoch")
        json.loads(rows[0])

        baseline = subprocess.run(
            [sys.executable, str(script), "--epochs", "12", "--seed", "1", "--baseline"],
            cwd=td,
            capture_output=True,
            text=True,
            timeout=180,
        )
        eq(baseline.returncode, 0, "baseline 变体也应能运行")
        check("FINAL accuracy=" in baseline.stdout, "baseline 也打印 FINAL accuracy=")


# --------------------------------------------------------------------------- #
# 脚本模式运行器
# --------------------------------------------------------------------------- #

_TESTS = [
    test_dotenv_precedence,
    test_dotenv_explicit_path_only,
    test_env_overrides_and_coercion,
    test_provider_defaults_and_api_key_mapping,
    test_ensure_dirs_and_constants,
    test_to_from_dict_roundtrip,
    test_redacted_dict,
    test_event_logger_jsonl_and_tail,
    test_configure_logging_idempotent,
    test_progress_reporter_no_ansi,
    test_intent_routing_all,
    test_intent_routing_on_real_prompt_templates,
    test_mock_intent_payload_shapes,
    test_mock_is_deterministic_and_logs_calls,
    test_mock_idea_respects_max_ideas,
    test_mock_review_score_monotonic,
    test_mock_debug_repairs_pasted_script,
    test_build_backend_and_lazy_cache,
    test_complete_json_parses_mock_json,
    test_extract_json_handles_fences_trailing_and_prose,
    test_complete_json_repair_path_returns_default,
    test_complete_json_repair_recovers,
    test_cache_hit_and_stats,
    test_llm_call_event_registered,
    test_generated_train_py_actually_runs,
]

_OPTIONAL = {"test_generated_train_py_actually_runs"}


def _run_all() -> int:
    _clear_autoresearch_env()
    failures: list[tuple[str, str]] = []
    skipped: list[str] = []
    for fn in _TESTS:
        try:
            fn()
            print(f"[ OK ] {fn.__name__}")
        except Exception as exc:
            if fn.__name__ in _OPTIONAL:
                skipped.append(fn.__name__)
                print(f"[SKIP] {fn.__name__}: {exc}")
                continue
            failures.append((fn.__name__, traceback.format_exc()))
            print(f"[FAIL] {fn.__name__}: {exc}")
    print("-" * 60)
    if failures:
        for name, tb in failures:
            print(f"\n=== {name} ===\n{tb}")
        print(f"FAILED {len(failures)} test(s); checks run: {CHECKS['n']}")
        return 1
    note = f" (skipped: {', '.join(skipped)})" if skipped else ""
    print(f"PASSED {CHECKS['n']} checks in {len(_TESTS) - len(skipped)} tests{note}")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
