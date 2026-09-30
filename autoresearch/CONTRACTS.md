# Auto-Research 内部接口契约 v1.3 (FROZEN)

> 本文件是**唯一权威接口定义**。所有模块必须严格遵守这里的签名与数据形状。
> 改动契约 = 破坏管线，必须先改本文件。
>
> **v1.3 变更（独立验证补漏）**：§13 登记 `template_placeholders_unfilled`；
> §15.2 补 `max_variants` 与三个 `codegen_*` 方法；§15.3 增加臂数上限与消融命名的
> 硬性约束；§12 补 `--max-variants`；§11 补 `GraphEngine.__init__` 的 `max_visits`/`on_step`。
>
> **v1.2 变更（文档审计修正）**：§1 的配置清单与 `config.py` 对齐
> （`max_tokens` 8192、`max_file_mb`、`experiment_adapter`、`adapter_params`、
> `max_variants`）；§9 的 `new_state` 签名与状态键表补全；§12 去掉不存在的
> `--resume`；§0 澄清提示词引擎不是 Jinja2；§14 修正套件数为 7。
>
> **v1.1 变更（阶段 B 重构）**：实验执行层从 `stages/s4_experiment.py` 抽出为
> 可插拔的 `adapters/` 包（见 §15）。`s4_experiment` 不再自己构造命令、不再自己
> 解析指标、不再硬编码 `baseline`/`method`；沙箱新增 `run_command()`。
> 其余各节的契约保持不变。

## 0. 项目布局（绝对路径，禁止越界写文件）

根目录：`D:\user\Documents\deepseekv4flash harness\autoresearch\`

```
autoresearch/
  cli.py                     # argparse 入口
  config.py                  # 配置层
  logging_utils.py           # 日志
  runner.py                  # RunContext 组装 + 顶层执行
  verify.py                  # 论文数字溯源校验
  adapters/                  # ★ 实验后端适配器（v1.1 新增，见 §15）
    base.py                  #   RunSpec + BaseExperimentAdapter 协议 + 解析
    synthetic_toy.py         #   内置默认：受控合成分类任务
    script_wrapper.py        #   用户自带训练脚本
  graph/
    state.py                 # 状态 schema + 常量 + Artifact
    checkpoint.py            # 断点续跑
    engine.py                # 自研状态机
    langgraph_adapter.py     # LangGraph 可选适配层
  llm/
    client.py                # LLM 客户端
    mock.py                  # 离线 MockBackend（全流程可跑通）
  tools/
    retrieve.py              # 文献检索 (arXiv/S2/OpenAlex/Crossref)
    pdfx.py                  # PDF 解析
    sandbox.py               # 子进程沙箱 + Docker 后端
    latex.py                 # tectonic 自动拉取 + 编译
    fetch_tectonic.py        # tectonic 按需下载脚本（不进版本控制）
    figures.py               # 图表生成
    metrics.py               # 指标解析 + 三线表
  stages/
    base.py                  # Stage 基类 + StageResult
    s1_literature.py
    s2_ideation.py
    s3_planning.py
    s4_experiment.py
    s5_analysis.py
    s6_writing.py
    s7_compile.py
    s8_review.py
    s9_finalize.py
  prompts/                   # 提示词模板（.md，内置极简引擎，**非 Jinja2**）
  templates/
    paper/                   # LaTeX 骨架
    experiment/              # 实验代码骨架
  tests/
```

**沙箱与外部工具一律禁止写入上述树之外的位置**，除运行根目录
`.autoresearch/runs/<run_id>/`（通过 `RunContext.run_dir` 获得）。

---

## 1. `config.py`

```python
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path          # = autoresearch/ 目录
WORKSPACE_ROOT: Path        # = 上一级工作区目录
DEFAULT_RUNS_DIR: Path      # = WORKSPACE_ROOT/.autoresearch/runs

@dataclass
class LLMConfig:
    provider: str = "openai"        # openai|deepseek|ollama|mock
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None      # 只从环境变量读，绝不落盘
    temperature: float = 0.3
    max_tokens: int = 8192   # deepseek 由 provider 默认值抬到 32768
    timeout: float = 120.0
    max_retries: int = 3
    json_repair_attempts: int = 2

@dataclass
class SandboxConfig:
    backend: str = "subprocess"     # subprocess|docker
    timeout: int = 900
    memory_mb: int = 4096
    cpus: float = 2.0
    allow_network: bool = True
    docker_image: str = "python:3.11-slim"
    extra_deny: list[str] = field(default_factory=list)
    max_file_mb: int = 4096   # 仅 POSIX 生效（RLIMIT_FSIZE）

@dataclass
class RetrieveConfig:
    sources: list[str] = field(default_factory=lambda: ["arxiv", "s2", "openalex", "crossref"])
    max_results_per_query: int = 8
    cache_dir: Path | None = None
    offline: bool = False
    timeout: float = 20.0
    mailto: str = "autoresearch@example.org"

@dataclass
class CompileConfig:
    engine: str = "tectonic"        # tectonic|pdflatex|xelatex|none
    auto_install_tectonic: bool = True
    tectonic_version: str = "0.15.0"
    venv_bin_dir: Path | None = None

@dataclass
class AutoResearchConfig:
    run_id: str = ""
    runs_dir: Path = DEFAULT_RUNS_DIR
    direction: str = ""
    venue: str = "NeurIPS"
    language: str = "zh"            # zh|en，控制产出报告语言
    seed: int = 0
    max_review_rounds: int = 3
    max_debug_rounds: int = 4
    max_ideas: int = 6
    keep_top_ideas: int = 3
    experiment_adapter: str | None = None   # 内置名 / .py 路径 / module:Class / entry point
    adapter_params: dict[str, Any] = field(default_factory=dict)
    max_variants: int = 6                   # 臂数上限，取与适配器 max_variants 的较小值
    resume: bool = False
    dry_run: bool = False
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: dict) -> "AutoResearchConfig": ...

def load_config(**overrides) -> AutoResearchConfig: ...
    # 优先级：显式 overrides > 环境变量 AUTORESEARCH_* > .env 文件 > 默认值
    # 环境变量清单见 §7

def load_dotenv(path: Path | None = None) -> dict[str, str]: ...
    # 极简 .env 解析（KEY=VALUE，# 注释，引号剥离），不写回 os.environ 之外
```

## 2. `logging_utils.py`

```python
def get_logger(name: str) -> logging.Logger
def configure_logging(level: str = "INFO", log_file: Path | None = None, quiet: bool = False) -> None
def event_logger(run_dir: Path) -> "EventLogger"

class EventLogger:
    """追加写 run_dir/events.jsonl，每行一个 JSON 对象。线程安全。"""
    def __init__(self, run_dir: Path) -> None: ...
    def log(self, event: str, **fields: Any) -> None
    def tail(self, n: int = 20) -> list[dict]
```

## 3. `llm/client.py`

```python
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    def __add__(self, other: "Usage") -> "Usage": ...

@dataclass
class LLMResponse:
    text: str
    raw: dict | None = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    cached: bool = False

class LLMError(RuntimeError): ...
class BackendError(LLMError): ...        # 网络/服务端错误，可重试
class ParseError(LLMError): ...          # JSON 解析失败
class BudgetExceeded(LLMError): ...      # mock/预算保护

class LLMBackend(Protocol):
    name: str
    def complete(self, prompt: str, system: str | None = None,
                 temperature: float | None = None,
                 max_tokens: int | None = None,
                 json_mode: bool = False) -> LLMResponse: ...

class LLMClient:
    def __init__(self, cfg: LLMConfig, cache_dir: Path | None = None,
                 event_logger: "EventLogger | None" = None) -> None: ...
    def backend(self) -> LLMBackend
    def complete(self, prompt: str, system: str | None = None, **kw) -> str
    def complete_json(self, prompt: str, system: str | None = None,
                      schema_hint: dict | None = None,
                      repair_attempts: int | None = None,
                      default: Any = None) -> Any
        # 解析失败 -> 自修复重试 repair_attempts 次 -> 仍失败则返回 default
        # 若 default is None 且解析失败 -> raise ParseError
    def usage(self) -> Usage
    def reset_usage(self) -> None
    def cache_stats(self) -> dict[str, int]
```

- 缓存：`cache_dir/<sha256(model+system+prompt+temp)>.json`，命中时 `LLMResponse.cached=True` 且不计 usage。
- 事件：每次调用向 event_logger 记 `llm_call`（含 model/tokens/latency/cached/ok）。
- `mock` provider → `llm/mock.py::MockBackend`。

## 4. `tools/retrieve.py`

```python
@dataclass
class Paper:
    id: str                   # "arxiv:2401.00001" / "doi:10.1/x" / "s2:<hash>"
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    url: str = ""
    pdf_url: str = ""
    citation_count: int = 0
    source: str = ""          # arxiv|s2|openalex|crossref
    tldr: str = ""
    keywords: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Paper": ...
    def key(self) -> str      # 归一化去重键（标题小写去标点）

class RetrievalError(RuntimeError): ...

class LiteratureSearch:
    def __init__(self, cfg: RetrieveConfig, cache_dir: Path | None = None,
                 event_logger=None) -> None: ...
    def search_arxiv(self, query: str, max_results: int = 8, sort_by: str = "relevance") -> list[Paper]
    def search_semantic_scholar(self, query: str, max_results: int = 8, year: str = "") -> list[Paper]
    def search_openalex(self, query: str, max_results: int = 8) -> list[Paper]
    def search_crossref(self, query: str, max_results: int = 8) -> list[Paper]
    def search(self, query: str, max_results: int | None = None,
               sources: list[str] | None = None) -> list[Paper]   # 聚合+去重+按相关度排序
    def multi_search(self, queries: list[str], per_query: int = 6) -> list[Paper]
    def similar_to(self, paper: Paper, max_results: int = 8) -> list[Paper]  # S2 recommendations 优先，失败回退关键词
    def to_bibtex(self, papers: list[Paper]) -> str
    def format_review(self, papers: list[Paper], max_papers: int = 12) -> str  # 综述脉络 markdown
```

- **必须**：`cfg.offline=True` 或所有源失败时，返回 `[]` 并记 `retrieve_fail` 事件，**绝不 raise 到管线顶层**（检索失败不能中断管线）。
- 用 `urllib.request` + `xml.etree` 实现（零第三方依赖），`requests` 可作可选加速。
- 磁盘缓存：`cache_dir/<source>_<sha1(query+params)>.json`，TTL 7 天（`time.time()` 判断）。

## 5. `tools/pdfx.py`

```python
def extract_text(pdf_path: Path, max_pages: int | None = None) -> str
def extract_sections(pdf_path: Path) -> dict[str, str]
    # {"abstract":..., "introduction":..., ...}；仅按常见标题启发式切分
def extract_references(pdf_path: Path) -> list[str]
def extract_metadata(pdf_path: Path) -> dict     # title/authors/year/doi，尽力而为
def download_pdf(url: str, dest: Path, timeout: float = 60.0) -> Path | None
def pdf_to_markdown(pdf_path: Path, max_pages: int | None = None) -> str
```
- 首选 `fitz`(PyMuPDF)，ImportError 时回退 `pdftotext`/返回空串并记警告。

## 6. `tools/sandbox.py`

```python
@dataclass
class ExecResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    oom: bool = False
    backend: str = ""
    cmd: list[str] = field(default_factory=list)
    def tail(self, n: int = 4000) -> str
    def to_dict(self) -> dict: ...

class SandboxError(RuntimeError): ...
class SecurityViolation(SandboxError): ...

class Sandbox:
    """统一执行接口。实现类：SubprocessSandbox, DockerSandbox"""
    name: str
    def __init__(self, cfg: SandboxConfig, workdir: Path, event_logger=None) -> None: ...
    def run(self, cmd: list[str], timeout: int | None = None, env: dict | None = None,
            cwd: str | None = None) -> ExecResult: ...
    def run_python(self, code: str | None = None, script: str | None = None,
                   args: list[str] | None = None, timeout: int | None = None,
                   allow_network: bool | None = None) -> ExecResult: ...
    def available(self) -> bool: ...
    def describe(self) -> dict: ...

def make_sandbox(cfg: SandboxConfig, workdir: Path, event_logger=None) -> Sandbox
    # cfg.backend=="docker" 且 docker 可用 -> DockerSandbox；否则降级 SubprocessSandbox 并记降级事件
def scan_code(code: str, extra_deny: list[str] | None = None) -> list[str]
    # 返回命中的危险模式列表（危险 shell 删除、fork bomb、写系统目录、提权等）
```

- SubprocessSandbox：Windows 用 `CREATE_NEW_PROCESS_GROUP` + `subprocess.run(timeout=)`；POSIX 用 `preexec_fn` 设 `resource.setrlimit`（CPU/内存/文件大小），两者都做 **代码静态扫描**（`scan_code`），命中即 `raise SecurityViolation`（除非 `extra_deny` 显式放行）。
- 超时 → `timed_out=True, ok=False`，不 raise。
- Docker：`docker run --rm -v {workdir}:/work -w /work [--network none] --memory --cpus`；docker 不存在时 `available()` 返回 False。

## 7. `tools/latex.py`

```python
class LatexError(RuntimeError): ...
@dataclass
class CompileResult:
    ok: bool
    pdf: Path | None
    log: str
    errors: list[str]        # 从日志抽取的 "! ..." 行
    warnings: list[str]
    engine: str = ""
    duration: float = 0.0
    def summary(self) -> str: ...

class LatexCompiler:
    def __init__(self, cfg: CompileConfig, workdir: Path, event_logger=None) -> None: ...
    def detect(self) -> str | None            # 返回可用引擎名或 None
    def install_tectonic(self) -> Path | None # 下载单文件二进制到 PROJECT_ROOT/vendor/tectonic/
    def compile(self, tex_file: Path, runs: int = 2, timeout: int = 300,
                engine: str | None = None) -> CompileResult: ...
    def extract_errors(self, log: str) -> list[str]: ...
    def bibtex_available(self) -> bool: ...
```

- `tectonic` 下载 URL 模板：`https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{version}/tectonic-{version}-x86_64-pc-windows-msvc.zip`（Windows）/ `...-x86_64-unknown-linux-musl.tar.gz`（Linux）。
- 下载目标 `PROJECT_ROOT/vendor/tectonic/`；**必须校验** `--version` 可执行。
- 网络不可用 → 返回 None 并记事件，不 raise。
- 编译时 tectonic 加 `--keep-logs --outdir <workdir>`；引擎为 pdflatex/xelatex 时连续跑 `runs` 次。

## 8. `tools/figures.py` + `tools/metrics.py`

```python
# metrics.py
def parse_metrics(path: Path) -> dict[str, list[float]]
    # 支持 CSV / JSONL / 纯文本 "epoch=1 loss=0.5 acc=0.9" / tensorboard-free 格式
def summarize(series: dict[str, list[float]]) -> dict[str, dict[str, float]]
    # {"loss": {"mean":..,"std":..,"min":..,"max":..,"final":..,"best":..}}
def compare_runs(runs: dict[str, dict[str, list[float]]]) -> "pd.DataFrame"
def to_latex_table(rows: list[dict], caption: str = "", label: str = "",
                   decimals: int = 2, bold_best: bool = True,
                   higher_is_better: dict[str, bool] | None = None) -> str
    # 标准三线表 booktabs：\toprule \midrule \bottomrule

# figures.py
def setup_style(venue: str = "NeurIPS") -> None
def plot_learning_curves(series: dict[str, dict[str, list[float]]], out_dir: Path,
                         name: str = "learning_curves", formats=("pdf", "png")) -> list[Path]
def plot_bar_comparison(summary: dict[str, dict[str, float]], metric: str, out_dir: Path,
                        name: str = "comparison", formats=("pdf", "png")) -> list[Path]
def plot_ablation(df, metric: str, out_dir: Path, name="ablation", formats=("pdf","png")) -> list[Path]
def plot_boxplot(runs: dict[str, list[float]], metric: str, out_dir: Path,
                 name="boxplot", formats=("pdf","png")) -> list[Path]
def plot_metric_grid(series_map: dict[str, dict[str, list[float]]], out_dir: Path,
                     name: str = "metric_grid", formats=("pdf","png")) -> list[Path]
def make_all_figures(metrics_dir: Path, out_dir: Path, formats=("pdf","png")) -> dict[str, list[Path]]
```
- 全部使用 `matplotlib.use("Agg")`；输出**同时** PDF(矢量) 与 PNG(300dpi)。
- 无数据时必须返回 `[]` 并记事件，不 raise。

## 9. `graph/state.py`（状态 schema）

```python
STAGES: tuple[str, ...] = (
    "s1_literature", "s2_ideation", "s3_planning", "s4_experiment",
    "s5_analysis", "s6_writing", "s7_compile", "s8_review", "s9_finalize",
)

@dataclass
class Artifact:
    path: str          # 相对 run_dir 的 POSIX 路径
    kind: str          # json|md|csv|py|tex|pdf|png|log|bib
    stage: str
    sha256: str = ""
    bytes: int = 0
    def to_dict(self) -> dict: ...

def new_state(run_id: str = "", direction: str = "", venue: str = "NeurIPS",
              language: str = "zh", seed: int = 0) -> dict[str, Any]
    # 完整键见下表（缺失键由 stage 自行 get）

def state_schema() -> dict[str, Any]:   # 文档用
```

状态键（顶层，扁平，值必须 JSON 可序列化）。**本表由 `graph/state.py::STATE_DEFAULTS` 生成**，
改动状态 schema 时必须同步刷新，避免文档与代码漂移。

**运行元信息（runner 写入，之后只读）**

| key | 类型 |
|---|---|
| `run_id` | str |
| `direction` | str |
| `venue` | str |
| `language` | str |
| `seed` | int |
| `started_at` | str |

**引擎簿记**

| key | 类型 |
|---|---|
| `stage_status` | dict |
| `stage_attempts` | dict |
| `stage_errors` | dict |
| `trace` | list |
| `current_stage` | str |
| `steps` | int |

**产物登记**

| key | 类型 |
|---|---|
| `artifacts` | list |

**s1 文献**

| key | 类型 |
|---|---|
| `queries` | list |
| `papers` | list |
| `lit_review` | str |
| `themes` | list |
| `gaps` | list |
| `bib_entries` | list |

**s2 构思**

| key | 类型 |
|---|---|
| `ideas` | list |
| `selected_idea` | dict |
| `novelty_summary` | dict |

**s3 规划**

| key | 类型 |
|---|---|
| `experiment_plan` | dict |

**s4 实验**

| key | 类型 |
|---|---|
| `code_files` | list |
| `workspace` | str |
| `baseline_results` | dict |
| `method_results` | dict |
| `debug_history` | list |
| `runs_executed` | list |

**s5 分析**

| key | 类型 |
|---|---|
| `metrics` | dict |
| `metrics_summary` | dict |
| `figured` | dict |
| `latex_tables` | dict |
| `analysis` | dict |

**s6 写作**

| key | 类型 |
|---|---|
| `paper_sections` | dict |
| `paper_title` | str |
| `paper_abstract` | str |
| `paper_tex` | str |
| `citations_used` | list |

**s7 编译**

| key | 类型 |
|---|---|
| `compile_result` | dict |
| `compile_fix_history` | list |

**s8 评审**

| key | 类型 |
|---|---|
| `reviews` | list |
| `review_score` | float |
| `review_verdict` | str |
| `review_round` | int |
| `revision_history` | list |

**s9 交付**

| key | 类型 |
|---|---|
| `final_pdf` | str |
| `final_report` | str |
| `deliverables` | list |

**全局**

| key | 类型 |
|---|---|
| `errors` | list |
| `warnings` | list |
| `notes` | dict |

`Idea` 形状（`s2` 内定义，JSON 化后进状态）：
```python
{
  "id": "I1", "title": str, "hypothesis": str, "motivation": str,
  "method_sketch": str, "novelty_claim": str, "feasibility": str,
  "risks": [str], "expected_metrics": [str],
  "novelty": {"verdict": "novel|incremental|duplicate|unknown",
              "closest": [{"title":str,"id":str,"year":int,"why":str}],
              "score": float, "rationale": str},
  "pilot": {"ok": bool, "cmd": [str], "result": str, "metrics": dict},
  "rank": float, "selected": bool
}
```

## 10. `stages/base.py`

```python
@dataclass
class StageResult:
    ok: bool
    state_updates: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    detail: str = ""
    fatal: bool = False          # True 则引擎立即终止（默认 False）
    retry: bool = False          # True 请求引擎重试本阶段

class Stage(ABC):
    name: str
    title: str
    requires: tuple[str, ...] = ()      # 依赖的状态键，缺失时记 warning
    produces: tuple[str, ...] = ()
    max_attempts: int = 2
    def __init__(self, ctx: "RunContext") -> None: ...
    @abstractmethod
    def run(self, state: dict) -> StageResult: ...
```

`RunContext`（`runner.py`）：
```python
@dataclass
class RunContext:
    cfg: AutoResearchConfig
    run_dir: Path
    llm: LLMClient
    events: EventLogger
    log: logging.Logger
    sandbox: Sandbox
    search: LiteratureSearch
    compiler: LatexCompiler
    prompts: "PromptLibrary"
    # 便捷方法
    def path(self, *parts: str) -> Path          # run_dir 下路径，自动建父目录
    def rel(self, p: Path) -> str                # 相对 run_dir 的 POSIX 路径
    def save_json(self, rel_path: str, obj: Any) -> Artifact
    def save_text(self, rel_path: str, text: str, kind: str = "md") -> Artifact
    def artifact(self, p: Path, kind: str) -> Artifact
    def load_json(self, rel_path: str, default=None) -> Any
    def log_event(self, event: str, **fields) -> None
```

`PromptLibrary`（`prompts/__init__.py`）：
```python
class PromptLibrary:
    def __init__(self, prompts_dir: Path, language: str = "zh") -> None
    def render(self, name: str, **vars) -> str   # name 对应 prompts/<name>.md
    def raw(self, name: str) -> str
```

## 11. `graph/engine.py`

```python
@dataclass
class Node:
    name: str
    fn: Callable[[dict], StageResult]
    max_attempts: int = 2
    optional: bool = False
    router: Callable[[dict], str | None] | None = None
        # 返回下一节点名；None 表示按声明顺序继续；"__end__" 终止

class GraphEngine:
    def __init__(self, nodes: list[Node], checkpoint: "Checkpointer | None" = None,
                 max_steps: int = 200, stop_on_error: bool = False,
                 event_logger=None, logger=None,
                 max_visits: int = 4, on_step=None) -> None: ...
    def run(self, state: dict, start_at: str | None = None) -> dict
    def trace(self) -> list[dict]
```

- 节点抛异常 → 捕获，记 `stage_errors`，`max_attempts` 内重试；耗尽后：`optional=True` 记 skipped 继续，否则若 `stop_on_error` 则终止，否则记 failed 继续后续节点（**尽量产出部分结果**）。
- 每步之后调用 `checkpoint.save(state)`。
- 有 `router` 的节点：路由到目标；目标若已 done 则跳过（防死循环），并计步数上限。

## 12. CLI

```
python -m autoresearch.cli run --direction "..." [--venue NeurIPS] [--runs-dir ...]
      [--llm-provider mock|openai|deepseek|ollama] [--model ...] [--offline]
      [--sandbox subprocess|docker] [--max-review-rounds 3] [--no-resume]
      [--stop-on-error]
      [--experiment-adapter SPEC] [--adapter-arg K=V ...] [--max-variants N]
      [--dry-run] [--language zh|en] [--quiet] [--json]
python -m autoresearch.cli resume <RUN_ID> [...]
python -m autoresearch.cli status <RUN_ID>
python -m autoresearch.cli verify <RUN_ID>   # 论文数字溯源校验
python -m autoresearch.cli doctor          # 环境自检：LLM/检索/沙箱/编译
python -m autoresearch.cli stages          # 列出阶段与依赖
python -m autoresearch.cli demo            # 离线端到端冒烟
```
**退出码**：0 成功；1 部分失败（有阶段 failed 但产出 PDF/报告）；2 致命失败。

## 13. 事件名清单（EventLogger）

事件日志是这套管线的审计底座：`events.jsonl` 里每一次 LLM 调用、沙箱执行、路由
决策、补丁与阻断都有记录。**新增事件必须同步更新本节**——清单与代码不一致时，
读日志的人会误判「某件事没发生」。

```
运行级    run_start, run_end, run_crash, run_interrupted, resume_start
阶段级    stage_start, stage_done, stage_retry, stage_failed, stage_skipped,
          stage_exception, stage_missing_input
图引擎    route, router_error, router_unknown_target, graph_abort, graph_cycle_break
LLM       llm_call, llm_truncated, llm_parse_failed
检索      retrieve_call, retrieve_fail, retrieve_cache_hit, retrieve_empty
沙箱      sandbox_run, sandbox_timeout, sandbox_violation, sandbox_degrade
编译      tectonic_install, compile_detect, compile_run, compile_fail, compile_blocked
实验      adapter_resolved, adapter_variants_filtered, code_generated, debug_round
分析      figure_made
构思      idea_generated, novelty_check
写作      paper_assembled, title_sanitized, citations_dropped, writing_iteration_noop,
          template_placeholders_unfilled
评审      review_round, review_loop, review_stop, revision_applied
持久化    checkpoint_save
```

## 14. 离线可跑原则（硬性）

`--llm-provider mock --offline` 必须能跑完整条管线并产出
`report/FINAL_REPORT.md` 与至少一个 PDF（若无可用 LaTeX 引擎，
则产出 `paper/main.tex` + `report/COMPILE_BLOCKED.md` 并正常退出）。
这是回归测试的基准，也是 CI 三平台矩阵能成立的前提：
**七个测试套件全部不需要网络与 API Key**。

## 15. 实验后端适配器（v1.1 新增）

### 15.1 职责边界（不可越界）

| 谁 | 负责 |
|---|---|
| 适配器（`adapters/`） | 「怎么跑」（`build_command`）+「怎么看结果」（`parse_results`）+ 环境预检、模板提供、变体能力声明 |
| `stages/s4_experiment.py` | 「写什么代码」（LLM）+「跑不通怎么修」（traceback 反思）+ **保真度护栏** |

**代码生成刻意不放进适配器**：否则调试闭环与保真度检查（补丁是否删掉指标输出、
是否加 `except: pass`、是否写死高分）对适配器产物完全失效——那正是自动科研最难
发现的一类错误。适配器只提供**模板**（`seed_code()`），LLM 负责按需生成与修补。

`owns_code = True` 是唯一的受控特例（用户明确要求「别让模型改我的脚本」）：
跳过 LLM 生成，但**调试闭环与保真度检查仍然生效**。

### 15.2 协议（`adapters/base.py`）

```python
@dataclass
class RunSpec:
    variant: str
    seed: int
    out_dir: str                                  # 工作区相对 POSIX 路径
    params: dict[str, Any] = field(default_factory=dict)   # 开放超参
    extra_args: list[str] = field(default_factory=list)    # 逃生舱

MetricSeries = dict[str, list[float]]             # 指标名 → 逐点序列（**必须**）

class BaseExperimentAdapter(ABC):
    name: str = "base"
    description: str = ""
    owns_code: bool = False
    entrypoint: str | None = None
    default_timeout: int | None = None
    max_variants: int | None = None   # 后端能承受的臂数上限，与配置取较小值

    def prepare(self, workspace: Path, plan: dict) -> None: ...
    def seed_code(self, workspace: Path, plan: dict) -> dict[str, str]: ...
    @abstractmethod
    def build_command(self, spec: RunSpec) -> list[str]: ...
    @abstractmethod
    def parse_results(self, out_dir: Path) -> MetricSeries: ...
    def validate_environment(self) -> tuple[bool, str]: ...
    def supported_variants(self) -> set[str] | None: ...
    def code_files(self, workspace: Path) -> list[Path]: ...
    def describe(self) -> dict: ...
    def quality_note(self) -> str: ...
    # 供 s4 构造代码生成提示词：把「本后端期望什么样的脚本」交还给适配器，
    # 而不是让阶段写死合成任务的约定（那正是重构前的病根）
    def codegen_conventions(self) -> str: ...
    def codegen_data_info(self) -> str: ...
    def codegen_variants(self) -> str: ...
```

模块级函数：`resolve_adapter(spec, params) -> BaseExperimentAdapter`、
`builtin_adapters() -> dict[str, type]`、`coerce_metric_series(raw)`、
`higher_is_better(name)`、`read_standard_metrics(out_dir)`；
异常 `AdapterError`；entry point 组名 `autoresearch.adapters`。

### 15.3 硬性约束

1. `parse_results()` 必须返回 `MetricSeries`。**返回标量而非序列会让管线约 30 张图、
   收敛判断与逐 epoch 统计同时失去意义**；结构非法时 `coerce_metric_series` 立即
   抛 `AdapterError`，不静默降级。
2. `build_command()` 返回完整 argv，**不经 shell**。第一个元素为 `"python"` 时由
   沙箱替换为 `sys.executable`。
3. 主对照 `baseline` / `method` 必须使用**同一套默认超参**（`params` 为空），
   否则「提升」无法归因到方法本身。
4. 消融格点的变体名是 `<轴名>=<取值>`（如 `momentum=0.9 (on)`）——保留轴名，
   否则消融曲线无法解读。
5. `quality_note()` 的文本会进入交付报告；适配器必须如实声明结果能支撑什么级别的
   结论，默认实现给出保守提示。
6. 臂数上限取 ``min(cfg.max_variants, adapter.max_variants)``：配置是用户本次预算，
   适配器上限是后端能力约束，**取小值**意味着用户不能把一个只愿意跑 2 臂的后端推到
   6 臂（「跑不完的消融」比「没有消融」更糟）。
7. 消融臂的**变体名是中性标识**（``abl-1``），轴与取值通过 ``RunSpec.params`` 传递
   （``{"momentum": 0.9}``）。把语义压进变体名等于强迫所有适配器解析管线的命名
   约定；而且 ``momentum=0.9 (on)`` 这类含空格/括号的名字进 argv 是单 token，
   极易出错。

### 15.4 沙箱新增接口（§6 的增补）

```python
def run_command(self, argv: list[str], timeout: int | None = None,
                env: dict | None = None, cwd: str | None = None,
                allow_network: bool | None = None) -> ExecResult: ...
```
- 字符串参数会被拒绝并给出可操作提示；含 shell 元字符的单元素命令同样被拒绝
  （不经 shell 时它们静默失效，报错会指向「找不到文件」这种无关原因）。
- POSIX 上 `preexec_fn + resource.setrlimit` 与 `start_new_session=True` **并存**：
  前者让 `memory_mb` / `max_file_mb` 真正生效，后者让整组进程可被回收。
  此前二者二选一，导致这两个配置项「存在但完全无效」。
