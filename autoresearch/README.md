# Auto-Research：自动化科研与论文生成管线

把科研全生命周期拆成九个**离散、可单独重跑**的阶段，用 LLM + 检索增强 + 沙箱执行 +
LaTeX 编译串成一条可断点续跑的管线。目标不是"一键生成一篇能中的论文"，而是
**把科研过程变得可审计、可复现、并且诚实地报告自己失败在哪一步**。

```text
① 文献检索 ──▶ ② 构思+查新 ──▶ ③ 实验规划 ──▶ ④ 沙箱执行+自纠错
                                                      │
       ⑨ 交付打包 ◀── ⑧ 评审+迭代 ◀── ⑦ LaTeX编译 ◀── ⑥ 分章撰写 ◀── ⑤ 分析制图
                          │
                          └── 分数不达标且仍有预算 → 回到 ⑥
```

---

## 当前能力与明确非目标

这一节放在最前面，因为开源项目最贵的成本是**用户预期错配**。

### ✅ 现在就能用

| 能力 | 说明 |
|---|---|
| 多源文献检索与结构化综述 | arXiv / Semantic Scholar / OpenAlex / Crossref 四源聚合、去重、主题聚类、缺口识别；单源失败不影响整体 |
| 假设生成 + 新颖性查新 | 语义扩散生成候选，对照近期文献判定 `novel`/`incremental`/`duplicate`/`unknown`，**证据不足时给 `unknown` 而不是硬凑 `novel`** |
| 可插拔实验后端 | 内置合成任务适配器 + `ScriptWrapperAdapter`（接你自己的训练脚本）；第三方可发布适配器包 |
| 沙箱内执行与自纠错 | 静态危险扫描、超时杀进程树、输出截断、traceback → 反思 → 最小补丁闭环 |
| **实验保真度护栏** | 补丁若删掉指标输出、加 `except: pass`、写死高分返回值，会被标记 `validity_flag` 并进入交付报告的未解决问题清单 |
| 多种子统计 | 每个变体多随机种子，**先在种子上取标量、再跨种子算 mean±std**（不是把所有 epoch 混在一起算方差） |
| 确定性图表与三线表 | 学习曲线 / 柱状 / 箱线 / 消融图（PDF+PNG）+ booktabs 表格，全部由代码生成，LLM 只负责解释 |
| 分章论文撰写 | 自底向上（method → experiments → results → related → intro → abstract），引用闭集校验 |
| LaTeX 编译与错误修复 | tectonic 自动拉取 + 编译 + 错误抽取 + LLM 补丁重编译 |
| 顶会标准自动评审 | 分数/verdict/分项评分 → 不达标回环修改，带收益递减检测 |
| **论文数字溯源校验** | `cli verify` 抽取论文里每个数值 token 与证据表比对，输出未溯源数字清单 |
| 断点续跑 | 每阶段落检查点；进程被杀后 `cli resume` 继续，不重跑已完成阶段 |
| 零依赖离线模式 | `cli demo` 在裸 Python 上跑完整条管线（无需网络、无需 API Key、无需 LaTeX） |

### ❌ 明确不做（当前版本）

| 非目标 | 现状 |
|---|---|
| GPU 调度 / 多机分布式 | 沙箱只做进程级隔离；GPU 由适配器自己声明与使用（如 `torchrun --nproc_per_node`） |
| 数据集管理 | 不下载、不缓存、不切分数据集；数据准备属于适配器的 `prepare()` |
| 依赖安装 | 不为实验安装 torch/HF 等；`validate_environment()` 只做**预检**并说清缺什么 |
| 长训练与断点续训 | 「断点续跑」指的是**管线**层面；单次实验内部的 checkpoint 恢复由用户脚本负责 |
| 超参搜索 | 只跑 s3 计划里声明的臂；不做贝叶斯/网格搜索 |
| 真实同行评审替代品 | s8 是**质量门禁**，不是同行评审；默认保守给分，不能当作录用预测 |
| 语义级事实验证 | `cli verify` 是**范围检查**：保证数字能落到证据网格上，但不保证语义正确（把 accuracy 写成 f1 仍会通过） |
| 湿实验 / 需要人工介入的流程 | 只支持可被一条 argv 启动、能把指标写到文件的实验 |

### 一句话定位

> 一个可审计、可断点续跑、会诚实报告失败位置的自动科研**流水线框架**，
> 附带一个受控小型实验的默认实现。**文献/构思/规划/分析/写作/评审/溯源链路是领域无关的；
> 实验执行层通过适配器扩展**——默认只支持「输出指标文件的小型 CPU 实验」。

---

## 快速开始

```powershell
# 0. 环境自检（LLM Key / 检索 / 沙箱 / 编译器 / 提示词 / 模板 / 适配器）
python -m autoresearch.cli doctor

# 1. 零依赖冒烟：mock LLM + 无网络，验证管线连通与降级路径（约 9 秒）
python -m autoresearch.cli demo

# 2. 真实运行（需要 API Key；provider 由 Key 自动推断）
$env:DEEPSEEK_API_KEY = "sk-..."
python -m autoresearch.cli run --direction "稀疏注意力在长序列上的效率-精度权衡" --venue NeurIPS

# 3. 接入你自己的训练脚本（不改代码，见 docs/adapters.md）
python -m autoresearch.cli run --direction "..." `
    --experiment-adapter script-wrapper `
    --adapter-arg script=my_train.py --adapter-arg program=python

# 4. 断点续跑（进程被杀 / 阶段失败后）
python -m autoresearch.cli resume <run_id>

# 5. 校验论文里的每个数字是否都能溯源到实验证据
python -m autoresearch.cli verify <run_id>
```

退出码：`0` 完整成功 / `1` 部分完成（有阶段失败或未编译出 PDF）/ `2` 致命失败。

## 九阶段职责

| 阶段 | 做什么 | 产物 | 失败后果 |
|---|---|---|---|
| `s1_literature` | 生成检索式 → 多源检索（arXiv / S2 / OpenAlex / Crossref）→ 主题聚类与缺口 | `literature/review.md` `papers.json` | **致命**（`fatal=True`） |
| `s2_ideation` | 语义扩散生成假设 → **查新淘汰** → 加权排序选出唯一方案 | `ideas/IDEATION_REPORT.md` | **致命**（`fatal=True`） |
| `s3_planning` | claim 驱动的里程碑、成功判据、消融矩阵、预算 | `plan/EXPERIMENT_PLAN.md` | 退出码计为 claim-fatal（无计划即无法实验） |
| `s4_experiment` | **经适配器**构造命令 → 静态危险扫描 → **多种子**执行 → Traceback 自纠错 | `experiment/EXPERIMENT_RESULTS.md` | **致命**（`fatal=True`，无实验即无 claim） |
| `s5_analysis` | 指标解析 → 跨种子统计 → 矢量图 + booktabs 三线表 → 证据分级 | `analysis/RESULTS_ANALYSIS.md` `figures/` | 可选（`OPTIONAL_STAGES`，跳过继续） |
| `s6_writing` | 自底向上分章撰写；回环时只应用评审修订，不重新生成 | `paper/main.tex` `sections/*.tex` | 非致命，但无正文则退出码为部分完成 |
| `s7_compile` | tectonic/pdflatex 编译 + 错误修复；无引擎则写 `COMPILE_BLOCKED.md` | `paper/main.pdf` 或阻断说明 | 可选（`OPTIONAL_STAGES`，跳过继续） |
| `s8_review` | 顶会标准评审 → 分数/verdict → 不达标回 `s6` 修改（带收益递减检测） | `review/AUTO_REVIEW.md` | 非致命，但交付报告会标出"未经评审" |
| `s9_finalize` | 打包交付物 + 最终报告 + **未解决问题清单** | `report/FINAL_REPORT.md` `06_deliverables/` | 非致命，但交付目录不完整 |

> **三种"致命"不是一回事**，上表刻意分开写：
> `fatal=True` 是阶段自己声明「没有我就没有意义」，引擎立即终止（只有 s1/s2/s4）；
> **claim-fatal** 是 `runner.compute_exit_code` 层面的判定（s1/s2/s3/s4 的失败会把退出码降级）；
> 「非致命」的阶段失败后仍会继续，产出带缺口但诚实的报告。

## 设计取舍（为什么这样做）

### 1. 自研状态机，LangGraph 作为可选适配层

引擎的核心语义只有「节点 + 条件路由 + 重试 + 检查点」四件事，用标准库约 250 行实现
（`graph/engine.py`），换来零依赖。`graph/langgraph_adapter.py` 用同一份 `list[Node]`
编译出等价的 `StateGraph`，装了 `langgraph` 就能切过去，**阶段代码一行不改**。

### 2. 失败要降级，不要崩

`OPTIONAL_STAGES`（分析、编译）失败后跳过并继续；其他阶段失败也默认继续，
因为它们后面的阶段仍能产出**部分**结果与一份诚实的报告。
整条管线只有 3 个阶段把 `fatal=True`（检索、构思、实验）——没有实验就没有论文，
继续跑只是在烧 token。另有「规划」参与**退出码**层面的致命判定：
`runner.compute_exit_code` 把 s1/s2/s3/s4 的失败都计为 claim-fatal。

### 3. 数字只能来自证据，不能来自 LLM

* 图表由 `tools/figures.py` + `tools/metrics.py` **确定性**生成，LLM 只被允许「解释」；
* 写作提示词里明确要求数值逐字引用证据块，禁止改写；
* `s5` 做**证据分级**（supported / partially_supported / not_supported / inconclusive），
  写作时必须遵守分级，不能把 `not_supported` 写成 `supported`；
* `cli verify` 事后抽取论文里的每个数值 token，与证据表逐一匹配，
  把「上千个数字」缩到「几个需要人看的东西」。它会抓出真实存在的幻觉数字
  （在 mock 冒烟里就抓到过「验证集在 20 epoch 后进入平台期」——实际只跑了 3 个 epoch）。

### 4. 统计口径：先在种子上取标量，再跨种子求 mean±std

实验默认对每个变体跑 3 个种子（`_SEEDS_PER_VARIANT`）。汇总时**不是**把所有 epoch
的点混在一起算方差——那算的是"收敛轨迹的散布"，会让误差棒系统性偏大。
正确做法是每个种子先取一个标量（best/final），再在种子维度上求均值与标准差。
`s5._cross_seed_summary` 实现了这一点，论文主表用的就是它。

### 5. 实验保真度不许被"修好"

自纠错会检查每一轮补丁：如果补丁删掉了 `metrics.csv` 输出、加了 `except: pass`、
或写死了高分返回值，会被标记 `validity_flag`，并出现在最终报告的**未解决问题**里。
指标可以难看，但不能是假的——这是自动科研最容易被静默破坏的一环。

### 6. 交付物包含"我哪里没做到"

`report/open_issues.json` 与 `FINAL_REPORT.md` 第 3 节列出所有 major/minor 未解决问题：
未修复的评审弱点、无法用现有证据解决的异议、保真度告警、失败阶段、编译阻断原因。
一份只报喜的自动科研报告是不可用的；**承认边界**才是它能被信任的前提。

### 7. 实验后端可插拔，但代码生成的主动权留在管线

`s4_experiment` 早期把「实验怎么跑」硬编码在阶段内部，后果是**整个项目只能跑小型
CPU 分类实验**。现在这部分被抽成 `autoresearch/adapters/`：适配器负责「怎么跑」
（`build_command`）与「怎么看结果」（`parse_results`），阶段负责编排、调试闭环与保真度护栏。

**为什么代码生成不交给适配器**（这是一个有代价的取舍）：如果适配器自己生成代码、
自己修 bug，那么「补丁删掉了指标输出」「补丁加了 `except: pass`」「补丁里写死高分」
这类问题就**不会流经保真度检查点**——而"指标被静默破坏"正是自动科研里最难发现的一类错误。
所以适配器只提供**模板**，LLM 负责按需生成与修补，任何适配器的产物都受同一套质量约束。

`owns_code = True` 是唯一的受控特例（用户明确要求「别让模型改我的训练脚本」）：
跳过 LLM 生成，但**调试闭环与保真度检查依然生效**。「这是用户的脚本」不构成绕过质量门禁的理由。

详见 [docs/adapters.md](../docs/adapters.md)。

## 实验后端适配器

```powershell
# 内置默认：受控合成分类任务（零依赖，用于 CI 与首次体验）
python -m autoresearch.cli run --direction "..."

# 用户自带脚本：跳过 LLM 代码生成，只负责跑起来与取指标
python -m autoresearch.cli run --direction "..." `
    --experiment-adapter script-wrapper `
    --adapter-arg script=path/to/train.py `
    --adapter-arg program=python `
    --adapter-arg metrics_file=metrics.csv

# 自己的适配器：.py 文件 / module:Class / 已安装包的 entry point
python -m autoresearch.cli run --direction "..." --experiment-adapter adapters/my_sim.py
python -m autoresearch.cli run --direction "..." --experiment-adapter my_pkg.adapters:MyAdapter
```

最小适配器只需实现两个方法：

```python
from autoresearch.adapters import BaseExperimentAdapter, RunSpec

class MyAdapter(BaseExperimentAdapter):
    name = "my-adapter"
    description = "在自有数据集上训练一个小模型"

    def build_command(self, spec: RunSpec) -> list[str]:
        # spec.params 承载来自 s3 消融矩阵的超参；spec.extra_args 是逃生舱
        return ["python", "train.py", "--seed", str(spec.seed),
                "--out", spec.out_dir, *spec.extra_args]

    def parse_results(self, out_dir) -> dict[str, list[float]]:
        # 必须返回 {指标名: 逐点序列}——返回标量会让约 30 张图与逐 epoch 统计失去意义
        return {"accuracy": [0.81, 0.88, 0.91]}
```

第三方包可以通过 entry point 注册适配器：

```toml
[project.entry-points."autoresearch.adapters"]
my-sim = "my_package.adapters:MySimAdapter"
```

## 环境要求

| 组件 | 必需 | 缺失后果 |
|---|---|---|
| Python ≥ 3.10 | ✅ | — |
| `openai` | 用真实 LLM 时必需 | 只能跑 `--llm-provider mock` |
| `matplotlib` `pandas` | 建议 | 无法制图与汇总（降级为内联统计） |
| `PyMuPDF`（`fitz`） | 可选 | 无法解析 PDF 全文 |
| LLM API Key | 真实运行时 | 用 `mock` 或降级为确定性回退 |
| tectonic 或 pdflatex | 可选 | 无 PDF，改为 `report/COMPILE_BLOCKED.md` |
| Docker | 可选 | 沙箱降级为受限子进程（静态扫描仍生效） |
| `langgraph` | 可选 | 用自研引擎（语义等价） |

**零第三方依赖的底线**：核心运行时 `dependencies = []`。`--llm-provider mock --offline`
能在只有 Python 标准库的机器上跑完整条管线并产出完整交付物（PDF 除外）。
这不是巧合而是设计目标——CI 的三平台矩阵**不安装任何第三方依赖**，就是为了让这条底线
每次提交都被验证一次。

## 配置

优先级：显式 CLI 参数 > `AUTORESEARCH_*` 环境变量 > `.env` 文件 > 默认值。

```ini
# .env（放在工作区根或 autoresearch/ 下）
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.deepseek.com/v1
# DEEPSEEK_API_KEY=...
AUTORESEARCH_VENUE=NeurIPS
AUTORESEARCH_LANGUAGE=zh
AUTORESEARCH_SANDBOX_BACKEND=subprocess
AUTORESEARCH_TECTONIC_VERSION=0.15.0
```

常用参数：

```powershell
python -m autoresearch.cli run `
  --direction "..." `
  --llm-provider deepseek --model deepseek-chat `
  --venue ICLR --language en --seed 0 `
  --sandbox subprocess --sandbox-timeout 900 `
  --max-review-rounds 3 --max-debug-rounds 4 --max-ideas 6 `
  --experiment-adapter synthetic-toy `
  --offline                       # 完全禁网（只用本地缓存/无检索）
```

## 目录结构

```text
autoresearch/
├─ cli.py              命令行入口（run/resume/status/verify/doctor/stages/demo）
├─ config.py           配置层（dataclass + .env 解析 + 环境变量）
├─ runner.py           RunContext 组装 + 管线启动 + 退出码
├─ verify.py           论文数字溯源校验
├─ adapters/           实验后端适配器（可插拔）
│  ├─ base.py           RunSpec + BaseExperimentAdapter 协议 + resolve_adapter
│  ├─ synthetic_toy.py  内置默认：受控合成分类任务
│  └─ script_wrapper.py 用户自带训练脚本
├─ graph/
│  ├─ state.py         状态 schema（扁平、JSON 可序列化）
│  ├─ checkpoint.py    检查点与断点续跑合并规则
│  ├─ engine.py        自研状态机（路由/重试/降级/环路保护）
│  └─ langgraph_adapter.py
├─ llm/
│  ├─ client.py        openai 兼容客户端 + 重试 + 磁盘缓存 + JSON 自修复
│  └─ mock.py          离线启发式后端（让全流程无 Key 可跑）
├─ tools/
│  ├─ retrieve.py      arXiv / Semantic Scholar / OpenAlex / Crossref
│  ├─ pdfx.py          PDF 全文/章节/参考文献提取
│  ├─ sandbox.py       子进程沙箱 + Docker 后端 + 静态危险扫描 + run_command
│  ├─ latex.py         tectonic 自动拉取 + 编译 + 错误抽取 + 可用性预检
│  ├─ fetch_tectonic.py tectonic 按需下载（二进制不入库）
│  ├─ metrics.py       指标解析 + 统计 + booktabs 三线表
│  └─ figures.py       学习曲线/柱状/箱线/消融图（PDF+PNG）
├─ stages/             s1…s9 阶段实现（base.py 定义 Stage/StageResult）
├─ prompts/            14 个阶段提示词 + 极简模板引擎（非 Jinja2）
├─ templates/
│  ├─ paper/           LaTeX 骨架
│  └─ experiment/      实验代码骨架（确定性、CPU 秒级、零下载）
├─ tests/              7 个测试套件（详情见下）
├─ CONTRACTS.md        模块接口契约（改签名前必读）
└─ README.md           本文件

仓库根目录另有：
├─ LICENSE             Apache-2.0
├─ pyproject.toml      PEP 621 打包与可选依赖组
├─ CONTRIBUTING.md     贡献指南（含"不变量"清单）
├─ .github/workflows/  CI：三平台矩阵 / 打包 / 静态检查含密钥扫描
├─ docs/adapters.md    适配器开发指南
└─ scratch/            开发期一次性脚本（不属于包）
```

运行产物（默认 `<workspace>/.autoresearch/runs/<run_id>/`）：

```text
state.json           最新状态（断点续跑读它）
checkpoints/         逐步快照（默认保留 20 份）
events.jsonl         结构化事件日志（每次 LLM 调用/沙箱执行/路由/补丁都有记录）
literature/          检索结果与综述
ideas/               候选假设、查新判定、IDEA 报告
plan/                实验计划与消融矩阵
experiment/          工作区、生成的代码、code_manifest.json、adapter.json、
                     results.json、debug_history.json、EXPERIMENT_RESULTS.md
metrics/<variant>/seed_N/   每个种子一份原始指标（跨种子统计的基础）
figures/             矢量图（PDF+PNG）
analysis/            指标汇总、图表清单、三线表、证据分级
paper/               main.tex + sections/ + references.bib + paper_meta.json
review/              每轮评审与修订记录、AUTO_REVIEW.md
report/              FINAL_REPORT.md（确定性报告）、RUN_SUMMARY.md（LLM 叙述）、
                     open_issues.json、manifest.json、REPRODUCE.md、
                     COMPILE_BLOCKED.md（无引擎时）
06_deliverables/     打包好的交付目录（论文+证据+评审+报告）
```

## 测试

```powershell
python -m autoresearch.tests.test_config_llm       # 配置 + LLM 客户端 + mock 意图路由
python -m autoresearch.tests.test_retrieve         # 检索 + PDF 解析（离线）
python -m autoresearch.tests.test_sandbox_latex    # 沙箱 + LaTeX（离线）
python -m autoresearch.tests.test_figures_metrics  # 指标统计 + 制图（离线）
python -m autoresearch.tests.test_prompts          # 提示词引擎 + 模板完整性
python -m autoresearch.tests.test_adapters         # 实验后端适配器协议（离线）
python -m autoresearch.tests.test_pipeline         # ★ 端到端：mock+offline 跑完整条管线
```

**七个套件全部不需要网络、不需要 API Key、不需要 LaTeX。**
这是设计不变量而不是巧合：它让 CI 能在三平台矩阵上直接跑，也让任何人都能在离线机器上
验证自己的改动没有破坏降级路径。

关于第三方依赖，准确的说法是：**没有任何套件因为缺少可选依赖而失败。**
`matplotlib` / `numpy` / `pandas` / `PyMuPDF` 都是可选依赖（见 `pyproject.toml` 的
`[figures]` / `[pdf]` 分组），缺席时相关检查会**明确跳过并打印安装提示**，其余照常验证。
注意区分：

| 环境 | `test_figures_metrics` | `test_retrieve` |
|---|---|---|
| 裸 Python（CI 的 `test` job 刻意不装任何第三方依赖） | 指标解析/统计/对照照跑；绘图与 DataFrame 相关检查跳过 | 检索部分照跑；需要 PyMuPDF 构造 PDF 的检查跳过 |
| 装了 `.[figures,pdf]`（CI 的 `extras` job） | 全部检查，含真实出图 | 全部检查，含真实 PDF 解析 |

**为什么要两个 job**：只跑裸环境会让绘图代码完全没有 CI 信号（跳过不等于测过）；
只跑装好依赖的环境又会让「零强制依赖」这条不变量失去可执行证据。两个都跑才对。
`parse_metrics` 是刻意的例外——CSV 解析属于核心路径，它**不依赖 pandas**，
自带标准库回退（两条路径的输出在测试中被断言必须一致）。

`test_pipeline` 是唯一的集成级测试，它断言的关键不变量包括：产物 sha256 与磁盘一致、
`\input` 目标全部存在、无悬空 `\cite`、跨种子统计口径正确、评审循环必然终止、
适配器确实接管了命令构造与指标解析、以及**故意植入的假数字必须被抓出来**。

## 已知限制

1. **本环境无法产出 PDF**。tectonic 二进制已下载到 `autoresearch/vendor/tectonic/`
   （48 MB，`0.15.0`，已通过 `--version` 验证），但它第一次编译时需要把 TeX bundle
   落盘并解包到自己的临时目录，而本会话的 Windows 文件沙箱 / ACL **拒绝进程在新建目录里
   写入**，报 `os error 5`。
   **注意归因**：这不是"缓存位置不对"。`tools/latex.py` 已经把 `TECTONIC_CACHE_DIR`
   指到工作区内的 `autoresearch/vendor/tectonic_cache`，失败依旧发生；同一个拒绝也命中
   tectonic 自己的 `.tectonic_dl_*` 临时目录（它们至今留在 `vendor/tectonic/` 下且无法删除）。
   所以这条路在当前会话里走不通，与配置无关。
   s7 会做**真实编译预检**（实际编译一个 3 行文档）并把根因写进 `report/COMPILE_BLOCKED.md`，
   `cli doctor` 也会显示这条诊断而不是笼统的"找不到引擎"。
   三条可行的替代路径：① 在有完整文件权限的终端里直接跑
   `autoresearch/vendor/tectonic/tectonic.exe -X compile paper/main.tex --outdir paper`；
   ② 换用系统 TeX（`pdflatex` / `xelatex`，不依赖 tectonic 的自建缓存）；
   ③ 把 `paper/` 上传 Overleaf。
   **因此 s7 的「编译失败 → 抽错 → LLM 修 → 重编译」闭环尚未在真实编译上验证过**——
   这是本项目目前最大的未验证路径。
2. **默认实验是合成任务**。内置 `synthetic-toy` 适配器用受控合成数据，只用于验证管线
   连通与降级路径；它的绝对指标与真实基准不可比，相对差异也不应外推。要产出可投稿的结果，
   必须通过适配器接入真实数据集与训练脚本。
3. **检索质量依赖网络与配额**。Semantic Scholar 无 Key 时限流严重（实测 429），
   arXiv 也有 3 秒节流；全部源失败时管线继续运行，但文献综述会明确标注"证据不足"。
4. **沙箱是进程级隔离，不是虚拟化**。POSIX 上启用 `setrlimit`（内存/文件大小）；
   Windows 没有等价的 rlimit，安全性靠 `scan_code` 静态扫描 + 工作目录约束 + 超时杀进程树。
   需要强隔离请用 `--sandbox docker`。
5. **mock 后端只用于验证连通性与降级路径**，它生成的内容不代表科研质量。
   接真实模型请去掉 `--llm-provider mock`。
6. **真实 LLM 的产出质量决定评审分数**。实测 DeepSeek 在合成数据实验上给出
   2.5/10 `reject`——这是**真实结论**（只有合成数据、种子偏少、缺少与文献强基线的对比），
   不是管线故障。要让分数上去，需要的是真实数据、更多种子与真基线。
