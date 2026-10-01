# Auto-Research Pipeline

[![CI](https://github.com/Zoloa-hub/autoresearch-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Zoloa-hub/autoresearch-pipeline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

> ### 大多数自动科研工具优化的是「产出得更多」。这个项目优化的是「**骗不了人**」。
>
> 它把科研全流程拆成九个可单独重跑的阶段，然后给每个可能作假的环节装上一道检查：
> 论文里每个数字都要**追回磁盘上的产物**，实验脚本被改动要**留下记录**，
> 跑不通就**写清楚哪里没跑到**——而不是交给你一份看起来成功的空结果。

这不是一句定位口号，是这个仓库的运行记录。下面这些**全部真实发生过**，每一条都能在
[CHANGELOG.md](CHANGELOG.md) 与提交历史里查到。

---

## 它抓到过什么

一个检查机制的价值不在于它写得多好，而在于**它真的抓到过东西**。这个项目的记录是：

<table>
<tr><th align="left">抓到了什么</th><th align="left">为什么这不 trivial</th></tr>
<tr><td>

**材料学：18 个常见指标里 10 个方向不可信**

把一份真实的材料光学脚本接进来后发现，通用指标词表把 `k`（消光系数）判成
「越大越好」——而它越小越好。同一份数据、同一组数字，`supports_claim`
**从 True 翻成 False**。

</td><td>

方向判错**不会报错**。它只会让「改善」的定义反过来，
然后把变差写成变好。这类错误再过一百次评审也发现不了，
因为两边看到的数字是一样的。

</td></tr>
<tr><td>

**COMSOL 的中文报错最高 88.8% 已经销毁**

66 个真实批处理日志里，20 个的损坏率超过 50%。原因不是编码不匹配——
是 COMSOL **写出时**就把字符变成了 `U+FFFD`，**转码救不回来**。

</td><td>

第一反应是「转个码就行」。实测证明那是错的：
信息已经没了。适配器因此改成**如实报告损坏比例**，
而不是把乱码当原文传下去——更不是假装能修好。

</td></tr>
<tr><td>

**参数覆盖静默失效，会让整个扫描变成"同一实验跑 N 次"**

被驱动模块把参数绑成了函数默认值（定义时即固定），
`setattr` 模块常量**完全无效**。后果是所有格点算出同一组数值 →
效应全为 0 → 结论变成「这些参数都不重要」。

</td><td>

这是最危险的一类失败：**没有任何报错**，
交出的是一条平坦曲线和一份看起来正常的结论。
现在 runner 会验证覆盖真的生效，没变就 `OVERRIDE_NO_EFFECT` 退出。

</td></tr>
<tr><td>

**CI 在真实 runner 上抓出 6 个缺陷——四个本地环境一个都没测出来**

包括两个会让用户**直接崩**的：容器后端一用就 `AttributeError`（`_merged_env`
定义在错误的类上），以及 Windows 控制台下中文输出让进程**静默死在半途**。
还有一个是产品缺陷：无 pandas 时**论文主表被静默丢弃**。

</td><td>

本地测过 Windows、WSL Ubuntu、隔离裸环境、Python 3.14——**全绿**。
CI 一次跑出 6 个。这是「本地全绿 ≠ 真的能跑」最直接的证据，
也是这个项目坚持三平台矩阵的原因。

</td></tr>
<tr><td>

**对真实科研脚本的一次完整运行：评审 2.5/10, reject**

24 次 LLM 调用、153k tokens、62 篇真实文献、7,400 词论文、2 轮评审。
评审理由是"实验只有合成数据、种子偏少、缺强基线"。

</td><td>

这是**真实的科研质量结论，不是管线故障**。
它被原样写进 README 和 CHANGELOG，而不是挑一个好看的数字放上来。
一个敢公布自己 2.5/10 的项目，比一个说"我们很可靠"的项目可信。

</td></tr>
</table>

最后一条也解释了**为什么这些检查必须存在**：自动化科研最缺的不是产能，
是"知道自己在哪一步说了假话"的能力。

---

## 为什么这样设计

有些选择看起来绕，但每一条都是被真实故障逼出来的：

**为什么数字溯源不做成"检查数字对不对"？**
因为它做不到，而假装能做到比不做更糟。`cli verify` 做的是**范围检查**：
把论文里的数值 token 与实验证据表逐个对齐。它抓得出"验证集在 20 epoch 后进入
平台期"而实际只跑了 3 个 epoch——但把 `accuracy` 写成 `f1` 这种语义错误它抓不出来。
**能证明的是"没编造超出证据的数字"，不是"每个数字都对"。** 这条差别被写进
了文档，因为它决定了你能拿它做什么承诺。

**为什么指标方向必须由适配器声明、而不是从名字猜？**
因为词表必然有领域偏向。管线里的通用表是 ML 中心的（`loss`/`accuracy`/`f1`/`bleu`），
换到材料学实测 10/18 不可信（5 个判反 + 5 个无固有方向却被静默选了一个）。
方向判错不报错——这是所有失败模式里最难发现的一种。

**为什么只有 3 个阶段是"致命"的？**
因为崩在半路比降级交付更没用。分析与编译失败会跳过并继续，产出一份写清
「哪里没做到」的报告；无 LaTeX 引擎时写出 `report/COMPILE_BLOCKED.md`
说明根因与三条自救路径，而不是生成一个空 PDF。

**为什么测试套件刻意不装任何第三方依赖？**
因为"零依赖"是一条承诺，而承诺需要可执行的证据，不是声明。CI 里核心套件
在 ubuntu / windows / macos × Python 3.10/3.12/3.13 上跑，**什么都不装**；
另有一个 `extras` job 装上可选依赖再跑一遍——因为**跳过不等于测过**。

**为什么要做「让 CI 失败时自动说出失败在哪」这件事？**
因为公开仓库的 job 日志正文需要 admin 权限才能通过 API 读取
（`GET /actions/runs/{id}/logs` → `403 Must have admin rights`），
而 GitHub annotation 是匿名可读的。在加这个之前，CI 红了之后能公开看到的
只有一句 `Process completed with exit code 1.`——**连作者都定位不到失败在哪**。
一个以"可验证"为卖点的项目，CI 红了却说不清为什么，是自相矛盾的。

---

## 九个阶段

```text
① 文献检索 ──▶ ② 构思+查新 ──▶ ③ 实验规划 ──▶ ④ 沙箱执行+自纠错
                                                      │
       ⑨ 交付打包 ◀── ⑧ 评审+迭代 ◀── ⑦ LaTeX编译 ◀── ⑥ 分章撰写 ◀── ⑤ 分析制图
                          │
                          └── 分数不达标且仍有预算 → 回到 ⑥
```

每个阶段可单独重跑、可断点续跑。每一次 LLM 调用、沙箱执行、路由与代码补丁都写进
`events.jsonl`：失败可归因，不靠猜。

---

## 30 秒上手

不需要 API Key、不需要网络、不需要安装任何第三方依赖：

```bash
python -m autoresearch.cli doctor    # 环境自检：如实报告能力与缺口
python -m autoresearch.cli demo      # 端到端冒烟，约 9 秒跑完九个阶段
```

接真实模型：

```bash
export DEEPSEEK_API_KEY=sk-...       # provider 由 Key 自动推断
python -m autoresearch.cli run --direction "你的研究大方向" --venue NeurIPS
```

接你自己的实验（**不改你的代码**）：

```bash
# 训练脚本
python -m autoresearch.cli run --direction "..." \
    --experiment-adapter script-wrapper --adapter-arg script=my_train.py

# 材料光学模块（需提供 nk_table(lam_um)）
python -m autoresearch.cli run --direction "..." \
    --experiment-adapter materials-optics --adapter-arg module=my_optics.py

# 已有 COMSOL 批处理日志（不需要装 COMSOL）
python -m autoresearch.cli run --direction "..." \
    --experiment-adapter comsol-batch --adapter-arg log_dir=./comsol_logs
```

---

## 实测驱动过的领域

适配器的意义就在这一栏。以下**全部真实执行**，不是设计声明：

| 后端 | 形态 | 实测结果 |
|---|---|---|
| `synthetic-toy`（内置） | 受控合成分类，零依赖 | 4 臂 × 3 种子；明确声明它只用于验证连通性 |
| `script-wrapper` | 用户自带训练脚本 | 2 变体 × 3 种子，`accuracy 0.8667 → 0.9111`（+5.1%） |
| `lorenz-governance` | **真实科研脚本**（Lorenz-63 混沌治理），未改一行 | 5 个模型种子、45 条序列；`d\|λ−lr\|` 种子间从 −0.001 到 −0.381 → **单种子会给出方向相反的结论** |
| `materials-optics` | **真实材料脚本**（La₂Ti₂O₇ 光学常数），未改一行 | 5 指标 × 重复测量；通用词表判反其中 4 个 → 结论翻转 |
| `materials-optics` + **多参数扫描** | 深紫外吸收边三参数研究 | OFAT 7 格点 × 3 重复，真实沙箱执行 → **没有单一主导参数**（见下） |
| `comsol-batch` | **66 个真实 COMSOL 日志** | ingest 实测；量化出中文错误最高 88.8% 已销毁 |

### 参数扫描给出的那个结论

材料/化学研究最自然的实验形态是参数扫描，而它和「方法 vs 基线」是两种设计：

| 设计 | 运行数 | 能力 |
|---|---|---|
| **OFAT**（一次一条轴） | `1 + Σ(kᵢ−1)` | 便宜，但**原理上测不出交互效应** |
| **网格**（全组合） | `Π kᵢ` | 能估主效应**与交互效应**，但组合爆炸（5×4×3 = 60） |

管线**不替你选**，但会在超出预算截断时标记 `orthogonal=False`——因为截断后的网格
不再是正交设计，主效应会与交互效应混淆，"某条轴重要"可能只是采样不均。

La₂Ti₂O₇ 深紫外吸收边的实测结果：

```
指标                    主导轴   相对强度          最优取值
n_deviation_1e3         n_vis    1.00 (次强 0.15)   2.3
transparent_window_nm   osc_g    1.00 (n_vis 0.40)  0.35
k_at_250nm              osc_f    1.00 (n_vis 0.69)  4.9
alpha_visible_mean      osc_g    1.00 (n_vis 0.40)  0.75
k_at_550nm              osc_g    1.00 (n_vis 0.40)  0.75
```

**没有单一「最重要参数」——哪条轴主导取决于你关心哪个性质。** 折射率偏差由 `n_vis`
主导（它就是这个量本身），透明窗口由展宽 `osc_g` 主导（展宽抹平吸收边），
深紫外消光由振子强度 `osc_f` 主导。这是"方法 vs 基线"式对照**给不出的**信息。

---

## 它凭什么可信

| 机制 | 它防的是什么 |
|---|---|
| **数字溯源** `cli verify <run_id>` | 论文里每个数值 token 与实验证据表逐一对齐，输出未溯源清单 |
| **保真度护栏** | 自纠错补丁若删掉指标输出、加 `except: pass`、或写死高分返回值，会被标记并进入未解决问题清单。**指标可以难看，但不能是假的。** |
| **多种子统计口径** | 先在每个种子上取标量、再跨种子算 mean±std。实测出现过 seed 0 打平而 seed 1 是 +15.2% |
| **指标方向由适配器声明** | 管线**不从指标名猜方向**；猜错不报错，只让「改善」的定义反过来 |
| **覆盖生效验证** | 参数覆盖若没真正改变结果，报 `OVERRIDE_NO_EFFECT` 退出，而不是交出一条平坦曲线 |
| **诚实降级** | 无 LaTeX 引擎时不产出空 PDF，而是写清根因与自救路径 |
| **审计底座** | 每次 LLM 调用、沙箱执行、路由、补丁都有结构化记录 |

**离线可验证**：七个测试套件、**1917 项检查**，全部不需要网络、API Key 或第三方依赖。

---

## 它明确不做什么

这些是**有意的拒绝**，不是尚未实现的待办：

- **不承诺生成可投稿论文。** 唯一一次真实全管线跑的评审结果是 **2.5/10 reject**。
  它现在能交付的是「结构完整、数字可追溯、实验强度不足的初稿」。
- **不保证没有幻觉数字。** `cli verify` 是**范围检查**：能证明"没编造超出证据的数字"，
  不能证明"每个数字都对"。
- **不做 GPU 调度、数据管理、依赖安装、长训练断点续训、超参搜索。**
  这些属于你的实验栈；适配器只负责「怎么跑」与「怎么看结果」。
- **不假装通用。** 机制领域无关，词汇不是——见下。

---

## 边界：实测过的 vs 没验证的

这个项目的核心承诺是文档诚实，所以这一节写得和别的项目反过来——
**不是「我们测试很充分」，而是「这些我们确实没验证过」**：

**已验证**

- CI 三平台 × 三 Python 全绿；核心套件零第三方依赖（可执行证据）
- 1917 项检查，本地 + WSL Ubuntu + 隔离裸环境三处一致
- 上述 6 个后端形态均真实执行过（含两个非 ML 领域）

**未验证（会挡住你的具体场景）**

- ~~PDF 编译闭环从未收敛过。~~ **已修复（2026-10）。** 真实根因不是沙箱：预检
  `_tectonic_cache_preflight()` 自己设了 `TECTONIC_CACHE_DIR` 并验证通过，
  而**真实编译没设**，于是 tectonic 回去用默认位置（`%LOCALAPPDATA%\Tectonic`，
  本环境不可写）——既读不到预检填充的缓存，也不会下载缺失宏包，最终在 TeX 层报
  `File `size11.clo' not found`，一个**指向完全错误方向**的错误。
  **预检验证了一个生产中从不使用的配置**，是典型的虚假信心检查。
  修复后：真实模板产出 **27 KB PDF**，且「注入错误 → 抽错 → 修复 → 重编译」闭环收敛。
- **`comsol-batch` 的 run 模式从未执行过。** COMSOL 是商业软件，开发机与 CI 都没有；
  已验证的只有命令构造与 ingest 解析。
- **`--sandbox docker` 只验证到命令构造。** 真实容器执行未被覆盖。
- **材料方向只覆盖光学常数。** 力学/热学/电学、XRD、显微、DSC 未验证。
- **参数扫描的效应没有不确定度。** `axis_effects` 给的是**点估计**
  （`span = 各水平均值极差`），没有置信区间、没有显著性检验、没有多重比较校正。
  **格点内噪声大时，"主导轴"可能就是噪声**——而代码不会告诉你。这是下一个要补的缺口。
- **能扫描的适配器目前只有 1 个。** 扫描机制是通用的（纯标准库、零领域词），
  但每个适配器必须自己声明参数轴与覆盖方式——因为**只有适配器知道被驱动模块
  怎么绑定参数**（实测：函数默认参数用 `setattr` 覆盖会静默失效）。
- **通用性有硬边界。** 有标量指标的实证领域（经济/心理/流行病）可用，但统计口径
  （多重比较校正 vs 跨种子 mean±std）与评审标准需要替换；**质性研究、人文、法学
  不适用**——没有数字、没有可执行实验，章节结构与评审标准都是错的。
- **真实训练规模未验证。** 所有实测都在秒级到分钟级的 CPU 任务上完成。

完整清单与每个缺陷的根因：[CHANGELOG.md](CHANGELOG.md)。

---

## 安装

```bash
pip install -e .                 # 核心零依赖
pip install -e ".[all]"          # 全部可选能力（LLM/绘图/PDF/HTTP/测试）
pip install -e ".[llm,figures]"  # 只装需要的
```

可选依赖组：`llm`（真实模型）、`figures`（制图与统计）、`pdf`（PDF 全文解析）、
`http`（更快的 HTTP）、`cjk`（中文图表字体）、`dev`、`all`。

## 文档

| 文档 | 内容 |
|---|---|
| [autoresearch/README.md](autoresearch/README.md) | **主文档**：当前能力、明确非目标、九阶段职责、设计取舍、配置、已知限制 |
| [docs/adapters.md](docs/adapters.md) | 写一个后端适配器（数据集 / 训练脚本 / 仿真器 / 日志） |
| [autoresearch/CONTRACTS.md](autoresearch/CONTRACTS.md) | 模块接口契约（改任何公开签名前必读） |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南与不可破坏的不变量 |

## 测试

```bash
python -m autoresearch.tests.test_config_llm       # 配置 + LLM 客户端 + mock 意图路由
python -m autoresearch.tests.test_retrieve         # 检索 + PDF 解析
python -m autoresearch.tests.test_sandbox_latex    # 沙箱 + LaTeX
python -m autoresearch.tests.test_figures_metrics  # 指标统计 + 制图
python -m autoresearch.tests.test_prompts          # 提示词引擎 + 模板完整性
python -m autoresearch.tests.test_adapters         # 适配器协议 + 参数扫描 + 方向声明
python -m autoresearch.tests.test_pipeline         # 端到端：mock+offline 跑完整条管线
```

每个套件既是可独立运行的脚本（打印 `PASSED n checks`），也可被 pytest 收集。
部分检查在缺可选依赖时**明确跳过并打印原因**——跳过不等于通过，`extras` job 会补上覆盖。

## 许可证

[Apache-2.0](LICENSE)
