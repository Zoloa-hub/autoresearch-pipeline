# Auto-Research Pipeline

[![CI](https://github.com/Zoloa-hub/autoresearch-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Zoloa-hub/autoresearch-pipeline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

> **自动科研的真正瓶颈不是「生成」，是「没人能验证生成的东西」。**
>
> 这个项目做的是那一层：让 AI 产出的**每一个数字都能追溯到磁盘上的一个产物**；
> 做不到的时候，它**说做不到**，而不是给你一份看起来成功的空结果。

九个离散阶段串成一条管线：检索 → 构思+查新 → 规划 → **沙箱执行+自纠错** → 分析制图 →
分章撰写 → LaTeX 编译 → 自动评审+迭代 → 交付打包。每个阶段可单独重跑，
可断点续跑，每一次 LLM 调用与沙箱执行都留在 `events.jsonl` 里。

```text
① 文献检索 ──▶ ② 构思+查新 ──▶ ③ 实验规划 ──▶ ④ 沙箱执行+自纠错
                                                      │
       ⑨ 交付打包 ◀── ⑧ 评审+迭代 ◀── ⑦ LaTeX编译 ◀── ⑥ 分章撰写 ◀── ⑤ 分析制图
                          │
                          └── 分数不达标且仍有预算 → 回到 ⑥
```

---

## 它凭什么可信

多数自动科研项目承诺「端到端生成论文」。这里把力气花在**让产出可被拆穿**上：

| 机制 | 它防的是什么 |
|---|---|
| **数字溯源** `cli verify <run_id>` | 论文里的每个数值 token 与实验证据表逐一对齐，输出未溯源清单。实测抓到过「验证集在 20 epoch 后进入平台期」而实际只跑了 3 个 epoch。 |
| **保真度护栏** | 自纠错补丁若删掉指标输出、加 `except: pass`、或写死高分返回值，会被标记并进入未解决问题清单。**指标可以难看，但不能是假的。** |
| **多种子统计口径** | 先在每个种子上取标量、再跨种子算 mean±std。实测出现过 seed 0 打平（−1.4%）而 seed 1 是 +15.2%——单种子会给出方向相反的结论。 |
| **指标方向由适配器声明** | 管线**不从指标名猜方向**。通用词表在 19 个材料学指标上实测 13/19 不可信（5 个判反），而方向判错不报错、只让「改善」的定义反过来。 |
| **诚实降级** | 无 LaTeX 引擎时不产出空 PDF，而是写 `report/COMPILE_BLOCKED.md` 说明根因与三条自救路径。`cli doctor` 如实报告能力缺口。 |
| **审计底座** | 每次 LLM 调用、沙箱执行、路由、补丁都有结构化记录；失败可归因，不靠猜。 |

**离线可验证**：七个测试套件、1848 项检查，全部不需要网络、API Key 或第三方依赖。
CI 在 ubuntu / windows / macos 三平台 × Python 3.10/3.12/3.13 上全绿，
且刻意**不安装任何第三方依赖**——那是「零强制依赖」这条承诺的可执行证据，
不是口头声明。另有 `extras` job 装上可选依赖再跑一遍，因为**跳过不等于测过**。

---

## 实测驱动过的领域

适配器层的存在意义就是这一栏。以下全部是**真实执行**，不是设计声明：

| 后端 | 形态 | 实测结果 |
|---|---|---|
| `synthetic-toy`（内置） | 受控合成分类，零依赖 | 4 臂 × 3 种子；声明它只用于验证连通性 |
| `script-wrapper` | 用户自带训练脚本 | 2 变体 × 3 种子，`accuracy 0.8667 → 0.9111`（+5.1%） |
| `lorenz-governance` | **真实科研脚本**（Lorenz-63 混沌系统治理），未改一行 | 5 个模型种子、45 条序列；`d\|lam−lr\|` 种子间从 −0.001 到 −0.381 |
| `materials-optics` | **真实材料脚本**（La₂Ti₂O₇ 薄膜光学常数），未改一行 | 5 指标 × 重复测量；通用词表判反其中 4 个 → 结论翻转 |
| `comsol-batch` | **66 个真实 COMSOL console 日志** | ingest 实测；量化出中文错误**最高 88.8% 已销毁**（U+FFFD，转码不可恢复） |

最后两行是这个项目目前最值得看的部分：**它们都是非 ML 领域，而且都暴露了具体缺陷。**

- 材料光学接进来之后才发现：**指标方向不能从名字猜**。同一个消光系数从 0.5 降到 0.2
  （材料变透明，真实改善），通用词表读成「没改善」，`supports_claim` 从 True 翻成 False。
- COMSOL 接进来之后才发现：它的中文报错**在写出时就已成不可恢复的替换字符**——
  不是「编码不匹配、转码即可」，是信息已经没了。适配器现在如实给出 `log_corruption_ratio`，
  并默认加 `-locale en_US` 让后续运行的错误可读。

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

## 它明确不做什么

这些是**有意的拒绝**，不是尚未实现的待办：

- **不承诺生成可投稿论文。** 唯一一次真实全管线跑（DeepSeek，24 次 LLM 调用、153k tokens、
  62 篇真实文献、7,400 词论文）得到的评审是 **2.5/10 reject** —— 理由是实验只有合成数据、
  种子偏少、缺强基线。**那是真实的科研质量结论，不是管线故障。** 它现在能交付的是
  「结构完整、数字可追溯、实验强度不足的初稿」。
- **不保证没有幻觉数字。** `cli verify` 是**范围检查**：它抓得出「20 epoch」而实际只跑 3 个，
  但把 accuracy 写成 f1 这种语义错误它抓不出来。**能证明的是"没编造超出证据的数字"，
  不是"每个数字都对"。**
- **不做 GPU 调度、数据管理、依赖安装、长训练断点续训、超参搜索。** 这些属于你的实验栈，
  适配器只负责「怎么跑」与「怎么看结果」。
- **不假装通用。** 机制是领域无关的，词汇不是。见下。

---

## 边界：实测过的 vs 没验证的

这个项目的核心承诺是文档诚实，所以这一节写得和别的项目反过来——
**不是「我们测试很充分」，而是「这些我们确实没验证过」**：

**已验证**

- CI 三平台 × 三 Python 全绿；核心套件零第三方依赖（可执行证据）
- 七个套件 1848 项检查，本地 + WSL Ubuntu + 隔离裸环境三处一致
- 上述 5 个后端均真实执行过
- `extras` job 里绘图与 PDF 解析有真实覆盖（不是跳过）

**未验证（会挡住你的具体场景）**

- **PDF 编译闭环从未收敛过。** tectonic 二进制就位，但沙箱/ACL 拒绝它在新建目录里写入
  （`os error 5`），一次成功编译都没跑通。需要在无沙箱终端跑一次
  `autoresearch/vendor/tectonic/tectonic.exe -X compile paper/main.tex --outdir paper`
  把 bundle 缓存建起来。
- **`comsol-batch` 的 run 模式从未执行过。** COMSOL 是商业软件，开发机与 CI 都没有；
  已验证的只有命令构造与 ingest 解析。
- **`--sandbox docker` 只验证到命令构造。** CI 的 Windows runner 有 docker CLI，
  但 `docker run` 因 Windows 容器限制失败；真实容器执行未被覆盖。
- **材料方向只覆盖光学常数。** 力学/热学/电学、XRD、显微、DSC 未验证；
  材料最自然的参数扫描（温度/成分/退火时间）还没接成变体轴。
- **通用性有硬边界。** 有标量指标的实证领域（经济/心理/流行病）可用，但统计口径
  （多重比较校正 vs 跨种子 mean±std）与评审标准需要替换；**质性研究、人文、法学不适用**——
  没有数字、没有可执行实验，章节结构与评审标准都是错的。
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
| [docs/adapters.md](docs/adapters.md) | 写一个后端适配器（接自己的数据集 / 训练脚本 / 仿真器 / 日志） |
| [autoresearch/CONTRACTS.md](autoresearch/CONTRACTS.md) | 模块接口契约（改任何公开签名前必读） |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南与不可破坏的不变量 |

## 测试

```bash
python -m autoresearch.tests.test_config_llm       # 配置 + LLM 客户端 + mock 意图路由
python -m autoresearch.tests.test_retrieve         # 检索 + PDF 解析
python -m autoresearch.tests.test_sandbox_latex    # 沙箱 + LaTeX
python -m autoresearch.tests.test_figures_metrics  # 指标统计 + 制图
python -m autoresearch.tests.test_prompts          # 提示词引擎 + 模板完整性
python -m autoresearch.tests.test_adapters         # 实验后端适配器协议
python -m autoresearch.tests.test_pipeline         # 端到端：mock+offline 跑完整条管线
```

每个套件既是可独立运行的脚本（打印 `PASSED n checks`），也可被 pytest 收集。
部分检查在缺可选依赖时**明确跳过并打印原因**——跳过不等于通过，`extras` job 会补上覆盖。

## 许可证

[Apache-2.0](LICENSE)
