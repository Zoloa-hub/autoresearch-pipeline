# Auto-Research Pipeline

[![CI](https://github.com/Zoloa-hub/autoresearch-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Zoloa-hub/autoresearch-pipeline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

把科研全生命周期拆成九个离散、可单独重跑的阶段，用 LLM + 检索增强 + 沙箱执行 + LaTeX
编译串成一条**可审计、可断点续跑、会诚实报告自己失败位置**的自动科研流水线。

> **不是**「一键生成一篇能中的论文」的工具。它的目标是把科研过程变得可复现、可核查，
> 并且在做不到的时候如实说出来。

CI 在 ubuntu / windows / macos 三平台 × Python 3.10/3.12/3.13 上全绿，且**核心套件
不安装任何第三方依赖**——那是"零强制依赖"这条承诺的可执行证据，不是口头声明。
另有 `extras` job 装上可选依赖再跑一遍，保证绘图与 PDF 解析真的有覆盖
（跳过不等于测过）。

```text
① 文献检索 ──▶ ② 构思+查新 ──▶ ③ 实验规划 ──▶ ④ 沙箱执行+自纠错
                                                      │
       ⑨ 交付打包 ◀── ⑧ 评审+迭代 ◀── ⑦ LaTeX编译 ◀── ⑥ 分章撰写 ◀── ⑤ 分析制图
                          │
                          └── 分数不达标且仍有预算 → 回到 ⑥
```

## 30 秒上手

不需要 API Key、不需要网络、不需要安装任何第三方依赖：

```bash
python -m autoresearch.cli doctor    # 环境自检
python -m autoresearch.cli demo      # 端到端冒烟，约 9 秒跑完九个阶段
```

接真实模型：

```bash
export DEEPSEEK_API_KEY=sk-...       # provider 由 Key 自动推断
python -m autoresearch.cli run --direction "你的研究大方向" --venue NeurIPS
```

接你自己的训练脚本（不改代码）：

```bash
python -m autoresearch.cli run --direction "..." \
    --experiment-adapter script-wrapper \
    --adapter-arg script=my_train.py
```

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
| [docs/adapters.md](docs/adapters.md) | 实验后端适配器开发指南（接自己的数据集 / 训练脚本 / 仿真器） |
| [autoresearch/CONTRACTS.md](autoresearch/CONTRACTS.md) | 模块接口契约（改任何公开签名前必读） |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南与不可破坏的不变量 |

## 它凭什么可信

多数自动科研项目承诺「端到端生成论文」。这个项目把力气花在**让人能核查**上：

- **实验保真度护栏** —— 自纠错补丁如果删掉了指标输出、加了 `except: pass`、
  或写死高分返回值，会被标记并进入交付报告的未解决问题清单。指标可以难看，但不能是假的。
- **数字溯源校验** —— `cli verify <run_id>` 抽取论文里的每个数值 token 与实验证据比对，
  输出「未溯源数字」清单。实测抓到过「验证集在 20 epoch 后进入平台期」而实际只跑了 3 个 epoch。
- **多种子统计口径** —— 先在每个种子上取标量、再跨种子算 mean±std，而不是把所有 epoch
  混在一起算方差（后者会把「收敛轨迹的散布」当成「结果不确定性」，误差棒系统性偏大）。
- **诚实降级** —— 只有 3 个阶段是致命的；分析、编译失败会跳过并继续，产出一份写清
  「哪里没做到」的报告，而不是崩在半路。
- **离线可验证** —— 七个测试套件全部不需要网络、API Key 或第三方依赖，CI 三平台矩阵直接跑。

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

## 许可证

[Apache-2.0](LICENSE)
