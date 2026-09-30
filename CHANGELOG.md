# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 已验证 / 已修复（本机 Windows + WSL Ubuntu 实测）

- **POSIX 分支首次真实执行，并因此发现一个会让 CI 必挂的缺陷。**
  开发机是 Windows，`os.name != "nt"` 的代码路径此前**一次都没跑过**。在 WSL
  Ubuntu（Python 3.14.4）上跑第一遍时暴露：

  `SubprocessSandbox.run_command()` 会把 ``argv[0] == "python"`` 规范化成
  ``sys.executable``，在 Linux 上那是 ``/usr/bin/python3``；而静态扫描的
  ``system_path_write`` 规则匹配 ``/usr/``，于是**每一条**以解释器开头的命令都被
  拒绝——也就是 s4 的全部实验执行。Windows 上 ``sys.executable`` 不含 ``/usr/``，
  所以本地与当时全部测试都看不见。
  修法：扫描只针对参数载荷，解释器路径是基础设施而非用户载荷；用户载荷里的危险路径
  仍然被拦（有回归测试同时覆盖这两种情况，且**不依赖当前平台**）。

  同时确认了两条此前只是"写了但没跑过"的能力：
  - `RLIMIT_AS` 生效——64 MB 上限下申请 400 MB 被阻止；
  - 超时后进程组整组回收生效。

- **测试套件的可选依赖处理被修正。** 此前 `test_figures_metrics` 在模块级
  `import matplotlib / pandas`（缺了整套 ImportError），`test_retrieve` 在缺
  PyMuPDF 时直接判失败。这与 README 的声明不符，也会让 CI 第一次运行就变红——
  而红的原因是"没装可选依赖"，这种红叉会训练人忽略 CI。
  现在：缺席时**明确跳过并打印安装提示**，其余检查照常执行。
  CI 相应拆成两个 job：`test`（裸环境，证明零强制依赖）与 `extras`
  （装 `[figures,pdf,http]`，真正覆盖绘图与 PDF 解析）。

- **`parse_metrics` 不再依赖 pandas。** CSV 解析属于核心路径，而 pandas 是可选依赖；
  现在自带标准库回退，并且测试断言**两条路径的输出完全一致**（含引号内逗号、
  分号、制表符、无表头、百分号等用例）。绘图与 DataFrame 形态的输出仍需要 pandas，
  缺席时抛 `OptionalDependencyError`（可区分"环境缺依赖"与"代码坏了"）。

- **消融臂此前是死路径。** `_plan_variants` 默认把臂数截成 2，s3 规划的消融矩阵
  一条都跑不到；同时消融臂的命名（`momentum=0.9 (on)`）含空格与括号，作为 argv
  token 不安全。现在：臂数上限取 `min(配置, 适配器自报)`，消融臂用中性标识
  `abl-1`，轴与取值通过 `RunSpec.params` 传递。

- **`--max-variants` 曾是死参数。** argparse 里注册了，但从未写进 config overrides，
  设了完全不生效；`AUTORESEARCH_MAX_VARIANTS` 也不在 `ENV_KEYS` 里。两者都已接通。

- **模板占位符漏替换。** LaTeX 模板声明 5 个占位符，而 s6 只替换 3 个，于是
  `\date{__DATE__}` 与 `Keywords: __KEYWORDS__` 会原样进入 PDF——这类残留**不会让
  编译失败**，因此不会被 s7 的编译修复闭环发现。现在全部替换，并加了装配后自检
  与事件 `template_placeholders_unfilled`。

- **评审回环会静默丢弃关键词。** `_apply_iteration` 不传 `keywords`，每次回环都用
  硬编码兜底串覆盖 LLM 生成的关键词。现在落盘到 `paper_meta.json` 并读回。

- **指标方向判定曾有 5+ 份副本且已漂移**（一处含 `fid/fdr`，另一处含 `flops/params`），
  同一个指标名在不同阶段会被判成不同方向——不报错，只让"改善"的定义在论文不同
  章节里悄悄改变。现在 `tools/metrics.py` 是唯一规范表。

- **`s4_codegen` 提示词与适配器协议正面冲突。** s4 已经把适配器约定注入提示词，
  但提示词文件里仍写着"Hard requirements — every one is checked"的固定表头、
  四个 CLI 参数与 `baseline|method` 白名单，把注入的约定压了过去。现在改为引用
  `{{ workspace_conventions }}` / `{{ data_info }}` / `{{ variant }}`。

- **显式配置的排版引擎曾被静默忽略。** _resolve_engine() 在显式引擎不可用时无条件
  
eturn self.detect()，于是 engine="xelatex" 而机器上只有 tectonic 时会**静默改用
  tectonic**：编译成功、产出 PDF、零提示，用户指定的引擎完全被忽略。排版引擎之间
  不是等价替换（字体、宏包、Unicode 处理都不同）。现在如实报告“指定的引擎
  不可用”并拒绝替换；只有 engine="" 才表示允许自动探测。

- **	est_latex_compile_degrades 里有一段是环境依赖的。** 它用 detect() is None
  判断是否走“无引擎限降”分支，但 detect() 可能返回 None 而紧接着的
  compile() 内部预检又把引擎变可用（第一次跑就以 
o engine -> ok is False
  失败，而报错里却写着 engine=tectonic 且编译成功）。已新增**确定性**的
  无引擎检查（用不存在的引擎名强制走降级路径），不再依赖“这台机器恰好没引擎”。

### ⚠️ 仍未验证

- **CI 工作流从未在 GitHub 上执行过。** 已完成的只是本地校验：YAML 可解析、
  4 个 job 结构正确、7 个套件引用真实存在、matrix 组合数正确、package-data glob
  命中真实文件、每个 `run` 步骤的命令在本地 shell 能跑通。
  **workflow 的运行时语义（actions 版本兼容性、矩阵并行、这一步的 heredoc 在
  GitHub runner 的默认 shell 下是否可执行）未经真实执行。**

- **s7 的「编译失败 → 抽错 → LLM 修复 → 重编译」闭环从未收敛过。**
  tectonic 二进制能下载、能通过 `--version`，但本机沙箱/ACL 拒绝进程在新建目录里
  写入（`os error 5`），一次成功编译都没有过。这是当前最大的未验证面。

- **`--sandbox docker` 路径未验证**（本机未安装 docker）。只有 `subprocess` 后端
  经过实测。

- **POSIX 上的三平台矩阵只覆盖到 WSL Ubuntu**（Python 3.14.4）。macOS 未验证；
  CI 里的 ubuntu job 会是首次在真实 runner 上执行。

### 待办

- 让 PDF 编译在真实编译上跑通一次，并把结果回灌到 s7 的错误修复规则。
- 首次 push 后核对 CI：`test`、`extras`、`packaging`、`lint` 四个 job。

## [0.1.0]

### 新增

- **九阶段管线**：文献检索 → 构思+查新 → 实验规划 → 沙箱执行+自纠错 → 分析制图 →
  分章撰写 → LaTeX 编译 → 自动评审+迭代 → 交付打包。
- **可插拔实验后端适配器**（`autoresearch/adapters/`）：
  - `BaseExperimentAdapter` 协议与 `RunSpec`（开放超参 `params` + `extra_args` 逃生舱）
  - `SyntheticToyAdapter`：内置受控合成任务，零第三方依赖、CPU 秒级、确定性种子
  - `ScriptWrapperAdapter`：接入用户自带训练脚本（`owns_code=True`，跳过 LLM 代码生成）
  - 三种注册方式：`.py` 文件（模块级 `ADAPTER`）/ `module:Class` / 包的 entry point
- **自研状态机** `graph/engine.py`：条件路由、重试、降级、环路保护，零依赖；
  语义与 LangGraph 对齐，并提供可选编译层 `graph/langgraph_adapter.py`。
- **断点续跑**：每阶段落检查点，`cli resume` 不重跑已完成阶段。
- **数字溯源校验** `cli verify`：抽取论文里的每个数值 token，与实验证据表比对，
  输出未溯源数字清单。
- **实验保真度护栏**：自纠错补丁若删掉指标输出、加 `except: pass`、或写死高分返回值，
  会被标记 `validity_flag` 并进入交付报告的未解决问题清单。
- **多种子统计口径**：先在每个种子上取标量，再跨种子算 mean±std
  （而不是把所有 epoch 混在一起算方差——那会把"收敛轨迹的散布"当成"结果不确定性"）。
- **诚实降级**：无 LaTeX 引擎时不产出空 PDF，而是写 `report/COMPILE_BLOCKED.md`
  说明根因与三条自救路径。
- **离线模式**：`cli demo` 无需网络、API Key、LaTeX 或任何第三方依赖跑通全流程。

### 设计取舍（为什么这样做）

- **代码生成不放进适配器。** 适配器只提供模板（`seed_code()`），LLM 负责按需生成与
  修补。若适配器自己生成代码、自己修 bug，调试闭环与保真度检查将对该适配器的产物
  完全失效——而"指标被静默破坏"正是自动科研里最难发现的一类错误。
  用户自带脚本（`owns_code=True`）是受控特例：跳过生成，但仍走调试与保真度检查。
- **失败降级而非中断。** 只有 3 个阶段设 `fatal=True`（检索、构思、实验）；
  另有 4 个阶段参与退出码层面的 claim-fatal 判定。分析、编译失败会跳过并继续，
  产出带缺口但诚实的报告。
- **臂数上限取 `min(配置, 适配器自报)`。** 配置是用户本次预算，适配器上限是后端能力
  约束；取小值意味着用户无法把一个只愿意跑 2 臂的后端推到 6 臂——
  "跑不完的消融"比"没有消融"更糟。
- **不做 PDF 编译失败的静默降级。** 宁可交付一份说明根因与替代路径的
  `COMPILE_BLOCKED.md`，也不产出一个看起来成功、实则空的 PDF。

### 测试

七个套件**全部不需要网络、API Key 或 LaTeX**。可选依赖缺席时会明确跳过相关检查
并打印安装提示，而不是失败——因此裸 Python 环境下套件仍全绿。CI 另有一个
`extras` job 装上可选依赖再跑一遍，以保证绘图与 PDF 解析真的有覆盖。

```bash
python -m autoresearch.tests.test_config_llm       # 配置 + LLM 客户端 + mock 意图路由
python -m autoresearch.tests.test_retrieve         # 检索 + PDF 解析
python -m autoresearch.tests.test_sandbox_latex    # 沙箱 + LaTeX
python -m autoresearch.tests.test_figures_metrics  # 指标统计 + 制图
python -m autoresearch.tests.test_prompts          # 提示词引擎 + 模板完整性
python -m autoresearch.tests.test_adapters         # 实验后端适配器协议
python -m autoresearch.tests.test_pipeline         # 端到端：mock+offline 跑完整条管线
```

> **不要引用本文档或 README 里的检查项数量。** 每个套件运行时打印
> `PASSED n checks`，具体数字随代码演进变化；写在 issue 或博客里的历史数字会过期
> 并误导。而且装了可选依赖与没装时数字**本就不同**（跳过的检查不计入）。

[Unreleased]: https://github.com/OWNER/autoresearch-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/autoresearch-pipeline/releases/tag/v0.1.0
