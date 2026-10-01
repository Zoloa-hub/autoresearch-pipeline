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

### 真实实验案例：Lorenz-63 governance（已接入并实测）

`autoresearch/adapters/lorenz_governance.py` 把一份**真实科研脚本**
（Lorenz-63 governance 实验，内部跑 5 个模型种子）接进了管线，**未修改其实验代码**。
它同时是一个「照抄即可」的适配器范例，因为真实脚本与合成玩具的形态差异很大。

实测结果（沙箱内真实执行，`update_epochs=2`）：

```
validate_environment : True（5 个 checkpoint）
prepare()            : 复制 5 个文件（含 AST 自动解析出的传递依赖）
沙箱执行             : returncode=0
parse_results()      : 45 条序列 / 9 个指标 / 5 个种子
```

跨种子统计（管线口径：先按种子取标量，再跨种子算 mean±std）：

| 指标 | mean | min | max | 极差 |
|---|---|---|---|---|
| `delta_source_attractor_dist` | 3.90 | 1.08 | 5.30 | **4.22** |
| `delta_source_lyapunov_gap` | −0.185 | −0.381 | −0.001 | **0.38** |

**这个案例恰好证明了多种子统计为何是硬要求**：`d|lam−lr|` 在 seed 5555 是 −0.001
（几乎没变），在 seed 7 是 −0.381（明显改善）——相差两个数量级。报单个种子会得出
方向相反的结论。

接这个真实脚本暴露了 4 个只有真实脚本才会有、合成玩具永远测不到的接口问题：

1. **脚本内部固定种子、命令行不接受 `--seed`**，而 `RunSpec.seed` 是管线必需的。
   适配器必须忽略它；把 `--seed` 传进不认识它的 argparse 会直接退出码 2。
2. **本地依赖是传递闭包**。第一版手工列了两个文件，运行时炸在
   `validate_faithful_chaos` → `validate_lorenz_lyapunov` 这条传递依赖上。
   现在用 AST 走 import 图自动解析（`local_deps()`），且只收同目录下真实存在的模块。
3. **指标必须按脚本内部种子拆成多条序列**，否则「种子间差异」会被当成「种子内噪声」。
4. **缺失值必须丢弃而不是填 0**（`pre` 侧 `vpt` 常为 `None`）——填 0 会造出一个
   不存在的观测。

顺带修掉一个测试基建问题：测试函数抛异常时会让整个套件中断、**掩盖后面所有测试**
的结果。现在 `main()` 逐个包 `try/except`，一个崩了其余照跑。

### ✅ CI 已在真实 runner 上跑通（三平台 × 三 Python）

```
ubuntu-latest    py3.10 / 3.12 / 3.13   ✅
windows-latest   py3.10 / 3.12 / 3.13   ✅
macos-latest     py3.12                 ✅
extras（装可选依赖后跑绘图与 PDF 解析）   ✅ py3.10 / 3.13
packaging（真跑 build + 装 wheel）        ✅
lint（byte-compile + 全模块导入 + 密钥扫描）✅
```

**首次真实执行抓出了 6 个缺陷，全部在本地怎样都测不出来**——这是"CI 值不值"
最直接的答案。逐条记录：

1. **`DockerSandbox` 一用就崩。** `run()` 调用 `self._merged_env(env)`，
   而 `_merged_env` 只定义在 `SubprocessSandbox` 上 → `AttributeError`。
   容器后端此前**零覆盖**：开发机与 WSL 都没装 Docker，这条路径从未被执行过。
   而它正是 README 里推荐的"强隔离"方案。

2. **`--pids-limit` 无条件传给 `docker run`。** Windows 容器不支持该选项，
   于是 `docker run` 直接失败。改为按宿主判断，并记录该能力缺口。

3. **Windows 控制台下非 ASCII `print()` 崩溃。** 日志与报告大量使用中文，
   在 cp1252 下抛 `UnicodeEncodeError`，进程当场死、stdout 缓冲区随之丢失
   ——表现成"输出停在中途 + 退出码 1"，没有任何 traceback。

4. **测试夹具依赖 numpy。** `test_script_wrapper_adapter` 拿
   `templates/experiment/train.py` 当夹具，而该模板在模块级 `import numpy`；
   这与"套件在裸环境全绿"的声明直接矛盾，使该套件在**所有平台**必然失败。
   改用纯标准库夹具，反而多验证了"零依赖脚本也能被 script-wrapper 驱动"。

5. **`to_latex_table` 无 pandas 时静默丢失论文主表。** 它唯一的 pandas 用法是
   `isinstance` 判断，渲染本身是纯 Python；而 `s5_analysis` 用 try/except 包住它，
   失败只留一条 warning。结果是**实验数据完好、但交付的论文里没有结果表**。
   这是本批改动里最严重的一个——它是产品缺陷，不是测试问题。

6. **密钥扫描误报 + 排除规则失效。** 正则 `DEEPSEEK_API_KEY=[^[:space:]]` 只要
   后面有非空白字符就命中，于是文档里的占位符 `sk-...` 必然误报；且
   `:!*.md` 这类 pathspec 通配**不跨目录**，`autoresearch/README.md` 从未被排除。

另有若干**测试把环境事实当断言**的写法（"本机没有 docker"、"本机没有 LaTeX 引擎"），
只在"机器恰好缺什么"时通过。已全部改为用 monkeypatch 强制造出条件。

以及一个自伤：我用脚本做块替换时把 `extras:` 的 job 头吞掉了，导致绘图与 PDF
解析的真实覆盖被静默删除（YAML 仍合法、校验也仍"通过"）。已恢复，并给
`tools/dev/check_ci_fix.py` 加了硬性结构断言防止重犯。

### 🧪 非 ML 领域验证：材料光学（第一个真实案例）

用一份**真实的材料学脚本**做了领域通用性实测：`ltp_optics.py`
（La₂Ti₂O₇ 薄膜光学常数，Kramers-Kronig 一致的 n,k 色散；文献锚点
Bayart et al., Optical Materials 92 (2019)）。**脚本未做任何修改。**

新增 `autoresearch/adapters/materials_optics.py`，实测结果：

```
加载（module:Class + .py 路径）        OK
validate_environment（检查 nk_table）  OK
命令形态：含 --replicates、不含 --epochs OK   <- 非 ML 领域的关键差异
沙箱执行真实模块                       rc=0
parse_results                          5 指标 × 4 次重复测量
metric_axis()                          'replicate'（不是 epoch）
```

**这个实测量化了一个真实的架构缺口。**

#### 缺口：指标方向靠"从名字猜"，在非 ML 领域会判反

管线原本用一张通用 token 表猜方向，而那张表是 ML 中心的
（loss/accuracy/f1/bleu…）。在 19 个材料学常见指标上实测：

| 判定 | 数量 | 例子 |
|---|---|---|
| **方向判反** | **5** | `k`（消光系数，应越小越好）、`alpha`、`resistivity`、`corrosion_rate`、`sintering_temp` |
| **无固有方向却被静默选了一个** | **6** | `n`、`reflectance`、`band_gap`、`youngs_modulus`、`thermal_conductivity` |
| 判对 | 8 | `transmittance`、`hardness`、`d33`、`zt`… |

即 **13/19 不可信**。而方向判错**不会报错**——它只会让「改善」的定义反过来。

在材料适配器的 5 个指标上，通用词表**判反 4 个**，并导致对照结论翻转：

```
不声明方向: supports_claim=False  improved=['transparent_window_nm']
声明方向  : supports_claim=True   improved=['alpha_visible_mean','k_at_250nm',
                                            'k_at_550nm','n_deviation_1e3',
                                            'transparent_window_nm']
```

同一份数据、同一组数字，**结论从"不支持 claim"变成"支持 claim"**。

#### 修法：领域知识归适配器，管线不猜

1. `BaseExperimentAdapter.metric_directions() -> dict[str, bool] | None`
   —— 适配器**声明** `{指标名: 越大越好?}`。返回 `None`=交给通用启发式；
   返回 `{}`=已声明但无需方向判断（两者语义不同）。
2. `BaseExperimentAdapter.metric_axis() -> str | None`
   —— 声明序列的物理轴（ML 是 `epoch`，材料是 `replicate`）。
   默认 `None`：**管线不假设序列有物理含义**。
3. 方向判定优先级：**适配器声明 → 调用方覆盖 → 通用词表（兜底）**。
4. `s4._compare(directions=...)` 接线；适配器声明了方向却没覆盖到的指标
   会记一条 warning —— 让"未声明的方向"可见，而不是静默走兜底猜。

#### 另一处发现：序列的语义在不同领域不同

ML 里序列轴是 `epoch`（"训练轨迹"），材料里是**重复测量**（"不确定度"）。
把不同波长的 n 求平均**在物理上没有意义**，所以材料适配器把代表性条件
（550nm / 250nm / 可见区均值）固定住，在**重复轴**上取序列——
这样 `mean±std` 才是材料学里成立的口径。这也说明"序列必须有顺序"是个
领域假设，不该写进契约。

#### 仍未验证（材料方向）

- 只覆盖**光学常数**这一类表征。力学/热学/电学性质、XRD、显微、DSC 等未验证。
- 材料学常见的**参数扫描**（温度/成分/退火时间）尚未接成变体轴。
- 扰动幅度 `jitter` 是人为设定的，真实不确定度需要来自实验重复。
- COMSOL 批处理输出**未接入**。顺带记录一个真实痛点：COMSOL 的中文报错是
  GBK 编码，在本机的 UTF-8 日志里显示为乱码，排错时需要先转码。

### ⚠️ 仍未验证

- **s7 的「编译失败 → 抽错 → LLM 修复 → 重编译」闭环从未收敛过。**
  tectonic 二进制能下载、能通过 `--version`，但本机沙箱/ACL 拒绝进程在新建目录里
  写入（`os error 5`），一次成功编译都没有过。这是当前最大的未验证面。

- **`--sandbox docker` 只验证到"命令构造正确"**。真实容器执行需要一台能跑
  容器的 Linux 机器：CI 的 Windows runner 虽有 docker CLI，但
  `docker run` 因 Windows 容器限制失败。`test_docker_command_construction`
  覆盖了挂载、命令顺序、网络开关与 `--pids-limit` 门控，但没有真的启动过容器。

- **在 Windows 宿主上跑 Linux 容器**（Docker Desktop + WSL2）的情形未验证：
  `--pids-limit` 是按**宿主**判断是否传递的，那种组合下本可以传。

- **真实训练规模未验证**：所有实测都在秒级到分钟级的 CPU 合成任务或小型
  Lorenz-63 实验上完成。

### 待办

- 让 PDF 编译在真实编译上跑通一次，并把结果回灌到 s7 的错误修复规则。
- 若有人真的在 Windows 宿主 + Linux 容器组合下使用，把 `--pids-limit` 的
  宿主判断改为运行时探测 daemon。

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

[Unreleased]: https://github.com/Zoloa-hub/autoresearch-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Zoloa-hub/autoresearch-pipeline/releases/tag/v0.1.0
