# 贡献指南

本项目是自动化科研管线的参考实现。它的价值不在于「能生成一篇论文」，而在于
**每一步都可审计、可复现、并且诚实地报告自己失败在哪一步**。因此对本仓库的改动，
除了「功能是否可用」，还有一条同等重要的标准：**它有没有削弱可审计性与诚实性**。

开发环境、目录结构、九阶段职责与已知限制见 `autoresearch/README.md`；
**接口契约见 `autoresearch/CONTRACTS.md`**。

---

## 1. 本地跑完整测试

七个套件，全部可以独立运行：

```powershell
python -m autoresearch.tests.test_config_llm        # 配置 + LLM 客户端 + mock 意图路由
python -m autoresearch.tests.test_retrieve          # 检索 + PDF 解析（离线）
python -m autoresearch.tests.test_sandbox_latex     # 沙箱 + LaTeX（离线）
python -m autoresearch.tests.test_figures_metrics   # 指标统计 + 制图（离线）
python -m autoresearch.tests.test_prompts           # 提示词引擎 + 模板完整性
python -m autoresearch.tests.test_adapters          # 实验后端适配器（协议 / 指标契约 / 变体推导）
python -m autoresearch.tests.test_pipeline          # ★ 端到端：mock+offline 跑完整条管线
```

一次跑完（退出码非 0 即失败）：

```powershell
python -m autoresearch.tests.test_config_llm; `
python -m autoresearch.tests.test_retrieve; `
python -m autoresearch.tests.test_sandbox_latex; `
python -m autoresearch.tests.test_figures_metrics; `
python -m autoresearch.tests.test_prompts; `
python -m autoresearch.tests.test_adapters; `
python -m autoresearch.tests.test_pipeline
```

也可以走 pytest（`pyproject.toml` 已配好 `testpaths` / `python_files` / `python_functions`，
并加了 `-q`）：

```powershell
pytest                       # 全部
pytest autoresearch/tests/test_adapters.py
```

### 1.1 `无网络、无 API Key` 是设计不变量

**这七个套件都不需要网络，也不需要 API Key。** 这不是巧合，而是设计目标：

* `--llm-provider mock` 提供离线启发式后端（`llm/mock.py`），全流程可跑通；
* 检索类测试一律用 `offline=True` 短路，或用纯函数、本地 fixture 覆盖；
* 沙箱与 LaTeX 测试只验证**降级路径**（无可执行引擎时返回 `None` 并记事件，不 raise）。

因此 CI 能在三平台绿灯，且任何人都能在断网机器上验证一颗改动。这一点由 CI 直接强制：
`.github/workflows/ci.yml` 在 ubuntu / windows / macos 三平台 × Python 3.10 / 3.12 / 3.13
上**不安装任何第三方依赖**，逐个跑这七个套件（`Suite 1/7` … `Suite 7/7`）。

**一个改动如果让某个套件需要网络或 Key，它必须被拒绝，或者该测试必须被改成离线的
（hermetic）。** 唯一的例外是**显式开关**的网络测试，它们默认跳过、不影响结果：

```powershell
# test_sandbox_latex.py::test_install_tectonic_real_network
# 默认打印 SKIPPED；只有显式置位才会真的去下载 tectonic
$env:AUTORESEARCH_TEST_NETWORK = "1"
python -m autoresearch.tests.test_sandbox_latex
```

新的网络相关测试请沿用同一个模式：默认跳过并打印原因，用环境变量显式启用。

---

## 2. 测试约定

**每个套件同时是「可 `python -m` 运行的脚本」和「pytest 可收集的测试」。** 具体做法：

* 模块里是若干 `def test_xxx() -> None:` 函数，每个只覆盖一个主题；
* 文件末尾有 `def main() -> int:` 顺序调用它们，并 `if __name__ == "__main__": sys.exit(main())`；
* **套件内部不依赖 pytest**：断言用本文件里定义的局部辅助函数，而不是裸 `assert`。

```python
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
```

```python
def main() -> int:
    ...
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} of {_CHECKS} checks")
        for failure in _FAILURES[:40]:
            print("  - " + failure)
        return 1
    print(f"PASSED {_CHECKS} checks")     # ← 每个套件的成功口径
    return 0


def test_adapter_suite() -> None:
    """pytest 收集入口。"""
    assert main() == 0
```

为什么用 `check()` / `eq()` 而不是 `assert`：

* 一个失败**不会中止整个套件**——失败会累积，最后一次性列出，你能一次看到所有问题；
* 失败信息自带标签与 `expected/got`，不需要读堆栈；
* 同一份代码既是脚本又是 pytest 用例，两种跑法给出**完全一致**的结论。

新套件的约定：

1. 打印 `PASSED n checks`（成功）或 `FAILED k of n checks` + 失败清单（失败）；
2. 失败时 `main()` 返回 1、`sys.exit(main())`；成功返回 0；
3. 只写**临时**文件，并且落在工作区的 `tmp/` 或 `.autoresearch/test_tmp/` 下，
   退出时清理（参考 `test_adapters._scratch()`）；
4. 不引入第三方测试依赖。`pytest` 只是可选的收集器（`[project.optional-dependencies] dev`）。

---

## 3. 接口契约：先读 `CONTRACTS.md`

**改动任何模块的公开签名之前，先读 `autoresearch/CONTRACTS.md`，并在同一个 commit 里
更新它。**

`CONTRACTS.md` 自称「唯一权威接口定义」，这句话是有约束力的：阶段之间靠状态键与
冻结的数据形状通信，签名漂移不会在改动处报错，而是在几个阶段之后才炸。同一个 commit
里同时改代码与契约，是让 reviewer 能一次性判断「这个改动是否所有调用方都跟上了」的
唯一办法。

涉及面较广的条目包括：

* `tools/sandbox.py` 的 `ExecResult` / `Sandbox.run_command(argv, ...)`；
* `tools/metrics.py` 的 `parse_metrics() -> dict[str, list[float]]` 与 `to_latex_table()`；
* `stages/base.py` 的 `Stage` / `StageResult`；
* `graph/state.py` 的状态键表与 `Artifact`；
* `llm/client.py` 的 `LLMClient` / `LLMResponse`；
* `adapters/base.py` 的适配器协议（`BaseExperimentAdapter`、`RunSpec`、`MetricSeries`）。

若新接口尚未稳定，先在 `CONTRACTS.md` 里标注，稳定后再去掉标注——不要留一个
「文档说 FROZEN、代码还在改」的中间状态。

---

## 4. 两条不可破坏的硬不变量

### (a) `python -m autoresearch.cli demo` 必须零第三方依赖、无网络地跑完

```powershell
python -m autoresearch.cli demo
```

它等价于 `run --llm-provider mock --offline`，并且是**零第三方依赖的底线**：
在只有 Python 标准库（+ 可选 numpy）的机器上也能产出完整交付物（PDF 除外，
此时改为写 `report/COMPILE_BLOCKED.md`）。这是三平台 CI 矩阵成立的前提，
也是新用户第一次接触这个项目时看到的东西。

`pyproject.toml` 里 `dependencies = []`，所有额外能力都在
`[project.optional-dependencies]` 下（`llm` / `figures` / `pdf` / `cjk` / `http` / `dev`）。
**给核心路径加一条第三方 import，就是破坏这条不变量**——正确做法是延迟导入 + 缺失时
降级并记事件（`figures.py`、`pdfx.py`、`latex.py` 都是这么写的）。

### (b) 不许提交任何密钥

* `.env` 已被 `.gitignore` 忽略；`.env.example` 只放占位符；
* `CONTRACTS.md` 里 `LLMConfig.api_key` 的约定是「**只从环境变量读，绝不落盘**」；
* 交付物、`state.json`、`events.jsonl` 里都不应出现 Key —— 需要外发配置时用
  `AutoResearchConfig.redacted_dict()`（`api_key` 渲染成 `***`，空值保持 `None`），
  `autoresearch/tests/test_config_llm.py::test_redacted_dict` 就在守这条线。

提交前自查：

```powershell
git diff --cached | Select-String -Pattern 'sk-|api[_-]?key\s*=|Bearer '
```

CI 与 reviewer 都会把这类内容当作阻断项，而不是「顺手修一下」的小问题。

---

## 5. 添加一个新的实验适配器

完整指南：**[`docs/adapters.md`](docs/adapters.md)**（为什么需要适配器、最小可用示例、
协议完整参考、指标契约、`RunSpec` 与消融、三种注册方式、`ScriptWrapperAdapter` 详解、
`supported_variants()`、一个 PyTorch + `torchrun` 的真实例子、诚实边界）。

最短路径：

1. 复制 `docs/adapters.md` §2 的最小适配器，只实现 `build_command()` 与
   `parse_results()`；
2. `parse_results()` 返回 `dict[str, list[float]]`——**逐 epoch / 逐 fold 的序列**，
   不是最终标量（标量会让学习曲线、boxplot、跨种子统计全部退化）；
3. 脚本 `argparse choices` 不认全部变体名时实现 `supported_variants()`；
4. 用 `validate_environment()` 把「缺 torch / 无 CUDA / 数据不存在」提前问清楚；
5. 覆盖 `quality_note()`，写清这组结果能支撑什么级别的结论；
6. 用 `--experiment-adapter path/to/adapter.py` 或 `module:Class` 或 entry point 注册
   （entry point 组名常量是 `autoresearch.adapters.ENTRY_POINT_GROUP`）；
7. 在 `autoresearch/tests/test_adapters.py` 里加断言，尤其是**契约边界**：返回类型、
   命令 argv 形状、环境预检失败的措辞、变体过滤。

最后一条特别重要：适配器重构最大的风险不是功能缺失，而是**契约漂移**——返回值结构、
变体命名、预检失败的处理方式一旦与主干不一致，错误会一路漂到画图或写作阶段才炸，
那时归因成本极高。

---

## 6. 一个好的改动长什么样

**优先降级，不要 raise。** 阶段、工具层、适配器的默认行为都应该是「尽力产出部分结果 +
如实记录为什么少了什么」。`OPTIONAL_STAGES`、`parse_metrics` 失败返回 `{}`、
Docker 不可用退回 `SubprocessSandbox`、无 LaTeX 引擎写 `COMPILE_BLOCKED.md`，
都是这个模式。只有真正致命的四件事（检索、构思、规划、实验）才允许终止管线——
没有实验就没有论文，继续跑只是在烧 token。**加一个会中断管线的 raise 之前，
先问：这个失败真的让后续所有阶段都失去意义了吗？**

**每个产物都要登记。** 落到磁盘的东西必须通过 `ctx.save_json()` / `ctx.save_text()` /
`ctx.artifact(p, kind, stage)` 走一遍，这样它才会进 `artifacts` 列表、带 `sha256` 与字节数、
进交付包、并能被 `cli verify` 与 `s9` 的产物清单看见。手工 `open(..., "w")` 写出来的
文件在断点续跑与交付打包里都是不存在的。

**绝不让管线声称它无法证明的成功。** 这条覆盖了很多具体规则：

* 指标可以难看，但不许是假的。自纠错补丁若删掉指标输出、加 `except: pass`、
  写死高分，必须被标记 `validity_flag` 并进入未解决问题清单；
* 数字只能来自证据。图表由 `tools/figures.py` / `tools/metrics.py` 确定性生成，
  LLM 只被允许「解释」；写作必须遵守 `s5` 的证据分级
  （`supported` / `partially_supported` / `not_supported` / `inconclusive`）；
* 有指标但不可对照时，要给出**可解释的失败**（「缺少 baseline 指标」
  「没有任何变体与它有共同指标名」），而不是静默返回空；
* 交付物必须包含「我哪里没做到」（`report/open_issues.json`、`FINAL_REPORT.md` 第 3 节）。

**改行为的 PR，请同时改测试，并说明降级路径。** reviewer 会问两个问题：
「这条新代码在依赖缺失/离线/无 GPU 时会怎样？」以及「它有没有把某个失败变成静默成功？」
能直接回答这两个问题的 PR，通常就是好 PR。
