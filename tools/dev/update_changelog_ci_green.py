"""CI 全绿后更新 CHANGELOG 的"未验证"清单 —— 过期的声明必须改掉。

这个项目的核心价值是文档诚实，所以"CI 从未执行过"这条一旦不再成立，
必须立刻改写；同时把 CI 真实抓到的缺陷如实记进去（那是这批改动最有价值的部分）。
"""

from __future__ import annotations

import pathlib

p = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"
t = p.read_text(encoding="utf-8")

OLD = """### ⚠️ 仍未验证

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
"""

NEW = """### ✅ CI 已在真实 runner 上跑通（三平台 × 三 Python）

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
"""

if OLD in t:
    t = t.replace(OLD, NEW, 1)
    p.write_text(t, encoding="utf-8")
    print("  CHANGELOG 未验证清单已更新")
else:
    print("  MISS: 未找到原文")

# README 侧也可能有同样的过期声明
for rel in ("autoresearch/README.md", "README.md"):
    q = pathlib.Path(__file__).resolve().parents[1] / rel
    if not q.is_file():
        continue
    s = q.read_text(encoding="utf-8")
    hits = [
        ln for ln in s.splitlines()
        if ("CI" in ln or "ci.yml" in ln) and ("未" in ln or "从未" in ln or "没有" in ln)
    ]
    if hits:
        print(f"  {rel} 里可能过期的 CI 声明:")
        for h in hits:
            print(f"    {h.strip()[:150]}")
    else:
        print(f"  {rel}: 无过期 CI 声明")
