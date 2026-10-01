"""更新文档：PDF 编译闭环已收敛（最大的未验证面被消除了）。

同时记录两个真实缺陷：
  1. 虚假信心检查：预检验证了一个生产中从不使用的配置
  2. s7 只按 ok 判断 → 带 LaTeX 错误的论文被静默放行
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ README --
r = ROOT / "README.md"
t = r.read_text(encoding="utf-8")

OLD = """- **PDF 编译闭环从未收敛过。** tectonic 二进制就位，但沙箱/ACL 拒绝它在新建目录里
  写入（`os error 5`），一次成功编译都没跑通。需要在无沙箱终端跑一次
  `autoresearch/vendor/tectonic/tectonic.exe -X compile paper/main.tex --outdir paper`
  把 bundle 缓存建起来。**这是当前最大的未验证面。**"""
NEW = """- ~~PDF 编译闭环从未收敛过。~~ **已修复（2026-10）。** 真实根因不是沙箱：预检
  `_tectonic_cache_preflight()` 自己设了 `TECTONIC_CACHE_DIR` 并验证通过，
  而**真实编译没设**，于是 tectonic 回去用默认位置（`%LOCALAPPDATA%\\Tectonic`，
  本环境不可写）——既读不到预检填充的缓存，也不会下载缺失宏包，最终在 TeX 层报
  `File `size11.clo' not found`，一个**指向完全错误方向**的错误。
  **预检验证了一个生产中从不使用的配置**，是典型的虚假信心检查。
  修复后：真实模板产出 **27 KB PDF**，且「注入错误 → 抽错 → 修复 → 重编译」闭环收敛。"""
if OLD in t:
    t = t.replace(OLD, NEW, 1)
    print("  README: 已更新 PDF 那一条")

# 补一条新发现的缺陷
OLD2 = """| **参数覆盖静默失效，会让整个扫描变成"同一实验跑 N 次"**"""
NEW2 = """| **带 LaTeX 错误的论文曾被静默放行** |
`s7` 的修复闭环只按 `ok` 判断，而 `ok` 的含义是「产出了 PDF」。实测：注入一个未闭合的
数学模式后 tectonic 退出码仍为 0、PDF 仍被写出，同时抽出 `! Missing $ inserted.`。
于是那个错误既不会被修，也不进任何警告——**而它是整条管线的最终产物**。
现在判据改为「产出 PDF **且**无抽出的错误」，残留错误会如实上报。
<tr><td>

| **参数覆盖静默失效，会让整个扫描变成"同一实验跑 N 次"**"""
if OLD2 in t and "带 LaTeX 错误的论文曾被静默放行" not in t:
    t = t.replace(OLD2, NEW2, 1)
    print("  README: 已补 LaTeX 静默放行那一条")

r.write_text(t, encoding="utf-8")

# --------------------------------------------------------------- CHANGELOG --
c = ROOT / "CHANGELOG.md"
ct = c.read_text(encoding="utf-8")

SECTION = '''### 🎯 PDF 编译闭环首次收敛 —— 消除最大的未验证面

CHANGELOG 顶部长期挂着一条「**s7 的编译→抽错→修复→重编译闭环从未收敛过**」，
并注明这是当前最大的未验证面。它现在解决了，而且根因不是原先判断的沙箱问题。

#### 根因：一个「虚假信心检查」

拦截真实子进程调用后确证：

```
预检 _tectonic_cache_preflight()：
    cache = <vendor>/tectonic/tectonic_cache     ← 设了
    → 成功，缓存被填充

真实编译 _run_tectonic()：
    cache = None                                  ← **没设**
    → 失败：File `size11.clo' not found（0.148 秒，没有尝试下载）
```

`TECTONIC_CACHE_DIR` 未设时 tectonic 用默认位置（`%LOCALAPPDATA%\\Tectonic`），
那里在本环境**不可写**——也就是最初诊断到的 `os error 5`。于是它既读不到预检
填充的缓存、也写不了默认缓存、**并且不会去下载缺失宏包**，直接在 TeX 层报

    ! LaTeX Error: File `size11.clo' not found.

这个错误**指向完全错误的方向**：看起来像"模板缺宏包"，实际是"缓存目录没配对"。

**教训（值得单独记）**：预检验证的是「缓存放在 X 时能编译」，而生产用的是
**没有 X**。一个检查如果验证的不是生产中真正跑的那套配置，它比没有检查更糟——
因为它会把注意力引开，而且给出绿灯。

修复：`_child_env(cache_dir=...)`，真实编译传入与预检**同一个**目录。

#### 顺带发现：`ok=True` 不等于「没有 LaTeX 错误」

实测确认：注入一个未闭合的数学模式（`$x = 1`）后，tectonic **退出码仍为 0、
PDF 仍被写出**，同时抽出 `! Missing $ inserted.`——即 `ok=True` 且 errors 非空。

而 `s7` 的闭环只按 `ok` 判断：

```python
if result is not None and result.ok:
    break          # ← 立刻退出，那个错误既不被修、也不进任何警告
```

**一篇带 LaTeX 错误的论文被静默放行**，而它是整条管线的最终产物。
（未闭合的 `$` 会让正文出现乱码或吞掉后续内容，而 PDF 本身完全正常打开。）

修复：判据改为「产出 PDF **且**无抽出的错误」；循环结束后若仍有残留错误，
**如实记进 warnings**——降级交付，而不是假装干净。

#### 验证结果

```
detect()                     -> 'tectonic'
compile(真实论文模板)         -> ok=True  27072 字节  %PDF-1.5
注入错误 -> 抽错 -> 修复 -> 重编译  -> ok=True  pdf=True   闭环收敛
```

#### 附带修正的一个测试假设

我最初的探针直接编译**裸模板**，得到 `main.tex:66: Missing $ inserted`——
而第 66 行是空行。真因是 `\\maketitle` 处理**未替换的 `__TITLE__`**：
LaTeX 里 `_` 是数学模式专用字符。模板本来就不该裸编译。
这顺带确认了占位符自检（`template_placeholders_unfilled`）的价值：
**未替换时报的错误完全不指向真正的原因。**

'''

anchor = "### ⚠️ 仍未验证"
if "PDF 编译闭环首次收敛" not in ct:
    # 插到"仍未验证"之前（最新的在前）
    idx = ct.find(anchor)
    if idx > 0:
        # 找到该节之前的最后一个空行位置，保持结构
        ct = ct[:idx] + SECTION + ct[idx:]
        c.write_text(ct, encoding="utf-8")
        print("  CHANGELOG: 已加 PDF 闭环收敛段")
    else:
        print("  MISS CHANGELOG: 找不到锚点")

# 更新仍未验证一节里的 PDF 条目
for old, new in [
    ("- **s7 的「编译失败 → 抽错 → LLM 修复 → 重编译」闭环从未收敛过。**",
     "- ~~s7 的编译闭环从未收敛~~ —— **已解决**，见上方「PDF 编译闭环首次收敛」。"),
]:
    ct = c.read_text(encoding="utf-8")
    if old in ct:
        c.write_text(ct.replace(old, new, 1), encoding="utf-8")
        print("  CHANGELOG: 已更新仍未验证里的 PDF 条目")
        break
