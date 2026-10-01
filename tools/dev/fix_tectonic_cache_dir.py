r"""修 PDF 编译闭环的真实根因：真实编译没有设置 TECTONIC_CACHE_DIR。

## 证据链（本次实测得到）

拦截管线的真实子进程调用：

    预检（_tectonic_cache_preflight）：
        cwd   = <vendor>/tectonic/.tectonic_dl_xxx
        cache = <vendor>/tectonic/tectonic_cache      ← 设了
        cmd   = tectonic -X compile preflight.tex --outdir ...
        -> 成功，缓存被填充

    真实编译（compile -> _run_tectonic）：
        cwd   = <.autoresearch/_pdfdiag3/paper>
        cache = None                                  ← **没设**
        cmd   = tectonic -X compile main.tex --keep-logs --outdir ...
        -> 失败：File `size11.clo' not found（0.148 秒，没有尝试下载）

## 为什么这导致「闭环从未收敛」

`TECTONIC_CACHE_DIR` 未设时，tectonic 用默认缓存位置
（Windows: `%LOCALAPPDATA%\Tectonic`）。那个位置**不可写**——正是最初诊断到的
`os error 5`。于是 tectonic 既读不到预检填充的缓存，又写不了默认缓存，
**不会去下载缺失的宏包**，直接在 TeX 层报：

    ! LaTeX Error: File `size11.clo' not found.

这个错误**完全指向错误的方向**：它看起来像"模板缺宏包"，
实际是"缓存目录没配对"。而预检之所以通过，是因为它自己设了缓存目录——
**预检验证的是一个生产中从不使用的配置**。

这是一个「虚假信心检查」：它证明"缓存放在 X 时能编译"，
而真实运行压根不用 X。比没有检查更糟，因为它把注意力引开了。

## 修法

真实编译的子进程环境里也设 `TECTONIC_CACHE_DIR`，与预检用**同一个**目录。
`_child_env()` 增加一个 `cache_dir` 参数，由 `_run_tectonic` 传入
`self.vendor_dir() / "tectonic_cache"`。
"""

from __future__ import annotations

import pathlib

p = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "tools" / "latex.py"
t = p.read_text(encoding="utf-8")
applied = 0

# --- 1) _child_env 接受 cache_dir ---
OLD_ENV = '''def _child_env() -> dict:
    """Environment for LaTeX child processes.

    Forces UTF-8 on the child's stdio so non-ASCII engine messages (e.g. the
    localized Windows "access denied" text) come back readable instead of
    mojibake, and points the temp vars at the current drive so no child has to
    reach a per-user temp path.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    return env'''
NEW_ENV = '''def _child_env(cache_dir: str | os.PathLike[str] | None = None) -> dict:
    """Environment for LaTeX child processes.

    Forces UTF-8 on the child's stdio so non-ASCII engine messages (e.g. the
    localized Windows "access denied" text) come back readable instead of
    mojibake, and points the temp vars at the current drive so no child has to
    reach a per-user temp path.

    ``cache_dir``：**必须传**。``tectonic`` 把宏包缓存放在
    ``TECTONIC_CACHE_DIR``，不设时它用默认位置（Windows 是
    ``%LOCALAPPDATA%\\Tectonic``）——那个位置在本项目的运行环境里**不可写**，
    于是 tectonic 既读不到已填充的缓存、也写不了新缓存，**而且不会去下载缺失宏包**，
    直接在 TeX 层报「File `size11.clo' not found」。

    这个错误指向完全错误的方向（看起来像模板缺宏包，实际是缓存目录没配对），
    而它曾经让 s7 的编译闭环**从未收敛**。教训值得写在这里：
    **预检必须验证生产中真正使用的那套配置**——早期预检自己设了缓存目录，
    真实编译却没设，于是预检通过而生产失败，是个典型的虚假信心检查。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    if cache_dir is not None:
        env["TECTONIC_CACHE_DIR"] = str(cache_dir)
    return env'''
if OLD_ENV in t:
    t = t.replace(OLD_ENV, NEW_ENV, 1)
    applied += 1
    print("  latex.py: _child_env 已接受 cache_dir")
else:
    print("  MISS: _child_env")

# --- 2) 真实编译传 cache_dir ---
OLD_CALL = '''                    cwd=str(workdir),
                    env=_child_env(),
                )'''
NEW_CALL = '''                    cwd=str(workdir),
                    # 必须与预检用同一个缓存目录，否则预检通过、生产失败
                    env=_child_env(self.vendor_dir() / "tectonic_cache"),
                )'''
if OLD_CALL in t:
    t = t.replace(OLD_CALL, NEW_CALL, 1)
    applied += 1
    print("  latex.py: 真实编译已传缓存目录")
else:
    print("  MISS: 真实编译调用点")

# --- 3) 其它 _child_env() 调用点（非 tectonic 的 pdflatex 等）不动 ---
n_other = t.count("_child_env()")
print(f"  其余 _child_env() 调用点: {n_other} 处（pdflatex 等不用 tectonic 缓存，保持不变）")

p.write_text(t, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(p), doraise=True)
print(f"  已改 {applied} 处；编译通过")
