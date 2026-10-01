"""给 PDF 闭环的两处修复加回归测试，并更新已经过期的文档声明。

## 测试什么（不需要网络）

关键不变量：**真实编译的子进程环境必须带 TECTONIC_CACHE_DIR**，
且指向与预检**同一个**目录。

这是本次找到的根因：预检设了缓存目录、真实编译没设，于是 tectonic 回去用
默认位置（不可写），既读不到已填充的缓存也不会下载缺失宏包，
最终在 TeX 层报「File `size11.clo' not found」——一个指向完全错误方向的错误。
**预检验证了一个生产中从不使用的配置**，是典型的虚假信心检查。

测试用拦截 subprocess.run 的方式验证环境，因此完全离线可跑。
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- 1) 回归测试 --
TA = ROOT / "autoresearch" / "tests" / "test_sandbox_latex.py"
t = TA.read_text(encoding="utf-8")

NEW = '''

def test_tectonic_cache_dir_is_passed_to_real_compile() -> None:
    """真实编译必须带 TECTONIC_CACHE_DIR，且与预检用同一个目录。

    这是「PDF 编译闭环从未收敛」的根因，也是本项目里最典型的一个
    **虚假信心检查**：

      * 预检 `_tectonic_cache_preflight()` 自己设了缓存目录 -> 通过
      * 真实编译 `_run_tectonic()` **没设** -> tectonic 用默认位置
        （Windows 的 `%LOCALAPPDATA%\\Tectonic`，在本环境不可写）
      * 结果：既读不到预检填充的缓存，也不会下载缺失宏包，
        直接在 TeX 层报 `File `size11.clo' not found`
      * 这个错误**指向完全错误的方向**（看起来像模板缺宏包）

    测试用拦截 `subprocess.run` 的方式验证，**不需要网络**。
    """
    section("tectonic: 真实编译必须收到缓存目录")
    import subprocess as _sp
    from pathlib import Path as _P

    import autoresearch.tools.latex as latex_mod

    calls: list[tuple[list[str], dict]] = []
    real_run = _sp.run

    def spy(cmd, **kwargs):  # noqa: ANN001
        calls.append((list(cmd), dict(kwargs)))
        raise FileNotFoundError("intercepted")  # 不真的执行

    work = _scratch("tectonic_cache_env")
    tex = work / "main.tex"
    tex.write_text(
        "\\\\documentclass{article}\\n\\\\begin{document}x\\\\end{document}\\n",
        encoding="utf-8",
    )

    comp = LatexCompiler(Cfg(), work, Recorder())
    # 跳过探测，直接进编译；同时避免真的跑 --version
    comp._detect_done = True
    comp._tectonic_supports_x = True
    comp._engine = "tectonic"

    saved_run = latex_mod.subprocess.run
    latex_mod.subprocess.run = spy
    try:
        try:
            comp.compile(tex, engine="tectonic")
        except Exception:  # noqa: BLE001 - 拦截会抛，属预期
            pass
    finally:
        latex_mod.subprocess.run = saved_run

    compile_calls = [
        (cmd, kw) for cmd, kw in calls if "compile" in cmd
    ]
    check(bool(compile_calls), "至少发出了一次 compile 调用", str(len(calls)))
    if compile_calls:
        _cmd, kwargs = compile_calls[0]
        env = kwargs.get("env") or {}
        cache = env.get("TECTONIC_CACHE_DIR")
        check(
            bool(cache),
            "**真实编译的子进程环境带 TECTONIC_CACHE_DIR**"
            "（不带时 tectonic 会用不可写的默认位置，且不会下载缺失宏包）",
            f"env keys={sorted(env)[:6]}",
        )
        if cache:
            expected = comp.vendor_dir() / "tectonic_cache"
            check(
                _P(cache) == expected,
                "缓存目录与预检指向**同一个**位置"
                "（不一致就是虚假信心检查：验证的配置 != 生产用的配置）",
                f"{cache} vs {expected}",
            )
        check(
            env.get("PYTHONIOENCODING") == "utf-8",
            "仍然强制 UTF-8（原有行为未被破坏）",
            repr(env.get("PYTHONIOENCODING")),
        )

    # _child_env 直接验：传了就设，不传就不设（pdflatex 等不需要）
    with_cache = latex_mod._child_env(work / "somewhere")
    check(
        with_cache.get("TECTONIC_CACHE_DIR") == str(work / "somewhere"),
        "_child_env(cache_dir=...) 会设 TECTONIC_CACHE_DIR",
        repr(with_cache.get("TECTONIC_CACHE_DIR")),
    )
    without = latex_mod._child_env()
    check(
        "TECTONIC_CACHE_DIR" not in without or without["TECTONIC_CACHE_DIR"] == os.environ.get(
            "TECTONIC_CACHE_DIR"
        ),
        "_child_env() 不传时不擅自改缓存目录（pdflatex 路径不受影响）",
        repr(without.get("TECTONIC_CACHE_DIR")),
    )


def test_s7_repairs_on_errors_not_only_ok() -> None:
    """s7 的修复闭环必须按**抽出的错误**触发，而不是只看 `ok`。

    `ok` 的含义是「**产出了 PDF**」。实测确认：注入一个未闭合的数学模式后，
    tectonic 退出码仍为 0、PDF 仍被写出，同时抽出 `! Missing $ inserted.`
    —— 即 `ok=True` 且 errors 非空。

    早期 s7 只要 `ok` 就 break，于是那个错误既不会被修复也不会进任何警告：
    **一篇带 LaTeX 错误的论文被静默放行**，而它是整条管线的最终产物。

    这里做**结构性断言**：源码里 break 的条件必须同时包含 errors 判据，
    且收尾处必须上报残留错误。真正跑一遍 s7 需要网络（首次要下宏包），
    不适合放在离线套件里。
    """
    section("s7: 修复判据必须包含 errors")
    src = (ROOT / "autoresearch" / "stages" / "s7_compile.py").read_text(encoding="utf-8")

    marker = "if result is not None and result.ok and not errors:"
    check(
        marker in src,
        "break 条件同时要求「无抽出的错误」"
        "（只判 ok 会放行带 LaTeX 错误的 PDF）",
        "找到" if marker in src else "未找到该条件",
    )
    check(
        "ok and not errors" in src or "and not errors" in src,
        "判据里出现了 errors 项",
    )
    check(
        "仍有" in src and "LaTeX 错误未被修复" in src,
        "循环结束后如实上报残留错误（降级交付，而不是假装干净）",
        "未找到残留错误上报" if "LaTeX 错误未被修复" not in src else "已上报",
    )
    # 反向：确认没有残留"只看 ok"的旧判据
    check(
        "if result is not None and result.ok:\\n                break" not in src,
        "旧的「只看 ok」判据已被替换（防止回退）",
    )
'''

anchor = "\ndef main() -> int:"
if "def test_tectonic_cache_dir_is_passed_to_real_compile(" not in t:
    t = t.replace(anchor, NEW + anchor, 1)
    reg = """    _progress('test_tectonic_cache_dir_is_passed_to_real_compile')
    try:
        test_tectonic_cache_dir_is_passed_to_real_compile()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_tectonic_cache_dir_is_passed_to_real_compile 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_s7_repairs_on_errors_not_only_ok')
    try:
        test_s7_repairs_on_errors_not_only_ok()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_s7_repairs_on_errors_not_only_ok 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    print("\\n" + "=" * 70)"""
    t = t.replace('    print("\\n" + "=" * 70)', reg, 1)
    TA.write_text(t, encoding="utf-8")
    print("  已加 2 个 PDF 闭环回归测试")
else:
    print("  已存在")

import py_compile  # noqa: E402

py_compile.compile(str(TA), doraise=True)
print("  编译通过")
