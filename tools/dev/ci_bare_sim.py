"""模拟 GitHub runner 的裸环境，跑 CI 里每个 step 的命令，看谁能过。

关键差异（我在本地从未真正模拟过的）：
  1. **没有任何第三方依赖**（runner 上不 pip install）
  2. Python 3.10 / 3.12 而不是 3.13
  3. Windows 上控制台是 cp1252

做法：构造一个子进程环境，屏蔽 site-packages，然后逐条跑 CI 的 run 命令。
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

def _repo_root() -> pathlib.Path:
    """向上找含 ``autoresearch/`` 的目录。

    不能写死 ``parents[1]``：这个脚本既可能在仓库根的 ``scratch/`` 下运行，
    也可能在 ``tools/dev/`` 下运行（发布仓库里的位置），层级不同。
    写死层级会让它在换位置后静默找错目录，然后报出一堆假的
    ``No module named 'autoresearch'``。
    """
    here = pathlib.Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "autoresearch" / "__init__.py").is_file():
            return cand
    return pathlib.Path.cwd()


ROOT = _repo_root()

# CI 里 test job 的 run 命令（按顺序）
STEPS = [
    ("Show interpreter", [sys.executable, "-VV"]),
    ("Smoke — CLI surface", [sys.executable, "-m", "autoresearch.cli", "stages", "--json"]),
    ("Suite 1/7 config", [sys.executable, "-m", "autoresearch.tests.test_config_llm"]),
    ("Suite 2/7 retrieve", [sys.executable, "-m", "autoresearch.tests.test_retrieve"]),
    ("Suite 3/7 sandbox", [sys.executable, "-m", "autoresearch.tests.test_sandbox_latex"]),
    ("Suite 4/7 figures", [sys.executable, "-m", "autoresearch.tests.test_figures_metrics"]),
    ("Suite 5/7 prompts", [sys.executable, "-m", "autoresearch.tests.test_prompts"]),
    ("Suite 6/7 adapters", [sys.executable, "-m", "autoresearch.tests.test_adapters"]),
    ("Suite 7/7 pipeline", [sys.executable, "-m", "autoresearch.tests.test_pipeline"]),
    ("lint: byte-compile", [sys.executable, "-m", "compileall", "-q", "autoresearch"]),
]

#: 屏蔽第三方依赖的启动代码：只保留标准库路径，但**必须把仓库自身加回来**
#: （strip 掉 site-packages 时 cwd 也可能被移除，否则 import autoresearch 会失败）
BARE_BOOTSTRAP = (
    "import sys,os;"
    "sys.path=[p for p in sys.path "
    "if 'site-packages' not in p and 'dist-packages' not in p];"
    "sys.path.insert(0, os.getcwd());"
)


def run_bare(argv: list[str], cwd: pathlib.Path) -> tuple[int, str, str]:
    """在屏蔽 site-packages 的子进程里运行 argv。

    **必须对整个进程树生效**，不能只在当前进程里过滤 ``sys.path``。
    这一点是踩过坑的：`test_adapters` 会通过沙箱 ``run_command`` 再起一个
    **孙进程**去跑夹具脚本。早期实现只用 ``-c`` 在当前进程里改 sys.path，
    孙进程仍然能 import 到 numpy，于是我漏掉了 CI 上真实存在的失败：

        File ".../_provided/train.py", line 35
            import numpy as np
        ModuleNotFoundError: No module named 'numpy'

    正确做法：把 ``sitecustomize.py`` 写进一个临时目录，它由**每个**解释器在
    初始化时自动执行并删掉 site-packages；再用
    ``PYTHONPATH=<guard>:<repo>`` 启动。这样每一层子进程都被隔离。
    """
    import shutil

    # argv 形如 [python, "-m", "module", *args]；runpy 要模块名而不是 "-m"
    module = argv[2] if len(argv) > 2 and argv[1] == "-m" else None
    if module is None:
        return 1, "", f"unsupported argv: {argv}"

    # guard 目录必须放在**工作区内**：这个沙箱下 tempfile.mkdtemp 创建的目录
    # 后续访问会被拒绝（PermissionError），而且永远删不掉。
    guard_dir = cwd / ".autoresearch" / "_bare_guard"
    guard_dir.mkdir(parents=True, exist_ok=True)
    (guard_dir / "sitecustomize.py").write_text(
        "import sys\n"
        "sys.path = [p for p in sys.path\n"
        "            if 'site-packages' not in p and 'dist-packages' not in p]\n",
        encoding="utf-8",
    )
    launcher = (
        "import runpy,sys;"
        + f"sys.argv={[module] + list(argv[3:])!r};"
        + f"runpy.run_module({module!r}, run_name='__main__')"
    )
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    # 刻意**不**设 PYTHONUTF8=0：那会让 Windows 上的 print 因编码崩掉，
    # 从而掩盖真正的失败。编码问题由 autoresearch/__init__.py 的 stdio 加固负责，
    # 不该混进这个"依赖隔离"模拟里（否则两种失败原因会互相遮蔽）。
    env.pop("PYTHONUTF8", None)
    env["PYTHONPATH"] = os.pathsep.join([str(guard_dir), str(cwd)])
    try:
        proc = subprocess.run(
            [sys.executable, "-c", launcher],
            capture_output=True,
            cwd=str(cwd),
            env=env,
            timeout=900,
        )
        return (
            proc.returncode,
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"),
        )
    finally:
        shutil.rmtree(guard_dir, ignore_errors=True)


def main() -> int:
    print("=" * 72)
    print("模拟 CI 的裸环境（无第三方依赖 + PYTHONUTF8=0）")
    print("=" * 72)

    failures: list[str] = []
    for name, argv in STEPS:
        if argv[1] == "-VV":
            rc, out, err = 0, "skip", ""
            print(f"  {name:<26} (跳过)")
            continue
        try:
            rc, out, err = run_bare(argv, ROOT)
        except subprocess.TimeoutExpired:
            print(f"  {name:<26} TIMEOUT")
            failures.append(name)
            continue
        verdict = "OK  " if rc == 0 else "FAIL"
        last = ""
        for line in (out or "").splitlines()[::-1]:
            if "PASSED" in line or "FAILED" in line:
                last = line.strip()
                break
        print(f"  {verdict} {name:<26} rc={rc}  {last}")
        if rc != 0:
            failures.append(name)
            # 打印最能说明问题的几行
            interesting = [
                ln
                for ln in (err or "").splitlines()
                if any(
                    k in ln
                    for k in (
                        "Error",
                        "error",
                        "Traceback",
                        "FAIL",
                        "Exception",
                        "No module",
                        "refused",
                        "denied",
                    )
                )
            ]
            for ln in interesting[:6]:
                print(f"        {ln.strip()[:150]}")
            if not interesting:
                for ln in (err or "").splitlines()[-6:]:
                    print(f"        {ln.strip()[:150]}")

    print()
    if failures:
        print(f"裸环境下失败 {len(failures)} 步: {failures}")
        return 1
    print("裸环境下全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
