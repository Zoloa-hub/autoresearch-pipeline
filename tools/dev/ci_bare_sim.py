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
    """在屏蔽 site-packages 的子进程里运行 argv。"""
    code = BARE_BOOTSTRAP + (
        "import runpy,sys;"
        f"sys.argv={argv!r};"
        "runpy.run_module(sys.argv[0].split('.')[0], run_name='__main__')"
        if False
        else ""
    )
    # 直接用 -c 包一层：先改 sys.path，再 exec 目标模块
    # argv 形如 [python, "-m", "module", *args]；runpy 要的是模块名而不是 "-m"
    module = argv[2] if len(argv) > 2 and argv[1] == "-m" else None
    if module is None:
        return 1, "", f"unsupported argv: {argv}"
    launcher = (
        BARE_BOOTSTRAP
        + "import runpy,sys;"
        + f"sys.argv={[module] + list(argv[3:])!r};"
        + f"runpy.run_module({module!r}, run_name='__main__')"
    )
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONUTF8"] = "0"  # 模拟 Windows runner 的非 UTF-8 默认
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
