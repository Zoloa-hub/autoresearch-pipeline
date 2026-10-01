"""修 ci_suite.py 的 stdio 编码崩溃 —— 并加自动检查防止第三次复发。

## 根因

`ci_suite.py` 把子进程的输出（含大量中文）用 `print()` 写到自己的 stdout。
在 Windows 的 cp1252 控制台下这会抛 `UnicodeEncodeError`，脚本当场死掉：

    UnicodeEncodeError: 'charmap' codec can't encode characters in position 1146-1148
    -> 0 条 annotation

于是 Windows 三个版本上**很多套件都"失败"**，而真实原因是包装器自己崩了，
不是套件有问题。这解释了为什么 run #10 里 Windows 的 Suite 1（上一轮还是通过的）
也挂了。

## 这是同一个坑的第二次

第一次是包入口 `autoresearch/__init__.py` 的日志/报告中文化。当时我加了
`_harden_stdio()`，但 `tools/dev/ci_suite.py` **在包外面**，没享受到。

所以这次除了修，还要**加自动检查**：让 lint job 在强制窄编码下跑一次
ci_suite.py，崩了就红。靠人记住"新入口要加固 stdio"是不可靠的。
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ------------------------------------------------------------ 1) 修 ci_suite --
CS = ROOT / "tools" / "dev" / "ci_suite.py"
t = CS.read_text(encoding="utf-8")

HARDEN = '''
def _harden_stdio() -> None:
    """让本脚本的标准输出**永不因非 ASCII 而崩**。

    这个包装器会把套件的完整输出（含大量中文标签与 SKIP 说明）打印出来。
    在非 UTF-8 控制台下（Windows 的 cp1252/cp936、某些 CI 容器的 POSIX locale），
    `print()` 一个中文串会抛 `UnicodeEncodeError`，**脚本当场死掉**——于是
    一个本来只是"套件失败"的情况，变成"包装器崩溃、连 annotation 都没输出"，
    而 CI 上看到的是多个套件同时失败，极难归因到真正的源头。

    这是**同一个坑的第二次**：第一次在 autoresearch/__init__.py 的包入口
    （那里的日志与报告大量使用中文）。区别是这个脚本在包外面，
    享受不到包入口的加固，所以必须自带一份。

    `backslashreplace` 而不是 `replace`：保留可读的 ``\\u4e2d``，
    信息量更大，也不会让编码问题掩盖真正的失败。
    """
    import sys as _sys

    for stream, errors in ((_sys.stdout, "backslashreplace"), (_sys.stderr, "replace")):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors=errors)
        except (ValueError, OSError, AttributeError):
            try:
                stream.reconfigure(errors=errors)
            except Exception:  # noqa: BLE001 - 加固本身绝不能成为失败源
                pass


_harden_stdio()

'''

if "_harden_stdio" not in t:
    # 插到 import 之后、常量之前
    anchor = "#: GitHub 每个 step 最多接受 10 条 error annotation"
    t = t.replace(anchor, HARDEN.strip("\\n") + "\n\n" + anchor, 1)
    CS.write_text(t, encoding="utf-8")
    print("  ci_suite.py: 已加 _harden_stdio")
else:
    print("  ci_suite.py: 已有 _harden_stdio")

# ------------------------------------------------------------ 2) 加自动检查 --
CI = ROOT / ".github" / "workflows" / "ci.yml"
ci = CI.read_text(encoding="utf-8")

CHECK_STEP = '''
      # 自动检查：任何"把非 ASCII 打到 stdout"的入口都必须在窄编码下存活。
      #
      # 这个坑踩过两次：一次在包入口（autoresearch/__init__.py 的日志中文化），
      # 一次在 tools/dev/ci_suite.py（包装器打印套件输出）。第二次让 Windows 上
      # 多个套件同时"失败"，而真凶是包装器自己崩了、连 annotation 都没输出——
      # 极难归因。靠人记住"新入口要加固 stdio"不可靠，所以固化成检查。
      - name: Entry points survive a non-UTF-8 console
        if: always()
        shell: bash
        run: |
          python - <<'PY'
          import pathlib, subprocess, sys

          #: 强制窄编码跑一遍每个入口，确认不会因非 ASCII 输出而崩
          CASES = [
              ("tools/dev/ci_suite.py", ["test_prompts"]),
          ]
          NARROW = (
              "import sys;"
              "sys.stdout.reconfigure(encoding='cp437', errors='strict');"
              "sys.stderr.reconfigure(encoding='cp437', errors='strict');"
          )
          failed = []
          for script, argv in CASES:
              if not pathlib.Path(script).is_file():
                  continue
              code = (
                  NARROW
                  + f"sys.argv={[script] + argv!r};"
                  + f"exec(compile(open({script!r}, encoding='utf-8').read(), {script!r}, 'exec'))"
              )
              proc = subprocess.run(
                  [sys.executable, "-c", code],
                  capture_output=True, text=True, encoding="utf-8", errors="replace",
              )
              if "UnicodeEncodeError" in (proc.stderr or ""):
                  failed.append(f"{script}: 在窄编码下抛 UnicodeEncodeError")
              elif proc.returncode != 0:
                  failed.append(f"{script}: rc={proc.returncode}（{proc.stderr[-300:]}）")
              else:
                  print(f"  ok {script} 在 cp437 下存活")
          if failed:
              for item in failed:
                  print(f"::error::{item}")
              sys.exit(1)
          print("all entry points survive a non-UTF-8 console")
          PY
'''

# 插到 lint job 的 "Byte-compile every module" 之前
anchor = "      - name: Byte-compile every module\n"
if anchor in ci and "Entry points survive a non-UTF-8 console" not in ci:
    ci = ci.replace(anchor, CHECK_STEP.strip("\n") + "\n\n" + anchor, 1)
    CI.write_text(ci, encoding="utf-8")
    print("  CI: 已加「窄编码存活」检查步骤")
else:
    print("  CI: 已有该检查或找不到锚点")

# ------------------------------------------------------------ 3) 校验 --
import py_compile  # noqa: E402

py_compile.compile(str(CS), doraise=True)
print("  ci_suite.py 编译通过")

import yaml  # noqa: E402

data = yaml.safe_load(CI.read_text(encoding="utf-8"))
lint_steps = [s.get("name") for s in data["jobs"]["lint"]["steps"]]
print(f"  lint steps: {lint_steps}")
