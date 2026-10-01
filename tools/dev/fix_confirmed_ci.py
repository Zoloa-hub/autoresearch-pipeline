"""修三个已确诊的 CI 缺陷。

1. **stdio 编码加固**（已加到 `autoresearch/__init__.py`）——
   Windows runner 上 `print()` 中文/emoji 会抛 UnicodeEncodeError，
   进程当场死、输出截断、退出码 1。这是「0.3 秒 + 输出断在 section header」
   的典型形态。已用 cp437 对照实验证明：不加固 rc=1，加固后 rc=0。

2. **lint 的「Import every module」与「零强制依赖」自相矛盾**——
   它要求每个模块都能在裸环境导入，但
   `tools/figures`（模块级 import matplotlib）与
   `templates/experiment/train`（模块级 import numpy）按设计就需要可选依赖。
   修法：把「需要可选依赖的模块」显式列出，并断言它们**失败得正确**
   （报缺失的依赖名），而不是当作错误。

3. **lint 的 heredoc 依赖默认 shell**——显式声明 `shell: bash`。

4. **test_adapters 加进度标记**——CI 上它 0.3 秒就退出且输出截断，
   说明进程在某个测试内死掉。加 per-test 进度打印（带 flush），
   下次运行就能定位到具体哪个测试，而不是只看到一个 section header。
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- CI lint --
CI = ROOT / ".github" / "workflows" / "ci.yml"
ci = CI.read_text(encoding="utf-8")

OLD_LINT = """      - name: Import every module
        if: always()
        run: |
          python - <<'PY'
          import importlib, pathlib, sys
          root = pathlib.Path("autoresearch")
          failed = []
          for path in sorted(root.rglob("*.py")):
              if "__pycache__" in path.parts:
                  continue
              if path.name.startswith("test_") or path.parent.name == "tests":
                  continue
              mod = ".".join(path.with_suffix("").parts)
              try:
                  importlib.import_module(mod)
              except Exception as exc:
                  failed.append(f"{mod}: {type(exc).__name__}: {exc}")
          if failed:
              print("\\n".join(failed))
              sys.exit(1)
          print("all modules import cleanly")
          PY
"""

NEW_LINT = """      # 导入每个模块，验证没有语法/循环导入/名字错误。
      #
      # 注意：**不能**要求"裸环境下每个模块都能导入"——那和"零强制依赖"矛盾。
      # `tools/figures` 在模块级 import matplotlib，`templates/experiment/train`
      # 在模块级 import numpy，它们**按设计**就需要可选依赖。早期这个步骤把它们
      # 的 ImportError 当成失败，于是 lint 在裸环境必然变红。
      #
      # 现在的判定分两类：
      #   * 核心模块：必须导入成功；
      #   * 声明了可选依赖的模块：**要么导入成功，要么报出缺失的那个依赖名**。
      #     后者仍然有价值——它证明失败原因是"缺可选依赖"而不是代码写坏了。
      - name: Import every module
        if: always()
        shell: bash
        run: |
          python - <<'PY'
          import importlib, pathlib, sys

          #: 模块 → 它要求的可选依赖（裸环境下允许 ImportError，但必须是这一个）
          OPTIONAL = {
              "autoresearch.tools.figures": "matplotlib",
              "autoresearch.tools.metrics": None,       # 已改为 pandas 可选、核心可用
              "autoresearch.templates.experiment.train": "numpy",
              "autoresearch.adapters.lorenz_governance": None,  # torch 只在方法内 import
          }

          root = pathlib.Path("autoresearch")
          core_failed: list[str] = []
          optional_ok: list[str] = []
          optional_bad: list[str] = []
          core_count = 0

          for path in sorted(root.rglob("*.py")):
              if "__pycache__" in path.parts:
                  continue
              if path.name.startswith("test_") or path.parent.name == "tests":
                  continue
              mod = ".".join(path.with_suffix("").parts)
              expect_missing = OPTIONAL.get(mod, "__core__")
              try:
                  importlib.import_module(mod)
              except ImportError as exc:
                  text = str(exc)
                  if expect_missing == "__core__":
                      core_failed.append(f"{mod}: ImportError: {text}")
                  elif expect_missing and expect_missing in text:
                      optional_ok.append(f"{mod} (需要 {expect_missing}，如实报告)")
                  else:
                      optional_bad.append(f"{mod}: 期望缺 {expect_missing!r}，实际: {text}")
              except Exception as exc:
                  core_failed.append(f"{mod}: {type(exc).__name__}: {exc}")
              else:
                  if expect_missing == "__core__":
                      core_count += 1

          for line in optional_ok:
              print(f"  optional: {line}")
          problems = core_failed + optional_bad
          if problems:
              print("\\n".join(problems))
              sys.exit(1)
          print(f"{core_count} core modules import cleanly; "
                f"{len(optional_ok)} optional-dependency modules degrade correctly")
          PY
"""

if OLD_LINT in ci:
    ci = ci.replace(OLD_LINT, NEW_LINT, 1)
    print("  CI: Import every module 已改为区分核心/可选依赖")
else:
    print("  CI: 未找到 Import every module 原文")

CI.write_text(ci, encoding="utf-8")

import yaml  # noqa: E402

data = yaml.safe_load(ci)
print(f"  YAML 校验: OK，jobs={list(data['jobs'])}")

# ------------------------------------------------- test_adapters 进度标记 --
TA = ROOT / "autoresearch" / "tests" / "test_adapters.py"
ta = TA.read_text(encoding="utf-8")

if "_progress(" not in ta:
    helper = '''

def _progress(name: str) -> None:
    """在每个测试**开始前**打印一行并 flush。

    CI 上这个套件曾以 0.3 秒退出、输出停在某个 section header 上——说明进程
    在某个测试内部直接死掉（缓冲区随之丢失）。有了这个标记，下次运行至少能
    定位到是哪一个测试，而不是只看到一个孤零零的 section 标题。
    ``flush=True`` 是关键：不 flush 的话崩溃时这行也在缓冲区里。
    """
    print(f"  ... {name}", flush=True)
'''
    anchor = "def _lorenz_source_dir()"
    ta = ta.replace(anchor, helper.strip("\n") + "\n\n\n" + anchor, 1)

    # 给 main() 里每个 try 块插入进度调用
    import re

    def _insert(match: "re.Match[str]") -> str:
        name = match.group(1)
        return f"    _progress({name!r})\n    try:\n        {name}()"

    ta2, n = re.subn(
        r"    try:\n        (test_\w+)\(\)",
        _insert,
        ta,
    )
    ta = ta2
    print(f"  test_adapters: 已插入 {n} 个进度标记")
else:
    print("  test_adapters: 已有进度标记")

TA.write_text(ta, encoding="utf-8")
print("完成")
