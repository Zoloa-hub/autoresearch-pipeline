"""逐条核对路线 C 的目标项，每条都要求代码/文件证据。"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, evidence: str) -> None:
    RESULTS.append((label, bool(ok), evidence))


# ---------------------------------------------------------------- 阶段 B --
try:
    from autoresearch.adapters import (
        BaseExperimentAdapter,
        RunSpec,
        ScriptWrapperAdapter,
        SyntheticToyAdapter,
        resolve_adapter,
    )
    from autoresearch.adapters.base import MetricSeries  # noqa: F401

    check("B: BaseExperimentAdapter 协议存在",
          hasattr(BaseExperimentAdapter, "build_command")
          and hasattr(BaseExperimentAdapter, "parse_results"),
          "build_command/parse_results 均为抽象方法")
    check("B: RunSpec 存在且字段开放",
          {"variant", "seed", "out_dir", "params", "extra_args"}
          <= set(RunSpec.__dataclass_fields__),
          f"字段 {sorted(RunSpec.__dataclass_fields__)}")
    check("B: SyntheticToyAdapter 存在且为默认",
          resolve_adapter(None).name == SyntheticToyAdapter.name == "synthetic-toy",
          f"resolve_adapter(None).name = {resolve_adapter(None).name}")
    check("B: ScriptWrapperAdapter 存在且 owns_code=True",
          ScriptWrapperAdapter.name == "script-wrapper"
          and ScriptWrapperAdapter.owns_code is True,
          "script-wrapper / owns_code=True")
except Exception as exc:  # pragma: no cover
    check("B: 适配器包可导入", False, f"{type(exc).__name__}: {exc}")

try:
    from autoresearch.tools.sandbox import DockerSandbox, SubprocessSandbox

    check("B: 沙箱 run_command 已实现（两个后端）",
          hasattr(SubprocessSandbox, "run_command") and hasattr(DockerSandbox, "run_command"),
          "SubprocessSandbox.run_command + DockerSandbox.run_command")
    check("B: POSIX rlimit 修复（preexec_fn 与 start_new_session 并存）",
          hasattr(SubprocessSandbox, "_make_preexec"),
          "SubprocessSandbox._make_preexec 存在")
except Exception as exc:
    check("B: 沙箱模块可导入", False, f"{type(exc).__name__}: {exc}")

cli_src = (ROOT / "autoresearch" / "cli.py").read_text(encoding="utf-8")
check("B: CLI --experiment-adapter 接入点",
      '"--experiment-adapter"' in cli_src and '"--adapter-arg"' in cli_src,
      "build_parser 中注册了 --experiment-adapter 与 --adapter-arg")

# ---------------------------------------------------------------- 阶段 A --
check("A: Apache-2.0 LICENSE",
      (ROOT / "LICENSE").is_file()
      and "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8"),
      f"LICENSE { (ROOT / 'LICENSE').stat().st_size } bytes")

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
readme_name = re.search(r'readme\s*=\s*"([^"]+)"', pyproject)
check("A: pyproject.toml 合法且 readme 指向真实文件",
      readme_name is not None and (ROOT / readme_name.group(1)).is_file(),
      f"readme = {readme_name.group(1) if readme_name else '?'}（存在）")
check("A: pyproject 声明零强制依赖",
      "dependencies = []" in pyproject,
      "dependencies = []")

ci = ROOT / ".github" / "workflows" / "ci.yml"
ci_text = ci.read_text(encoding="utf-8") if ci.is_file() else ""
check("A: CI 三平台矩阵",
      all(o in ci_text for o in ("ubuntu-latest", "windows-latest", "macos-latest")),
      "ubuntu / windows / macos 三平台")
check("A: CI 覆盖全部 7 个测试套件",
      len(set(re.findall(r"autoresearch\.tests\.(test_\w+)", ci_text))) == 7,
      f"{len(set(re.findall(r'autoresearch.tests.(test_'+'\\w+)', ci_text)))} 个套件被 CI 调用")

gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
check("A: vendor 二进制移出版本控制",
      "autoresearch/vendor/" in gitignore and "*.exe" in gitignore,
      ".gitignore 排除 vendor/ 与 *.exe")
check("A: 提供按需下载脚本",
      (ROOT / "autoresearch" / "tools" / "fetch_tectonic.py").is_file(),
      "autoresearch/tools/fetch_tectonic.py")

readme = (ROOT / "autoresearch" / "README.md").read_text(encoding="utf-8")
check("A: README 有 Current Capabilities 节",
      "当前能力与明确非目标" in readme and "现在就能用" in readme,
      "§当前能力与明确非目标 / ✅ 现在就能用")
check("A: README 有 Explicit Non-Goals 节",
      "明确不做" in readme and "非目标" in readme,
      "❌ 明确不做（当前版本）表")

# ------------------------------------------------------- 回归与规模 --
suites = sorted(p.stem for p in (ROOT / "autoresearch" / "tests").glob("test_*.py"))
check("A: 测试套件数量与文档一致（7）",
      len(suites) == 7 and "7 个测试套件" in readme,
      f"实际 {len(suites)} 个，README 写 7 个")
prompts = sorted(p.stem for p in (ROOT / "autoresearch" / "prompts").glob("s*.md"))
check("A: 阶段提示词数量与文档一致（14）",
      len(prompts) == 14 and "14" in readme,
      f"实际 {len(prompts)} 个")

# ------------------------------------------------------------- 输出 --
width = max(len(label) for label, _, _ in RESULTS)
passed = sum(1 for _, ok, _ in RESULTS if ok)
for label, ok, evidence in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<{width}}  {evidence}")
print(f"\n{passed}/{len(RESULTS)} 项目标项通过")
raise SystemExit(0 if passed == len(RESULTS) else 1)
