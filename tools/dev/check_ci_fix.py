"""校验 CI 修复：YAML 结构、if:always() 覆盖、无平台相关 shell 语法。"""

from __future__ import annotations

import pathlib
import re

def _repo_root() -> pathlib.Path:
    """向上找含 ``autoresearch/`` 的目录（脚本可能在 scratch/ 或 tools/dev/ 下）。"""
    here = pathlib.Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "autoresearch" / "__init__.py").is_file():
            return cand
    return pathlib.Path.cwd()


ROOT = _repo_root()
text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

import yaml  # noqa: E402

data = yaml.safe_load(text)
print("YAML 解析: OK")
print(f"jobs: {list(data['jobs'])}")
print()
for job_name, job in data["jobs"].items():
    steps = job.get("steps") or []
    always = sum(1 for s in steps if str(s.get("if", "")) == "always()")
    cont = sum(1 for s in steps if s.get("continue-on-error"))
    print(f"  {job_name:<12} {len(steps):>2} steps   if:always()={always}   continue-on-error={cont}")

print()
print("冒烟步骤:")
for s in data["jobs"]["test"]["steps"]:
    if str(s.get("name", "")).startswith("Smoke"):
        print(f"  run = {s.get('run')!r}")
        print(f"  continue-on-error = {s.get('continue-on-error')}")

print()
print("危险语法检查:")
checks = {
    "GITHUB_ENV（Windows 上 bash 专有）": "GITHUB_ENV" in text,
    "$GITHUB_OUTPUT": "$GITHUB_OUTPUT" in text,
    "heredoc <<'PY'（extras job 用，bash-only）": "<<'PY'" in text,
    "shell: bash 显式声明": "shell: bash" in text,
}
for label, hit in checks.items():
    print(f"  {'命中' if hit else '未出现'}  {label}")

# 关键：extras job 用了 heredoc，必须显式声明 bash，否则默认 shell 可能是别的
extras = data["jobs"]["extras"]
heredoc_steps = [
    s for s in extras.get("steps") or [] if "<<'PY'" in str(s.get("run", ""))
]
print()
print(f"extras job 里含 heredoc 的步骤: {len(heredoc_steps)}")
for s in heredoc_steps:
    print(f"  - {s.get('name')!r}  shell={s.get('shell')!r}")

# extras 跑在 ubuntu-latest 上，默认就是 bash，所以 heredoc 没问题；
# 但显式写出 shell 更稳（runner 的默认 shell 可能变）
print()
print(f"extras runs-on = {extras.get('runs-on')}（ubuntu 默认 bash，heredoc 可用）")
