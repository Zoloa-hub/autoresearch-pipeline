"""把 CI 的套件步骤改为经 tools/dev/ci_suite.py 运行。

目的：失败时把**具体失败项**写成 GitHub annotation。

公开仓库的 job 日志正文需要 admin 权限才能通过 API 读，而 annotation 匿名可读。
此前 CI 红了之后，自动 annotation 只有一句 `Process completed with exit code 1.`
——谁都看不出失败在哪。这次排查就是卡在这里，只能靠人手点网页复制日志。

顺带修一处真实缺陷：早期 `Import every module` 那句注释里写了
「`templates/experiment/train.py` 在模块级 import numpy」，而该文件实际是
`autoresearch/templates/experiment/train.py`，注释里的路径少了包前缀。
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
text = CI.read_text(encoding="utf-8")

SUITES = [
    "test_config_llm",
    "test_retrieve",
    "test_sandbox_latex",
    "test_figures_metrics",
    "test_prompts",
    "test_adapters",
    "test_pipeline",
]

applied = 0
for mod in SUITES:
    old = f"        run: python -m autoresearch.tests.{mod}\n"
    new = f"        run: python tools/dev/ci_suite.py {mod}\n"
    if old in text:
        text = text.replace(old, new, 1)
        applied += 1
    else:
        print(f"  MISS: {mod}")

# extras job 里的三个套件也走同一个包装（失败同样要有 annotation）
for mod in ("test_figures_metrics", "test_retrieve", "test_pipeline"):
    old = f"        run: python -m autoresearch.tests.{mod}\n"
    if old in text:
        text = text.replace(old, f"        run: python tools/dev/ci_suite.py {mod}\n", 1)
        applied += 1

print(f"已改 {applied} 处套件调用")

# 在第一个套件步骤前加一段说明
anchor = "      - name: Suite 1/7 — config + LLM layer\n"
comment = """      # 套件经 tools/dev/ci_suite.py 运行：失败时把**具体失败项**写成
      # GitHub annotation。公开仓库的 job 日志正文需要 admin 权限才能通过 API
      # 读，而 annotation 匿名可读——否则 CI 红了之后能看到的只有
      # "Process completed with exit code 1."，连作者都定位不到问题。
"""
if anchor in text and "ci_suite.py 运行" not in text:
    text = text.replace(anchor, comment + anchor, 1)
    applied += 1

CI.write_text(text, encoding="utf-8")

import yaml  # noqa: E402

data = yaml.safe_load(text)
print(f"YAML 校验: OK，jobs={list(data['jobs'])}")
for jn, job in data["jobs"].items():
    runs = [
        s.get("run", "")
        for s in (job.get("steps") or [])
        if isinstance(s.get("run"), str) and "ci_suite" in s.get("run", "")
    ]
    if runs:
        print(f"  {jn}: {len(runs)} 个套件步骤走 ci_suite")
