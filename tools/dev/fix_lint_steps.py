"""修两个 lint 步骤的设计缺陷。

1. `No committed secrets` 有误报，且正则/排除都写错了
   - 正则 `DEEPSEEK_API_KEY=[^[:space:]]` 只要后面有非空白字符就命中，
     于是文档里的占位符 `DEEPSEEK_API_KEY=sk-...` 必然误报；它**没有长度约束**，
     等于没有"像真密钥"的判据。
   - 排除用 `:!*.md`，而 pathspec 通配**不跨目录**（`git ls-files '*.md'` 只列
     根目录的），所以 `autoresearch/README.md` 从未被排除。

2. `Assert optional-dependency skips are reported` 对环境的假设不成立
   - 它断言"裸环境下必须报告 SKIPPED"，却被放在一个条件混合的位置：
     Linux 上恰好成立、macOS/Windows 上不成立。
   - 跳过上报是**裸环境专有**的性质，应该由专门确保无依赖的那一步来断言。

修法：密钥扫描区分"真像密钥"（要求足够的字符数与字符集、排除占位符形态）；
跳过上报在**显式清空环境**的子进程里验证，而不是依赖 job 的隐含状态。
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
text = CI.read_text(encoding="utf-8")

# ---------------------------------------------------------------- 1) 密钥扫描 --
old_secrets = """      - name: No committed secrets
        run: |
          # .env 必须被 gitignore 挡住；仓库里出现 sk- 开头的密钥即 CI 失败。
          if git ls-files --error-unmatch .env >/dev/null 2>&1; then
            echo "::error::.env is tracked by git — secrets must never be committed"
            exit 1
          fi
          if git grep -nIE '(sk-[A-Za-z0-9]{16,}|DEEPSEEK_API_KEY=[^[:space:]]|OPENAI_API_KEY=sk-)' -- . ':!*.md' ':!*.example' ':!.env.example'; then
            echo "::error::possible committed API key"
            exit 1
          fi
          echo "no committed secrets found"
"""

new_secrets = """      - name: No committed secrets
        shell: bash
        run: |
          # .env 必须被 gitignore 挡住。
          if git ls-files --error-unmatch .env >/dev/null 2>&1; then
            echo "::error::.env is tracked by git — secrets must never be committed"
            exit 1
          fi

          # 扫"像真密钥"的字符串。
          #
          # 两条经验都来自真实误报：
          #  1. **必须有长度约束**。早期用 `DEEPSEEK_API_KEY=[^[:space:]]`，
          #     于是文档里的占位符 `DEEPSEEK_API_KEY=sk-...` 直接命中——
          #     等于没有判据，只会在自己的文档上反复变红。
          #  2. **排除要能跨目录**。`:!*.md` 这类 pathspec 通配不跨目录
          #     （`git ls-files '*.md'` 只列根目录），所以
          #     `autoresearch/README.md` 从未被排除过。
          #
          # 真密钥形态：`sk-` + 至少 20 个字母数字（OpenAI/DeepSeek 都远长于此）。
          # 占位符形态（`sk-...`、`sk-xxx`、`sk-在此填入你的密钥`）都不会命中。
          PATTERN='(^|[^A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}'
          if git grep -nIE "$PATTERN" -- . \\
               ':(exclude,glob)**/*.md' \\
               ':(exclude,glob)**/*.example' \\
               ':(exclude,glob)**/*.rst' \\
               ':(exclude).env.example'; then
            echo "::error::possible committed API key (a real-looking sk- token)"
            exit 1
          fi
          echo "no committed secrets found"
"""

if old_secrets in text:
    text = text.replace(old_secrets, new_secrets, 1)
    print("  已替换 No committed secrets")
else:
    print("  MISS: No committed secrets 原文")

# ------------------------------------------------- 2) 跳过上报改为显式无依赖 --
old_skip = """      - name: Assert optional-dependency skips are reported
        if: always()
        run: |
          python - <<'PY'
"""
i = text.find(old_skip)
if i < 0:
    print("  MISS: Assert optional-dependency skips 原文")
else:
    # 找到这个 step 的结尾（下一个 "      - name:" 或 job 结束）
    j = text.find("\n      - name:", i + len(old_skip))
    if j < 0:
        j = text.find("\n  # ---", i)
    old_block = text[i:j]
    new_block = """      - name: Assert optional-dependency skips are reported
        if: always()
        shell: bash
        # 这一步必须**自己保证**没有第三方依赖，而不是依赖 job 的隐含状态。
        # 早期版本在混合条件下运行，于是 Linux 上恰好成立、macOS/Windows 上
        # 不成立——"跳过上报"本来就是裸环境专有的性质。
        # 用 sitecustomize 把 site-packages 从 sys.path 里摘掉，
        # 并且用 PYTHONPATH 注入，使**子进程**（沙箱里的实验脚本）同样被隔离。
        run: |
          GUARD="$PWD/.autoresearch/_bare_guard"
          mkdir -p "$GUARD"
          cat > "$GUARD/sitecustomize.py" <<'PY'
          import sys
          sys.path = [p for p in sys.path
                      if "site-packages" not in p and "dist-packages" not in p]
          PY
          OUT=$(PYTHONPATH="$GUARD:$PWD" python -c "import runpy,sys; sys.argv=['autoresearch.tests.test_figures_metrics']; runpy.run_module('autoresearch.tests.test_figures_metrics', run_name='__main__')" 2>&1)
          echo "$OUT" | tail -20
          echo "$OUT" | grep -q "SKIPPED" || {
            echo "::error::裸环境下应当明确报告跳过，而不是静默通过"
            exit 1
          }
          echo "$OUT" | grep -qE "PASSED [0-9]+ checks" || {
            echo "::error::测试仍应实际执行并给出计数"
            exit 1
          }
          echo "skip reporting ok"
"""
    text = text[:i] + new_block + text[j:]
    print("  已替换 Assert optional-dependency skips")

CI.write_text(text, encoding="utf-8")

import yaml  # noqa: E402

data = yaml.safe_load(text)
print(f"  YAML 校验: OK，jobs={list(data['jobs'])}")
for jn, job in data["jobs"].items():
    steps = job.get("steps") or []
    print(f"    {jn:<12} {len(steps):>2} steps")
