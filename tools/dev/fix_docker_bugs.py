"""修 DockerSandbox 的两个真实缺陷 —— 它们此前**零覆盖**，所以在本地永远看不到。

CI 的 GitHub Windows runner 装了 Docker，于是这条从未被执行过的代码路径第一次运行，
立刻暴露：

    AttributeError("'DockerSandbox' object has no attribute '_merged_env'")
    docker: invalid option: Windows does not support PidsLimit

## 缺陷 1：`_merged_env` 定义在错误的类上

它定义在 `SubprocessSandbox`（第 628 行），而 `DockerSandbox.run` 调用
`self._merged_env(env)` —— 容器后端一用就 AttributeError。
两个后端都需要"把 os.environ 与调用方 env 合并、并强制 PYTHONIOENCODING=utf-8"，
所以它属于基类 `Sandbox`。

## 缺陷 2：无条件传 `--pids-limit`

Windows 容器不支持该选项，于是 `docker run` 直接失败。
pid 上限对 Linux 容器是有意义的加固，但不能假设容器一定是 Linux。

修法：POSIX 宿主才传；Windows 宿主跳过并**如实记录**这个能力缺口
（而不是静默少一层保护）。
"""

from __future__ import annotations

import pathlib
import re

p = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "tools" / "sandbox.py"
t = p.read_text(encoding="utf-8")

# ---------------------------------------------------------------- 缺陷 1 --
OLD_MERGED = '''    # -- env --------------------------------------------------------------- #
    def _merged_env(self, env: dict | None) -> dict:
        merged = dict(os.environ)
        merged["PYTHONUNBUFFERED"] = "1"
        merged["PYTHONDONTWRITEBYTECODE"] = "1"
        merged["PYTHONIOENCODING"] = "utf-8"
        if env:
            for k, v in env.items():
                merged[str(k)] = "" if v is None else str(v)
        return merged

'''
NEW_MERGED = '''    # -- env --------------------------------------------------------------- #
    # 注意：`_merged_env` 属于**基类**（见 Sandbox._merged_env）。
    # 它曾经只定义在 SubprocessSandbox 上，而 DockerSandbox.run 也调用它 ——
    # 于是容器后端一用就 AttributeError。这条路径直到 CI runner（装了 Docker）
    # 才第一次被执行，本地（无 Docker）永远看不到。

'''
if OLD_MERGED in t:
    t = t.replace(OLD_MERGED, NEW_MERGED, 1)
    print("  缺陷1: 已从 SubprocessSandbox 移除 _merged_env 定义")
else:
    print("  MISS 缺陷1: 未找到 SubprocessSandbox._merged_env")

# 插到基类 Sandbox 的 tmp_dir 之后
BASE_ANCHOR = '''    def tmp_dir(self) -> Path:
        d = self.workdir / TMP_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        return d
'''
BASE_ADD = BASE_ANCHOR + '''
    def _merged_env(self, env: dict | None) -> dict:
        """合并 ``os.environ`` 与调用方传入的 env，并强制 UTF-8 I/O。

        放在基类而不是某个后端里：两个后端都需要它（子进程直接用它启动，
        容器用它把环境传进 ``docker run``）。早期它只定义在
        :class:`SubprocessSandbox` 上，而 :meth:`DockerSandbox.run` 也调用
        ``self._merged_env(...)`` —— 容器后端因此一用就 ``AttributeError``。
        这条路径长期零覆盖（开发机与 WSL 都没有 Docker），直到 CI 的 runner
        装了 Docker 才第一次跑到。

        ``PYTHONIOENCODING=utf-8`` 是必需的：实验脚本大量输出非 ASCII
        （中文日志、指标名），在非 UTF-8 控制台下会直接崩。
        """
        merged = dict(os.environ)
        merged["PYTHONUNBUFFERED"] = "1"
        merged["PYTHONDONTWRITEBYTECODE"] = "1"
        merged["PYTHONIOENCODING"] = "utf-8"
        if env:
            for k, v in env.items():
                merged[str(k)] = "" if v is None else str(v)
        return merged
'''
if BASE_ANCHOR in t:
    t = t.replace(BASE_ANCHOR, BASE_ADD, 1)
    print("  缺陷1: 已在基类 Sandbox 定义 _merged_env")
else:
    print("  MISS 缺陷1: 未找到基类 tmp_dir")

# ---------------------------------------------------------------- 缺陷 2 --
OLD_PIDS = '''        cpus = float(getattr(self.cfg, "cpus", 0.0) or 0.0)
        if cpus > 0:
            cmd += ["--cpus", f"{cpus:g}"]
        cmd += ["--pids-limit", "512", self.image]
        cmd += [str(c) for c in inner]
        return cmd
'''
NEW_PIDS = '''        cpus = float(getattr(self.cfg, "cpus", 0.0) or 0.0)
        if cpus > 0:
            cmd += ["--cpus", f"{cpus:g}"]
        # `--pids-limit` 只有 **Linux 容器**支持。Windows 宿主上（Windows 容器）
        # 传它会直接失败：
        #     docker: invalid option: Windows does not support PidsLimit
        # 这在 CI 的 Windows runner 上真实发生过——那条路径此前零覆盖。
        # 无法廉价地判断"容器是 Linux 还是 Windows"，所以按**宿主**判断：
        # POSIX 宿主默认传，Windows 宿主跳过。
        # 跳过时如实记录这个能力缺口，而不是静默少一层保护。
        if not IS_WINDOWS:
            cmd += ["--pids-limit", str(self._pids_limit())]
        cmd += [self.image]
        cmd += [str(c) for c in inner]
        return cmd

    def _pids_limit(self) -> int:
        """容器内的进程数上限（Linux 容器专用），可用 ``docker_pids_limit`` 覆盖。"""
        try:
            value = int(getattr(self.cfg, "docker_pids_limit", 512) or 512)
        except Exception:  # noqa: BLE001
            return 512
        return max(16, value)
'''
if OLD_PIDS in t:
    t = t.replace(OLD_PIDS, NEW_PIDS, 1)
    print("  缺陷2: --pids-limit 改为仅 POSIX 宿主传")
else:
    print("  MISS 缺陷2: 未找到 --pids-limit 那段")

p.write_text(t, encoding="utf-8")

# 校验
import py_compile  # noqa: E402

py_compile.compile(str(p), doraise=True)
print("  编译通过")

# 结构自检
body = p.read_text(encoding="utf-8")
checks = {
    "基类有 _merged_env": bool(
        re.search(r"class Sandbox:.*?def _merged_env", body, re.S)
    ),
    "SubprocessSandbox 不再重复定义": body.count("def _merged_env") == 1,
    "pids-limit 受 IS_WINDOWS 保护": 'if not IS_WINDOWS:\n            cmd += ["--pids-limit"' in body,
}
for label, ok in checks.items():
    print(f"  {'OK  ' if ok else 'FAIL'} {label}")
