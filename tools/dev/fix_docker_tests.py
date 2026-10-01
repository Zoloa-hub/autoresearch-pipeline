"""把依赖"本机没有 docker"的测试改成确定性，并补上 docker 后端的真实覆盖。

## 问题

GitHub 的 Windows runner **装了 Docker**，而三个测试硬假设"本机没有 docker"：

    check("this machine really has no docker", shutil.which("docker") is None, ...)
    check("available() is False without docker", dk.available() is False)

于是它们只在**恰好没有 docker 的机器**上通过——本地通过、CI 失败。
这与之前 LaTeX 引擎那个问题同构：**测试把环境事实当成了断言**。

## 修法

用 monkeypatch **强制**造出"无 docker"的条件（并且也测"有 docker"那一侧），
再补一段真正验证 `_docker_cmd` 构造的测试——那是刚修好的两处缺陷的所在，
此前零覆盖。
"""

from __future__ import annotations

import pathlib

p = (
    pathlib.Path(__file__).resolve().parents[1]
    / "autoresearch"
    / "tests"
    / "test_sandbox_latex.py"
)
t = p.read_text(encoding="utf-8")

# ---------------------------------------------------- 1) docker 降级（强制无） --
OLD_DEG = '''    with section("make_sandbox: docker degradation on a docker-less machine"):
        check(
            "this machine really has no docker",
            __import__("shutil").which("docker") is None,
            "docker unexpectedly found; the degradation assertion is weaker",
        )
        logs = Recorder()
        sb = make_sandbox(Cfg(backend="docker"), _WORK / "docker_deg", logs)
        check("returns SubprocessSandbox", isinstance(sb, SubprocessSandbox), type(sb).__name__)
        check("not a DockerSandbox", not isinstance(sb, DockerSandbox))
'''
NEW_DEG = '''    with section("make_sandbox: docker 不可用时降级到 subprocess"):
        # 这里**强制**造出"docker 不可用"，而不是断言本机没有 docker。
        #
        # 早期写法是 `check(shutil.which("docker") is None, ...)` —— 把环境事实
        # 当成了断言。本地没有 docker 所以通过，而 GitHub 的 Windows runner
        # **装了 Docker**，于是同一份代码在 CI 上必然失败。这与 LaTeX 引擎那个
        # 问题同构：**测试不该依赖机器恰好缺什么**。
        _real_probe = DockerSandbox._probe
        DockerSandbox._probe = lambda self: False  # type: ignore[assignment]
        try:
            logs = Recorder()
            sb = make_sandbox(Cfg(backend="docker"), _WORK / "docker_deg", logs)
        finally:
            DockerSandbox._probe = _real_probe  # type: ignore[assignment]
        check("returns SubprocessSandbox", isinstance(sb, SubprocessSandbox), type(sb).__name__)
        check("not a DockerSandbox", not isinstance(sb, DockerSandbox))
'''
if OLD_DEG in t:
    t = t.replace(OLD_DEG, NEW_DEG, 1)
    print("  已改：docker 降级测试改为强制模拟")
else:
    print("  MISS: docker 降级段")

# ---------------------------------------------------- 2) None logger 用 subprocess --
OLD_NONE = '''        sb_none = make_sandbox(Cfg(backend="docker"), _WORK / "none_logger", None)
        check("event_logger=None is null-safe", isinstance(sb_none, SubprocessSandbox))
        res = sb_none.run_python(code="print('null-safe')\\n")
        check("run works with event_logger=None", res.ok and "null-safe" in res.stdout, res.tail(200))
'''
NEW_NONE = '''        # 这一段验的是「logger 为 None 时不崩」，不该顺带依赖 docker 的有无。
        # 早期用 backend="docker"：本机没 docker 时拿到 SubprocessSandbox 所以通过，
        # 而 CI 的 Windows runner 有 docker，于是拿到 DockerSandbox、断言失败，
        # 并进一步触发 docker 后端的真实缺陷。改用显式 subprocess 后端。
        sb_none = make_sandbox(Cfg(backend="subprocess"), _WORK / "none_logger", None)
        check("event_logger=None is null-safe", isinstance(sb_none, SubprocessSandbox))
        res = sb_none.run_python(code="print('null-safe')\\n")
        check("run works with event_logger=None", res.ok and "null-safe" in res.stdout, res.tail(200))
'''
if OLD_NONE in t:
    t = t.replace(OLD_NONE, NEW_NONE, 1)
    print("  已改：None logger 段改用 subprocess 后端")
else:
    print("  MISS: None logger 段")

# ---------------------------------------------------- 3) DockerSandbox availability --
OLD_AVAIL = '''    with section("DockerSandbox availability"):
        logs = Recorder()
        dk = DockerSandbox(Cfg(backend="docker"), _WORK / "dk", logs)
        check("available() is False without docker", dk.available() is False
'''
# 原文件中该行末尾没有换行符对齐问题，单独处理
OLD_AVAIL_LINE = '        check("available() is False without docker", dk.available() is False)'
NEW_AVAIL_BLOCK = '''    with section("DockerSandbox availability（强制模拟，不依赖本机）"):
        logs = Recorder()
        _real_probe2 = DockerSandbox._probe
        DockerSandbox._probe = lambda self: False  # type: ignore[assignment]
        try:
            dk = DockerSandbox(Cfg(backend="docker"), _WORK / "dk", logs)
            check("_probe=False 时 available() is False", dk.available() is False)
        finally:
            DockerSandbox._probe = _real_probe2  # type: ignore[assignment]
'''
if OLD_AVAIL_LINE in t:
    t = t.replace(OLD_AVAIL_LINE, NEW_AVAIL_BLOCK.rstrip("\n"), 1)
    print("  已改：DockerSandbox availability 改为强制模拟")
else:
    print("  MISS: available() 那行")

# ---------------------------------------------------- 4) 新增：docker 命令构造覆盖 --
NEW_SECTION = '''

def test_docker_command_construction() -> None:
    """`DockerSandbox._docker_cmd` 与 `_merged_env` —— 此前**零覆盖**的两处缺陷所在。

    容器后端长期没有被执行过（开发机与 WSL 都没有 Docker），于是两处缺陷一直
    潜伏到 CI 的 Windows runner 才暴露：

      1. `DockerSandbox.run` 调用 `self._merged_env(env)`，而 `_merged_env` 只
         定义在 `SubprocessSandbox` 上 -> AttributeError，容器后端一用就崩。
      2. 无条件传 `--pids-limit`，而 Windows 容器不支持该选项 ->
         `docker: invalid option: Windows does not support PidsLimit`。

    这两件事**不需要真的跑容器**就能验证，所以这个测试在所有平台都能跑，
    也就不会再出现"零覆盖"。
    """
    section("DockerSandbox: 命令构造与环境合并（无需真实 docker）")
    logs = Recorder()
    dk = DockerSandbox(Cfg(backend="docker"), _WORK / "dkcmd", logs)

    # 1) _merged_env 必须可用（缺陷 1 的回归）
    check("_merged_env 在基类上可用", hasattr(dk, "_merged_env"))
    merged = dk._merged_env({"MY_VAR": "42", "NONE_VAR": None})
    check("env 被合并进去", merged.get("MY_VAR") == "42", str(merged.get("MY_VAR")))
    check("None 值被规范成空串", merged.get("NONE_VAR") == "", repr(merged.get("NONE_VAR")))
    check(
        "强制 PYTHONIOENCODING=utf-8（实验脚本输出大量非 ASCII）",
        merged.get("PYTHONIOENCODING") == "utf-8",
        repr(merged.get("PYTHONIOENCODING")),
    )
    check("保留宿主环境变量", "PATH" in merged or "Path" in merged)

    # 2) 命令构造
    cmd = dk._docker_cmd(["python", "-c", "print(1)"], allow_network=False)
    check("以 docker run 开头", cmd[:2] == ["docker", "run"], str(cmd[:2]))
    check("带 --rm", "--rm" in cmd)
    check("挂载工作区到 /work", any(str(dk.workdir) in str(c) for c in cmd), str(cmd))
    check("工作目录是 /work", "-w" in cmd and "/work" in cmd)
    check("含镜像名", dk.image in cmd, dk.image)
    check("内层命令在镜像名之后", cmd.index("print(1)") > cmd.index(dk.image))
    check("allow_network=False 时禁网", "--network" in cmd and "none" in cmd)

    cmd_net = dk._docker_cmd(["python", "-c", "print(1)"], allow_network=True)
    check("allow_network=True 时不加 --network", "--network" not in cmd_net)

    # 3) --pids-limit 的平台门控（缺陷 2 的回归）
    if IS_WINDOWS:
        check(
            "Windows 上不传 --pids-limit（Windows 容器不支持该选项）",
            "--pids-limit" not in cmd,
            str(cmd),
        )
    else:
        check(
            "POSIX 上仍传 --pids-limit（Linux 容器支持）",
            "--pids-limit" in cmd,
            str(cmd),
        )
    check("_pids_limit() 有下限保护", dk._pids_limit() >= 16, str(dk._pids_limit()))

    # 4) describe 如实报告后端
    info = dk.describe()
    check("describe 报告 backend", info.get("name") == "docker", str(info.get("name")))
'''
anchor = "\ndef main() -> int:"
if "def test_docker_command_construction(" not in t:
    t = t.replace(anchor, NEW_SECTION + anchor, 1)
    t = t.replace(
        "    test_exec_result_and_factory()\n",
        "    test_exec_result_and_factory()\n    test_docker_command_construction()\n",
        1,
    )
    print("  已新增：test_docker_command_construction 并注册")
else:
    print("  已存在 test_docker_command_construction")

p.write_text(t, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(p), doraise=True)
print("  编译通过")
