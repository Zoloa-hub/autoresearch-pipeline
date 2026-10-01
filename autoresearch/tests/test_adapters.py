"""实验后端适配器测试（离线，无网络，无需 API Key）。

阶段 B 重构把「实验怎么跑」从 ``s4_experiment`` 里抽成可插拔适配器。这个重构
最大的风险不是功能缺失，而是**契约漂移**：适配器返回的指标结构、变体命名、
环境预检失败的处理方式一旦和主干不一致，错误会一路漂到画图或写作阶段才炸，
那时归因成本极高。所以这里的断言集中在**边界契约**上：

1. ``RunSpec`` 承载开放超参，不再是写死的 variant/seed/out_dir 三元组；
2. ``parse_results`` 的返回被强制规范成 ``dict[str, list[float]]``，畸形结构立刻报错；
3. 适配器解析支持四种写法（内置名 / .py 路径 / module:Class / entry point），
   未知名字的报错要列出可用项；
4. 沙箱 ``run_command`` 接受 argv，明确拒绝字符串与 shell 元字符；
5. ``s4`` 的变体推导不把消融**取值**当变体名，且会过滤适配器不支持的变体；
6. 对照逻辑在缺 baseline / 无共同指标时必须给出**可解释的失败**，而不是静默返回空。
"""

from __future__ import annotations

import json
import re
import shutil
import sys

from autoresearch.adapters import RunSpec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_CHECKS = 0
_FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    global _CHECKS
    _CHECKS += 1
    if condition:
        return True
    message = f"FAIL: {label}" + (f" — {detail}" if detail else "")
    _FAILURES.append(message)
    print("  " + message)
    return False


def eq(actual, expected, label: str) -> bool:
    return check(actual == expected, label, f"expected {expected!r}, got {actual!r}")


def _progress(name: str) -> None:
    """在每个测试**开始前**打印一行并 flush。

    CI 上这个套件曾以 0.3 秒退出、输出停在某个 section header 上——说明进程
    在某个测试内部直接死掉（缓冲区随之丢失）。有了这个标记，下次运行至少能
    定位到是哪一个测试，而不是只看到一个孤零零的 section 标题。
    ``flush=True`` 是关键：不 flush 的话崩溃时这行也在缓冲区里。
    """
    print(f"  ... {name}", flush=True)


def _lorenz_source_dir() -> Path | None:
    """真实实验脚本目录（供门控的真实执行测试使用）。

    优先读 ``AUTORESEARCH_LORENZ_SOURCE``；否则试一个已知的本地位置。
    两者都不存在时返回 ``None``，让调用方**跳过**而不是失败——
    发布仓库里本来就不该包含用户那份实验脚本。
    """
    import os

    env = os.environ.get("AUTORESEARCH_LORENZ_SOURCE")
    candidates: list[Path] = [Path(env)] if env else []
    candidates.append(Path(r"C:\Users\user\Desktop\Phd-foundations\sakanaai ctm"))
    for cand in candidates:
        if (cand / "gov_naive_update.py").is_file():
            return cand
    return None


def _lorenz_fixture_dir() -> Path:
    """真实实验适配器的指标 fixture 目录。

    用 fixture 而不是真跑实验：套件的硬性不变量是「全绿不需要网络/API Key」，
    而真跑那份脚本还要 PyTorch 与用户目录。解析逻辑本身完全可以用
    一段**真实的**脚本输出（从真跑中截取）来验证。
    """
    return (
        _workspace_root()
        / "autoresearch"
        / "tests"
        / "fixtures"
        / "lorenz_governance"
    )


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scratch(name: str) -> Path:
    root = _workspace_root() / ".autoresearch" / "test_tmp" / f"adapters_{name}"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# RunSpec 与指标契约
# --------------------------------------------------------------------------- #


def test_runspec() -> None:
    section("RunSpec")
    from autoresearch.adapters import RunSpec

    spec = RunSpec(variant="method", seed=3, out_dir="runs/method/seed_3",
                   params={"lr": 0.01, "epochs": 5}, extra_args=["--amp"])
    eq(spec.variant, "method", "variant")
    eq(spec.seed, 3, "seed")
    eq(spec.params["lr"], 0.01, "params 承载任意超参")
    eq(spec.extra_args, ["--amp"], "extra_args 是逃生舱")
    check("slug" in dir(spec), "有 slug()")
    slug = spec.slug()
    check(all(ch.isalnum() or ch in "-_." for ch in slug), "slug 是文件系统安全字符", slug)
    payload = spec.to_dict()
    check(json.dumps(payload), "to_dict 可 JSON 序列化")

    # 默认值：params/extra_args 互不共享（可变默认值陷阱）
    a, b = RunSpec("x", 0, "d"), RunSpec("y", 1, "d")
    a.params["k"] = 1
    eq(b.params, {}, "params 不共享默认实例")


def test_metric_series_contract() -> None:
    section("指标契约")
    from autoresearch.adapters import AdapterError, coerce_metric_series, higher_is_better

    # 合法形态
    eq(coerce_metric_series({"acc": [0.9, 0.91]}), {"acc": [0.9, 0.91]}, "标准形态原样通过")
    eq(coerce_metric_series({"acc": 0.9}), {"acc": [0.9]}, "标量被包成单元素序列")
    eq(coerce_metric_series({"acc": (1, 2)}), {"acc": [1.0, 2.0]}, "元组被接受")
    eq(coerce_metric_series({}), {}, "空 dict 合法")
    eq(coerce_metric_series(None), {}, "None 视作无指标")

    # 非数值被跳过而不是让整批失败
    eq(coerce_metric_series({"acc": [1, "x", 2]}), {"acc": [1.0, 2.0]},
       "序列里的非数值被跳过")

    # numpy 风格的可迭代对象
    class _Arr:
        def __iter__(self):
            return iter([0.1, 0.2])

    eq(coerce_metric_series({"loss": _Arr()}), {"loss": [0.1, 0.2]}, "可迭代对象被接受")

    # 违反契约必须**立刻**报错，而不是静默返回空
    for bad, label in ((5, "int"), ("str", "str"), (["a"], "list")):
        try:
            coerce_metric_series(bad)
            check(False, f"非 mapping（{label}）应报 AdapterError")
        except AdapterError:
            check(True, f"非 mapping（{label}）报 AdapterError")

    # 指标方向启发式
    check(higher_is_better("accuracy"), "accuracy 越大越好")
    check(not higher_is_better("val_loss"), "val_loss 越小越好")
    check(not higher_is_better("test_rmse"), "rmse 越小越好")
    check(higher_is_better("f1"), "f1 越大越好")
    check(higher_is_better("自定义指标"), "未知指标默认越大越好")


def test_standard_metrics_reader() -> None:
    section("标准指标读取")
    from autoresearch.adapters import read_standard_metrics

    root = _scratch("metrics")
    (root / "metrics.csv").write_text(
        "epoch,loss,accuracy\n1,0.5,0.9\n2,0.4,0.92\n", encoding="utf-8"
    )
    series = read_standard_metrics(root)
    eq(sorted(series), ["accuracy", "epoch", "loss"], "读到 CSV 的三列")
    eq(series["accuracy"], [0.9, 0.92], "accuracy 序列正确")

    # jsonl 也要能读
    root2 = _scratch("metrics_jsonl")
    (root2 / "metrics.jsonl").write_text(
        '{"acc": 0.8}\n{"acc": 0.85}\n', encoding="utf-8"
    )
    eq(read_standard_metrics(root2)["acc"], [0.8, 0.85], "读到 JSONL")

    # 空目录不报错
    eq(read_standard_metrics(_scratch("metrics_empty")), {}, "空目录返回空 dict")


# --------------------------------------------------------------------------- #
# 适配器解析
# --------------------------------------------------------------------------- #


def test_resolve_and_inspect() -> None:
    section("适配器解析")
    from autoresearch.adapters import (
        AdapterError,
        BaseExperimentAdapter,
        RunSpec,
        builtin_adapters,
        resolve_adapter,
    )

    registry = builtin_adapters()
    check("synthetic-toy" in registry and "script-wrapper" in registry,
          "两个内置适配器都已注册", str(sorted(registry)))

    for spec, expected in ((None, "synthetic-toy"), ("", "synthetic-toy"),
                           ("default", "synthetic-toy"), ("synthetic", "synthetic-toy"),
                           ("synthetic-toy", "synthetic-toy")):
        eq(resolve_adapter(spec).name, expected, f"resolve_adapter({spec!r})")

    # 未知名字必须列出可用项
    try:
        resolve_adapter("no-such-adapter")
        check(False, "未知适配器名应报 AdapterError")
    except AdapterError as exc:
        text = str(exc)
        check("synthetic-toy" in text and "script-wrapper" in text,
              "未知名字的报错列出可用适配器", text[:120])

    # 脚本适配器缺 script 参数要给出可操作提示
    try:
        resolve_adapter("script-wrapper")
        check(False, "script-wrapper 缺 script 应报错")
    except AdapterError as exc:
        check("script" in str(exc), "缺 script 的报错提示了该传什么", str(exc)[:140])

    # 从 .py 文件加载
    root = _scratch("resolve")
    adapter_file = root / "my_adapter.py"
    adapter_file.write_text(
        "from autoresearch.adapters import BaseExperimentAdapter, RunSpec\n"
        "class MyAdapter(BaseExperimentAdapter):\n"
        "    name = 'my-adapter'\n"
        "    def build_command(self, spec: RunSpec):\n"
        "        return ['python', 'x.py']\n"
        "    def parse_results(self, out_dir):\n"
        "        return {'accuracy': [0.5]}\n"
        "ADAPTER = MyAdapter\n",
        encoding="utf-8",
    )
    loaded = resolve_adapter(str(adapter_file))
    eq(loaded.name, "my-adapter", "从 .py 文件加载适配器")
    eq(loaded.build_command(RunSpec("v", 0, "d")), ["python", "x.py"], "加载后的方法可用")

    # 文件里没有适配器类 → 明确报错
    empty = root / "empty.py"
    empty.write_text("x = 1\n", encoding="utf-8")
    try:
        resolve_adapter(str(empty))
        check(False, "无适配器类的文件应报错")
    except AdapterError as exc:
        check("BaseExperimentAdapter" in str(exc), "提示要继承哪个基类", str(exc)[:140])

    # 多个适配器类 → 要求显式指定
    multi = root / "multi.py"
    multi.write_text(
        "from autoresearch.adapters import BaseExperimentAdapter\n"
        "class A(BaseExperimentAdapter):\n"
        "    def build_command(self, spec): return []\n"
        "    def parse_results(self, out_dir): return {}\n"
        "class B(BaseExperimentAdapter):\n"
        "    def build_command(self, spec): return []\n"
        "    def parse_results(self, out_dir): return {}\n",
        encoding="utf-8",
    )
    try:
        resolve_adapter(str(multi))
        check(False, "多个适配器类应要求显式指定")
    except AdapterError as exc:
        check("ADAPTER" in str(exc), "提示用 ADAPTER 指定", str(exc)[:140])

    # 抽象基类不可直接实例化
    try:
        BaseExperimentAdapter()
        check(False, "抽象基类不可实例化")
    except TypeError:
        check(True, "抽象基类不可实例化")


# --------------------------------------------------------------------------- #
# 内置适配器行为
# --------------------------------------------------------------------------- #


def test_synthetic_adapter() -> None:
    section("合成任务适配器")
    import ast

    from autoresearch.adapters import RunSpec, resolve_adapter

    adapter = resolve_adapter(None)
    eq(adapter.name, "synthetic-toy", "默认适配器")
    check(adapter.owns_code is False, "合成适配器允许 LLM 生成/修补代码")

    ws = _scratch("synthetic")
    adapter.prepare(ws, {})
    templates = adapter.seed_code(ws, {})
    check("train.py" in templates, "提供 train.py 模板")
    check(len(templates["train.py"]) > 1000, "模板是有实质内容的脚本")
    # 契约细节：seed_code() **返回**模板内容，由 s4 负责落盘——适配器不直接写文件，
    # 这样 s4 才能在模板之上做 LLM 修补与保真度检查。
    for rel, content in templates.items():
        (ws / rel).write_text(content, encoding="utf-8")

    # 模板必须是合法 Python（模板语法错误会在运行时才炸，代价很高）
    try:
        ast.parse(templates["train.py"])
        check(True, "模板是合法 Python")
    except SyntaxError as exc:
        check(False, "模板是合法 Python", str(exc))

    ok, reason = adapter.validate_environment()
    check(ok, "合成适配器环境永远可用", reason)
    # 合成适配器**不限制**变体名：消融臂用中性标识（abl-1…），轴取值走 params。
    # 早期版本白名单 {"baseline","method"}，导致 s3 规划的消融一条都跑不到。
    check(adapter.supported_variants() is None,
          "合成适配器不限制变体（消融臂得以执行）",
          str(adapter.supported_variants()))
    eq(adapter.max_variants, 2,
       "合成适配器自报臂数上限 2（内置任务不为消费超参设计，不展开消融）")

    spec = RunSpec("method", 2, "runs/method/seed_2", params={"lr": 0.3})
    cmd = adapter.build_command(spec)
    eq(cmd[0], "python", "argv[0] 是 python（由沙箱替换为当前解释器）")
    check("--variant" in cmd and "method" in cmd, "命令含 variant")
    check("--seed" in cmd and "2" in cmd, "命令含 seed")
    check("--out-dir" in cmd and "runs/method/seed_2" in cmd, "命令含 out_dir")
    check("--lr" in cmd and "0.3" in cmd, "命令含额外超参")

    # 真实执行模板：必须写出指标、且确定性可复现
    import subprocess

    run_dir = ws / "runs" / "method" / "seed_2"
    run_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(ws / "train.py"), "--variant", "method", "--epochs", "2",
         "--seed", "2", "--out-dir", str(run_dir)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        cwd=str(ws),
    )
    check(proc.returncode == 0, "模板脚本能跑通", proc.stderr[-300:])
    parsed = adapter.parse_results(run_dir)
    check("accuracy" in parsed and len(parsed["accuracy"]) == 2,
          "适配器解析出 2 个 epoch 的 accuracy", str(sorted(parsed)))
    check("val_accuracy" in parsed, "解析出 val_accuracy")
    check(all(isinstance(v, float) for vs in parsed.values() for v in vs),
          "所有指标都是 float")

    # 同一 seed 必须复现同一结果（否则跨种子统计无意义）
    proc2 = subprocess.run(
        [sys.executable, str(ws / "train.py"), "--variant", "method", "--epochs", "2",
         "--seed", "2", "--out-dir", str(run_dir)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        cwd=str(ws),
    )
    eq(adapter.parse_results(run_dir)["accuracy"], parsed["accuracy"],
       "同 seed 复现同一指标")

    # 任务必须"不饱和"：准确率不能是 1.0，否则 baseline/method 对照会失去意义
    best = max(parsed["accuracy"])
    check(best < 0.999, "合成任务未饱和（留出了方法差异空间）", f"best accuracy={best}")

    note = adapter.quality_note()
    check("合成" in note and "不构成" in note, "质量声明明确了结果边界", note[:80])


def test_script_wrapper_adapter() -> None:
    section("用户脚本适配器")
    from autoresearch.adapters import RunSpec, resolve_adapter

    root = _workspace_root()
    # 夹具用**纯标准库**脚本，而不是 `templates/experiment/train.py`。
    #
    # 原因：那个模板在模块级 `import numpy`，而 CI 的 test job 刻意不装任何
    # 第三方依赖，于是本套件在所有平台都失败：
    #     File ".../_provided/train.py", line 35
    #         import numpy as np
    #     ModuleNotFoundError: No module named 'numpy'
    # 两者各自都对——模板是给人参考的完整实现，用 numpy 合理；而套件必须在裸环境
    # 跑，所以夹具不能有第三方依赖。换成一个纯标准库实现，反而多验证了一件更重要的
    # 事：**零依赖脚本也能被 script-wrapper 正常驱动**。
    script = root / "autoresearch" / "tests" / "fixtures" / "pure_python_train.py"
    if not check(script.is_file(), "存在可用的测试脚本（纯标准库夹具）", str(script)):
        return

    adapter = resolve_adapter("script-wrapper", {"script": str(script), "epochs": 2})
    eq(adapter.name, "script-wrapper", "适配器名")
    check(adapter.owns_code is True,
          "声明 owns_code=True（s4 必须跳过 LLM 代码生成）")
    check(adapter.supported_variants() is None, "不限制变体（用户脚本自己决定）")
    ok, reason = adapter.validate_environment()
    check(ok, "环境预检通过", reason)

    ws = _scratch("script_wrapper")
    adapter.prepare(ws, {})
    # entrypoint 是给命令行用的**相对**路径（必须能在工作目录里解析），
    # 因此"文件是否存在"要看 code_files()，而不是直接对 entrypoint 做 is_file()
    # ——后者按当前 cwd 解析，在工作目录不同时会误判。
    entry = adapter.entrypoint
    check(entry.startswith("_provided/") and entry.endswith(".py"),
          "entrypoint 是工作区相对路径", entry)
    check("/" in adapter.entrypoint and "\\" not in adapter.entrypoint,
          "entrypoint 用正斜杠（跨平台一致）", adapter.entrypoint)
    files = adapter.code_files(ws)
    check(len(files) == 1 and files[0].is_file(),
          "code_files() 返回可读的脚本", str(files))
    check(files[0].is_absolute(), "code_files() 返回绝对路径", str(files))
    check(str(files[0]).startswith(str(ws.resolve())) or ws.resolve() in files[0].parents,
          "脚本确实落在工作区内", str(files[0]))
    eq(adapter.seed_code(ws, {}), {}, "自带脚本时 seed_code 返回空（不交给 LLM 改写）")

    spec = RunSpec("method", 1, "runs/method/seed_1")
    cmd = adapter.build_command(spec)
    eq(cmd[0], "python", "程序名是 python")
    check("--variant" in cmd and "method" in cmd, "命令含 variant")
    check("--seed" in cmd and "1" in cmd, "命令含 seed")
    check("runs/method/seed_1" in cmd, "命令含 out_dir")
    # 脚本路径必须是带正斜杠的相对路径（跨平台一致）
    check(entry in cmd, "脚本路径是工作区内的相对路径", str(cmd))

    # 真实执行并解析
    import subprocess

    run_dir = ws / "runs" / "method" / "seed_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    # 直接执行时也要用适配器给的 entrypoint，避免夹具改名后这里静默跑错文件
    proc = subprocess.run(
        [sys.executable, str(ws / entry), "--variant", "method",
         "--epochs", "2", "--seed", "1", "--out-dir", str(run_dir)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        cwd=str(ws),
    )
    check(proc.returncode == 0, "用户脚本能跑通", proc.stderr[-300:])
    parsed = adapter.parse_results(run_dir)
    check("accuracy" in parsed, "适配器解析出指标", str(sorted(parsed)))

    # 指标列名重映射
    mapped = resolve_adapter("script-wrapper", {
        "script": str(script), "metrics_map": {"accuracy": "acc", "val_accuracy": "vacc"},
    })
    mapped.prepare(ws, {})
    remapped = mapped.parse_results(run_dir)
    check("acc" in remapped and "vacc" in remapped, "metrics_map 生效", str(sorted(remapped)))
    check("accuracy" not in remapped, "原名被替换而不是并存")

    # 参数模板定制
    custom = resolve_adapter("script-wrapper", {
        "script": str(script),
        "args_template": ["{script}", "--variant", "{variant}", "--out-dir", "{out_dir}"],
    })
    custom.prepare(ws, {})
    # 用适配器给的 entrypoint，而不是写死夹具文件名——换夹具不该让断言碎掉
    entry_custom = custom.entrypoint
    eq(custom.build_command(spec),
       ["python", entry_custom, "--variant", "method", "--out-dir",
        "runs/method/seed_1"],
       "自定义 args_template 生效")

    # 模板里无法解析的占位符必须报错（静默丢参会造成"跑了但设置不对"）
    from autoresearch.adapters import AdapterError

    broken = resolve_adapter("script-wrapper", {
        "script": str(script), "args_template": ["{script}", "--lr", "{lr}"],
    })
    broken.prepare(ws, {})
    try:
        broken.build_command(spec)
        check(False, "无法解析的占位符应报错")
    except AdapterError as exc:
        check("lr" in str(exc), "报错指明了是哪个占位符", str(exc)[:140])

    # 缺脚本文件
    missing = resolve_adapter("script-wrapper", {"script": str(root / "nope.py")})
    ok, reason = missing.validate_environment()
    check(not ok, "脚本不存在时预检失败")
    check("nope.py" in reason, "预检原因里包含脚本名", reason[:120])

    note = adapter.quality_note()
    check("不对实验设计负责" in note, "质量声明点明了适配器的责任边界", note[:90])


# --------------------------------------------------------------------------- #
# 沙箱 run_command
# --------------------------------------------------------------------------- #


def test_sandbox_run_command() -> None:
    section("沙箱 run_command")
    from autoresearch.config import load_config
    from autoresearch.tools.sandbox import SandboxError, make_sandbox

    ws = _scratch("sandbox")
    sandbox = make_sandbox(load_config().sandbox, ws)

    result = sandbox.run_command(["python", "-c", "print(6*7)"])
    check(result.ok, "执行 python 成功", result.stderr[-200:])
    eq(result.stdout.strip(), "42", "stdout 正确")

    # argv[0] == "python" 必须被替换为当前解释器
    result2 = sandbox.run_command(["python", "-c", "import sys; print(sys.executable)"])
    check(result2.stdout.strip().replace("\\", "/").lower().endswith(
        Path(sys.executable).name.lower()) or sys.executable in result2.stdout,
        "python 被替换为当前解释器", result2.stdout.strip())

    # 字符串必须被拒绝，并给出可操作提示
    try:
        sandbox.run_command("python -c 'print(1)'")
        check(False, "字符串 argv 应被拒绝")
    except SandboxError as exc:
        check("列表" in str(exc) and "shell" in str(exc),
              "拒绝字符串并解释原因", str(exc)[:140])

    # shell 元字符必须被拒绝（否则会静默变成"找不到该文件"）
    try:
        sandbox.run_command(["python -c 'print(1)' | cat"])
        check(False, "shell 元字符应被拒绝")
    except SandboxError as exc:
        check("元字符" in str(exc), "拒绝元字符并解释原因", str(exc)[:140])

    # 空 argv
    try:
        sandbox.run_command([])
        check(False, "空 argv 应被拒绝")
    except SandboxError:
        check(True, "空 argv 被拒绝")

    # 不存在的程序：返回失败结果而不是抛异常
    result3 = sandbox.run_command(["definitely-not-a-real-program-xyz", "--help"])
    check(not result3.ok, "不存在的程序返回 ok=False")
    check(result3.returncode != 0, "返回码非 0")

    # describe 要如实说明限制（"配置项存在但不生效"是最糟的情况）
    info = sandbox.describe()
    check("limits" in info and "os" in info, "describe 声明了限制与平台", str(info)[:160])
    if info.get("os") == "posix":
        check("setrlimit" in info["limits"],
              "POSIX 上声明启用了 setrlimit（rlimit 修复）", info["limits"])
    else:
        check("Windows" in info["limits"] or "none" in info["limits"],
              "Windows 上如实声明无 rlimit", info["limits"])


# --------------------------------------------------------------------------- #
# s4 的变体推导与对照
# --------------------------------------------------------------------------- #


def test_variant_derivation() -> None:
    section("变体推导")
    from autoresearch.stages.s4_experiment import (
        _ablation_cells,
        _coerce_scalar,
        _loose_command,
        _plan_variants,
        _safe_variant,
        _variant_params,
    )

    plan = {
        "ablation_matrix": [
            {"name": "momentum", "variants": ["0.0 (off)", "0.9 (on)"], "hypothesis": "h"},
            {"name": "noise_level", "variants": ["0.70", "0.80"], "hypothesis": "h"},
        ]
    }
    eq(_plan_variants(plan), ["baseline", "method"], "默认只返回主对照")
    allv = _plan_variants(plan, max_variants=None)
    eq(allv[0], "baseline", "第一个是 baseline")
    eq(allv[1], "method", "第二个是 method")
    check(len(allv) == 6, "baseline + method + 4 个消融格点", str(allv))

    # 关键回归：消融**取值**不能变成变体名的一部分
    check("0.9_on" not in allv and "0.9 (on)" not in allv,
          "取值本身没有成为变体名", str(allv))
    # 消融变体名必须是中性标识，且不含空格/括号/等号——它们会成为单个 argv token
    abl = [v for v in allv if v not in ("baseline", "method")]
    check(all(v.startswith("abl-") for v in abl),
          "消融臂用中性标识 abl-<n>", str(abl))
    bad_tokens = [v for v in allv if any(ch in v for ch in " =()")]
    eq(bad_tokens, [], "所有变体名都能安全地作为 argv token")

    # 空计划仍然给出最基本的对照
    eq(_plan_variants({}, max_variants=None), ["baseline", "method"], "空计划给主对照")

    # 超参提取：轴与取值走 params，而不是编码进变体名
    eq(_variant_params(plan, "abl-1"), {"momentum": 0.0}, "第 1 个格点的轴取值")
    eq(_variant_params(plan, "abl-2"), {"momentum": 0.9}, "第 2 个格点的轴取值")
    eq(_variant_params(plan, "baseline"), {}, "主对照不附加超参（必须同超参比较）")
    eq(_variant_params(plan, "method"), {}, "method 同样不附加超参")
    eq(_variant_params(plan, "不存在"), {}, "未知变体返回空")

    cells = _ablation_cells(plan)
    eq(len(cells), 4, "格点数量")
    check(all(isinstance(c, tuple) and len(c) == 4 for c in cells), "格点结构一致")
    check(all(c[0].startswith("abl-") for c in cells), "格点名是中性标识")
    check(all(isinstance(c[3], dict) and c[3] for c in cells), "每个格点都带轴取值")
    # 轴名必须能直接当 argparse 的长选项（否则多数训练脚本会直接报错）
    check(all(re.fullmatch(r"[a-z_][a-z_0-9]*", c[1].replace("-", "_")) for c in cells),
          "轴名规范成合法标识符", str([c[1] for c in cells]))
    eq(
        _ablation_cells({"ablation_matrix": [{"name": "noise level 2", "variants": ["0.7"]}]})[0][1],
        "noise_level_2",
        "含空格/数字的轴名被规范化",
    )

    eq(_coerce_scalar("2e-4 (on)"), 0.0002, "科学计数法解析")
    eq(_coerce_scalar("0.70"), 0.7, "小数解析")
    eq(_coerce_scalar("constant"), "constant", "非数值保留字符串")
    eq(_safe_variant("ablation no-budget"), "ablation_no-budget", "变体名规范化")

    # 宽松命令：保留程序名、位置参数、--variant/--out-dir
    loose = _loose_command(["python", "train.py", "--variant", "method", "--epochs", "3",
                            "--seed", "1", "--out-dir", "runs/x", "--lr", "0.3"])
    eq(loose, ["python", "train.py", "--variant", "method", "--out-dir", "runs/x"],
       "宽松命令保留关键参数")
    eq(_loose_command([]), [], "空命令不崩")


def test_comparison_robustness() -> None:
    section("对照逻辑")
    from autoresearch.stages.s4_experiment import _compare

    # 多种子指标名（带 @seed）也要能对照——这是重构前会误报"无结果"的场景
    results = {
        "baseline": {"accuracy@seed=0": [0.9], "accuracy@seed=1": [0.8]},
        "method": {"accuracy@seed=0": [0.93], "accuracy@seed=1": [0.85]},
    }
    cmp1 = _compare(results)
    check(cmp1["available"], "带 @seed 后缀的指标能对照", str(cmp1.get("summary_line")))
    eq(cmp1["baseline"], "baseline", "记录 baseline 是谁")
    eq(cmp1["treatment"], "method", "记录 treatment 是谁")
    check(cmp1["supports_claim"], "两个种子都提升 → 支持 claim")

    # 多臂：选相对提升最大的作为 treatment
    multi = dict(results)
    multi["momentum=0.9"] = {"accuracy@seed=0": [0.95], "accuracy@seed=1": [0.9]}
    cmp2 = _compare(multi)
    eq(cmp2["treatment"], "momentum=0.9", "多臂时选最佳的作为 treatment")
    eq(cmp2["other_variants"], ["method"], "其余变体被列出")

    # 方向正确性：loss 下降算改善
    lossy = {
        "baseline": {"loss": [0.5]},
        "method": {"loss": [0.3]},
    }
    cmp3 = _compare(lossy)
    check(cmp3["supports_claim"], "loss 下降算改善（方向判断正确）")
    eq(cmp3["per_metric"]["loss"]["direction"], "lower", "loss 方向为 lower")

    # 缺 baseline：必须给出可解释的失败
    cmp4 = _compare({"method": {"accuracy": [0.9]}})
    check(not cmp4["available"], "缺 baseline 时不可对照")
    check("baseline" in cmp4["summary_line"], "失败原因提到 baseline",
          cmp4["summary_line"])

    # 无共同指标
    cmp5 = _compare({"baseline": {"a": [1.0]}, "method": {"b": [2.0]}})
    check(not cmp5["available"], "无共同指标时不可对照")
    check("共同指标" in cmp5["summary_line"], "失败原因说明是无共同指标",
          cmp5["summary_line"])

    # 完全空
    cmp6 = _compare({})
    check(not cmp6["available"], "空结果不可对照")
    check(bool(cmp6.get("summary_line")), "空结果也有可读说明")


def test_s4_uses_adapter() -> None:
    """s4 必须走适配器，而不是回退到内置硬编码逻辑。"""
    section("s4 适配器集成")
    from autoresearch.adapters import resolve_adapter
    from autoresearch.stages.s4_experiment import ExperimentStage

    import inspect

    source = inspect.getsource(ExperimentStage)
    check("resolve_adapter" in source, "s4 通过 resolve_adapter 取适配器")
    check("adapter.build_command" in source, "命令由适配器构造")
    check("adapter.parse_results" in source, "指标由适配器解析")
    check("validate_environment" in source, "运行前做环境预检")
    check("supported_variants" in source, "运行前过滤不支持的变体")
    check("run_command" in source, "通过沙箱 run_command 执行")

    # 重构后不应再有硬编码的 baseline/method 循环
    check('for variant in ("baseline", "method")' not in source,
          "不再硬编码 baseline/method 两个变体")
    check("_MINIMAL_TRAIN_PY" not in source, "内置保底脚本已移入适配器")
    check("run_python(" not in source, "不再用 run_python 执行实验（改用 run_command）")

    # 关键：owns_code 的适配器必须跳过 LLM 代码生成
    check("owns_code" in source, "s4 尊重 owns_code 声明")

    # 适配器信息要进产物清单（读者需要知道结果来自哪种后端）
    check("adapter_info" in source, "适配器能力声明写入产物")
    check("adapter.describe()" in source or "describe()" in source,
          "s4 通过 describe() 收集能力声明")
    check("code_files(" in source, "s4 通过 code_files() 定位适配器自带代码")

    adapter = resolve_adapter(None)
    info = adapter.describe()
    for key in ("name", "description", "owns_code", "quality_note"):
        check(key in info, f"describe() 含 {key}")
    check(bool(info["quality_note"]), "describe() 的质量声明非空")


def test_adapter_informs_codegen() -> None:
    """代码生成提示词必须由**适配器**提供，而不是阶段里写死。

    审计发现的问题：`s4` 无条件把合成任务的约定塞进提示词
    （「必须支持 --epochs --seed --variant --out-dir」「metrics.csv 表头固定为
    epoch,loss,accuracy,f1,val_loss,val_accuracy」「变体只能是 baseline|method」）。
    于是一个跑材料仿真的自定义适配器也会收到为小型分类任务写的规范，
    生成的代码自然对不上它真正的指标——而这一点在合成适配器上看不出来。
    """
    section("适配器驱动的代码生成约定")
    from autoresearch.adapters import resolve_adapter

    root = _workspace_root()
    script = root / "autoresearch" / "templates" / "experiment" / "train.py"

    synthetic = resolve_adapter(None)
    sw = resolve_adapter("script-wrapper", {"script": str(script)})

    for adapter in (synthetic, sw):
        conv = adapter.codegen_conventions()
        data = adapter.codegen_data_info()
        variants = adapter.codegen_variants()
        check(isinstance(conv, str) and len(conv) > 20,
              f"{adapter.name}.codegen_conventions() 有实质内容")
        check(isinstance(data, str) and len(data) > 10,
              f"{adapter.name}.codegen_data_info() 有实质内容")
        check(isinstance(variants, str) and variants,
              f"{adapter.name}.codegen_variants() 非空")

    # 合成适配器应当说明它是合成数据；脚本适配器不应当把用户数据说成合成的
    check("合成" in synthetic.codegen_data_info(),
          "合成适配器声明了数据是内部合成的")
    sw_data = sw.codegen_data_info()
    check(sw_data.count("合成") <= 1 and "不得假设数据是管线内合成的" in sw_data,
          "脚本适配器只把「合成」用在否定句里（不声称用户数据是合成的）",
          sw_data[:120])
    # 脚本适配器必须禁止重写用户脚本逻辑
    check("不得重写" in sw.codegen_conventions(),
          "脚本适配器明确禁止重写用户脚本逻辑", sw.codegen_conventions()[:80])

    # s4 必须从适配器取这三段文本，而不是自带字面量
    import inspect

    from autoresearch.stages.s4_experiment import ExperimentStage

    source = inspect.getsource(ExperimentStage)
    check("adapter.codegen_conventions()" in source, "s4 从适配器取脚本规范")
    check("adapter.codegen_data_info()" in source, "s4 从适配器取数据说明")
    check("adapter.codegen_variants()" in source, "s4 从适配器取变体名")
    check("baseline|method" not in source,
          "s4 不再硬编码 variant=\"baseline|method\"")
    check("epoch,loss,accuracy,f1,val_loss,val_accuracy" not in source,
          "s4 不再硬编码合成任务的 metrics.csv 表头")


def test_variant_budget_respects_adapter_cap() -> None:
    """臂数预算取「配置上限」与「适配器上限」的较小值。

    两侧都要有：配置是用户的本次预算，适配器是后端能力约束。取小值意味着用户
    不能把一个只愿意跑 2 臂的适配器推到 6 臂——对真实训练这是必要保护，
    因为「跑不完的消融」比「没有消融」更糟。
    """
    section("臂数预算")
    from autoresearch.adapters import resolve_adapter
    from autoresearch.config import load_config
    from autoresearch.stages.s4_experiment import _plan_variants, _variant_budget

    cfg = load_config()
    synthetic = resolve_adapter(None)
    sw = resolve_adapter("script-wrapper",
                         {"script": str(_workspace_root() / "autoresearch" /
                                        "templates" / "experiment" / "train.py")})

    base = _variant_budget(cfg, synthetic)
    check(base == min(cfg.max_variants, synthetic.max_variants),
          "合成适配器取较小值", f"{base} vs cfg={cfg.max_variants} adapter={synthetic.max_variants}")
    script_budget = _variant_budget(cfg, sw)
    check(script_budget == min(cfg.max_variants, sw.max_variants),
          "脚本适配器取较小值", f"{script_budget}")
    check(script_budget >= 2, "预算不会低于 2（主对照必须能跑）")

    # 配置上限为 0/负数时退化为适配器上限（不能因为一个坏配置而跑不了）
    class _Cfg:
        max_variants = 0

    check(_variant_budget(_Cfg(), synthetic) == synthetic.max_variants,
          "配置上限为 0 时退回适配器上限")

    # 无上限的适配器受配置约束
    class _Uncapped:
        max_variants = None

    check(_variant_budget(cfg, _Uncapped()) == cfg.max_variants,
          "适配器无上限时受配置约束")

    # 消融格点确实能被推导出来（不再被默认值悄悄截成 2）
    plan = {"ablation_matrix": [
        {"name": "momentum", "variants": ["0.0 (off)", "0.9 (on)"]},
        {"name": "noise", "variants": ["0.7", "0.8"]},
    ]}
    allv = _plan_variants(plan, max_variants=None)
    check(len(allv) > 2, "max_variants=None 时消融格点被纳入", str(allv))
    eq(_plan_variants(plan, max_variants=2), ["baseline", "method"],
       "显式限制为 2 时只跑主对照")
    capped = _plan_variants(plan, max_variants=4)
    eq(len(capped), 4, "max_variants=4 时截断到 4 个臂")
    eq(capped[:2], ["baseline", "method"], "截断时主对照优先保留")


def test_metric_direction_is_unified() -> None:
    """指标方向在全项目只能有一套规则。

    审计发现 s3/s4/s5/s9 与 adapters 各自维护了一份 token 表，且已经漂移
    （有的有 fid/fdr，有的有 flops/params）。后果不是报错，而是**同一个指标名
    在不同章节里被判定为不同方向**，让「改善」的定义悄悄改变。
    """
    section("指标方向一致性")
    from autoresearch.adapters import higher_is_better
    from autoresearch.stages.s3_planning import _direction
    from autoresearch.stages.s4_experiment import _higher_is_better as s4_hib
    from autoresearch.stages.s5_analysis import _is_lower_better as s5_lib
    from autoresearch.stages.s9_finalize import _lower_is_better as s9_lib
    from autoresearch.tools.metrics import _higher_is_better as shared

    names = ["accuracy", "f1", "bleu", "exact_match", "val_loss", "loss", "rmse",
             "fid", "fdr", "flops", "params", "latency", "wer", "自定义指标"]
    diverged: list[tuple[str, list[bool]]] = []
    for name in names:
        values = [
            bool(shared(name)),
            bool(s4_hib(name)),
            not bool(s5_lib(name)),
            not bool(s9_lib(name)),
            bool(higher_is_better(name)),
            _direction(name) == "higher",
        ]
        if len(set(values)) != 1:
            diverged.append((name, values))
    eq(diverged, [], "六处方向判定完全一致（含 fid/fdr/flops/params）")

    # 规范表是全项目共用的：adapters 侧的别名必须与 tools.metrics 对齐
    from autoresearch.adapters import LOWER_IS_BETTER_TOKENS

    for token in ("fid", "fdr", "flops", "params", "loss", "latency"):
        check(not higher_is_better(token), f"{token} 被判为越小越好")
    check(len(LOWER_IS_BETTER_TOKENS) >= 14, "别名表非空且完整")


def test_json_metrics_with_bom() -> None:
    """带 BOM 的 JSON 指标文件必须能读出来。

    实测：PowerShell 的 `Set-Content -Encoding utf8` 默认写 BOM，而
    `json.loads` 拒绝带 BOM 的文本。同一条路径下 `tools.metrics.parse_metrics`
    能读出指标、适配器却返回 `{}`——**同一个文件两处解析结果不同**是极难归因的 bug。
    """
    section("JSON 指标与 BOM")
    import json as _json

    from autoresearch.adapters import resolve_adapter

    root = _workspace_root()
    script = root / "autoresearch" / "templates" / "experiment" / "train.py"
    adapter = resolve_adapter("script-wrapper",
                              {"script": str(script), "metrics_file": "m.json",
                               "metrics_format": "json"})
    ws = _scratch("bom")
    adapter.prepare(ws, {})
    run_dir = ws / "runs" / "v" / "seed_0"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = [{"epoch": 1, "accuracy": 0.9}, {"epoch": 2, "accuracy": 0.95}]
    # 无 BOM
    (run_dir / "m.json").write_text(_json.dumps(payload), encoding="utf-8")
    clean = adapter.parse_results(run_dir)
    eq(clean.get("accuracy"), [0.9, 0.95], "无 BOM 的 list-of-records 可读")

    # 带 BOM
    (run_dir / "m.json").write_bytes(b"\xef\xbb\xbf" + _json.dumps(payload).encode("utf-8"))
    bom = adapter.parse_results(run_dir)
    eq(bom.get("accuracy"), [0.9, 0.95], "带 BOM 的 JSON 同样可读")

    # 与 tools.metrics 的结论一致（两处解析不能给出不同答案）
    from autoresearch.tools.metrics import parse_metrics

    tool_view = parse_metrics(run_dir / "m.json")
    eq(tool_view.get("accuracy"), bom.get("accuracy"),
       "适配器与 tools.metrics 对同一文件结论一致")


def test_no_unfilled_template_placeholders() -> None:
    """生成的 main.tex 里不能残留 ``__TOKEN__``。

    LaTeX 模板声明了五个占位符，而早期实现只替换三个，于是
    ``\\date{__DATE__}`` 与 ``\\textbf{Keywords:} __KEYWORDS__`` 会原样进入 PDF。
    这类残留不会让编译失败，因此**不会被 s7 的编译修复闭环发现**——
    它只是安静地把编译期 token 印在论文上。所以必须有一道显式自检。
    """
    section("模板占位符自检")
    import re as _re

    from autoresearch.config import PROJECT_ROOT

    template = PROJECT_ROOT / "templates" / "paper" / "main.tex"
    if not check(template.is_file(), "模板存在", str(template)):
        return
    text = template.read_text(encoding="utf-8")
    declared = sorted(set(_re.findall(r"__([A-Z][A-Z0-9_]{2,})__", text)))
    check(len(declared) >= 3, "模板声明了占位符", str(declared))

    # s6 必须对每一个声明过的占位符都有替换语句
    stage_src = (PROJECT_ROOT / "stages" / "s6_writing.py").read_text(encoding="utf-8")
    missing = [tok for tok in declared if f"__{tok}__" not in stage_src]
    eq(missing, [], "s6 替换了模板声明的全部占位符")

    # 自检逻辑本身要存在（否则下次漏替换又只能靠肉眼）
    check("placeholders_left" in stage_src, "装配后会自检残留占位符")
    check("template_placeholders_unfilled" in stage_src,
          "发现残留时会记事件（可被审计）")

    # 关键词必须有来源（LLM 失败时也要有兜底，不能留空）
    check("_fallback_keywords" in stage_src, "关键词有确定性兜底")


def test_codegen_prompt_defers_to_adapter() -> None:
    """代码生成**提示词文件**不能写死只有适配器才知道的契约。

    这是独立验证发现的盲区：`test_adapters` 只对 `.py` 源码做字面量断言，
    `test_prompts` 只校验提示词里的变量名与文件存在性——**没有任何测试把
    「提示词里写死的硬性契约」与「适配器声明的能力」对齐**。于是此前那轮修复
    只做了 `.py` 半边：s4 把适配器约定注入了，但 `s4_codegen.md` 里仍写着
    「Hard requirements — every one is checked」的固定表头 / 四个 CLI 参数 /
    `baseline|method`，把注入的约定压了过去。
    """
    section("提示词与适配器契约对齐")
    import inspect

    from autoresearch.config import PROJECT_ROOT
    from autoresearch.stages.s4_experiment import ExperimentStage

    prompt_path = PROJECT_ROOT / "prompts" / "s4_codegen.md"
    if not check(prompt_path.is_file(), "s4_codegen.md 存在", str(prompt_path)):
        return
    prompt = prompt_path.read_text(encoding="utf-8")

    for var in ("{{ workspace_conventions }}", "{{ data_info }}", "{{ variant }}"):
        check(var in prompt, f"提示词引用了 {var}")

    check("epoch,loss,accuracy,f1,val_loss,val_accuracy" not in prompt,
          "提示词不再写死 metrics.csv 表头（改由适配器 conventions 提供）")
    check("Valid variant names are" not in prompt, "提示词不再写死变体白名单")
    check("run_command" not in prompt,
          "提示词不再要求 run_command（命令由 adapter.build_command 提供）")

    source = inspect.getsource(ExperimentStage)
    schema_part = source.split("_CODEGEN_SCHEMA")[1][:700] if "_CODEGEN_SCHEMA" in source else ""
    check('"run_command"' not in schema_part,
          "_CODEGEN_SCHEMA 不再声明 run_command")

    for call in ("codegen_conventions()", "codegen_data_info()", "codegen_variants()"):
        check(f"adapter.{call}" in source, f"s4 传入 adapter.{call}")

    # 渲染一遍：变量名拼错会渲染成空串，必须用哨兵值验证真的被替换进去
    from autoresearch.prompts import PromptLibrary

    lib = PromptLibrary(PROJECT_ROOT / "prompts", language="zh")
    rendered = lib.render(
        "s4_codegen",
        plan_block="PLAN",
        existing_code_block="CODE",
        data_info="DATA-SENTINEL",
        variant="VARIANT-SENTINEL",
        workspace_conventions="CONVENTIONS-SENTINEL",
    )
    for sentinel in ("DATA-SENTINEL", "VARIANT-SENTINEL", "CONVENTIONS-SENTINEL"):
        check(sentinel in rendered, f"渲染后 {sentinel} 出现在提示词里")
    check(len(rendered) > 800, "渲染结果完整", f"{len(rendered)} chars")


def test_section_prompt_has_no_phantom_fields() -> None:
    """章节撰写提示词不能要求一个没有来源的字段。

    独立验证发现：`s6_section.md` 要求每个 claim 填 `evidence_id`（并说
    「use the ids from the evidence block」），但 `_evidence_block()` 输出的各段里
    **没有任何 id 字段**，代码也从不读 `claims`。模型被要求填一个不存在的东西，
    只能编造——这正是自动科研最该避免的行为。
    """
    section("章节提示词无幻影字段")
    import inspect

    from autoresearch.config import PROJECT_ROOT
    from autoresearch.stages.s6_writing import WritingStage

    prompt = (PROJECT_ROOT / "prompts" / "s6_section.md").read_text(encoding="utf-8")
    check("evidence_id" not in prompt,
          "提示词不再要求 evidence_id（证据块里没有 id 可引用）")
    check("use the ids from the evidence block" not in prompt,
          "不再指示模型引用不存在的 id")

    source = inspect.getsource(WritingStage)
    schema_part = source.split("_SECTION_SCHEMA")[1][:600] if "_SECTION_SCHEMA" in source else ""
    check('"claims"' not in schema_part,
          "_SECTION_SCHEMA 不再声明未被消费的 claims 字段")

    # 「提示词要求 id」与「代码提供 id」必须同时成立或同时不成立，不能只做一半
    eq("evidence_id" in source, "evidence_id" in prompt,
       "提示词与代码对 evidence_id 的立场一致")


def test_keywords_survive_review_loop() -> None:
    """评审回环不能把 LLM 生成的关键词换成兜底串。

    独立验证发现：`_apply_iteration` 调 `_assemble_main` 时不传 `keywords`，
    于是只要发生过一轮评审回环，`main.tex` 的 Keywords 行就会被硬编码英文兜底串覆盖
    ——不报错、不记事件，恰好是模板占位符那个修复想消灭的「安静地把错的东西印进 PDF」。
    """
    section("关键词跨回环保持")
    import inspect

    from autoresearch.stages.s6_writing import WritingStage

    source = inspect.getsource(WritingStage)
    # 用整个方法体，别用固定长度切片——方法长起来断言就会静默失效
    body = source.split("def _apply_iteration")[-1].split("\n    def ")[0]
    check("keywords" in body, "_apply_iteration 会读取/传递 keywords",
          f"方法体 {len(body)} chars")
    check("paper_meta.json" in body, "从 paper_meta.json 读回关键词")
    check("load_json" in body, "通过 ctx.load_json 读取（可容错）")
    check('"keywords": keywords' in source or "'keywords': keywords" in source,
          "首轮把关键词落盘到 paper_meta.json")
    # 回环路径必须把 keywords 传进装配，否则又被兜底串覆盖
    check("_assemble_main(paper_dir, title, abstract, warnings, keywords)" in body,
          "_apply_iteration 把 keywords 传给 _assemble_main")


def test_real_experiment_adapter_real_script() -> None:
    """真实实验适配器：接一份**真实科研脚本**时才暴露的接口摩擦。

    被测对象是 ``adapters/lorenz_governance.py``，它驱动 Kong 的
    ``gov_naive_update.py``（Lorenz-63 governance 实验，内部跑 5 个模型种子）。
    合成玩具适配器永远不会暴露下面这些问题：

    1. **脚本内部固定种子、命令行不接受 seed**，而 ``RunSpec.seed`` 是管线必需的。
       适配器必须忽略它——把 ``--seed`` 传进不认识它的 argparse 会直接退出码 2。
    2. **本地依赖是传递闭包**。第一版手工列了两个文件，运行时炸在
       ``validate_faithful_chaos`` → ``validate_lorenz_lyapunov`` 这条传递依赖上。
    3. **指标要按脚本内部种子拆成多条序列**，否则「种子间差异」会被当成
       「种子内噪声」；这个实验里 ``d|lam-lr|`` 的种子间极差是 0.38（两个数量级）。
    4. **缺失值必须丢弃而不是填 0**（pre 侧 ``vpt`` 常为 ``None``）。
    5. **双模式导入**：作为 ``.py`` 文件独立加载时相对导入会失败，需退回绝对导入。

    这里**不实际执行**脚本（那要 1-2 分钟）；真实执行由下面
    ``test_lorenz_adapter_runs_for_real`` 在 ``AUTORESEARCH_TEST_SLOW=1`` 时门控。
    """
    section("真实实验适配器：Lorenz governance")
    from autoresearch.adapters import resolve_adapter
    from autoresearch.adapters.lorenz_governance import (
        ADAPTER,
        LorenzGovernanceAdapter,
    )

    # --- 契约与注册 --------------------------------------------------- #
    eq(ADAPTER, LorenzGovernanceAdapter, "模块级 ADAPTER 已导出（支持 .py 路径加载）")
    check(LorenzGovernanceAdapter.owns_code is True,
          "owns_code=True：跳过 LLM 代码生成，但保留调试闭环与保真度检查")
    eq(LorenzGovernanceAdapter.max_variants, 1,
       "臂数压到 1（脚本一次跑完全部种子，再加臂只是重复同样的计算）")
    _probe = LorenzGovernanceAdapter({})
    check(_probe.supported_variants() is None,
          "supported_variants() 返回 None（不做白名单过滤）")

    # 加载路径：module:Class 必须能用
    loaded = resolve_adapter(
        "autoresearch.adapters.lorenz_governance:LorenzGovernanceAdapter", {}
    )
    eq(loaded.name, "lorenz-governance", "module:Class 加载路径可用")

    root = _workspace_root()
    adapter = LorenzGovernanceAdapter({"source_dir": str(root), "update_epochs": 1})

    # --- 1) 命令行绝不能含脚本不认识的 --seed -------------------------- #
    spec = RunSpec(variant="naive-update", seed=7, out_dir="runs/naive-update/seed_7")
    argv = adapter.build_command(spec)
    check("--seed" not in argv,
          "不把脚本不认识的 --seed 传进命令行（否则 argparse 退出码 2）",
          " ".join(argv))
    check("--update-epochs" in argv, "传递了脚本真正支持的参数", " ".join(argv))
    check("--model-dir" in argv and "--data" in argv,
          "传递了脚本需要的输入目录参数", " ".join(argv))
    check(any(str(a).endswith("gov_naive_update.py") for a in argv),
          "入口脚本指向 _provided 下的副本", argv[1])
    check("runs/naive-update/seed_7" in " ".join(argv),
          "输出路径跟随 spec.out_dir（含由 spec.seed 决定的目录名）",
          " ".join(argv))

    # --- 2) 依赖闭包是**自动解析**的（不是手工清单） -------------------- #
    # 发布仓库里没有用户那份实验脚本，所以这里的闭包应当为空——但不能报错。
    # 真正要钉住的是「解析由 AST 驱动、缺失文件被安静跳过」这个契约：
    # 手工清单会漏传递依赖（第一版就漏了 validate_lorenz_lyapunov），
    # 而自动解析在任何目录下都不会因为单文件缺失而崩。
    deps = LorenzGovernanceAdapter({"source_dir": str(root)}).local_deps()
    eq(deps, [], "入口脚本不存在时闭包为空且不抛异常")
    check(hasattr(LorenzGovernanceAdapter, "local_deps"),
          "依赖闭包由 local_deps() 自动解析（不是写死的清单）")

    # 用真实脚本目录验证解析确实工作（该目录不在发布仓库里时跳过）
    real_src = Path(r"C:\Users\user\Desktop\Phd-foundations\sakanaai ctm")
    # 需要 PyTorch 才能通过 validate_environment —— 但 torch 是**可选依赖**，
    # 裸环境（CI 的 test job 刻意不装任何第三方依赖）里没有它。
    # 只有关闭 PyTorch 的相关断言，依赖解析与命令构造这两件事与 torch 无关，照测。
    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False
    if (real_src / "gov_naive_update.py").is_file():
        real_deps = LorenzGovernanceAdapter({"source_dir": str(real_src)}).local_deps()
        check(len(real_deps) >= 2,
              "真实脚本目录下解析出传递依赖闭包", str(real_deps))
        check("validate_lorenz_lyapunov.py" in real_deps,
              "**传递**依赖被解析到（手工清单曾漏掉这一个）", str(real_deps))
        for third in ("torch.py", "numpy.py", "argparse.py", "pathlib.py"):
            check(third not in real_deps, f"第三方/标准库不进闭包：{third}")
        real = LorenzGovernanceAdapter(
            {"source_dir": str(real_src), "update_epochs": 1}
        )
        if has_torch:
            env_ok, env_why = real.validate_environment()
            check(env_ok, "真实目录下环境预检通过（含 5 个 checkpoint）", env_why)
        else:
            env_ok, env_why = real.validate_environment()
            check(env_ok is False and "PyTorch" in env_why,
                  "缺 PyTorch 时 validate_environment 如实报出能力缺口而不是崩溃",
                  env_why)
        real_argv = real.build_command(
            RunSpec(variant="naive-update", seed=7, out_dir="runs/v/seed_7")
        )
        check("--seed" not in real_argv,
              "真实脚本的命令行同样不含 --seed（它不接受该参数）",
              " ".join(real_argv))
    else:
        print(f"  SKIP 真实脚本目录不存在: {real_src}")

    # --- 3)(4) 指标解析：按种子拆序列 + 缺失值不填 0 -------------------- #
    series = adapter.parse_results(_lorenz_fixture_dir())
    check(bool(series), "fixture 能解析出指标")
    if series:
        names = {k.split("@seed=")[0] for k in series}
        seeds = {k.split("@seed=")[-1] for k in series if "@seed=" in k}
        check(len(names) >= 4, "解析出多个指标名", str(sorted(names)))
        check(seeds == {"42", "7", "5555"},
              "按脚本内部种子拆成多条独立序列（跨种子统计的基础）",
              str(sorted(seeds)))
        check(all(isinstance(v, list) and v for v in series.values()),
              "每条都是非空序列（不是标量）")
        # pre 侧 vpt 为 null：必须整条缺失，而不是出现一个 0
        check("source_vpt@seed=42" in series,
              "post 侧存在的 vpt 被采纳", str(sorted(k for k in series if "vpt" in k)))
        zeros = [
            k for k, vs in series.items() if "vpt" in k and any(v == 0.0 for v in vs)
        ]
        eq(zeros, [], "缺失的 vpt 没有被充成 0（宁缺勿造）")
        # 关键：种子间的差异必须**可见**，否则说明被合并成了一条序列
        gap_keys = sorted(k for k in series if k.startswith("delta_source_lyapunov_gap"))
        eq(len(gap_keys), 3,
           "delta_source_lyapunov_gap 保留了 3 个种子各自的取值")
        gaps = sorted(series[k][0] for k in gap_keys)
        check(gaps[-1] - gaps[0] > 0.3,
              "种子间极差被完整保留（这个实验里 d|lam-lr| 相差两个数量级）",
              str(gaps))

    # --- 5) 双模式导入：作为独立 .py 文件加载 --------------------------- #
    import importlib.util

    path = root / "autoresearch" / "adapters" / "lorenz_governance.py"
    if check(path.is_file(), "适配器文件存在", str(path)):
        spec2 = importlib.util.spec_from_file_location("_probe_lorenz", path)
        module = importlib.util.module_from_spec(spec2)  # type: ignore[arg-type]
        try:
            spec2.loader.exec_module(module)  # type: ignore[union-attr]
            ok, err = True, ""
        except Exception as exc:  # noqa: BLE001
            ok, err = False, f"{type(exc).__name__}: {exc}"
        check(ok,
              "能被独立加载（.py 路径加载时相对导入需退回绝对导入）",
              err)

    # --- 自述必须如实说明接口摩擦 --------------------------------------- #
    note = LorenzGovernanceAdapter({"source_dir": str(root)}).quality_note()
    check(any(k in note for k in ("固定 5 个种子", "不接受外部 seed", "重复执行")),
          "quality_note 如实说明了「内部固定种子 / 会重复执行」这一摩擦",
          note[:140])


def test_lorenz_adapter_runs_for_real() -> None:
    """可选：真实执行一次（约 1-2 分钟），由 AUTORESEARCH_TEST_SLOW=1 门控。

    默认不跑：套件的硬性不变量是「全绿不需要网络/API Key」，而这条还需要
    PyTorch 与那个实验脚本。门控而不是删除，是因为「适配器能不能真的驱动
    这份脚本」只能靠真实执行证明——命令构造正确不等于脚本会成功。
    """
    import os
    import shutil

    import sys as _sys

    section("真实实验适配器：真实执行（门控）")
    if os.environ.get("AUTORESEARCH_TEST_SLOW") != "1":
        print("  SKIP 真实执行（设 AUTORESEARCH_TEST_SLOW=1 启用）")
        return

    from autoresearch.adapters.lorenz_governance import LorenzGovernanceAdapter
    from autoresearch.config import load_config
    from autoresearch.tools.sandbox import SubprocessSandbox

    root = _lorenz_source_dir()
    if root is None:
        print(
            "  SKIP 未找到真实实验脚本目录"
            "（设 AUTORESEARCH_LORENZ_SOURCE=<gov_naive_update.py 所在目录> 启用）"
        )
        return
    adapter = LorenzGovernanceAdapter({"source_dir": str(root), "update_epochs": 1})

    ws = _scratch("lorenz_real")
    adapter.prepare(ws, {})
    spec = RunSpec(variant="naive-update", seed=0, out_dir="runs/v/seed_0")
    run_dir = ws / spec.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    sandbox = SubprocessSandbox(load_config().sandbox, ws)
    try:
        result = sandbox.run_command(adapter.build_command(spec), timeout=900)
    except Exception as exc:  # noqa: BLE001
        check("沙箱执行未抛异常", False, f"{type(exc).__name__}: {exc}")
        return
    if not check(result.ok, "沙箱执行成功", (result.stderr or "")[-400:]):
        return
    series = adapter.parse_results(run_dir)
    check(bool(series), "真实执行的输出能被解析成指标", str(list(series)[:4]))
    check(len({k.split("@seed=")[-1] for k in series}) == 5,
          "解析出全部 5 个脚本内部种子",
          str(sorted({k.split("@seed=")[-1] for k in series})))
    _sys.stdout.flush()
    shutil.rmtree(ws, ignore_errors=True)



def test_metric_direction_declaration() -> None:
    """适配器**声明**指标方向，而不是让管线从名字猜。

    这是非 ML 领域（材料/化学/临床…）能否复用的关键。管线的通用词表是 ML 中心的
    （loss/accuracy/f1/bleu…），在材料学实测 18 个指标里 **5 个判反、5 个无固有方向
    却被静默选了一个**，合计 10/18 不可信。

    最要命的是：方向判错**不会报错**，它只会让「改善」的定义反过来。
    下面用一个真实的材料学量（消光系数 k）演示结论翻转。
    """
    section("指标方向：适配器声明优先")
    from autoresearch.tools.metrics import _higher_is_better

    # 1) 词表本身会判反：k（消光系数）应当越小越好（越小越透明）
    check(
        _higher_is_better("k") is True,
        "通用词表把单字母 k 猜成「越大越好」（这是它会判反的证据）",
        str(_higher_is_better("k")),
    )
    check(
        _higher_is_better("k", declared={"k": False}) is False,
        "声明优先：k 被显式声明为越小越好后，判定随之改变",
    )
    # 2) 声明不影响未声明的指标（仍回落词表）
    check(
        _higher_is_better("loss", declared={"k": False}) is False,
        "未声明的指标仍回落通用词表",
    )
    # 3) 声明也覆盖"无固有方向"的量（把 n 编码成偏差量后方向就明确了）
    check(
        _higher_is_better("n_deviation_1e3", declared={"n_deviation_1e3": False}) is False,
        "无固有方向的量可以编码成偏差量来获得明确方向",
    )

    # 4) 端到端：同一份数据，声明与否会**翻转**对照结论
    from autoresearch.stages.s4_experiment import _compare

    # k 从 0.5 降到 0.2（材料变透明，真实改善）
    results = {
        "baseline": {"k_at_550nm": [0.5]},
        "treatment": {"k_at_550nm": [0.2]},
    }
    naive = _compare(results)
    aware = _compare(results, directions={"k_at_550nm": False})
    check(
        naive.get("supports_claim") is False,
        "不声明方向时，真实的改善被判成「不支持 claim」",
        str(naive.get("supports_claim")),
    )
    check(
        aware.get("supports_claim") is True,
        "声明方向后，同一份数据被正确判为「支持 claim」——**结论翻转**",
        str(aware.get("supports_claim")),
    )
    check(
        naive.get("improved") != aware.get("improved"),
        "两边的 improved 列表不同，证明差异来自方向而非数据",
        f"{naive.get('improved')} vs {aware.get('improved')}",
    )


def test_materials_adapter_contract() -> None:
    """材料光学适配器：非 ML 领域的接口契约。

    它是第一个非 ML 实测案例，暴露了三处与 ML 不同的形态：

    1. **没有 epoch** —— 命令里必须是物理参数（--replicates），不能出现 ML 概念
    2. **序列轴是重复测量**，不是训练轨迹 —— `metric_axis()` 如实声明
    3. **方向必须由适配器声明** —— 5 个指标里通用词表会判反 4 个
    """
    section("材料光学适配器（非 ML 领域契约）")
    from autoresearch.adapters import resolve_adapter
    from autoresearch.adapters.materials_optics import (
        DIRECTIONS,
        ADAPTER,
        MaterialsOpticsAdapter,
    )
    from autoresearch.tools.metrics import _higher_is_better

    eq(ADAPTER, MaterialsOpticsAdapter, "模块级 ADAPTER 已导出（支持 .py 路径加载）")
    check(MaterialsOpticsAdapter.owns_code is False,
          "owns_code=False：适配器给驱动模板，LLM 可在跑不通时修补")

    # 加载路径
    for label, spec in (
        ("module:Class", "autoresearch.adapters.materials_optics:MaterialsOpticsAdapter"),
        ("文件路径", "autoresearch/adapters/materials_optics.py"),
    ):
        try:
            got = resolve_adapter(spec, {})
            eq(got.name, "materials-optics", f"{label} 加载路径可用")
        except Exception as exc:  # noqa: BLE001
            check(f"{label} 加载路径可用", False, f"{type(exc).__name__}: {exc}")

    # 缺 module 参数时必须给出可操作的报错，而不是崩
    bare = MaterialsOpticsAdapter({})
    ok, why = bare.validate_environment()
    check(ok is False and "module" in why,
          "缺 module 参数时如实报错并说明需要什么", why)

    # 契约：模块必须提供 nk_table()
    probe = MaterialsOpticsAdapter({"module": str(ROOT / "pyproject.toml")})
    ok2, why2 = probe.validate_environment()
    check(ok2 is False and "nk_table" in why2,
          "模块缺少必需 API 时报出缺什么", why2)

    # 命令形态：物理参数在、ML 概念不在
    real = Path(r"C:\Users\user\Desktop\Phd-foundations")
    adapter = MaterialsOpticsAdapter({
        "module": str(ROOT / "autoresearch" / "adapters" / "materials_optics.py"),
        "replicates": 3,
    })
    argv = adapter.build_command(RunSpec(variant="baseline", seed=0, out_dir="runs/v/seed_0"))
    joined = " ".join(str(a) for a in argv)
    check("--replicates" in argv, "命令用物理概念 --replicates（重复测量）", joined)
    check("--epochs" not in argv, "命令里没有 ML 概念 --epochs", joined)
    check("--variant" in argv and "--out-dir" in argv, "保留了适配器通用契约参数", joined)

    # 方向声明：必须覆盖全部产出，且与通用词表**明确不同**
    declared = MaterialsOpticsAdapter({}).metric_directions()
    eq(declared, dict(DIRECTIONS), "metric_directions() 返回全部声明")
    check(len(declared) >= 4, "声明了足够多的指标", str(sorted(declared)))
    disagree = [n for n, v in declared.items() if _higher_is_better(n) != v]
    check(len(disagree) >= 3,
          "其中多个指标的声明与通用词表**不一致**（若无冲突就无需声明）",
          f"不一致: {disagree}")

    # 序列轴如实声明（不是 epoch）
    eq(MaterialsOpticsAdapter({}).metric_axis(), "replicate",
       "metric_axis() 声明为重复测量（非 epoch）")

    # quality_note 必须说明边界
    note = MaterialsOpticsAdapter({}).quality_note()
    check("重复测量" in note and ("波长" in note or "不确定度" in note),
          "quality_note 说明了序列语义与边界", note[:120])


def test_comsol_adapter_encoding_and_contract() -> None:
    """COMSOL 适配器：非 ML 领域专属的两类问题。

    1. **日志编码** —— COMSOL console 是 UTF-16LE + BOM，按 UTF-8 读会乱码。
    2. **中文错误不可恢复** —— 实测 21 个真实日志里非 ASCII 字符含 168 个
       U+FFFD（REPLACEMENT CHARACTER）。这说明信息在 COMSOL **写出时**就丢了，
       **转码救不回来**。适配器因此如实量化损坏度，并默认加 `-locale en_US`
       让后续运行的错误可读，而不是把乱码当原文往下传。

    这个测试**不需要 COMSOL**，用合成的 UTF-16 日志即可覆盖全部逻辑。
    """
    section("COMSOL 适配器：编码与契约")
    import tempfile

    from autoresearch.adapters import resolve_adapter
    from autoresearch.adapters.comsol_batch import (
        DIRECTIONS,
        ADAPTER,
        ComsolBatchAdapter,
        corruption_ratio,
        read_comsol_log,
    )
    from autoresearch.tools.metrics import _higher_is_better

    eq(ADAPTER, ComsolBatchAdapter, "模块级 ADAPTER 已导出")

    for label, spec in (
        ("module:Class", "autoresearch.adapters.comsol_batch:ComsolBatchAdapter"),
        ("文件路径", "autoresearch/adapters/comsol_batch.py"),
    ):
        try:
            got = resolve_adapter(spec, {})
            eq(got.name, "comsol-batch", f"{label} 加载路径可用")
        except Exception as exc:  # noqa: BLE001
            check(f"{label} 加载路径可用", False, f"{type(exc).__name__}: {exc}")

    # 无参数时报错必须同时给出两种模式的可操作指引
    ok, why = ComsolBatchAdapter({}).validate_environment()
    check(ok is False and "log_dir" in why and "model" in why,
          "缺参数时同时给出 ingest 与 run 两种模式的指引", why)

    # --- UTF-16 日志读写 ---
    work = _scratch("comsol_enc")
    # 构造一个"真实形态"的日志：UTF-16LE + BOM，含失败步骤与损坏字符。
    # 刻意**同时**放入合法的非 ASCII（ε、δ）与被毁的 U+FFFD ——
    # 真实日志就是这样（实测里有 U+03B4 等正常希腊字母混在 168 个 U+FFFD 中）。
    # 若只有 U+FFFD，损坏率恒为 1.0，就测不出"占比"这个语义了。
    body = (
        "set epsilonr -> 4.84\n"
        "std.run: FAILED -> com.comsol.util.exceptions.FlException: Exception:\n"
        "\tFlException: \ufffd\ufffd\ufffd\ufffd\n"
        "S11: FAILED -> FlException: \ufffd\ufffd\n"
        "epsilon = \u03b5  delta = \u03b4\n"
        "OK   WaveEquationElectric -> type=WaveEquationElectric\n"
        "OK   CrossSectionCalculation -> type=CrossSectionCalculation\n"
    )
    log = work / "probe.console.txt"
    log.write_bytes(body.encode("utf-16"))

    text, enc = read_comsol_log(log)
    check(enc in ("utf-16", "utf-16-le"), "按 BOM 识别出 UTF-16", enc)
    check("FAILED" in text, "内容解码正确（含 FAILED）")
    check("WaveEquationElectric" in text, "英文标识符完整保留")

    ratio = corruption_ratio(text)
    check(0.0 < ratio < 1.0,
          "损坏比例按非 ASCII 中的占比算出（合法字符与被毁字符并存）", f"{ratio:.3f}")
    check(corruption_ratio("全部是 ASCII") == 0.0, "纯 ASCII 的损坏率为 0")
    check(corruption_ratio("\ufffd\ufffd") == 1.0, "全损坏时为 1.0")

    # 一个"干净"的日志损坏率为 0
    clean = work / "clean.console.txt"
    clean.write_bytes("std.run: OK\nOK   A -> type=X\n".encode("utf-16"))
    check(corruption_ratio(read_comsol_log(clean)[0]) == 0.0,
          "干净日志的损坏率为 0", "无 U+FFFD")

    # --- parse_results ---
    series = ComsolBatchAdapter({"log_dir": str(work)}).parse_results(work)
    check(bool(series), "能从日志目录解析出指标", str(sorted(series)))
    if series:
        eq(len(series["failed_steps"]), 2, "两个日志各贡献一个观测")
        check(sum(series["failed_steps"]) >= 2.0,
              "识别出 FAILED 步骤", str(series["failed_steps"]))
        check(max(series["log_corruption_ratio"]) > 0.5,
              "损坏率被作为指标产出", str(series["log_corruption_ratio"]))

    # --- 方向声明 ---
    declared = ComsolBatchAdapter({}).metric_directions()
    eq(declared, dict(DIRECTIONS), "metric_directions() 返回全部声明")
    disagree = [n for n, v in (declared or {}).items() if _higher_is_better(n) != v]
    check(len(disagree) >= 2,
          "多个指标的声明与通用词表不一致（否则无需声明）", f"不一致: {disagree}")
    eq(ComsolBatchAdapter({}).metric_axis(), "log", "序列轴声明为 log（无物理顺序）")

    # --- run 模式命令形态 ---
    runner = ComsolBatchAdapter({"model": "m.mph", "study": "std1"})
    argv = runner.build_command(RunSpec(variant="baseline", seed=0, out_dir="runs/v/seed_0"))
    joined = " ".join(str(a) for a in argv)
    check("batch" in argv, "使用 comsol batch 子命令", joined)
    check("-locale" in argv and "en_US" in argv,
          "默认加 -locale en_US（否则中文错误会被写成不可恢复的 U+FFFD）", joined)
    check("--epochs" not in argv, "命令里没有 ML 概念", joined)

    # --- quality_note 必须说明两个边界 ---
    note = ComsolBatchAdapter({}).quality_note()
    check("U+FFFD" in note or "不可恢复" in note or "转码" in note,
          "quality_note 说明中文错误不可恢复", note[:140])
    check("从未被执行" in note or "没有" in note,
          "quality_note 说明 run 模式未被执行", note[:200])


def test_parameter_sweep() -> None:
    """多参数扫描：展开、预算护栏、效应分析。

    材料/化学/器件研究最自然的实验形态是参数扫描（温度 × 成分 × 退火时间），
    而它和"方法 vs 基线"是两种不同的设计。这里测的是**会静默出错**的地方：
    截断破坏正交性、OFAT 测不出交互、参考格点拿不到参数、
    覆盖 target 没进命令行——每一个都不会报错，只会给出看起来正常的错误结论。
    """
    section("多参数扫描：展开与预算护栏")
    from autoresearch.adapters.sweep import (
        SweepAxis,
        axis_effects,
        effect_ratio,
        expand_sweep,
        interaction_effects,
        rank_axes,
    )

    # --- 参数校验 ---
    try:
        SweepAxis("", (1, 2))
        check("空轴名被拒绝", False, "未抛异常")
    except ValueError:
        check("空轴名被拒绝", True)
    try:
        SweepAxis("x", (1,))
        check("少于 2 个取值被拒绝", False, "未抛异常")
    except ValueError:
        check("少于 2 个取值被拒绝", True)

    T = SweepAxis("temp_C", (600, 700, 800), "C")
    X = SweepAxis("comp_x", (0.0, 0.1), "")
    H = SweepAxis("time_h", (1, 4), "h")

    # --- OFAT 数量 = 1 + Σ(k-1) ---
    ofat = expand_sweep([T, X, H], mode="ofat", max_runs=None)
    eq(ofat.n_runs, 1 + 2 + 1 + 1, "OFAT 运行数 = 1 + Σ(kᵢ−1)")
    eq(len(ofat.cells[0]), 3, "参考格点含全部轴")
    eq(ofat.reference["temp_C"], 600, "默认参考点取每轴首值")
    check(
        any("无法发现交互" in n for n in ofat.notes),
        "OFAT 必须说明它测不出交互效应（否则会被当成完整结论）",
        str(ofat.notes)[:120],
    )

    # --- 网格数量 = Πk ---
    grid = expand_sweep([T, X, H], mode="grid", max_runs=None)
    eq(grid.n_runs, 3 * 2 * 2, "网格运行数 = Πkᵢ")
    check(grid.orthogonal is True, "未截断的网格是正交的")
    check(not any("无法发现交互" in n for n in grid.notes), "网格不报 OFAT 的告警")

    # --- 截断护栏（核心）---
    cut = expand_sweep([T, X, H], mode="grid", max_runs=5)
    eq(cut.n_runs, 5, "截断到预算")
    eq(cut.dropped, 12 - 5, "如实报告丢了多少组")
    check(cut.truncated is True, "标记为已截断")
    check(
        cut.orthogonal is False,
        "**截断后必须标记为非正交**——否则主效应会与交互效应混淆，"
        "而结果看起来完全正常",
    )
    check(
        any("不再正交" in n for n in cut.notes),
        "截断时必须说明后果", str(cut.notes)[:140],
    )

    # --- 自定义参考点 ---
    ref = expand_sweep([T, X], mode="ofat", max_runs=None, reference={"temp_C": 700})
    eq(ref.reference["temp_C"], 700, "参考点可指定")
    eq(ref.reference["comp_x"], 0.0, "未指定的轴取首值")
    try:
        expand_sweep([T], mode="ofat", reference={"temp_C": 999})
        check("非法参考点被拒绝", False, "未抛异常")
    except ValueError:
        check("非法参考点被拒绝（不在取值内）", True)
    try:
        expand_sweep([T, X], mode="bogus")
        check("非法 mode 被拒绝", False, "未抛异常")
    except ValueError:
        check("非法 mode 被拒绝", True)

    # --- 效应分析：合成数据验证判据本身 ---
    section("多参数扫描：效应分析")
    A = SweepAxis("a", (0, 1, 2))
    B = SweepAxis("b", (0, 1))
    cells = expand_sweep([A, B], mode="grid", max_runs=None).cells
    # 可加模型：val = 10*a + 3*b，无交互
    add_obs = [(c, 10.0 * c["a"] + 3.0 * c["b"]) for c in cells]
    add_eff = rank_axes(axis_effects(add_obs, [A, B]))
    eq([e.axis for e in add_eff], ["a", "b"], "可加数据里效应排序正确")
    eq(round(add_eff[0].span, 6), 20.0, "轴 a 的效应幅度 = 10×2")
    eq(round(add_eff[1].span, 6), 3.0, "轴 b 的效应幅度 = 3×1")
    ratios = effect_ratio(add_eff)
    check(abs(ratios["b"] - 3.0 / 20.0) < 1e-9, "相对强度比例正确", str(ratios))

    add_int = interaction_effects(add_obs, [A, B])
    check(
        (not add_int) or add_int[0].strength < 1e-9,
        "可加数据的交互强度为 0（判据不会凭空造出交互）",
        str([i.to_dict() for i in add_int]),
    )

    # 有交互：b 会放大 a 的作用
    inter_obs = [(c, (10.0 * c["a"]) * (1.0 + 2.0 * c["b"])) for c in cells]
    inter_int = interaction_effects(inter_obs, [A, B])
    check(bool(inter_int), "能报出交互", str(inter_int))
    if inter_int:
        # b=0 时 a 的效应 = 20；b=1 时 = 60 -> 交互强度 = 40
        check(
            abs(inter_int[0].strength - 40.0) < 1e-6,
            "交互强度数值正确（b 放大 a 的作用）",
            f"{inter_int[0].strength}",
        )

    # --- OFAT 在有交互时会给出错误结论（这正是要告警的理由）---
    ofat_cells = expand_sweep([A, B], mode="ofat", max_runs=None).cells
    # 构造"只有 a=0,b=0 是坏的、其余都好"的交互，OFAT 从 (0,0) 出发
    tricky = [(c, 0.0 if (c["a"] == 0 and c["b"] == 0) else 100.0) for c in ofat_cells]
    tricky_eff = axis_effects(tricky, [A, B])
    check(
        bool(tricky_eff),
        "OFAT 仍会算出数值（所以必须靠 note 提示它测不出交互）",
        str([e.to_dict() for e in tricky_eff])[:150],
    )

    # --- 缺失值不参与统计 ---
    sparse = [({"a": 0}, 1.0), ({"a": 1}, float("nan")), ({"a": 2}, 3.0)]
    sp = axis_effects(sparse, [A])
    check(bool(sp), "部分缺失仍可算", str(sp))
    if sp:
        check(abs(sp[0].span - 2.0) < 1e-9, "nan 被忽略（跨度 = 3-1）", str(sp[0].span))


def test_sweep_integration() -> None:
    """扫描接进适配器协议与 s4：命令里必须有覆盖、参考格点必须有参数。"""
    section("多参数扫描：适配器与 s4 接线")
    from autoresearch.adapters.base import RunSpec
    from autoresearch.adapters.materials_optics import MaterialsOpticsAdapter
    from autoresearch.adapters.sweep import SweepAxis, expand_sweep
    from autoresearch.stages.s4_experiment import (
        _sweep_plan,
        _sweep_variant_names,
        _variant_params,
    )

    # 基类默认不扫描。BaseExperimentAdapter 是抽象类，用最小具体子类验证默认行为。
    from autoresearch.adapters.base import BaseExperimentAdapter

    class _Minimal(BaseExperimentAdapter):
        name = "minimal"

        def build_command(self, spec):  # pragma: no cover - 本测试不用
            return ["python", "x.py"]

        def parse_results(self, out_dir):  # pragma: no cover - 本测试不用
            return {}

    _min = _Minimal({})
    check(_min.sweep_axes() is None,
          "基类默认不做参数扫描（不改变既有适配器行为）")
    eq(_min.sweep_mode(), "ofat", "默认设计是 OFAT")

    # 材料适配器声明了 3 条轴，且带 target
    ad = MaterialsOpticsAdapter({})
    axes = ad.sweep_axes()
    check(bool(axes) and len(axes) == 3, "材料适配器声明 3 条扫描轴", str(axes))
    if axes:
        eq([a.name for a in axes], ["osc_g", "osc_f", "n_vis"], "轴名正确")
        check(
            all(a.target for a in axes),
            "**每条轴都显式声明 target**（绑定方式只有适配器知道，猜错会静默失效）",
            str([(a.name, a.target) for a in axes]),
        )
        check(
            any(":" in a.target for a in axes),
            "其中至少一条是函数默认参数绑定（必须用 函数名:参数名 形式）",
            str([a.target for a in axes]),
        )

    # 可关闭 / 可覆盖
    check(MaterialsOpticsAdapter({"no_sweep": True}).sweep_axes() is None,
          "no_sweep 可关闭扫描")
    custom = MaterialsOpticsAdapter({"sweeps": "osc_g:0.3|0.5|0.7"}).sweep_axes()
    check(bool(custom) and len(custom) == 1 and custom[0].levels == 3,
          "sweeps 可自定义轴与取值", str(custom))
    check(custom is not None and custom[0].target == "eps_lorentz:g",
          "自定义轴继承已知轴的 target（否则又会静默失效）",
          str(custom[0].target if custom else None))

    # _sweep_plan / 变体命名
    plan = _sweep_plan(ad, 12)
    check(plan is not None, "_sweep_plan 对声明了轴的适配器返回方案")
    names = _sweep_variant_names(plan) if plan else []
    eq(names[0], "baseline", "参考格点命名为 baseline")
    check(all(n.startswith("sw-") for n in names[1:]), "其余格点是 sw-<n>", str(names))

    # **参考格点必须拿到参数**，否则对照变成"默认值 vs 被改过的值"
    base_params = _variant_params({}, "baseline", ad, 12)
    check(bool(base_params) and len(base_params) == 3,
          "参考格点拿到参考条件的参数（不是空 dict）", str(base_params))
    sw1 = _variant_params({}, "sw-1", ad, 12)
    check(bool(sw1) and sw1 != base_params,
          "其余格点拿到与参考格点不同的参数", f"{base_params} vs {sw1}")

    # **覆盖必须进命令行**——否则所有格点算出同一组数值，
    # 扫描退化成重复实验，效应全为 0，然后被误读成"参数都不重要"
    argv = ad.build_command(
        RunSpec(variant="sw-1", seed=0, out_dir="runs/sw-1/seed_0", params=sw1)
    )
    eq(sum(1 for a in argv if a == "--set"), len(sw1), "每个参数都变成一次 --set")
    joined = " ".join(str(a) for a in argv)
    check("eps_lorentz:g=" in joined or "eps_lorentz:f=" in joined or "N_VIS=" in joined,
          "命令里用的是声明过的 target 形式", joined[-120:])
    check("--epochs" not in argv, "扫描命令里没有 ML 概念", joined)

    # 未声明轴的 adapter 不受影响
    from autoresearch.stages.s4_experiment import _plan_variants

    eq(_plan_variants({}, max_variants=2, adapter=_Minimal({})),
       ["baseline", "method"], "未声明轴的适配器仍走原有变体逻辑")

def main() -> int:
    print("=" * 70)
    print("实验后端适配器测试（离线）")
    print("=" * 70)

    _progress('test_runspec')
    try:
        test_runspec()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_runspec 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_metric_series_contract')
    try:
        test_metric_series_contract()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_metric_series_contract 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_standard_metrics_reader')
    try:
        test_standard_metrics_reader()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_standard_metrics_reader 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_resolve_and_inspect')
    try:
        test_resolve_and_inspect()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_resolve_and_inspect 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_synthetic_adapter')
    try:
        test_synthetic_adapter()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_synthetic_adapter 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_script_wrapper_adapter')
    try:
        test_script_wrapper_adapter()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_script_wrapper_adapter 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_sandbox_run_command')
    try:
        test_sandbox_run_command()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_sandbox_run_command 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_variant_derivation')
    try:
        test_variant_derivation()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_variant_derivation 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_comparison_robustness')
    try:
        test_comparison_robustness()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_comparison_robustness 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_s4_uses_adapter')
    try:
        test_s4_uses_adapter()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_s4_uses_adapter 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_adapter_informs_codegen')
    try:
        test_adapter_informs_codegen()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_adapter_informs_codegen 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_variant_budget_respects_adapter_cap')
    try:
        test_variant_budget_respects_adapter_cap()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_variant_budget_respects_adapter_cap 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_metric_direction_is_unified')
    try:
        test_metric_direction_is_unified()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_metric_direction_is_unified 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_json_metrics_with_bom')
    try:
        test_json_metrics_with_bom()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_json_metrics_with_bom 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_no_unfilled_template_placeholders')
    try:
        test_no_unfilled_template_placeholders()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_no_unfilled_template_placeholders 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_codegen_prompt_defers_to_adapter')
    try:
        test_codegen_prompt_defers_to_adapter()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_codegen_prompt_defers_to_adapter 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_section_prompt_has_no_phantom_fields')
    try:
        test_section_prompt_has_no_phantom_fields()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_section_prompt_has_no_phantom_fields 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_keywords_survive_review_loop')
    try:
        test_keywords_survive_review_loop()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_keywords_survive_review_loop 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_real_experiment_adapter_real_script')
    try:
        test_real_experiment_adapter_real_script()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_real_experiment_adapter_real_script 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_lorenz_adapter_runs_for_real')
    try:
        test_lorenz_adapter_runs_for_real()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_lorenz_adapter_runs_for_real 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_metric_direction_declaration')
    try:
        test_metric_direction_declaration()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_metric_direction_declaration 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_materials_adapter_contract')
    try:
        test_materials_adapter_contract()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_materials_adapter_contract 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_comsol_adapter_encoding_and_contract')
    try:
        test_comsol_adapter_encoding_and_contract()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_comsol_adapter_encoding_and_contract 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_parameter_sweep')
    try:
        test_parameter_sweep()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_parameter_sweep 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    _progress('test_sweep_integration')
    try:
        test_sweep_integration()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_sweep_integration 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"FAILED {len(_FAILURES)} of {_CHECKS} checks")
        for failure in _FAILURES[:40]:
            print("  - " + failure)
        return 1
    print(f"PASSED {_CHECKS} checks")
    return 0


def test_adapter_suite() -> None:
    """pytest 收集入口。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
