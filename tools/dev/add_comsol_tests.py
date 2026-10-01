"""注册 COMSOL 适配器并加回归测试（含编码损坏这个非 ML 专属问题）。"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ------------------------------------------------ 1) 注册 --
ini = ROOT / "autoresearch" / "adapters" / "__init__.py"
s = ini.read_text(encoding="utf-8")
if "ComsolBatchAdapter" not in s:
    s = s.replace(
        "from .lorenz_governance import LorenzGovernanceAdapter",
        "from .comsol_batch import ComsolBatchAdapter\nfrom .lorenz_governance import LorenzGovernanceAdapter",
        1,
    )
    s = s.replace('    "LorenzGovernanceAdapter",',
                  '    "ComsolBatchAdapter",\n    "LorenzGovernanceAdapter",', 1)
    ini.write_text(s, encoding="utf-8")
    print("  __init__.py: 已注册 ComsolBatchAdapter")

# ------------------------------------------------ 2) 测试 --
TA = ROOT / "autoresearch" / "tests" / "test_adapters.py"
ta = TA.read_text(encoding="utf-8")

NEW = '''

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
    # 构造一个"真实形态"的日志：UTF-16LE + BOM，含失败步骤与损坏字符
    body = (
        "set epsilonr -> 4.84\\n"
        "std.run: FAILED -> com.comsol.util.exceptions.FlException: Exception:\\n"
        "\\tFlException: \\ufffd\\ufffd\\ufffd\\ufffd\\n"
        "S11: FAILED -> FlException: \\ufffd\\ufffd\\n"
        "OK   WaveEquationElectric -> type=WaveEquationElectric\\n"
        "OK   CrossSectionCalculation -> type=CrossSectionCalculation\\n"
    )
    log = work / "probe.console.txt"
    log.write_bytes(body.encode("utf-16"))

    text, enc = read_comsol_log(log)
    check(enc in ("utf-16", "utf-16-le"), "按 BOM 识别出 UTF-16", enc)
    check("FAILED" in text, "内容解码正确（含 FAILED）")
    check("WaveEquationElectric" in text, "英文标识符完整保留")

    ratio = corruption_ratio(text)
    check(0.5 < ratio < 1.0, "损坏比例被如实算出", f"{ratio:.3f}")
    check(corruption_ratio("全部是 ASCII") == 0.0, "纯 ASCII 的损坏率为 0")
    check(corruption_ratio("\\ufffd\\ufffd") == 1.0, "全损坏时为 1.0")

    # 一个"干净"的日志损坏率为 0
    clean = work / "clean.console.txt"
    clean.write_bytes("std.run: OK\\nOK   A -> type=X\\n".encode("utf-16"))
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
'''

anchor = "\ndef main() -> int:"
if "def test_comsol_adapter_encoding_and_contract(" not in ta:
    ta = ta.replace(anchor, NEW + anchor, 1)
    # 注册到 main（带崩溃隔离）
    reg = """    _progress('test_comsol_adapter_encoding_and_contract')
    try:
        test_comsol_adapter_encoding_and_contract()
    except Exception as _exc:  # noqa: BLE001
        import traceback as _tb
        _FAILURES.append(f"test_comsol_adapter_encoding_and_contract 崩溃: {type(_exc).__name__}: {_exc}")
        _tb.print_exc()

    print("\\n" + "=" * 70)"""
    ta = ta.replace('    print("\\n" + "=" * 70)', reg, 1)
    TA.write_text(ta, encoding="utf-8")
    print("  已加 COMSOL 测试并注册")
else:
    print("  已存在")

import py_compile  # noqa: E402

py_compile.compile(str(ini), doraise=True)
py_compile.compile(str(TA), doraise=True)
print("  编译通过")
