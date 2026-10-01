"""收尾材料适配器：修 docstring 警告 + 加方向声明协议的回归测试。

方向声明是本轮最重要的架构改动，必须被测试钉住：
**同一份数据下，声明与否会翻转对照结论**——这正是非 ML 领域会出错的点。
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------- 1) raw docstring --
p = ROOT / "autoresearch" / "adapters" / "materials_optics.py"
t = p.read_text(encoding="utf-8")
if not t.startswith('r"""'):
    t = t.replace('"""材料光学适配器', 'r"""材料光学适配器', 1)
    p.write_text(t, encoding="utf-8")
    print("  materials_optics.py: docstring 改为 raw（消除 \\| 转义警告）")

# ---------------------------------------------------- 2) 回归测试 --
TA = ROOT / "autoresearch" / "tests" / "test_adapters.py"
ta = TA.read_text(encoding="utf-8")

NEW_TESTS = '''

def test_metric_direction_declaration() -> None:
    """适配器**声明**指标方向，而不是让管线从名字猜。

    这是非 ML 领域（材料/化学/临床…）能否复用的关键。管线的通用词表是 ML 中心的
    （loss/accuracy/f1/bleu…），在材料学实测 19 个指标里 **5 个判反、6 个无方向
    却被静默选了一个**，合计 13/19 不可信。

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
    real = Path(r"C:\\Users\\user\\Desktop\\Phd-foundations")
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
'''

anchor = "\ndef main() -> int:"
if "def test_metric_direction_declaration(" not in ta:
    ta = ta.replace(anchor, NEW_TESTS + anchor, 1)
    # ROOT 与 Path 需要可用
    if "\nROOT = " not in ta:
        ta = ta.replace(
            "from pathlib import Path",
            "from pathlib import Path\n\nROOT = Path(__file__).resolve().parents[2]",
            1,
        )
    ta = ta.replace(
        "    test_lorenz_adapter_runs_for_real()\n",
        "    test_lorenz_adapter_runs_for_real()\n"
        "    test_metric_direction_declaration()\n"
        "    test_materials_adapter_contract()\n",
        1,
    )
    TA.write_text(ta, encoding="utf-8")
    print("  已加 2 个回归测试并注册")
else:
    print("  已存在")

import py_compile  # noqa: E402

py_compile.compile(str(p), doraise=True)
py_compile.compile(str(TA), doraise=True)
print("  编译通过")
