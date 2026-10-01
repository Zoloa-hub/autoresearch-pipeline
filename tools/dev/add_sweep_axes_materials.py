"""给 SweepAxis 加显式 target，并让材料适配器声明真实扫描轴 + 覆盖机制。

## 为什么 target 必须显式

实测用户的 ``ltp_optics.py``：
  * ``N_VIS`` 在 ``nk_table`` **函数体内**被读取 → ``setattr`` 生效
  * ``eps_lorentz(ev, f=OSC_F, e0=OSC_E0, g=OSC_G)`` 的默认参数在**定义时**就绑定
    → ``setattr(mod, 'OSC_G', ...)`` **完全无效**（实测确认）
  * ``CAUCHY_A`` 等是死常量 —— ``cauchy_n()`` 从未被 ``nk_table`` 调用

所以"改模块常量"这条路**不可靠，而且失败是静默的**：所有格点得到同一组数值，
扫描退化成"同一实验跑 N 次"，效应全为 0，然后被解读成
「这些参数都不重要」——一个看起来正常、实际完全错误的结论。

结论：**覆盖方式必须由适配器显式声明**（只有适配器知道被驱动模块怎么绑定参数），
并在运行后**验证覆盖真的生效**。声明格式：

  * ``target="N_VIS"``            → 普通模块属性，``setattr``
  * ``target="eps_lorentz:g"``    → 重建该函数、把参数 ``g`` 的默认值换掉，再替换回模块
"""

from __future__ import annotations

import pathlib

# ---------------------------------------------------------------- 1) SweepAxis.target --
SW = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "adapters" / "sweep.py"
s = SW.read_text(encoding="utf-8")

old = '''    name: str
    values: tuple[Any, ...]
    unit: str = ""

    def __post_init__(self) -> None:'''
new = '''    name: str
    values: tuple[Any, ...]
    unit: str = ""
    #: 该参数在**被驱动模块**里的绑定方式。空 = 与 ``name`` 同名的模块属性。
    #:
    #: 格式：``"N_VIS"``（普通属性，直接 setattr）或
    #: ``"eps_lorentz:g"``（该参数是函数 ``eps_lorentz`` 的默认参数——
    #: 这种绑定在函数**定义时**就固定了，setattr 模块常量无效，必须重建函数）。
    #:
    #: 之所以必须显式：绑定方式只有适配器知道，而**猜错的失败是静默的**——
    #: 所有格点算出同一组数值，扫描退化成重复实验，效应全为 0，
    #: 然后被解读成「这些参数都不重要」。
    target: str = ""

    def __post_init__(self) -> None:'''
if old in s:
    s = s.replace(old, new, 1)
    print("  sweep.py: SweepAxis 已加 target")

# to_dict 带上 target
s = s.replace(
    '''        return {
            "name": self.name,
            "values": list(self.values),
            "unit": self.unit,
            "levels": self.levels,
        }''',
    '''        return {
            "name": self.name,
            "values": list(self.values),
            "unit": self.unit,
            "levels": self.levels,
            "target": self.target or self.name,
        }''',
    1,
)
SW.write_text(s, encoding="utf-8")

# ---------------------------------------------------------------- 2) 材料适配器 --
MO = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "adapters" / "materials_optics.py"
m = MO.read_text(encoding="utf-8")

# 2a) 声明扫描轴
DIRECTIONS_ANCHOR = "#: 计算这些指标需要材料模块提供什么"
SWEEP_DECL = '''#: 默认扫描轴 —— 深紫外吸收边的三个物理参数。
#:
#: ``target`` 是**实测出来的**，不是猜的：``ltp_optics.py`` 把 ``OSC_F/OSC_E0/OSC_G``
#: 绑成 ``eps_lorentz`` 的默认参数，直接 ``setattr`` 模块常量**不会生效**
#: （实测确认），必须重建该函数。而 ``N_VIS`` 在 ``nk_table`` 函数体内被读取，
#: 直接 setattr 即可。
DEFAULT_SWEEP_AXES = (
    SweepAxis("osc_g", (0.35, 0.55, 0.75), "eV", target="eps_lorentz:g"),
    SweepAxis("osc_f", (3.30, 4.10, 4.90), "eV^2", target="eps_lorentz:f"),
    SweepAxis("n_vis", (2.10, 2.20, 2.30), "", target="N_VIS"),
)

''' + DIRECTIONS_ANCHOR

if "DEFAULT_SWEEP_AXES" not in m:
    m = m.replace(DIRECTIONS_ANCHOR, SWEEP_DECL, 1)
    print("  materials_optics.py: 已声明 DEFAULT_SWEEP_AXES")

# 2b) import SweepAxis
m = m.replace(
    "from typing import Any\n",
    "from typing import Any\n\nfrom .sweep import SweepAxis\n",
    1,
)
if "from .sweep import SweepAxis" not in m:
    m = m.replace(
        "from typing import Any  # noqa: F401\n",
        "from typing import Any  # noqa: F401\nfrom .sweep import SweepAxis\n",
        1,
    )

# 2c) 适配器开放扫描
if "def sweep_axes(" not in m:
    m = m.replace(
        "    # -- 领域知识：方向与轴 ---------------------------------------------- #",
        '''    # -- 参数扫描 ------------------------------------------------------- #
    #: 一次扫描最多跑多少格点（可由 AUTORESEARCH_MAX_SWEEP_RUNS 与配置收紧）
    max_sweep_runs = 12
    #: 默认 OFAT —— 3 条轴各 3 水平 => 1+2+2+2 = 7 组。
    #: 改 "grid" 是 27 组（3³），能测交互效应但要确认预算。
    default_sweep_mode = "ofat"

    def sweep_axes(self) -> tuple[SweepAxis, ...] | None:
        """默认扫描深紫外吸收边的三条参数轴。

        可用 ``adapter-arg sweeps=osc_g:0.3|0.5|0.7;n_vis:2.1|2.3`` 覆盖，
        或 ``sweep_mode=grid`` 切换设计。
        """
        raw = self.params.get("sweeps")
        if raw:
            parsed = _parse_sweep_spec(str(raw))
            if parsed:
                return parsed
        if self.params.get("no_sweep"):
            return None
        return DEFAULT_SWEEP_AXES

    def sweep_mode(self) -> str:
        mode = str(self.params.get("sweep_mode") or self.default_sweep_mode)
        return mode if mode in ("ofat", "grid") else self.default_sweep_mode

    # -- 领域知识：方向与轴 ---------------------------------------------- #''',
        1,
    )
    print("  materials_optics.py: 已开放 sweep_axes / sweep_mode")

# 2d) build_command 传 --set
old_cmd = '''            "--out-dir",
            spec.out_dir,
            *[str(a) for a in spec.extra_args],
        ]'''
new_cmd = '''            "--out-dir",
            spec.out_dir,
        ]
        # 把扫描格点的参数交给 runner。**用适配器声明的 target**，
        # 而不是假设"参数名就是模块属性名"——见 DEFAULT_SWEEP_AXES 的说明。
        for axis in (self.sweep_axes() or ()):
            if axis.name in (spec.params or {}):
                argv += ["--set", f"{axis.target or axis.name}={spec.params[axis.name]}"]
        argv.extend(str(a) for a in spec.extra_args)
        return argv'''
if old_cmd in m:
    m = m.replace(old_cmd, new_cmd, 1)
    print("  materials_optics.py: build_command 已传 --set")

MO.write_text(m, encoding="utf-8")

# 2e) 解析 sweeps 规格的辅助
if "_parse_sweep_spec" not in m:
    helper = '''

def _parse_sweep_spec(raw: str) -> tuple[SweepAxis, ...]:
    """解析 ``"osc_g:0.3|0.5|0.7;n_vis:2.1|2.3"`` 形式的扫描规格。"""
    axes: list[SweepAxis] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, values_raw = chunk.split(":", 1)
        values: list[Any] = []
        for token in values_raw.split("|"):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token) if ("." in token or "e" in token.lower()) else int(token))
            except ValueError:
                values.append(token)
        if len(values) < 2:
            continue
        known = {a.name: a for a in DEFAULT_SWEEP_AXES}
        src = known.get(name.strip())
        axes.append(
            SweepAxis(
                name.strip(),
                tuple(values),
                src.unit if src else "",
                target=src.target if src else name.strip(),
            )
        )
    return tuple(axes)
'''
    m = m + helper
    MO.write_text(m, encoding="utf-8")
    print("  已加 _parse_sweep_spec")

import py_compile  # noqa: E402

py_compile.compile(str(SW), doraise=True)
py_compile.compile(str(MO), doraise=True)
print("  编译通过")
