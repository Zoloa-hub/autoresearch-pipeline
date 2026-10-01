"""把材料领域的发现写进 CHANGELOG，并记录指标方向声明这个架构改动。"""

from __future__ import annotations

import pathlib

p = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"
t = p.read_text(encoding="utf-8")

NEW_SECTION = '''### 🧪 非 ML 领域验证：材料光学（第一个真实案例）

用一份**真实的材料学脚本**做了领域通用性实测：`ltp_optics.py`
（La₂Ti₂O₇ 薄膜光学常数，Kramers-Kronig 一致的 n,k 色散；文献锚点
Bayart et al., Optical Materials 92 (2019)）。**脚本未做任何修改。**

新增 `autoresearch/adapters/materials_optics.py`，实测结果：

```
加载（module:Class + .py 路径）        OK
validate_environment（检查 nk_table）  OK
命令形态：含 --replicates、不含 --epochs OK   <- 非 ML 领域的关键差异
沙箱执行真实模块                       rc=0
parse_results                          5 指标 × 4 次重复测量
metric_axis()                          'replicate'（不是 epoch）
```

**这个实测量化了一个真实的架构缺口。**

#### 缺口：指标方向靠"从名字猜"，在非 ML 领域会判反

管线原本用一张通用 token 表猜方向，而那张表是 ML 中心的
（loss/accuracy/f1/bleu…）。在 19 个材料学常见指标上实测：

| 判定 | 数量 | 例子 |
|---|---|---|
| **方向判反** | **5** | `k`（消光系数，应越小越好）、`alpha`、`resistivity`、`corrosion_rate`、`sintering_temp` |
| **无固有方向却被静默选了一个** | **6** | `n`、`reflectance`、`band_gap`、`youngs_modulus`、`thermal_conductivity` |
| 判对 | 8 | `transmittance`、`hardness`、`d33`、`zt`… |

即 **13/19 不可信**。而方向判错**不会报错**——它只会让「改善」的定义反过来。

在材料适配器的 5 个指标上，通用词表**判反 4 个**，并导致对照结论翻转：

```
不声明方向: supports_claim=False  improved=['transparent_window_nm']
声明方向  : supports_claim=True   improved=['alpha_visible_mean','k_at_250nm',
                                            'k_at_550nm','n_deviation_1e3',
                                            'transparent_window_nm']
```

同一份数据、同一组数字，**结论从"不支持 claim"变成"支持 claim"**。

#### 修法：领域知识归适配器，管线不猜

1. `BaseExperimentAdapter.metric_directions() -> dict[str, bool] | None`
   —— 适配器**声明** `{指标名: 越大越好?}`。返回 `None`=交给通用启发式；
   返回 `{}`=已声明但无需方向判断（两者语义不同）。
2. `BaseExperimentAdapter.metric_axis() -> str | None`
   —— 声明序列的物理轴（ML 是 `epoch`，材料是 `replicate`）。
   默认 `None`：**管线不假设序列有物理含义**。
3. 方向判定优先级：**适配器声明 → 调用方覆盖 → 通用词表（兜底）**。
4. `s4._compare(directions=...)` 接线；适配器声明了方向却没覆盖到的指标
   会记一条 warning —— 让"未声明的方向"可见，而不是静默走兜底猜。

#### 另一处发现：序列的语义在不同领域不同

ML 里序列轴是 `epoch`（"训练轨迹"），材料里是**重复测量**（"不确定度"）。
把不同波长的 n 求平均**在物理上没有意义**，所以材料适配器把代表性条件
（550nm / 250nm / 可见区均值）固定住，在**重复轴**上取序列——
这样 `mean±std` 才是材料学里成立的口径。这也说明"序列必须有顺序"是个
领域假设，不该写进契约。

#### 仍未验证（材料方向）

- 只覆盖**光学常数**这一类表征。力学/热学/电学性质、XRD、显微、DSC 等未验证。
- 材料学常见的**参数扫描**（温度/成分/退火时间）尚未接成变体轴。
- 扰动幅度 `jitter` 是人为设定的，真实不确定度需要来自实验重复。
- COMSOL 批处理输出**未接入**。顺带记录一个真实痛点：COMSOL 的中文报错是
  GBK 编码，在本机的 UTF-8 日志里显示为乱码，排错时需要先转码。

'''

anchor = "### ⚠️ 仍未验证"
if "非 ML 领域验证：材料光学" not in t:
    t = t.replace(anchor, NEW_SECTION + anchor, 1)
    p.write_text(t, encoding="utf-8")
    print("  CHANGELOG: 已加材料领域验证段")
else:
    print("  CHANGELOG: 已有该段")
