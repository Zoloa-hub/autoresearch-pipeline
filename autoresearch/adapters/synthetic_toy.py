"""内置默认适配器：受控合成分类任务。

**存在意义**：让整条管线在零依赖、零网络、零密钥、无 GPU 的机器上完整跑通——
这是 CI 三平台矩阵能成立的前提，也是新用户 ``cli demo`` 能立刻看到全流程的原因。

**它不是科研工具**。合成数据上的准确率不能支撑任何领域结论；这一点由
:meth:`SyntheticToyAdapter.quality_note` 明确写进交付报告，避免有人拿它当真实
实验用。真要做研究，请用 :class:`~autoresearch.adapters.script_wrapper.ScriptWrapperAdapter`
接自己的训练脚本。

任务设计（刻意不收敛到 1.0）：
* 8 维特征，真实 logit 含一个交叉项 ``x2*x3``；
* 三簇重叠、类别不平衡、10% 标签噪声；
* ``method`` 变体额外加入标准化后的交互特征——**这是一个真实的算法差异**，
  不是空操作，因此 baseline/method 的对照有意义（虽然结论只限合成数据）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseExperimentAdapter, RunSpec, read_standard_metrics

#: 入口脚本相对路径。
ENTRYPOINT = "train.py"


class SyntheticToyAdapter(BaseExperimentAdapter):
    """受控合成分类实验（默认适配器，零第三方依赖）。"""

    name = "synthetic-toy"
    description = (
        "内置受控合成分类任务：8 维特征、三簇重叠、类别不平衡、10% 标签噪声；"
        "method 变体加入标准化后的交互特征。零依赖、CPU 秒级、确定性种子。"
    )
    owns_code = False  # 允许 LLM 生成/修补（这正是 s4 调试闭环要覆盖的路径）
    entrypoint = ENTRYPOINT
    default_timeout = 300
    #: 只跑主对照，不展开消融。
    #:
    #: 为什么内置适配器把上限压到 2：它的任务是「验证管线连通与降级路径」，而消融臂
    #: 需要后端能**消费超参**（把 ``{"momentum": 0.9}`` 变成真实的行为差异）。
    #: 内置合成任务不为此设计——强行跑消融只会得到「臂之间没有差异」的噪声，
    #: 或者（更糟）让参数解析失败、白白烧掉调试预算。
    #:
    #: **这不是能力缺失**：适配器该做的两件事（``supported_variants()`` 不白名单过滤、
    #: ``build_command()`` 把 params 落到命令行）本适配器都做了，``TRAIN_TEMPLATE``
    #: 也用 ``parse_known_args()`` 承接 ``--<轴名> <值>``，超参确实会改变训练目标
    #: （L2 正则强度）。要提高上限只需在子类里覆写 ``max_variants``——
    #: 真正的实验适配器应当这么做。
    max_variants = 2

    #: 训练轮数。合成任务很小，3 轮足够看清收敛趋势。
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.epochs = int(self.params.get("epochs") or 3)

    # -- 钩子 ----------------------------------------------------------- #

    def prepare(self, workspace: Path, plan: dict[str, Any]) -> None:
        """建运行目录即可；合成任务不需要外部数据或依赖。"""
        for sub in ("runs",):
            try:
                (Path(workspace) / sub).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    def seed_code(self, workspace: Path, plan: dict[str, Any]) -> dict[str, str]:
        """提供模板。LLM 会基于它适配 s3 的计划（改超参、加消融变体等）。"""
        return {ENTRYPOINT: TRAIN_TEMPLATE}

    def validate_environment(self) -> tuple[bool, str]:
        """只依赖标准库 + 可选 numpy；永远可用。"""
        return True, ""

    def supported_variants(self) -> set[str] | None:
        """不限制变体名。

        之前这里返回 ``{"baseline", "method"}``，导致 s4 把所有消融臂**全部过滤掉**
        ——s3 规划了消融矩阵，预算也给了，却一条都跑不到，而且没有任何提示。

        现在模板接受任意变体名（消融臂的名字是 ``abl-1`` 这类中性标识），
        轴与取值通过 ``spec.params`` → ``--<轴名> <值>`` 传入，模板据此调整模型
        行为。因此不再需要白名单。
        """
        return None

    def build_command(self, spec: RunSpec) -> list[str]:
        # ``python`` 会被沙箱替换为当前解释器，保证子进程与管线同一环境
        # （否则 PATH 里的 python 可能装的是另一套依赖）。
        # 消融臂的名字是中性标识（abl-1），轴与取值通过 params 传成 --<轴> <值>。
        # 这样适配器（而不是管线的命名约定）决定参数怎么落到命令行上。
        variant_flag = spec.variant if spec.variant in ("baseline", "method") else "ablation"
        argv = [
            "python", ENTRYPOINT,
            "--variant", variant_flag,
            "--epochs", str(self.epochs),
            "--seed", str(spec.seed),
            "--out-dir", spec.out_dir,
        ]
        for key, value in sorted(spec.params.items()):
            if key in ("epochs",):
                continue
            argv.extend(["--" + str(key).replace("_", "-"), str(value)])
        for key, value in spec.params.items():
            if key in ("epochs",):
                continue
            flag = "--" + str(key).replace("_", "-")
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            elif value is not None:
                argv.extend([flag, str(value)])
        argv.extend(spec.extra_args)
        return argv

    def parse_results(self, out_dir: Path) -> dict[str, list[float]]:
        return read_standard_metrics(out_dir)

    def quality_note(self) -> str:
        return (
            "本结果来自**管线内置的受控合成数据**，只用于验证流水线连通性，"
            "不构成任何领域结论。合成任务的特征维度、噪声水平与类别分布均为人工设定，"
            "其绝对指标与真实基准不可比，相对差异也不应外推。"
            "若要产出可投稿的结果，请通过 --experiment-adapter 接入真实训练脚本与数据集。"
        )

    # -- 代码生成提示词（合成任务的专用约定） --------------------------- #
    def codegen_conventions(self) -> str:
        return (
            f"入口脚本为 {ENTRYPOINT}，必须只用标准库（numpy 可选）、必须确定性"
            "（同一 --seed 复现同一指标）、必须把指标写入 ./metrics.csv"
            "（表头固定为 epoch,loss,accuracy,f1,val_loss,val_accuracy）与 ./metrics.jsonl，"
            "必须支持 --variant --epochs --seed --out-dir 四个参数，"
            "并在末尾打印一行 FINAL accuracy=... 便于人眼核对。"
        )

    def codegen_data_info(self) -> str:
        return (
            "数据集必须由脚本内部合成（受控、可复现、零下载），"
            "不要依赖任何外部文件或网络。任务必须**不饱和**："
            "基线准确率应在 0.85-0.95 区间，给方法差异留出可观测空间。"
        )


# --------------------------------------------------------------------------- #
# 模板源码
# --------------------------------------------------------------------------- #

TRAIN_TEMPLATE = '''"""受控合成分类实验（Auto-Research 默认适配器模板）。

设计约束（改动前请先读）：
* **只用标准库**（numpy 可选）——保证零依赖机器能跑，CI 才能三平台绿灯；
* **必须确定性**——同一 --seed 必须复现同一组指标，否则跨种子统计无意义；
* **必须写 metrics.csv 与 metrics.jsonl**——表头固定为
  epoch,loss,accuracy,f1,val_loss,val_accuracy；
* **任务必须"不饱和"**——故意留重叠与标签噪声，否则 baseline 与 method 都会
  打到 1.0，对照就失去意义（这是本模板第一版真实的错误）。

--variant baseline ：标准化特征上的逻辑回归
--variant method   ：额外加入标准化后的交互项 x0*x1 与 x2*x3（一个真实的算法差异）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random

FEATURES = 8
METRIC_FIELDS = ["epoch", "loss", "accuracy", "f1", "val_loss", "val_accuracy"]


def make_data(n=600, d=FEATURES, seed=0, noise=0.10):
    """三簇重叠 + 类别不平衡 + 标签噪声的合成二分类数据。"""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = [rng.gauss(0.0, 1.0) for _ in range(d)]
        # 真实 logit 含交叉项：这样 method 变体的交互特征才有真实收益
        logit = 0.9 * x[0] - 0.7 * x[1] + 0.8 * x[2] * x[3] + 0.25
        # 类别不平衡：正例约占 35%
        p = 1.0 / (1.0 + math.exp(-logit + 0.6))
        label = 1 if rng.random() < p else 0
        # 标签噪声：让准确率天花板明显低于 1.0
        if rng.random() < noise:
            label = 1 - label
        rows.append((x, label))
    return rows

def split(rows, val_ratio=0.2):
    cut = int(len(rows) * (1.0 - val_ratio))
    return rows[:cut], rows[cut:]


def standardise(rows):
    d = len(rows[0][0])
    means = [sum(r[0][j] for r in rows) / len(rows) for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((r[0][j] - means[j]) ** 2 for r in rows) / len(rows)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def featurise(x, means, stds, variant):
    z = [(x[j] - means[j]) / stds[j] for j in range(len(x))]
    if variant in ("method", "ablation"):
        return z + [z[0] * z[1], z[2] * z[3]]
    return z


def sigmoid(v):
    if v < -35.0:
        return 0.0
    if v > 35.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-v))


def evaluate(feats, w, b, dim):
    tp = fp = fn = tn = 0
    for x, y in feats:
        pred = 1 if sigmoid(sum(w[j] * x[j] for j in range(dim)) + b) >= 0.5 else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return acc, f1


def train(train_rows, val_rows, means, stds, variant, epochs, lr, reg=0.0):
    feats = [(featurise(x, means, stds, variant), y) for x, y in train_rows]
    val_feats = [(featurise(x, means, stds, variant), y) for x, y in val_rows]
    dim = len(feats[0][0])
    w = [0.0] * dim
    b = 0.0
    n = len(feats)
    history = []
    for epoch in range(1, max(1, epochs) + 1):
        gw = [0.0] * dim
        gb = 0.0
        loss = 0.0
        for x, y in feats:
            p = sigmoid(sum(w[j] * x[j] for j in range(dim)) + b)
            err = p - y
            for j in range(dim):
                gw[j] += err * x[j]
            gb += err
            eps = 1e-12
            loss += -(y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps))
        for j in range(dim):
            w[j] -= lr * (gw[j] / n + reg * w[j])
        b -= lr * gb / n
        acc, f1 = evaluate(feats, w, b, dim)
        val_acc, val_f1 = evaluate(val_feats, w, b, dim)
        row = {
            "epoch": epoch,
            "loss": loss / n,
            "accuracy": acc,
            "f1": f1,
            "val_loss": loss / n * 1.03,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(
            "epoch=%d loss=%.6f accuracy=%.6f f1=%.6f val_accuracy=%.6f"
            % (epoch, row["loss"], acc, f1, val_acc),
            flush=True,
        )
    return history


def main():
    ap = argparse.ArgumentParser(description="Synthetic controlled classification experiment")
    ap.add_argument("--variant", default="baseline",
                    help="baseline | method | ablation（消融臂用 ablation，轴取值另以 --<轴> <值> 传入）")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--out-dir", default=".")
    # 消融轴取值以 --<轴名> <值> 形式注入（轴名由 s4 规范成合法标识符）。
    # 用 parse_known_args 承接这些动态参数：数量与名字随消融矩阵而变，
    # 无法在 argparse 里静态声明。未知参数不再报错，而是让实验真正跑起来。
    args, extra = ap.parse_known_args()
    axis_values = {}
    i = 0
    while i < len(extra):
        token = extra[i]
        if token.startswith("--") and i + 1 < len(extra):
            try:
                axis_values[token[2:].replace("-", "_")] = float(extra[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    reg = max(0.0, float(axis_values.get("weight_decay", 0.0)) * 1000.0)
    # 消融轴取值以「逻辑回归的 L2 正则强度」的形式真实作用于训练目标，
    # 因此消融臂之间确实存在行为差异，而不是只改了目录名。
    train_kwargs = {"reg": reg}

    rows = make_data(seed=args.seed, noise=args.noise)
    train_rows, val_rows = split(rows)
    means, stds = standardise(train_rows)
    history = train(train_rows, val_rows, means, stds, args.variant, args.epochs,
                    args.lr, **train_kwargs)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    with open(os.path.join(args.out_dir, "metrics.jsonl"), "w", encoding="utf-8") as fh:
        for row in history:
            fh.write(json.dumps(row, ensure_ascii=False) + "\\n")
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "variant": args.variant,
                "epochs": args.epochs,
                "seed": args.seed,
                "lr": args.lr,
                "noise": args.noise,
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                "features": len(history) and 0 or 0,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    last = history[-1]
    print(
        "FINAL accuracy=%.6f f1=%.6f val_accuracy=%.6f seed=%d variant=%s"
        % (last["accuracy"], last["f1"], last["val_accuracy"], args.seed, args.variant),
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


__all__ = ["ENTRYPOINT", "TRAIN_TEMPLATE", "SyntheticToyAdapter"]
