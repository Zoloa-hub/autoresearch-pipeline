#!/usr/bin/env python3
"""纯标准库的合成训练脚本 —— `ScriptWrapperAdapter` 的测试夹具。

## 为什么需要它

`test_script_wrapper_adapter` 最初拿 `templates/experiment/train.py` 当夹具，而那个
模板在模块级 `import numpy`。CI 的 `test` job 刻意**不安装任何第三方依赖**，
于是这个套件在所有平台都失败：

    File ".../_provided/train.py", line 35
        import numpy as np
    ModuleNotFoundError: No module named 'numpy'

两者都是"对的"，矛盾出在用错了夹具：
  * 模板是给人参考的**完整实现**，用 numpy 是合理选择；
  * 而套件必须能在裸环境跑，所以夹具本身不能有第三方依赖。

这个脚本就是对那件事的正面回答：**一个只用标准库、却完全满足适配器契约的训练脚本**。
它同时验证了一个更重要的性质——「零依赖脚本也能被 script-wrapper 驱动」。

## 契约（与 `templates/experiment/train.py` 完全一致）

* ``--variant {baseline,method}`` ``--epochs N`` ``--seed N`` ``--out-dir DIR``
* 写 ``<out-dir>/metrics.csv``，表头固定
  ``epoch,loss,accuracy,f1,val_loss,val_accuracy``
* 写 ``<out-dir>/metrics.jsonl``，每行一个 epoch 的同样六个键
* 末尾打印 ``FINAL accuracy=<x> seed=<s> variant=<v>``
* 同 ``--seed`` 必然复现同一串指标（无时间依赖、无未播种随机源）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

METRIC_FIELDS = ["epoch", "loss", "accuracy", "f1", "val_loss", "val_accuracy"]
N_CLASSES = 3
N_FEATURES = 4


def make_data(seed: int, n: int = 240):
    """确定性合成数据：三簇、有重叠、类别不平衡、带标签噪声。

    刻意**不饱和**：基线的验证准确率落在 0.7–0.9 区间，给 method 变体留出
    可观测的差异空间。一个一开始就 0.99 的夹具无法区分任何东西。
    """
    rng = random.Random(seed)
    centers = [[-1.6, -0.9, 0.4, 0.1], [1.5, -0.7, -0.3, 0.2], [0.1, 1.7, 0.2, -0.4]]
    weights = [1.0, 0.62, 0.4]  # 类别不平衡
    rows = []
    for cls in range(N_CLASSES):
        count = int(round(n * weights[cls] / sum(weights)))
        for _ in range(count):
            x = [centers[cls][j] + rng.gauss(0.0, 1.0) for j in range(N_FEATURES)]
            label = cls
            if rng.random() < 0.12:  # 12% 标签噪声：压低准确率天花板
                label = rng.randrange(N_CLASSES)
            rows.append((x, label))
    rng.shuffle(rows)
    cut = int(len(rows) * 0.8)
    return rows[:cut], rows[cut:]


def _softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _forward(w, b, x, use_interaction: bool):
    feats = list(x)
    if use_interaction:
        # method 变体：加入一个非线性交互项。这是它唯一的行为差异，
        # 因此两个变体之间的差距可以干净地归因到它。
        feats.append(x[0] * x[1])
        feats.append(math.tanh(x[2] - x[3]))
    logits = [sum(w[c][j] * feats[j] for j in range(len(feats))) + b[c] for c in range(N_CLASSES)]
    return logits, feats


def _evaluate(w, b, rows, use_interaction: bool):
    if not rows:
        return 0.0, 0.0, 0.0
    correct = 0
    loss = 0.0
    tp = [0] * N_CLASSES
    fp = [0] * N_CLASSES
    fn = [0] * N_CLASSES
    for x, y in rows:
        logits, _ = _forward(w, b, x, use_interaction)
        probs = _softmax(logits)
        pred = max(range(N_CLASSES), key=lambda c: probs[c])
        loss += -math.log(max(probs[y], 1e-12))
        if pred == y:
            correct += 1
        for c in range(N_CLASSES):
            if pred == c and y == c:
                tp[c] += 1
            elif pred == c and y != c:
                fp[c] += 1
            elif pred != c and y == c:
                fn[c] += 1
    n = len(rows)
    scores = []
    for c in range(N_CLASSES):
        denom = 2 * tp[c] + fp[c] + fn[c]
        if denom:
            scores.append(2 * tp[c] / denom)
    return loss / n, correct / n, (sum(scores) / len(scores) if scores else 0.0)


def train(variant: str, epochs: int, seed: int):
    train_rows, val_rows = make_data(seed)
    use_interaction = variant != "baseline"
    rng = random.Random(seed + 991)
    dim = N_FEATURES + (2 if use_interaction else 0)
    w = [[rng.gauss(0.0, 0.1) for _ in range(dim)] for _ in range(N_CLASSES)]
    b = [0.0] * N_CLASSES
    lr = 0.30 if not use_interaction else 0.42

    history = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_rows)
        for x, y in train_rows:
            logits, feats = _forward(w, b, x, use_interaction)
            probs = _softmax(logits)
            for c in range(N_CLASSES):
                g = probs[c] - (1.0 if c == y else 0.0)
                for j in range(dim):
                    w[c][j] -= lr * g * feats[j]
                b[c] -= lr * g
        tr_loss, tr_acc, tr_f1 = _evaluate(w, b, train_rows, use_interaction)
        va_loss, va_acc, va_f1 = _evaluate(w, b, val_rows, use_interaction)
        history.append(
            {
                "epoch": epoch,
                "loss": round(tr_loss, 6),
                "accuracy": round(tr_acc, 6),
                "f1": round(tr_f1, 6),
                "val_loss": round(va_loss, 6),
                "val_accuracy": round(va_acc, 6),
            }
        )
        print(
            f"epoch={epoch} loss={tr_loss:.4f} acc={tr_acc:.4f} val_acc={va_acc:.4f}",
            flush=True,
        )
    return history


def write_outputs(out_dir: Path, rows, variant: str, seed: int, epochs: int) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in METRIC_FIELDS})
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    best = max((r["val_accuracy"] for r in rows), default=0.0)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {"variant": variant, "seed": seed, "epochs": epochs, "best_val_accuracy": best},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pure-stdlib synthetic classifier (test fixture)")
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)

    rows = train(args.variant, args.epochs, args.seed)
    best = write_outputs(Path(args.out_dir), rows, args.variant, args.seed, args.epochs)
    print(f"FINAL accuracy={best:.6f} seed={args.seed} variant={args.variant}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
