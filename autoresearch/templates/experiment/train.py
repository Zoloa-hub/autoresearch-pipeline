#!/usr/bin/env python
"""Dependency-light experiment template (Auto-Research ``templates/experiment``).

This file is the *reference implementation* the pipeline copies into a run
directory and then patches (see README.md). It is deliberately small, fully
deterministic, CPU-only, and downloads nothing.

Output contract (what ``autoresearch/tools/metrics.py`` expects):

    metrics.csv   header exactly: epoch,loss,accuracy,f1,val_loss,val_accuracy
    metrics.jsonl one JSON object per epoch with the same six keys

and a final stdout line::

    FINAL accuracy=<x> seed=<s> variant=<v>

Usage
-----
    python train.py --epochs 3 --seed 0 --variant baseline \
        --out-dir runs/baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

try:  # optional accelerator; never required
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on torch-free machines
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

METRIC_FIELDS = ["epoch", "loss", "accuracy", "f1", "val_loss", "val_accuracy"]
VARIANTS = ("baseline", "method")
N_CLASSES = 3
N_FEATURES = 2


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG we might touch, including torch when present."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# data: synthetic, local, zero downloads
# ---------------------------------------------------------------------------
def make_dataset(seed: int, n_per_class: int = 200):
    """Deterministic 2-D, 4-class Gaussian blobs with class imbalance.

    Returns ``(x_train, y_train, x_val, y_val, x_test, y_test)`` as float64 /
    int64 numpy arrays, in a fixed order determined only by ``seed``.

    The blobs deliberately overlap and the class counts are skewed, so the
    default baseline does *not* saturate: there is headroom for the ``method``
    variant to show a real (if modest) difference. An uncontested near-1.0
    baseline would make the template useless as a discriminative harness.
    """
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[-2.0, -1.1], [2.0, -1.1], [0.0, 2.1]], dtype=np.float64
    )
    counts = [int(round(n_per_class * f)) for f in (1.0, 0.6, 0.4)]
    xs, ys = [], []
    for cls in range(N_CLASSES):
        pts = rng.normal(loc=centers[cls], scale=0.8, size=(counts[cls], N_FEATURES))
        # 10% of each class is drawn from the global noise cloud: irreducible
        # label noise, which caps achievable accuracy well below 1.0.
        n_noise = max(1, counts[cls] // 10)
        pts[:n_noise] = rng.normal(loc=0.0, scale=2.6, size=(n_noise, N_FEATURES))
        xs.append(pts)
        ys.append(np.full(counts[cls], cls, dtype=np.int64))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # One deterministic permutation, then a 60/20/20 split.
    order = rng.permutation(x.shape[0])
    x, y = x[order], y[order]
    n = x.shape[0]
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    return (
        x[:n_train], y[:n_train],
        x[n_train : n_train + n_val], y[n_train : n_train + n_val],
        x[n_train + n_val :], y[n_train + n_val :],
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Macro-averaged F1 over classes present in ``y_true``."""
    scores = []
    for cls in range(n_classes):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fp = float(np.sum((y_pred == cls) & (y_true != cls)))
        fn = float(np.sum((y_pred != cls) & (y_true == cls)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        scores.append(2 * precision * recall / denom if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# model: 2-layer MLP, softmax cross-entropy
# ---------------------------------------------------------------------------
class MLP:
    """Tiny MLP with an Adam or SGD update rule (pure numpy)."""

    def __init__(self, n_features: int, n_hidden: int, n_classes: int, seed: int):
        rng = np.random.default_rng(seed + 1234)
        self.params = {
            "w1": rng.normal(0.0, math.sqrt(2.0 / n_features),
                             size=(n_features, n_hidden)),
            "b1": np.zeros(n_hidden),
            "w2": rng.normal(0.0, math.sqrt(1.0 / n_hidden),
                             size=(n_hidden, n_classes)),
            "b2": np.zeros(n_classes),
        }
        self._moment = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._velocity = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.params["w1"] + self.params["b1"]
        a1 = np.maximum(z1, 0.0)
        logits = a1 @ self.params["w2"] + self.params["b2"]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def update(self, grads: dict, lr: float, variant: str, weight_decay: float) -> None:
        self.step += 1
        if variant == "method":  # Adam
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for name, grad in grads.items():
                g = grad + weight_decay * self.params[name]
                self._moment[name] = beta1 * self._moment[name] + (1 - beta1) * g
                self._velocity[name] = beta2 * self._velocity[name] + (1 - beta2) * g * g
                m_hat = self._moment[name] / (1 - beta1 ** self.step)
                v_hat = self._velocity[name] / (1 - beta2 ** self.step)
                self.params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        else:  # plain SGD, no momentum, no decay
            for name, grad in grads.items():
                self.params[name] -= lr * grad


def train_numpy(
    x_train, y_train, x_val, y_val, x_test, y_test, args
) -> list[dict]:
    set_seed(args.seed)
    model = MLP(N_FEATURES, args.hidden, N_CLASSES, args.seed)
    rng = np.random.default_rng(args.seed + 7)
    n = x_train.shape[0]
    rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        # cosine decay only for the `method` variant
        if args.variant == "method":
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs, 1)))
        else:
            lr = args.lr
        weight_decay = args.weight_decay if args.variant == "method" else 0.0

        order = rng.permutation(n)
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            probs = model.forward(xb)
            onehot = np.zeros_like(probs)
            onehot[np.arange(yb.size), yb] = 1.0
            grad_logits = (probs - onehot) / xb.shape[0]

            a1 = np.maximum(xb @ model.params["w1"] + model.params["b1"], 0.0)
            grads = {
                "w2": a1.T @ grad_logits,
                "b2": grad_logits.sum(axis=0),
            }
            da1 = grad_logits @ model.params["w2"].T
            dz1 = da1 * (a1 > 0)
            grads["w1"] = xb.T @ dz1
            grads["b1"] = dz1.sum(axis=0)
            model.update(grads, lr, args.variant, weight_decay)

        metrics = evaluate_numpy(model, x_train, y_train, x_val, y_val)
        metrics["epoch"] = epoch
        rows.append(metrics)
        print(
            f"epoch={epoch} loss={metrics['loss']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_accuracy={metrics['val_accuracy']:.4f}",
            flush=True,
        )
    test_acc = accuracy(y_test, model.forward(x_test).argmax(axis=1))
    print(f"test_accuracy={test_acc:.4f} (not used for selection)", flush=True)
    return rows


def evaluate_numpy(model: MLP, x_tr, y_tr, x_va, y_va) -> dict:
    out: dict[str, float] = {}
    for split, x, y in (("train", x_tr, y_tr), ("val", x_va, y_va)):
        probs = model.forward(x)
        pred = probs.argmax(axis=1)
        clipped = np.clip(probs[np.arange(y.size), y], 1e-12, 1.0)
        loss = float(-np.log(clipped).mean())
        prefix = "" if split == "train" else "val_"
        out[f"{prefix}loss"] = loss
        out[f"{prefix}accuracy"] = accuracy(y, pred)
        out[f"{prefix}f1" if split == "val" else "f1"] = macro_f1(y, pred)
    return out


def train_torch(x_train, y_train, x_val, y_val, x_test, y_test, args) -> list[dict]:
    """Same model, same schedule, expressed in torch (optional backend)."""
    set_seed(args.seed)
    torch.manual_seed(args.seed + 1234)
    dev = torch.device("cpu")
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    xv = torch.tensor(x_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)
    xte = torch.tensor(x_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    net = torch.nn.Sequential(
        torch.nn.Linear(N_FEATURES, args.hidden), torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, N_CLASSES),
    ).to(dev)
    if args.variant == "method":
        opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(net.parameters(), lr=args.lr)
    lossf = torch.nn.CrossEntropyLoss()

    rows: list[dict] = []
    g = torch.Generator().manual_seed(args.seed + 7)
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(xt.shape[0], generator=g)
        for start in range(0, xt.shape[0], args.batch_size):
            idx = perm[start : start + args.batch_size]
            opt.zero_grad()
            loss = lossf(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        metrics = evaluate_torch(net, lossf, xt, yt, xv, yv)
        metrics["epoch"] = epoch
        rows.append(metrics)
        print(
            f"epoch={epoch} loss={metrics['loss']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_accuracy={metrics['val_accuracy']:.4f}",
            flush=True,
        )
    net.eval()
    with torch.no_grad():
        test_acc = accuracy(y_test, net(xte).argmax(dim=1).numpy())
    print(f"test_accuracy={test_acc:.4f} (not used for selection)", flush=True)
    return rows


def evaluate_torch(net, lossf, x_tr, y_tr, x_va, y_va) -> dict:
    out: dict[str, float] = {}
    net.eval()
    with torch.no_grad():
        for split, x, y in (("train", x_tr, y_tr), ("val", x_va, y_va)):
            logits = net(x)
            pred = logits.argmax(dim=1).numpy()
            prefix = "" if split == "train" else "val_"
            out[f"{prefix}loss"] = float(lossf(logits, y).item())
            out[f"{prefix}accuracy"] = accuracy(y.numpy(), pred)
            out[f"{prefix}f1" if split == "val" else "f1"] = macro_f1(y.numpy(), pred)
    return out


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def write_outputs(out_dir: Path, rows: list[dict], args) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in METRIC_FIELDS})

    jsonl_path = out_dir / "metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: row[k] for k in METRIC_FIELDS}) + "\n")

    best = max(rows, key=lambda r: r["val_accuracy"]) if rows else None
    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "backend": args.backend,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "final_val_accuracy": rows[-1]["val_accuracy"] if rows else None,
        "best_val_accuracy": best["val_accuracy"] if best else None,
        "best_epoch": best["epoch"] if best else None,
        "best_f1": best["f1"] if best else None,
        "final_f1": rows[-1]["f1"] if rows else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return float(best["val_accuracy"]) if best else float("nan")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-Research training template")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "torch"),
        default="numpy",
        help="numpy (default, deterministic) or torch when installed",
    )
    parser.add_argument("--n-per-class", type=int, default=200)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.backend == "torch" and not _TORCH_AVAILABLE:
        parser.error("--backend torch requested but torch is not importable")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    set_seed(args.seed)
    x_train, y_train, x_val, y_val, x_test, y_test = make_dataset(
        args.seed, args.n_per_class
    )
    print(
        f"backend={args.backend} torch_available={_TORCH_AVAILABLE} "
        f"variant={args.variant} seed={args.seed} epochs={args.epochs} "
        f"train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]}",
        flush=True,
    )

    if args.backend == "torch":
        rows = train_torch(x_train, y_train, x_val, y_val, x_test, y_test, args)
    else:
        rows = train_numpy(x_train, y_train, x_val, y_val, x_test, y_test, args)

    best_val_accuracy = write_outputs(args.out_dir, rows, args)
    elapsed = time.time() - started
    print(
        f"wrote {args.out_dir / 'metrics.csv'} and {args.out_dir / 'metrics.jsonl'} "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    print(
        f"FINAL accuracy={best_val_accuracy:.4f} seed={args.seed} "
        f"variant={args.variant}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
