#!/usr/bin/env python
"""Thin wrapper: run the ``baseline`` arm of the experiment template.

    python run_baseline.py [--epochs 3] [--seed 0]
                           [--out-dir runs/baseline] [--train-args "..."]

Equivalent to::

    python train.py --variant baseline --out-dir runs/baseline [options]

Exits with train.py's exit code (or 124 on timeout).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / "runs" / "baseline"
VARIANT = "baseline"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--train-args",
        default="",
        help="extra flags passed verbatim to train.py, e.g. \"--lr 0.05\"",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cmd = [
        sys.executable,
        str(HERE / "train.py"),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--variant", VARIANT,
        "--out-dir", str(Path(args.out_dir)),
    ]
    if args.train_args:
        cmd.extend(args.train_args.split())

    print("[run_baseline] " + " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, cwd=str(HERE), timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[run_baseline] timed out after {args.timeout}s", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
