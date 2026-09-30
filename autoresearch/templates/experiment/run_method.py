#!/usr/bin/env python
"""Thin wrapper: run the ``method`` arm of the experiment template.

    python run_method.py [--epochs 3] [--seed 0]
                         [--out-dir runs/method] [--train-args "..."]

Equivalent to::

    python train.py --variant method --out-dir runs/method [options]

Both wrappers pass through to the *same* train.py code path, so the two arms
differ only in ``--variant`` and output directory -- that is what makes the
comparison apples-to-apples.

Exits with train.py's exit code (or 124 on timeout).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / "runs" / "method"
VARIANT = "method"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--train-args",
        default="",
        help="extra flags passed verbatim to train.py, e.g. \"--hidden 32\"",
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

    print("[run_method] " + " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, cwd=str(HERE), timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"[run_method] timed out after {args.timeout}s", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
