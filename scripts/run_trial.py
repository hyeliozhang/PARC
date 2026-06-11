#!/usr/bin/env python3
"""Run one PARC benchmark trial in an isolated Python process.

The script exits with os._exit(0) after flushing outputs.  This makes repeated
trial generation robust in constrained notebook containers where interpreter
cleanup of many pandas/numpy objects can be expensive.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_benchmark import run_once  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--entities", type=int, required=True)
    ap.add_argument("--sources", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--mode", default="coordinated")
    ap.add_argument("--ops", nargs="*", default=None)
    ap.add_argument("--trust-alpha", type=float, default=1.0)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    rows, _ = run_once(
        args.entities, args.sources, args.budget, args.seed,
        methods=args.methods, enabled_ops=args.ops, trust_alpha=args.trust_alpha, mode=args.mode,
    )
    extras = {}
    for item in args.extra:
        if "=" in item:
            k, v = item.split("=", 1)
            extras[k] = v
    for row in rows:
        row.update(extras)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
