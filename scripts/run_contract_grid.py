#!/usr/bin/env python3
"""Run a wider contract-certification grid for the PARC paper.

This is a paper-evidence script: it evaluates whether repaired group-by SUM
answers are accompanied by intervals that cover the clean answer across both the
controlled synthetic generator and public tabular workloads.  It does not expose
hidden truth to the repair algorithm; hidden truth is used only after repair for
scoring interval coverage and point distortion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.certified_query_bounds import run_one  # noqa: E402

DEFAULT_DATASETS = ["synthetic", "iris", "wine", "breast_cancer", "diabetes", "digits"]
DEFAULT_BUDGETS = [0.10, 0.20, 0.30, 0.35]
DEFAULT_SEEDS = [11, 31, 47, 59]
DEFAULT_METHODS = ["no_repair", "dependency_truth", "parc"]


def _summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_dataset = df.groupby(["suite", "dataset", "budget", "method", "cert_kind"], as_index=False).agg({
        "point_distortion": ["mean", "std"],
        "coverage": ["mean", "std"],
        "mean_relative_width": ["mean", "std"],
        "miss_distance": ["mean", "std"],
        "cell_accuracy": ["mean", "std"],
        "integrated_quality": ["mean", "std"],
        "runtime_sec": ["mean", "std"],
    })
    by_dataset.columns = ["_".join([p for p in col if p]).rstrip("_") for col in by_dataset.columns.to_flat_index()]
    by_suite = df.groupby(["suite", "budget", "method", "cert_kind"], as_index=False).agg({
        "point_distortion": ["mean", "std"],
        "coverage": ["mean", "std"],
        "mean_relative_width": ["mean", "std"],
        "miss_distance": ["mean", "std"],
        "cell_accuracy": ["mean", "std"],
        "integrated_quality": ["mean", "std"],
        "runtime_sec": ["mean", "std"],
    })
    by_suite.columns = ["_".join([p for p in col if p]).rstrip("_") for col in by_suite.columns.to_flat_index()]
    # A compact paper table at the two review-relevant budgets.
    paper = by_suite[by_suite["budget"].isin([0.20, 0.35])].copy()
    return by_dataset, by_suite, paper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "results"))
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for dataset in args.datasets:
        for budget in args.budgets:
            for seed in args.seeds:
                rows.extend(run_one(dataset, budget, seed, args.methods))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "contract_grid_results.csv", index=False)
    by_dataset, by_suite, paper = _summarize(df)
    by_dataset.to_csv(out_dir / "contract_grid_by_dataset.csv", index=False)
    by_suite.to_csv(out_dir / "contract_grid_by_suite.csv", index=False)
    paper.to_csv(out_dir / "contract_grid_paper.csv", index=False)
    print(f"contract grid rows={len(df)} datasets={len(args.datasets)} budgets={len(args.budgets)} seeds={len(args.seeds)}")


if __name__ == "__main__":
    main()
