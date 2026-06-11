#!/usr/bin/env python3
"""Selective aggregate-contract frontier for PARC.

The script evaluates a review-facing question that point repair accuracy does not
answer: for which downstream aggregate groups can a repaired relation return a
compact certificate that (i) contains the clean answer and (ii) is narrow enough
to be operationally useful? Hidden clean aggregates are used only for scoring the
certificate after the repair is produced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_repair.generator import Scenario, generate_master, generate_sources  # noqa: E402
from robust_repair.corruption import apply_corruption  # noqa: E402
from robust_repair.repair import integrate  # noqa: E402
from scripts.certified_query_bounds import _numeric_candidates  # noqa: E402
from scripts.run_external_and_sql_suite import DATASET_DESCRIPTIONS, master_from_real_dataset  # noqa: E402

DATASETS = ["synthetic", "iris", "wine", "breast_cancer"]
BUDGETS = [0.20, 0.35]
SEEDS = [11]
WIDTH_THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
MARGIN_THRESHOLDS = [0.0, 1.0, 1.5, 2.0]
METHODS = ["parc", "dependency_truth"]


def _make_master(dataset: str, seed: int) -> tuple[pd.DataFrame, str]:
    if dataset == "synthetic":
        return generate_master(n_entities=300, seed=seed), "synthetic-control"
    return master_from_real_dataset(dataset), "public-tabular"


def _truth_by_group(master: pd.DataFrame) -> Dict[str, float]:
    return master.groupby("category")["revenue"].sum().astype(float).to_dict()


def _parc_group_intervals(out: pd.DataFrame, details: dict, margin_threshold: float) -> Dict[str, Tuple[float, float]]:
    intervals: Dict[str, Tuple[float, float]] = {}
    certs = details.get("certificates", {})
    for row in out.itertuples(index=False):
        cat = str(getattr(row, "category"))
        revenue = float(getattr(row, "revenue"))
        cluster = str(getattr(row, "cluster")) if hasattr(row, "cluster") else ""
        cert = certs.get(cluster, {}).get("cells", {}).get("revenue", {})
        margin = float(cert.get("margin", 0.0))
        # High-margin values are exact certified point contributions. Low-margin
        # values expose the recorded candidate span. This is evidence-only: the
        # clean truth never enters interval construction.
        if margin >= margin_threshold:
            lo, hi = revenue, revenue
        else:
            vals = _numeric_candidates(cert, revenue)
            lo, hi = min(vals), max(vals)
        old_lo, old_hi = intervals.get(cat, (0.0, 0.0))
        intervals[cat] = (old_lo + lo, old_hi + hi)
    return intervals


def _point_intervals(out: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    if len(out) == 0:
        return {}
    return {str(k): (float(v), float(v)) for k, v in out.groupby("category")["revenue"].sum().items()}


def _score_frontier(truth: Dict[str, float], intervals: Dict[str, Tuple[float, float]], width_threshold: float) -> dict:
    keys = sorted(set(truth) | set(intervals))
    certified = 0
    covered = 0
    widths: List[float] = []
    misses: List[float] = []
    for k in keys:
        tv = float(truth.get(k, 0.0))
        lo, hi = intervals.get(k, (0.0, 0.0))
        lo, hi = min(float(lo), float(hi)), max(float(lo), float(hi))
        rel_width = (hi - lo) / max(1.0, abs(tv))
        if rel_width <= width_threshold:
            certified += 1
            widths.append(rel_width)
            if lo <= tv <= hi:
                covered += 1
            else:
                misses.append(min(abs(tv - lo), abs(tv - hi)) / max(1.0, abs(tv)))
    return {
        "n_groups": len(keys),
        "certified_groups": certified,
        "certified_fraction": certified / max(1, len(keys)),
        "conditional_coverage": covered / max(1, certified),
        "mean_certified_width": float(np.mean(widths)) if widths else np.nan,
        "mean_relative_miss": float(np.mean(misses)) if misses else 0.0,
    }


def run() -> pd.DataFrame:
    rows: List[dict] = []
    for dataset in DATASETS:
        print(f"frontier dataset={dataset}", flush=True)
        for budget in BUDGETS:
            for seed in SEEDS:
                master, suite = _make_master(dataset, seed)
                scenario = Scenario(n_entities=len(master), n_sources=12, coverage=0.72, seed=seed)
                clean, meta = generate_sources(master, scenario)
                rec, meta_c, _attack = apply_corruption(master, clean, meta, budget=budget, seed=seed, mode="coordinated")
                truth = _truth_by_group(master)
                for method in METHODS:
                    out, details = integrate(rec, meta_c, method, seed=seed)
                    for margin_threshold in (MARGIN_THRESHOLDS if method == "parc" else [0.0]):
                        intervals = _parc_group_intervals(out, details, margin_threshold) if method == "parc" else _point_intervals(out)
                        for width_threshold in WIDTH_THRESHOLDS:
                            score = _score_frontier(truth, intervals, width_threshold)
                            rows.append({
                                "suite": suite,
                                "dataset": dataset,
                                "budget": budget,
                                "seed": seed,
                                "method": method,
                                "margin_threshold": margin_threshold,
                                "width_threshold": width_threshold,
                                **score,
                            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "results"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(out_dir / "selective_contract_frontier.csv", index=False)
    summary = df.groupby(["suite", "budget", "method", "margin_threshold", "width_threshold"], as_index=False).agg({
        "certified_fraction": ["mean", "std"],
        "conditional_coverage": ["mean", "std"],
        "mean_certified_width": ["mean", "std"],
        "mean_relative_miss": ["mean", "std"],
        "certified_groups": "sum",
        "n_groups": "sum",
    })
    summary.columns = ["_".join([x for x in c if x]).rstrip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(out_dir / "selective_contract_frontier_summary.csv", index=False)
    paper = summary[
        (summary["budget"].isin([0.20, 0.35]))
        & (summary["width_threshold"].isin([0.05, 0.10, 0.25]))
        & ((summary["method"] == "dependency_truth") | (summary["margin_threshold"].isin([1.0, 1.5])))
    ].copy()
    paper.to_csv(out_dir / "selective_contract_frontier_paper.csv", index=False)
    print(f"selective contract rows={len(df)} summary={len(summary)}")


if __name__ == "__main__":
    main()
