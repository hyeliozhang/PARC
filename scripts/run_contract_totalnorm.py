#!/usr/bin/env python3
"""Total-normalized aggregate certificate quality.

Mean relative interval width can be dominated by tiny groups in public tabular
workloads.  This script scores aggregate contracts by the scale of the whole
query answer: total interval width divided by total absolute clean aggregate,
and total miss distance divided by the same denominator.  The repair algorithm
never sees clean answers; they are used only for post-hoc certificate scoring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_repair.generator import Scenario, generate_master, generate_sources  # noqa: E402
from robust_repair.corruption import apply_corruption  # noqa: E402
from robust_repair.metrics import evaluate  # noqa: E402
from robust_repair.repair import integrate  # noqa: E402
from scripts.certified_query_bounds import _numeric_candidates  # noqa: E402
from scripts.run_external_and_sql_suite import DATASET_DESCRIPTIONS, master_from_real_dataset  # noqa: E402

DATASETS = ["synthetic", "iris", "wine", "breast_cancer", "diabetes"]
BUDGETS = [0.20, 0.35]
SEEDS = [11, 12, 13]
METHODS = ["no_repair", "dependency_truth", "parc"]


def _make_master(dataset: str, seed: int) -> tuple[pd.DataFrame, str]:
    if dataset == "synthetic":
        return generate_master(n_entities=300, seed=seed), "synthetic-control"
    return master_from_real_dataset(dataset), "public-tabular"


def _truth_aggregate(master: pd.DataFrame) -> Dict[str, float]:
    return master.groupby("category")["revenue"].sum().astype(float).to_dict()


def _point_aggregate(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0:
        return {}
    return df.groupby("category")["revenue"].sum().astype(float).to_dict()


def _point_intervals(out: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    return {k: (v, v) for k, v in _point_aggregate(out).items()}


def _candidate_intervals(out: pd.DataFrame, details: dict, margin_threshold: float = 1.5) -> Dict[str, Tuple[float, float]]:
    """Build a fair uncertainty interval from a method's own candidate panel.

    This gives strong point baselines the same opportunity as PARC to express
    uncertainty when the top candidate has a low margin.  Hidden truth is not
    used.  Methods without certificates fall back to point intervals.
    """
    certs = details.get("certificates", {}) if isinstance(details, dict) else {}
    if not certs:
        return _point_intervals(out)
    intervals: Dict[str, Tuple[float, float]] = {}
    for row in out.itertuples(index=False):
        cat = str(getattr(row, "category"))
        revenue = float(getattr(row, "revenue"))
        cluster = str(getattr(row, "cluster")) if hasattr(row, "cluster") else ""
        c = certs.get(cluster, {}).get("cells", {}).get("revenue", {})
        margin = float(c.get("margin", 0.0))
        if margin >= margin_threshold:
            lo, hi = revenue, revenue
        else:
            vals = _numeric_candidates(c, revenue)
            lo, hi = min(vals), max(vals)
        old_lo, old_hi = intervals.get(cat, (0.0, 0.0))
        intervals[cat] = (old_lo + float(lo), old_hi + float(hi))
    return intervals


def _parc_intervals(out: pd.DataFrame, details: dict, margin_threshold: float = 1.5) -> Dict[str, Tuple[float, float]]:
    intervals: Dict[str, Tuple[float, float]] = {}
    certs = details.get("certificates", {})
    for row in out.itertuples(index=False):
        cat = str(getattr(row, "category"))
        revenue = float(getattr(row, "revenue"))
        cluster = str(getattr(row, "cluster")) if hasattr(row, "cluster") else ""
        c = certs.get(cluster, {}).get("cells", {}).get("revenue", {})
        margin = float(c.get("margin", 0.0))
        if margin >= margin_threshold:
            lo, hi = revenue, revenue
        else:
            vals = _numeric_candidates(c, revenue)
            lo, hi = min(vals), max(vals)
        old_lo, old_hi = intervals.get(cat, (0.0, 0.0))
        intervals[cat] = (old_lo + lo, old_hi + hi)
    return intervals


def _distortion(truth: Dict[str, float], pred: Dict[str, float]) -> float:
    keys = sorted(set(truth) | set(pred))
    denom = sum(abs(float(truth.get(k, 0.0))) for k in keys)
    err = sum(abs(float(pred.get(k, 0.0)) - float(truth.get(k, 0.0))) for k in keys)
    return float(err / max(1.0, denom))


def _contract_metrics(truth: Dict[str, float], intervals: Dict[str, Tuple[float, float]]) -> dict:
    keys = sorted(set(truth) | set(intervals))
    denom = sum(abs(float(truth.get(k, 0.0))) for k in keys)
    if not keys:
        return {"coverage": 1.0, "total_norm_width": 0.0, "total_norm_miss": 0.0, "mean_relative_width": 0.0}
    covers = 0
    total_width = 0.0
    total_miss = 0.0
    rel_widths: List[float] = []
    for k in keys:
        tv = float(truth.get(k, 0.0))
        lo, hi = intervals.get(k, (0.0, 0.0))
        lo, hi = min(float(lo), float(hi)), max(float(lo), float(hi))
        total_width += hi - lo
        rel_widths.append((hi - lo) / max(1.0, abs(tv)))
        if lo <= tv <= hi:
            covers += 1
        else:
            total_miss += min(abs(tv - lo), abs(tv - hi))
    return {
        "coverage": covers / max(1, len(keys)),
        "total_norm_width": total_width / max(1.0, denom),
        "total_norm_miss": total_miss / max(1.0, denom),
        "mean_relative_width": float(np.mean(rel_widths)),
    }


def run_one(dataset: str, budget: float, seed: int, methods: Iterable[str]) -> List[dict]:
    master, suite = _make_master(dataset, seed)
    scen = Scenario(n_entities=len(master), n_sources=12, coverage=0.72, seed=seed)
    clean, meta = generate_sources(master, scen)
    records, meta_c, _attack = apply_corruption(master, clean, meta, budget=budget, seed=seed, mode="coordinated")
    truth = _truth_aggregate(master)
    rows: List[dict] = []
    for method in methods:
        out, details = integrate(records, meta_c, method, seed=seed)
        point = _point_aggregate(out)
        intervals = _parc_intervals(out, details) if method == "parc" else (_candidate_intervals(out, details) if method in {"dependency_truth", "iterative_truth", "source_dependence"} else _point_intervals(out))
        cm = _contract_metrics(truth, intervals)
        met = evaluate(master, records, out, details.get("runtime_sec", 0.0))
        rows.append({
            "suite": suite,
            "dataset": dataset,
            "budget": budget,
            "seed": seed,
            "method": method,
            "n_entities": len(master),
            "point_distortion": _distortion(truth, point),
            **cm,
            "cell_accuracy": met["cell_accuracy"],
            "integrated_quality": met["integrated_quality"],
            "runtime_sec": details.get("runtime_sec", np.nan),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "results"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for dataset in DATASETS:
        print(f"totalnorm dataset={dataset}", flush=True)
        for budget in BUDGETS:
            print(f"  beta={budget}", flush=True)
            for seed in SEEDS:
                rows.extend(run_one(dataset, budget, seed, METHODS))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "contract_totalnorm_results.csv", index=False)
    summary = df.groupby(["suite", "budget", "method"], as_index=False).agg({
        "point_distortion": ["mean", "std"],
        "coverage": ["mean", "std"],
        "total_norm_width": ["mean", "std"],
        "total_norm_miss": ["mean", "std"],
        "mean_relative_width": ["mean", "std"],
        "cell_accuracy": ["mean", "std"],
        "integrated_quality": ["mean", "std"],
        "runtime_sec": ["mean", "std"],
    })
    summary.columns = ["_".join([x for x in c if x]).rstrip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(out_dir / "contract_totalnorm_summary.csv", index=False)
    print(f"total-normalized contract rows={len(df)}")


if __name__ == "__main__":
    main()
