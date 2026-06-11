#!/usr/bin/env python3
"""Evaluate query-level certification for PARC repair outputs.

The script constructs simple group-by SUM certificates from the cell-level PARC
certificates. A certified aggregate interval covers the true aggregate when the
sum of lower/upper cell-level candidate intervals contains the clean query value.
The interval is intentionally conservative and uses only candidates recorded in
PARC certificates plus the accepted value; hidden truth is used only for scoring.
"""
from __future__ import annotations

import argparse
import json
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
from scripts.run_external_and_sql_suite import master_from_real_dataset  # noqa: E402

METHODS = ["no_repair", "dependency_truth", "parc"]
PUBLIC_DATASETS = ["iris", "wine", "breast_cancer", "diabetes"]
BUDGETS = [0.20, 0.35]
SEEDS = [11, 31]


def _truth_aggregate(master: pd.DataFrame) -> Dict[str, float]:
    return master.groupby("category")["revenue"].sum().astype(float).to_dict()


def _point_aggregate(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0:
        return {}
    return df.groupby("category")["revenue"].sum().astype(float).to_dict()


def _distortion(truth: Dict[str, float], pred: Dict[str, float]) -> float:
    keys = sorted(set(truth) | set(pred))
    denom = sum(abs(float(truth.get(k, 0.0))) for k in keys)
    err = sum(abs(float(pred.get(k, 0.0)) - float(truth.get(k, 0.0))) for k in keys)
    return float(err / max(1.0, denom))


def _numeric_candidates(cert: dict, accepted: float) -> List[float]:
    vals = [float(accepted)]
    for k in cert.get("candidates", {}).keys():
        try:
            vals.append(float(k))
        except Exception:
            continue
    if not vals:
        vals = [float(accepted)]
    return vals


def _parc_intervals(out: pd.DataFrame, details: dict, margin_threshold: float = 1.5) -> Dict[str, Tuple[float, float]]:
    intervals: Dict[str, Tuple[float, float]] = {}
    certs = details.get("certificates", {})
    for row in out.itertuples(index=False):
        cat = str(getattr(row, "category"))
        revenue = float(getattr(row, "revenue"))
        cluster = str(getattr(row, "cluster")) if hasattr(row, "cluster") else ""
        c = certs.get(cluster, {}).get("cells", {}).get("revenue", {})
        margin = float(c.get("margin", 0.0))
        vals = _numeric_candidates(c, revenue)
        if margin >= margin_threshold:
            lo, hi = revenue, revenue
        else:
            lo, hi = min(vals), max(vals)
        old_lo, old_hi = intervals.get(cat, (0.0, 0.0))
        intervals[cat] = (old_lo + float(lo), old_hi + float(hi))
    return intervals


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


def _interval_metrics(truth: Dict[str, float], intervals: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    keys = sorted(set(truth) | set(intervals))
    if not keys:
        return {"coverage": 1.0, "mean_relative_width": 0.0, "miss_distance": 0.0}
    covers = 0
    rel_widths = []
    miss_distance = 0.0
    denom_total = 0.0
    for k in keys:
        tv = float(truth.get(k, 0.0))
        lo, hi = intervals.get(k, (0.0, 0.0))
        lo, hi = min(lo, hi), max(lo, hi)
        if lo <= tv <= hi:
            covers += 1
        else:
            miss_distance += min(abs(tv - lo), abs(tv - hi))
        rel_widths.append((hi - lo) / max(1.0, abs(tv)))
        denom_total += abs(tv)
    return {
        "coverage": float(covers / len(keys)),
        "mean_relative_width": float(np.mean(rel_widths)),
        "miss_distance": float(miss_distance / max(1.0, denom_total)),
    }


def _make_master(dataset: str, n_entities: int, seed: int) -> Tuple[pd.DataFrame, str]:
    if dataset == "synthetic":
        return generate_master(n_entities=n_entities, seed=seed), "synthetic-control"
    return master_from_real_dataset(dataset), "public-tabular"


def run_one(dataset: str, budget: float, seed: int, methods: Iterable[str]) -> List[dict]:
    master, suite = _make_master(dataset, 300, seed)
    scen = Scenario(n_entities=len(master), n_sources=12, coverage=0.72, seed=seed)
    clean, meta = generate_sources(master, scen)
    records, meta_c, attack = apply_corruption(master, clean, meta, budget=budget, seed=seed, mode="coordinated")
    truth = _truth_aggregate(master)
    rows = []
    for method in methods:
        out, details = integrate(records, meta_c, method, seed=seed)
        point = _point_aggregate(out)
        if method == "parc":
            intervals = _parc_intervals(out, details)
            cert_kind = "parc-certificate"
        elif method in {"dependency_truth", "iterative_truth", "source_dependence"}:
            intervals = _candidate_intervals(out, details)
            cert_kind = "baseline-candidate-certificate"
        else:
            intervals = _point_intervals(out)
            cert_kind = "point-output"
        im = _interval_metrics(truth, intervals)
        met = evaluate(master, records, out, details.get("runtime_sec", 0.0))
        rows.append({
            "suite": suite,
            "dataset": dataset,
            "budget": float(budget),
            "seed": int(seed),
            "method": method,
            "cert_kind": cert_kind,
            "n_entities": int(len(master)),
            "n_records": int(len(records)),
            "n_attacked_records": int(records.get("attacked", pd.Series(dtype=int)).sum()),
            "point_distortion": _distortion(truth, point),
            "coverage": im["coverage"],
            "mean_relative_width": im["mean_relative_width"],
            "miss_distance": im["miss_distance"],
            "cell_accuracy": met.get("cell_accuracy", np.nan),
            "integrated_quality": met.get("integrated_quality", np.nan),
            "runtime_sec": details.get("runtime_sec", np.nan),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/results")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    datasets = ["synthetic"] + PUBLIC_DATASETS
    for dataset in datasets:
        for budget in BUDGETS:
            for seed in SEEDS:
                print(f"cert-bounds {dataset} beta={budget} seed={seed}", flush=True)
                all_rows.extend(run_one(dataset, budget, seed, METHODS))
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "certified_query_bounds.csv", index=False)
    summary = df.groupby(["suite", "dataset", "budget", "method", "cert_kind"], as_index=False).agg({
        "point_distortion": ["mean", "std"],
        "coverage": ["mean", "std"],
        "mean_relative_width": ["mean", "std"],
        "miss_distance": ["mean", "std"],
        "cell_accuracy": ["mean", "std"],
        "integrated_quality": ["mean", "std"],
        "runtime_sec": ["mean", "std"],
    })
    summary.columns = ["_".join([c for c in col if c]).rstrip("_") for col in summary.columns.to_flat_index()]
    summary.to_csv(out_dir / "certified_query_bounds_summary.csv", index=False)
    paper = df.groupby(["suite", "budget", "method", "cert_kind"], as_index=False).agg({
        "point_distortion": "mean",
        "coverage": "mean",
        "mean_relative_width": "mean",
        "miss_distance": "mean",
        "cell_accuracy": "mean",
        "integrated_quality": "mean",
    })
    paper.to_csv(out_dir / "certified_query_bounds_paper.csv", index=False)
    payload = {
        "rows": int(len(df)),
        "datasets": datasets,
        "budgets": BUDGETS,
        "seeds": SEEDS,
        "methods": METHODS,
    }
    (out_dir / "certified_query_bounds_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
