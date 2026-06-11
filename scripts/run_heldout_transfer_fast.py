#!/usr/bin/env python3
"""Held-out public-tabular and SQL transfer check for submission audit.

This intentionally uses seeds and budgets not used in the paper headline tables.
It is meant to catch overfitting or cherry-picking: PARC is compared against the
strong dependency-aware truth baseline and no-repair on unseen public-tabular
conversions and query workloads.  The script is CPU-only and writes compact CSVs
that can be inspected independently of the main benchmark.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_external_and_sql_suite import (  # noqa: E402
    master_from_real_dataset,
    _query_results,
    _l1_distortion,
)
from robust_repair.generator import Scenario, generate_sources  # noqa: E402
from robust_repair.corruption import apply_corruption  # noqa: E402
from robust_repair.metrics import evaluate  # noqa: E402
from robust_repair.repair import integrate  # noqa: E402


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric = [c for c in df.columns if c not in set(group_cols) and pd.api.types.is_numeric_dtype(df[c])]
    out = df.groupby(group_cols, as_index=False).agg({c: ["mean", "std", "min", "max"] for c in numeric})
    out.columns = ["_".join([p for p in col if p]) for col in out.columns.to_flat_index()]
    return out


def run_trial(dataset: str, budget: float, seed: int, methods: List[str]) -> tuple[list[Dict], list[Dict]]:
    master = master_from_real_dataset(dataset)
    scenario = Scenario(n_entities=len(master), n_sources=12, coverage=0.72, seed=seed)
    clean_records, meta = generate_sources(master, scenario)
    records, meta_c, _ = apply_corruption(master, clean_records, meta, budget=budget, seed=seed, mode="coordinated")
    truth_q = _query_results(master, master)
    out0, det0 = integrate(records, meta_c, "no_repair", seed=seed)
    base = evaluate(master, records, out0, det0["runtime_sec"])
    rows: list[Dict] = []
    qrows: list[Dict] = []
    for method in methods:
        if method == "no_repair":
            out, det, met = out0, det0, base
        else:
            out, det = integrate(records, meta_c, method, seed=seed)
            met = evaluate(master, records, out, det["runtime_sec"], baseline_distortion=base["aggregate_distortion"])
        row = {
            "suite": "heldout_public_transfer",
            "dataset": dataset,
            "budget": float(budget),
            "seed": int(seed),
            "method": method,
            "n_entities": int(len(master)),
            "n_records": int(len(records)),
            "n_attacked_records": int(records["attacked"].sum()),
        }
        row.update(met)
        rows.append(row)
        qres = _query_results(out, master)
        for qname in sorted(truth_q):
            qrows.append({
                "suite": "heldout_public_sql",
                "dataset": dataset,
                "budget": float(budget),
                "seed": int(seed),
                "method": method,
                "query": qname,
                "query_l1_distortion": _l1_distortion(truth_q[qname], qres[qname]),
                "runtime_sec": float(det.get("runtime_sec", 0.0)),
            })
    return rows, qrows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "diabetes"])
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.17, 0.27])
    ap.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44])
    ap.add_argument("--methods", nargs="+", default=["no_repair", "dependency_truth", "parc"])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows: list[Dict] = []
    qrows: list[Dict] = []
    for dataset in args.datasets:
        for budget in args.budgets:
            for seed in args.seeds:
                rr, qq = run_trial(dataset, budget, seed, args.methods)
                rows.extend(rr); qrows.extend(qq)
    real = pd.DataFrame(rows)
    sql = pd.DataFrame(qrows)
    real.to_csv(out / "heldout_transfer_results.csv", index=False)
    summarize(real, ["dataset", "budget", "method"]).to_csv(out / "heldout_transfer_summary.csv", index=False)
    summarize(real, ["budget", "method"]).to_csv(out / "heldout_transfer_overall.csv", index=False)
    sql.to_csv(out / "heldout_sql_results.csv", index=False)
    summarize(sql, ["budget", "method"]).to_csv(out / "heldout_sql_overall.csv", index=False)
    print(f"wrote {len(real)} repair rows and {len(sql)} SQL rows to {out}")


if __name__ == "__main__":
    main()
