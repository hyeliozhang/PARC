#!/usr/bin/env python3
"""Public-tabular metadata-stress evaluation for PARC.

This suite isolates corruptions that are specific to integration metadata:
source-level schema swaps, duplicate/provenance laundering, declared-trust
inflation, and aggregate outliers.  It complements the broad corruption suite by
making the data-engineering motivation explicit on real public tabular records.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_repair.corruption import apply_corruption  # noqa: E402
from robust_repair.generator import Scenario, generate_sources  # noqa: E402
from robust_repair.metrics import evaluate  # noqa: E402
from robust_repair.repair import integrate  # noqa: E402
from scripts.run_external_and_sql_suite import master_from_real_dataset, _query_results, _l1_distortion  # noqa: E402

METHODS = ["no_repair", "majority", "iterative_truth", "dependency_truth", "parc"]
OPS = ["schema", "duplicate", "provenance", "trust", "outlier"]


def trial(dataset: str, seed: int, budget: float, methods: Iterable[str]) -> tuple[list[Dict], list[Dict]]:
    master = master_from_real_dataset(dataset)
    scenario = Scenario(n_entities=len(master), n_sources=12, coverage=0.72, seed=seed)
    clean_records, meta = generate_sources(master, scenario)
    records, meta_c, attack = apply_corruption(master, clean_records, meta, budget=budget, seed=seed, enabled_ops=OPS, mode="coordinated")
    truth_q = _query_results(master, master)
    out0, det0 = integrate(records, meta_c, "no_repair", seed=seed)
    base = evaluate(master, records, out0, det0["runtime_sec"])
    real_rows: list[Dict] = []
    sql_rows: list[Dict] = []
    for method in methods:
        if method == "no_repair":
            out, det, met = out0, det0, base
        else:
            out, det = integrate(records, meta_c, method, seed=seed)
            met = evaluate(master, records, out, det["runtime_sec"], baseline_distortion=base["aggregate_distortion"])
        row = {
            "suite": "public_metadata_stress",
            "dataset": dataset,
            "budget": budget,
            "seed": seed,
            "method": method,
            "operators": "+".join(OPS),
            "n_entities": len(master),
            "n_records": len(records),
            "n_attacked_records": int(records["attacked"].sum()),
            "n_duplicates": int(records["is_duplicate"].sum()),
            "n_schema_sources": len(attack.get("schema_sources", [])),
        }
        row.update(met)
        real_rows.append(row)
        qres = _query_results(out, master)
        for qname in sorted(truth_q):
            sql_rows.append({
                "suite": "public_metadata_stress_sql",
                "dataset": dataset,
                "budget": budget,
                "seed": seed,
                "method": method,
                "query": qname,
                "query_l1_distortion": _l1_distortion(truth_q[qname], qres[qname]),
                "runtime_sec": float(det.get("runtime_sec", 0.0)),
                "n_entities": len(master),
            })
    return real_rows, sql_rows


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric = [c for c in df.columns if c not in set(group_cols) and pd.api.types.is_numeric_dtype(df[c])]
    out = df.groupby(group_cols, as_index=False).agg({c: ["mean", "std"] for c in numeric})
    out.columns = ["_".join([p for p in col if p]) for col in out.columns.to_flat_index()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "diabetes"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[31, 32])
    ap.add_argument("--budget", type=float, default=0.25)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    real: list[Dict] = []
    sql: list[Dict] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            r, s = trial(dataset, seed, args.budget, METHODS)
            real.extend(r)
            sql.extend(s)
    real_df = pd.DataFrame(real)
    sql_df = pd.DataFrame(sql)
    real_df.to_csv(out_dir / "public_metadata_stress_results.csv", index=False)
    sql_df.to_csv(out_dir / "public_metadata_stress_sql_results.csv", index=False)
    summarize(real_df, ["budget", "method"]).to_csv(out_dir / "public_metadata_stress_overall.csv", index=False)
    summarize(sql_df, ["budget", "method"]).to_csv(out_dir / "public_metadata_stress_sql_overall.csv", index=False)
    print(f"wrote {len(real_df)} rows and {len(sql_df)} SQL rows")


if __name__ == "__main__":
    main()
