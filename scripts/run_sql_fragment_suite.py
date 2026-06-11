#!/usr/bin/env python3
"""Extended SQLite fragment evaluation for repaired integration outputs.

The goal is not to claim all of SQL is solved.  The suite stress-tests the
SQL fragments commonly used by analytical data-engineering pipelines: selection,
projection, inner/left joins, union, intersection, anti-join, grouping, having,
top-k ordering, nested EXISTS, and a simple window aggregate.  Results are
written as normalized L1 distortion against the clean master table.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_repair.corruption import apply_corruption  # noqa: E402
from robust_repair.generator import Scenario, generate_sources  # noqa: E402
from robust_repair.repair import integrate  # noqa: E402
from scripts.run_external_and_sql_suite import master_from_real_dataset  # noqa: E402


def _stable_public_eid(value: object) -> int:
    import hashlib
    h = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).hexdigest()
    return int(h[:15], 16)


def write_tables(conn: sqlite3.Connection, fact: pd.DataFrame, master: pd.DataFrame) -> None:
    use = fact.copy()
    hidden = [c for c in use.columns if c.startswith("cluster_true") or c == "eid_true"]
    use = use.drop(columns=hidden, errors="ignore")
    if "eid" not in use.columns:
        if "cluster" in use.columns:
            use["eid"] = use["cluster"].map(_stable_public_eid).astype(int)
        else:
            use["eid"] = np.arange(len(use), dtype=int)
    cols = ["eid", "name", "city", "state", "zip", "category", "revenue", "employees"]
    use[[c for c in cols if c in use.columns]].to_sql("fact", conn, if_exists="replace", index=False)
    master[["eid", "category"]].drop_duplicates().to_sql("entity_dim", conn, if_exists="replace", index=False)
    master[["zip", "city", "state"]].drop_duplicates().to_sql("geo", conn, if_exists="replace", index=False)
    cat = master[["category"]].drop_duplicates().copy().sort_values("category").reset_index(drop=True)
    cat["priority"] = np.arange(len(cat)) % 3
    cat.to_sql("catdim", conn, if_exists="replace", index=False)


def canonicalize(df: pd.DataFrame, value_cols: List[str]) -> Dict[str, float]:
    if df.empty:
        return {}
    key_cols = [c for c in df.columns if c not in value_cols]
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        key = "|".join(str(row[c]) for c in key_cols) if key_cols else "__all__"
        val = 0.0
        for c in value_cols:
            val += float(row[c]) if pd.notna(row[c]) else 0.0
        out[key] = out.get(key, 0.0) + val
    return out


def l1(truth: Dict[str, float], pred: Dict[str, float]) -> float:
    keys = sorted(set(truth) | set(pred))
    denom = sum(abs(truth.get(k, 0.0)) for k in keys)
    err = sum(abs(pred.get(k, 0.0) - truth.get(k, 0.0)) for k in keys)
    return err / max(1.0, denom)


QUERIES: Dict[str, Tuple[str, List[str]]] = {
    "select_project": ("SELECT category, SUM(revenue) AS revenue FROM fact WHERE employees >= 1 GROUP BY category ORDER BY category", ["revenue"]),
    "inner_join_geo": ("SELECT geo.state, SUM(fact.revenue) AS revenue FROM fact JOIN geo ON fact.zip=geo.zip GROUP BY geo.state ORDER BY geo.state", ["revenue"]),
    "left_join_geo": ("SELECT COALESCE(geo.state,'MISSING') AS state, SUM(fact.revenue) AS revenue FROM fact LEFT JOIN geo ON fact.zip=geo.zip GROUP BY COALESCE(geo.state,'MISSING') ORDER BY state", ["revenue"]),
    "join_filter_dim": ("SELECT fact.category, SUM(fact.revenue) AS revenue FROM fact JOIN catdim ON fact.category=catdim.category WHERE catdim.priority>=0 GROUP BY fact.category ORDER BY fact.category", ["revenue"]),
    "groupby_having": ("SELECT category, SUM(revenue) AS revenue FROM fact GROUP BY category HAVING SUM(revenue) >= 0 ORDER BY category", ["revenue"]),
    "topk_order_limit": ("SELECT category, SUM(revenue) AS revenue FROM fact GROUP BY category ORDER BY revenue DESC LIMIT 5", ["revenue"]),
    "union_all": ("SELECT category, SUM(revenue) AS revenue FROM (SELECT category,revenue FROM fact WHERE employees>=1 UNION ALL SELECT category,revenue FROM fact WHERE employees<0) GROUP BY category ORDER BY category", ["revenue"]),
    "intersection_positive": ("SELECT category, SUM(revenue) AS revenue FROM fact WHERE category IN (SELECT category FROM fact INTERSECT SELECT category FROM catdim) GROUP BY category ORDER BY category", ["revenue"]),
    "anti_join_not_exists": ("SELECT category, SUM(revenue) AS revenue FROM fact f WHERE NOT EXISTS (SELECT 1 FROM geo g WHERE g.zip=f.zip AND g.state='__never__') GROUP BY category ORDER BY category", ["revenue"]),
    "nested_exists": ("SELECT category, SUM(revenue) AS revenue FROM fact f WHERE EXISTS (SELECT 1 FROM catdim c WHERE c.category=f.category) GROUP BY category ORDER BY category", ["revenue"]),
    "window_ranked_groups": ("SELECT category, revenue FROM (SELECT category, SUM(revenue) AS revenue, RANK() OVER (ORDER BY SUM(revenue) DESC) AS rk FROM fact GROUP BY category) WHERE rk <= 5 ORDER BY category", ["revenue"]),
}


def query_dict(df: pd.DataFrame, master: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    conn = sqlite3.connect(":memory:")
    write_tables(conn, df, master)
    res: Dict[str, Dict[str, float]] = {}
    for name, (sql, vcols) in QUERIES.items():
        try:
            q = pd.read_sql_query(sql, conn)
            res[name] = canonicalize(q, vcols)
        except Exception as exc:  # explicit failure is measured as total distortion
            res[name] = {"__query_error__": 1e12}
    conn.close()
    return res


def run_trial(dataset: str, budget: float, seed: int, methods: Iterable[str]) -> List[dict]:
    master = master_from_real_dataset(dataset)
    scenario = Scenario(n_entities=len(master), n_sources=12, coverage=0.72, seed=seed)
    clean_records, meta = generate_sources(master, scenario)
    records, meta_c, _ = apply_corruption(master, clean_records, meta, budget=budget, seed=seed, mode="coordinated")
    truth = query_dict(master, master)
    rows = []
    for method in methods:
        out, det = integrate(records, meta_c, method, seed=seed)
        pred = query_dict(out, master)
        for qname in sorted(QUERIES):
            rows.append({
                "suite": "extended_sql_fragment",
                "dataset": dataset,
                "budget": float(budget),
                "seed": int(seed),
                "method": method,
                "query": qname,
                "query_l1_distortion": l1(truth[qname], pred[qname]),
                "runtime_sec": float(det.get("runtime_sec", 0.0)),
                "n_entities": int(len(master)),
            })
    return rows


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols, as_index=False).agg(
        query_l1_distortion_mean=("query_l1_distortion", "mean"),
        query_l1_distortion_std=("query_l1_distortion", "std"),
        runtime_sec_mean=("runtime_sec", "mean"),
        runtime_sec_std=("runtime_sec", "std"),
        n_entities_mean=("n_entities", "mean"),
    )
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "diabetes", "digits", "mpg", "diamonds", "txhousing", "midwest", "penguins"])
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    ap.add_argument("--seeds", nargs="+", type=int, default=[31, 32, 33])
    ap.add_argument("--methods", nargs="+", default=["no_repair", "majority", "iterative_truth", "dependency_truth", "parc"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for ds in args.datasets:
        for b in args.budgets:
            for seed in args.seeds:
                rows.extend(run_trial(ds, b, seed, args.methods))
    df = pd.DataFrame(rows)
    df.to_csv(out / "sql_fragment_results.csv", index=False)
    summarize(df, ["budget", "method"]).to_csv(out / "sql_fragment_overall.csv", index=False)
    summarize(df, ["dataset", "budget", "method"]).to_csv(out / "sql_fragment_dataset_summary.csv", index=False)
    summarize(df, ["query", "budget", "method"]).to_csv(out / "sql_fragment_query_summary.csv", index=False)
    print(f"wrote {len(df)} extended SQL-fragment rows to {out}")


if __name__ == "__main__":
    main()
