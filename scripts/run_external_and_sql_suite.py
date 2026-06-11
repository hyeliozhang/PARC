#!/usr/bin/env python3
"""External-tabular and SQL-workflow evaluation for PARC.

This script uses public tabular datasets bundled with scikit-learn and converts
one clean table into multiple conflicting source feeds. It then evaluates the
same repair methods on (i) cell/entity/aggregate quality and (ii) SQL workflow
queries over the repaired output. No network access, paid services, or hidden
labels are used by the repair algorithms; original dataset rows are used only as
evaluation truth.
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
from robust_repair.generator import Scenario, generate_sources, make_zip_city_maps  # noqa: E402
from robust_repair.metrics import aggregate_distortion, evaluate  # noqa: E402
from robust_repair.repair import integrate  # noqa: E402

DATASET_DESCRIPTIONS = {
    "iris": "Fisher iris measurements, 150 rows, 4 numeric attributes, 3 classes",
    "wine": "UCI wine recognition, 178 rows, 13 numeric attributes, 3 classes",
    "breast_cancer": "Wisconsin diagnostic breast-cancer measurements, 569 rows, 30 numeric attributes, 2 classes",
    "diabetes": "diabetes regression table, 442 rows, 10 numeric attributes",
    "digits": "digits handwritten-image feature table, deterministically subsampled to 600 rows",
    "mpg": "plotnine automobile fuel-economy table with real categorical and numeric vehicle attributes",
    "diamonds": "plotnine diamonds transaction-like table, deterministically subsampled",
    "txhousing": "plotnine Texas housing market table with city, month, sales, volume, and listings",
    "midwest": "plotnine Midwest county demographics table",
    "penguins": "plotnine Palmer penguins biological measurement table",
}

PLOTNINE_DATASETS = {"mpg", "diamonds", "txhousing", "midwest", "penguins"}


def _load_dataset(name: str, max_rows: int = 300) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Load a bundled public tabular dataset without network access.

    The first five datasets come from scikit-learn.  The additional datasets
    come from plotnine's packaged example data.  They are not enterprise
    integration corpora, but they provide non-synthetic value distributions,
    categories, missingness, and correlations that are independent of the
    PARC generator.
    """
    from sklearn import datasets

    if name == "iris":
        ds = datasets.load_iris(as_frame=True)
        X = ds.data.copy()
        y = np.asarray(ds.target)
    elif name == "wine":
        ds = datasets.load_wine(as_frame=True)
        X = ds.data.copy()
        y = np.asarray(ds.target)
    elif name == "breast_cancer":
        ds = datasets.load_breast_cancer(as_frame=True)
        X = ds.data.copy()
        y = np.asarray(ds.target)
    elif name == "diabetes":
        ds = datasets.load_diabetes(as_frame=True)
        X = ds.data.copy()
        y = pd.qcut(pd.Series(ds.target), q=4, labels=False, duplicates="drop").to_numpy()
    elif name == "digits":
        ds = datasets.load_digits(as_frame=True)
        X = ds.data.copy()
        y = np.asarray(ds.target)
    elif name in PLOTNINE_DATASETS:
        from plotnine import data as p9data
        raw = getattr(p9data, name).copy()
        raw = raw.replace([np.inf, -np.inf], np.nan)
        numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        cat_cols = [c for c in raw.columns if c not in numeric_cols]
        if len(numeric_cols) < 2:
            raise ValueError(f"plotnine dataset {name} lacks two numeric columns")
        # Pick a real categorical column when available; otherwise use quantiles.
        if cat_cols:
            cat_col = cat_cols[0]
            y = raw[cat_col].astype(str).fillna("missing").astype("category").cat.codes.to_numpy()
        else:
            y = pd.qcut(raw[numeric_cols[0]], q=4, labels=False, duplicates="drop").to_numpy()
        X = raw[numeric_cols].copy()
        for c in numeric_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(X[c].median())
    else:
        raise ValueError(f"unknown dataset {name}")
    if len(X) > max_rows:
        idx = np.linspace(0, len(X) - 1, max_rows, dtype=int)
        X = X.iloc[idx].reset_index(drop=True)
        y = np.asarray(y)[idx]
    return X.reset_index(drop=True), np.asarray(y), list(map(str, X.columns))


def _scale_numeric(s: pd.Series, lo: float, hi: float) -> np.ndarray:
    vals = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    vals = np.nan_to_num(vals, nan=float(np.nanmedian(vals)))
    mn, mx = float(vals.min()), float(vals.max())
    if abs(mx - mn) < 1e-12:
        return np.full(len(vals), (lo + hi) / 2.0)
    return lo + (vals - mn) * (hi - lo) / (mx - mn)


def master_from_real_dataset(name: str) -> pd.DataFrame:
    X, y, cols = _load_dataset(name)
    zip_to_city, _ = make_zip_city_maps()
    zips = sorted(zip_to_city.keys())
    # Use genuine numeric measurements, scaled into two relational measures.  A
    # pair of correlated but not identical features is used so aggregate queries
    # are tied to real variation rather than to a synthetic log-normal draw.
    first_col = cols[0]
    second_col = cols[min(1, len(cols) - 1)]
    rev = np.round(_scale_numeric(X[first_col], 1_000, 1_000_000)).astype(int)
    emp = np.maximum(1, np.round(_scale_numeric(X[second_col], 1, 4000)).astype(int))
    target = pd.Series(y).astype(int).to_numpy()
    rows = []
    for i in range(len(X)):
        cls = int(target[i])
        z = zips[(cls * 17 + i) % len(zips)]
        city, state = zip_to_city[z]
        rows.append({
            "eid": int(i),
            "name": f"{name.replace('_', ' ').title()} entity {i:04d}",
            "city": city,
            "state": state,
            "zip": z,
            "category": f"class_{cls}",
            "revenue": int(rev[i]),
            "employees": int(emp[i]),
        })
    return pd.DataFrame(rows)


def evaluate_real_dataset(name: str, budget: float, seed: int, methods: Iterable[str], n_sources: int = 12) -> List[Dict]:
    master = master_from_real_dataset(name)
    scenario = Scenario(n_entities=len(master), n_sources=n_sources, coverage=0.72, seed=seed)
    clean_records, meta = generate_sources(master, scenario)
    records, meta_c, attack = apply_corruption(master, clean_records, meta, budget=budget, seed=seed, mode="coordinated")
    out0, det0 = integrate(records, meta_c, "no_repair", seed=seed)
    base = evaluate(master, records, out0, det0["runtime_sec"])
    rows = []
    for method in methods:
        if method == "no_repair":
            out, det, met = out0, det0, base
        else:
            out, det = integrate(records, meta_c, method, seed=seed)
            met = evaluate(master, records, out, det["runtime_sec"], baseline_distortion=base["aggregate_distortion"])
        row = {
            "suite": "external_tabular",
            "dataset": name,
            "budget": float(budget),
            "seed": int(seed),
            "method": method,
            "n_entities": int(len(master)),
            "n_sources": int(n_sources),
            "n_records": int(len(records)),
            "n_attacked_records": int(records["attacked"].sum()),
            "description": DATASET_DESCRIPTIONS.get(name, name),
        }
        row.update(met)
        rows.append(row)
    return rows


def _stable_public_eid(value: object) -> int:
    import hashlib
    h = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).hexdigest()
    return int(h[:15], 16)


def _write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    """Write a production-like SQL fact table.

    Hidden evaluator columns such as cluster_true_eid are intentionally ignored.
    If a repaired output has no public eid, we derive a stable surrogate from the
    materialized cluster label. This prevents SQL evaluation from using clean
    entity identifiers while keeping query execution deterministic.
    """
    cols = ["eid", "name", "city", "state", "zip", "category", "revenue", "employees"]
    use = df.copy()
    hidden = [c for c in use.columns if c.startswith("cluster_true") or c == "eid_true"]
    use = use.drop(columns=hidden, errors="ignore")
    if "eid" not in use.columns:
        if "cluster" in use.columns:
            use["eid"] = use["cluster"].map(_stable_public_eid).astype(int)
        else:
            use["eid"] = np.arange(len(use), dtype=int)
    use = use[[c for c in cols if c in use.columns]].copy()
    use.to_sql(name, conn, if_exists="replace", index=False)


def _prepare_dims(master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    geo = master[["zip", "city", "state"]].drop_duplicates().copy()
    cat = master[["category"]].drop_duplicates().copy()
    cat["priority"] = np.arange(len(cat)) % 3
    return geo, cat


def _query_results(df: pd.DataFrame, master: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    conn = sqlite3.connect(":memory:")
    _write_table(conn, "fact", df)
    geo, cat = _prepare_dims(master)
    geo.to_sql("geo", conn, if_exists="replace", index=False)
    cat.to_sql("catdim", conn, if_exists="replace", index=False)
    queries = {
        "selection_groupby": """
            SELECT category, SUM(revenue) AS revenue, SUM(employees) AS employees
            FROM fact WHERE revenue >= 0 GROUP BY category ORDER BY category
        """,
        "geo_join_groupby": """
            SELECT geo.state AS category, SUM(fact.revenue) AS revenue, COUNT(*) AS employees
            FROM fact JOIN geo ON fact.zip = geo.zip
            GROUP BY geo.state ORDER BY geo.state
        """,
        "category_join_filter": """
            SELECT fact.category AS category, SUM(fact.revenue) AS revenue, SUM(fact.employees) AS employees
            FROM fact JOIN catdim ON fact.category = catdim.category
            WHERE catdim.priority >= 0 GROUP BY fact.category ORDER BY fact.category
        """,
        "having_top_groups": """
            SELECT category, SUM(revenue) AS revenue, COUNT(*) AS employees
            FROM fact GROUP BY category HAVING SUM(revenue) >= 0 ORDER BY revenue DESC LIMIT 5
        """,
    }
    out: Dict[str, Dict[str, float]] = {}
    for qname, sql in queries.items():
        qr = pd.read_sql_query(sql, conn)
        out[qname] = {str(r["category"]): float(r["revenue"]) for _, r in qr.iterrows()}
    conn.close()
    return out


def _l1_distortion(truth: Dict[str, float], pred: Dict[str, float]) -> float:
    keys = sorted(set(truth) | set(pred))
    denom = sum(abs(truth.get(k, 0.0)) for k in keys)
    err = sum(abs(pred.get(k, 0.0) - truth.get(k, 0.0)) for k in keys)
    return err / max(1.0, denom)


def evaluate_sql_workflow(dataset: str, budget: float, seed: int, methods: Iterable[str]) -> List[Dict]:
    master = master_from_real_dataset(dataset)
    scenario = Scenario(n_entities=len(master), n_sources=12, coverage=0.72, seed=seed)
    clean_records, meta = generate_sources(master, scenario)
    records, meta_c, _attack = apply_corruption(master, clean_records, meta, budget=budget, seed=seed, mode="coordinated")
    truth_q = _query_results(master, master)
    rows = []
    for method in methods:
        out, det = integrate(records, meta_c, method, seed=seed)
        qres = _query_results(out, master)
        for qname in sorted(truth_q):
            rows.append({
                "suite": "sql_workflow",
                "dataset": dataset,
                "budget": float(budget),
                "seed": int(seed),
                "method": method,
                "query": qname,
                "query_l1_distortion": _l1_distortion(truth_q[qname], qres[qname]),
                "runtime_sec": float(det.get("runtime_sec", 0.0)),
                "n_entities": int(len(master)),
            })
    return rows


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric = [c for c in df.columns if c not in set(group_cols) and pd.api.types.is_numeric_dtype(df[c])]
    agg = {c: ["mean", "std"] for c in numeric}
    out = df.groupby(group_cols, as_index=False).agg(agg)
    out.columns = ["_".join([p for p in col if p]) for col in out.columns.to_flat_index()]
    return out


def evaluate_external_and_sql_trial(dataset: str, budget: float, seed: int, methods: Iterable[str]) -> Tuple[List[Dict], List[Dict]]:
    master = master_from_real_dataset(dataset)
    scenario = Scenario(n_entities=len(master), n_sources=12, coverage=0.72, seed=seed)
    clean_records, meta = generate_sources(master, scenario)
    records, meta_c, _attack = apply_corruption(master, clean_records, meta, budget=budget, seed=seed, mode="coordinated")
    truth_q = _query_results(master, master)
    real_rows: List[Dict] = []
    sql_rows: List[Dict] = []
    out0, det0 = integrate(records, meta_c, "no_repair", seed=seed)
    base = evaluate(master, records, out0, det0["runtime_sec"])
    cache = {"no_repair": (out0, det0, base)}
    for method in methods:
        if method not in cache:
            out, det = integrate(records, meta_c, method, seed=seed)
            met = evaluate(master, records, out, det["runtime_sec"], baseline_distortion=base["aggregate_distortion"])
            cache[method] = (out, det, met)
        out, det, met = cache[method]
        row = {
            "suite": "external_tabular",
            "dataset": dataset,
            "budget": float(budget),
            "seed": int(seed),
            "method": method,
            "n_entities": int(len(master)),
            "n_sources": 12,
            "n_records": int(len(records)),
            "n_attacked_records": int(records["attacked"].sum()),
            "description": DATASET_DESCRIPTIONS.get(dataset, dataset),
        }
        row.update(met)
        real_rows.append(row)
        qres = _query_results(out, master)
        for qname in sorted(truth_q):
            sql_rows.append({
                "suite": "sql_workflow",
                "dataset": dataset,
                "budget": float(budget),
                "seed": int(seed),
                "method": method,
                "query": qname,
                "query_l1_distortion": _l1_distortion(truth_q[qname], qres[qname]),
                "runtime_sec": float(det.get("runtime_sec", 0.0)),
                "n_entities": int(len(master)),
            })
    return real_rows, sql_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "diabetes", "digits", "mpg", "diamonds", "txhousing", "midwest", "penguins"])
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    ap.add_argument("--seeds", nargs="+", type=int, default=[21, 22, 23, 24, 25])
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["no_repair", "majority", "iterative_truth", "dependency_truth", "parc"]

    real_rows: List[Dict] = []
    sql_rows: List[Dict] = []
    for ds in args.datasets:
        for b in args.budgets:
            for seed in args.seeds:
                rr, sr = evaluate_external_and_sql_trial(ds, b, seed, methods)
                real_rows.extend(rr)
                sql_rows.extend(sr)

    real = pd.DataFrame(real_rows)
    sql = pd.DataFrame(sql_rows)
    real.to_csv(out_dir / "external_tabular_results.csv", index=False)
    summarize(real, ["dataset", "budget", "method"]).to_csv(out_dir / "external_tabular_summary.csv", index=False)
    summarize(real, ["budget", "method"]).to_csv(out_dir / "external_tabular_overall.csv", index=False)
    sql.to_csv(out_dir / "sql_workflow_results.csv", index=False)
    summarize(sql, ["dataset", "budget", "method", "query"]).to_csv(out_dir / "sql_workflow_summary.csv", index=False)
    summarize(sql, ["budget", "method"]).to_csv(out_dir / "sql_workflow_overall.csv", index=False)
    print(f"wrote {len(real)} external rows and {len(sql)} SQL-workflow rows to {out_dir}")


if __name__ == "__main__":
    main()
