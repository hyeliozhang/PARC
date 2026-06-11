#!/usr/bin/env python3
"""Evaluate whether PARC repair certificates are calibrated.

This is an artifact-only analysis script. It compares certificate margins with
benchmark truth that is withheld from repair algorithms and used only after
integration. The output answers two reviewer questions: do high-margin
certificates correspond to correct repairs, and do low-margin warnings capture
residual errors?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_repair.corruption import apply_corruption  # noqa: E402
from robust_repair.constraints import value_equal  # noqa: E402
from robust_repair.generator import ATTRS, NUM_ATTRS, generate_scenario  # noqa: E402
from robust_repair.repair import integrate  # noqa: E402


def _margin_bin(m: float) -> str:
    if m < 0.25:
        return "[0,0.25)"
    if m < 0.75:
        return "[0.25,0.75)"
    if m < 1.50:
        return "[0.75,1.5)"
    if m < 3.00:
        return "[1.5,3)"
    return ">=3"


def _truth(master: pd.DataFrame) -> Dict[int, dict]:
    return {int(r["eid"]): r.to_dict() for _, r in master.iterrows()}


def trial_rows(n_entities: int, n_sources: int, budget: float, seed: int, method: str) -> List[dict]:
    master, clean_records, meta = generate_scenario(n_entities=n_entities, n_sources=n_sources, seed=seed)
    records, meta_c, attack = apply_corruption(master, clean_records, meta, budget=budget, seed=seed, mode="coordinated")
    output, details = integrate(records, meta_c, method, seed=seed)
    certs = details.get("certificates", {})
    truth = _truth(master)
    rows: List[dict] = []
    for out in output.to_dict(orient="records"):
        cid = str(out.get("cluster"))
        eid = int(out.get("cluster_true_eid", -1))
        if eid < 0 or eid not in truth:
            continue
        cert = certs.get(cid, {})
        for attr in ATTRS:
            cell = cert.get("cells", {}).get(attr, {}) if isinstance(cert, dict) else {}
            margin = float(cell.get("margin", 0.0))
            indep = int(cell.get("independent_groups", 0))
            filtered = int(cell.get("filtered_claims", 0))
            correct = bool(value_equal(attr, out.get(attr), truth[eid][attr]))
            rows.append({
                "method": method,
                "budget": float(budget),
                "seed": int(seed),
                "cluster": cid,
                "attribute": attr,
                "attribute_type": "numeric" if attr in NUM_ATTRS else "categorical",
                "margin": margin,
                "margin_bin": _margin_bin(margin),
                "independent_groups": indep,
                "filtered_claims": filtered,
                "correct": int(correct),
                "incorrect": int(not correct),
                "low_margin": int(margin < 0.75),
                "has_independent_support": int(indep >= 2),
                "n_attacked_records": int(records["attacked"].sum()),
                "n_duplicates": int(records["is_duplicate"].sum()),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--sources", type=int, default=12)
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.20, 0.30, 0.35])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=["parc", "dependency_truth"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for b in args.budgets:
        for seed in args.seeds:
            for method in args.methods:
                rows.extend(trial_rows(args.entities, args.sources, b, seed, method))
    df = pd.DataFrame(rows)
    df.to_csv(out / "certificate_cell_rows.csv", index=False)
    by_bin = df.groupby(["method", "budget", "margin_bin"], as_index=False).agg(
        n_cells=("correct", "size"),
        accuracy=("correct", "mean"),
        error_rate=("incorrect", "mean"),
        mean_margin=("margin", "mean"),
        mean_independent_groups=("independent_groups", "mean"),
        low_margin_rate=("low_margin", "mean"),
    )
    by_bin.to_csv(out / "certificate_margin_calibration.csv", index=False)
    summary = df.groupby(["method", "budget"], as_index=False).agg(
        n_cells=("correct", "size"),
        accuracy=("correct", "mean"),
        error_rate=("incorrect", "mean"),
        low_margin_rate=("low_margin", "mean"),
        error_captured_by_low_margin=("low_margin", lambda x: float(x[df.loc[x.index, "incorrect"].eq(1)].mean()) if int(df.loc[x.index, "incorrect"].sum()) else 0.0),
        high_margin_accuracy=("correct", lambda x: float(x[df.loc[x.index, "margin"].ge(3.0)].mean()) if int(df.loc[x.index, "margin"].ge(3.0).sum()) else 0.0),
        high_margin_count=("margin", lambda x: int(x.ge(3.0).sum())),
    )
    summary.to_csv(out / "certificate_calibration_summary.csv", index=False)
    print(f"wrote {len(df)} certificate-cell rows to {out}")


if __name__ == "__main__":
    main()
