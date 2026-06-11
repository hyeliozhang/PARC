#!/usr/bin/env python3
"""Reviewer-style adversarial configuration search for PARC.

This experiment does not tune PARC.  It samples held-out mixtures of corruption
operators, modes, trust inflation levels, and budgets, then reports how often
PARC remains on the Pareto frontier against strong point-repair baselines.  The
purpose is to test robustness of the repair contract beyond the fixed headline
budget sweep.
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_repair.corruption import ALL_OPS  # noqa: E402
from scripts.run_benchmark import _summarize, run_once  # noqa: E402

METHODS = ["no_repair", "dependency_truth", "parc"]


def _operator_mixes(rng: random.Random, n_cases: int) -> list[list[str]]:
    ops = sorted(ALL_OPS)
    fixed = [
        ["duplicate", "provenance", "trust"],
        ["schema", "fd", "outlier"],
        ["key", "duplicate", "schema", "provenance"],
        ["key", "fd", "trust", "provenance", "outlier"],
        ops,
    ]
    mixes = [list(x) for x in fixed]
    while len(mixes) < n_cases:
        k = rng.randint(2, min(len(ops), 6))
        mixes.append(sorted(rng.sample(ops, k)))
    return mixes[:n_cases]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--cases", type=int, default=18)
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--sources", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    budgets = [0.15, 0.20, 0.25, 0.30, 0.35]
    modes = ["coordinated", "independent", "sybil"]
    trust_alphas = [0.25, 0.50, 0.75, 1.0]
    mixes = _operator_mixes(rng, args.cases)
    rows: List[dict] = []
    for i, ops in enumerate(mixes):
        budget = budgets[i % len(budgets)]
        mode = modes[(i // len(budgets)) % len(modes)]
        trust_alpha = trust_alphas[(i * 2 + 1) % len(trust_alphas)]
        seed = args.seed + i * 17
        print(f"case={i:02d} beta={budget} mode={mode} trust={trust_alpha} ops={'+'.join(ops)}", flush=True)
        r, _ = run_once(args.entities, args.sources, budget, seed, methods=METHODS, enabled_ops=ops, trust_alpha=trust_alpha, mode=mode)
        for row in r:
            row.update({
                "search_case": i,
                "operator_mix": "+".join(ops),
                "search_seed": seed,
            })
        rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "adversarial_search_results.csv", index=False)
    _summarize(df, ["method"]).to_csv(out / "adversarial_search_overall.csv", index=False)
    # Per-case frontier diagnostics.
    front_rows = []
    for case, g in df.groupby("search_case"):
        best_iq = float(g["integrated_quality"].max())
        best_dist = float(g["aggregate_distortion"].min())
        parc = g[g["method"] == "parc"].iloc[0]
        dep = g[g["method"] == "dependency_truth"].iloc[0]
        front_rows.append({
            "search_case": int(case),
            "operator_mix": str(parc["operator_mix"]),
            "budget": float(parc["budget"]),
            "attack_mode": str(parc["attack_mode"]),
            "trust_alpha": float(parc["trust_alpha"]),
            "parc_integrated_quality": float(parc["integrated_quality"]),
            "dependency_truth_integrated_quality": float(dep["integrated_quality"]),
            "parc_minus_dependency_truth_iq": float(parc["integrated_quality"] - dep["integrated_quality"]),
            "parc_aggregate_distortion": float(parc["aggregate_distortion"]),
            "dependency_truth_aggregate_distortion": float(dep["aggregate_distortion"]),
            "parc_minus_dependency_truth_distortion": float(parc["aggregate_distortion"] - dep["aggregate_distortion"]),
            "parc_within_001_of_best_iq": bool(best_iq - float(parc["integrated_quality"]) <= 0.01),
            "parc_within_001_of_best_distortion": bool(float(parc["aggregate_distortion"]) - best_dist <= 0.01),
        })
    front = pd.DataFrame(front_rows)
    front.to_csv(out / "adversarial_search_frontier.csv", index=False)
    print(f"wrote {len(df)} method rows and {len(front)} search cases")


if __name__ == "__main__":
    main()
