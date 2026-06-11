#!/usr/bin/env python3
"""Parallel full benchmark runner for the PARC artifact.

This script is optional; `run_benchmark.py` remains a simple sequential runner.
The parallel runner is used to regenerate the stronger result set reported in
this version of the paper while keeping each trial deterministic.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_benchmark import (  # noqa: E402
    ABLATION_METHODS,
    MAIN_METHODS,
    OPERATOR_METHODS,
    TRUST_METHODS,
    _summarize,
    run_once,
)
from robust_repair.corruption import ALL_OPS  # noqa: E402


def _task(args_tuple):
    n_entities, n_sources, budget, seed, methods, enabled_ops, trust_alpha, mode, extra = args_tuple
    rows, _ = run_once(
        n_entities, n_sources, budget, seed,
        methods=methods, enabled_ops=enabled_ops, trust_alpha=trust_alpha, mode=mode,
    )
    if extra:
        for r in rows:
            r.update(extra)
    return rows, None


def _run_tasks(tasks, workers: int):
    all_rows = []
    if workers <= 1:
        for t in tasks:
            rows, _ = _task(t)
            all_rows.extend(rows)
    else:
        with mp.Pool(processes=workers) as pool:
            for rows, _ in pool.imap_unordered(_task, tasks):
                all_rows.extend(rows)
    return pd.DataFrame(all_rows), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--sources", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--scale-entities", type=int, nargs="+", default=[150, 300, 600, 900])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-extra", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_tasks = [
        (args.entities, args.sources, float(b), int(seed), MAIN_METHODS, None, 1.0, "coordinated", None)
        for b in args.budgets for seed in args.seeds
    ]
    df, _ = _run_tasks(main_tasks, args.workers)
    df = df.sort_values(["budget", "seed", "method"]).reset_index(drop=True)
    df.to_csv(out_dir / "benchmark_results.csv", index=False)
    _summarize(df, ["budget", "method"]).to_csv(out_dir / "benchmark_summary.csv", index=False)
    _, evidence = run_once(args.entities, args.sources, 0.20, int(args.seeds[0]), methods=MAIN_METHODS)
    with open(out_dir / "evidence_sample.json", "w") as f:
        json.dump(evidence, f, indent=2)

    if not args.skip_extra:
        ab_tasks = [
            (args.entities, args.sources, 0.30, int(seed) + 7000, ABLATION_METHODS, None, 1.0, "coordinated", None)
            for seed in args.seeds
        ]
        ab, _ = _run_tasks(ab_tasks, args.workers)
        ab.to_csv(out_dir / "ablation_results.csv", index=False)
        _summarize(ab, ["budget", "method"]).to_csv(out_dir / "ablation_summary.csv", index=False)

        operator_sets: list[tuple[str, Optional[list[str]]]] = [("all", None)] + [(op, [op]) for op in sorted(ALL_OPS)]
        operator_sets.append(("trust+provenance", ["trust", "provenance"]))
        op_tasks = [
            (args.entities, args.sources, 0.25, int(seed) + 9000, OPERATOR_METHODS, ops, 1.0, "coordinated", {"operator_label": label})
            for label, ops in operator_sets for seed in args.seeds[:3]
        ]
        op, _ = _run_tasks(op_tasks, args.workers)
        op.to_csv(out_dir / "operator_ablation_results.csv", index=False)
        _summarize(op, ["operator_label", "method"]).to_csv(out_dir / "operator_ablation_summary.csv", index=False)

        trust_tasks = [
            (args.entities, args.sources, 0.20, int(seed) + 11000, TRUST_METHODS, None, float(alpha), "coordinated", None)
            for alpha in [0.0, 0.25, 0.50, 0.75, 1.0] for seed in args.seeds[:3]
        ]
        tr, _ = _run_tasks(trust_tasks, args.workers)
        tr.to_csv(out_dir / "trust_sensitivity_results.csv", index=False)
        _summarize(tr, ["trust_alpha", "method"]).to_csv(out_dir / "trust_sensitivity_summary.csv", index=False)

        stress_tasks = [
            (args.entities, args.sources, 0.30, int(seed) + 15000,
             ["no_repair", "majority", "iterative_truth", "source_dependence", "dependency_truth", "parc"],
             None, 1.0, mode, None)
            for mode in ["coordinated", "independent", "sybil"] for seed in args.seeds[:3]
        ]
        st, _ = _run_tasks(stress_tasks, args.workers)
        st.to_csv(out_dir / "stress_modes_results.csv", index=False)
        _summarize(st, ["attack_mode", "method"]).to_csv(out_dir / "stress_modes_summary.csv", index=False)

        scale_tasks = [
            (int(n), args.sources, 0.20, int(seed) + 13000,
             ["majority", "dependency_truth", "parc"], None, 1.0, "coordinated", None)
            for n in args.scale_entities for seed in args.seeds[:3]
        ]
        sc, _ = _run_tasks(scale_tasks, args.workers)
        sc.to_csv(out_dir / "scalability_results.csv", index=False)
        _summarize(sc, ["n_entities", "method"]).to_csv(out_dir / "scalability_summary.csv", index=False)

    print(f"Wrote {out_dir / 'benchmark_results.csv'} with {len(df)} main rows")


if __name__ == "__main__":
    main()
