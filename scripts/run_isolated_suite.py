#!/usr/bin/env python3
"""Run the stronger PARC benchmark suite through isolated subprocesses."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _collect(files: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def _run_many(tasks: list[list[str]], workers: int) -> None:
    print(f"Running {len(tasks)} missing trials with workers={workers}", flush=True)
    if workers <= 1:
        for i, cmd in enumerate(tasks, 1):
            print(f"  trial {i}/{len(tasks)}: {Path(cmd[cmd.index('--out') + 1]).name}", flush=True)
            _run(cmd)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run, cmd) for cmd in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                print(f"  completed {i}/{len(tasks)}", flush=True)


def _complete(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 64


def trial_cmd(path: Path, entities: int, sources: int, budget: float, seed: int, methods: list[str], *, mode="coordinated", ops=None, trust_alpha=1.0, extra=None) -> list[str]:
    cmd = [sys.executable, "scripts/run_trial.py", "--out", str(path), "--entities", str(entities), "--sources", str(sources), "--budget", str(budget), "--seed", str(seed), "--methods", *methods, "--mode", mode, "--trust-alpha", str(trust_alpha)]
    if ops is not None:
        cmd.extend(["--ops", *list(ops)])
    if extra:
        cmd.extend(["--extra", *[f"{k}={v}" for k, v in extra.items()]])
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--entities", type=int, default=400)
    ap.add_argument("--sources", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--scale-entities", type=int, nargs="+", default=[200, 400, 800, 1600])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-extra", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    tmp = out_dir / "_trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    main_files = []
    tasks = []
    for b in args.budgets:
        for seed in args.seeds:
            path = tmp / f"main_b{b:.2f}_s{seed}.csv"
            main_files.append(path)
            if not _complete(path):
                tasks.append(trial_cmd(path, args.entities, args.sources, float(b), int(seed), MAIN_METHODS))
    _run_many(tasks, args.workers)
    df = _collect(main_files).sort_values(["budget", "seed", "method"]).reset_index(drop=True)
    df.to_csv(out_dir / "benchmark_results.csv", index=False)
    _summarize(df, ["budget", "method"]).to_csv(out_dir / "benchmark_summary.csv", index=False)

    # Evidence sample is generated once and intentionally not part of the repeated-trial loop.
    _, evidence = run_once(args.entities, args.sources, 0.20, int(args.seeds[0]), methods=MAIN_METHODS)
    with open(out_dir / "evidence_sample.json", "w") as f:
        json.dump(evidence, f, indent=2)

    if not args.skip_extra:
        ab_files, tasks = [], []
        for seed in args.seeds:
            path = tmp / f"ablation_s{seed}.csv"
            ab_files.append(path)
            if not _complete(path):
                tasks.append(trial_cmd(path, args.entities, args.sources, 0.30, int(seed) + 7000, ABLATION_METHODS))
        _run_many(tasks, args.workers)
        ab = _collect(ab_files)
        ab.to_csv(out_dir / "ablation_results.csv", index=False)
        _summarize(ab, ["budget", "method"]).to_csv(out_dir / "ablation_summary.csv", index=False)

        op_files, tasks = [], []
        operator_sets = [("all", None)] + [(op, [op]) for op in sorted(ALL_OPS)] + [("trust+provenance", ["trust", "provenance"])]
        for label, ops in operator_sets:
            for seed in args.seeds[:3]:
                path = tmp / f"op_{label}_s{seed}.csv".replace("+", "_")
                op_files.append(path)
                if not _complete(path):
                    tasks.append(trial_cmd(path, args.entities, args.sources, 0.25, int(seed) + 9000, OPERATOR_METHODS, ops=ops, extra={"operator_label": label}))
        _run_many(tasks, args.workers)
        op = _collect(op_files)
        op.to_csv(out_dir / "operator_ablation_results.csv", index=False)
        _summarize(op, ["operator_label", "method"]).to_csv(out_dir / "operator_ablation_summary.csv", index=False)

        tr_files, tasks = [], []
        for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
            for seed in args.seeds[:3]:
                path = tmp / f"trust_a{alpha:.2f}_s{seed}.csv"
                tr_files.append(path)
                if not _complete(path):
                    tasks.append(trial_cmd(path, args.entities, args.sources, 0.20, int(seed) + 11000, TRUST_METHODS, trust_alpha=float(alpha)))
        _run_many(tasks, args.workers)
        tr = _collect(tr_files)
        tr.to_csv(out_dir / "trust_sensitivity_results.csv", index=False)
        _summarize(tr, ["trust_alpha", "method"]).to_csv(out_dir / "trust_sensitivity_summary.csv", index=False)

        st_files, tasks = [], []
        stress_methods = ["no_repair", "majority", "iterative_truth", "source_dependence", "dependency_truth", "parc"]
        for mode in ["coordinated", "independent", "sybil"]:
            for seed in args.seeds[:3]:
                path = tmp / f"stress_{mode}_s{seed}.csv"
                st_files.append(path)
                if not _complete(path):
                    tasks.append(trial_cmd(path, max(100, args.entities // 2), args.sources, 0.30, int(seed) + 15000, stress_methods, mode=mode))
        _run_many(tasks, args.workers)
        st = _collect(st_files)
        st.to_csv(out_dir / "stress_modes_results.csv", index=False)
        _summarize(st, ["attack_mode", "method"]).to_csv(out_dir / "stress_modes_summary.csv", index=False)

        sc_files, tasks = [], []
        for n in args.scale_entities:
            for seed in args.seeds[:3]:
                path = tmp / f"scale_n{n}_s{seed}.csv"
                sc_files.append(path)
                if not _complete(path):
                    tasks.append(trial_cmd(path, int(n), args.sources, 0.20, int(seed) + 13000, ["majority", "dependency_truth", "parc"]))
        _run_many(tasks, args.workers)
        sc = _collect(sc_files)
        sc.to_csv(out_dir / "scalability_results.csv", index=False)
        _summarize(sc, ["n_entities", "method"]).to_csv(out_dir / "scalability_summary.csv", index=False)

    print(f"Wrote {out_dir / 'benchmark_results.csv'} with {len(df)} rows")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
