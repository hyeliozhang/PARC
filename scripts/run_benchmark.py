#!/usr/bin/env python3
"""Run the PARC synthetic relational repair benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_repair.corruption import ALL_OPS, apply_corruption
from robust_repair.generator import generate_scenario
from robust_repair.metrics import evaluate
from robust_repair.repair import integrate

MAIN_METHODS = [
    "no_repair", "majority", "standard_fd", "rule_only", "random_weighting",
    "outlier_removal", "provenance_unaware", "iterative_truth", "source_dependence",
    "dependency_truth", "parc",
]
ABLATION_METHODS = [
    "parc", "parc_no_prov", "parc_no_constraints", "parc_no_trust",
    "parc_no_schema", "parc_name_blocks",
]
OPERATOR_METHODS = ["majority", "dependency_truth", "parc"]
TRUST_METHODS = ["provenance_unaware", "dependency_truth", "parc"]


def _pairwise_group_metrics(group_map: dict, meta: pd.DataFrame) -> dict:
    """Precision/recall of inferred source-dependence groups against generator truth."""
    if not group_map or "true_group" not in meta.columns:
        return {"group_precision": 0.0, "group_recall": 0.0, "group_f1": 0.0}
    truth = {str(r.source): str(r.true_group) for r in meta.itertuples(index=False)}
    sources = sorted(set(map(str, group_map.keys())) & set(truth.keys()))
    tp = fp = fn = 0
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            pred_same = str(group_map.get(a)) == str(group_map.get(b))
            true_same = truth[a] == truth[b]
            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif (not pred_same) and true_same:
                fn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {"group_precision": prec, "group_recall": rec, "group_f1": f1}


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        "cell_accuracy", "integrated_quality", "aggregate_distortion", "data_quality_score",
        "attack_success_reduction", "entity_precision", "entity_recall", "entity_f1",
        "group_precision", "group_recall", "group_f1",
        "duplicate_output_rate", "missing_entity_rate", "false_positive_rate",
        "false_negative_rate", "repair_precision", "repair_recall", "runtime_sec",
        "n_output_rows", "n_records", "n_attacked_records", "n_duplicates",
    ]
    agg = {m: ["mean", "std"] for m in metrics if m in df.columns}
    summary = df.groupby(group_cols, as_index=False).agg(agg)
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.to_flat_index()]
    return summary


def run_once(
    n_entities: int,
    n_sources: int,
    budget: float,
    seed: int,
    methods: Optional[Iterable[str]] = None,
    enabled_ops: Optional[Iterable[str]] = None,
    trust_alpha: float = 1.0,
    mode: str = "coordinated",
) -> tuple[list[dict], dict]:
    if methods is None:
        methods = MAIN_METHODS
    methods = list(methods)
    master, clean_records, meta = generate_scenario(n_entities=n_entities, n_sources=n_sources, seed=seed)
    records, meta_c, attack = apply_corruption(
        master, clean_records, meta, budget=budget, seed=seed, enabled_ops=enabled_ops, trust_alpha=trust_alpha, mode=mode
    )
    outputs = {}
    details = {}
    out0, det0 = integrate(records, meta_c, "no_repair", seed=seed)
    m0 = evaluate(master, records, out0, det0["runtime_sec"])
    base_dist = m0["aggregate_distortion"]
    rows = []
    for method in methods:
        if method == "no_repair":
            out, det, met = out0, det0, m0
        else:
            out, det = integrate(records, meta_c, method, seed=seed)
            met = evaluate(master, records, out, det["runtime_sec"], baseline_distortion=base_dist)
        row = {
            "method": method,
            "budget": float(budget),
            "seed": int(seed),
            "n_entities": int(n_entities),
            "n_sources": int(n_sources),
            "n_records": int(len(records)),
            "n_attacked_records": int(records["attacked"].sum()),
            "n_duplicates": int(records["is_duplicate"].sum()),
            "operators": "+".join(sorted(enabled_ops)) if enabled_ops is not None else "all",
            "trust_alpha": float(trust_alpha),
            "attack_mode": mode,
        }
        row.update(met)
        row.update(_pairwise_group_metrics(det.get("dependency_groups", {}), meta_c))
        rows.append(row)
        outputs[method] = out
        compact = {k: v for k, v in det.items() if k != "certificates"}
        compact["certificate_sample"] = None
        certs = det.get("certificates", {})
        if certs:
            first_key = sorted(certs.keys())[0]
            compact["certificate_sample"] = {first_key: certs[first_key]}
        details[method] = compact
    evidence = {
        "attack": {k: (v[:12] if isinstance(v, list) else v) for k, v in attack.items()},
        "details": details,
        "sample_parc_output": outputs.get("parc", pd.DataFrame()).head(5).to_dict(orient="records"),
        "sample_records": records.head(8).to_dict(orient="records"),
    }
    return rows, evidence


def run_main(args, out_dir: Path) -> pd.DataFrame:
    all_rows = []
    evidence = None
    for b in args.budgets:
        for seed in args.seeds:
            rows, ev = run_once(args.entities, args.sources, float(b), int(seed), methods=MAIN_METHODS)
            all_rows.extend(rows)
            if evidence is None and abs(float(b) - 0.20) < 1e-9:
                evidence = ev
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "benchmark_results.csv", index=False)
    _summarize(df, ["budget", "method"]).to_csv(out_dir / "benchmark_summary.csv", index=False)
    if evidence is None:
        _, evidence = run_once(args.entities, args.sources, 0.20, int(args.seeds[0]), methods=MAIN_METHODS)
    with open(out_dir / "evidence_sample.json", "w") as f:
        json.dump(evidence, f, indent=2)
    return df


def run_ablation(args, out_dir: Path) -> None:
    rows = []
    for seed in args.seeds:
        r, _ = run_once(args.entities, args.sources, 0.30, int(seed) + 7000, methods=ABLATION_METHODS)
        rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    _summarize(df, ["budget", "method"]).to_csv(out_dir / "ablation_summary.csv", index=False)


def run_operator_ablation(args, out_dir: Path) -> None:
    rows = []
    operator_sets: list[tuple[str, Optional[list[str]]]] = [("all", None)] + [(op, [op]) for op in sorted(ALL_OPS)]
    # Include a realistic metadata-only case: trust and provenance manipulated together.
    operator_sets.append(("trust+provenance", ["trust", "provenance"]))
    for label, ops in operator_sets:
        for seed in args.seeds[:3]:
            r, _ = run_once(args.entities, args.sources, 0.25, int(seed) + 9000, methods=OPERATOR_METHODS, enabled_ops=ops)
            for row in r:
                row["operator_label"] = label
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "operator_ablation_results.csv", index=False)
    _summarize(df, ["operator_label", "method"]).to_csv(out_dir / "operator_ablation_summary.csv", index=False)


def run_trust_sensitivity(args, out_dir: Path) -> None:
    rows = []
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for seed in args.seeds[:3]:
            r, _ = run_once(args.entities, args.sources, 0.20, int(seed) + 11000, methods=TRUST_METHODS, trust_alpha=alpha)
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "trust_sensitivity_results.csv", index=False)
    _summarize(df, ["trust_alpha", "method"]).to_csv(out_dir / "trust_sensitivity_summary.csv", index=False)


def run_stress_modes(args, out_dir: Path) -> None:
    rows = []
    for mode in ["coordinated", "independent", "sybil"]:
        for seed in args.seeds[:3]:
            r, _ = run_once(
                args.entities, args.sources, 0.30, int(seed) + 15000,
                methods=["no_repair", "majority", "iterative_truth", "source_dependence", "dependency_truth", "parc"],
                mode=mode,
            )
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stress_modes_results.csv", index=False)
    _summarize(df, ["attack_mode", "method"]).to_csv(out_dir / "stress_modes_summary.csv", index=False)


def run_scalability(args, out_dir: Path) -> None:
    scale_rows = []
    for n in args.scale_entities:
        for seed in args.seeds[:3]:
            r, _ = run_once(int(n), args.sources, 0.20, int(seed) + 13000, methods=["majority", "dependency_truth", "parc"])
            scale_rows.extend(r)
    pd.DataFrame(scale_rows).to_csv(out_dir / "scalability_results.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "results"))
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--sources", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--scale-entities", type=int, nargs="+", default=[150, 300, 600, 900])
    ap.add_argument("--skip-extra", action="store_true", help="only run the main budget sweep")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_main(args, out_dir)
    if not args.skip_extra:
        run_ablation(args, out_dir)
        run_operator_ablation(args, out_dir)
        run_trust_sensitivity(args, out_dir)
        run_stress_modes(args, out_dir)
        run_scalability(args, out_dir)
    print(f"Wrote {out_dir / 'benchmark_results.csv'} with {len(df)} main rows")


if __name__ == "__main__":
    main()
