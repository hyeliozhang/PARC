# Reviewer Guide

This guide is intended for a first pass over the artifact.

## Recommended Order

1. Read `README.md` for the repository boundary and quick commands.
2. Run the five-minute check:

   ```powershell
   python -m pip install -r requirements.txt
   python -m pytest -q
   python scripts/audit_artifact.py --root .
   ```

3. Inspect `docs/claims_to_evidence.md` to map paper claims to CSV files and scripts.
4. Open `data/results/benchmark_summary.csv`, `data/results/headline_ci_beta35.csv`, `data/results/contract_grid_paper.csv`, and `data/results/sql_workflow_overall.csv` for the main result families.
5. Inspect `robust_repair/repair.py`, `robust_repair/constraints.py`, and `robust_repair/metrics.py` for the core implementation.

## What the Quick Check Covers

The test suite checks core repair invariants, certificate behavior, contract-grid consistency, SQL no-leakage behavior, paper-facing result assumptions, and artifact self-containment. The audit script is read-only by default and scans the public tree for missing result files, manuscript-only files, archive packages, local absolute paths, internal workflow traces, and obvious credential patterns. Use `--clean` only when you want it to remove deterministic local caches created by Python or pytest.

## Full Regeneration

Full reruns are optional for an initial review and can overwrite committed CSV files. To preserve the frozen tables, run regeneration in a new checkout or pass an alternate output directory where supported.

Representative commands:

```powershell
python scripts/run_benchmark.py --out data/results
python scripts/statistical_analysis.py --results data/results
python scripts/run_external_and_sql_suite.py --out data/results
python scripts/run_sql_fragment_suite.py --out data/results
python scripts/run_contract_grid.py --out_dir data/results
python scripts/run_contract_totalnorm.py --out_dir data/results
python scripts/certified_query_bounds.py --out_dir data/results
python scripts/certificate_calibration.py --out data/results
python scripts/runtime_memory_profile.py --out data/results/runtime_memory_profile.csv
python scripts/plot_results.py --results data/results --figures figures --tables data/results
```

The external-tabular/SQL suite uses both scikit-learn and plotnine packaged datasets by default. A shorter scikit-learn-only check is:

```powershell
python scripts/run_external_and_sql_suite.py --out data/results --datasets iris wine breast_cancer diabetes digits
```

## Artifact Boundary

The repository contains implementation, frozen result tables, generated figures, tests, and reproducibility notes. It intentionally excludes manuscript source, bibliography files, compiled PDFs, paper archive packages, private notes, local paths, and credentials.
