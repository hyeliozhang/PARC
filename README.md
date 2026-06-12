# PARC

This repository contains the public artifact for **PARC: Certified Provenance-Bounded Repair for Corrupted Relational Integration**.

PARC treats evidence governance as part of relational integration. It converts source records into provenance-bearing claims, discounts correlated or manipulated support, repairs constraint violations, and emits replayable per-cell certificates. The repository is organized so that reviewers can inspect the implementation, validate the frozen result tables, and rerun the main experiment families from a clean checkout.

## Repository Contents

- `robust_repair/`: core data generation, corruption, constraints, repair operators, certificate construction, and metrics.
- `scripts/`: benchmark, transfer, SQL, certificate, contract, plotting, and artifact-audit entry points.
- `data/results/`: frozen CSV/JSON outputs used by the paper.
- `figures/`: generated figure assets in vector and raster formats.
- `tests/`: lightweight regression and consistency checks for the artifact.
- `docs/`: reviewer guide, environment notes, claim-to-evidence map, and result inventory.

The manuscript source, bibliography, and PDF are outside this public artifact. This repository is limited to code, committed result data, figures, and reproducibility documentation.

## Five-Minute Check

Use a fresh Python environment with Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/audit_artifact.py --root .
```

Expected result: the tests pass and the audit prints `artifact audit passed`.

## Reproducing Result Families

The committed CSV files are the frozen results. The following commands regenerate the main families in-place, so use a clean branch or copy if you want to compare regenerated files with the committed tables.

```powershell
python scripts/run_benchmark.py --out data/results
python scripts/statistical_analysis.py --results data/results
python scripts/run_contract_grid.py --out_dir data/results
python scripts/run_contract_totalnorm.py --out_dir data/results
python scripts/certified_query_bounds.py --out_dir data/results
python scripts/run_external_and_sql_suite.py --out data/results
python scripts/run_sql_fragment_suite.py --out data/results
python scripts/runtime_memory_profile.py --out data/results/runtime_memory_profile.csv
python scripts/plot_results.py --results data/results --figures figures --tables data/results
```

`docs/reviewer_guide.md` gives a shorter review path, while `docs/claims_to_evidence.md` maps empirical claims to result files and scripts.

## Availability

Public URL: https://github.com/hyeliozhang/PARC
