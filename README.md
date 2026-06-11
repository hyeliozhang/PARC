# PARC

This repository contains the public reproducibility artifact for PARC, a provenance-bounded repair method for corrupted relational integration.

The artifact includes:

- `robust_repair/`: implementation of corruption generation, constraint handling, repair, and metrics.
- `scripts/`: benchmark, stress-test, plotting, and audit utilities.
- `data/results/`: frozen result tables used by the submission.
- `figures/`: generated figure assets and source-friendly exports.
- `tests/`: lightweight consistency checks for the public artifact.
- `docs/`: reproducibility, synchronization, and evidence notes.

The manuscript source and submission PDFs are intentionally excluded from the public artifact repository. This keeps the GitHub repository focused on code, result data, and reproducibility metadata.

## Quick Check

Use a fresh Python environment:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/audit_artifact.py --root .
```

Expected result: all tests pass and the artifact audit reports `artifact audit passed`.

## Reproduction Notes

The committed CSV files are the frozen result summaries for the paper. Scripts that regenerate benchmark results are available under `scripts/`; substantial benchmark reruns should be executed in a clean cloud or workstation environment rather than inside the submission workspace.

Representative commands:

```powershell
python scripts/run_benchmark.py --out_dir data/results
python scripts/run_contract_grid.py
python scripts/run_contract_totalnorm.py
python scripts/certified_query_bounds.py --out_dir data/results
python scripts/plot_results.py
```

Some full benchmark scripts may take longer than the quick check and can overwrite result files. Preserve the committed CSVs when comparing reproductions.

## Availability

Public URL: https://github.com/hyeliozhang/PARC
