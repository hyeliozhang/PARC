# Proof and Reproducibility

PARC exposes three reproducibility layers.

1. `robust_repair/` contains the implementation of corruption generation, candidate repair, provenance-aware scoring, constraints, and metrics.
2. `scripts/` contains the experiment drivers used to regenerate result families.
3. `data/results/` contains the committed result tables used for paper-facing analysis and figures.

For a quick check:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/audit_artifact.py --root .
```

For full regeneration, start from a clean checkout, record the commit hash and Python version, then run the relevant script from `scripts/`. The scripts use deterministic seeds by default. When changing seeds, budgets, datasets, or methods, write outputs to a separate directory before comparing them with the committed CSV files.

The artifact does not require network access, paid APIs, or private data for the quick checks. Public-tabular transfer experiments use datasets loaded through the standard Python scientific stack; keep dependency versions fixed when reproducing paper-level numbers.
