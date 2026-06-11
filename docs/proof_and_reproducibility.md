# Proof And Reproducibility

The artifact separates three reproducibility layers:

- Core implementation: `robust_repair/`.
- Experiment and analysis scripts: `scripts/`.
- Frozen paper-result summaries: `data/results/`.

For a quick repository check, run:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/audit_artifact.py --root .
```

For full regeneration, use a clean cloud/workstation environment and run the relevant scripts under `scripts/`. Record the commit hash, Python version, dependency versions, and any changed random seeds before comparing regenerated CSVs with the committed summaries.

