# Robustness Audit

The public artifact includes checks for the following review risks.

- SQL-fragment tests exercise repaired outputs without reading hidden ground-truth columns.
- Held-out, stress-mode, metadata-perturbation, and adversarial-search result tables are committed as CSV summaries.
- Certificate and contract tables include PARC certificate behavior as well as baseline point-output behavior.
- The artifact audit excludes manuscript files, archive packages, runtime caches, local absolute paths, internal workflow notes, and common credential patterns.
- Result tables are versioned directly because they are small enough for normal Git review.

Residual risk: full benchmark scripts overwrite files under `data/results/` by default. Use a clean branch or a separate output directory when comparing a regenerated run against the committed evidence tables.
