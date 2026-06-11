# Robustness Audit

PARC was checked against the following artifact-level risks before public release:

- No ground-truth leakage in SQL-fragment integration tests.
- Held-out and stress-mode result tables are included as frozen CSV summaries.
- Certificate and contract tables include both PARC certificates and baseline point outputs.
- Public artifact checks exclude manuscript files, submission archives, runtime caches, and obvious credential patterns.
- The result tables are small enough to be versioned directly with the code.

Residual risk: full benchmark reruns can overwrite `data/results/`; reviewers should keep a clean checkout or branch before regeneration.

