# Structure Variation Plan

PARC uses an artifact-oriented repository layout rather than a manuscript or venue-template layout.

Differentiated dimensions:

- Repository layout: implementation, scripts, frozen results, figures, tests, and sync metadata are at repository root; manuscript files are excluded.
- Module naming: core code lives under `robust_repair/`, matching the repair-and-certificate contribution rather than a generic benchmark scaffold.
- Experiment layout: frozen public summaries are grouped under `data/results/`, with rerun scripts under `scripts/` and cloud reruns documented separately.
- Claim spine: `docs/claims_to_evidence.md` maps empirical claims directly to result files and reproduction entry points.
- Cloud policy: full reruns are isolated under `/data/ICDE2027/PARC` and are not mixed with local manuscript work.

Similarities forced by common tooling: Python package structure, pytest tests, CSV result summaries, and standard Git ignore rules.

