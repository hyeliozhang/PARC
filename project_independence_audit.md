# Project Independence Audit

- Project: PARC
- compare_root: none
compare_root: none
- Local repository path: local submission workspace, `PARC_REPO_READY`
- GitHub repository: `https://github.com/hyeliozhang/PARC`
- Intended cloud root: `/data/ICDE2027/PARC`
- Environment scope: project-specific Python environment under `/data/ICDE2027/PARC/env`
- Data scope: project-local committed result summaries under `data/results/`; any regenerated data should live under `/data/ICDE2027/PARC/data` or `/data/ICDE2027/PARC/runs`
- Cache scope: local and cloud caches are ignored and must not be shared as mutable state across projects
- Paper tree: local-only in the submission workspace; excluded from GitHub and cloud sync
- Bibliography scope: local-only in the submission workspace; excluded from GitHub and cloud sync
- Figure scope: public generated figure assets are committed under this repository's `figures/`
- Review logs: public artifact notes under `docs/`; private submission logs remain outside the repository

No active shared mutable repo, environment, cache, run directory, paper tree, or bibliography path is configured for this project.
