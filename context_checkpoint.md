# Context Checkpoint

Active phase: public GitHub artifact package published for PARC.

Current local paths:

- Full submission workspace: local-only parent workspace
- Public artifact staging repository: `PARC_REPO_READY`
- Submitted no-repo package: local-only sibling package, outside the public artifact repository

Current status:

- Clean public repository: `https://github.com/hyeliozhang/PARC`.
- Manuscript source, PDF, references, template files, submission archives, and credentials are excluded from GitHub.
- The ICDE availability URL is the public repository URL above.

Future maintenance:

- Keep result-summary CSVs immutable unless a new reproduction run intentionally replaces them.
- Run `python -m pytest -q` and `python scripts/audit_artifact.py --root .` before any follow-up update.

Skills to load in future sessions: `research-sync-governor`, `auto-research-engineering` if cloud reruns are needed, and `paper-writing-orchestrator` only for local manuscript updates.
