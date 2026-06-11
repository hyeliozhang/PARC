# Cleanup Checklist

- Removed from public repository scope: manuscript source/PDF, bibliography, IEEE template files, submission package, patch file, and local paper-review notes.
- Excluded by ignore rules: `paper/`, secrets, Python caches, virtual environments, LaTeX auxiliaries, archives, scratch directories, and local run directories.
- Retained intentionally: implementation code, experiment scripts, frozen result CSVs, figure assets, tests, and reproducibility metadata.
- Pre-push gates: run pytest, run artifact audit, scan for token/key patterns, scan for forbidden manuscript/submission files, and verify clean git status.

