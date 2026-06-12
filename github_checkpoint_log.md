# GitHub Checkpoint Log

## 2026-06-12

- Branch: `main`
- Repository: `https://github.com/hyeliozhang/PARC`
- Remote status: public repository, default branch `main`, verified after upload.
- User trigger: post-submission request to prepare the public availability repository.
- Pushed files summary: implementation, benchmark scripts, frozen result CSVs, figure assets, tests, and reproducibility metadata.
- Excluded files: manuscript source/PDF, bibliography, IEEE template files, submission zip/folder, local caches, build auxiliaries, patch files, private notes, and credentials.
- Checks before push: pytest, artifact audit, project-isolation audit, secret scan, forbidden-file scan, and clean git status.
- Residual risks: full benchmark reruns may overwrite committed result summaries; run them in a clean branch or cloud checkout.
