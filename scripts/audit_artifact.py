#!/usr/bin/env python3
"""Repository audit checks for the public PARC artifact."""
from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path
import pandas as pd

REQUIRED_RESULT_FILES = [
    "benchmark_results.csv",
    "benchmark_summary.csv",
    "ablation_results.csv",
    "scalability_results.csv",
    "paired_statistical_tests.csv",
    "headline_ci_beta35.csv",
    "external_tabular_results.csv",
    "sql_workflow_results.csv",
    "sql_fragment_results.csv",
    "sql_fragment_overall.csv",
    "contract_grid_paper.csv",
    "contract_totalnorm_summary.csv",
    "certified_query_bounds_paper.csv",
]

FORBIDDEN_PUBLIC_NAMES = {
    "main.tex",
    "main.pdf",
    "main.bbl",
    "main.aux",
    "main.blg",
    "main.log",
    "references.bib",
    "IEEEtran.cls",
    "IEEEtran.bst",
    "ICDE2027_SUBMISSION_NO_REPO_READY.zip",
}

SECRET_PATTERNS = [
    "gh" + r"p_[A-Za-z0-9_]+",
    r"BEGIN (RSA|OPENSSH|PRIVATE) KEY",
    "OPENAI" + r"_API_KEY\s*=",
    "ANTHROPIC" + r"_API_KEY\s*=",
    "AWS" + r"_SECRET",
]


def clean_runtime_artifacts(root: Path) -> list[str]:
    """Remove local caches that are created by validation commands.

    The shipped artifact must not contain cache/VCS files, but reviewers often
    run tests before the package audit.  Pytest and Python then create
    `.pytest_cache` and `__pycache__` directories in an otherwise clean tree.
    Removing only these deterministic runtime caches makes the documented
    validation order executable while keeping stale reports and real backups
    forbidden.
    """
    removed: list[str] = []
    for name in ('.pytest_cache',):
        target = root / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target.relative_to(root)))
    for target in list(root.rglob('__pycache__')):
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target.relative_to(root)))
    for pyc in list(root.rglob('*.pyc')):
        try:
            pyc.unlink()
            removed.append(str(pyc.relative_to(root)))
        except FileNotFoundError:
            pass
    # LaTeX and standalone-TikZ builds can create deterministic auxiliaries.
    for pattern in ('*.aux', '*.log', '*.blg'):
        for aux in list(root.rglob(pattern)):
            if ".git" in aux.relative_to(root).parts:
                continue
            try:
                aux.unlink()
                removed.append(str(aux.relative_to(root)))
            except FileNotFoundError:
                pass
    return removed


def check_results(root: Path) -> list[str]:
    errs=[]
    res=root/'data'/'results'
    for r in REQUIRED_RESULT_FILES:
        if not (res/r).exists():
            errs.append(f'missing result file: {r}')
    if (res/'benchmark_results.csv').exists():
        df=pd.read_csv(res/'benchmark_results.csv')
        if len(df) < 400:
            errs.append(f'benchmark_results has only {len(df)} rows')
        if 'parc' not in set(df.method):
            errs.append('PARC method missing from benchmark_results')
        if df[['budget','seed','method']].duplicated().any():
            errs.append('duplicate budget/seed/method rows in benchmark_results')
        bad=df.select_dtypes(include='number').isna().sum().sum()
        if bad:
            errs.append(f'numeric NaNs in results: {bad}')
    return errs


def check_package_cleanliness(root: Path) -> list[str]:
    errs=[]
    forbidden_dirs={'.pytest_cache','__pycache__'}
    for path in root.rglob('*'):
        rel_path = path.relative_to(root)
        if ".git" in rel_path.parts:
            continue
        rel=str(path.relative_to(root))
        parts=set(path.relative_to(root).parts)
        if parts & forbidden_dirs:
            errs.append(f'forbidden cache/VCS artifact in package: {rel}')
            if len(errs) > 20:
                break
        if path.name in FORBIDDEN_PUBLIC_NAMES:
            errs.append(f'paper/submission-only file must not be in public artifact: {rel}')
        if path.suffix.lower() == ".zip":
            errs.append(f'zip archive must not be committed: {rel}')
        if path.name == "ARTIFACT_AUDIT.md":
            continue
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv", ".ini", ".yml", ".yaml"}:
            txt = path.read_text(errors="ignore")
            for pat in SECRET_PATTERNS:
                if re.search(pat, txt):
                    errs.append(f'possible secret pattern in {rel}: {pat}')
            if re.search(r"[A-Za-z]:[\\/](Users|Documents and Settings)[\\/]", txt):
                errs.append(f'absolute local path leaked in text file: {rel}')
            if re.search(r"(^|[^A-Za-z])/(tmp|mnt/data)/", txt):
                errs.append(f'absolute local path leaked in text file: {rel}')
    if not (root / "README.md").exists():
        errs.append("missing README.md")
    if not (root / ".gitignore").exists():
        errs.append("missing .gitignore")
    return errs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args=ap.parse_args()
    root=Path(args.root)
    removed=clean_runtime_artifacts(root)
    errs=check_results(root)+check_package_cleanliness(root)
    out=root/'ARTIFACT_AUDIT.md'
    if errs:
        out.write_text('# Artifact Audit\n\nFAILED\n\n'+'\n'.join(f'- {e}' for e in errs)+'\n')
        print('\n'.join(errs))
        raise SystemExit(1)
    extra = '' if not removed else '\nRemoved runtime caches before audit: ' + ', '.join(removed[:12]) + (' ...' if len(removed) > 12 else '') + '\n'
    out.write_text('# Artifact Audit\n\nPASSED. Core result files are present, benchmark tables pass basic consistency checks, runtime Python caches are cleaned before scanning, and the public repository excludes manuscript/submission files, archives, local absolute paths, and obvious credential patterns.' + extra + '\n')
    print('artifact audit passed')

if __name__=='__main__':
    main()
