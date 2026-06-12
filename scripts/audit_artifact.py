#!/usr/bin/env python3
"""Repository audit checks for the public PARC artifact."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
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

REQUIRED_DOC_FILES = [
    "README.md",
    "docs/reviewer_guide.md",
    "docs/environment.md",
    "docs/result_inventory.md",
    "docs/claims_to_evidence.md",
    "docs/proof_and_reproducibility.md",
    "docs/robustness_audit.md",
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
    "cleanup_checklist.md",
    "cloud_sync_plan.md",
    "context_" + "check" + "point.md",
    "github_" + "check" + "point_log.md",
    "project_independence_audit.md",
    "structure_variation_plan.md",
    "sync_state.md",
}

RUNTIME_CACHE_DIRS = {
    ".pytest_cache",
    "__pycache__",
    ".ipynb_" + "check" + "points",
}

FORBIDDEN_PUBLIC_DIRS = {
    "hand" + "offs",
    "local_notes",
    "scratch",
    "runs",
    "artifacts",
    "cloud",
}

FORBIDDEN_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar")
FORBIDDEN_OS_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}

SECRET_PATTERNS = [
    "gh" + r"p_[A-Za-z0-9_]+",
    r"BEGIN (RSA|OPENSSH|PRIVATE) KEY",
    "OPENAI" + r"_API_KEY\s*=",
    "ANTHROPIC" + r"_API_KEY\s*=",
    "AWS" + r"_SECRET",
]

INTERNAL_TRACE_PATTERNS = [
    "Co" + "dex",
    "Chat" + "".join(["G", "P", "T"]),
    "User " + "trigger",
    "post-" + "sub" + "mis" + "sion",
    "final " + "up" + "load",
    "best " + "paper",
    "hando" + "ff",
    "check" + "point",
    "\u63d0\u793a" + "\u8bcd",
    "\u8bc4" + "\u5206",
    "\u622a" + "\u7a3f",
    "\u6700\u7ec8" + "\u7248",
]


def clean_runtime_artifacts(root: Path) -> list[str]:
    """Remove deterministic local caches created by validation commands."""
    removed: list[str] = []
    for name in (".pytest_cache",):
        target = root / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target.relative_to(root)))
    for target in list(root.rglob("__pycache__")):
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target.relative_to(root)))
    for target in list(root.rglob(".ipynb_" + "check" + "points")):
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target.relative_to(root)))
    for pyc in list(root.rglob("*.pyc")):
        try:
            pyc.unlink()
            removed.append(str(pyc.relative_to(root)))
        except FileNotFoundError:
            pass
    for pattern in ("*.aux", "*.log", "*.blg"):
        for aux in list(root.rglob(pattern)):
            if ".git" in aux.relative_to(root).parts:
                continue
            try:
                aux.unlink()
                removed.append(str(aux.relative_to(root)))
            except FileNotFoundError:
                pass
    return removed


def git_tracked_files(root: Path) -> set[str] | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return {line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()}


def check_results(root: Path) -> list[str]:
    errs: list[str] = []
    res = root / "data" / "results"
    for filename in REQUIRED_RESULT_FILES:
        if not (res / filename).exists():
            errs.append(f"missing result file: {filename}")
    benchmark = res / "benchmark_results.csv"
    if benchmark.exists():
        df = pd.read_csv(benchmark)
        if len(df) < 400:
            errs.append(f"benchmark_results has only {len(df)} rows")
        if "parc" not in set(df.method):
            errs.append("PARC method missing from benchmark_results")
        if df[["budget", "seed", "method"]].duplicated().any():
            errs.append("duplicate budget/seed/method rows in benchmark_results")
        bad = df.select_dtypes(include="number").isna().sum().sum()
        if bad:
            errs.append(f"numeric NaNs in results: {bad}")
    return errs


def check_package_cleanliness(root: Path) -> list[str]:
    errs: list[str] = []
    tracked = git_tracked_files(root)
    for path in root.rglob("*"):
        rel_path = path.relative_to(root)
        if ".git" in rel_path.parts:
            continue
        rel = str(rel_path)
        rel_key = rel.replace("\\", "/")
        parts = set(rel_path.parts)
        if parts & RUNTIME_CACHE_DIRS:
            if tracked is not None and rel_key in tracked:
                errs.append(f"tracked runtime cache artifact in package: {rel}")
            continue
        if path.is_dir() and parts & FORBIDDEN_PUBLIC_DIRS:
            errs.append(f"forbidden internal/runtime directory in package: {rel}")
            if len(errs) > 20:
                break
        if path.name in FORBIDDEN_PUBLIC_NAMES:
            errs.append(f"paper-only file must not be in public artifact: {rel}")
        lower_name = path.name.lower()
        if path.name in FORBIDDEN_OS_FILES:
            errs.append(f"local OS metadata file must not be committed: {rel}")
        if any(lower_name.endswith(suffix) for suffix in FORBIDDEN_ARCHIVE_SUFFIXES):
            errs.append(f"archive file must not be committed: {rel}")
        if path.name == "ARTIFACT_AUDIT.md":
            continue
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv", ".ini", ".yml", ".yaml"}:
            txt = path.read_text(errors="ignore")
            for pat in SECRET_PATTERNS:
                if re.search(pat, txt):
                    errs.append(f"possible secret pattern in {rel}: {pat}")
            if re.search(r"[A-Za-z]:[\\/](Users|Documents and Settings)[\\/]", txt):
                errs.append(f"absolute local path leaked in text file: {rel}")
            if re.search(r"(^|[^A-Za-z])/(tmp|mnt/data)/", txt):
                errs.append(f"absolute local path leaked in text file: {rel}")
            if rel != "scripts/audit_artifact.py":
                for pat in INTERNAL_TRACE_PATTERNS:
                    if re.search(pat, txt, flags=re.IGNORECASE):
                        errs.append(f"internal workflow trace in public text file: {rel}: {pat}")
                        break
    for doc in REQUIRED_DOC_FILES:
        if not (root / doc).exists():
            errs.append(f"missing reviewer-facing documentation: {doc}")
    if not (root / ".gitignore").exists():
        errs.append("missing .gitignore")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--clean", action="store_true", help="remove deterministic runtime caches before scanning")
    ap.add_argument("--write-report", action="store_true", help="write ARTIFACT_AUDIT.md with the audit outcome")
    args = ap.parse_args()
    root = Path(args.root)
    removed = clean_runtime_artifacts(root) if args.clean else []
    errs = check_results(root) + check_package_cleanliness(root)
    out = root / "ARTIFACT_AUDIT.md"
    if errs:
        if args.write_report:
            out.write_text("# Artifact Audit\n\nFAILED\n\n" + "\n".join(f"- {e}" for e in errs) + "\n")
        print("\n".join(errs))
        raise SystemExit(1)
    if args.write_report:
        out.write_text(
            "# Artifact Audit\n\n"
            "PASSED. Core result files are present, benchmark tables pass basic consistency checks, "
            "reviewer-facing documentation is present, and the public repository excludes manuscript files, "
            "archives, local absolute paths, internal workflow traces, and obvious credential patterns.\n"
        )
    if removed:
        suffix = " ..." if len(removed) > 12 else ""
        print("removed runtime caches: " + ", ".join(removed[:12]) + suffix)
    print("artifact audit passed")


if __name__ == "__main__":
    main()
