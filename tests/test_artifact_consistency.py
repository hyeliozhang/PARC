from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_parallel_runner_defaults_match_manuscript():
    txt = (ROOT / "scripts" / "run_full_benchmark.py").read_text()
    assert 'default=300' in txt
    assert 'default=[150, 300, 600, 900]' in txt


def test_adversarial_search_frontier_evidence_present():
    front = pd.read_csv(ROOT / "data" / "results" / "adversarial_search_frontier.csv")
    assert len(front) >= 6
    assert bool(front["parc_within_001_of_best_iq"].all())
    overall = pd.read_csv(ROOT / "data" / "results" / "adversarial_search_overall.csv")
    parc = overall[overall["method"] == "parc"].iloc[0]
    dep = overall[overall["method"] == "dependency_truth"].iloc[0]
    assert float(parc["aggregate_distortion_mean"]) <= float(dep["aggregate_distortion_mean"]) + 1e-12


def test_public_artifact_excludes_manuscript_and_submission_files():
    forbidden = [
        "main.tex",
        "main.pdf",
        "references.bib",
        "main.bbl",
        "ICDE2027_SUBMISSION_NO_REPO_READY.zip",
    ]
    for name in forbidden:
        assert not (ROOT / name).exists(), name
    readme = (ROOT / "README.md").read_text()
    assert "manuscript source and submission PDFs are intentionally excluded" in readme


def test_artifact_audit_passes_after_runtime_cache_generation():
    import subprocess, sys
    # Simulate local runtime caches that appear after pytest.  The audit should
    # remove these deterministic caches and still fail on real stale artifacts.
    (ROOT / '.pytest_cache').mkdir(exist_ok=True)
    (ROOT / 'robust_repair' / '__pycache__').mkdir(exist_ok=True)
    (ROOT / 'robust_repair' / '__pycache__' / 'dummy.pyc').write_bytes(b'cache')
    subprocess.run([sys.executable, 'scripts/audit_artifact.py', '--root', '.'], check=True)


def test_r28_scalability_memory_throughput_files_present():
    src = pd.read_csv(ROOT / "data" / "results" / "source_scale_parc_summary.csv")
    assert sorted(src["n_sources"].astype(int).unique().tolist()) == [6, 12, 18, 24, 30, 36]
    mem = pd.read_csv(ROOT / "data" / "results" / "runtime_memory_profile.csv")
    assert sorted(mem["n_entities"].astype(int).unique().tolist()) == [150, 300, 600, 900, 3000]
    assert float(mem["peak_rss_mb"].max()) < 450.0
