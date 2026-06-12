from pathlib import Path


def test_public_artifact_manifest_is_self_contained():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "data" / "results" / "certified_query_bounds_manifest.json").read_text()
    tmp_prefix = "/" + "tmp/"
    assert tmp_prefix not in manifest
    assert "data/results/certified_query_bounds_paper.csv" in manifest
    assert (root / "docs" / "claims_to_evidence.md").exists()
    assert (root / "docs" / "reviewer_guide.md").exists()
    assert (root / "docs" / "result_inventory.md").exists()
