#!/usr/bin/env python3
"""Standard-library regression tests for the core PARC invariants."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from robust_repair.generator import Scenario, generate_master, generate_sources
from robust_repair.corruption import apply_corruption
from robust_repair.repair import _weighted_vote, _infer_dependency_groups, _apply_schema_repair, integrate


class CoreInvariantTests(unittest.TestCase):
    def test_duplicate_support_is_capped_by_group(self) -> None:
        values = ["bad"] * 20 + ["good"] * 2
        sources = [f"S{i:02d}" for i in range(20)] + ["G1", "G2"]
        groups = ["copied_family"] * 20 + ["clean1", "clean2"]
        weights = {s: 1.0 for s in sources}
        _val_raw, _score_raw, scores_raw = _weighted_vote(values, sources, groups, weights, cap_groups=False)
        val_cap, _score_cap, scores_cap = _weighted_vote(values, sources, groups, weights, cap_groups=True)
        self.assertEqual(scores_raw["bad"], 20.0)
        self.assertEqual(scores_cap["bad"], 1.0)
        self.assertEqual(scores_cap["good"], 2.0)
        self.assertEqual(val_cap, "good")

    def test_hidden_truth_columns_are_not_required_for_repair(self) -> None:
        master = generate_master(80, seed=101)
        records, meta = generate_sources(master, Scenario(n_entities=80, n_sources=8, coverage=0.7, seed=101))
        corrupted, meta_c, _ = apply_corruption(master, records, meta, budget=0.2, seed=101)
        # Remove evaluator-only identifiers.  A repair should still run because it
        # uses observed keys/names/source metadata, not the clean master entity id.
        repair_view = corrupted.drop(columns=[c for c in ["eid_true"] if c in corrupted.columns])
        out, details = integrate(repair_view, meta_c, "parc", seed=101)
        self.assertGreater(len(out), 0)
        self.assertIn("runtime_sec", details)
        self.assertNotIn("eid_true", out.columns)

    def test_dependency_group_inference_merges_copied_sources(self) -> None:
        master = generate_master(60, seed=7)
        records, meta = generate_sources(master, Scenario(n_entities=60, n_sources=6, coverage=0.8, seed=7))
        # Force S00 and S04 to share stable fingerprints even if declared groups differ.
        s0 = records[records.source == "S00"].copy()
        s4 = s0.copy()
        s4["source"] = "S04"
        records2 = pd.concat([records[records.source != "S04"], s4], ignore_index=True)
        meta2 = meta.copy()
        meta2.loc[meta2.source == "S04", "declared_group"] = "independent-looking"
        groups = _infer_dependency_groups(records2, meta2, threshold=0.75)
        self.assertEqual(groups["S00"], groups["S04"])

    def test_schema_repair_requires_peer_validation(self) -> None:
        master = generate_master(40, seed=202)
        records, meta = generate_sources(master, Scenario(n_entities=40, n_sources=5, coverage=0.8, seed=202))
        # Make one source look swapped, but keep too few peer-validated rows for
        # the conservative schema repair to act.  The repair should abstain
        # rather than blindly swapping an entire source.
        tiny = records[records.source == "S00"].head(5).copy()
        tmp = tiny["revenue"].copy()
        tiny["revenue"] = tiny["employees"]
        tiny["employees"] = tmp
        mixed = pd.concat([tiny, records[records.source != "S00"]], ignore_index=True)
        repaired, actions = _apply_schema_repair(mixed)
        if "S00" in actions:
            self.assertEqual(actions["S00"].get("rows_repaired", 0), 0)
        self.assertEqual(len(repaired), len(mixed))

    def test_parc_hybrid_preserves_numeric_measure_anchor(self) -> None:
        master = generate_master(100, seed=303)
        records, meta = generate_sources(master, Scenario(n_entities=100, n_sources=10, coverage=0.72, seed=303))
        corrupted, meta_c, _ = apply_corruption(master, records, meta, budget=0.3, seed=303)
        out_default, _ = integrate(corrupted, meta_c, "parc", seed=303)
        out_hybrid, _ = integrate(corrupted, meta_c, "parc_hybrid", seed=303)
        # Hybrid may repair symbolic entity evidence, but it should not create
        # new numeric measure values that were absent from observed-key output.
        default_revenues = set(map(int, out_default["revenue"].dropna().astype(int).tolist()))
        hybrid_revenues = set(map(int, out_hybrid["revenue"].dropna().astype(int).tolist()))
        self.assertTrue(hybrid_revenues.issubset(default_revenues))


if __name__ == "__main__":
    unittest.main()


def test_production_output_hides_truth_metadata():
    from robust_repair.generator import generate_scenario
    from robust_repair.corruption import apply_corruption
    from robust_repair.repair import integrate
    master, clean_records, meta = generate_scenario(n_entities=40, n_sources=6, seed=909)
    corrupted, meta_c, _ = apply_corruption(master, clean_records, meta, budget=0.20, seed=909)
    out, _ = integrate(corrupted, meta_c, "parc", seed=909, attach_eval_metadata=False)
    assert "cluster_true_eid" not in out.columns
    assert "cluster_purity" not in out.columns
