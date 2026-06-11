from __future__ import annotations

import sqlite3

from robust_repair.corruption import apply_corruption
from robust_repair.generator import Scenario, generate_master, generate_sources
from robust_repair.repair import integrate
from scripts.run_external_and_sql_suite import _query_results, _write_table
from scripts.run_sql_fragment_suite import query_dict


def _scenario():
    master = generate_master(60, seed=123)
    clean, meta = generate_sources(master, Scenario(60, 6, coverage=0.75, seed=123))
    records, meta_c, _ = apply_corruption(master, clean, meta, budget=0.2, seed=123, mode="coordinated")
    return master, records, meta_c


def test_production_outputs_have_public_eid_without_truth_metadata_when_requested():
    _master, records, meta = _scenario()
    out, _ = integrate(records, meta, "parc", seed=123, attach_eval_metadata=False)
    assert "eid" in out.columns
    assert "cluster_true_eid" not in out.columns
    assert "cluster_purity" not in out.columns


def test_sql_writer_drops_evaluator_truth_even_if_present():
    master, records, meta = _scenario()
    out, _ = integrate(records, meta, "parc", seed=123)
    assert "cluster_true_eid" in out.columns  # benchmark output has evaluator metadata
    conn = sqlite3.connect(":memory:")
    _write_table(conn, "fact", out)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fact)").fetchall()]
    assert "eid" in cols
    assert "cluster_true_eid" not in cols
    assert "cluster_purity" not in cols
    conn.close()


def test_sql_query_helpers_do_not_require_hidden_truth_columns():
    master, records, meta = _scenario()
    out, _ = integrate(records, meta, "parc", seed=123, attach_eval_metadata=False)
    q1 = _query_results(out, master)
    q2 = query_dict(out, master)
    assert q1 and q2
