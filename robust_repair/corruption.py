"""Adversarial corruption model for multi-source relational integration.

The corruption operators intentionally target data-management artifacts: keys,
functional dependencies, schema mappings, duplicate/provenance evidence, source
trust priors, and downstream aggregates. The implementation is deterministic
under a seed and is designed for reproducible data-repair experiments rather
than for cryptographic threat modeling.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .generator import FIRST, STATES, make_zip_city_maps

ALL_OPS = {"key", "name", "fd", "outlier", "schema", "duplicate", "trust", "provenance"}


def _append_attack(existing: str, new: str) -> str:
    if not existing:
        return new
    parts = set(str(existing).split(";")) if str(existing) != "nan" else set()
    parts.add(new)
    return ";".join(sorted(parts))


def _enabled(enabled_ops: Optional[Iterable[str]]) -> Set[str]:
    if enabled_ops is None:
        return set(ALL_OPS)
    ops = {str(x).strip().lower() for x in enabled_ops if str(x).strip()}
    unknown = ops - ALL_OPS
    if unknown:
        raise ValueError(f"unknown corruption operators: {sorted(unknown)}")
    return ops


def _name_alias(name: str, rng: np.random.Generator) -> str:
    """Return a plausible but not necessarily matchable business-name alias."""
    parts = str(name).split()
    if not parts:
        return str(name)
    mode = int(rng.integers(0, 5))
    if mode == 0 and len(parts) >= 2:
        parts[0] = FIRST[int(rng.integers(0, len(FIRST)))]
        return " ".join(parts)
    if mode == 1 and len(parts) >= 3:
        return " ".join(parts[:-1])
    if mode == 2:
        return str(name).replace("Market", "Mkt").replace("Supply", "Sup.").replace("Depot", "Dpt")
    if mode == 3 and len(parts[0]) > 3:
        token = list(parts[0])
        j = int(rng.integers(1, len(token)))
        token[j - 1], token[j] = token[j], token[j - 1]
        parts[0] = "".join(token)
        return " ".join(parts)
    return f"{name} Holdings"


def apply_corruption(
    master: pd.DataFrame,
    records: pd.DataFrame,
    meta: pd.DataFrame,
    budget: float,
    seed: int = 0,
    enabled_ops: Optional[Iterable[str]] = None,
    trust_alpha: float = 1.0,
    mode: str = "coordinated",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Apply a source/tuple/value/provenance corruption budget.

    ``enabled_ops`` supports operator ablations. ``trust_alpha`` controls how
    aggressively attackers inflate declared source trust. ``mode`` supports
    stress tests: ``coordinated`` attacks copied sources, ``independent`` spreads
    corruptions across independent-looking sources while keeping dependencies
    mostly valid, and ``sybil`` mints synthetic source identities as an explicit
    negative case outside authenticated-source assumptions.
    """
    ops = _enabled(enabled_ops)
    if mode not in {"coordinated", "independent", "sybil"}:
        raise ValueError(f"unknown corruption mode {mode}")
    trust_alpha = float(max(0.0, min(1.0, trust_alpha)))
    rng = np.random.default_rng(seed + 101)
    rec = records.copy(deep=True).reset_index(drop=True)
    met = meta.copy(deep=True).reset_index(drop=True)
    n = len(master)
    s_count = len(met)
    if budget <= 0 or not ops:
        return rec, met, {"target_entities": [], "attacker_sources": [], "budget": budget, "mode": mode, "operators": sorted(ops)}

    n_target = max(1, int(round(n * min(0.60, budget))))
    target_entities = set(map(int, rng.choice(master["eid"].to_numpy(), size=n_target, replace=False)))
    n_attack_sources = max(1, min(s_count, int(np.ceil(1 + budget * s_count * 0.75))))
    dep_cluster_sources = met[met["true_group"] == "G0"]["source"].tolist()
    independent_sources = met[~met["true_group"].isin(["G0", "G1", "G2"] if s_count >= 9 else ["G0"])]["source"].tolist()
    remainder = [x for x in met["source"].tolist() if x not in dep_cluster_sources and x not in independent_sources]
    rng.shuffle(remainder)
    rng.shuffle(independent_sources)
    if mode in {"independent", "sybil"} and independent_sources:
        attacker_sources = (independent_sources + remainder + dep_cluster_sources)[:n_attack_sources]
    else:
        attacker_sources = (dep_cluster_sources + remainder + independent_sources)[:n_attack_sources]

    if "trust" in ops or "provenance" in ops:
        for src in attacker_sources:
            m = met["source"] == src
            if "trust" in ops:
                old = float(met.loc[m, "declared_trust"].iloc[0])
                inflated = float(0.93 + 0.04 * rng.random())
                new = (1.0 - trust_alpha) * old + trust_alpha * inflated
                met.loc[m, "declared_trust"] = min(0.99, max(0.50, new))
                met.loc[m, "trust_manipulated"] = True
            if "provenance" in ops and rng.random() < min(0.95, 0.35 + budget):
                fake_group = f"U_FAKE_{src}"
                met.loc[m, "declared_group"] = fake_group
                met.loc[m, "provenance_manipulated"] = True
                rec.loc[rec["source"] == src, "declared_group"] = fake_group
            rec.loc[rec["source"] == src, "declared_trust"] = float(met.loc[m, "declared_trust"].iloc[0])

    zip_to_city_state, _ = make_zip_city_maps()
    zips = list(zip_to_city_state.keys())
    target_mask = rec["eid_true"].isin(target_entities) & rec["source"].isin(attacker_sources)
    idx = rec.index[target_mask].to_numpy()
    if len(idx) == 0:
        return rec, met, {"target_entities": list(target_entities), "attacker_sources": attacker_sources, "budget": budget, "mode": mode, "operators": sorted(ops)}

    target_category = "electronics"
    target_multiplier = 1.0 + 5.0 * budget
    for i in idx:
        eid = int(rec.at[i, "eid_true"])
        if "key" in ops and rng.random() < 0.55 + 0.35 * budget:
            new_eid = int((eid + rng.integers(1, max(2, n // 7))) % n)
            rec.at[i, "obs_key"] = f"E{new_eid:05d}"
            rec.at[i, "attacked"] = True
            rec.at[i, "attack_types"] = _append_attack(rec.at[i, "attack_types"], "join_key")
        if "name" in ops and rng.random() < 0.20 + 0.35 * budget:
            rec.at[i, "name"] = _name_alias(str(rec.at[i, "name"]), rng)
            rec.at[i, "attacked"] = True
            rec.at[i, "attack_types"] = _append_attack(rec.at[i, "attack_types"], "name_alias")
        if "fd" in ops and rng.random() < 0.70:
            z = zips[int(rng.integers(0, len(zips)))]
            city, state = zip_to_city_state[z]
            rec.at[i, "zip"] = z
            rec.at[i, "city"] = city
            if mode == "independent":
                # Harder adaptive case: the corrupt claim satisfies zip -> city,state,
                # so dependency repair cannot reject it by violation evidence alone.
                rec.at[i, "state"] = state
                rec.at[i, "attack_types"] = _append_attack(rec.at[i, "attack_types"], "constraint_satisfying")
            else:
                wrong_state = STATES[(STATES.index(state) + int(rng.integers(1, len(STATES)))) % len(STATES)]
                rec.at[i, "state"] = wrong_state
                rec.at[i, "attack_types"] = _append_attack(rec.at[i, "attack_types"], "fd_violation")
            rec.at[i, "attacked"] = True
        if "outlier" in ops and rng.random() < 0.85:
            rec.at[i, "category"] = target_category
            rec.at[i, "revenue"] = int(max(1, rec.at[i, "revenue"] * target_multiplier * rng.uniform(1.2, 2.1)))
            rec.at[i, "employees"] = int(max(1, rec.at[i, "employees"] * rng.uniform(0.2, 0.7)))
            rec.at[i, "attacked"] = True
            rec.at[i, "attack_types"] = _append_attack(rec.at[i, "attack_types"], "aggregate_outlier")

    schema_sources: List[str] = []
    if "schema" in ops:
        n_schema_sources = max(1, int(np.floor(max(0.10, budget) * len(attacker_sources))))
        schema_sources = attacker_sources[:n_schema_sources]
        for src in schema_sources:
            # Schema-mapping corruption is source-level: an adapter for an
            # untrusted feed has swapped two mapped columns.  This makes the
            # repair problem a data-integration/schema-mapping problem rather
            # than an isolated cell-noise problem.
            m = rec["source"].eq(src)
            tmp = rec.loc[m, "revenue"].copy()
            rec.loc[m, "revenue"] = rec.loc[m, "employees"].astype(int)
            rec.loc[m, "employees"] = tmp.astype(int)
            rec.loc[m, "attacked"] = True
            rec.loc[m, "attack_types"] = rec.loc[m, "attack_types"].map(lambda x: _append_attack(x, "schema_mapping"))
            met.loc[met["source"] == src, "mapping_corrupted"] = True

    duplicates: List[Dict] = []
    if "duplicate" in ops:
        max_dupes_per_budget = 1 + int(np.ceil(5 * budget))
        dup_candidates = rec.index[target_mask].to_numpy()
        if len(dup_candidates) > 0:
            sample_size = min(len(dup_candidates), int(len(dup_candidates) * min(0.8, 0.25 + budget)))
            chosen = rng.choice(dup_candidates, size=sample_size, replace=False)
            for i in chosen:
                row = rec.loc[i].to_dict()
                for d in range(max_dupes_per_budget):
                    new = dict(row)
                    if mode == "sybil":
                        fake_src = f"X{row['source']}_{d}_{abs(hash(row['rid'])) % 100000}"
                        new["source"] = fake_src
                        new["rid"] = f"{fake_src}_{row['rid']}"
                        new["source_idx"] = 100000 + d
                        new["declared_group"] = f"U_FAKE_SYBIL_{fake_src}"
                        new["true_group"] = f"U_FAKE_SYBIL_{fake_src}"
                        new["declared_trust"] = 0.96
                        new["attack_types"] = _append_attack(new["attack_types"], "sybil_duplicate")
                    else:
                        new["rid"] = f"{row['rid']}_D{d}"
                        new["declared_group"] = f"U_FAKE_DUP_{row['source']}_{d}"
                        new["attack_types"] = _append_attack(new["attack_types"], "duplicate")
                    new["is_duplicate"] = True
                    new["attacked"] = True
                    duplicates.append(new)
    if duplicates:
        rec = pd.concat([rec, pd.DataFrame(duplicates)], ignore_index=True)

    summary = {
        "target_entities": sorted(target_entities),
        "attacker_sources": attacker_sources,
        "schema_sources": schema_sources,
        "budget": float(budget),
        "mode": mode,
        "operators": sorted(ops),
        "trust_alpha": trust_alpha,
        "n_attacked_records": int(rec["attacked"].sum()),
        "n_duplicates": int(rec["is_duplicate"].sum()),
    }
    return rec.reset_index(drop=True), met, summary
