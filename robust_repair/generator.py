"""Synthetic multi-source relational integration generator for PARC.

The generator creates a clean master relation and several overlapping sources.
Each source has a declared trust score and a dependency/provenance group. The
relation intentionally contains functional dependencies such as zip -> city,state
and city -> state, plus numerical attributes used by downstream aggregates.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ATTRS = ["name", "city", "state", "zip", "category", "revenue", "employees"]
CAT_ATTRS = ["name", "city", "state", "zip", "category"]
NUM_ATTRS = ["revenue", "employees"]

STATES = ["CA", "TX", "NY", "WA", "IL", "PA", "OH", "GA"]
CATEGORIES = [
    "grocery", "pharmacy", "electronics", "auto", "hardware", "apparel",
    "books", "furniture", "sports", "garden", "pet", "office"
]
FIRST = [
    "Aster", "Cedar", "River", "Nova", "Summit", "Harbor", "Maple", "Atlas",
    "Orchid", "Prairie", "Beacon", "Cobalt", "Willow", "Jade", "Pioneer",
    "Granite", "Amber", "Canyon", "Aurora", "Silver", "Juniper", "Vertex",
]
SECOND = [
    "Market", "Supply", "Depot", "Outlet", "Works", "Hub", "Trading", "Mart",
    "Exchange", "Collective", "Source", "Gallery", "Center", "Stores", "Lane",
]

@dataclass
class Scenario:
    n_entities: int = 1200
    n_sources: int = 12
    coverage: float = 0.64
    seed: int = 0
    dependent_clusters: int = 3


def make_zip_city_maps() -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str]]:
    zip_to_city_state: Dict[str, Tuple[str, str]] = {}
    city_to_state: Dict[str, str] = {}
    idx = 10000
    for s_i, state in enumerate(STATES):
        for j in range(9):
            city = f"{state}_city_{j:02d}"
            z = f"{idx + s_i * 100 + j:05d}"
            zip_to_city_state[z] = (city, state)
            city_to_state[city] = state
    return zip_to_city_state, city_to_state


def generate_master(n_entities: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zip_to_city_state, _ = make_zip_city_maps()
    zips = list(zip_to_city_state.keys())
    rows = []
    used_names = set()
    for eid in range(n_entities):
        z = zips[int(rng.integers(0, len(zips)))]
        city, state = zip_to_city_state[z]
        cat = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
        # Unique but human-readable business-like names.
        name = f"{FIRST[eid % len(FIRST)]} {SECOND[(eid // len(FIRST)) % len(SECOND)]} {eid:04d}"
        if name in used_names:
            name += f" {eid}"
        used_names.add(name)
        cat_factor = 1.0 + (CATEGORIES.index(cat) % 5) * 0.28
        state_factor = 1.0 + (STATES.index(state) % 4) * 0.10
        revenue = int(rng.lognormal(mean=12.0, sigma=0.45) * cat_factor * state_factor)
        employees = max(1, int(revenue / rng.uniform(78000, 135000)))
        rows.append({
            "eid": eid,
            "name": name,
            "city": city,
            "state": state,
            "zip": z,
            "category": cat,
            "revenue": revenue,
            "employees": employees,
        })
    return pd.DataFrame(rows)


def _source_dependency_group(source_id: int, n_sources: int, dependent_clusters: int) -> str:
    # Several groups intentionally share upstream provenance; the remaining
    # sources are independent.
    if source_id < dependent_clusters * 3:
        return f"G{source_id // 3}"
    return f"U{source_id}"


def generate_sources(master: pd.DataFrame, scenario: Scenario) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(scenario.seed + 11)
    rnd = random.Random(scenario.seed + 17)
    rows: List[Dict] = []
    source_meta: List[Dict] = []
    for s in range(scenario.n_sources):
        # Quality is heterogeneous; dependent clusters copy some errors.
        base_quality = float(rng.beta(11, 2))
        if s % 5 == 0:
            base_quality = float(rng.beta(7, 3))
        dep = _source_dependency_group(s, scenario.n_sources, scenario.dependent_clusters)
        declared = min(0.98, max(0.52, base_quality + rng.normal(0.03, 0.04)))
        source_meta.append({
            "source": f"S{s:02d}",
            "source_idx": s,
            "true_quality": base_quality,
            "declared_trust": declared,
            "declared_group": dep,
            "true_group": dep,
            "mapping_corrupted": False,
            "trust_manipulated": False,
            "provenance_manipulated": False,
        })
        for rec in master.itertuples(index=False):
            if rng.random() > scenario.coverage:
                continue
            local = rec._asdict()
            # Natural dirty-data errors before adversarial corruption.
            attacked = False
            attack_types: List[str] = []
            for attr in CAT_ATTRS:
                if rng.random() > base_quality:
                    if attr == "zip":
                        local[attr] = str(int(local[attr]) + int(rng.integers(1, 40))).zfill(5)
                    elif attr == "state":
                        local[attr] = STATES[int(rng.integers(0, len(STATES)))]
                    elif attr == "city":
                        local[attr] = f"{STATES[int(rng.integers(0, len(STATES)))]}_city_{int(rng.integers(0, 9)):02d}"
                    elif attr == "category":
                        local[attr] = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
                    elif attr == "name":
                        local[attr] = local[attr].replace(" ", "") if rng.random() < 0.5 else local[attr].lower()
            for attr in NUM_ATTRS:
                if rng.random() > base_quality:
                    if attr == "revenue":
                        local[attr] = int(max(1, local[attr] * rng.uniform(0.65, 1.65)))
                    else:
                        local[attr] = int(max(1, local[attr] * rng.uniform(0.5, 1.8)))
            # Local schema id visible to systems; attackers can poison it later.
            obs_key = f"E{local['eid']:05d}"
            rows.append({
                "rid": f"S{s:02d}_R{int(local['eid']):06d}",
                "source": f"S{s:02d}",
                "source_idx": s,
                "declared_group": dep,
                "true_group": dep,
                "declared_trust": declared,
                "eid_true": int(local["eid"]),
                "obs_key": obs_key,
                "name": local["name"],
                "city": local["city"],
                "state": local["state"],
                "zip": str(local["zip"]),
                "category": local["category"],
                "revenue": int(local["revenue"]),
                "employees": int(local["employees"]),
                "attacked": attacked,
                "attack_types": ";".join(attack_types),
                "is_duplicate": False,
            })
    return pd.DataFrame(rows), pd.DataFrame(source_meta)


def generate_scenario(n_entities: int = 1200, n_sources: int = 12, coverage: float = 0.64, seed: int = 0):
    scenario = Scenario(n_entities=n_entities, n_sources=n_sources, coverage=coverage, seed=seed)
    master = generate_master(n_entities, seed)
    records, meta = generate_sources(master, scenario)
    return master, records, meta
