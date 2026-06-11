"""Constraint checking and lightweight normalization utilities."""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .generator import ATTRS, CAT_ATTRS, NUM_ATTRS, make_zip_city_maps


def norm_name(x: object) -> str:
    """Normalize business names without using ground truth.

    The normalization deliberately handles only common integration-time
    abbreviations and legal suffixes. It is not an oracle: person-token
    substitutions and character transpositions still create hard cases.
    """
    s = str(x).lower().strip()
    replacements = {
        r"\bmkt\b": "market",
        r"\bsup\b": "supply",
        r"\bdpt\b": "depot",
        r"\bco\b": "company",
        r"\bcorp\b": "corporation",
        r"\binc\b": "",
        r"\bllc\b": "",
        r"\bholdings\b": "",
    }
    for pat, repl in replacements.items():
        s = re.sub(pat, repl, s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def norm_cat(x: object) -> str:
    return str(x).strip().lower()


def close_num(a: object, b: object, rel_tol: float = 0.05) -> bool:
    try:
        av, bv = float(a), float(b)
    except Exception:
        return False
    if av == bv:
        return True
    return abs(av - bv) <= rel_tol * max(1.0, abs(bv))


def value_equal(attr: str, a: object, b: object) -> bool:
    if attr in NUM_ATTRS:
        return close_num(a, b)
    if attr == "name":
        return norm_name(a) == norm_name(b)
    return str(a) == str(b)


def fd_violation_count(df: pd.DataFrame) -> int:
    """Count row-level FD/range violations in an integrated table."""
    zip_to_city_state, city_to_state = make_zip_city_maps()
    count = 0
    for r in df.itertuples(index=False):
        z = str(getattr(r, "zip"))
        city = getattr(r, "city")
        state = getattr(r, "state")
        if z in zip_to_city_state:
            true_city, true_state = zip_to_city_state[z]
            if city != true_city or state != true_state:
                count += 1
        if city in city_to_state and state != city_to_state[city]:
            count += 1
        try:
            rev = float(getattr(r, "revenue"))
            emp = float(getattr(r, "employees"))
            if rev <= 0 or emp <= 0 or rev / max(emp, 1) < 1000 or rev / max(emp, 1) > 2000000:
                count += 1
        except Exception:
            count += 1
    return count


def violation_rate(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 1.0
    return fd_violation_count(df) / (3.0 * len(df))


def repair_zip_city_state(row: dict, candidates: Iterable[dict] | None = None) -> dict:
    zip_to_city_state, city_to_state = make_zip_city_maps()
    out = dict(row)
    z = str(out.get("zip", ""))
    if z in zip_to_city_state:
        c, s = zip_to_city_state[z]
        out["city"] = c
        out["state"] = s
    else:
        c = out.get("city")
        if c in city_to_state:
            out["state"] = city_to_state[c]
    return out


def robust_median(values, weights=None):
    vals = np.array([float(v) for v in values if pd.notna(v)])
    if len(vals) == 0:
        return np.nan
    if weights is None:
        return float(np.median(vals))
    w = np.array(weights, dtype=float)
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    c = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    return float(vals[np.searchsorted(c, cutoff)])


def iqr_filter(values):
    vals = np.array([float(v) for v in values if pd.notna(v)], dtype=float)
    if len(vals) < 4:
        return np.ones(len(values), dtype=bool)
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    iqr = max(q3 - q1, 1.0)
    lo, hi = q1 - 1.75 * iqr, q3 + 1.75 * iqr
    return np.array([(float(v) >= lo and float(v) <= hi) if pd.notna(v) else False for v in values])
