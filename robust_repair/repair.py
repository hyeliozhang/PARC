"""Repair and integration algorithms used in the PARC benchmark."""
from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .constraints import iqr_filter, norm_name, repair_zip_city_state, robust_median, value_equal
from .generator import ATTRS, CAT_ATTRS, NUM_ATTRS

PARC_VARIANTS = {
    "parc", "parc_hybrid", "parc_no_prov", "parc_no_constraints", "parc_no_trust",
    "parc_no_schema", "parc_name_blocks"
}


def _source_weights(meta: pd.DataFrame, mode: str, rng: np.random.Generator | None = None) -> Dict[str, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    weights: Dict[str, float] = {}
    for r in meta.itertuples(index=False):
        src = getattr(r, "source")
        declared = float(getattr(r, "declared_trust", 0.7))
        if mode == "uniform":
            w = 1.0
        elif mode == "declared":
            # Positive clipped logit weight; clipping is part of the semantics and
            # prevents manipulated trust priors from dominating support alone.
            w = max(0.05, min(4.0, math.log(declared / max(1e-6, 1 - declared))))
        elif mode == "random":
            w = float(rng.uniform(0.5, 2.5))
        else:
            w = 1.0
        weights[src] = float(w)
    return weights


def _detect_schema_swaps(records: pd.DataFrame) -> Dict[str, dict]:
    """Detect source-level revenue/employees swaps via range/ratio evidence."""
    swaps: Dict[str, dict] = {}
    for src, g in records.groupby("source"):
        if len(g) < 20:
            continue
        rev_med = float(np.median(g["revenue"].astype(float)))
        emp_med = float(np.median(g["employees"].astype(float)))
        # Clean relation normally has revenue much larger than employees.  A source
        # with an inverted median ratio receives a schema-repair certificate.
        if rev_med < emp_med * 20 and emp_med > 5000:
            swaps[src] = {"median_revenue": rev_med, "median_employees": emp_med, "action": "swap revenue/employees"}
    return swaps


def _apply_schema_repair(records: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    out = records.copy(deep=True)
    swaps = _detect_schema_swaps(out)
    if not swaps:
        return out, swaps

    # Peer evidence by observed key.  We use only plausible rows as anchors so a
    # corrupted source slice does not make its own swapped values look normal.
    plausible = (out["revenue"].astype(float) > out["employees"].astype(float) * 20.0) & (out["employees"].astype(float) < 5000.0)
    peer_stats: Dict[str, Tuple[float, float]] = {}
    for key, g in out[plausible].groupby("obs_key"):
        if len(g) >= 2:
            peer_stats[str(key)] = (float(np.median(g["revenue"].astype(float))), float(np.median(g["employees"].astype(float))))

    for src in list(swaps.keys()):
        src_mask = out["source"].eq(src)
        inv = src_mask & (out["revenue"].astype(float) < out["employees"].astype(float) * 20.0) & (out["employees"].astype(float) > 5000.0)
        repair_idx = []
        for idx, row in out[inv].iterrows():
            key = str(row["obs_key"])
            if key not in peer_stats:
                continue
            med_rev, med_emp = peer_stats[key]
            cand_rev = float(row["employees"])
            cand_emp = float(row["revenue"])
            rev_ok = abs(cand_rev - med_rev) <= 0.12 * max(1.0, med_rev)
            emp_ok = abs(cand_emp - med_emp) <= max(3.0, 0.35 * max(1.0, med_emp))
            if rev_ok and emp_ok:
                repair_idx.append(idx)
        # Abstain unless the source has a substantial peer-validated slice;
        # otherwise the repair risks converting ordinary value errors into a
        # systematic mapping edit.
        min_rows = max(25, int(0.10 * max(1, int(src_mask.sum()))))
        if len(repair_idx) < min_rows:
            swaps[src]["rows_repaired"] = 0
            swaps[src]["abstained_rows"] = int(len(repair_idx))
            continue
        swaps[src]["rows_repaired"] = int(len(repair_idx))
        tmp = out.loc[repair_idx, "revenue"].copy()
        out.loc[repair_idx, "revenue"] = out.loc[repair_idx, "employees"].astype(int)
        out.loc[repair_idx, "employees"] = tmp.astype(int)
    return out, swaps

def _stable_fingerprint_frame(g: pd.DataFrame) -> set[str]:
    if len(g) == 0:
        return set()
    f = g[["name", "zip", "category"]].copy()
    f["name"] = f["name"].map(norm_name)
    return set(f.astype(str).agg("|".join, axis=1).tolist())


def _infer_dependency_groups(records: pd.DataFrame, meta: pd.DataFrame, threshold: float = 0.98) -> Dict[str, str]:
    """Merge declared groups with copy evidence from near-identical source feeds."""
    sources = sorted(records["source"].unique())
    parent = {s: s for s in sources}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Declared provenance is evidence.  When attackers launder provenance into
    # singleton fake groups, this merge disappears; copy fingerprints below are a
    # second, independent cue.
    for _, group in meta.groupby("declared_group"):
        ss = [s for s in group["source"].tolist() if s in parent]
        if len(ss) >= 2:
            for s in ss[1:]:
                union(ss[0], s)

    src_sets = {src: _stable_fingerprint_frame(g) for src, g in records.groupby("source")}

    # Avoid the quadratic all-pairs comparison when a stress test mints many
    # synthetic source identities.  Only pairs that share at least one stable
    # fingerprint can reach the overlap threshold.
    inverted: Dict[str, List[str]] = defaultdict(list)
    for src, fps in src_sets.items():
        for fp in fps:
            inverted[fp].append(src)
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for ss in inverted.values():
        if len(ss) < 2:
            continue
        ss = sorted(set(ss))
        for i, a in enumerate(ss):
            for b in ss[i + 1:]:
                pair_counts[(a, b)] += 1
    for (a, b), inter in pair_counts.items():
        denom = min(len(src_sets.get(a, set())), len(src_sets.get(b, set())))
        if denom and inter / denom >= threshold:
            union(a, b)
    return {s: find(s) for s in sources}


def _cluster_records(records: pd.DataFrame, mode: str) -> Dict[str, pd.DataFrame]:
    rec = records.copy()
    if mode == "obs_key":
        rec["cluster"] = "K:" + rec["obs_key"].astype(str)
    elif mode == "robust_blocks":
        # Keys are evidence, not truth.  PARC uses conservative hybrid blocking:
        # stable normalized names can bridge poisoned keys, while one-off aliases
        # inside a coherent key block are folded back into the local majority.
        # No hidden entity id or clean master table is used here.
        rec["_name_key"] = rec["name"].map(norm_name)
        global_name_support = rec.groupby("_name_key")["source"].nunique().to_dict()
        key_majority: Dict[str, Tuple[str, int, int]] = {}
        for key, g in rec.groupby("obs_key"):
            vc = g["_name_key"].value_counts()
            if len(vc) > 0:
                key_majority[str(key)] = (str(vc.index[0]), int(vc.iloc[0]), int(len(g)))

        labels: List[str] = []
        for _, row in rec.iterrows():
            nk = str(row["_name_key"])
            key = str(row["obs_key"])
            maj, maj_count, key_size = key_majority.get(key, ("", 0, 0))
            stable_name = len(nk) >= 4 and int(global_name_support.get(nk, 0)) >= 2
            strong_key_majority = maj_count >= 2 and maj_count >= max(2, int(0.45 * max(1, key_size)))
            if stable_name:
                labels.append("N:" + nk)
            elif strong_key_majority and len(maj) >= 4:
                labels.append("N:" + maj)
            elif len(nk) >= 4:
                labels.append("N:" + nk)
            else:
                labels.append("K:" + key)
        rec["cluster"] = labels
        rec = rec.drop(columns=["_name_key"])
    else:
        raise ValueError(mode)
    return {k: g.copy() for k, g in rec.groupby("cluster")}


def _weighted_vote(
    values: Iterable,
    sources: Iterable[str],
    groups: Iterable[str],
    weights: Dict[str, float],
    cap_groups: bool = False,
) -> Tuple[object | None, float, Dict[object, float]]:
    score: Dict[object, float] = defaultdict(float)
    group_value_score: Dict[Tuple[str, object], float] = defaultdict(float)
    if cap_groups:
        for v, s, g in zip(values, sources, groups):
            group_value_score[(g, v)] = max(group_value_score[(g, v)], weights.get(s, 1.0))
        for (_g, v), w in group_value_score.items():
            score[v] += w
    else:
        for v, s in zip(values, sources):
            score[v] += weights.get(s, 1.0)
    if not score:
        return None, 0.0, {}
    val = max(score.items(), key=lambda kv: (kv[1], str(kv[0])))[0]
    return val, float(score[val]), dict(score)


def _cell_certificate(attr: str, accepted: object, scores: Dict[object, float], groups: List[str], filtered: int = 0) -> dict:
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    top = float(ordered[0][1]) if ordered else 0.0
    second = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    return {
        "attribute": attr,
        "accepted": accepted,
        "margin": top - second,
        "top_score": top,
        "second_score": second,
        "independent_groups": len(set(groups)),
        "filtered_claims": int(filtered),
        "candidates": {str(k): float(v) for k, v in ordered[:6]},
    }


def _integrate_cluster(
    g: pd.DataFrame,
    weights: Dict[str, float],
    group_map: Dict[str, str],
    cap_groups: bool,
    apply_constraints: bool,
    outlier_filter: bool,
) -> Tuple[dict, dict]:
    out: Dict[str, object] = {}
    cert = {"cells": {}, "source_count": int(g["source"].nunique()), "record_count": int(len(g)), "constraint_actions": []}
    groups = [group_map.get(s, str(s)) for s in g["source"]]
    for attr in CAT_ATTRS:
        val, _score, scores = _weighted_vote(g[attr].tolist(), g["source"].tolist(), groups, weights, cap_groups=cap_groups)
        out[attr] = val
        cert["cells"][attr] = _cell_certificate(attr, val, scores, groups)

    for attr in NUM_ATTRS:
        values = g[attr].astype(float).to_numpy()
        srcs = g["source"].tolist()
        grps = groups
        mask = np.ones(len(values), dtype=bool)
        if outlier_filter and len(values) >= 4:
            mask = iqr_filter(values)
            if mask.sum() < 2:
                mask = np.ones(len(values), dtype=bool)
        vals = values[mask]
        srcs_m = [s for s, m in zip(srcs, mask) if m]
        grps_m = [gr for gr, m in zip(grps, mask) if m]
        filtered = int(len(values) - mask.sum())
        if cap_groups:
            reps: List[float] = []
            rep_weights: List[float] = []
            by_group: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
            for v, s, gr in zip(vals, srcs_m, grps_m):
                by_group[gr].append((float(v), weights.get(s, 1.0), str(s)))
            for _gr, items in by_group.items():
                v, w, _s = max(items, key=lambda x: x[1])
                reps.append(v)
                rep_weights.append(w)
            capped = int(round(robust_median(reps, rep_weights))) if reps else int(round(np.nanmedian(values)))
            capped_scores = {int(round(v)): w for v, w in zip(reps, rep_weights)}

            # Adaptive contract rule: the capped representative protects against
            # copied-source amplification, but when an uncapped candidate is
            # supported by at least two independent groups and has a strictly
            # larger evidence margin, use it as the point value while preserving
            # the capped/candidate panel in the certificate.  This is an
            # unsupervised decision; it prevents genuine independent agreement
            # from being mistaken for copy dependence.
            uncapped, _unc_score, unc_scores = _weighted_vote([int(round(v)) for v in vals], srcs_m, grps_m, weights, cap_groups=False)
            accepted = capped
            if uncapped is not None:
                ordered_unc = sorted(unc_scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
                unc_margin = float(ordered_unc[0][1] - (ordered_unc[1][1] if len(ordered_unc) > 1 else 0.0)) if ordered_unc else 0.0
                ordered_cap = sorted(capped_scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
                cap_margin = float(ordered_cap[0][1] - (ordered_cap[1][1] if len(ordered_cap) > 1 else 0.0)) if ordered_cap else 0.0
                unc_groups = {gr for v, gr in zip([int(round(x)) for x in vals], grps_m) if v == uncapped}
                if len(unc_groups) >= 2 and unc_margin >= cap_margin + 0.25:
                    accepted = int(uncapped)
                    capped_scores[int(uncapped)] = max(capped_scores.get(int(uncapped), 0.0), float(ordered_unc[0][1]))
            out[attr] = accepted
            cert["cells"][attr] = _cell_certificate(attr, accepted, capped_scores, grps_m, filtered=filtered)
        else:
            val, _score, scores = _weighted_vote([int(round(v)) for v in vals], srcs_m, grps_m, weights, cap_groups=False)
            if val is None:
                val = int(round(np.nanmedian(values)))
            out[attr] = int(val)
            cert["cells"][attr] = _cell_certificate(attr, int(val), scores, grps_m, filtered=filtered)

    if apply_constraints:
        before = dict(out)
        out = repair_zip_city_state(out)
        for a in ["city", "state"]:
            if before.get(a) != out.get(a):
                cert["constraint_actions"].append({"rule": "zip->city,state", "attribute": a, "from": before.get(a), "to": out.get(a)})
    return out, cert


def _trust_update(
    records: pd.DataFrame,
    clusters: Dict[str, pd.DataFrame],
    outputs: Dict[str, dict],
    base_weights: Dict[str, float],
) -> Dict[str, float]:
    matches = defaultdict(list)
    for cid, g in clusters.items():
        out = outputs.get(cid)
        if not out:
            continue
        for r in g.itertuples(index=False):
            for attr in ATTRS:
                src = getattr(r, "source")
                matches[src].append(1.0 if value_equal(attr, getattr(r, attr), out[attr]) else 0.0)
    updated: Dict[str, float] = {}
    for src, vals in matches.items():
        if vals:
            p = 0.15 + 0.70 * float(np.mean(vals))
            prior = 1 / (1 + math.exp(-base_weights.get(src, 1.0)))
            t = 0.70 * p + 0.30 * prior
        else:
            t = 0.55
        t = min(0.97, max(0.08, t))
        updated[src] = max(0.05, min(4.0, math.log(t / (1 - t))))
    return updated



def _dominant_name_label(g: pd.DataFrame) -> str | None:
    """Return the robust-block label suggested by the dominant normalized name."""
    if len(g) == 0 or "name" not in g.columns:
        return None
    tmp = g["name"].map(norm_name)
    tmp = tmp[tmp.map(lambda x: len(str(x)) >= 4)]
    if len(tmp) == 0:
        return None
    return "N:" + str(tmp.value_counts().index[0])


def _certificate_margin(certs: Dict[str, dict], cid: str, attr: str) -> float:
    try:
        return float(certs.get(cid, {}).get("cells", {}).get(attr, {}).get("margin", 0.0))
    except Exception:
        return 0.0


def _certificate_groups(certs: Dict[str, dict], cid: str, attr: str) -> int:
    try:
        return int(certs.get(cid, {}).get("cells", {}).get(attr, {}).get("independent_groups", 0))
    except Exception:
        return 0


def _hybridize_outputs(
    rec: pd.DataFrame,
    obs_outputs: Dict[str, dict],
    obs_certs: Dict[str, dict],
    name_outputs: Dict[str, dict],
    name_certs: Dict[str, dict],
    apply_constraints: bool = True,
) -> Tuple[Dict[str, dict], Dict[str, dict], int]:
    """Use robust name blocks as a value-repair oracle without changing row admission.

    This preserves the observed-key materialization policy that is stable for
    aggregates, but allows high-margin independent name evidence to repair cell
    values inside poisoned-key blocks. It is an unsupervised model-selection step:
    no hidden entity id or clean master relation is consulted.
    """
    clusters = _cluster_records(rec, "obs_key")
    out: Dict[str, dict] = {}
    certs: Dict[str, dict] = {}
    adopted = 0
    for cid, g in clusters.items():
        base = dict(obs_outputs.get(cid, {}))
        base_cert = dict(obs_certs.get(cid, {}))
        nlabel = _dominant_name_label(g)
        if nlabel and nlabel in name_outputs:
            for attr in ATTRS:
                # Name-block evidence is used only for symbolic integration
                # attributes. Numeric measures drive downstream aggregates; moving
                # them across ambiguous entity blocks can lower cell error while
                # silently increasing query distortion. PARC therefore keeps
                # numeric measures anchored to observed keys and exposes ambiguous
                # numeric cells through margins instead of migrating them.
                if attr in NUM_ATTRS:
                    continue
                nm_groups = _certificate_groups(name_certs, nlabel, attr)
                nm_margin = _certificate_margin(name_certs, nlabel, attr)
                ob_margin = _certificate_margin(obs_certs, cid, attr)
                if nm_groups >= 2 and nm_margin >= max(1e-9, ob_margin):
                    base[attr] = name_outputs[nlabel].get(attr, base.get(attr))
                    adopted += 1
        if apply_constraints and base:
            base = repair_zip_city_state(base)
        base_cert["hybrid_name_label"] = nlabel
        out[cid] = base
        certs[cid] = base_cert
    return out, certs, adopted

def _run_configured(
    rec: pd.DataFrame,
    meta: pd.DataFrame,
    rng: np.random.Generator,
    *,
    cluster_mode: str,
    cap_groups: bool,
    apply_constraints: bool,
    outlier_filter: bool,
    schema_repair: bool,
    weight_mode: str,
    trust_iterations: int,
) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, str], Dict[str, dict], Dict[str, float], pd.DataFrame]:
    schema_actions: Dict[str, dict] = {}
    if schema_repair:
        rec, schema_actions = _apply_schema_repair(rec)
    clusters = _cluster_records(rec, cluster_mode)
    group_map = _infer_dependency_groups(rec, meta, threshold=0.98) if cap_groups else {s: s for s in rec["source"].unique()}
    weights = _source_weights(meta, weight_mode, rng)
    outputs: Dict[str, dict] = {}
    certs: Dict[str, dict] = {}
    for _ in range(max(1, trust_iterations + 1)):
        outputs, certs = {}, {}
        for cid, g in clusters.items():
            outputs[cid], certs[cid] = _integrate_cluster(g, weights, group_map, cap_groups, apply_constraints, outlier_filter)
        if trust_iterations > 0:
            weights = _trust_update(rec, clusters, outputs, weights)
            trust_iterations -= 1
    return outputs, certs, group_map, schema_actions, weights, rec


def integrate(
    records: pd.DataFrame,
    meta: pd.DataFrame,
    method: str,
    seed: int = 0,
    attach_eval_metadata: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Run one integration/repair method.

    ``attach_eval_metadata`` controls whether benchmark-only hidden identifiers
    are appended to the returned DataFrame.  The repair pipeline never reads
    hidden identifiers; they are added after outputs are produced solely for
    metric computation.  Setting the flag to ``False`` returns a production-like
    repaired relation with no hidden truth columns.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    rec = records.copy(deep=True)
    details: Dict = {"method": method}

    if method == "no_repair":
        clusters = _cluster_records(rec, "obs_key")
        outputs, certs = {}, {}
        for cid, g in clusters.items():
            first = g.iloc[0]
            outputs[cid] = {attr: first[attr] for attr in ATTRS}
            certs[cid] = {"record_count": int(len(g)), "source_count": int(g["source"].nunique()), "cells": {}}
        group_map, schema_actions, weights = {s: s for s in rec["source"].unique()}, {}, _source_weights(meta, "uniform")
    elif method in {"majority", "standard_fd", "rule_only", "random_weighting", "outlier_removal", "provenance_unaware", "iterative_truth", "source_dependence", "dependency_truth"}:
        cap_groups = method in {"source_dependence", "dependency_truth"}
        apply_constraints = method in {"standard_fd", "rule_only", "outlier_removal", "iterative_truth", "dependency_truth"}
        outlier_filter = method in {"outlier_removal", "dependency_truth"}
        if method in {"majority", "standard_fd", "rule_only", "outlier_removal", "source_dependence"}:
            weight_mode = "uniform"
        elif method == "random_weighting":
            weight_mode = "random"
        else:
            weight_mode = "declared"
        trust_iterations = 1 if method in {"iterative_truth", "dependency_truth"} else 0
        outputs, certs, group_map, schema_actions, weights, rec = _run_configured(
            rec, meta, rng,
            cluster_mode="obs_key", cap_groups=cap_groups, apply_constraints=apply_constraints,
            outlier_filter=outlier_filter, schema_repair=False, weight_mode=weight_mode,
            trust_iterations=trust_iterations,
        )
    elif method in PARC_VARIANTS:
        if method == "parc_hybrid":
            obs_outputs, obs_certs, group_map, schema_actions, weights, rec = _run_configured(
                rec, meta, rng, cluster_mode="obs_key", cap_groups=True, apply_constraints=True,
                outlier_filter=True, schema_repair=True, weight_mode="declared", trust_iterations=1,
            )
            name_outputs, name_certs, name_group_map, name_schema, name_weights, rec_name = _run_configured(
                records.copy(deep=True), meta, rng, cluster_mode="robust_blocks", cap_groups=True, apply_constraints=True,
                outlier_filter=True, schema_repair=True, weight_mode="declared", trust_iterations=1,
            )
            outputs, certs, adopted = _hybridize_outputs(rec, obs_outputs, obs_certs, name_outputs, name_certs, apply_constraints=True)
            details["hybrid_cells_adopted"] = int(adopted)
        else:
            outputs, certs, group_map, schema_actions, weights, rec = _run_configured(
                rec, meta, rng,
                # The optional parc_name_blocks variant materializes robust name blocks.
                # The default PARC remains observed-key materialized for aggregate stability;
                # parc_hybrid uses name blocks only as high-margin value evidence.
                cluster_mode="robust_blocks" if method == "parc_name_blocks" else "obs_key",
                cap_groups=method != "parc_no_prov",
                apply_constraints=method != "parc_no_constraints",
                outlier_filter=True,
                schema_repair=method != "parc_no_schema",
                weight_mode="uniform" if method == "parc_no_trust" else "declared",
                trust_iterations=0 if method == "parc_no_trust" else 1,
            )
    else:
        raise ValueError(f"unknown method {method}")

    rows: List[dict] = []
    clusters = _cluster_records(rec, "obs_key") if method == "no_repair" else None
    # Rebuild the same clusters used for output to attach evaluation metadata.
    if method == "no_repair" or method not in PARC_VARIANTS and method not in {"source_dependence", "dependency_truth"}:
        eval_clusters = _cluster_records(rec, "obs_key")
    elif method in {"source_dependence", "dependency_truth"}:
        eval_clusters = _cluster_records(rec, "obs_key")
    else:
        eval_clusters = _cluster_records(rec, "robust_blocks" if method == "parc_name_blocks" else "obs_key")

    suppressed_clusters = 0
    for cid, g in eval_clusters.items():
        if cid not in outputs:
            continue
        # Admission is part of PARC's provenance-bounded semantics: a robust
        # name/key block emitted by a single source is retained as evidence in
        # the certificate but not materialized as an integrated entity.  This
        # prevents single-source alias and duplicate injections from amplifying
        # downstream aggregates without consulting hidden truth.
        if method == "parc_name_blocks":
            independent_groups = {group_map.get(str(src), str(src)) for src in g["source"].astype(str).tolist()}
            # Low-provenance name blocks are the main amplification path for
            # duplicate/name-alias injections.  They remain in certificates but
            # are not materialized as integrated entities.
            if len(independent_groups) < 2 and int(len(g)) == 1 and str(cid).startswith("N:"):
                suppressed_clusters += 1
                continue
        out = dict(outputs[cid])
        if "eid_true" in g.columns and len(g):
            vc = g["eid_true"].value_counts()
            true_eid = int(vc.index[0])
            purity = float(vc.iloc[0] / len(g))
        else:
            true_eid = -1
            purity = 0.0
        # Public surrogate key for downstream SQL workflows. It is derived only
        # from the materialized cluster label and never from evaluator truth.
        eid_public = int(hashlib.blake2b(str(cid).encode("utf-8"), digest_size=8).hexdigest()[:15], 16)
        out.update({
            "eid": eid_public,
            "cluster": cid,
            "record_count": int(len(g)),
            "source_count": int(g["source"].nunique()),
        })
        if attach_eval_metadata:
            out.update({
                "cluster_true_eid": true_eid,
                "cluster_purity": purity,
            })
        rows.append(out)
    result = pd.DataFrame(rows)
    details["runtime_sec"] = time.perf_counter() - start
    details["n_output_rows"] = int(len(result))
    details["suppressed_clusters"] = int(suppressed_clusters)
    details["dependency_groups"] = group_map
    details["schema_actions"] = schema_actions
    details["source_weights"] = {k: round(float(v), 4) for k, v in weights.items()}
    details["certificates"] = certs
    return result.reset_index(drop=True), details
