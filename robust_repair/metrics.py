"""Evaluation metrics for repaired integration outputs."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .constraints import value_equal, violation_rate
from .generator import ATTRS


def _truth_map(master: pd.DataFrame) -> Dict[int, dict]:
    return {int(r["eid"]): r.to_dict() for _, r in master.iterrows()}


def _best_output_by_eid(output: pd.DataFrame) -> Dict[int, dict]:
    """Choose the highest-evidence output row per hidden true id for cell scoring.

    Duplicate or split clusters are not removed silently: they are penalized by
    entity_precision/entity_f1 and duplicate_output_rate in addition to cell
    accuracy and aggregate distortion.
    """
    if len(output) == 0 or "cluster_true_eid" not in output.columns:
        return {}
    tmp = output.sort_values(["cluster_true_eid", "record_count", "source_count"], ascending=[True, False, False])
    tmp = tmp.drop_duplicates(subset=["cluster_true_eid"], keep="first")
    return {int(r["cluster_true_eid"]): r.to_dict() for _, r in tmp.iterrows()}


def entity_metrics(master: pd.DataFrame, output: pd.DataFrame) -> Dict[str, float]:
    n_truth = len(master)
    if len(output) == 0 or "cluster_true_eid" not in output.columns:
        return {
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_f1": 0.0,
            "duplicate_output_rate": 0.0,
            "missing_entity_rate": 1.0,
        }
    valid = output[output["cluster_true_eid"] >= 0]
    unique_eids = set(map(int, valid["cluster_true_eid"].unique()))
    tp = len(unique_eids & set(map(int, master["eid"].tolist())))
    precision = tp / max(1, len(output))
    recall = tp / max(1, n_truth)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    duplicate_rate = max(0, len(output) - tp) / max(1, len(output))
    missing_rate = max(0, n_truth - tp) / max(1, n_truth)
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "duplicate_output_rate": duplicate_rate,
        "missing_entity_rate": missing_rate,
    }


def cell_accuracy(master: pd.DataFrame, output: pd.DataFrame) -> float:
    truth = _truth_map(master)
    correct = 0
    total = len(master) * len(ATTRS)
    out_by_eid = _best_output_by_eid(output)
    for eid, tr in truth.items():
        out = out_by_eid.get(eid)
        if out is None:
            continue
        for attr in ATTRS:
            if value_equal(attr, out.get(attr), tr[attr]):
                correct += 1
    return correct / total if total else 0.0


def integrated_quality(master: pd.DataFrame, output: pd.DataFrame) -> float:
    em = entity_metrics(master, output)
    return cell_accuracy(master, output) * em["entity_f1"]


def aggregate_distortion(master: pd.DataFrame, output: pd.DataFrame) -> float:
    if len(output) == 0:
        return 1.0
    truth = master.groupby("category")["revenue"].sum()
    pred = output.groupby("category")["revenue"].sum()
    cats = sorted(set(truth.index) | set(pred.index))
    err = 0.0
    denom = 0.0
    for c in cats:
        t = float(truth.get(c, 0.0))
        p = float(pred.get(c, 0.0))
        err += abs(p - t)
        denom += abs(t)
    return err / max(1.0, denom)


def per_source_detection(master: pd.DataFrame, records: pd.DataFrame, output: pd.DataFrame) -> Dict[str, float]:
    truth = _truth_map(master)
    out_by_eid = _best_output_by_eid(output)
    fp = tn = fn = tp = 0
    for r in records.itertuples(index=False):
        eid = int(getattr(r, "eid_true"))
        tr = truth.get(eid)
        out = out_by_eid.get(eid)
        if tr is None or out is None:
            continue
        for attr in ATTRS:
            claim = getattr(r, attr)
            dirty = not value_equal(attr, claim, tr[attr])
            flagged = not value_equal(attr, claim, out[attr])
            if flagged and dirty:
                tp += 1
            elif flagged and not dirty:
                fp += 1
            elif not flagged and dirty:
                fn += 1
            else:
                tn += 1
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"false_positive_rate": fpr, "false_negative_rate": fnr, "repair_precision": precision, "repair_recall": recall}


def evaluate(
    master: pd.DataFrame,
    records: pd.DataFrame,
    output: pd.DataFrame,
    runtime_sec: float,
    baseline_distortion: float | None = None,
) -> Dict[str, float]:
    acc = cell_accuracy(master, output)
    dist = aggregate_distortion(master, output)
    q = 1.0 - violation_rate(output)
    det = per_source_detection(master, records, output)
    ent = entity_metrics(master, output)
    if baseline_distortion is not None and baseline_distortion > 1e-9:
        attack_reduction = 1.0 - dist / baseline_distortion
    else:
        attack_reduction = 0.0
    row = {
        "cell_accuracy": acc,
        "integrated_quality": acc * ent["entity_f1"],
        "aggregate_distortion": dist,
        "data_quality_score": q,
        "attack_success_reduction": attack_reduction,
        "runtime_sec": runtime_sec,
        "n_output_rows": float(len(output)),
    }
    row.update(ent)
    row.update(det)
    return row
