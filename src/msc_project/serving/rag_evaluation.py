"""CPU-only calibration and paired evaluation of complete-review RAG scores.

The calibration objective filters labels AND candidate scores to seen aspects.
Evaluation includes every registered review, with failed/missing predictions
counted as empty outputs and separately reported as operational failures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from msc_project.serving.presence_policy import (
    active_presence_threshold, finite_margin, presence_space, presence_value,
)

SENTIMENTS = ("negative", "neutral", "positive")
SCOPES = ("overall", "seen", "heldout")
METRICS = ("presence", "pair")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def records_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = str(row.get("request_id") or row.get("response", {}).get("request_id") or "")
        if not key or key in result:
            raise ValueError("Empty or duplicate request_id; repeated attempts must be resolved first")
        response_id = row.get("response", {}).get("request_id")
        if response_id is not None and str(response_id) != key:
            raise ValueError("Response identity disagrees with its prediction record")
        result[key] = dict(row)
    return result


def gold_by_id(rows: Sequence[Mapping[str, Any]], aspects: Sequence[str]) -> dict[str, set]:
    result = {}
    allowed = set(aspects)
    for row in rows:
        key = str(row.get("request_id", ""))
        if not key or key in result:
            raise ValueError("Empty or duplicate label request_id")
        pairs = {(str(p["aspect"]), str(p["sentiment"])) for p in row["pairs"]}
        if any(a not in allowed or s not in SENTIMENTS for a, s in pairs):
            raise ValueError("Unknown aspect or sentiment in gold labels")
        if "aspects" in row and set(row["aspects"]) != {a for a, _ in pairs}:
            raise ValueError("Gold presence labels disagree with gold aspect-sentiment pairs")
        result[key] = pairs
    if not result:
        raise ValueError("Empty labelled evaluation population")
    return result


def successful(record: Mapping[str, Any] | None) -> bool:
    if not record:
        return False
    if "success" in record:
        return bool(record["success"])
    return record.get("status", "ok") in ("ok", "success", "completed", 200) and bool(
        record.get("response"))


def _probability(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected finite probability in [0, 1]")
    return value


def validate_record_spaces(records: Mapping[str, Mapping], expected: str) -> None:
    presence_space({"presence_score_space": expected})
    for record in records.values():
        if successful(record) and presence_space(record["response"]) != expected:
            raise ValueError("Mixed or unexpected presence score space in method records")


def score_maps(record: Mapping[str, Any], aspects: Sequence[str], *,
               presence_score_space: str = "probability") -> tuple[dict, dict]:
    presence_space({"presence_score_space": presence_score_space})
    response = record["response"]
    presence = {}
    for item in response["aspect_scores"]:
        aspect = str(item["aspect"])
        if aspect in presence:
            raise ValueError("Duplicate aspect score")
        _probability(item["score"])
        presence[aspect] = presence_value(item, presence_score_space)
    sentiment = {}
    for item in response.get("full_grid_scores", []):
        aspect = str(item["aspect"])
        if aspect in sentiment:
            raise ValueError("Duplicate full-grid sentiment score")
        values = item["sentiment_scores"]
        if set(values) != set(SENTIMENTS):
            raise ValueError("Full-grid sentiment labels differ from the scoring contract")
        scores = [_probability(values[s]) for s in SENTIMENTS]
        if not np.isclose(sum(scores), 1.0, atol=1e-4):
            raise ValueError("Sentiment score vector is not normalized")
        sentiment[aspect] = scores
    if set(presence) != set(aspects) or set(sentiment) != set(aspects):
        raise ValueError("Successful records must provide the complete candidate score grid")
    return presence, sentiment


def decode_scores(presence: Mapping[str, float], sentiment: Mapping[str, Sequence[float]],
                  aspect_threshold: float, runner_threshold: float) -> set[tuple[str, str]]:
    result = set()
    for aspect, score in presence.items():
        if score < aspect_threshold:
            continue
        values = np.asarray(sentiment[aspect], dtype=float)
        order = np.argsort(-values, kind="stable")
        result.add((aspect, SENTIMENTS[int(order[0])]))
        if values[order[1]] >= runner_threshold:
            result.add((aspect, SENTIMENTS[int(order[1])]))
    return result


def decode_records(records: Mapping[str, Mapping], keys: Sequence[str], aspects: Sequence[str],
                   aspect_threshold: float, runner_threshold: float, *,
                   presence_score_space: str = "probability") -> dict[str, set]:
    presence_space({"presence_score_space": presence_score_space})
    (finite_margin if presence_score_space == "logit_margin" else _probability)(aspect_threshold)
    _probability(runner_threshold)
    result = {}
    for key in keys:
        record = records.get(key)
        if successful(record):
            presence, sentiment = score_maps(record, aspects, presence_score_space=presence_score_space)
            result[key] = decode_scores(presence, sentiment, aspect_threshold, runner_threshold)
        else:
            result[key] = set()
    return result


def threshold_candidates(scores: Sequence[float], old_threshold: float) -> list[float]:
    old = _probability(old_threshold)
    values = np.asarray(scores, dtype=float)
    if values.size and (not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
        raise ValueError("Invalid calibration scores")
    quantiles = np.quantile(values, np.arange(1, 100) / 100) if values.size else []
    return sorted({0.0, 1.0, old, *map(float, quantiles)})


def margin_threshold_candidates(scores: Sequence[float], old_probability: float) -> list[float]:
    old = finite_margin(old_probability)
    values = np.asarray(scores, dtype=np.float64)
    if values.size and not np.isfinite(values).all():
        raise ValueError("Invalid finite calibration margins")
    if values.size:
        endpoints = [float(np.nextafter(values.min(), -np.inf)),
                     float(np.nextafter(values.max(), np.inf))]
        quantiles = np.quantile(values, np.arange(1, 100) / 100)
    else:
        endpoints, quantiles = [-1.0, 1.0], []
    old_margin = [math.log(old) - math.log1p(-old)] if 0 < old < 1 else []
    result = sorted({*endpoints, *map(float, quantiles), *old_margin})
    if not all(math.isfinite(value) for value in result) or len(result) > 102:
        raise ValueError("Margin grid requires at most102 finite thresholds")
    return result


def count_tensor(keys: Sequence[str], predictions: Mapping[str, set], gold: Mapping[str, set],
                 aspects: Sequence[str], heldout: str) -> np.ndarray:
    if heldout not in aspects:
        raise ValueError("Heldout aspect absent from taxonomy")
    scopes = (set(aspects), set(aspects) - {heldout}, {heldout})
    counts = np.zeros((len(keys), 3, 2, 3), dtype=np.int64)
    for i, key in enumerate(keys):
        for j, allowed in enumerate(scopes):
            pred = {p for p in predictions.get(key, set()) if p[0] in allowed}
            target = {p for p in gold[key] if p[0] in allowed}
            for k, (p, t) in enumerate((({a for a, _ in pred}, {a for a, _ in target}),
                                        (pred, target))):
                counts[i, j, k] = len(p & t), len(p - t), len(t - p)
    return counts


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = np.moveaxis(np.asarray(counts), -1, 0)
    denominator = 2 * tp + fp + fn
    return np.divide(2.0 * tp, denominator, out=np.zeros_like(tp, dtype=float),
                     where=denominator > 0)


def select_threshold(records: Mapping[str, Mapping], gold: Mapping[str, set],
                     aspects: Sequence[str], heldout: str, old_threshold: float,
                     runner_threshold: float, *, presence_score_space: str = "probability") -> dict:
    """No heldout label or heldout candidate score enters candidate/objective selection."""
    presence_space({"presence_score_space": presence_score_space})
    if set(records) - set(gold):
        raise ValueError("Prediction IDs outside registered calibration population")
    keys = sorted(gold)
    seen = set(aspects) - {heldout}
    if heldout not in aspects or not seen:
        raise ValueError("Expected heldout and seen aspects")
    seen_gold = {key: {p for p in gold[key] if p[0] in seen} for key in keys}
    maps = {key: score_maps(record, aspects, presence_score_space=presence_score_space) for key, record in records.items()
            if successful(record)}
    values = [value for p, _ in maps.values() for a, value in p.items() if a in seen]
    candidates = (margin_threshold_candidates if presence_score_space == "logit_margin"
                  else threshold_candidates)(values, old_threshold)
    objectives = []
    for threshold in candidates:
        tp_pair = fp_pair = fn_pair = tp_aspect = fp_aspect = fn_aspect = calls = 0
        for key in keys:
            p, s = maps.get(key, ({}, {}))
            prediction = decode_scores({a: v for a, v in p.items() if a in seen}, s,
                                       threshold, runner_threshold)
            target = seen_gold[key]
            pa, ta = {a for a, _ in prediction}, {a for a, _ in target}
            tp_pair += len(prediction & target)
            fp_pair += len(prediction - target)
            fn_pair += len(target - prediction)
            tp_aspect += len(pa & ta)
            fp_aspect += len(pa - ta)
            fn_aspect += len(ta - pa)
            calls += len(pa)
        pair_f1 = float(f1_from_counts(np.array([tp_pair, fp_pair, fn_pair])))
        presence_f1 = float(f1_from_counts(np.array([tp_aspect, fp_aspect, fn_aspect])))
        objectives.append((pair_f1, presence_f1, -calls, threshold))
    best = max(objectives)
    result = {"aspect_threshold": best[3], "runner_up_threshold": runner_threshold,
            "old_aspect_threshold": old_threshold, "candidate_count": len(candidates),
            "candidates": candidates, "seen_pair_f1": best[0], "seen_presence_f1": best[1],
            "seen_aspect_calls": -best[2], "calibration_review_count": len(keys),
            "selection_labels": "seen_aspects_only", "heldout_used_for_selection": False,
            "tie_rule": "pair_f1,presence_f1,fewer_seen_calls,higher_threshold"}
    if presence_score_space == "logit_margin":
        result.update(presence_score_space="logit_margin", presence_margin_threshold=best[3],
                      aspect_threshold=old_threshold,
                      candidate_space="logit_margin_float64",
                      endpoint_scope="finite_all_accept_all_reject_on_seen_calibration_margins")
    return result


def bootstrap_weights(n: int, draws: int = 2000, seed: int = 20261111) -> np.ndarray:
    if n < 1 or draws < 1:
        raise ValueError("Positive review count and bootstrap draws required")
    indices = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    weights = np.zeros((draws, n), dtype=np.int32)
    np.add.at(weights, (np.arange(draws)[:, None], indices), 1)
    return weights


def sampled_totals(counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights @ counts.reshape(len(counts), -1)).reshape(len(weights), 3, 2, 3)


def summarize_counts(counts: np.ndarray, weights: np.ndarray | None = None) -> dict:
    totals = counts.sum(axis=0)
    sampled = sampled_totals(counts, weights) if weights is not None else None
    scores = f1_from_counts(totals)
    result = {"review_count": len(counts), "f1_scale": "0_to_1", "scopes": {}}
    for j, scope in enumerate(SCOPES):
        entry = {}
        for k, metric in enumerate(METRICS):
            tp, fp, fn = map(int, totals[j, k])
            item = {"tp": tp, "fp": fp, "fn": fn, "gold_support": tp + fn,
                    "micro_f1": float(scores[j, k]),
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0}
            if sampled is not None:
                item["f1_percentile_95_interval"] = np.quantile(
                    f1_from_counts(sampled[:, j, k]), [.025, .975]).tolist()
            entry[metric] = item
        result["scopes"][scope] = entry
    return result


def _relative(new: float, baseline: float, *, reduction: bool = False) -> float | None:
    if baseline == 0:
        return None
    return float(100 * ((baseline - new) / baseline if reduction else new / baseline - 1))


def _relative_interval(new: np.ndarray, baseline: np.ndarray,
                       *, reduction: bool = False) -> dict:
    valid = baseline != 0
    values = (100 * ((baseline[valid] - new[valid]) / baseline[valid] if reduction else
                     new[valid] / baseline[valid] - 1))
    return {"percentile_95_interval": np.quantile(values, [.025, .975]).tolist()
            if len(values) else None, "undefined_zero_baseline_draws": int((~valid).sum())}


def paired_contrast(new: np.ndarray, baseline: np.ndarray, weights: np.ndarray) -> dict:
    if new.shape != baseline.shape:
        raise ValueError("Paired population differs")
    a, b = new.sum(axis=0), baseline.sum(axis=0)
    sa, sb = sampled_totals(new, weights), sampled_totals(baseline, weights)
    result = {"review_count": len(new), "direction": "new_vs_baseline", "scopes": {}}
    for j, scope in enumerate(SCOPES):
        fa, fb = float(f1_from_counts(a[j, 1])), float(f1_from_counts(b[j, 1]))
        sfa, sfb = f1_from_counts(sa[:, j, 1]), f1_from_counts(sb[:, j, 1])
        entry = {"pair_f1_relative_gain_percent": _relative(fa, fb),
                 "pair_f1_difference_points": 100 * (fa - fb),
                 "pair_f1_difference_points_interval":
                     np.quantile(100 * (sfa - sfb), [.025, .975]).tolist(),
                 "pair_f1_relative_gain_uncertainty": _relative_interval(sfa, sfb)}
        for label, index in (("fp", 1), ("fn", 2)):
            entry[f"presence_{label}_relative_reduction_percent"] = _relative(
                int(a[j, 0, index]), int(b[j, 0, index]), reduction=True)
            entry[f"presence_{label}_reduction_uncertainty"] = _relative_interval(
                sa[:, j, 0, index], sb[:, j, 0, index], reduction=True)
        result["scopes"][scope] = entry
    return result
