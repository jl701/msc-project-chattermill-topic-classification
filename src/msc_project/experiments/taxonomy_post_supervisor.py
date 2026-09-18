"""Utilities for the post-supervisor Level 2 N/D/R experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from msc_project.experiments.taxonomy_protocol import TaxonomyFold, registered_folds
from msc_project.experiments.taxonomy_two_stage import (
    capped_two_sentiment_prediction_mask,
    conditional_sentiment_metrics,
    evaluate_prediction_mask,
    two_stage_candidates,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS


L2_CONDITIONS = ("N", "D", "R")


def post_supervisor_l2_folds() -> tuple[TaxonomyFold, ...]:
    """Return the twelve legacy-compatible LOAO folds with the new N/D/R sweep."""

    folds = tuple(
        replace(fold, conditions=L2_CONDITIONS)
        for fold in registered_folds("L2")
    )
    if len(folds) != 12 or any(len(fold.heldout_aspects) != 1 for fold in folds):
        raise AssertionError("Post-supervisor Level 2 must contain twelve LOAO folds.")
    return folds


def score_frame_sha256(frame: pd.DataFrame) -> str:
    columns = (
        "row_uid",
        "candidate_aspect",
        "candidate_sentiment",
        "representation_variant",
        "target",
        "aspect_score",
        "sentiment_score",
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Score frame is missing columns: {missing}.")
    records = (
        frame[list(columns)]
        .sort_values(
            ["row_uid", "candidate_aspect", "candidate_sentiment"],
            kind="stable",
        )
        .to_dict(orient="records")
    )
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _presence_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    if targets.shape != scores.shape or targets.ndim != 1 or not len(targets):
        raise ValueError("Presence targets and scores must be aligned one-dimensional arrays.")
    if not np.isfinite(scores).all():
        raise ValueError("Presence scores must be finite.")
    truth = targets.astype(bool)
    prediction = scores >= float(threshold)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * tp / denominator if denominator else 0.0),
        "average_precision": float(average_precision_score(truth, scores)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_rows": int(truth.sum()),
        "predicted_positive_rows": int(prediction.sum()),
        "false_positive_rows_per_100": float(fp * 100 / len(truth)),
        "positive_score_mean": float(scores[truth].mean()) if truth.any() else 0.0,
        "negative_score_mean": float(scores[~truth].mean()) if (~truth).any() else 0.0,
    }


def _heldout_rank_diagnostics(
    scored: pd.DataFrame,
    heldout_aspect: str,
) -> dict[str, float | int]:
    candidates = two_stage_candidates(scored)
    gold_presence = (
        scored.assign(_gold=scored["target"].astype(int).eq(1))
        .groupby(["row_uid", "candidate_aspect"], sort=False)["_gold"]
        .any()
    )
    candidates["is_gold_aspect"] = [
        bool(gold_presence.get((str(uid), str(aspect)), False))
        for uid, aspect in zip(candidates["row_uid"], candidates["candidate_aspect"])
    ]
    candidates = candidates.sort_values(
        ["row_uid", "aspect_score", "candidate_aspect"],
        ascending=[True, False, True],
        kind="stable",
    )
    candidates["rank"] = candidates.groupby("row_uid", sort=False).cumcount() + 1
    heldout = candidates[
        candidates["candidate_aspect"].astype(str).eq(heldout_aspect)
        & candidates["is_gold_aspect"]
    ]
    if heldout.empty:
        return {
            "positive_rows": 0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mean_reciprocal_rank": 0.0,
            "top_1_rate": 0.0,
            "top_3_rate": 0.0,
        }
    ranks = heldout["rank"].to_numpy(dtype=float)
    return {
        "positive_rows": int(len(ranks)),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "top_1_rate": float(np.mean(ranks <= 1)),
        "top_3_rate": float(np.mean(ranks <= 3)),
    }


def _oracle_gate_mask(
    scored: pd.DataFrame,
    *,
    second_sentiment_threshold: float,
) -> pd.Series:
    selected = capped_two_sentiment_prediction_mask(
        scored,
        aspect_threshold=-1.0,
        second_sentiment_threshold=second_sentiment_threshold,
    )
    gold_presence = (
        scored.assign(_gold=scored["target"].astype(int).eq(1))
        .groupby(["row_uid", "candidate_aspect"], sort=False)["_gold"]
        .any()
    )
    open_gate = pd.Series(
        [
            bool(gold_presence.get((str(uid), str(aspect)), False))
            for uid, aspect in zip(scored["row_uid"], scored["candidate_aspect"])
        ],
        index=scored.index,
        dtype=bool,
    )
    return selected & open_gate


def evaluate_l2_condition(
    scored: pd.DataFrame,
    fold: TaxonomyFold,
    *,
    aspect_threshold: float,
    second_sentiment_threshold: float,
) -> dict[str, object]:
    """Derive the L2-S oracle view and L2-E full pipeline from one score frame."""

    if fold.level != "L2" or len(fold.heldout_aspects) != 1:
        raise ValueError("L2 evaluation requires one post-supervisor LOAO fold.")
    heldout_aspect = fold.heldout_aspects[0]
    available = set(scored["candidate_aspect"].astype(str))
    if available != set(fold.evaluation_aspects):
        raise ValueError("L2 score frame does not cover all twelve candidates.")
    scores = scored[["aspect_score", "sentiment_score"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("L2 score frame contains non-finite values.")

    heldout = scored[
        scored["candidate_aspect"].astype(str).eq(heldout_aspect)
    ].copy()
    heldout_candidates = two_stage_candidates(heldout)
    heldout_presence = (
        heldout.assign(_gold=heldout["target"].astype(int).eq(1))
        .groupby("row_uid", sort=False)["_gold"]
        .any()
    )
    target = np.asarray(
        [bool(heldout_presence.get(str(uid), False)) for uid in heldout_candidates["row_uid"]],
        dtype=int,
    )
    presence = _presence_metrics(
        target,
        heldout_candidates["aspect_score"].to_numpy(dtype=float),
        aspect_threshold,
    )
    primary_mask = capped_two_sentiment_prediction_mask(
        scored,
        aspect_threshold=aspect_threshold,
        second_sentiment_threshold=second_sentiment_threshold,
    )
    selected_counts = (
        scored.loc[primary_mask]
        .groupby(["row_uid", "candidate_aspect"], sort=False)
        .size()
    )
    if len(selected_counts) and int(selected_counts.max()) > 2:
        raise AssertionError("The post-supervisor decoder emitted a third sentiment.")
    oracle_mask = _oracle_gate_mask(
        heldout,
        second_sentiment_threshold=second_sentiment_threshold,
    )

    partitions = {
        "overall": fold.evaluation_aspects,
        "seen": fold.seen_aspects,
        "heldout": fold.heldout_aspects,
    }
    return {
        "score_sha256": score_frame_sha256(scored),
        "L2_S": {
            "aspect_presence": presence,
            "positive_review_rank": _heldout_rank_diagnostics(
                scored, heldout_aspect
            ),
            "oracle_aspect_gated_sentiment": {
                "top_one_conditional_accuracy": conditional_sentiment_metrics(
                    heldout
                ),
                "capped_two_pair_metrics": evaluate_prediction_mask(
                    heldout,
                    oracle_mask,
                    aspects=fold.heldout_aspects,
                ),
            },
        },
        "L2_E": {
            "partitions": {
                name: evaluate_prediction_mask(
                    scored,
                    primary_mask,
                    aspects=aspects,
                )
                for name, aspects in partitions.items()
            },
            "maximum_sentiments_per_selected_aspect": (
                int(selected_counts.max()) if len(selected_counts) else 0
            ),
            "selected_aspect_instances": int(len(selected_counts)),
        },
    }


def aggregate_l2_folds(
    fold_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(fold_results) != 12:
        raise ValueError("A full Level 2 aggregate requires twelve folds.")
    rows: list[dict[str, object]] = []
    for fold in fold_results:
        conditions = fold.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != set(L2_CONDITIONS):
            raise ValueError("Every Level 2 fold must contain N, D and R.")
        for condition in L2_CONDITIONS:
            result = conditions[condition]
            if not isinstance(result, Mapping):
                raise ValueError("Condition result is not an object.")
            l2_s = result["L2_S"]  # type: ignore[index]
            l2_e = result["L2_E"]  # type: ignore[index]
            rows.append(
                {
                    "fold_id": str(fold["fold_id"]),
                    "condition": condition,
                    "heldout_presence_f1": float(l2_s["aspect_presence"]["f1"]),  # type: ignore[index]
                    "heldout_presence_ap": float(l2_s["aspect_presence"]["average_precision"]),  # type: ignore[index]
                    "heldout_pair_micro_f1": float(l2_e["partitions"]["heldout"]["pair_micro_f1"]),  # type: ignore[index]
                    "overall_pair_micro_f1": float(l2_e["partitions"]["overall"]["pair_micro_f1"]),  # type: ignore[index]
                    "seen_pair_micro_f1": float(l2_e["partitions"]["seen"]["pair_micro_f1"]),  # type: ignore[index]
                }
            )
    frame = pd.DataFrame.from_records(rows)
    metrics = (
        "heldout_presence_f1",
        "heldout_presence_ap",
        "heldout_pair_micro_f1",
        "overall_pair_micro_f1",
        "seen_pair_micro_f1",
    )
    means = {
        condition: {
            metric: float(
                frame.loc[frame["condition"].eq(condition), metric].mean()
            )
            for metric in metrics
        }
        for condition in L2_CONDITIONS
    }
    pivot = frame.pivot(index="fold_id", columns="condition", values=list(metrics))
    contrasts = {}
    for left, right, name in (("N", "D", "D_minus_N"), ("D", "R", "R_minus_D")):
        contrasts[name] = {
            metric: {
                "mean": float((pivot[(metric, right)] - pivot[(metric, left)]).mean()),
                "improved_folds": int((pivot[(metric, right)] > pivot[(metric, left)]).sum()),
                "tied_folds": int((pivot[(metric, right)] == pivot[(metric, left)]).sum()),
                "worsened_folds": int((pivot[(metric, right)] < pivot[(metric, left)]).sum()),
            }
            for metric in metrics
        }
    return {
        "records": frame.to_dict(orient="records"),
        "condition_macro_means": means,
        "paired_fold_contrasts": contrasts,
    }
