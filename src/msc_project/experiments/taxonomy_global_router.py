"""Validation-only global routing for matched two-stage taxonomy systems."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_protocol import PAIR_KEY
from msc_project.experiments.taxonomy_two_stage import (
    capped_two_sentiment_prediction_mask,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS


@dataclass(frozen=True)
class ComponentArrays:
    """Threshold-aligned scores and frozen predictions for one component."""

    method_id: str
    ordered_frame: pd.DataFrame
    target: np.ndarray
    prediction: np.ndarray
    aspect_score: np.ndarray
    sentiment_score: np.ndarray
    aspect_present: np.ndarray
    row_uid: np.ndarray
    candidate_aspect: np.ndarray
    is_heldout: np.ndarray


def threshold_aligned_scores(
    scores: np.ndarray | pd.Series,
    threshold: float,
) -> np.ndarray:
    """Map a component threshold to 0.5 without changing its decisions.

    Values below the threshold are mapped linearly to [0, 0.5), and values at
    or above it are mapped to [0.5, 1].  Endpoint thresholds are handled
    explicitly so saturated probability systems retain exact binary decisions.
    """

    values = np.asarray(scores, dtype=float)
    boundary = float(threshold)
    if values.ndim != 1 or not len(values):
        raise ValueError("Threshold alignment requires a non-empty score vector.")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("Threshold alignment requires finite probabilities in [0,1].")
    if not np.isfinite(boundary) or not 0.0 <= boundary <= 1.0:
        raise ValueError("The component threshold must lie in [0,1].")

    output = np.empty_like(values, dtype=float)
    positive = values >= boundary
    if boundary <= 0.0:
        output[:] = 0.5 + 0.5 * values
    elif boundary >= 1.0:
        output[~positive] = 0.5 * values[~positive]
        output[positive] = 1.0
    else:
        output[~positive] = 0.5 * values[~positive] / boundary
        output[positive] = 0.5 + 0.5 * (
            values[positive] - boundary
        ) / (1.0 - boundary)
    if not np.array_equal(output >= 0.5, positive):
        raise AssertionError("Threshold alignment changed a component decision.")
    return np.clip(output, 0.0, 1.0)


def _ordered_group_frame(scored: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    required = {
        *PAIR_KEY,
        "target",
        "aspect_score",
        "sentiment_score",
        "method_id",
        "split",
        "is_heldout",
    }
    missing = sorted(required - set(scored.columns))
    if missing or scored.empty:
        raise ValueError(f"Router score frame is invalid; missing={missing}.")
    if set(scored["split"].astype(str)) != {"validation"}:
        raise ValueError("The global router accepts validation scores only.")
    if scored.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Router score frame contains duplicate pair identities.")
    ordered = scored.sort_values(list(PAIR_KEY), kind="stable").reset_index(drop=True)
    width = len(CANDIDATE_SENTIMENTS)
    group_sizes = ordered.groupby(
        ["row_uid", "candidate_aspect"], sort=False
    ).size()
    if set(group_sizes.astype(int)) != {width}:
        raise ValueError("Every router aspect instance must contain three sentiments.")
    positions = np.arange(len(ordered), dtype=np.int64).reshape(-1, width)
    for column in ("aspect_score", "sentiment_score"):
        values = ordered[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"Router {column} values must be finite probabilities.")
    aspect = ordered["aspect_score"].to_numpy(dtype=float).reshape(-1, width)
    if not np.all(aspect == aspect[:, :1]):
        raise ValueError("Aspect scores differ across sentiments for one candidate.")
    return ordered, positions


def build_component_arrays(
    scored: pd.DataFrame,
    *,
    aspect_threshold: float,
    runner_up_sentiment_threshold: float,
) -> ComponentArrays:
    """Create threshold-aligned arrays while preserving the frozen decoder."""

    ordered, _positions = _ordered_group_frame(scored)
    method_values = set(ordered["method_id"].astype(str))
    if len(method_values) != 1:
        raise ValueError("One component frame must contain exactly one method.")
    method_id = next(iter(method_values))
    aligned = ordered.copy()
    aligned["aspect_score"] = threshold_aligned_scores(
        aligned["aspect_score"], aspect_threshold
    )
    aligned["sentiment_score"] = threshold_aligned_scores(
        aligned["sentiment_score"], runner_up_sentiment_threshold
    )
    original_prediction = capped_two_sentiment_prediction_mask(
        ordered,
        aspect_threshold=float(aspect_threshold),
        second_sentiment_threshold=float(runner_up_sentiment_threshold),
    ).to_numpy(dtype=bool)
    aligned_prediction = capped_two_sentiment_prediction_mask(
        aligned,
        aspect_threshold=0.5,
        second_sentiment_threshold=0.5,
    ).to_numpy(dtype=bool)
    if not np.array_equal(original_prediction, aligned_prediction):
        raise AssertionError("Threshold alignment changed the capped-two decoder.")

    width = len(CANDIDATE_SENTIMENTS)
    aspect_scores = aligned["aspect_score"].to_numpy(dtype=float).reshape(-1, width)
    heldout = aligned["is_heldout"].astype(bool).to_numpy().reshape(-1, width)
    if not np.all(heldout == heldout[:, :1]):
        raise ValueError("Held-out status differs across sentiments for one aspect.")
    return ComponentArrays(
        method_id=method_id,
        ordered_frame=aligned,
        target=aligned["target"].astype(int).to_numpy().reshape(-1, width).astype(bool),
        prediction=aligned_prediction.reshape(-1, width),
        aspect_score=aspect_scores[:, 0],
        sentiment_score=aligned["sentiment_score"].to_numpy(dtype=float).reshape(-1, width),
        aspect_present=aspect_scores[:, 0] >= 0.5,
        row_uid=aligned["row_uid"].astype(str).to_numpy()[::width],
        candidate_aspect=aligned["candidate_aspect"].astype(str).to_numpy()[::width],
        is_heldout=heldout[:, 0],
    )


def validate_component_alignment(
    left: ComponentArrays,
    right: ComponentArrays,
) -> None:
    """Require two components to cover exactly the same labelled grid."""

    left_keys = left.ordered_frame[list(PAIR_KEY)].astype(str).to_numpy()
    right_keys = right.ordered_frame[list(PAIR_KEY)].astype(str).to_numpy()
    if not np.array_equal(left_keys, right_keys):
        raise ValueError("Router component pair identities differ.")
    if not np.array_equal(left.target, right.target):
        raise ValueError("Router component targets differ.")
    if not np.array_equal(left.is_heldout, right.is_heldout):
        raise ValueError("Router component held-out partitions differ.")


def route_mask(
    base: ComponentArrays,
    *,
    rescue_band: float,
    confirmation_band: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global route, rescue and confirmation masks per aspect instance."""

    rescue = float(rescue_band)
    confirmation = float(confirmation_band)
    for name, value in (("rescue", rescue), ("confirmation", confirmation)):
        if not np.isfinite(value) or value < -1.0 or value > 0.5:
            raise ValueError(f"{name} band must lie in [-1,0.5].")
    distance = np.abs(base.aspect_score - 0.5)
    rescue_mask = (
        (~base.aspect_present)
        & (rescue >= 0.0)
        & (distance <= rescue + 1e-15)
    )
    confirmation_mask = (
        base.aspect_present
        & (confirmation >= 0.0)
        & (distance <= confirmation + 1e-15)
    )
    return rescue_mask | confirmation_mask, rescue_mask, confirmation_mask


def _pair_counts(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(target, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError("Pair targets and predictions must be aligned matrices.")
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    return {
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "pair_precision": float(precision),
        "pair_recall": float(recall),
        "pair_micro_f1": float(2 * tp / denominator if denominator else 0.0),
        "pair_gold_label_count": int(truth.sum()),
        "pair_predicted_label_count": int(pred.sum()),
    }


def _presence_counts(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(target, dtype=bool).any(axis=1)
    pred = np.asarray(prediction, dtype=bool).any(axis=1)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    denominator = 2 * tp + fp + fn
    return {
        "presence_tp": tp,
        "presence_fp": fp,
        "presence_fn": fn,
        "presence_tn": tn,
        "presence_f1": float(2 * tp / denominator if denominator else 0.0),
    }


def policy_metrics(
    base: ComponentArrays,
    expert: ComponentArrays,
    *,
    rescue_band: float,
    confirmation_band: float,
) -> dict[str, float | int | str]:
    """Evaluate one hard router without using any outcome as an input feature."""

    validate_component_alignment(base, expert)
    routed, rescue, confirmation = route_mask(
        base,
        rescue_band=rescue_band,
        confirmation_band=confirmation_band,
    )
    prediction = np.where(routed[:, None], expert.prediction, base.prediction)
    heldout = base.is_heldout
    seen = ~heldout
    overall_counts = _pair_counts(base.target, prediction)
    heldout_counts = _pair_counts(base.target[heldout], prediction[heldout])
    seen_counts = _pair_counts(base.target[seen], prediction[seen])
    presence = _presence_counts(base.target[heldout], prediction[heldout])
    return {
        "base_method_id": base.method_id,
        "expert_method_id": expert.method_id,
        "rescue_band": float(rescue_band),
        "confirmation_band": float(confirmation_band),
        **{f"overall_{key}": value for key, value in overall_counts.items()},
        **{f"heldout_{key}": value for key, value in heldout_counts.items()},
        **{f"seen_{key}": value for key, value in seen_counts.items()},
        **{f"heldout_{key}": value for key, value in presence.items()},
        "aspect_instance_count": len(routed),
        "route_count": int(routed.sum()),
        "rescue_count": int(rescue.sum()),
        "confirmation_count": int(confirmation.sum()),
        "route_rate": float(routed.mean()),
    }


def routed_score_frame(
    base: ComponentArrays,
    expert: ComponentArrays,
    *,
    rescue_band: float,
    confirmation_band: float,
    method_id: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construct a deterministic routed score frame for full metric evaluation."""

    validate_component_alignment(base, expert)
    routed, rescue, confirmation = route_mask(
        base,
        rescue_band=rescue_band,
        confirmation_band=confirmation_band,
    )
    width = len(CANDIDATE_SENTIMENTS)
    broadcast = np.repeat(routed, width)
    frame = base.ordered_frame.copy()
    frame["aspect_score"] = np.where(
        broadcast,
        expert.ordered_frame["aspect_score"].to_numpy(dtype=float),
        base.ordered_frame["aspect_score"].to_numpy(dtype=float),
    )
    frame["sentiment_score"] = np.where(
        broadcast,
        expert.ordered_frame["sentiment_score"].to_numpy(dtype=float),
        base.ordered_frame["sentiment_score"].to_numpy(dtype=float),
    )
    frame["method_id"] = str(method_id)
    expected = np.where(routed[:, None], expert.prediction, base.prediction).reshape(-1)
    observed = capped_two_sentiment_prediction_mask(
        frame,
        aspect_threshold=0.5,
        second_sentiment_threshold=0.5,
    ).to_numpy(dtype=bool)
    if not np.array_equal(observed, expected):
        raise AssertionError("Routed aligned scores do not reproduce routed predictions.")
    route_frame = pd.DataFrame(
        {
            "row_uid": base.row_uid,
            "candidate_aspect": base.candidate_aspect,
            "base_present": base.aspect_present,
            "base_threshold_distance": np.abs(base.aspect_score - 0.5),
            "routed": routed,
            "rescue": rescue,
            "confirmation": confirmation,
            "is_heldout": base.is_heldout,
        }
    )
    return frame, pd.Series(observed, index=frame.index, dtype=bool), route_frame


def confusion_additive_f1(records: pd.DataFrame, prefix: str) -> float:
    """Pool pair confusion counts from policy records."""

    tp = int(records[f"{prefix}_pair_tp"].sum())
    fp = int(records[f"{prefix}_pair_fp"].sum())
    fn = int(records[f"{prefix}_pair_fn"].sum())
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator if denominator else 0.0)


def policy_id(record: Mapping[str, object]) -> str:
    return (
        f"{record['base_method_id']}__to__{record['expert_method_id']}"
        f"__rescue-{float(record['rescue_band']):.3f}"
        f"__confirm-{float(record['confirmation_band']):.3f}"
    )
