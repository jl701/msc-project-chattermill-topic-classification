"""Deterministic description-to-classifier-weight transfer utilities.

The target aspect is represented only by its descriptor.  Target labels and a
target-trained classifier are intentionally absent from every fitting API in
this module so that the outer LOAO boundary is difficult to violate by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score


GENERATOR_FAMILIES = (
    "kernel_ridge",
    "mean_seen_weight",
    "nearest_description_weight",
    "cosine_barycentric_weight",
)


@dataclass(frozen=True)
class PresenceThresholdSelection:
    threshold: float
    f1: float
    average_precision: float
    precision: float
    recall: float
    candidates_evaluated: int


def sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Logits must be finite.")
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Descriptor matrix must be finite and two-dimensional.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Descriptor vectors must have non-zero norm.")
    return matrix / norms


def _validate_transfer_inputs(
    descriptors: np.ndarray,
    weights: np.ndarray,
    target_descriptor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    descriptor_matrix = _normalise_rows(descriptors)
    weight_matrix = np.asarray(weights, dtype=float)
    target = np.asarray(target_descriptor, dtype=float).reshape(1, -1)
    target = _normalise_rows(target)[0]
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != descriptor_matrix.shape[0]:
        raise ValueError("Seen descriptors and classifier weights must align.")
    if descriptor_matrix.shape[0] < 2 or target.shape[0] != descriptor_matrix.shape[1]:
        raise ValueError("Transfer needs at least two seen aspects and a compatible target.")
    if not np.isfinite(weight_matrix).all():
        raise ValueError("Classifier parameters must be finite.")
    return descriptor_matrix, weight_matrix, target


def synthesise_weight(
    descriptors: np.ndarray,
    weights: np.ndarray,
    target_descriptor: np.ndarray,
    *,
    family: str,
    alpha: float | None = None,
    temperature: float | None = None,
) -> np.ndarray:
    """Synthesise one target parameter vector from seen descriptors and weights."""

    x, w, target = _validate_transfer_inputs(descriptors, weights, target_descriptor)
    if family == "mean_seen_weight":
        predicted = w.mean(axis=0)
    elif family == "nearest_description_weight":
        predicted = w[int(np.argmax(x @ target))]
    elif family == "cosine_barycentric_weight":
        if temperature is None or float(temperature) <= 0:
            raise ValueError("Barycentric transfer requires a positive temperature.")
        logits = (x @ target) / float(temperature)
        logits -= logits.max()
        coefficients = np.exp(logits)
        coefficients /= coefficients.sum()
        predicted = coefficients @ w
    elif family == "kernel_ridge":
        if alpha is None or float(alpha) <= 0:
            raise ValueError("Kernel-ridge transfer requires positive alpha.")
        x_mean = x.mean(axis=0, keepdims=True)
        w_mean = w.mean(axis=0, keepdims=True)
        x_centered = x - x_mean
        w_centered = w - w_mean
        kernel = x_centered @ x_centered.T
        dual = np.linalg.solve(
            kernel + float(alpha) * np.eye(len(kernel), dtype=float),
            w_centered,
        )
        predicted = w_mean[0] + ((target - x_mean[0]) @ x_centered.T) @ dual
    else:
        raise ValueError(f"Unknown transfer family: {family!r}.")
    result = np.asarray(predicted, dtype=float).reshape(-1)
    if result.shape != (w.shape[1],) or not np.isfinite(result).all():
        raise ValueError("Synthesised classifier parameters are invalid.")
    return result


def parameter_direction_cosine(observed: np.ndarray, predicted: np.ndarray) -> float:
    left = np.asarray(observed, dtype=float).reshape(-1)
    right = np.asarray(predicted, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0 else 0.0


def select_presence_threshold(
    targets: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> PresenceThresholdSelection:
    truth = np.asarray(targets, dtype=int).reshape(-1)
    probability = np.asarray(scores, dtype=float).reshape(-1)
    if truth.shape != probability.shape or not len(truth):
        raise ValueError("Presence targets and scores must align.")
    if set(np.unique(truth)) - {0, 1} or len(np.unique(truth)) != 2:
        raise ValueError("Presence threshold selection requires both target classes.")
    if not np.isfinite(probability).all():
        raise ValueError("Presence scores must be finite.")
    average_precision = float(average_precision_score(truth, probability))
    order = np.argsort(-probability, kind="stable")
    sorted_scores = probability[order]
    sorted_truth = truth[order]
    cumulative_tp = np.cumsum(sorted_truth, dtype=np.int64)
    cumulative_predicted = np.arange(1, len(truth) + 1, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    total_positive = int(truth.sum())
    candidates: list[tuple[float, int, int]] = [
        (np.nextafter(float(sorted_scores[0]), float("inf")), 0, 0)
    ]
    candidates.extend(
        (
            float(sorted_scores[index]),
            int(cumulative_tp[index]),
            int(cumulative_predicted[index]),
        )
        for index in group_ends
    )
    best: tuple[tuple[float, ...], PresenceThresholdSelection] | None = None
    for threshold, tp, predicted_count in candidates:
        fp = predicted_count - tp
        fn = total_positive - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        selection = PresenceThresholdSelection(
            threshold=float(threshold),
            f1=float(f1),
            average_precision=average_precision,
            precision=float(precision),
            recall=float(recall),
            candidates_evaluated=int(len(candidates)),
        )
        rank = (f1, precision, recall, float(threshold))
        if best is None or rank > best[0]:
            best = (rank, selection)
    if best is None:
        raise AssertionError("Presence threshold selection produced no candidate.")
    return best[1]


def pseudo_unseen_scores(
    descriptors: np.ndarray,
    weights: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    family: str,
    alpha: float | None = None,
    temperature: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run leave-one-seen-aspect-out meta-validation over seen aspects only."""

    x = _normalise_rows(descriptors)
    w = np.asarray(weights, dtype=float)
    features = np.asarray(validation_features, dtype=float)
    targets = np.asarray(validation_targets, dtype=int)
    aspect_count = x.shape[0]
    if w.shape[0] != aspect_count or features.ndim != 2:
        raise ValueError("Pseudo-unseen inputs do not align.")
    if targets.shape != (features.shape[0], aspect_count):
        raise ValueError("Pseudo-unseen validation target matrix is invalid.")
    if features.shape[1] + 1 != w.shape[1]:
        raise ValueError("Latent review features and classifier parameters do not align.")
    pooled_scores: list[np.ndarray] = []
    pooled_targets: list[np.ndarray] = []
    cosines: list[float] = []
    for target_index in range(aspect_count):
        keep = np.arange(aspect_count) != target_index
        predicted = synthesise_weight(
            x[keep],
            w[keep],
            x[target_index],
            family=family,
            alpha=alpha,
            temperature=temperature,
        )
        pooled_scores.append(
            sigmoid(features @ predicted[:-1] + predicted[-1])
        )
        pooled_targets.append(targets[:, target_index])
        cosines.append(parameter_direction_cosine(w[target_index], predicted))
    return (
        np.concatenate(pooled_targets),
        np.concatenate(pooled_scores),
        float(np.mean(cosines)),
    )
