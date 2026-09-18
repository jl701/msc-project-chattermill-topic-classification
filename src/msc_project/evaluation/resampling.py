"""Synchronised review-cluster resampling for the validation policy comparison.

Axes are system, fold, review and confusion count (TP, FP, FN). Every system and
fold must use the same review order. Resampling changes review multiplicities,
never model weights, thresholds or the set of aspects. The original official
test uses its separately frozen analysis implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def f1_from_counts(counts: ArrayLike) -> NDArray[np.float64]:
    """Compute micro-F1 along a final TP/FP/FN axis; an empty denominator gives 0."""
    values = np.asarray(counts, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("Confusion counts must have a final TP/FP/FN axis of length 3")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Confusion counts must be finite and non-negative")
    denominator = 2 * values[..., 0] + values[..., 1] + values[..., 2]
    return np.divide(
        2 * values[..., 0],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )


def review_cluster_bootstrap(
    counts: ArrayLike,
    *,
    draws: int = 20_000,
    seed: int = 13,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return equal-fold and pooled F1 samples, each shaped ``(draws, systems)``.

    A multinomial draw is equivalent to sampling ``reviews`` review identities
    with replacement. Its weights are shared across all systems and folds.
    The batch size of 200 and RNG call sequence preserve the original policy
    analysis exactly for valid input. Intervals are computed by callers from
    paired differences, not by subtracting separate confidence intervals.
    """
    values = np.asarray(counts)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or min(values.shape) < 1
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Expected finite non-negative system x fold x review x TP/FP/FN counts")
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    systems, folds, reviews, _ = values.shape
    matrix = values.transpose(2, 0, 1, 3).reshape(reviews, -1).astype(float)
    rng = np.random.default_rng(seed)
    balanced, pooled = [], []
    for start in range(0, draws, 200):
        weights = rng.multinomial(
            reviews,
            np.full(reviews, 1 / reviews),
            size=min(200, draws - start),
        )
        totals = (weights @ matrix).reshape(-1, systems, folds, 3)
        balanced.append(f1_from_counts(totals).mean(axis=2))
        pooled.append(f1_from_counts(totals.sum(axis=2)))
    return np.concatenate(balanced), np.concatenate(pooled)
