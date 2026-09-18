"""Outcome-blind row-level evidence for the taxonomy scientific freeze."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def row_pair_confusions(
    scored_grid: pd.DataFrame,
    prediction_mask: Sequence[bool] | pd.Series,
    *,
    aspects: Iterable[str],
    method_id: str,
    fold_id: str,
    condition: str,
    level: str = "L2",
) -> pd.DataFrame:
    """Return one paired TP/FP/FN sufficient-statistics row per review.

    The function deliberately stores counts rather than review text or labels.  This
    is sufficient for an exact review-cluster bootstrap of pair micro-F1 while
    keeping the formal test partition sealed.
    """

    required = {
        "row_uid",
        "candidate_aspect",
        "candidate_sentiment",
        "pair_label",
        "target",
    }
    missing = sorted(required - set(scored_grid.columns))
    if missing:
        raise ValueError(f"Score grid is missing columns: {missing}")
    if scored_grid.empty:
        raise ValueError("Score grid must not be empty.")
    if scored_grid.duplicated(
        ["row_uid", "candidate_aspect", "candidate_sentiment"]
    ).any():
        raise ValueError("Score grid contains duplicate candidate identities.")

    allowed = {str(value) for value in aspects}
    frame = scored_grid[
        scored_grid["candidate_aspect"].astype(str).isin(allowed)
    ].copy()
    if frame.empty:
        raise ValueError("Requested aspect partition is empty.")
    mask = pd.Series(prediction_mask, index=scored_grid.index, dtype=bool).loc[
        frame.index
    ]
    frame["_gold"] = frame["target"].astype(int).eq(1)
    frame["_pred"] = mask.to_numpy(dtype=bool)
    frame["_tp"] = frame["_gold"] & frame["_pred"]
    frame["_fp"] = ~frame["_gold"] & frame["_pred"]
    frame["_fn"] = frame["_gold"] & ~frame["_pred"]
    grouped = (
        frame.assign(row_uid=frame["row_uid"].astype(str))
        .groupby("row_uid", sort=True)[["_tp", "_fp", "_fn", "_gold", "_pred"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "_tp": "pair_tp",
                "_fp": "pair_fp",
                "_fn": "pair_fn",
                "_gold": "pair_gold_count",
                "_pred": "pair_predicted_count",
            }
        )
    )
    for column in (
        "pair_tp",
        "pair_fp",
        "pair_fn",
        "pair_gold_count",
        "pair_predicted_count",
    ):
        grouped[column] = grouped[column].astype(np.int64)
    grouped.insert(0, "level", str(level))
    grouped.insert(1, "method_id", str(method_id))
    grouped.insert(2, "fold_id", str(fold_id))
    grouped.insert(3, "condition", str(condition))
    return grouped


def pair_micro_f1_from_counts(
    true_positive: float | np.ndarray,
    false_positive: float | np.ndarray,
    false_negative: float | np.ndarray,
) -> float | np.ndarray:
    """Compute pair micro-F1 from additive sufficient statistics."""

    denominator = 2.0 * true_positive + false_positive + false_negative
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(np.asarray(denominator), dtype=float),
        where=np.asarray(denominator) != 0,
    )
