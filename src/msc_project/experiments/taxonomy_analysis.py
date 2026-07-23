"""Protocol-specific result tables and Level 3 crossover diagnostics."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.experiments.taxonomy_protocol import (
    PAIR_KEY,
    TaxonomyFold,
    _prediction_sets,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS


def _validate_scored_grid(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *PAIR_KEY,
        "pair_label",
        "target",
        "score",
        "representation_variant",
        "is_seen",
        "is_heldout",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Scored taxonomy grid is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Scored taxonomy grid must not be empty.")
    if frame.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Scored taxonomy grid contains duplicate pair identities.")
    result = frame.copy()
    result["target"] = pd.to_numeric(result["target"], errors="raise").astype(int)
    result["score"] = pd.to_numeric(result["score"], errors="raise").astype(float)
    if not result["target"].isin([0, 1]).all():
        raise ValueError("Scored taxonomy targets must be binary.")
    if not np.isfinite(result["score"]).all():
        raise ValueError("Scored taxonomy probabilities must be finite.")
    if ((result["score"] < 0.0) | (result["score"] > 1.0)).any():
        raise ValueError("Scored taxonomy probabilities must lie in [0,1].")
    return result


def per_aspect_metrics(
    scored_grid: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    frame = _validate_scored_grid(scored_grid)
    rows: list[dict[str, object]] = []
    for aspect, group in frame.groupby("candidate_aspect", sort=True):
        _, gold, predicted, classes = _prediction_sets(
            group,
            threshold,
            allowed_aspects={str(aspect)},
        )
        metrics = evaluate_pair_and_aspect(gold, predicted, classes)
        rows.append(
            {
                "candidate_aspect": str(aspect),
                "is_seen": bool(group["is_seen"].iloc[0]),
                "is_heldout": bool(group["is_heldout"].iloc[0]),
                "representation_variant": str(
                    group["representation_variant"].iloc[0]
                ),
                "examples": int(group["row_uid"].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def per_sentiment_metrics(
    scored_grid: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    frame = _validate_scored_grid(scored_grid)
    rows: list[dict[str, object]] = []
    for sentiment in CANDIDATE_SENTIMENTS:
        group = frame[frame["candidate_sentiment"].astype(str) == sentiment].copy()
        if group.empty:
            raise ValueError(f"Scored grid is missing sentiment {sentiment!r}.")
        gold = group["target"].to_numpy(dtype=bool)
        predicted = group["score"].to_numpy(dtype=float) >= threshold
        tp = int(np.sum(gold & predicted))
        fp = int(np.sum(~gold & predicted))
        fn = int(np.sum(gold & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append(
            {
                "candidate_sentiment": sentiment,
                "pair_f1": float(f1),
                "pair_precision": float(precision),
                "pair_recall": float(recall),
                "pair_tp": tp,
                "pair_fp": fp,
                "pair_fn": fn,
                "candidate_pairs": int(len(group)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _aspect_presence_by_row(
    scored_grid: pd.DataFrame,
    aspect: str,
    threshold: float,
) -> pd.DataFrame:
    group = scored_grid[
        scored_grid["candidate_aspect"].astype(str) == aspect
    ].copy()
    if group.empty:
        raise ValueError(f"Scored grid is missing held-out aspect {aspect!r}.")
    rows = []
    for row_uid, row_group in group.groupby(
        group["row_uid"].astype(str), sort=True
    ):
        gold_pairs = row_group[row_group["target"] == 1]
        predicted_pairs = row_group[row_group["score"] >= threshold]
        rows.append(
            {
                "row_uid": str(row_uid),
                "gold_present": bool(len(gold_pairs)),
                "predicted_present": bool(len(predicted_pairs)),
                "gold_sentiments": tuple(
                    sorted(gold_pairs["candidate_sentiment"].astype(str))
                ),
                "predicted_sentiments": tuple(
                    sorted(predicted_pairs["candidate_sentiment"].astype(str))
                ),
            }
        )
    return pd.DataFrame.from_records(rows).set_index("row_uid")


def level3_condition_diagnostics(
    scored_grid: pd.DataFrame,
    fold: TaxonomyFold,
    condition: str,
    threshold: float,
) -> dict[str, object]:
    frame = _validate_scored_grid(scored_grid)
    if fold.level != "L3" or len(fold.heldout_aspects) != 2:
        raise ValueError("Level 3 diagnostics require a dual-heldout Level 3 fold.")
    if condition not in fold.conditions:
        raise ValueError(f"Condition {condition!r} is not registered for this fold.")
    aspect_a, aspect_b = fold.heldout_aspects
    a = _aspect_presence_by_row(frame, aspect_a, threshold).add_suffix("_a")
    b = _aspect_presence_by_row(frame, aspect_b, threshold).add_suffix("_b")
    aligned = a.join(b, how="inner", validate="one_to_one")
    expected_rows = set(frame["row_uid"].astype(str))
    if set(aligned.index) != expected_rows:
        raise ValueError("Held-out aspects do not share the complete review set.")

    both_gold = aligned["gold_present_a"] & aligned["gold_present_b"]
    neither_gold = ~aligned["gold_present_a"] & ~aligned["gold_present_b"]
    both_predicted = (
        aligned["predicted_present_a"] & aligned["predicted_present_b"]
    )
    any_predicted = (
        aligned["predicted_present_a"] | aligned["predicted_present_b"]
    )

    represented = {
        str(row.candidate_aspect): str(row.representation_variant)
        for row in frame[
            frame["candidate_aspect"].isin(fold.heldout_aspects)
        ]
        .drop_duplicates("candidate_aspect")
        .itertuples(index=False)
    }
    described = [
        aspect
        for aspect in fold.heldout_aspects
        if represented.get(aspect) == "minimal"
    ]
    undescribed = [
        aspect
        for aspect in fold.heldout_aspects
        if represented.get(aspect) == "name_only"
    ]

    def safe_rate(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    diagnostics: dict[str, object] = {
        "fold_id": fold.fold_id,
        "condition": condition,
        "threshold": float(threshold),
        "examples": int(len(aligned)),
        "heldout_aspect_a": aspect_a,
        "heldout_aspect_b": aspect_b,
        "both_gold_rows": int(both_gold.sum()),
        "both_present_recall": safe_rate(
            int((both_gold & both_predicted).sum()),
            int(both_gold.sum()),
        ),
        "neither_gold_rows": int(neither_gold.sum()),
        "neither_false_positive_rate": safe_rate(
            int((neither_gold & any_predicted).sum()),
            int(neither_gold.sum()),
        ),
        "described_aspects": described,
        "name_only_aspects": undescribed,
    }
    if len(described) == 1 and len(undescribed) == 1:
        described_aspect = described[0]
        undescribed_aspect = undescribed[0]
        described_suffix = "a" if described_aspect == aspect_a else "b"
        undescribed_suffix = "a" if undescribed_aspect == aspect_a else "b"
        only_undescribed = (
            aligned[f"gold_present_{undescribed_suffix}"]
            & ~aligned[f"gold_present_{described_suffix}"]
        )
        described_selected = aligned[f"predicted_present_{described_suffix}"]
        undescribed_selected = aligned[f"predicted_present_{undescribed_suffix}"]
        diagnostics.update(
            {
                "only_name_only_gold_rows": int(only_undescribed.sum()),
                "name_only_recall_when_only_name_only_is_gold": safe_rate(
                    int((only_undescribed & undescribed_selected).sum()),
                    int(only_undescribed.sum()),
                ),
                "described_false_positive_rate_when_only_name_only_is_gold": safe_rate(
                    int((only_undescribed & described_selected).sum()),
                    int(only_undescribed.sum()),
                ),
                "described_candidate_selection_rate": float(
                    described_selected.mean()
                ),
                "name_only_candidate_selection_rate": float(
                    undescribed_selected.mean()
                ),
                "described_minus_name_only_selection_rate": float(
                    described_selected.mean() - undescribed_selected.mean()
                ),
                "described_minus_name_only_selection_rate_when_neither_gold": (
                    float(
                        described_selected[neither_gold].mean()
                        - undescribed_selected[neither_gold].mean()
                    )
                    if neither_gold.any()
                    else None
                ),
            }
        )
    else:
        diagnostics.update(
            {
                "only_name_only_gold_rows": None,
                "name_only_recall_when_only_name_only_is_gold": None,
                "described_false_positive_rate_when_only_name_only_is_gold": None,
                "described_candidate_selection_rate": None,
                "name_only_candidate_selection_rate": None,
                "described_minus_name_only_selection_rate": None,
                "described_minus_name_only_selection_rate_when_neither_gold": None,
            }
        )
    diagnostics["heldout_per_aspect"] = per_aspect_metrics(
        frame[frame["candidate_aspect"].isin(fold.heldout_aspects)],
        threshold,
    ).to_dict(orient="records")
    return diagnostics


def assert_shared_seen_threshold(
    selections: Mapping[str, object],
    *,
    tolerance: float = 1e-12,
) -> float:
    if not selections:
        raise ValueError("At least one threshold selection is required.")
    thresholds = {}
    calibration_sets = {}
    for condition, selection in selections.items():
        threshold = getattr(selection, "threshold", None)
        calibration_aspects = getattr(selection, "calibration_aspects", None)
        if threshold is None or calibration_aspects is None:
            raise ValueError(f"Condition {condition!r} lacks strict selection evidence.")
        thresholds[str(condition)] = float(threshold)
        calibration_sets[str(condition)] = tuple(calibration_aspects)
    first = next(iter(thresholds.values()))
    if any(abs(value - first) > tolerance for value in thresholds.values()):
        raise ValueError(f"Condition-specific strict thresholds differ: {thresholds}")
    if len(set(calibration_sets.values())) != 1:
        raise ValueError(
            f"Condition-specific calibration aspect sets differ: {calibration_sets}"
        )
    return first
