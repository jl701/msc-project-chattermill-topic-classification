from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from msc_project.data.fabsa import format_pair_label
from msc_project.evaluation.metrics import evaluate_label_sets, evaluate_pair_and_aspect


SENTIMENTS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metrics: dict[str, float]
    sweep: pd.DataFrame


def candidate_pair_classes(aspect: str) -> list[str]:
    return [format_pair_label(aspect, sentiment) for sentiment in SENTIMENTS]


def score_matrix_from_grid(
    grid: pd.DataFrame,
    scores: Iterable[float],
    row_count: int,
) -> np.ndarray:
    """Align a scored evaluation grid to row x sentiment order.

    The grid must contain exactly one row for every ``row_index`` and frozen
    sentiment. Explicit validation here prevents silent metric corruption when
    a cached scorer output is resumed in the wrong order.
    """

    required = {"row_index", "candidate_sentiment"}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Evaluation grid is missing columns: {sorted(missing)}")

    values = np.asarray(list(scores), dtype=float)
    if len(values) != len(grid):
        raise ValueError("The score count does not match the evaluation grid.")
    if not np.isfinite(values).all():
        raise ValueError("Applicability scores must all be finite.")

    matrix = np.full((row_count, len(SENTIMENTS)), np.nan, dtype=float)
    sentiment_index = {sentiment: index for index, sentiment in enumerate(SENTIMENTS)}
    for position, row in enumerate(grid.itertuples(index=False)):
        row_index = int(row.row_index)
        sentiment = str(row.candidate_sentiment)
        if not 0 <= row_index < row_count:
            raise ValueError(f"Invalid row_index {row_index} for {row_count} rows.")
        if sentiment not in sentiment_index:
            raise ValueError(f"Unknown candidate sentiment: {sentiment}")
        column = sentiment_index[sentiment]
        if not np.isnan(matrix[row_index, column]):
            raise ValueError(f"Duplicate evaluation pair at row {row_index}, sentiment {sentiment}.")
        matrix[row_index, column] = values[position]

    if np.isnan(matrix).any():
        raise ValueError("Evaluation grid does not contain every row-sentiment pair.")
    return matrix


def prediction_sets_from_scores(
    scores: np.ndarray,
    aspect: str,
    threshold: float,
) -> list[list[str]]:
    if scores.ndim != 2 or scores.shape[1] != len(SENTIMENTS):
        raise ValueError(f"scores must have shape (rows, {len(SENTIMENTS)}).")
    classes = candidate_pair_classes(aspect)
    return [
        [classes[index] for index, value in enumerate(row) if float(value) >= threshold]
        for row in scores
    ]


def _presence_metrics(
    true_pairs: list[list[str]],
    pred_pairs: list[list[str]],
) -> dict[str, float]:
    true_present = np.asarray([bool(row) for row in true_pairs], dtype=bool)
    pred_present = np.asarray([bool(row) for row in pred_pairs], dtype=bool)
    tp = int(np.sum(true_present & pred_present))
    fp = int(np.sum(~true_present & pred_present))
    fn = int(np.sum(true_present & ~pred_present))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rows = len(true_pairs)
    return {
        "presence_precision": float(precision),
        "presence_recall": float(recall),
        "presence_f1": float(f1),
        "presence_tp_rows": tp,
        "presence_fp_rows": fp,
        "presence_fn_rows": fn,
        "presence_prevalence": float(np.mean(true_present)) if rows else 0.0,
        "presence_false_positive_rows_per_100": float(fp * 100 / rows) if rows else 0.0,
        "presence_false_negative_rows_per_100": float(fn * 100 / rows) if rows else 0.0,
    }


def _sentiments_from_pairs(pairs: list[str]) -> list[str]:
    return sorted({label.rsplit(" | ", maxsplit=1)[1] for label in pairs})


def _conditional_sentiment_metrics(
    true_pairs: list[list[str]],
    pred_pairs: list[list[str]],
) -> dict[str, float]:
    matched = [
        index
        for index, (true_row, pred_row) in enumerate(zip(true_pairs, pred_pairs))
        if true_row and pred_row
    ]
    gold_present = sum(bool(row) for row in true_pairs)
    if not matched:
        return {
            "conditional_sentiment_rows": 0,
            "conditional_sentiment_detection_coverage": 0.0,
            "conditional_sentiment_exact_match": 0.0,
            "conditional_sentiment_micro_f1": 0.0,
            "conditional_sentiment_macro_f1": 0.0,
        }

    true_sentiments = [_sentiments_from_pairs(true_pairs[index]) for index in matched]
    pred_sentiments = [_sentiments_from_pairs(pred_pairs[index]) for index in matched]
    scores = evaluate_label_sets(true_sentiments, pred_sentiments, list(SENTIMENTS))
    exact = float(np.mean([gold == pred for gold, pred in zip(true_sentiments, pred_sentiments)]))
    return {
        "conditional_sentiment_rows": int(len(matched)),
        "conditional_sentiment_detection_coverage": float(len(matched) / gold_present) if gold_present else 0.0,
        "conditional_sentiment_exact_match": exact,
        "conditional_sentiment_micro_f1": float(scores["micro_f1"]),
        "conditional_sentiment_macro_f1": float(scores["macro_f1"]),
    }


def evaluate_score_matrix(
    true_pairs: list[list[str]],
    scores: np.ndarray,
    aspect: str,
    threshold: float,
) -> tuple[dict[str, float], list[list[str]]]:
    predictions = prediction_sets_from_scores(scores, aspect, threshold)
    metrics = evaluate_pair_and_aspect(true_pairs, predictions, candidate_pair_classes(aspect))
    metrics.update(_presence_metrics(true_pairs, predictions))
    metrics.update(_conditional_sentiment_metrics(true_pairs, predictions))
    metrics["examples"] = int(len(true_pairs))
    metrics["threshold"] = float(threshold)
    return metrics, predictions


def threshold_candidates(scores: np.ndarray) -> list[float]:
    flat = np.unique(np.asarray(scores, dtype=float).ravel())
    flat = flat[np.isfinite(flat)]
    midpoints = (flat[:-1] + flat[1:]) / 2 if len(flat) > 1 else np.asarray([], dtype=float)
    regular = np.arange(0.01, 1.00, 0.01, dtype=float)
    values = np.concatenate([regular, midpoints])
    values = values[(values >= 0.0) & (values <= 1.0)]
    return sorted({round(float(value), 12) for value in values})


def select_threshold(
    true_pairs: list[list[str]],
    scores: np.ndarray,
    aspect: str,
) -> ThresholdSelection:
    if scores.ndim != 2 or scores.shape != (len(true_pairs), len(SENTIMENTS)):
        raise ValueError(
            f"scores must have shape ({len(true_pairs)}, {len(SENTIMENTS)})."
        )
    classes = candidate_pair_classes(aspect)
    class_index = {label: index for index, label in enumerate(classes)}
    gold = np.zeros(scores.shape, dtype=bool)
    for row_index, labels in enumerate(true_pairs):
        for label in labels:
            if label in class_index:
                gold[row_index, class_index[label]] = True

    gold_counts = gold.sum(axis=1)
    gold_present = gold_counts > 0
    rows: list[dict[str, float]] = []
    for threshold in threshold_candidates(scores):
        predicted = scores >= threshold
        tp = int(np.sum(gold & predicted))
        fp = int(np.sum(~gold & predicted))
        fn = int(np.sum(gold & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        pred_counts = predicted.sum(axis=1)
        row_tp = (gold & predicted).sum(axis=1)
        denominators = gold_counts + pred_counts
        samples_f1 = np.divide(
            2 * row_tp,
            denominators,
            out=np.zeros_like(denominators, dtype=float),
            where=denominators != 0,
        )
        pred_present = pred_counts > 0
        false_positive_rows = int(np.sum(~gold_present & pred_present))
        rows.append(
            {
                "threshold": float(threshold),
                "pair_micro_f1": float(micro_f1),
                "pair_micro_precision": float(precision),
                "pair_micro_recall": float(recall),
                "pair_samples_f1": float(samples_f1.mean()),
                "presence_false_positive_rows_per_100": float(
                    false_positive_rows * 100 / len(true_pairs)
                )
                if true_pairs
                else 0.0,
            }
        )

    if not rows:
        raise ValueError("No candidate thresholds were generated.")
    rows.sort(
        key=lambda row: (
            float(row["pair_micro_f1"]),
            float(row["pair_samples_f1"]),
            float(row["pair_micro_precision"]),
            -float(row["presence_false_positive_rows_per_100"]),
            float(row["threshold"]),
        ),
        reverse=True,
    )
    best_threshold = float(rows[0]["threshold"])
    best, _ = evaluate_score_matrix(true_pairs, scores, aspect, best_threshold)
    sweep = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return ThresholdSelection(threshold=best_threshold, metrics=best, sweep=sweep)


def paired_aspect_statistics(
    enhanced: Iterable[float],
    control: Iterable[float],
    seed: int = 13,
    bootstrap_samples: int = 20000,
) -> dict[str, float | int | list[float]]:
    enhanced_values = np.asarray(list(enhanced), dtype=float)
    control_values = np.asarray(list(control), dtype=float)
    if enhanced_values.shape != control_values.shape or enhanced_values.ndim != 1:
        raise ValueError("Enhanced and control scores must be aligned one-dimensional arrays.")
    if len(enhanced_values) == 0:
        raise ValueError("At least one paired score is required.")

    differences = enhanced_values - control_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    ci = np.quantile(bootstrap_means, [0.025, 0.975])

    observed = abs(float(differences.mean()))
    if len(differences) <= 20:
        signed_means = []
        for signs in product((-1.0, 1.0), repeat=len(differences)):
            signed_means.append(abs(float(np.mean(differences * np.asarray(signs)))))
        p_value = float(np.mean(np.asarray(signed_means) >= observed - 1e-15))
    else:
        sign_samples = rng.choice((-1.0, 1.0), size=(100000, len(differences)))
        random_means = np.abs((sign_samples * differences).mean(axis=1))
        p_value = float((np.sum(random_means >= observed) + 1) / (len(random_means) + 1))

    tolerance = 1e-12
    return {
        "pairs": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "median_delta": float(np.median(differences)),
        "bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        "exact_or_monte_carlo_sign_flip_p": p_value,
        "wins": int(np.sum(differences > tolerance)),
        "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "losses": int(np.sum(differences < -tolerance)),
    }


def pilot_gate(
    enhanced_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    minimum_mean_delta: float = 0.02,
    minimum_wins: int = 2,
    minimum_recall_fraction: float = 0.5,
) -> dict[str, object]:
    keys = ["heldout_aspect", "pair_micro_f1", "pair_micro_recall"]
    for frame, name in ((enhanced_rows, "enhanced"), (control_rows, "control")):
        missing = set(keys) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} rows are missing columns: {sorted(missing)}")

    merged = enhanced_rows[keys].merge(
        control_rows[keys],
        on="heldout_aspect",
        suffixes=("_enhanced", "_control"),
        validate="one_to_one",
    )
    if len(merged) != len(enhanced_rows) or len(merged) != len(control_rows):
        raise ValueError("Enhanced and control pilot folds are not aligned.")

    deltas = merged["pair_micro_f1_enhanced"] - merged["pair_micro_f1_control"]
    recall_floor = minimum_recall_fraction * merged["pair_micro_recall_control"]
    mean_delta = float(deltas.mean())
    wins = int((deltas > 0).sum())
    recall_ok = bool((merged["pair_micro_recall_enhanced"] >= recall_floor).all())
    passed = mean_delta >= minimum_mean_delta and wins >= minimum_wins and recall_ok
    return {
        "passed": bool(passed),
        "mean_pair_micro_f1_delta": mean_delta,
        "wins": wins,
        "required_mean_delta": float(minimum_mean_delta),
        "required_wins": int(minimum_wins),
        "minimum_recall_fraction": float(minimum_recall_fraction),
        "recall_floor_passed": recall_ok,
        "per_fold": merged.assign(pair_micro_f1_delta=deltas).to_dict(orient="records"),
    }
