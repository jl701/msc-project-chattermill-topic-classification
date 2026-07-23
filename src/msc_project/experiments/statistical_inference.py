"""Paired, cluster-aware uncertainty utilities for taxonomy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SUPPORTED_METRICS = (
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
    "pair_samples_f1",
    "pair_exact_match_rate",
    "presence_f1",
    "presence_precision",
    "presence_recall",
    "pair_false_positive_rows_per_100",
    "pair_false_negative_rows_per_100",
)


def _normalise_label_set(value: object, *, column: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{column} must contain a list-like label set per row.")
    labels = [str(item) for item in value]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{column} contains duplicate labels within a row.")
    return tuple(sorted(labels))


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    gold_column: str,
    prediction_column: str,
    observation_key_columns: Sequence[str],
) -> pd.DataFrame:
    required = {
        cluster_column,
        gold_column,
        prediction_column,
        *observation_key_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction frame must not be empty.")

    result = frame.copy()
    for column in (cluster_column, *observation_key_columns):
        if result[column].isna().any() or (result[column].astype(str).str.len() == 0).any():
            raise ValueError(f"{column} values must be non-empty.")
        result[column] = result[column].astype(str)
    if result.duplicated(list(observation_key_columns)).any():
        raise ValueError("Prediction frame contains duplicate observation keys.")
    result[gold_column] = [
        _normalise_label_set(value, column=gold_column)
        for value in result[gold_column]
    ]
    result[prediction_column] = [
        _normalise_label_set(value, column=prediction_column)
        for value in result[prediction_column]
    ]
    return result.sort_values(list(observation_key_columns), kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class ClusterSufficientStatistics:
    cluster_ids: tuple[str, ...]
    pair_tp: np.ndarray
    pair_fp: np.ndarray
    pair_fn: np.ndarray
    pair_class_tp: np.ndarray
    pair_class_fp: np.ndarray
    pair_class_fn: np.ndarray
    sample_f1_sum: np.ndarray
    exact_match_rows: np.ndarray
    presence_tp: np.ndarray
    presence_fp: np.ndarray
    presence_fn: np.ndarray
    false_positive_rows: np.ndarray
    false_negative_rows: np.ndarray
    rows: np.ndarray

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_ids)


def build_cluster_sufficient_statistics(
    frame: pd.DataFrame,
    pair_classes: Sequence[str],
    *,
    cluster_column: str = "row_uid",
    gold_column: str = "gold_pairs",
    prediction_column: str = "pred_pairs",
    observation_key_columns: Sequence[str] = ("row_uid",),
) -> ClusterSufficientStatistics:
    classes = tuple(str(value) for value in pair_classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("pair_classes must be non-empty and unique.")
    class_index = {label: index for index, label in enumerate(classes)}
    validated = _validate_prediction_frame(
        frame,
        cluster_column=cluster_column,
        gold_column=gold_column,
        prediction_column=prediction_column,
        observation_key_columns=observation_key_columns,
    )

    unknown_gold = sorted(
        {
            label
            for labels in validated[gold_column]
            for label in labels
            if label not in class_index
        }
    )
    unknown_pred = sorted(
        {
            label
            for labels in validated[prediction_column]
            for label in labels
            if label not in class_index
        }
    )
    if unknown_gold or unknown_pred:
        raise ValueError(
            f"Prediction frame contains labels outside pair_classes: "
            f"gold={unknown_gold}, pred={unknown_pred}."
        )

    row_count = len(validated)
    class_count = len(classes)
    gold = np.zeros((row_count, class_count), dtype=np.int64)
    predicted = np.zeros((row_count, class_count), dtype=np.int64)
    for row_index, labels in enumerate(validated[gold_column]):
        for label in labels:
            gold[row_index, class_index[label]] = 1
    for row_index, labels in enumerate(validated[prediction_column]):
        for label in labels:
            predicted[row_index, class_index[label]] = 1

    cluster_codes, unique_clusters = pd.factorize(
        validated[cluster_column],
        sort=True,
    )
    clusters = len(unique_clusters)
    gold_counts = gold.sum(axis=1)
    predicted_counts = predicted.sum(axis=1)
    row_tp = (gold & predicted).sum(axis=1)
    row_fp = ((1 - gold) & predicted).sum(axis=1)
    row_fn = (gold & (1 - predicted)).sum(axis=1)
    sample_denominators = gold_counts + predicted_counts
    sample_f1 = np.divide(
        2 * row_tp,
        sample_denominators,
        out=np.zeros(row_count, dtype=float),
        where=sample_denominators != 0,
    )
    gold_present = gold_counts > 0
    predicted_present = predicted_counts > 0

    def aggregate_1d(values: np.ndarray, dtype: object = np.int64) -> np.ndarray:
        result = np.zeros(clusters, dtype=dtype)
        np.add.at(result, cluster_codes, values)
        return result

    def aggregate_2d(values: np.ndarray) -> np.ndarray:
        result = np.zeros((clusters, class_count), dtype=np.int64)
        np.add.at(result, cluster_codes, values)
        return result

    return ClusterSufficientStatistics(
        cluster_ids=tuple(str(value) for value in unique_clusters.tolist()),
        pair_tp=aggregate_1d(row_tp),
        pair_fp=aggregate_1d(row_fp),
        pair_fn=aggregate_1d(row_fn),
        pair_class_tp=aggregate_2d(gold & predicted),
        pair_class_fp=aggregate_2d((1 - gold) & predicted),
        pair_class_fn=aggregate_2d(gold & (1 - predicted)),
        sample_f1_sum=aggregate_1d(sample_f1, dtype=float),
        exact_match_rows=aggregate_1d((gold == predicted).all(axis=1).astype(np.int64)),
        presence_tp=aggregate_1d((gold_present & predicted_present).astype(np.int64)),
        presence_fp=aggregate_1d((~gold_present & predicted_present).astype(np.int64)),
        presence_fn=aggregate_1d((gold_present & ~predicted_present).astype(np.int64)),
        false_positive_rows=aggregate_1d((~gold_present & predicted_present).astype(np.int64)),
        false_negative_rows=aggregate_1d((gold_present & ~predicted_present).astype(np.int64)),
        rows=aggregate_1d(np.ones(row_count, dtype=np.int64)),
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def metric_from_cluster_weights(
    statistics: ClusterSufficientStatistics,
    weights: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported bootstrap metric: {metric!r}")
    values = np.asarray(weights, dtype=np.int64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != statistics.cluster_count:
        raise ValueError("weights must have shape [replicates, cluster_count].")
    if (values < 0).any():
        raise ValueError("Cluster weights must be non-negative.")

    pair_tp = values @ statistics.pair_tp
    pair_fp = values @ statistics.pair_fp
    pair_fn = values @ statistics.pair_fn
    rows = values @ statistics.rows
    precision = _safe_ratio(pair_tp, pair_tp + pair_fp)
    recall = _safe_ratio(pair_tp, pair_tp + pair_fn)

    if metric == "pair_micro_precision":
        return precision
    if metric == "pair_micro_recall":
        return recall
    if metric == "pair_micro_f1":
        return _safe_ratio(2 * pair_tp, 2 * pair_tp + pair_fp + pair_fn)
    if metric == "pair_samples_f1":
        return _safe_ratio(values @ statistics.sample_f1_sum, rows)
    if metric == "pair_exact_match_rate":
        return _safe_ratio(values @ statistics.exact_match_rows, rows)
    if metric == "pair_false_positive_rows_per_100":
        return 100 * _safe_ratio(values @ statistics.false_positive_rows, rows)
    if metric == "pair_false_negative_rows_per_100":
        return 100 * _safe_ratio(values @ statistics.false_negative_rows, rows)

    if metric == "pair_macro_f1":
        class_tp = values @ statistics.pair_class_tp
        class_fp = values @ statistics.pair_class_fp
        class_fn = values @ statistics.pair_class_fn
        class_f1 = _safe_ratio(2 * class_tp, 2 * class_tp + class_fp + class_fn)
        return class_f1.mean(axis=1)

    presence_tp = values @ statistics.presence_tp
    presence_fp = values @ statistics.presence_fp
    presence_fn = values @ statistics.presence_fn
    if metric == "presence_precision":
        return _safe_ratio(presence_tp, presence_tp + presence_fp)
    if metric == "presence_recall":
        return _safe_ratio(presence_tp, presence_tp + presence_fn)
    return _safe_ratio(
        2 * presence_tp,
        2 * presence_tp + presence_fp + presence_fn,
    )


def _bootstrap_weight_batches(
    cluster_count: int,
    replicates: int,
    seed: int,
    *,
    batch_size: int = 512,
) -> Iterable[np.ndarray]:
    if cluster_count < 1:
        raise ValueError("At least one cluster is required.")
    if replicates < 1:
        raise ValueError("replicates must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        sampled = rng.integers(
            0,
            cluster_count,
            size=(size, cluster_count),
        )
        weights = np.zeros((size, cluster_count), dtype=np.int64)
        row_indices = np.repeat(np.arange(size), cluster_count)
        np.add.at(weights, (row_indices, sampled.ravel()), 1)
        yield weights


def cluster_bootstrap_interval(
    statistics: ClusterSufficientStatistics,
    metric: str,
    *,
    replicates: int = 20_000,
    seed: int = 13,
) -> dict[str, object]:
    point_weights = np.ones(statistics.cluster_count, dtype=np.int64)
    point = float(metric_from_cluster_weights(statistics, point_weights, metric)[0])
    samples = np.concatenate(
        [
            metric_from_cluster_weights(statistics, weights, metric)
            for weights in _bootstrap_weight_batches(
                statistics.cluster_count,
                replicates,
                seed,
            )
        ]
    )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "metric": metric,
        "point_estimate": point,
        "confidence_level": 0.95,
        "interval_method": "paired_cluster_percentile_bootstrap",
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "clusters": statistics.cluster_count,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def align_paired_prediction_frames(
    challenger: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    observation_key_columns: Sequence[str] = ("row_uid",),
    cluster_column: str = "row_uid",
    gold_column: str = "gold_pairs",
    prediction_column: str = "pred_pairs",
) -> pd.DataFrame:
    challenger_valid = _validate_prediction_frame(
        challenger,
        cluster_column=cluster_column,
        gold_column=gold_column,
        prediction_column=prediction_column,
        observation_key_columns=observation_key_columns,
    )
    reference_valid = _validate_prediction_frame(
        reference,
        cluster_column=cluster_column,
        gold_column=gold_column,
        prediction_column=prediction_column,
        observation_key_columns=observation_key_columns,
    )
    left_keys = challenger_valid[list(observation_key_columns)]
    right_keys = reference_valid[list(observation_key_columns)]
    if not left_keys.equals(right_keys):
        raise ValueError("Matched prediction frames have different observation keys or order.")
    if challenger_valid[cluster_column].tolist() != reference_valid[cluster_column].tolist():
        raise ValueError("Matched prediction frames have different cluster assignments.")
    if challenger_valid[gold_column].tolist() != reference_valid[gold_column].tolist():
        raise ValueError("Matched prediction frames have different gold labels.")

    aligned = challenger_valid[
        [*observation_key_columns, cluster_column, gold_column]
    ].copy()
    # Avoid duplicate insertion when cluster_column is already an observation key.
    aligned = aligned.loc[:, ~aligned.columns.duplicated()].copy()
    aligned["challenger_pairs"] = challenger_valid[prediction_column]
    aligned["reference_pairs"] = reference_valid[prediction_column]
    return aligned


def paired_cluster_bootstrap_difference(
    challenger: pd.DataFrame,
    reference: pd.DataFrame,
    pair_classes: Sequence[str],
    metric: str,
    *,
    observation_key_columns: Sequence[str] = ("row_uid",),
    cluster_column: str = "row_uid",
    gold_column: str = "gold_pairs",
    prediction_column: str = "pred_pairs",
    replicates: int = 20_000,
    seed: int = 13,
) -> dict[str, object]:
    aligned = align_paired_prediction_frames(
        challenger,
        reference,
        observation_key_columns=observation_key_columns,
        cluster_column=cluster_column,
        gold_column=gold_column,
        prediction_column=prediction_column,
    )
    shared_columns = [*observation_key_columns, cluster_column, gold_column]
    shared_columns = list(dict.fromkeys(shared_columns))
    challenger_frame = aligned[shared_columns].copy()
    challenger_frame["pred_pairs"] = aligned["challenger_pairs"]
    reference_frame = aligned[shared_columns].copy()
    reference_frame["pred_pairs"] = aligned["reference_pairs"]
    challenger_stats = build_cluster_sufficient_statistics(
        challenger_frame,
        pair_classes,
        cluster_column=cluster_column,
        gold_column=gold_column,
        prediction_column="pred_pairs",
        observation_key_columns=observation_key_columns,
    )
    reference_stats = build_cluster_sufficient_statistics(
        reference_frame,
        pair_classes,
        cluster_column=cluster_column,
        gold_column=gold_column,
        prediction_column="pred_pairs",
        observation_key_columns=observation_key_columns,
    )
    if challenger_stats.cluster_ids != reference_stats.cluster_ids:
        raise AssertionError("Aligned paired statistics produced different clusters.")

    point_weights = np.ones(challenger_stats.cluster_count, dtype=np.int64)
    challenger_point = float(
        metric_from_cluster_weights(challenger_stats, point_weights, metric)[0]
    )
    reference_point = float(
        metric_from_cluster_weights(reference_stats, point_weights, metric)[0]
    )
    differences = []
    for weights in _bootstrap_weight_batches(
        challenger_stats.cluster_count,
        replicates,
        seed,
    ):
        differences.append(
            metric_from_cluster_weights(challenger_stats, weights, metric)
            - metric_from_cluster_weights(reference_stats, weights, metric)
        )
    bootstrap_differences = np.concatenate(differences)
    lower, upper = np.quantile(bootstrap_differences, [0.025, 0.975])
    return {
        "metric": metric,
        "challenger_point": challenger_point,
        "reference_point": reference_point,
        "point_difference": challenger_point - reference_point,
        "confidence_level": 0.95,
        "interval_method": "paired_cluster_percentile_bootstrap",
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "clusters": challenger_stats.cluster_count,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def paired_unit_statistics(
    challenger: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    unit_column: str,
    score_column: str,
    bootstrap_replicates: int = 20_000,
    seed: int = 13,
) -> dict[str, object]:
    required = {unit_column, score_column}
    for frame, name in ((challenger, "challenger"), (reference, "reference")):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} unit frame is missing columns: {missing}")
        if frame[unit_column].astype(str).duplicated().any():
            raise ValueError(f"{name} unit frame contains duplicate units.")
    merged = challenger[[unit_column, score_column]].merge(
        reference[[unit_column, score_column]],
        on=unit_column,
        suffixes=("_challenger", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(challenger) or len(merged) != len(reference):
        raise ValueError("Paired unit frames do not contain identical units.")
    merged = merged.sort_values(unit_column, kind="stable")
    differences = (
        merged[f"{score_column}_challenger"].to_numpy(dtype=float)
        - merged[f"{score_column}_reference"].to_numpy(dtype=float)
    )
    if not np.isfinite(differences).all():
        raise ValueError("Paired unit differences must be finite.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(bootstrap_replicates, len(differences)),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    observed = abs(float(differences.mean()))
    if len(differences) <= 20:
        null_values = [
            abs(float(np.mean(differences * np.asarray(signs))))
            for signs in product((-1.0, 1.0), repeat=len(differences))
        ]
        sign_flip_p = float(np.mean(np.asarray(null_values) >= observed - 1e-15))
        sign_flip_method = "exact"
    else:
        sign_samples = rng.choice((-1.0, 1.0), size=(100_000, len(differences)))
        null_values = np.abs((sign_samples * differences).mean(axis=1))
        sign_flip_p = float(
            (np.sum(null_values >= observed - 1e-15) + 1)
            / (len(null_values) + 1)
        )
        sign_flip_method = "monte_carlo_100000"

    tolerance = 1e-12
    return {
        "units": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
        "sign_flip_method": sign_flip_method,
        "sign_flip_p_value_two_sided": sign_flip_p,
        "wins": int(np.sum(differences > tolerance)),
        "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "losses": int(np.sum(differences < -tolerance)),
        "per_unit": [
            {
                "unit": str(unit),
                "difference": float(difference),
            }
            for unit, difference in zip(merged[unit_column], differences)
        ],
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite one-dimensional sequence.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must lie in [0, 1].")
    if len(values) == 0:
        return []

    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(values) - rank) * values[index])
        running = max(running, adjusted)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]
