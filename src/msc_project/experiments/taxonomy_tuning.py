"""Finite, preregistered seen-validation hyperparameter selection."""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_methods import (
    load_method_registry,
    resolve_method_spec,
)


TUNING_METRICS = (
    "pair_micro_f1",
    "pair_samples_f1",
    "pair_micro_precision",
    "presence_false_positive_rows_per_100",
)


def registered_tuning_candidates(
    method_id: str,
    registry: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    source = dict(registry or load_method_registry())
    spec = resolve_method_spec(method_id, source)
    starting = dict(spec.starting_recipe)
    tuning = spec.seen_only_tuning
    if method_id == "strict_train_only_tfidf":
        grid = [
            {
                **starting,
                "classifier_c": float(classifier_c),
                "feature_ablation": str(ablation),
            }
            for classifier_c, ablation in product(
                tuning["classifier_c"],
                tuning["feature_ablation"],
            )
        ]
    elif method_id == "distilbert_review_candidate_cross_encoder":
        grid = [
            {
                **starting,
                "learning_rate": float(learning_rate),
                "selected_checkpoint_epoch": int(epoch),
            }
            for learning_rate, epoch in product(
                tuning["learning_rate"],
                tuning["checkpoint_epochs"],
            )
        ]
    elif method_id == "qwen_candidate_pair_qlora":
        grid = [
            {
                **starting,
                "learning_rate": float(learning_rate),
                "selected_checkpoint_epoch": int(epoch),
            }
            for learning_rate, epoch in product(
                tuning["learning_rate"],
                tuning["checkpoint_epochs"],
            )
        ]
    else:
        grid = [starting]
    unique = {
        canonical_sha256(parameters): parameters
        for parameters in grid
    }
    if len(unique) != len(grid):
        raise AssertionError("Registered tuning grid contains duplicate parameter sets.")
    return [
        {
            "candidate_id": f"{method_id}-{index:03d}",
            "parameters_sha256": digest,
            "parameters": parameters,
        }
        for index, (digest, parameters) in enumerate(unique.items(), start=1)
    ]


def expected_tuning_folds(
    method_id: str,
    registry: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    spec = resolve_method_spec(method_id, registry or load_method_registry())
    folds = spec.seen_only_tuning.get("pilot_outer_folds")
    if folds is None:
        return ()
    values = tuple(str(value) for value in folds)
    if not values or len(values) != len(set(values)):
        raise ValueError("Registered pilot_outer_folds must be non-empty and unique.")
    return values


def select_registered_tuning_candidate(
    observations: pd.DataFrame,
    method_id: str,
    registry: Mapping[str, object] | None = None,
    *,
    required_folds: Sequence[str] | None = None,
) -> dict[str, object]:
    source = dict(registry or load_method_registry())
    candidates = registered_tuning_candidates(method_id, source)
    candidate_by_id = {
        str(value["candidate_id"]): value
        for value in candidates
    }
    configured_folds = expected_tuning_folds(method_id, source)
    if configured_folds and required_folds is not None:
        raise ValueError("Configured and explicit tuning folds cannot both be used.")
    expected_folds = tuple(str(value) for value in (required_folds or configured_folds))
    if not expected_folds:
        fold_sets = {
            tuple(sorted(group["fold_id"].astype(str).unique()))
            for _, group in observations.groupby("candidate_id", sort=False)
        }
        if len(fold_sets) != 1:
            raise ValueError("Registered candidates have different tuning-fold coverage.")
        expected_folds = next(iter(fold_sets))
    if not expected_folds or len(expected_folds) != len(set(expected_folds)):
        raise ValueError("Tuning selection requires non-empty unique fold coverage.")
    required = {
        "method_id",
        "candidate_id",
        "parameters_sha256",
        "fold_id",
        "calibration_scope",
        *TUNING_METRICS,
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"Tuning observations are missing columns: {missing}")
    frame = observations.copy()
    if set(frame["method_id"].astype(str)) != {method_id}:
        raise ValueError("Tuning observations mix method IDs.")
    if set(frame["candidate_id"].astype(str)) != set(candidate_by_id):
        raise ValueError("Tuning observations do not cover the exact registered candidates.")
    if frame.duplicated(["candidate_id", "fold_id"]).any():
        raise ValueError("Tuning observations contain duplicate candidate-fold results.")
    if set(frame["calibration_scope"].astype(str)) != {"seen_aspects_only"}:
        raise ValueError("Tuning observations include non-strict calibration evidence.")
    for candidate_id, group in frame.groupby("candidate_id", sort=False):
        expected_hash = candidate_by_id[str(candidate_id)]["parameters_sha256"]
        if set(group["parameters_sha256"].astype(str)) != {expected_hash}:
            raise ValueError(f"Parameter hash mismatch for {candidate_id}.")
        if set(group["fold_id"].astype(str)) != set(expected_folds):
            raise ValueError(f"Pilot-fold coverage mismatch for {candidate_id}.")
        for metric in TUNING_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite tuning metric {metric} for {candidate_id}.")

    aggregate = (
        frame.groupby("candidate_id", as_index=False)[list(TUNING_METRICS)]
        .mean()
        .rename(columns={metric: f"mean_{metric}" for metric in TUNING_METRICS})
    )
    aggregate["parameters_sha256"] = aggregate["candidate_id"].map(
        lambda value: candidate_by_id[str(value)]["parameters_sha256"]
    )

    def complexity_key(candidate_id: str) -> tuple[float, float]:
        parameters = candidate_by_id[candidate_id]["parameters"]
        if method_id == "strict_train_only_tfidf":
            ablation_order = {
                "char_cosine_and_cues": 0.0,
                "word_cosine_and_cues": 1.0,
                "all_six": 2.0,
            }
            return (
                ablation_order[str(parameters["feature_ablation"])],
                float(parameters["classifier_c"]),
            )
        epoch = float(parameters.get("selected_checkpoint_epoch", 0))
        return (epoch, float(parameters.get("learning_rate", 0.0)))

    ranked = sorted(
        aggregate.to_dict(orient="records"),
        key=lambda row: (
            -float(row["mean_pair_micro_f1"]),
            -float(row["mean_pair_samples_f1"]),
            -float(row["mean_pair_micro_precision"]),
            float(row["mean_presence_false_positive_rows_per_100"]),
            complexity_key(str(row["candidate_id"])),
            str(row["candidate_id"]),
        ),
    )
    selected = ranked[0]
    selected_candidate = candidate_by_id[str(selected["candidate_id"])]
    return {
        "method_id": method_id,
        "selection_scope": "registered_seen_validation_grid",
        "selection_folds": list(expected_folds),
        "registered_candidates": len(candidates),
        "selected_candidate_id": selected["candidate_id"],
        "selected_parameters_sha256": selected_candidate["parameters_sha256"],
        "selected_parameters": selected_candidate["parameters"],
        "selected_mean_metrics": {
            metric.removeprefix("mean_"): float(value)
            for metric, value in selected.items()
            if metric.startswith("mean_")
        },
        "ranked_candidates": ranked,
        "stopping_rule": resolve_method_spec(
            method_id, source
        ).seen_only_tuning["stopping_rule"],
        "adaptive_grid_expansion_permitted": False,
    }


def fixed_registered_selection(
    method_id: str,
    registry: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Freeze a method whose registry intentionally exposes no tuning grid."""

    source = dict(registry or load_method_registry())
    candidates = registered_tuning_candidates(method_id, source)
    folds = expected_tuning_folds(method_id, source)
    if len(candidates) != 1 or folds:
        raise ValueError(
            "Fixed selection is only valid for a single-candidate method "
            "without registered pilot folds."
        )
    selected = candidates[0]
    return {
        "method_id": method_id,
        "selection_scope": "registered_fixed_recipe",
        "training_scope_id": "global_fixed_recipe",
        "selection_folds": [],
        "registered_candidates": 1,
        "selected_candidate_id": selected["candidate_id"],
        "selected_parameters_sha256": selected["parameters_sha256"],
        "selected_parameters": selected["parameters"],
        "selected_mean_metrics": None,
        "ranked_candidates": None,
        "stopping_rule": resolve_method_spec(
            method_id, source
        ).seen_only_tuning["stopping_rule"],
        "adaptive_grid_expansion_permitted": False,
    }
