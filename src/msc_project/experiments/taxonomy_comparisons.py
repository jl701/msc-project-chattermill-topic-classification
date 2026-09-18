"""Registered matched and cross-protocol comparisons for final thesis evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from msc_project.experiments.statistical_inference import (
    holm_adjust,
    paired_cluster_bootstrap_difference,
    paired_unit_statistics,
    shared_cluster_bootstrap_difference,
)
from msc_project.experiments.taxonomy_analysis import prediction_frame_for_aspects
from msc_project.experiments.taxonomy_protocol import (
    TaxonomyFold,
    canonical_aspects,
    evaluate_scored_grid,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS


@dataclass(frozen=True)
class ProtocolResult:
    unit_id: str
    fold: TaxonomyFold
    condition: str
    scored_grid: pd.DataFrame
    threshold: float


def canonical_pair_classes() -> tuple[str, ...]:
    return tuple(
        f"{aspect} | {sentiment}"
        for aspect in canonical_aspects()
        for sentiment in CANDIDATE_SENTIMENTS
    )


def _partition_aspects(result: ProtocolResult, partition: str) -> set[str]:
    available = set(result.scored_grid["candidate_aspect"].astype(str))
    if partition == "overall":
        return available
    if partition == "seen":
        return available & set(result.fold.seen_aspects)
    if partition == "unseen":
        return available & set(result.fold.heldout_aspects)
    raise ValueError("partition must be overall, seen, or unseen.")


def aggregate_protocol_predictions(
    results: Sequence[ProtocolResult],
    *,
    partition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        raise ValueError("At least one protocol result is required.")
    unit_ids = [value.unit_id for value in results]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Protocol result unit IDs must be unique.")
    prediction_frames = []
    unit_metrics = []
    for result in results:
        aspects = _partition_aspects(result, partition)
        if not aspects:
            raise ValueError(
                f"Partition {partition!r} is empty for {result.unit_id!r}."
            )
        frame, _ = prediction_frame_for_aspects(
            result.scored_grid,
            result.threshold,
            aspects,
        )
        frame.insert(0, "experiment_unit", result.unit_id)
        prediction_frames.append(frame)
        metrics = evaluate_scored_grid(
            result.scored_grid,
            result.threshold,
            seen_aspects=result.fold.seen_aspects,
            heldout_aspects=result.fold.heldout_aspects,
        )[partition]
        if not isinstance(metrics, dict):
            raise ValueError(
                f"Partition {partition!r} has no metrics for {result.unit_id!r}."
            )
        unit_metrics.append(
            {
                "experiment_unit": result.unit_id,
                "pair_micro_f1": float(metrics["pair_micro_f1"]),
            }
        )
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame.from_records(unit_metrics),
    )


def matched_protocol_comparison(
    challenger: Sequence[ProtocolResult],
    reference: Sequence[ProtocolResult],
    *,
    partition: str = "overall",
    metric: str = "pair_micro_f1",
    replicates: int = 20_000,
    seed: int = 13,
) -> dict[str, object]:
    """Compare systems/conditions with identical experiment units and gold sets."""

    challenger_frame, challenger_units = aggregate_protocol_predictions(
        challenger,
        partition=partition,
    )
    reference_frame, reference_units = aggregate_protocol_predictions(
        reference,
        partition=partition,
    )
    cluster = paired_cluster_bootstrap_difference(
        challenger_frame,
        reference_frame,
        canonical_pair_classes(),
        metric,
        observation_key_columns=("experiment_unit", "row_uid"),
        replicates=replicates,
        seed=seed,
    )
    unit = paired_unit_statistics(
        challenger_units,
        reference_units,
        unit_column="experiment_unit",
        score_column="pair_micro_f1",
        bootstrap_replicates=replicates,
        seed=seed,
    )
    return {
        "comparison_design": "matched_identical_task_grid",
        "partition": partition,
        "review_cluster_interval": cluster,
        "paired_unit_statistics": unit,
    }


def cross_protocol_degradation(
    harder: Sequence[ProtocolResult],
    easier: Sequence[ProtocolResult],
    *,
    partition: str = "overall",
    metric: str = "pair_micro_f1",
    replicates: int = 20_000,
    seed: int = 13,
) -> dict[str, object]:
    """Estimate harder-minus-easier change without claiming identical tasks."""

    harder_frame, _ = aggregate_protocol_predictions(harder, partition=partition)
    easier_frame, _ = aggregate_protocol_predictions(easier, partition=partition)
    cluster = shared_cluster_bootstrap_difference(
        harder_frame,
        easier_frame,
        canonical_pair_classes(),
        metric,
        challenger_observation_keys=("experiment_unit", "row_uid"),
        reference_observation_keys=("experiment_unit", "row_uid"),
        replicates=replicates,
        seed=seed,
    )
    return {
        "comparison_design": "different_task_grids_shared_review_resampling",
        "partition": partition,
        "review_cluster_interval": cluster,
        "paired_fold_sign_flip_permitted": False,
        "interpretation": (
            "The interval captures shared review-sampling uncertainty. "
            "It does not make the two task definitions identical."
        ),
    }


def holm_adjust_comparison_family(
    comparisons: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Attach Holm-adjusted p-values to one preregistered sign-flip family."""

    p_values = [
        float(value["paired_unit_statistics"]["sign_flip_p_value_two_sided"])  # type: ignore[index]
        for value in comparisons
    ]
    adjusted = holm_adjust(p_values)
    return [
        {
            **value,
            "holm_family_size": len(comparisons),
            "holm_adjusted_sign_flip_p_value": adjusted[index],
        }
        for index, value in enumerate(comparisons)
    ]
