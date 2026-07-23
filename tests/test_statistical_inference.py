from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.statistical_inference import (
    build_cluster_sufficient_statistics,
    cluster_bootstrap_interval,
    holm_adjust,
    paired_cluster_bootstrap_difference,
    paired_unit_statistics,
)


CLASSES = ["A | negative", "A | neutral", "A | positive"]


def prediction_frame(predictions: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_uid": ["r1", "r2", "r3"],
            "gold_pairs": [["A | positive"], ["A | negative"], []],
            "pred_pairs": predictions,
        }
    )


def test_cluster_statistics_preserve_every_observation_in_a_review() -> None:
    frame = pd.DataFrame(
        {
            "fold_id": ["f1", "f2", "f1"],
            "row_uid": ["r1", "r1", "r2"],
            "gold_pairs": [["A | positive"], [], ["A | negative"]],
            "pred_pairs": [["A | positive"], [], []],
        }
    )
    statistics = build_cluster_sufficient_statistics(
        frame,
        CLASSES,
        observation_key_columns=("fold_id", "row_uid"),
    )
    assert statistics.cluster_ids == ("r1", "r2")
    assert statistics.rows.tolist() == [2, 1]
    assert statistics.pair_tp.tolist() == [1, 0]
    assert statistics.pair_fn.tolist() == [0, 1]


def test_bootstrap_point_recomputes_micro_f1_from_global_counts() -> None:
    frame = prediction_frame(
        [["A | positive"], ["A | positive"], ["A | neutral"]]
    )
    statistics = build_cluster_sufficient_statistics(frame, CLASSES)
    result = cluster_bootstrap_interval(
        statistics,
        "pair_micro_f1",
        replicates=200,
        seed=13,
    )
    # TP=1, FP=2, FN=1 -> 2/(2+2+1) = 0.4.
    assert result["point_estimate"] == pytest.approx(0.4)
    assert 0.0 <= result["ci_lower"] <= result["ci_upper"] <= 1.0


def test_identical_paired_predictions_have_zero_difference_interval() -> None:
    frame = prediction_frame([["A | positive"], ["A | negative"], []])
    result = paired_cluster_bootstrap_difference(
        frame,
        frame.copy(),
        CLASSES,
        "pair_micro_f1",
        replicates=200,
        seed=13,
    )
    assert result["point_difference"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)


def test_paired_bootstrap_uses_same_clusters_and_finds_uniform_gain() -> None:
    challenger = prediction_frame([["A | positive"], ["A | negative"], []])
    reference = prediction_frame([[], [], ["A | neutral"]])
    result = paired_cluster_bootstrap_difference(
        challenger,
        reference,
        CLASSES,
        "pair_micro_f1",
        replicates=500,
        seed=13,
    )
    assert result["point_difference"] > 0.0
    assert result["ci_lower"] >= 0.0


def test_paired_alignment_fails_on_gold_or_key_mismatch() -> None:
    challenger = prediction_frame([["A | positive"], ["A | negative"], []])
    reference = challenger.copy(deep=True)
    reference.at[0, "gold_pairs"] = ["A | neutral"]
    with pytest.raises(ValueError, match="gold"):
        paired_cluster_bootstrap_difference(
            challenger,
            reference,
            CLASSES,
            "pair_micro_f1",
            replicates=10,
        )

    reference = challenger.copy(deep=True)
    reference.at[0, "row_uid"] = "different"
    with pytest.raises(ValueError, match="observation keys"):
        paired_cluster_bootstrap_difference(
            challenger,
            reference,
            CLASSES,
            "pair_micro_f1",
            replicates=10,
        )


def test_paired_unit_statistics_align_and_report_exact_sign_flip() -> None:
    challenger = pd.DataFrame({"aspect": ["a", "b", "c"], "f1": [0.5, 0.6, 0.7]})
    reference = pd.DataFrame({"aspect": ["c", "a", "b"], "f1": [0.6, 0.4, 0.5]})
    result = paired_unit_statistics(
        challenger,
        reference,
        unit_column="aspect",
        score_column="f1",
        bootstrap_replicates=500,
    )
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["wins"] == 3
    assert result["sign_flip_method"] == "exact"


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
