from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_comparisons import (
    ProtocolResult,
    cross_protocol_degradation,
    holm_adjust_comparison_family,
    matched_protocol_comparison,
)
from msc_project.experiments.taxonomy_protocol import registered_folds


def scored(fold, score: float, *, rows: tuple[str, ...] = ("r1", "r2")):
    aspect = fold.evaluation_aspects[0]
    records = []
    for row_uid in rows:
        for sentiment in ("negative", "neutral", "positive"):
            records.append(
                {
                    "row_uid": row_uid,
                    "candidate_aspect": aspect,
                    "candidate_sentiment": sentiment,
                    "pair_label": f"{aspect} | {sentiment}",
                    "target": int(row_uid == "r1" and sentiment == "positive"),
                    "score": score if sentiment == "positive" else 0.0,
                    "representation_variant": "minimal",
                    "is_seen": False,
                    "is_heldout": True,
                }
            )
    return pd.DataFrame.from_records(records)


def result(unit: str, score: float) -> ProtocolResult:
    fold = registered_folds("L1")[0]
    return ProtocolResult(unit, fold, "D", scored(fold, score), 0.5)


def test_matched_comparison_enforces_pairing_and_reports_unit_evidence() -> None:
    comparison = matched_protocol_comparison(
        [result("u1", 0.9), result("u2", 0.9)],
        [result("u1", 0.1), result("u2", 0.1)],
        replicates=100,
    )
    assert comparison["comparison_design"] == "matched_identical_task_grid"
    assert comparison["review_cluster_interval"]["point_difference"] > 0
    assert comparison["paired_unit_statistics"]["wins"] == 2

    with pytest.raises(ValueError, match="observation keys"):
        matched_protocol_comparison(
            [result("u1", 0.9)],
            [result("different", 0.1)],
            replicates=10,
        )


def test_cross_protocol_result_is_explicitly_not_an_identical_task() -> None:
    harder = [result("hard-a", 0.1), result("hard-b", 0.1)]
    easier = [result("easy", 0.9)]
    comparison = cross_protocol_degradation(
        harder,
        easier,
        replicates=100,
    )
    assert (
        comparison["comparison_design"]
        == "different_task_grids_shared_review_resampling"
    )
    assert comparison["paired_fold_sign_flip_permitted"] is False


def test_holm_is_applied_only_to_the_supplied_family() -> None:
    comparisons = [
        {"paired_unit_statistics": {"sign_flip_p_value_two_sided": 0.01}},
        {"paired_unit_statistics": {"sign_flip_p_value_two_sided": 0.04}},
    ]
    adjusted = holm_adjust_comparison_family(comparisons)
    assert adjusted[0]["holm_adjusted_sign_flip_p_value"] == pytest.approx(0.02)
    assert adjusted[1]["holm_adjusted_sign_flip_p_value"] == pytest.approx(0.04)
