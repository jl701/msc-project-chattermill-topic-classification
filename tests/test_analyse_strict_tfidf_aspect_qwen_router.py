from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_strict_tfidf_aspect_qwen_router.py"
SPEC = importlib.util.spec_from_file_location(
    "analyse_strict_tfidf_aspect_qwen_router", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(uid: str, score: float, threshold: float = 0.5) -> dict:
    present = score >= threshold
    return {
        "row_uid": uid,
        "gold_pair_labels": [],
        "local_pred_pair_labels": ["A | neutral"] if present else [],
        "qwen_pred_pair_labels": ["A | positive"],
        "presence_score": score,
        "threshold": threshold,
        "local_present": present,
        "distance": abs(score - threshold),
    }


def test_ranked_cutoff_uses_registered_ceil_fraction_rule() -> None:
    rows = [
        row("a", 0.49),
        row("b", 0.48),
        row("c", 0.47),
        row("d", 0.46),
        row("e", 0.70),
    ]
    assert MODULE.ranked_cutoff(rows, present=False, fraction=0.0) is None
    assert MODULE.ranked_cutoff(rows, present=False, fraction=0.25) == pytest.approx(
        0.01
    )
    assert MODULE.ranked_cutoff(rows, present=False, fraction=0.5) == pytest.approx(
        0.02
    )


def test_asymmetric_route_uses_qwen_only_inside_frozen_side_cutoffs() -> None:
    rows = [row("abs-near", 0.49), row("abs-far", 0.2), row("yes-near", 0.51)]
    predictions, calls = MODULE.route_predictions(
        rows,
        rescue_cutoff=0.02,
        confirm_cutoff=None,
    )
    assert calls == [True, False, False]
    assert predictions == [["A | positive"], [], ["A | neutral"]]


def test_global_policy_respects_call_budget_and_registered_tie_breaks() -> None:
    base = {
        "pair_samples_f1_mean": 0.1,
        "pair_false_positive_rows_per_100_mean": 4.0,
        "pair_micro_precision_mean": 0.5,
    }
    rows = [
        {
            **base,
            "scenario": "over_budget",
            "pair_micro_f1_mean": 0.9,
            "qwen_call_rate_mean": 0.51,
        },
        {
            **base,
            "scenario": "b_policy",
            "pair_micro_f1_mean": 0.4,
            "qwen_call_rate_mean": 0.2,
        },
        {
            **base,
            "scenario": "a_policy",
            "pair_micro_f1_mean": 0.4,
            "qwen_call_rate_mean": 0.2,
        },
    ]
    selected = MODULE.select_global_policy(rows, maximum_call_rate=0.5)
    assert selected["scenario"] == "a_policy"


def test_row_uid_join_accepts_different_physical_order() -> None:
    strict = [
        {
            "row_uid": "validation:1",
            "method": MODULE.LOCAL_METHOD,
            "split": "validation",
            "heldout_aspect": "A",
            "candidate_aspects": ["A"],
            "gold_pair_labels": [],
            "presence_score": 0.1,
            "selected_threshold": 0.2,
            "predicted_present": False,
            "pred_pair_labels": [],
            "sentiment_features": {"predicted_sentiment": "neutral"},
        },
        {
            "row_uid": "validation:2",
            "method": MODULE.LOCAL_METHOD,
            "split": "validation",
            "heldout_aspect": "A",
            "candidate_aspects": ["A"],
            "gold_pair_labels": ["A | positive"],
            "presence_score": 0.4,
            "selected_threshold": 0.2,
            "predicted_present": True,
            "pred_pair_labels": ["A | neutral"],
            "sentiment_features": {"predicted_sentiment": "neutral"},
        },
    ]
    sentiment = [
        {
            "row_uid": "validation:1",
            "original_split": "validation",
            "gold_pair_labels": [],
            "sentiment_features": {"A": {"predicted_sentiment": "negative"}},
        },
        {
            "row_uid": "validation:2",
            "original_split": "validation",
            "gold_pair_labels": ["A | positive"],
            "sentiment_features": {"A": {"predicted_sentiment": "positive"}},
        },
    ]
    qwen = [
        {
            "row_uid": "validation:2",
            "gold_pair_labels": ["A | positive"],
            "pred_pair_labels": ["A | positive"],
        },
        {
            "row_uid": "validation:1",
            "gold_pair_labels": [],
            "pred_pair_labels": [],
        },
    ]
    aspect, aligned = MODULE.align_router_rows(
        strict, sentiment, qwen, stage="validation", threshold=0.3
    )
    assert aspect == "A"
    assert [item["row_uid"] for item in aligned] == ["validation:1", "validation:2"]
    assert aligned[1]["local_pred_pair_labels"] == ["A | positive"]


def test_row_uid_or_gold_mismatch_is_rejected() -> None:
    strict = [
        {
            "row_uid": "test:1",
            "method": MODULE.LOCAL_METHOD,
            "split": "test",
            "heldout_aspect": "A",
            "candidate_aspects": ["A"],
            "gold_pair_labels": [],
            "presence_score": 0.1,
            "selected_threshold": 0.2,
            "predicted_present": False,
            "pred_pair_labels": [],
            "sentiment_features": {"predicted_sentiment": "neutral"},
        }
    ]
    sentiment = [
        {
            "row_uid": "test:1",
            "original_split": "test",
            "gold_pair_labels": [],
            "sentiment_features": {"A": {"predicted_sentiment": "neutral"}},
        }
    ]
    with pytest.raises(ValueError, match="row_uid set mismatch"):
        MODULE.align_router_rows(
            strict,
            sentiment,
            [{"row_uid": "test:2", "gold_pair_labels": [], "pred_pair_labels": []}],
            stage="test",
            threshold=0.2,
        )
