from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rescore_similarity_loao_with_aspect_sentiment import (
    METHODS,
    align_prediction_rows,
    predicted_pairs,
    select_threshold,
    validate_selection_manifest,
)


ASPECT = "Staff support: Email"


def strict_row(
    uid: str,
    *,
    score: float,
    threshold: float,
    gold: list[str],
    sentiment: str,
) -> dict[str, object]:
    present = score >= threshold
    return {
        "method": METHODS[0],
        "split": "validation",
        "row_uid": uid,
        "heldout_aspect": ASPECT,
        "candidate_aspects": [ASPECT],
        "gold_pair_labels": gold,
        "pred_pair_labels": [f"{ASPECT} | {sentiment}"] if present else [],
        "presence_score": score,
        "selected_threshold": threshold,
        "predicted_present": present,
        "sentiment_features": {"predicted_sentiment": sentiment},
    }


def sentiment_row(
    uid: str,
    *,
    gold: list[str],
    sentiment: str,
) -> dict[str, object]:
    return {
        "row_uid": uid,
        "original_split": "validation",
        "gold_pair_labels": gold,
        "sentiment_features": {
            ASPECT: {
                "predicted_sentiment": sentiment,
            }
        },
    }


class AspectSentimentRescoreTests(unittest.TestCase):
    def test_alignment_extracts_both_sentiment_heads(self) -> None:
        gold = [f"{ASPECT} | negative"]
        aspect, rows = align_prediction_rows(
            [strict_row("validation:1", score=0.8, threshold=0.5, gold=gold, sentiment="positive")],
            [sentiment_row("validation:1", gold=gold, sentiment="negative")],
            expected_method=METHODS[0],
            expected_stage="validation",
        )

        self.assertEqual(aspect, ASPECT)
        self.assertEqual(rows[0]["global_sentiment"], "positive")
        self.assertEqual(rows[0]["aspect_sentiment"], "negative")

    def test_alignment_rejects_row_uid_reordering(self) -> None:
        strict = [
            strict_row("validation:1", score=0.8, threshold=0.5, gold=[], sentiment="positive"),
            strict_row("validation:2", score=0.2, threshold=0.5, gold=[], sentiment="positive"),
        ]
        sentiment = [
            sentiment_row("validation:2", gold=[], sentiment="negative"),
            sentiment_row("validation:1", gold=[], sentiment="negative"),
        ]

        with self.assertRaisesRegex(ValueError, "row_uid sequence mismatch"):
            align_prediction_rows(
                strict,
                sentiment,
                expected_method=METHODS[0],
                expected_stage="validation",
            )

    def test_fixed_threshold_replaces_sentiment_without_changing_presence(self) -> None:
        rows = [
            {
                "presence_score": 0.8,
                "aspect_sentiment": "negative",
            },
            {
                "presence_score": 0.2,
                "aspect_sentiment": "positive",
            },
        ]
        predictions = predicted_pairs(rows, ASPECT, 0.5, "aspect_sentiment")
        self.assertEqual(predictions, [[f"{ASPECT} | negative"], []])

    def test_threshold_selection_uses_pair_micro_f1_and_higher_threshold_tie_break(self) -> None:
        rows = [
            {
                "row_uid": "validation:1",
                "gold_pair_labels": [f"{ASPECT} | positive"],
                "presence_score": 0.8,
                "original_threshold": 0.5,
                "global_pred_pair_labels": [f"{ASPECT} | positive"],
                "global_sentiment": "positive",
                "aspect_sentiment": "positive",
            },
            {
                "row_uid": "validation:2",
                "gold_pair_labels": [],
                "presence_score": 0.2,
                "original_threshold": 0.5,
                "global_pred_pair_labels": [],
                "global_sentiment": "positive",
                "aspect_sentiment": "positive",
            },
        ]
        threshold, sweep, metrics = select_threshold(rows, ASPECT)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(float(sweep.iloc[0]["pair_micro_f1"]), 1.0)
        self.assertEqual(float(metrics["presence_f1"]), 1.0)

    def test_incomplete_selection_manifest_is_rejected(self) -> None:
        payload = {
            "stage": "validation",
            "protocol_complete": False,
            "methods": METHODS,
            "folds": [],
        }
        with self.assertRaisesRegex(ValueError, "complete validation"):
            validate_selection_manifest(payload)


if __name__ == "__main__":
    unittest.main()
