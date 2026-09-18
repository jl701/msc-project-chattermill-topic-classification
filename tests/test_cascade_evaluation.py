from __future__ import annotations

import unittest
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.evaluation.cascade import (
    combine_predictions,
    label_reliability,
    prediction_features,
    score_margin_features,
    validate_aligned_rows,
)


class CascadeEvaluationTests(unittest.TestCase):
    def test_combine_predictions_modes(self) -> None:
        local = ["A | positive", "B | negative"]
        gemini = ["B | negative", "C | neutral"]

        self.assertEqual(combine_predictions(local, gemini, "replace"), ["B | negative", "C | neutral"])
        self.assertEqual(combine_predictions(local, [], "gemini_nonempty_else_local"), local)
        self.assertEqual(combine_predictions(local, gemini, "union"), ["A | positive", "B | negative", "C | neutral"])
        self.assertEqual(combine_predictions(local, gemini, "intersection"), ["B | negative"])
        self.assertEqual(combine_predictions(local, gemini, "agreement_or_gemini"), ["B | negative"])
        self.assertEqual(combine_predictions(local, ["C | neutral"], "agreement_or_local"), local)

    def test_label_reliability_counts_pair_precision_recall(self) -> None:
        rows = [
            {"gold_pair_labels": ["A | positive"], "pred_pair_labels": ["A | positive", "B | negative"]},
            {"gold_pair_labels": ["B | negative"], "pred_pair_labels": ["A | positive"]},
        ]

        stats = label_reliability(rows, ["A | positive", "B | negative"], "pair")

        self.assertEqual(stats["A | positive"].true_positive, 1)
        self.assertEqual(stats["A | positive"].false_positive, 1)
        self.assertEqual(stats["A | positive"].false_negative, 0)
        self.assertAlmostEqual(stats["A | positive"].precision, 0.5)
        self.assertAlmostEqual(stats["A | positive"].recall, 1.0)
        self.assertEqual(stats["B | negative"].true_positive, 0)
        self.assertEqual(stats["B | negative"].false_positive, 1)
        self.assertEqual(stats["B | negative"].false_negative, 1)

    def test_prediction_features_use_validation_reliability(self) -> None:
        rows = [
            {"gold_pair_labels": ["A | positive"], "pred_pair_labels": ["A | positive", "B | negative"]},
            {"gold_pair_labels": ["B | negative"], "pred_pair_labels": ["A | positive"]},
        ]
        pair_stats = label_reliability(rows, ["A | positive", "B | negative"], "pair")
        aspect_stats = label_reliability(rows, ["A", "B"], "aspect")
        sentiment_stats = label_reliability(rows, ["positive", "negative"], "sentiment")

        features = prediction_features(rows[0], pair_stats, aspect_stats, sentiment_stats)

        self.assertEqual(features["pred_count"], 2.0)
        self.assertEqual(features["has_multi_prediction"], 1.0)
        self.assertGreater(features["low_min_pair_precision"], features["low_mean_pair_precision"])

    def test_prediction_features_include_score_margin_uncertainty_when_available(self) -> None:
        row = {
            "gold_pair_labels": ["A | positive"],
            "pred_pair_labels": ["A | positive"],
            "score_features": {
                "top_score": 0.55,
                "second_score": 0.49,
                "score_margin": 0.06,
                "min_abs_distance_to_threshold": 0.02,
                "selected_score_min": 0.55,
                "selected_score_mean": 0.55,
                "above_threshold_count": 1,
            },
        }
        pair_stats = label_reliability([row], ["A | positive"], "pair")
        aspect_stats = label_reliability([row], ["A"], "aspect")
        sentiment_stats = label_reliability([row], ["positive"], "sentiment")

        features = prediction_features(row, pair_stats, aspect_stats, sentiment_stats)

        self.assertAlmostEqual(features["aspect_score_margin"], 0.06)
        self.assertAlmostEqual(features["low_aspect_score_margin"], 0.94)
        self.assertAlmostEqual(features["score_near_threshold"], 0.98)
        self.assertEqual(features["aspect_above_threshold_count"], 1.0)

    def test_score_margin_features_ignores_missing_or_non_numeric_values(self) -> None:
        features = score_margin_features({"top_score": "bad", "score_margin": None})

        self.assertEqual(features, {})

    def test_validate_aligned_rows_checks_gold_mismatch(self) -> None:
        local = [{"row_uid": "validation:1", "gold_pair_labels": ["A | positive"]}]
        gemini = [{"row_uid": "validation:1", "gold_pair_labels": ["A | negative"]}]

        with self.assertRaises(ValueError):
            validate_aligned_rows(local, gemini)


if __name__ == "__main__":
    unittest.main()
