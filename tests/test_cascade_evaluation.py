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

    def test_validate_aligned_rows_checks_gold_mismatch(self) -> None:
        local = [{"row_uid": "validation:1", "gold_pair_labels": ["A | positive"]}]
        gemini = [{"row_uid": "validation:1", "gold_pair_labels": ["A | negative"]}]

        with self.assertRaises(ValueError):
            validate_aligned_rows(local, gemini)


if __name__ == "__main__":
    unittest.main()
