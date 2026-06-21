from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.evaluation.metrics import evaluate_pair_and_aspect, pair_to_components, pairs_to_aspects
from msc_project.baselines.classical import threshold_predictions


class MetricsTest(unittest.TestCase):
    def test_pair_and_aspect_scores(self) -> None:
        true_pairs = [["A | positive"], ["B | negative"]]
        pred_pairs = [["A | positive"], ["B | positive"]]
        classes = ["A | positive", "B | negative", "B | positive"]

        scores = evaluate_pair_and_aspect(true_pairs, pred_pairs, classes)

        self.assertAlmostEqual(scores["pair_micro_f1"], 0.5)
        self.assertAlmostEqual(scores["pair_micro_precision"], 0.5)
        self.assertAlmostEqual(scores["pair_micro_recall"], 0.5)
        self.assertAlmostEqual(scores["aspect_micro_f1"], 1.0)
        self.assertAlmostEqual(scores["sentiment_accuracy_when_gold_aspect_predicted"], 0.5)

    def test_pair_to_components_splits_aspect_and_sentiment(self) -> None:
        self.assertEqual(pair_to_components("A: B | positive"), ("A: B", "positive"))

    def test_pairs_to_aspects(self) -> None:
        self.assertEqual(
            pairs_to_aspects([["A | positive", "A | negative"], ["B | neutral"]]),
            [["A"], ["B"]],
        )

    def test_single_class_sample_f1_with_empty_rows(self) -> None:
        scores = evaluate_pair_and_aspect(
            [["A | positive"], []],
            [["A | positive"], []],
            ["A | positive"],
        )
        self.assertAlmostEqual(scores["pair_samples_f1"], 0.5)
        self.assertAlmostEqual(scores["aspect_samples_f1"], 0.5)
        self.assertEqual(scores["pair_empty_gold_rows"], 1)
        self.assertEqual(scores["pair_empty_gold_and_prediction_rows"], 1)
        self.assertEqual(scores["pair_false_positive_rows"], 0)

    def test_false_positive_diagnostics_for_empty_gold_rows(self) -> None:
        scores = evaluate_pair_and_aspect(
            [["A | positive"], []],
            [["A | positive"], ["A | negative"]],
            ["A | positive", "A | negative"],
        )
        self.assertEqual(scores["pair_label_tp"], 1)
        self.assertEqual(scores["pair_label_fp"], 1)
        self.assertEqual(scores["pair_false_positive_rows"], 1)
        self.assertAlmostEqual(scores["pair_false_positive_rows_per_100"], 50.0)

    def test_threshold_predictions_ensures_one_label(self) -> None:
        pred = threshold_predictions(
            y_score=__import__("numpy").array([[0.1, 0.2], [0.8, 0.1]]),
            threshold=0.5,
            ensure_one=True,
        )
        self.assertEqual(pred.tolist(), [[0, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
