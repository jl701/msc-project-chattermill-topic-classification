from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.evaluation.metrics import evaluate_pair_and_aspect, pairs_to_aspects
from msc_project.baselines.classical import threshold_predictions


class MetricsTest(unittest.TestCase):
    def test_pair_and_aspect_scores(self) -> None:
        true_pairs = [["A | positive"], ["B | negative"]]
        pred_pairs = [["A | positive"], ["B | positive"]]
        classes = ["A | positive", "B | negative", "B | positive"]

        scores = evaluate_pair_and_aspect(true_pairs, pred_pairs, classes)

        self.assertAlmostEqual(scores["pair_micro_f1"], 0.5)
        self.assertAlmostEqual(scores["aspect_micro_f1"], 1.0)

    def test_pairs_to_aspects(self) -> None:
        self.assertEqual(
            pairs_to_aspects([["A | positive", "A | negative"], ["B | neutral"]]),
            [["A"], ["B"]],
        )

    def test_threshold_predictions_ensures_one_label(self) -> None:
        pred = threshold_predictions(
            y_score=__import__("numpy").array([[0.1, 0.2], [0.8, 0.1]]),
            threshold=0.5,
            ensure_one=True,
        )
        self.assertEqual(pred.tolist(), [[0, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
