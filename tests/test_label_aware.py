from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.label_aware import (
    aspect_predictions_from_scores,
    build_pair_examples,
    candidate_aspect_text,
    candidate_pair_text,
    predictions_from_scores,
)


class LabelAwareTest(unittest.TestCase):
    def test_candidate_pair_text_expands_pair_label(self) -> None:
        self.assertEqual(
            candidate_pair_text("Company brand: Competitor | negative"),
            "Aspect: Company brand: Competitor. Sentiment: negative.",
        )

    def test_candidate_aspect_text_expands_leaf(self) -> None:
        self.assertEqual(
            candidate_aspect_text("Value: Discounts promotions"),
            "Aspect: Value: Discounts promotions. Topic keywords: Discounts promotions.",
        )

    def test_build_pair_examples_samples_negatives(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "text": "The app is good.",
                    "supervision_pair_labels": ["Online experience: App website | positive"],
                }
            ]
        )
        examples = build_pair_examples(
            frame,
            [
                "Online experience: App website | positive",
                "Online experience: App website | negative",
                "Staff support: Email | negative",
            ],
            "supervision_pair_labels",
            negatives_per_positive=2,
            seed=13,
        )
        self.assertEqual(examples["target"].tolist().count(1), 1)
        self.assertEqual(examples["target"].tolist().count(0), 2)

    def test_predictions_keep_one_sentiment_per_aspect(self) -> None:
        scores = np.array([[0.8, 0.7, 0.2]])
        labels = [
            "A | positive",
            "A | negative",
            "B | positive",
        ]
        self.assertEqual(predictions_from_scores(scores, labels, threshold=0.5), [["A | positive"]])

    def test_predictions_can_limit_labels_per_row(self) -> None:
        scores = np.array([[0.8, 0.6, 0.7]])
        labels = [
            "A | positive",
            "B | negative",
            "C | positive",
        ]
        self.assertEqual(
            predictions_from_scores(scores, labels, threshold=0.5, max_predictions_per_row=1),
            [["A | positive"]],
        )

    def test_aspect_predictions_can_limit_labels_per_row(self) -> None:
        scores = np.array([[0.4, 0.9, 0.8]])
        self.assertEqual(
            aspect_predictions_from_scores(scores, ["A", "B", "C"], threshold=0.3, max_predictions_per_row=1),
            [["B"]],
        )


if __name__ == "__main__":
    unittest.main()
