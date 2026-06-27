from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.transformer_sentiment import (
    aspect_sentiment_prompt,
    build_aspect_sentiment_examples,
    class_weights,
    selected_keys,
)


class TransformerSentimentTest(unittest.TestCase):
    def test_aspect_sentiment_prompt_contains_canonical_aspect_and_leaf(self) -> None:
        prompt = aspect_sentiment_prompt("Staff support: Email")

        self.assertIn("Aspect: Staff support: Email.", prompt)
        self.assertIn("Aspect keywords: Email.", prompt)

    def test_build_aspect_sentiment_examples_expands_supervision_labels(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "text": "Email was good but delivery was bad.",
                    "supervision_labels": [
                        ("Staff support: Email", "positive"),
                        ("Operations: Delivery", "negative"),
                    ],
                }
            ]
        )

        examples = build_aspect_sentiment_examples(frame)

        self.assertEqual(examples["aspect"].tolist(), ["Staff support: Email", "Operations: Delivery"])
        self.assertEqual(examples["sentiment"].tolist(), ["positive", "negative"])
        self.assertEqual(examples["target"].tolist(), [2, 0])

    def test_class_weights_support_balanced_and_sqrt_modes(self) -> None:
        examples = pd.DataFrame({"target": [0, 0, 2]})

        balanced = class_weights(examples, "balanced", torch.device("cpu"))
        sqrt = class_weights(examples, "sqrt", torch.device("cpu"))

        self.assertIsNotNone(balanced)
        self.assertIsNotNone(sqrt)
        self.assertGreater(float(balanced[2]), float(balanced[0]))
        self.assertGreater(float(sqrt[2]), float(sqrt[0]))

    def test_selected_keys_promotes_primary_metric(self) -> None:
        self.assertEqual(selected_keys("macro_f1"), ("macro_f1", "accuracy", "micro_f1"))

    def test_unknown_selected_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            selected_keys("loss")


if __name__ == "__main__":
    unittest.main()
