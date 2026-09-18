from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import (
    aspect_conditioned_sentiment_text,
    build_sentiment_feature_lookup,
    build_sentiment_lookup,
    pair_predictions_from_aspects,
    sentiment_lookup_from_features,
    sentiment_probability_features,
    train_aspect_conditioned_sentiment_model,
    validate_sentiment_mode,
)


class CandidateLabelSentimentTest(unittest.TestCase):
    def test_aspect_conditioned_input_contains_review_and_aspect(self) -> None:
        text = aspect_conditioned_sentiment_text("Email was quick.", "Staff support: Email")

        self.assertIn("Review: Email was quick.", text)
        self.assertIn("Aspect: Staff support: Email", text)
        self.assertIn("Aspect keywords: Email", text)

    def test_pair_predictions_use_per_aspect_sentiment(self) -> None:
        predictions = pair_predictions_from_aspects(
            [["Staff support: Email", "Operations: Delivery"]],
            [{"Staff support: Email": "positive", "Operations: Delivery": "negative"}],
        )

        self.assertEqual(
            predictions,
            [["Staff support: Email | positive", "Operations: Delivery | negative"]],
        )

    def test_global_lookup_applies_one_sentiment_to_all_candidate_aspects(self) -> None:
        class FakeGlobalSentiment:
            def predict(self, texts):
                return np.array(["negative" for _ in texts])

        lookup = build_sentiment_lookup(
            FakeGlobalSentiment(),
            ["The delivery was late."],
            ["Staff support: Email", "Operations: Delivery"],
            "global",
        )

        self.assertEqual(
            lookup,
            [{"Staff support: Email": "negative", "Operations: Delivery": "negative"}],
        )

    def test_probability_features_include_margin_and_entropy(self) -> None:
        features = sentiment_probability_features({"negative": 0.1, "neutral": 0.2, "positive": 0.7})

        self.assertEqual(features["predicted_sentiment"], "positive")
        self.assertAlmostEqual(features["top_probability"], 0.7)
        self.assertAlmostEqual(features["second_probability"], 0.2)
        self.assertAlmostEqual(features["sentiment_margin"], 0.5)
        self.assertGreater(features["sentiment_entropy"], 0.0)

    def test_global_feature_lookup_reuses_document_sentiment_scores_per_aspect(self) -> None:
        class FakeGlobalSentiment:
            classes_ = np.array(["negative", "neutral", "positive"])

            def predict(self, texts):
                return np.array(["positive" for _ in texts])

            def predict_proba(self, texts):
                return np.array([[0.1, 0.2, 0.7] for _ in texts])

        features = build_sentiment_feature_lookup(
            FakeGlobalSentiment(),
            ["The delivery was great."],
            ["Staff support: Email", "Operations: Delivery"],
            "global",
        )

        self.assertEqual(features[0]["Staff support: Email"]["predicted_sentiment"], "positive")
        self.assertEqual(features[0]["Operations: Delivery"]["predicted_sentiment"], "positive")
        self.assertAlmostEqual(features[0]["Staff support: Email"]["sentiment_margin"], 0.5)
        self.assertEqual(
            sentiment_lookup_from_features(features),
            [{"Staff support: Email": "positive", "Operations: Delivery": "positive"}],
        )

    def test_aspect_conditioned_lookup_can_return_different_sentiments_for_same_review(self) -> None:
        rows = []
        for _ in range(8):
            rows.extend(
                [
                    {
                        "text": "The same review mentions email and delivery.",
                        "supervision_labels": [("Staff support: Email", "positive")],
                    },
                    {
                        "text": "The same review mentions email and delivery.",
                        "supervision_labels": [("Operations: Delivery", "negative")],
                    },
                    {
                        "text": "The same review mentions email and delivery.",
                        "supervision_labels": [("Account management: Billing", "neutral")],
                    },
                ]
            )
        frame = pd.DataFrame(rows)
        model = train_aspect_conditioned_sentiment_model(frame)

        lookup = build_sentiment_lookup(
            model,
            ["The same review mentions email and delivery."],
            [
                "Staff support: Email",
                "Operations: Delivery",
                "Account management: Billing",
            ],
            "aspect_conditioned",
        )

        self.assertEqual(lookup[0]["Staff support: Email"], "positive")
        self.assertEqual(lookup[0]["Operations: Delivery"], "negative")
        self.assertEqual(lookup[0]["Account management: Billing"], "neutral")

    def test_single_class_aspect_conditioned_sentiment_uses_constant_fallback(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "text": "Email was good.",
                    "supervision_labels": [("Staff support: Email", "positive")],
                }
            ]
        )
        model = train_aspect_conditioned_sentiment_model(frame)

        self.assertEqual(model.predict_pairs(["Anything"], ["Any aspect"]), ["positive"])

        features = model.predict_pairs_with_scores(["Anything"], ["Any aspect"])
        self.assertEqual(features[0]["predicted_sentiment"], "positive")
        self.assertEqual(features[0]["top_probability"], 1.0)
        self.assertEqual(features[0]["sentiment_margin"], 1.0)

    def test_unknown_sentiment_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_sentiment_mode("document")


if __name__ == "__main__":
    unittest.main()
