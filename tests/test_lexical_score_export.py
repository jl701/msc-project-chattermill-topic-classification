from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from msc_project.baselines.candidate_label import (
    candidate_aspect_predictions_from_scores,
    candidate_score_features,
)
from run_generalisation_baselines import write_candidate_prediction_rows


class LexicalScoreExportTests(unittest.TestCase):
    def test_candidate_score_features_record_threshold_distance_and_selection(self) -> None:
        aspect = "Company brand: Competitor"
        features = candidate_score_features(
            np.array([0.38]),
            [aspect],
            [aspect],
            0.37,
        )

        self.assertAlmostEqual(features["candidate_aspect_scores"][aspect], 0.38)
        self.assertAlmostEqual(features["threshold"], 0.37)
        self.assertAlmostEqual(features["top_distance_to_threshold"], 0.01)
        self.assertAlmostEqual(features["min_abs_distance_to_threshold"], 0.01)
        self.assertEqual(features["selected_count"], 1)

    def test_candidate_selection_allows_empty_predictions(self) -> None:
        predictions = candidate_aspect_predictions_from_scores(
            np.array([[0.10], [0.40]]),
            ["Company brand: Competitor"],
            threshold=0.37,
            ensure_one=False,
        )

        self.assertEqual(predictions, [[], ["Company brand: Competitor"]])

    def test_prediction_export_is_qwen_alignable_and_omits_review_text(self) -> None:
        aspect = "Company brand: Competitor"
        frame = pd.DataFrame(
            [
                {
                    "id": "610309432",
                    "row_uid": "validation:610309432",
                    "original_split": "validation",
                    "org_index": 600,
                    "text": "This text must not be exported.",
                    "supervision_pair_labels": [f"{aspect} | positive"],
                }
            ]
        )
        sentiment_features = [
            {
                aspect: {
                    "predicted_sentiment": "positive",
                    "sentiment_probabilities": {"negative": 0.1, "neutral": 0.2, "positive": 0.7},
                    "sentiment_margin": 0.5,
                }
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.jsonl"
            write_candidate_prediction_rows(
                frame,
                [[f"{aspect} | positive"]],
                np.array([[0.38]]),
                [aspect],
                [[aspect]],
                0.37,
                sentiment_features,
                path,
            )
            row = json.loads(path.read_text(encoding="utf-8").strip())

        self.assertEqual(row["row_uid"], "validation:610309432")
        self.assertEqual(row["heldout_aspect"], aspect)
        self.assertNotIn("text", row)
        self.assertEqual(row["gold_pair_labels"], [f"{aspect} | positive"])
        self.assertEqual(row["final_local_pair_prediction"], [f"{aspect} | positive"])
        self.assertAlmostEqual(row["lexical_features"][aspect]["cosine_similarity"], 0.38)
        self.assertAlmostEqual(row["lexical_features"][aspect]["signed_distance_from_threshold"], 0.01)
        self.assertAlmostEqual(row["lexical_features"][aspect]["absolute_distance_from_threshold"], 0.01)
        self.assertTrue(row["lexical_features"][aspect]["local_selected"])
        self.assertEqual(row["lexical_features"][aspect]["global_sentiment_prediction"], "positive")


if __name__ == "__main__":
    unittest.main()
