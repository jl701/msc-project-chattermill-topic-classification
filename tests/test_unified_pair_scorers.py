from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.unified_pair_scorers import (
    UnifiedPairCrossEncoderConfig,
    UnifiedPairDataset,
    UnifiedTfidfPairScorer,
    build_unified_pair_eval_grid,
    build_unified_pair_optimizer_and_scheduler,
    make_unified_pair_train_loader,
    score_unified_pair_manifest,
    train_unified_pair_epoch,
    validate_pair_manifest,
)


def pair_manifest() -> pd.DataFrame:
    rows = [
        ("The app is easy and excellent", "Mobile app and website experience", "Online: App", "positive", 1),
        ("The app is easy and excellent", "Email support from staff", "Support: Email", "positive", 0),
        ("Email support was rude and slow", "Email support from staff", "Support: Email", "negative", 1),
        ("Email support was rude and slow", "Mobile app and website experience", "Online: App", "negative", 0),
        ("The discount offer was great", "Discounts and promotional offers", "Value: Discounts", "positive", 1),
        ("The discount offer was great", "Email support from staff", "Support: Email", "positive", 0),
        ("The website had a terrible problem", "Mobile app and website experience", "Online: App", "negative", 1),
        ("The website had a terrible problem", "Discounts and promotional offers", "Value: Discounts", "negative", 0),
    ]
    return pd.DataFrame(
        [
            {
                "text": text,
                "candidate_text": candidate_text,
                "candidate_aspect": aspect,
                "candidate_sentiment": sentiment,
                "target": target,
                "row_uid": f"row-{index // 2}",
                "negative_type": "positive" if target else "hard_negative",
            }
            for index, (text, candidate_text, aspect, sentiment, target) in enumerate(rows)
        ]
    )


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(self, first, second, **kwargs):
        self.calls.append((list(first), list(second)))
        lengths = [len(a.split()) + len(b.split()) for a, b in zip(first, second)]
        input_ids = torch.tensor([[length, 1] for length in lengths], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class TinyPairModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.projection(input_ids.float())
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


class UnifiedPairScorersTest(unittest.TestCase):
    def test_cross_encoder_defaults_match_preregistered_recipe(self) -> None:
        config = UnifiedPairCrossEncoderConfig()
        self.assertIsNone(config.model_revision)
        self.assertEqual(config.max_length, 256)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.eval_batch_size, 96)
        self.assertEqual(config.learning_rate, 3e-5)
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.warmup_ratio, 0.1)
        self.assertEqual(config.seed, 13)
        self.assertTrue(config.use_amp)

    def test_manifest_validation_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_text"):
            validate_pair_manifest(pd.DataFrame({"text": ["x"]}), require_target=True)

        invalid = pair_manifest()
        invalid["target"] = invalid["target"].astype(float)
        invalid.loc[0, "target"] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_pair_manifest(invalid, require_target=True)

    def test_tfidf_scorer_fits_and_scores_unseen_candidate_text(self) -> None:
        train = pair_manifest()
        scorer = UnifiedTfidfPairScorer().fit(train)
        evaluation = train.iloc[[0, 1]].drop(columns="target").copy()
        evaluation["candidate_text"] = [
            "Unseen competitor brand comparison zyxwv",
            "Unseen competitor brand comparison zyxwv",
        ]
        evaluation["candidate_aspect"] = "Company: Competitor"

        probabilities = scorer.predict_proba(evaluation)
        scores = scorer.score_manifest(evaluation)

        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities[:, 1], scores)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
        self.assertNotIn("zyxwv", scorer.word_vectorizer.vocabulary_)
        self.assertEqual(scorer.transform_features(evaluation).shape, (2, 6))

    def test_tfidf_features_match_preregistered_six_feature_contract(self) -> None:
        scorer = UnifiedTfidfPairScorer().fit(pair_manifest())
        evaluation = pair_manifest().iloc[[0]].drop(columns="target").copy()
        evaluation.loc[:, "text"] = "email email support was great great bad filler"
        evaluation.loc[:, "candidate_text"] = "Email support from staff"
        evaluation.loc[:, "candidate_aspect"] = "Support: Email"
        evaluation.loc[:, "candidate_sentiment"] = "positive"

        features = scorer.transform_features(evaluation)

        self.assertEqual(
            scorer.feature_names,
            (
                "word_review_candidate_cosine",
                "character_review_candidate_cosine",
                "aspect_cue_token_coverage",
                "candidate_token_coverage",
                "sentiment_cue_token_coverage",
                "review_sentiment_cue_density",
            ),
        )
        self.assertEqual(features.shape, (1, len(scorer.feature_names)))
        self.assertAlmostEqual(features[0, 2], 1.0)
        self.assertAlmostEqual(features[0, 3], 2.0 / 4.0)
        self.assertAlmostEqual(features[0, 4], 1.0 / 13.0)
        self.assertAlmostEqual(features[0, 5], 2.0 / 8.0)

    def test_pair_dataset_uses_sentence_pairs_without_download(self) -> None:
        tokenizer = FakeTokenizer()
        dataset = UnifiedPairDataset(pair_manifest(), tokenizer, max_length=256, include_labels=True)

        self.assertEqual(len(dataset), len(pair_manifest()))
        self.assertEqual(tokenizer.calls[0][0][0], "The app is easy and excellent")
        self.assertEqual(tokenizer.calls[0][1][0], "Mobile app and website experience")
        self.assertEqual(int(dataset[0]["labels"]), 1)

    def test_cross_encoder_trains_one_epoch_and_scores_manifest_offline(self) -> None:
        manifest = pair_manifest()
        tokenizer = FakeTokenizer()
        config = UnifiedPairCrossEncoderConfig(
            batch_size=4,
            eval_batch_size=3,
            epochs=1,
            use_amp=False,
        )
        model = TinyPairModel()
        loader = make_unified_pair_train_loader(manifest, tokenizer, config)
        optimizer, scheduler = build_unified_pair_optimizer_and_scheduler(model, loader, config)

        loss = train_unified_pair_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            torch.device("cpu"),
            config,
        )
        scores = score_unified_pair_manifest(
            model,
            tokenizer,
            manifest.drop(columns="target"),
            config,
            torch.device("cpu"),
        )

        self.assertTrue(np.isfinite(loss))
        self.assertEqual(scores.shape, (len(manifest),))
        self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).all())

    def test_eval_grid_is_stable_row_major_cross_product(self) -> None:
        reviews = pd.DataFrame(
            {"row_uid": ["r1", "r2"], "text": ["first", "second"]}
        )
        candidates = pd.DataFrame(
            {
                "candidate_text": ["candidate a", "candidate b"],
                "candidate_aspect": ["A", "B"],
                "candidate_sentiment": ["positive", "negative"],
            }
        )

        grid = build_unified_pair_eval_grid(reviews, candidates)

        self.assertEqual(grid["row_uid"].tolist(), ["r1", "r1", "r2", "r2"])
        self.assertEqual(grid["candidate_aspect"].tolist(), ["A", "B", "A", "B"])
        self.assertEqual(set(grid["negative_type"]), {"eval_candidate"})


if __name__ == "__main__":
    unittest.main()
