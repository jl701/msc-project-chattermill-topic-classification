from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_similarity import (
    REGISTERED_CONFIGS,
    SparseCandidateScorer,
    masked_mean_pool,
    resolve_configs,
)


class CandidateSimilarityTests(unittest.TestCase):
    def test_registered_sentence_encoders_are_pinned(self) -> None:
        mini = REGISTERED_CONFIGS["minilm_l6_v2"]
        e5 = REGISTERED_CONFIGS["e5_base_v2"]

        self.assertEqual(mini.model_id, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(mini.revision, "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
        self.assertEqual(e5.candidate_prefix, "query: ")
        self.assertEqual(e5.review_prefix, "passage: ")
        self.assertEqual(e5.revision, "f52bf8ec8c7124536f0efb74aca902b2995e5bcd")

    def test_config_resolution_rejects_duplicates_and_unknown_names(self) -> None:
        self.assertEqual(resolve_configs(["bow_count_1_2_train_vocab"])[0].family, "bag_of_words")
        with self.assertRaisesRegex(ValueError, "unique"):
            resolve_configs(["minilm_l6_v2", "minilm_l6_v2"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_configs(["missing"])

    def test_bow_vocabulary_is_train_only_and_oov_candidate_is_safe(self) -> None:
        config = REGISTERED_CONFIGS["bow_count_1_2_train_vocab"]
        scorer = SparseCandidateScorer(config).fit(["delivery was quick", "friendly support team"])
        vocabulary_before = dict(scorer.vectorizer.vocabulary_)

        scores, diagnostics = scorer.score(["zebra issue", "delivery issue"], "zebra taxonomy")

        self.assertEqual(vocabulary_before, scorer.vectorizer.vocabulary_)
        self.assertNotIn("zebra", scorer.vectorizer.vocabulary_)
        np.testing.assert_allclose(scores, [0.0, 0.0])
        self.assertTrue(diagnostics["candidate_zero_vector"])
        self.assertEqual(diagnostics["candidate_vector_nnz"], 0)

    def test_bow_scores_known_candidate_overlap(self) -> None:
        config = REGISTERED_CONFIGS["bow_count_1_2_train_vocab"]
        scorer = SparseCandidateScorer(config).fit(
            ["delivery speed was good", "support response was good"]
        )

        scores, diagnostics = scorer.score(
            ["delivery speed", "support response"],
            "delivery speed",
        )

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(diagnostics["candidate_vector_nnz"], 0)
        self.assertFalse(diagnostics["candidate_zero_vector"])

    def test_strict_tfidf_does_not_fit_candidate_text(self) -> None:
        config = REGISTERED_CONFIGS["tfidf_char_3_5_train_vocab"]
        scorer = SparseCandidateScorer(config).fit(["phone support", "email support"])
        vocabulary_before = dict(scorer.vectorizer.vocabulary_)

        scorer.score(["competitor mention"], "competitor")

        self.assertEqual(vocabulary_before, scorer.vectorizer.vocabulary_)

    def test_masked_mean_pool_ignores_padding(self) -> None:
        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])

        pooled = masked_mean_pool(hidden, mask)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
