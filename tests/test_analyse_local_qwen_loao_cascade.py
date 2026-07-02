from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyse_local_qwen_loao_cascade import choose_predictions, policy_grid, qwen_call_count


class AnalyseLocalQwenLoaoCascadeTests(unittest.TestCase):
    def test_choose_predictions_keeps_local_when_local_nonempty_else_qwen(self) -> None:
        local = ["Company brand: Competitor | positive"]
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(choose_predictions(local, qwen, "local_nonempty_else_qwen"), local)

    def test_choose_predictions_uses_qwen_when_local_empty(self) -> None:
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(choose_predictions([], qwen, "local_nonempty_else_qwen"), qwen)

    def test_pair_agreement_requires_same_sentiment(self) -> None:
        local = ["Company brand: Competitor | positive"]
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(choose_predictions(local, qwen, "pair_agreement"), [])

    def test_aspect_agreement_can_keep_local_sentiment(self) -> None:
        local = ["Company brand: Competitor | positive"]
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(choose_predictions(local, qwen, "aspect_agreement_local_sentiment"), local)

    def test_qwen_call_count_matches_gate_policy(self) -> None:
        self.assertEqual(qwen_call_count([], "local_only"), 0)
        self.assertEqual(qwen_call_count([], "qwen_only"), 1)
        self.assertEqual(qwen_call_count([], "local_nonempty_else_qwen"), 1)
        self.assertEqual(
            qwen_call_count(["Company brand: Competitor | positive"], "local_nonempty_else_qwen"),
            0,
        )
        self.assertEqual(qwen_call_count([], "aspect_agreement_local_sentiment"), 0)
        self.assertEqual(
            qwen_call_count(["Company brand: Competitor | positive"], "aspect_agreement_local_sentiment"),
            1,
        )

    def test_score_abs_replace_uses_qwen_near_threshold(self) -> None:
        local_row = {
            "score_features": {
                "candidate_aspect_scores": {"Company brand: Competitor": 0.38},
                "threshold": 0.37,
            }
        }
        local = ["Company brand: Competitor | positive"]
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(
            choose_predictions(
                local,
                qwen,
                "score_abs_replace_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            qwen,
        )
        self.assertEqual(
            qwen_call_count(
                local,
                "score_abs_replace_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            1,
        )

    def test_score_below_rescue_keeps_local_when_far_from_threshold(self) -> None:
        local_row = {
            "score_features": {
                "candidate_aspect_scores": {"Company brand: Competitor": 0.10},
                "threshold": 0.37,
            }
        }
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(
            choose_predictions(
                [],
                qwen,
                "score_below_rescue_le_0.05",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            [],
        )
        self.assertEqual(
            qwen_call_count(
                [],
                "score_below_rescue_le_0.05",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            0,
        )

    def test_policy_grid_adds_feature_policies_only_when_available(self) -> None:
        base = policy_grid({"score_features": False, "sentiment_features": False})
        expanded = policy_grid({"score_features": True, "sentiment_features": True})

        self.assertIn("local_only", base)
        self.assertNotIn("score_abs_replace_le_0.05", base)
        self.assertIn("score_abs_replace_le_0.05", expanded)
        self.assertIn("sentiment_margin_confirm_le_0.20", expanded)


if __name__ == "__main__":
    unittest.main()
