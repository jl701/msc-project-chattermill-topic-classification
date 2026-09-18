from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyse_local_qwen_loao_cascade import build_pareto_rows, choose_predictions, policy_grid, qwen_call_count


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

    def test_asymmetric_policy_rescues_near_below_threshold(self) -> None:
        local_row = {
            "score_features": {
                "candidate_aspect_scores": {"Company brand: Competitor": 0.35},
                "threshold": 0.37,
            }
        }
        qwen = ["Company brand: Competitor | positive"]

        self.assertEqual(
            choose_predictions(
                [],
                qwen,
                "score_asym_rescue_le_0.05_confirm_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            qwen,
        )
        self.assertEqual(
            qwen_call_count(
                [],
                "score_asym_rescue_le_0.05_confirm_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            1,
        )

    def test_asymmetric_confirm_uses_qwen_for_weak_positive(self) -> None:
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
                "score_asym_rescue_le_0.05_confirm_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            qwen,
        )
        self.assertEqual(
            choose_predictions(
                local,
                [],
                "score_asym_rescue_le_0.05_confirm_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            [],
        )

    def test_asymmetric_veto_preserves_local_sentiment_when_qwen_confirms_aspect(self) -> None:
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
                "score_asym_rescue_le_0.05_veto_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            local,
        )
        self.assertEqual(
            choose_predictions(
                local,
                [],
                "score_asym_rescue_le_0.05_veto_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            [],
        )

    def test_asymmetric_policy_keeps_local_far_above_threshold(self) -> None:
        local_row = {
            "score_features": {
                "candidate_aspect_scores": {"Company brand: Competitor": 0.80},
                "threshold": 0.37,
            }
        }
        local = ["Company brand: Competitor | positive"]
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(
            choose_predictions(
                local,
                qwen,
                "score_asym_rescue_le_0.05_confirm_le_0.02",
                local_row=local_row,
                aspect="Company brand: Competitor",
            ),
            local,
        )
        self.assertEqual(
            qwen_call_count(
                local,
                "score_asym_rescue_le_0.05_confirm_le_0.02",
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
        self.assertIn("score_asym_rescue_le_0.05_confirm_le_0.02", expanded)
        self.assertIn("score_asym_rescue_le_0.05_veto_le_0.02", expanded)
        self.assertIn("sentiment_margin_confirm_le_0.20", expanded)

    def test_build_pareto_rows_marks_dominated_policy(self) -> None:
        aggregate_rows = [
            {
                "split": "validation",
                "policy": "local_only",
                "aspects": 12,
                "pair_micro_f1_mean": 0.30,
                "pair_samples_f1_mean": 0.05,
                "pair_micro_precision_mean": 0.40,
                "pair_micro_recall_mean": 0.25,
                "pair_false_positive_rows_per_100_mean": 10.0,
                "pair_false_negative_rows_per_100_mean": 9.0,
                "qwen_call_rate_mean": 0.0,
            },
            {
                "split": "validation",
                "policy": "score_asym_rescue_le_0.05_confirm_le_0.02",
                "aspects": 12,
                "pair_micro_f1_mean": 0.38,
                "pair_samples_f1_mean": 0.10,
                "pair_micro_precision_mean": 0.36,
                "pair_micro_recall_mean": 0.45,
                "pair_false_positive_rows_per_100_mean": 14.0,
                "pair_false_negative_rows_per_100_mean": 5.0,
                "qwen_call_rate_mean": 0.20,
            },
            {
                "split": "validation",
                "policy": "score_abs_replace_le_0.08",
                "aspects": 12,
                "pair_micro_f1_mean": 0.37,
                "pair_samples_f1_mean": 0.09,
                "pair_micro_precision_mean": 0.34,
                "pair_micro_recall_mean": 0.45,
                "pair_false_positive_rows_per_100_mean": 16.0,
                "pair_false_negative_rows_per_100_mean": 5.2,
                "qwen_call_rate_mean": 0.25,
            },
        ]

        rows = build_pareto_rows(aggregate_rows, "score_asym_rescue_le_0.05_confirm_le_0.02")
        by_policy = {row["policy"]: row for row in rows}

        self.assertEqual(by_policy["local_only"]["is_pareto_frontier"], 1)
        self.assertEqual(
            by_policy["score_asym_rescue_le_0.05_confirm_le_0.02"]["is_global_validation_selected"],
            1,
        )
        self.assertEqual(
            by_policy["score_asym_rescue_le_0.05_confirm_le_0.02"]["policy_family"],
            "asymmetric_rescue_confirm",
        )
        self.assertEqual(by_policy["score_abs_replace_le_0.08"]["is_pareto_frontier"], 0)
        self.assertEqual(by_policy["score_asym_rescue_le_0.05_confirm_le_0.02"]["precision_mean"], 0.36)


if __name__ == "__main__":
    unittest.main()
