from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_qwen_loao_heldout_aspect import aggregate_spread, existing_predictions_are_usable


class QwenLoaoRunnerTest(unittest.TestCase):
    def test_existing_predictions_must_match_prefix_row_ids(self) -> None:
        eval_df = pd.DataFrame({"id": ["r1", "r2", "r3"]})
        rows = [
            {"row_index": 0, "id": "r1", "pred_pair_labels": []},
            {"row_index": 1, "id": "r2", "pred_pair_labels": ["A | positive"]},
        ]

        self.assertTrue(existing_predictions_are_usable(rows, eval_df))
        self.assertFalse(existing_predictions_are_usable([{**rows[0], "id": "wrong"}], eval_df))
        self.assertFalse(existing_predictions_are_usable([{**rows[0], "row_index": 1}], eval_df))

    def test_aggregate_spread_groups_by_split_and_prompt(self) -> None:
        spread = aggregate_spread(
            [
                {
                    "heldout_aspect": "A",
                    "split": "validation",
                    "prompt_variant": "indexed",
                    "pair_micro_f1": 0.2,
                    "valid_json_rate": 1.0,
                    "seconds_per_example": 1.5,
                },
                {
                    "heldout_aspect": "B",
                    "split": "validation",
                    "prompt_variant": "indexed",
                    "pair_micro_f1": 0.6,
                    "valid_json_rate": 0.5,
                    "seconds_per_example": 2.5,
                },
            ]
        )

        row = spread.iloc[0].to_dict()
        self.assertEqual(row["aspects"], 2)
        self.assertAlmostEqual(row["pair_micro_f1_mean"], 0.4)
        self.assertAlmostEqual(row["pair_micro_f1_std"], 0.2)
        self.assertAlmostEqual(row["pair_micro_f1_min"], 0.2)
        self.assertAlmostEqual(row["pair_micro_f1_max"], 0.6)
        self.assertAlmostEqual(row["valid_json_rate_mean"], 0.75)
        self.assertAlmostEqual(row["seconds_per_example_mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
