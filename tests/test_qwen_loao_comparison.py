from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyse_qwen_loao_comparison import aggregate_metric, aggregate_row, compare_per_aspect, positive_gap_rows


class QwenLoaoComparisonTest(unittest.TestCase):
    def test_aggregate_row_adds_spread_and_weighted_seconds(self) -> None:
        rows = [
            {
                "heldout_aspect": "A",
                "pair_micro_f1": 0.2,
                "seconds": 2.0,
                "examples": 2,
            },
            {
                "heldout_aspect": "B",
                "pair_micro_f1": 0.6,
                "seconds": 6.0,
                "examples": 2,
            },
        ]

        metric = aggregate_metric(rows, "pair_micro_f1")
        row = aggregate_row("system", "test", rows)

        self.assertAlmostEqual(metric["mean"], 0.4)
        self.assertAlmostEqual(metric["spread"], 0.4)
        self.assertEqual(row["aspects"], 2)
        self.assertAlmostEqual(row["pair_micro_f1_mean"], 0.4)
        self.assertAlmostEqual(row["weighted_seconds_per_example"], 2.0)

    def test_compare_per_aspect_computes_qwen_minus_distilbert(self) -> None:
        qwen_rows = [
            {
                "heldout_aspect": "A",
                "pair_micro_f1": 0.7,
                "pair_micro_precision": 0.5,
                "pair_micro_recall": 1.0,
            }
        ]
        distilbert_rows = [
            {
                "heldout_aspect": "A",
                "pair_micro_f1": 0.4,
                "pair_micro_precision": 0.8,
                "pair_micro_recall": 0.3,
            }
        ]

        rows = compare_per_aspect(qwen_rows, distilbert_rows)

        self.assertEqual(rows[0]["winner_pair_micro"], "Qwen")
        self.assertAlmostEqual(rows[0]["delta_pair_micro_f1"], 0.3)
        self.assertAlmostEqual(rows[0]["delta_pair_micro_precision"], -0.3)
        self.assertAlmostEqual(rows[0]["delta_pair_micro_recall"], 0.7)

    def test_positive_gap_rows_computes_positive_minus_all_row(self) -> None:
        qwen_rows = [
            {
                "heldout_aspect": "A",
                "pair_micro_f1": 0.2,
                "pair_micro_precision": 0.1,
                "pair_false_positive_rows_per_100": 30.0,
                "pair_false_negative_rows_per_100": 2.0,
            }
        ]
        positive_rows = [
            {
                "heldout_aspect": "A",
                "pair_micro_f1": 0.9,
                "pair_micro_precision": 1.0,
            }
        ]

        rows = positive_gap_rows(qwen_rows, positive_rows)

        self.assertAlmostEqual(rows[0]["gap_pair_micro_f1"], 0.7)
        self.assertAlmostEqual(rows[0]["gap_pair_micro_precision"], 0.9)
        self.assertAlmostEqual(rows[0]["all_row_fp_rows_per_100"], 30.0)
        self.assertAlmostEqual(rows[0]["all_row_fn_rows_per_100"], 2.0)


if __name__ == "__main__":
    unittest.main()
