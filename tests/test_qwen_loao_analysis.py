from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyse_qwen_loao_predictions import aggregate_spread, positive_gold_result


class QwenLoaoAnalysisTest(unittest.TestCase):
    def test_positive_gold_result_filters_empty_gold_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_path = Path(tmpdir) / "predictions.jsonl"
            rows = [
                {
                    "gold_pair_labels": ["A | positive"],
                    "pred_pair_labels": ["A | positive"],
                },
                {
                    "gold_pair_labels": [],
                    "pred_pair_labels": ["A | negative"],
                },
            ]
            with predictions_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            result = positive_gold_result(
                {
                    "split": "test",
                    "heldout_aspect": "A",
                    "predictions_file": str(predictions_path),
                }
            )

        self.assertEqual(result["positive_rows"], 1)
        self.assertAlmostEqual(result["pair_micro_f1"], 1.0)
        self.assertEqual(result["pair_false_positive_rows"], 0)

    def test_aggregate_spread_summarises_by_split(self) -> None:
        spread = aggregate_spread(
            [
                {
                    "split": "test",
                    "heldout_aspect": "A",
                    "pair_samples_f1": 0.5,
                    "pair_micro_f1": 0.4,
                    "pair_micro_precision": 0.8,
                    "pair_micro_recall": 0.3,
                    "pair_macro_f1": 0.2,
                    "sentiment_accuracy_when_gold_aspect_predicted": 1.0,
                },
                {
                    "split": "test",
                    "heldout_aspect": "B",
                    "pair_samples_f1": 1.0,
                    "pair_micro_f1": 0.8,
                    "pair_micro_precision": 1.0,
                    "pair_micro_recall": 0.7,
                    "pair_macro_f1": 0.6,
                    "sentiment_accuracy_when_gold_aspect_predicted": 0.5,
                },
            ]
        )

        self.assertEqual(spread[0]["aspects"], 2)
        self.assertAlmostEqual(spread[0]["pair_micro_f1_mean"], 0.6)
        self.assertAlmostEqual(spread[0]["pair_micro_f1_min"], 0.4)
        self.assertAlmostEqual(spread[0]["pair_micro_f1_max"], 0.8)


if __name__ == "__main__":
    unittest.main()
