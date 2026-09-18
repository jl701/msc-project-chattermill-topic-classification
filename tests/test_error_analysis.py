from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.evaluation.error_analysis import count_labels, row_error_record, sample_f1, summarise_records


class ErrorAnalysisTest(unittest.TestCase):
    def test_sample_f1(self) -> None:
        self.assertEqual(sample_f1({"A", "B"}, {"A", "C"}), 0.5)

    def test_row_error_record_detects_sentiment_error(self) -> None:
        record = row_error_record(
            {
                "id": "1",
                "org_index": 1,
                "text": "good app",
                "gold_pair_labels": ["App | positive"],
                "pred_pair_labels": ["App | negative"],
            }
        )
        self.assertEqual(record["aspect_tp"], ["App"])
        self.assertEqual(record["sentiment_error_aspects"], ["App"])
        self.assertIn("sentiment_error", record["error_categories"])

    def test_count_labels(self) -> None:
        records = [
            row_error_record(
                {
                    "gold_pair_labels": ["A | positive"],
                    "pred_pair_labels": ["A | positive", "B | negative"],
                }
            )
        ]
        frame = count_labels(records, "aspect")
        row_a = frame[frame["label"] == "A"].iloc[0]
        row_b = frame[frame["label"] == "B"].iloc[0]
        self.assertEqual(int(row_a["tp"]), 1)
        self.assertEqual(int(row_b["fp"]), 1)

    def test_summarise_records(self) -> None:
        records = [
            row_error_record({"gold_pair_labels": ["A | positive"], "pred_pair_labels": ["A | positive"]}),
            row_error_record({"gold_pair_labels": ["B | positive"], "pred_pair_labels": []}),
        ]
        summary = summarise_records(records)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["exact_rows"], 1)
        self.assertEqual(summary["missed_all_rows"], 1)


if __name__ == "__main__":
    unittest.main()

