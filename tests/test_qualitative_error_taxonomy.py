from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyse_qualitative_error_taxonomy import fixed_row_analysis, row_without_text


def base_row() -> dict:
    return {
        "row_uid": "test:1",
        "id": "1",
        "text": "A review text that must stay out of no-text exports.",
        "gold_pair_labels": ["Company brand: Competitor | positive"],
        "predictions": {
            "local": [
                "Company brand: Competitor | positive",
                "Account management: Account access | positive",
            ],
            "qwen": ["Account management: Account access | positive"],
            "flash_lite": [
                "Company brand: Competitor | positive",
                "Account management: Account access | positive",
            ],
            "flash": [],
            "pro": [],
            "flash_lite_desc": ["Company brand: Competitor | positive"],
            "flash_lite_boundary": [],
            "flash_desc": [],
            "pro_cascade": ["Company brand: Competitor | positive"],
        },
    }


class QualitativeErrorTaxonomyTest(unittest.TestCase):
    def test_fixed_row_analysis_tags_cross_model_mechanisms(self) -> None:
        analysis = fixed_row_analysis(base_row())

        self.assertIn("competitor_positive_miss", analysis["row_tags"])
        self.assertIn("account_access_overprediction", analysis["row_tags"])
        self.assertIn("pro_empty_cascade_recovery", analysis["row_tags"])
        self.assertIn("description_row_gain", analysis["row_tags"])
        self.assertIn("description_recall_loss", analysis["row_tags"])

    def test_no_text_export_excludes_review_text(self) -> None:
        row = base_row()
        no_text = row_without_text(row, fixed_row_analysis(row))

        self.assertNotIn("text", no_text)
        self.assertEqual(no_text["row_uid"], "test:1")


if __name__ == "__main__":
    unittest.main()
