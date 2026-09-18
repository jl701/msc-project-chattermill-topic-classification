from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_aspect_label_aware_baseline import selection_keys
from run_generalisation_baselines import selection_columns


class SelectionMetricTest(unittest.TestCase):
    def test_selection_columns_promote_primary_metric(self) -> None:
        self.assertEqual(
            selection_columns("pair_micro_f1"),
            ["pair_micro_f1", "pair_samples_f1", "pair_macro_f1"],
        )

    def test_selection_keys_promote_primary_metric(self) -> None:
        self.assertEqual(
            selection_keys("pair_macro_f1"),
            ("pair_macro_f1", "pair_samples_f1", "pair_micro_f1"),
        )

    def test_unknown_selection_metric_raises(self) -> None:
        with self.assertRaises(ValueError):
            selection_columns("accuracy")
        with self.assertRaises(ValueError):
            selection_keys("accuracy")


if __name__ == "__main__":
    unittest.main()
