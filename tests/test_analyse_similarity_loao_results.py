from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyse_similarity_loao_results import (
    METHODS,
    comparison_table,
    exact_sign_flip_p_value,
    holm_adjusted_p_values,
    paired_bootstrap_interval,
)


class SimilarityLoaoAnalysisTests(unittest.TestCase):
    def test_exact_sign_flip_uses_all_two_sided_permutations(self) -> None:
        self.assertEqual(exact_sign_flip_p_value(np.asarray([1.0, 1.0, 1.0])), 0.25)

    def test_constant_paired_difference_has_degenerate_bootstrap_interval(self) -> None:
        lower, upper = paired_bootstrap_interval(
            np.asarray([0.5, 0.5, 0.5]),
            resamples=1_000,
            seed=13,
        )
        self.assertEqual(lower, 0.5)
        self.assertEqual(upper, 0.5)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = holm_adjusted_p_values(np.asarray([0.04, 0.01, 0.03]))
        np.testing.assert_allclose(adjusted, np.asarray([0.06, 0.03, 0.06]))

    def test_comparison_table_aligns_methods_by_aspect(self) -> None:
        rows = []
        for method_index, method in enumerate(METHODS):
            for aspect_index in range(12):
                rows.append(
                    {
                        "method": method,
                        "split": "test",
                        "heldout_aspect": f"aspect-{11 - aspect_index:02d}",
                        "pair_micro_f1": method_index + aspect_index / 100,
                    }
                )
        result = comparison_table(
            pd.DataFrame(reversed(rows)),
            split="test",
            bootstrap_resamples=1_000,
        )

        e5_vs_tfidf = result.iloc[0]
        self.assertAlmostEqual(float(e5_vs_tfidf["mean_paired_difference"]), 2.0)
        self.assertEqual(int(e5_vs_tfidf["challenger_wins"]), 12)
        self.assertEqual(len(result), 6)

    def test_incomplete_registered_method_set_is_rejected(self) -> None:
        rows = [
            {
                "method": METHODS[0],
                "split": "test",
                "heldout_aspect": f"aspect-{index:02d}",
                "pair_micro_f1": 0.1,
            }
            for index in range(12)
        ]
        with self.assertRaisesRegex(ValueError, "exactly 48"):
            comparison_table(pd.DataFrame(rows), split="test", bootstrap_resamples=10)


if __name__ == "__main__":
    unittest.main()
