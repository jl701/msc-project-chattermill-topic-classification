from __future__ import annotations

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_scientific_freeze import (
    pair_micro_f1_from_counts,
    row_pair_confusions,
)


def test_row_pair_confusions_are_exact_and_partitioned() -> None:
    frame = pd.DataFrame(
        {
            "row_uid": ["r1", "r1", "r2", "r2"],
            "candidate_aspect": ["A", "A", "A", "A"],
            "candidate_sentiment": ["negative", "positive"] * 2,
            "pair_label": ["A#negative", "A#positive"] * 2,
            "target": [1, 0, 0, 1],
        }
    )
    evidence = row_pair_confusions(
        frame,
        [True, True, False, False],
        aspects=["A"],
        method_id="m",
        fold_id="f",
        condition="D",
    )
    assert evidence[["pair_tp", "pair_fp", "pair_fn"]].to_dict("records") == [
        {"pair_tp": 1, "pair_fp": 1, "pair_fn": 0},
        {"pair_tp": 0, "pair_fp": 0, "pair_fn": 1},
    ]
    assert float(
        pair_micro_f1_from_counts(
            evidence["pair_tp"].sum(),
            evidence["pair_fp"].sum(),
            evidence["pair_fn"].sum(),
        )
    ) == 0.5


def test_pair_micro_f1_zero_denominator_is_zero() -> None:
    observed = pair_micro_f1_from_counts(
        np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([0, 0])
    )
    np.testing.assert_allclose(observed, [0.0, 2.0 / 3.0])
