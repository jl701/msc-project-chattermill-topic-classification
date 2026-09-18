from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_taxonomy_level1_closed_reference import build_score_grid


def test_build_level1_score_grid_has_unique_36_label_identities() -> None:
    classes = [f"aspect-{index // 3} | sentiment-{index % 3}" for index in range(36)]
    targets = np.zeros((2, 36), dtype=int)
    scores = np.full((2, 36), 0.25, dtype=float)
    grid = build_score_grid(pd.Series(["validation:1", "validation:2"]), classes, targets, scores)
    assert len(grid) == 72
    assert not grid.duplicated(
        ["row_uid", "candidate_aspect", "candidate_sentiment"]
    ).any()
    assert np.isfinite(grid["score"]).all()
