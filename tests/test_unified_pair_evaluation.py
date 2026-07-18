from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msc_project.experiments.unified_pair_evaluation import (
    candidate_pair_classes,
    paired_aspect_statistics,
    pilot_gate,
    score_matrix_from_grid,
    select_threshold,
)


ASPECT = "Value: Price value for money"


def test_score_grid_alignment_is_explicit_and_multi_sentiment_safe() -> None:
    grid = pd.DataFrame(
        [
            {"row_index": 0, "candidate_sentiment": "negative"},
            {"row_index": 0, "candidate_sentiment": "neutral"},
            {"row_index": 0, "candidate_sentiment": "positive"},
            {"row_index": 1, "candidate_sentiment": "negative"},
            {"row_index": 1, "candidate_sentiment": "neutral"},
            {"row_index": 1, "candidate_sentiment": "positive"},
        ]
    )
    matrix = score_matrix_from_grid(grid, [0.9, 0.1, 0.8, 0.2, 0.3, 0.4], row_count=2)
    assert matrix.tolist() == [[0.9, 0.1, 0.8], [0.2, 0.3, 0.4]]


def test_threshold_selection_can_emit_two_sentiments_for_one_aspect() -> None:
    classes = candidate_pair_classes(ASPECT)
    true = [[classes[0], classes[2]], [], [classes[1]]]
    scores = np.asarray([[0.9, 0.1, 0.8], [0.1, 0.2, 0.1], [0.1, 0.7, 0.2]])
    selected = select_threshold(true, scores, ASPECT)
    assert selected.metrics["pair_micro_f1"] == pytest.approx(1.0)
    assert selected.metrics["presence_f1"] == pytest.approx(1.0)
    assert selected.metrics["conditional_sentiment_exact_match"] == pytest.approx(1.0)


def test_paired_statistics_report_wins_and_exact_sign_flip() -> None:
    result = paired_aspect_statistics([0.5, 0.6, 0.7], [0.4, 0.5, 0.6], bootstrap_samples=1000)
    assert result["mean_delta"] == pytest.approx(0.1)
    assert result["wins"] == 3
    assert result["losses"] == 0
    assert 0.0 <= result["exact_or_monte_carlo_sign_flip_p"] <= 1.0


def test_pilot_gate_requires_delta_wins_and_recall_floor() -> None:
    control = pd.DataFrame(
        {
            "heldout_aspect": ["a", "b", "c"],
            "pair_micro_f1": [0.2, 0.3, 0.4],
            "pair_micro_recall": [0.4, 0.5, 0.6],
        }
    )
    enhanced = pd.DataFrame(
        {
            "heldout_aspect": ["a", "b", "c"],
            "pair_micro_f1": [0.24, 0.34, 0.39],
            "pair_micro_recall": [0.3, 0.4, 0.4],
        }
    )
    result = pilot_gate(enhanced, control)
    assert result["passed"] is True
    assert result["wins"] == 2


def test_score_grid_rejects_duplicate_pair() -> None:
    grid = pd.DataFrame(
        [
            {"row_index": 0, "candidate_sentiment": "negative"},
            {"row_index": 0, "candidate_sentiment": "negative"},
            {"row_index": 0, "candidate_sentiment": "positive"},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        score_matrix_from_grid(grid, [0.1, 0.2, 0.3], row_count=1)
