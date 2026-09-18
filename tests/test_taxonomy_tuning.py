from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_tuning import (
    expected_tuning_folds,
    fixed_registered_selection,
    registered_tuning_candidates,
    select_registered_tuning_candidate,
)


def observations(method_id: str) -> pd.DataFrame:
    rows = []
    candidates = registered_tuning_candidates(method_id)
    folds = expected_tuning_folds(method_id) or ("single",)
    for candidate_index, candidate in enumerate(candidates):
        for fold in folds:
            rows.append(
                {
                    "method_id": method_id,
                    "candidate_id": candidate["candidate_id"],
                    "parameters_sha256": candidate["parameters_sha256"],
                    "fold_id": fold,
                    "calibration_scope": "seen_aspects_only",
                    "pair_micro_f1": 0.4 + candidate_index / 1000,
                    "pair_samples_f1": 0.3,
                    "pair_micro_precision": 0.2,
                    "presence_false_positive_rows_per_100": 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_registered_grids_have_expected_finite_sizes() -> None:
    assert len(registered_tuning_candidates("strict_train_only_tfidf")) == 15
    assert len(
        registered_tuning_candidates(
            "distilbert_review_candidate_cross_encoder"
        )
    ) == 3
    assert len(registered_tuning_candidates("qwen_candidate_pair_qlora")) == 3
    assert len(registered_tuning_candidates("e5_base_v2")) == 1
    assert len(registered_tuning_candidates("frozen_qwen_candidate_pair")) == 1


def test_selection_requires_complete_registered_seen_only_grid() -> None:
    method = "strict_train_only_tfidf"
    frame = observations(method)
    result = select_registered_tuning_candidate(frame, method)
    assert result["registered_candidates"] == 15
    assert result["selected_candidate_id"] == frame["candidate_id"].iloc[-1]
    assert result["adaptive_grid_expansion_permitted"] is False

    with pytest.raises(ValueError, match="exact registered"):
        select_registered_tuning_candidate(
            frame[frame["candidate_id"] != frame["candidate_id"].iloc[0]],
            method,
        )
    changed = frame.copy()
    changed.loc[0, "calibration_scope"] = "target_aspect"
    with pytest.raises(ValueError, match="non-strict"):
        select_registered_tuning_candidate(changed, method)


def test_tie_break_prefers_lower_complexity() -> None:
    method = "strict_train_only_tfidf"
    frame = observations(method)
    for metric in (
        "pair_micro_f1",
        "pair_samples_f1",
        "pair_micro_precision",
        "presence_false_positive_rows_per_100",
    ):
        frame[metric] = 0.5
    result = select_registered_tuning_candidate(frame, method)
    assert result["selected_parameters"]["feature_ablation"] == "char_cosine_and_cues"
    assert result["selected_parameters"]["classifier_c"] == pytest.approx(0.1)


def test_fixed_selection_is_only_available_without_a_tuning_grid() -> None:
    result = fixed_registered_selection("e5_base_v2")
    assert result["selection_scope"] == "registered_fixed_recipe"
    assert result["registered_candidates"] == 1
    assert result["selected_mean_metrics"] is None

    with pytest.raises(ValueError, match="single-candidate"):
        fixed_registered_selection("strict_train_only_tfidf")
