from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for value in (SCRIPTS_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import analyse_taxonomy_two_stage_stage_hybrid_calibration as study

from msc_project.experiments.taxonomy_global_router import build_component_arrays
from msc_project.experiments.taxonomy_two_stage import (
    capped_two_sentiment_prediction_mask,
)

SENTIMENTS = ("negative", "neutral", "positive")


def _score_frame(method: str, aspect_scores: list[float]) -> pd.DataFrame:
    rows = []
    aspects = [f"aspect-{index:02d}" for index in range(len(aspect_scores))]
    for review_index in range(2):
        for aspect_index, (aspect, aspect_score) in enumerate(
            zip(aspects, aspect_scores)
        ):
            for sentiment_index, sentiment in enumerate(SENTIMENTS):
                rows.append(
                    {
                        "row_uid": f"validation:{review_index}",
                        "candidate_aspect": aspect,
                        "candidate_sentiment": sentiment,
                        "pair_label": f"{aspect} | {sentiment}",
                        "target": int(
                            review_index == 0
                            and aspect_index == 0
                            and sentiment == "positive"
                        ),
                        "aspect_score": aspect_score,
                        "sentiment_score": (0.1, 0.2, 0.7)[sentiment_index],
                        "method_id": method,
                        "split": "validation",
                        "is_heldout": aspect_index == 0,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_frozen_config_is_validation_only() -> None:
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs/experiments/taxonomy_two_stage_stage_hybrid_calibration_v1.json"
        ).read_text(encoding="utf-8")
    )
    study.validate_config(config)
    config["include_official_test"] = True
    try:
        study.validate_config(config)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsafe test access was not rejected.")


def test_stage_hybrid_retains_stage_one_gate() -> None:
    stage_1 = build_component_arrays(
        _score_frame("stage-1", [0.8, 0.2]),
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.5,
    )
    stage_2_frame = _score_frame("stage-2", [0.1, 0.9])
    stage_2_frame["sentiment_score"] = np.tile([0.6, 0.55, 0.1], 4)
    stage_2 = build_component_arrays(
        stage_2_frame,
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.5,
    )
    hybrid = study.build_stage_hybrid_frame(
        stage_1, stage_2, method_id="hybrid"
    )
    prediction = capped_two_sentiment_prediction_mask(
        hybrid, aspect_threshold=0.5, second_sentiment_threshold=0.5
    ).to_numpy(dtype=bool)
    assert np.array_equal(prediction.reshape(-1, 3).any(axis=1), stage_1.aspect_present)
    assert int(prediction.reshape(-1, 3).sum(axis=1).max()) == 2


def test_review_block_is_deterministic_and_seeded() -> None:
    first = [study.review_block(f"validation:{index}") for index in range(100)]
    second = [study.review_block(f"validation:{index}") for index in range(100)]
    changed = [
        study.review_block(f"validation:{index}", seed=14) for index in range(100)
    ]
    assert first == second
    assert first != changed
    assert set(first) == {0, 1, 2, 3, 4}


def test_relative_features_have_exact_two_feature_shape() -> None:
    scores = [0.05 + 0.07 * index for index in range(12)]
    frame = _score_frame("qlora", scores)
    component = build_component_arrays(
        frame,
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.5,
    )
    features = study.relative_feature_matrix(frame, component, epsilon=1e-6)
    assert features.shape == (24, 2)
    assert np.isfinite(features).all()
    assert not np.allclose(features[:, 1], 0.0)


def test_threshold_selection_uses_registered_lexicographic_rule() -> None:
    probabilities = np.asarray([0.9, 0.8, 0.2, 0.1])
    sentiment_mask = np.asarray(
        [[True, False, False]] * len(probabilities), dtype=bool
    )
    target = np.asarray(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )
    selected = study.select_calibrated_threshold(
        probabilities,
        sentiment_mask,
        target,
        np.ones(4, dtype=bool),
        [0.0, 0.5, 1.0],
    )
    assert selected["threshold"] == 0.5
    assert selected["pair_micro_f1"] == 1.0
