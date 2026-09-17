from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msc_project.data.fabsa import format_pair_label
from msc_project.experiments.taxonomy_global_router import (
    build_component_arrays,
    policy_metrics,
    route_mask,
    routed_score_frame,
    threshold_aligned_scores,
    validate_component_alignment,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS

ASPECTS = ("Group: A", "Group: B")


def _component_frame(
    method_id: str,
    *,
    aspect_scores: tuple[float, float] = (0.49, 0.90),
    selected_sentiments: tuple[str, str] = ("negative", "positive"),
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for aspect_index, aspect in enumerate(ASPECTS):
        for sentiment in CANDIDATE_SENTIMENTS:
            selected = sentiment == selected_sentiments[aspect_index]
            records.append(
                {
                    "row_uid": "validation:000",
                    "candidate_aspect": aspect,
                    "candidate_sentiment": sentiment,
                    "pair_label": format_pair_label(aspect, sentiment),
                    "target": int(
                        aspect == "Group: A" and sentiment == "positive"
                    ),
                    "aspect_score": aspect_scores[aspect_index],
                    "sentiment_score": 0.9 if selected else 0.1,
                    "method_id": method_id,
                    "split": "validation",
                    "is_heldout": aspect == "Group: A",
                }
            )
    return pd.DataFrame.from_records(records)


@pytest.mark.parametrize("threshold", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_threshold_alignment_preserves_binary_decisions(threshold: float) -> None:
    values = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
    aligned = threshold_aligned_scores(values, threshold)
    np.testing.assert_array_equal(aligned >= 0.5, values >= threshold)
    assert np.all(np.diff(aligned) >= 0.0)
    assert np.isfinite(aligned).all()


def test_build_component_alignment_preserves_capped_two_decoder() -> None:
    frame = _component_frame("base")
    frame.loc[
        frame["candidate_sentiment"].isin(["negative", "positive"]),
        "sentiment_score",
    ] = [0.91, 0.85, 0.90, 0.84]
    component = build_component_arrays(
        frame,
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.8,
    )
    assert component.prediction.shape == (2, 3)
    assert int(component.prediction[1].sum()) == 2
    assert int(component.prediction.max(axis=1).sum()) == 1


def test_router_rescues_uncertain_absence_and_retains_confident_presence() -> None:
    base = build_component_arrays(
        _component_frame("base"),
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.8,
    )
    expert = build_component_arrays(
        _component_frame(
            "expert",
            aspect_scores=(0.90, 0.10),
            selected_sentiments=("positive", "negative"),
        ),
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.8,
    )
    routed, rescued, confirmed = route_mask(
        base,
        rescue_band=0.025,
        confirmation_band=0.025,
    )
    np.testing.assert_array_equal(routed, [True, False])
    np.testing.assert_array_equal(rescued, [True, False])
    np.testing.assert_array_equal(confirmed, [False, False])

    metrics = policy_metrics(
        base,
        expert,
        rescue_band=0.025,
        confirmation_band=0.025,
    )
    assert metrics["heldout_pair_micro_f1"] == 1.0
    assert metrics["route_count"] == 1


def test_routed_frame_reproduces_hard_decisions_and_capped_two_contract() -> None:
    base = build_component_arrays(
        _component_frame("base"),
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.8,
    )
    expert_frame = _component_frame(
        "expert",
        aspect_scores=(0.90, 0.90),
        selected_sentiments=("positive", "negative"),
    )
    expert_frame.loc[
        expert_frame["candidate_sentiment"].isin(["negative", "positive"]),
        "sentiment_score",
    ] = [0.91, 0.85, 0.90, 0.84]
    expert = build_component_arrays(
        expert_frame,
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.8,
    )
    routed, prediction, route_frame = routed_score_frame(
        base,
        expert,
        rescue_band=0.025,
        confirmation_band=-1.0,
        method_id="router",
    )
    counts = routed.loc[prediction].groupby(
        ["row_uid", "candidate_aspect"]
    ).size()
    assert int(counts.max()) <= 2
    assert int(route_frame["routed"].sum()) == 1
    assert set(routed["method_id"]) == {"router"}


def test_component_alignment_rejects_mismatched_targets() -> None:
    left = build_component_arrays(
        _component_frame("left"),
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.8,
    )
    changed = _component_frame("right")
    changed.loc[0, "target"] = 1
    right = build_component_arrays(
        changed,
        aspect_threshold=0.5,
        runner_up_sentiment_threshold=0.8,
    )
    with pytest.raises(ValueError, match="targets differ"):
        validate_component_alignment(left, right)
