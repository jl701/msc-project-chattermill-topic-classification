from __future__ import annotations

import pandas as pd

from scripts.run_taxonomy_two_stage_precloud_validation import (
    _combine_l3_cached_seen_scores,
    _evaluation_partitions,
    _missing_resume_conditions,
    _pipeline_conditional_sentiment_accuracy,
    _score_hash,
)
from msc_project.experiments.taxonomy_protocol import registered_folds


def _scored() -> pd.DataFrame:
    records = []
    for sentiment, score, target in (
        ("negative", 0.2, 0),
        ("neutral", 0.8, 1),
        ("positive", 0.1, 0),
    ):
        records.append(
            {
                "row_uid": "validation:1",
                "candidate_aspect": "Parent: Alpha",
                "candidate_sentiment": sentiment,
                "representation_variant": "name_and_description",
                "pair_label": f"Parent: Alpha | {sentiment}",
                "target": target,
                "score": 0.9,
                "aspect_score": 0.9,
                "sentiment_score": score,
            }
        )
    return pd.DataFrame.from_records(records)


def test_pipeline_sentiment_metric_is_conditioned_on_selected_gold_aspects() -> None:
    value = _pipeline_conditional_sentiment_accuracy(
        _scored(), aspect_threshold=0.5
    )
    assert value["selected_gold_aspect_instances"] == 1
    assert value["pipeline_conditional_sentiment_accuracy"] == 1.0


def test_seen_score_hash_changes_with_representation_or_score() -> None:
    frame = _scored()
    first = _score_hash(frame)
    frame.loc[0, "aspect_score"] = 0.8
    assert _score_hash(frame) != first


def test_l1_does_not_create_an_empty_seen_partition() -> None:
    fold = registered_folds("L1")[0]
    partitions = _evaluation_partitions(fold)
    assert partitions["seen"] == ()
    assert partitions["overall"] == fold.heldout_aspects
    assert partitions["heldout"] == fold.heldout_aspects


def test_l3_seen_cache_reconstructs_exact_pair_grid() -> None:
    seen = _scored()
    heldout = _scored().assign(candidate_aspect="Parent: Beta")
    full = pd.concat([seen, heldout], ignore_index=True)
    combined = _combine_l3_cached_seen_scores(full, seen, heldout)
    assert len(combined) == len(full)
    duplicate = pd.concat([seen, seen], ignore_index=True)
    try:
        _combine_l3_cached_seen_scores(full, duplicate, heldout)
    except AssertionError as error:
        assert "exact evaluation grid" in str(error)
    else:
        raise AssertionError("Duplicate cached identities must be rejected.")


def test_resume_condition_plan_only_returns_missing_conditions() -> None:
    fold = registered_folds("L3")[0]
    value = {
        "protocol_id": "taxonomy_two_stage_precloud_v2",
        "method_id": "frozen_qwen_candidate_pair",
        "level": "L3",
        "fold_id": fold.fold_id,
        "training_scope_id": f"heldout-a01-a02",
        "conditions": {"NN": {}, "DD": {}},
    }
    assert _missing_resume_conditions(
        value,
        method="frozen_qwen_candidate_pair",
        level="L3",
        fold=fold,
        requested=("NN", "DD", "RR"),
    ) == ("RR",)
