from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_protocol import (
    build_budgeted_training_manifest,
    build_candidate_table,
    build_taxonomy_eval_grid,
    build_taxonomy_fold_splits,
    build_taxonomy_training_manifest,
    candidate_representation_variants,
    canonical_aspects,
    evaluate_scored_grid,
    pair_identity_hash,
    registered_folds,
    select_strict_seen_threshold,
)
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions


ASPECTS = canonical_aspects()
A1, A2, A3, A4 = ASPECTS[:4]


def tiny_official_frame() -> pd.DataFrame:
    rows = [
        {
            "id": 1,
            "original_split": "train",
            "row_uid": "train:1",
            "text": "a1 and a2",
            "labels": [(A1, "positive"), (A2, "negative")],
        },
        {
            "id": 2,
            "original_split": "train",
            "row_uid": "train:2",
            "text": "a2",
            "labels": [(A2, "positive")],
        },
        {
            "id": 3,
            "original_split": "train",
            "row_uid": "train:3",
            "text": "a3",
            "labels": [(A3, "neutral")],
        },
        {
            "id": 4,
            "original_split": "validation",
            "row_uid": "validation:4",
            "text": "a1",
            "labels": [(A1, "negative")],
        },
        {
            "id": 5,
            "original_split": "validation",
            "row_uid": "validation:5",
            "text": "a2",
            "labels": [(A2, "positive")],
        },
        {
            "id": 6,
            "original_split": "test",
            "row_uid": "test:6",
            "text": "a1 and a3",
            "labels": [(A1, "positive"), (A3, "negative")],
        },
        {
            "id": 7,
            "original_split": "test",
            "row_uid": "test:7",
            "text": "none",
            "labels": [],
        },
    ]
    frame = pd.DataFrame(rows)
    frame["org_index"] = range(1, len(frame) + 1)
    frame["pair_labels"] = frame["labels"].apply(
        lambda labels: [f"{aspect} | {sentiment}" for aspect, sentiment in labels]
    )
    frame["aspect_labels"] = frame["labels"].apply(
        lambda labels: sorted({aspect for aspect, _ in labels})
    )
    frame["industry"] = "synthetic"
    frame["data_source"] = "synthetic"
    return frame


def test_registered_fold_schedule_is_fixed_and_complete() -> None:
    l1 = registered_folds("L1")
    l2 = registered_folds("L2")
    l3 = registered_folds("L3")
    l4 = registered_folds("L4")
    assert len(l1) == len(l2) == len(l3) == 12
    assert len(l4) == 3
    assert l1[0].heldout_aspects == (ASPECTS[0],)
    assert l1[0].evaluation_aspects == (ASPECTS[0],)
    assert l2[0].evaluation_aspects == ASPECTS
    assert l3[0].heldout_aspects == (ASPECTS[0], ASPECTS[1])
    assert l3[-1].heldout_aspects == (ASPECTS[-1], ASPECTS[0])
    assert all(fold.conditions == ("NN", "DN", "ND", "DD") for fold in l3)
    appearances = {
        aspect: sum(aspect in fold.heldout_aspects for fold in l3)
        for aspect in ASPECTS
    }
    assert set(appearances.values()) == {2}


def test_splits_use_example_filtering_and_all_official_eval_rows() -> None:
    frame = tiny_official_frame()
    fold = registered_folds("L1")[0]
    splits = build_taxonomy_fold_splits(frame, fold)
    assert splits["train"]["row_uid"].tolist() == ["train:2", "train:3"]
    assert set(splits["validation"]["row_uid"]) == {"validation:4", "validation:5"}
    assert set(splits["test"]["row_uid"]) == {"test:6", "test:7"}
    assert splits["validation"].set_index("row_uid").at[
        "validation:5", "supervision_pair_labels"
    ] == []
    assert all(
        A1 not in {aspect for aspect, _ in labels}
        for labels in splits["train"]["labels"]
    )


def test_l1_and_l2_separate_candidate_scope_from_gold_scope() -> None:
    frame = tiny_official_frame()
    resource = load_minimal_descriptions(require_approved=False)
    l1 = registered_folds("L1")[0]
    l2 = registered_folds("L2")[0]
    l1_splits = build_taxonomy_fold_splits(frame, l1)
    l2_splits = build_taxonomy_fold_splits(frame, l2)
    l1_candidates = build_candidate_table(l1, "D", resource)
    l2_candidates = build_candidate_table(l2, "D", resource)
    l1_grid = build_taxonomy_eval_grid(
        l1_splits["test"], l1_candidates, fold_id=l1.fold_id, condition="D"
    )
    l2_grid = build_taxonomy_eval_grid(
        l2_splits["test"], l2_candidates, fold_id=l2.fold_id, condition="D"
    )
    assert len(l1_candidates) == 3
    assert len(l2_candidates) == 36
    assert len(l1_grid) == 2 * 3
    assert len(l2_grid) == 2 * 36
    assert l1_grid["target"].sum() == 1
    assert l2_grid["target"].sum() == 2


def test_l3_crossover_changes_only_heldout_text_not_pair_identity() -> None:
    frame = tiny_official_frame()
    fold = registered_folds("L3")[0]
    splits = build_taxonomy_fold_splits(frame, fold)
    resource = load_minimal_descriptions(require_approved=False)
    hashes = []
    texts: dict[str, dict[tuple[str, str], str]] = {}
    for condition in fold.conditions:
        candidates = build_candidate_table(fold, condition, resource)
        grid = build_taxonomy_eval_grid(
            splits["validation"],
            candidates,
            fold_id=fold.fold_id,
            condition=condition,
        )
        hashes.append(pair_identity_hash(grid))
        texts[condition] = {
            (row.candidate_aspect, row.candidate_sentiment): row.candidate_text
            for row in candidates.itertuples(index=False)
        }
    assert len(set(hashes)) == 1
    for aspect in fold.seen_aspects:
        assert texts["NN"][(aspect, "positive")] == texts["DD"][(aspect, "positive")]
    first, second = fold.heldout_aspects
    assert texts["DN"][(first, "positive")] == texts["DD"][(first, "positive")]
    assert texts["DN"][(second, "positive")] == texts["NN"][(second, "positive")]
    assert texts["ND"][(first, "positive")] == texts["NN"][(first, "positive")]
    assert texts["ND"][(second, "positive")] == texts["DD"][(second, "positive")]


def test_candidate_variants_keep_seen_descriptions_in_every_condition() -> None:
    fold = registered_folds("L3")[0]
    for condition in fold.conditions:
        variants = candidate_representation_variants(fold, condition)
        assert {variants[aspect] for aspect in fold.seen_aspects} == {"minimal"}
    assert [
        candidate_representation_variants(fold, condition)[fold.heldout_aspects[0]]
        for condition in fold.conditions
    ] == ["name_only", "minimal", "name_only", "minimal"]


def test_training_manifest_excludes_heldout_and_is_deterministic() -> None:
    frame = tiny_official_frame()
    fold = registered_folds("L1")[0]
    splits = build_taxonomy_fold_splits(frame, fold)
    resource = load_minimal_descriptions(require_approved=False)
    first = build_taxonomy_training_manifest(splits["train"], fold, resource)
    second = build_taxonomy_training_manifest(
        splits["train"].sample(frac=1, random_state=7), fold, resource
    )
    assert set(first["candidate_aspect"]).isdisjoint(fold.heldout_aspects)
    assert set(first["representation_variant"]) == {"minimal"}
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(
        ["row_uid", "candidate_aspect", "candidate_sentiment"]
    ).any()
    for positive in first[first["target"] == 1].itertuples(index=False):
        wrong = first[
            (first["row_uid"] == positive.row_uid)
            & (first["candidate_aspect"] == positive.candidate_aspect)
            & (first["candidate_sentiment"] != positive.candidate_sentiment)
        ]
        assert set(wrong["candidate_sentiment"]) == {
            "negative",
            "neutral",
            "positive",
        } - {positive.candidate_sentiment}
    budgeted = build_budgeted_training_manifest(
        splits["train"],
        fold,
        resource,
        total_budget=6,
        positive_budget=3,
    )
    assert len(budgeted) == min(6, len(first))


def test_training_manifest_never_turns_a_second_gold_sentiment_into_a_negative() -> None:
    fold = registered_folds("L1")[0]
    resource = load_minimal_descriptions(require_approved=False)
    frame = pd.DataFrame(
        [
            {
                "row_uid": "train:multi",
                "text": "mixed",
                "supervision_labels": [(A2, "positive"), (A2, "negative")],
            }
        ]
    )
    manifest = build_taxonomy_training_manifest(frame, fold, resource)
    same_aspect = manifest[manifest["candidate_aspect"] == A2].set_index(
        "candidate_sentiment"
    )
    assert same_aspect.at["positive", "target"] == 1
    assert same_aspect.at["negative", "target"] == 1
    assert same_aspect.at["neutral", "target"] == 0
    assert not manifest.groupby(
        ["row_uid", "candidate_aspect", "candidate_sentiment"]
    )["target"].nunique().gt(1).any()


def scored_validation_grid() -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    seen = (A2,)
    heldout = (A1,)
    rows = []
    gold = {
        "r1": f"{A2} | positive",
        "r2": f"{A1} | negative",
        "r3": None,
    }
    for row_uid in gold:
        for aspect in (A1, A2):
            for sentiment in ("negative", "neutral", "positive"):
                pair = f"{aspect} | {sentiment}"
                target = int(pair == gold[row_uid])
                score = 0.9 if target else 0.1
                if aspect == A1:
                    score = 0.99 if target else 0.98
                rows.append(
                    {
                        "row_uid": row_uid,
                        "candidate_aspect": aspect,
                        "candidate_sentiment": sentiment,
                        "pair_label": pair,
                        "target": target,
                        "score": score,
                    }
                )
    return pd.DataFrame(rows), seen, heldout


def test_strict_threshold_ignores_heldout_scores_and_labels() -> None:
    frame, seen, heldout = scored_validation_grid()
    first = select_strict_seen_threshold(
        frame, seen_aspects=seen, heldout_aspects=heldout
    )
    changed = frame.copy()
    mask = changed["candidate_aspect"] == A1
    changed.loc[mask, "score"] = [0.01, 0.02, 0.03] * 3
    changed.loc[mask, "target"] = 1 - changed.loc[mask, "target"]
    second = select_strict_seen_threshold(
        changed, seen_aspects=seen, heldout_aspects=heldout
    )
    assert first.threshold == second.threshold
    pd.testing.assert_frame_equal(first.sweep, second.sweep)
    assert first.calibration_aspects == seen


def test_partition_metrics_and_harmonic_mean_use_common_threshold() -> None:
    frame, seen, heldout = scored_validation_grid()
    result = evaluate_scored_grid(
        frame,
        0.5,
        seen_aspects=seen,
        heldout_aspects=heldout,
    )
    assert result["overall"]["pair_micro_f1"] == pytest.approx(1 / 3)
    assert result["seen"]["pair_micro_f1"] == pytest.approx(1.0)
    assert result["unseen"]["pair_micro_f1"] == pytest.approx(0.2)
    assert result["seen_unseen_harmonic_pair_micro_f1"] == pytest.approx(1 / 3)
