from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_pipeline import (
    assert_condition_pair_identity,
    build_run_contract,
    prepare_fold_evaluation,
    prepare_l1_strict_calibration,
    prepare_strict_seen_calibration,
    prepare_fold_training,
    score_prepared_shard,
    score_prepared_conditions_shard,
    unique_rendered_claim_count,
)
from msc_project.experiments.taxonomy_protocol import registered_folds
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions
from msc_project.experiments.unified_candidate_pairs import manifest_hash


def synthetic_frame() -> pd.DataFrame:
    fold = registered_folds("L3")[0]
    seen = fold.seen_aspects[0]
    rows = [
        ("train", 1, [(seen, "positive")]),
        ("train", 2, [(seen, "negative")]),
        ("validation", 3, [(fold.heldout_aspects[0], "positive")]),
        ("validation", 4, []),
    ]
    frame = pd.DataFrame(
        [
            {
                "id": index,
                "original_split": split,
                "row_uid": f"{split}:{index}",
                "text": f"synthetic {index}",
                "labels": labels,
                "org_index": index,
                "industry": "synthetic",
                "data_source": "synthetic",
            }
            for split, index, labels in rows
        ]
    )
    frame["pair_labels"] = frame["labels"].apply(
        lambda labels: [f"{aspect} | {sentiment}" for aspect, sentiment in labels]
    )
    frame["aspect_labels"] = frame["labels"].apply(
        lambda labels: sorted({aspect for aspect, _ in labels})
    )
    return frame


class DeterministicRuntime:
    method_id = "strict_train_only_tfidf"

    def fit(self, train_manifest):
        return self

    def score(self, pair_manifest):
        return np.asarray(
            [
                (index + 1) / (len(pair_manifest) + 1)
                for index in range(len(pair_manifest))
            ]
        )

    def close(self):
        return None


def prepared_conditions():
    frame = synthetic_frame()
    fold = registered_folds("L3")[0]
    resource = load_minimal_descriptions(require_approved=False)
    return [
        prepare_fold_evaluation(
            frame,
            fold,
            condition,
            "validation",
            resource,
            total_budget=16,
            positive_budget=8,
        )
        for condition in fold.conditions
    ]


def test_prepared_l3_conditions_share_training_and_evaluation_identities() -> None:
    prepared = prepared_conditions()
    assert len(assert_condition_pair_identity(prepared)) == 64
    assert {
        tuple(value.training_manifest["candidate_aspect"].unique())
        for value in prepared
    }


def test_training_preparation_needs_only_the_official_train_split() -> None:
    frame = synthetic_frame()
    train_only = frame[frame["original_split"] == "train"].copy()
    fold = registered_folds("L3")[0]
    resource = load_minimal_descriptions(require_approved=False)
    split, manifest = prepare_fold_training(
        train_only,
        fold,
        resource,
        total_budget=16,
        positive_budget=8,
    )
    assert set(split["original_split"]) == {"train"}
    assert not manifest.empty
    assert not set(manifest["candidate_aspect"]) & set(fold.heldout_aspects)


def test_l1_strict_calibration_uses_seen_gold_on_a_matched_l2_grid() -> None:
    frame = synthetic_frame()
    fold = registered_folds("L1")[0]
    resource = load_minimal_descriptions(require_approved=False)
    target = prepare_fold_evaluation(
        frame,
        fold,
        "D",
        "validation",
        resource,
        total_budget=16,
        positive_budget=8,
    )
    calibration = prepare_l1_strict_calibration(
        frame,
        fold,
        resource,
        total_budget=16,
        positive_budget=8,
    )
    assert set(target.evaluation_grid["candidate_aspect"]) == set(
        fold.heldout_aspects
    )
    assert set(calibration.evaluation_grid["candidate_aspect"]) == set(
        fold.seen_aspects
    )
    assert not set(calibration.evaluation_grid["candidate_aspect"]) & set(
        fold.heldout_aspects
    )
    assert manifest_hash(target.training_manifest) == manifest_hash(
        calibration.training_manifest
    )


def test_every_level_strict_calibration_excludes_heldout_candidates() -> None:
    frame = synthetic_frame()
    resource = load_minimal_descriptions(require_approved=False)
    for level in ("L1", "L2", "L3", "L4"):
        fold = registered_folds(level)[0]
        calibration = prepare_strict_seen_calibration(frame, fold, resource)
        observed = set(calibration.evaluation_grid["candidate_aspect"])
        assert observed == set(fold.seen_aspects)
        assert not observed & set(fold.heldout_aspects)


def test_pipeline_builds_hashed_contract_scores_and_resumes(tmp_path: Path) -> None:
    prepared = prepared_conditions()[0]
    run = build_run_contract(
        prepared,
        "strict_train_only_tfidf",
        {"classifier_c": 1.0},
        shard_count=1,
        formal=False,
    )
    runtime = DeterministicRuntime().fit(prepared.training_manifest)
    first, first_state = score_prepared_shard(
        prepared,
        runtime,
        run,
        tmp_path,
        shard_index=0,
        resume=False,
    )
    second, second_state = score_prepared_shard(
        prepared,
        runtime,
        run,
        tmp_path,
        shard_index=0,
        resume=True,
    )
    assert first_state == "scored"
    assert second_state == "resumed"
    pd.testing.assert_frame_equal(first, second)


def test_condition_identity_guard_rejects_changed_training_pairs() -> None:
    prepared = prepared_conditions()
    changed_manifest = prepared[1].training_manifest.copy()
    changed_manifest.loc[changed_manifest.index[0], "target"] ^= 1
    changed = prepared[1].__class__(
        **{
            **prepared[1].__dict__,
            "training_manifest": changed_manifest,
        }
    )
    with pytest.raises(ValueError, match="training pair"):
        assert_condition_pair_identity([prepared[0], changed])


class CountingRuntime(DeterministicRuntime):
    def __init__(self):
        self.scored_rows = 0

    def score(self, pair_manifest):
        self.scored_rows += len(pair_manifest)
        return super().score(pair_manifest)


def test_condition_scoring_deduplicates_identical_rendered_claims(tmp_path: Path) -> None:
    prepared = prepared_conditions()
    runtime = CountingRuntime().fit(prepared[0].training_manifest)
    contracts = {
        value.condition: build_run_contract(
            value,
            runtime.method_id,
            {"classifier_c": 1.0},
            shard_count=1,
            formal=False,
        )
        for value in prepared
    }
    artifacts, states, unique_count = score_prepared_conditions_shard(
        prepared,
        runtime,
        contracts,
        tmp_path,
        shard_index=0,
        resume=False,
    )
    naive_count = sum(len(value.evaluation_grid) for value in prepared)
    assert set(artifacts) == set(states) == {"NN", "DN", "ND", "DD"}
    assert unique_count == runtime.scored_rows
    assert unique_count < naive_count
    # Two synthetic reviews x (10 seen aspects * 3 sentiments +
    # 2 held-out aspects * 2 representation variants * 3 sentiments).
    assert unique_count == 2 * 42
    assert unique_rendered_claim_count(prepared) == unique_count
