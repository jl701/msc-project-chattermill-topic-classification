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
    score_prepared_shard,
)
from msc_project.experiments.taxonomy_protocol import registered_folds
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions


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
