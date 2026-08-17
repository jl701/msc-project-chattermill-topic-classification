from __future__ import annotations

import pandas as pd
import pytest

from msc_project.experiments.taxonomy_two_stage_training import (
    build_two_stage_training_manifests,
    training_manifest_sha256,
)


ASPECTS = ("Parent: Alpha", "Parent: Beta", "Parent: Gamma")


def _rows() -> pd.DataFrame:
    records = []
    sentiments = ("negative", "neutral", "positive")
    for index in range(30):
        aspect = ASPECTS[index % len(ASPECTS)]
        records.append(
            {
                "row_uid": f"train:{index}",
                "text": f"review {index}",
                "original_split": "train",
                "supervision_labels": [(aspect, sentiments[index % 3])],
            }
        )
    return pd.DataFrame.from_records(records)


def _resource() -> dict[str, object]:
    return {
        "minimal_aspects": {
            aspect: f"Definition for {aspect}." for aspect in ASPECTS
        }
    }


def test_two_stage_training_manifests_are_balanced_and_stable() -> None:
    first = build_two_stage_training_manifests(
        _rows(), ASPECTS, _resource(), total_budget=36, aspect_budget=18
    )
    second = build_two_stage_training_manifests(
        _rows().sample(frac=1.0, random_state=4),
        ASPECTS,
        _resource(),
        total_budget=36,
        aspect_budget=18,
    )
    assert first["aspect_presence"]["answer"].value_counts().to_dict() == {
        "Y": 9,
        "N": 9,
    }
    assert first["sentiment"]["answer"].value_counts().to_dict() == {
        "A": 6,
        "B": 6,
        "C": 6,
    }
    assert training_manifest_sha256(first["aspect_presence"]) == (
        training_manifest_sha256(second["aspect_presence"])
    )
    assert training_manifest_sha256(first["sentiment"]) == (
        training_manifest_sha256(second["sentiment"])
    )


def test_two_stage_training_rejects_non_train_rows() -> None:
    rows = _rows()
    rows.loc[0, "original_split"] = "validation"
    with pytest.raises(ValueError, match="only train"):
        build_two_stage_training_manifests(
            rows, ASPECTS, _resource(), total_budget=30, aspect_budget=15
        )
