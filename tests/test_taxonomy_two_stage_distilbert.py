from __future__ import annotations

import pandas as pd
import pytest

from msc_project.experiments.taxonomy_two_stage_distilbert import (
    TwoStageDistilBertConfig,
    _validate_manifest,
)


def _manifest(task: str) -> pd.DataFrame:
    value = {
        "row_uid": ["train:1"],
        "text": ["review"],
        "candidate_aspect": ["Parent: Alpha"],
        "candidate_text": ["Aspect: Parent: Alpha. Definition: alpha"],
        "task": [task],
        "answer": ["Y" if task == "aspect_presence" else "A"],
    }
    if task == "sentiment":
        value["candidate_sentiment"] = ["negative"]
    return pd.DataFrame(value)


def test_distilbert_two_stage_contract_is_stable_and_revision_bound() -> None:
    config = TwoStageDistilBertConfig()
    assert config.contract_sha256 == TwoStageDistilBertConfig().contract_sha256
    assert config.contract_sha256 != TwoStageDistilBertConfig(
        learning_rate=2e-5
    ).contract_sha256


def test_distilbert_manifests_keep_tasks_separate() -> None:
    _validate_manifest(
        _manifest("aspect_presence"),
        task="aspect_presence",
        labels={"N": 0, "Y": 1},
        require_answer=True,
    )
    with pytest.raises(ValueError, match="does not contain only"):
        _validate_manifest(
            _manifest("sentiment"),
            task="aspect_presence",
            labels={"N": 0, "Y": 1},
            require_answer=True,
        )
