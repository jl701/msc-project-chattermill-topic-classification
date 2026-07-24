from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_checkpoints import TrainingContract
from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_methods import resolve_method_spec
from msc_project.experiments.taxonomy_runtime_factory import (
    create_runtime_for_training,
    load_runtime_checkpoint,
    save_frozen_registry_checkpoint,
    save_runtime_checkpoint,
)


def manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_uid": "r1",
                "text": "email good",
                "candidate_text": "email positive",
                "candidate_aspect": "Support: Email",
                "candidate_sentiment": "positive",
                "negative_type": "gold",
                "target": 1,
            },
            {
                "row_uid": "r1",
                "text": "email good",
                "candidate_text": "phone positive",
                "candidate_aspect": "Support: Phone",
                "candidate_sentiment": "positive",
                "negative_type": "absent",
                "target": 0,
            },
            {
                "row_uid": "r2",
                "text": "phone bad",
                "candidate_text": "phone negative",
                "candidate_aspect": "Support: Phone",
                "candidate_sentiment": "negative",
                "negative_type": "gold",
                "target": 1,
            },
            {
                "row_uid": "r2",
                "text": "phone bad",
                "candidate_text": "email negative",
                "candidate_aspect": "Support: Email",
                "candidate_sentiment": "negative",
                "negative_type": "absent",
                "target": 0,
            },
        ]
    )


def training_contract(parameters: dict[str, object]) -> TrainingContract:
    spec = resolve_method_spec("strict_train_only_tfidf")
    return TrainingContract(
        protocol_id="taxonomy_generalisation_precloud_v1",
        scientific_protocol_sha256="0" * 64,
        method_id=spec.method_id,
        method_spec_sha256=spec.spec_sha256,
        method_registry_sha256="a" * 64,
        training_scope_id="heldout-a01",
        seed=13,
        training_manifest_sha256="b" * 64,
        scientific_parameters_sha256=canonical_sha256(parameters),
        model_id=None,
        model_revision=None,
        training_pairs=len(manifest()),
    )


def test_tfidf_runtime_real_checkpoint_round_trip(tmp_path: Path) -> None:
    parameters = resolve_method_spec("strict_train_only_tfidf").starting_recipe
    runtime = create_runtime_for_training(
        "strict_train_only_tfidf",
        parameters,
        seed=13,
        device=torch.device("cpu"),
    )
    runtime.fit(manifest())
    expected = runtime.score(manifest().drop(columns="target"))
    contract = training_contract(parameters)
    save_runtime_checkpoint(runtime, tmp_path / "checkpoint", contract)
    loaded = load_runtime_checkpoint(
        "strict_train_only_tfidf",
        parameters,
        tmp_path / "checkpoint",
        contract,
        seed=13,
        device=torch.device("cpu"),
    )
    observed = loaded.score(manifest().drop(columns="target"))
    np.testing.assert_allclose(expected, observed)


def test_frozen_checkpoint_does_not_load_or_fit_the_model(tmp_path: Path) -> None:
    spec = resolve_method_spec("e5_base_v2")
    parameters = spec.starting_recipe
    contract = TrainingContract(
        protocol_id="taxonomy_generalisation_precloud_v1",
        scientific_protocol_sha256="0" * 64,
        method_id=spec.method_id,
        method_spec_sha256=spec.spec_sha256,
        method_registry_sha256="a" * 64,
        training_scope_id="heldout-a01",
        seed=13,
        training_manifest_sha256="b" * 64,
        scientific_parameters_sha256=canonical_sha256(parameters),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        training_pairs=len(manifest()),
    )
    checkpoint = tmp_path / "frozen"
    save_frozen_registry_checkpoint(spec.method_id, checkpoint, contract)
    marker = json.loads((checkpoint / "reload.json").read_text(encoding="utf-8"))
    assert marker["task_specific_fit_performed"] is False

    with pytest.raises(ValueError, match="trainable"):
        save_frozen_registry_checkpoint(
            "strict_train_only_tfidf",
            tmp_path / "invalid",
            training_contract(
                resolve_method_spec("strict_train_only_tfidf").starting_recipe
            ),
        )
