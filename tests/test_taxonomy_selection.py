from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_protocol import (
    StrictThresholdSelection,
    registered_folds,
)
from msc_project.experiments.taxonomy_selection import (
    build_threshold_transfer_artifact,
    load_threshold_transfer_artifact,
    write_threshold_transfer_artifact,
)


def selections():
    fold = registered_folds("L3")[0]
    return {
        condition: StrictThresholdSelection(
            threshold=0.42,
            metrics={},
            sweep=pd.DataFrame(),
            calibration_aspects=fold.seen_aspects,
        )
        for condition in fold.conditions
    }


def artifact():
    fold = registered_folds("L3")[0]
    return build_threshold_transfer_artifact(
        fold,
        "strict_train_only_tfidf",
        selections(),
        protocol_id="taxonomy_generalisation_precloud_v1",
        scientific_parameters_sha256=canonical_sha256({"c": 1}),
        training_contract_sha256="a" * 64,
        validation_score_contracts={
            condition: canonical_sha256({"condition": condition})
            for condition in fold.conditions
        },
        validation_pair_identity_sha256="b" * 64,
    )


def test_threshold_artifact_round_trip_and_test_compatibility(tmp_path: Path) -> None:
    value = artifact()
    path = tmp_path / "threshold.json"
    write_threshold_transfer_artifact(path, value)
    fold = registered_folds("L3")[0]
    loaded = load_threshold_transfer_artifact(
        path,
        fold=fold,
        method_id="strict_train_only_tfidf",
        scientific_parameters_sha256=canonical_sha256({"c": 1}),
        training_contract_sha256="a" * 64,
    )
    assert loaded.threshold == pytest.approx(0.42)
    assert loaded.artifact_sha256 == value.artifact_sha256
    with pytest.raises(FileExistsError):
        write_threshold_transfer_artifact(path, value)


def test_threshold_artifact_rejects_condition_threshold_or_contract_mismatch() -> None:
    fold = registered_folds("L3")[0]
    changed = selections()
    changed["DN"] = StrictThresholdSelection(
        threshold=0.43,
        metrics={},
        sweep=pd.DataFrame(),
        calibration_aspects=fold.seen_aspects,
    )
    with pytest.raises(ValueError, match="thresholds differ"):
        build_threshold_transfer_artifact(
            fold,
            "strict_train_only_tfidf",
            changed,
            protocol_id="p",
            scientific_parameters_sha256="a" * 64,
            training_contract_sha256="b" * 64,
            validation_score_contracts={
                condition: "c" * 64 for condition in fold.conditions
            },
            validation_pair_identity_sha256="d" * 64,
        )
