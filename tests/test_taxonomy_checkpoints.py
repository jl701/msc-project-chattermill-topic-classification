from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_checkpoints import (
    TrainingContract,
    checkpoint_resume_state,
    validate_checkpoint,
    write_checkpoint_atomic,
)
from msc_project.experiments.taxonomy_execution import canonical_sha256


def contract() -> TrainingContract:
    return TrainingContract(
        protocol_id="taxonomy_generalisation_precloud_v1",
        scientific_protocol_sha256="0" * 64,
        method_id="distilbert_review_candidate_cross_encoder",
        method_spec_sha256="a" * 64,
        method_registry_sha256="b" * 64,
        training_scope_id="heldout-a01",
        seed=13,
        training_manifest_sha256="c" * 64,
        scientific_parameters_sha256=canonical_sha256({"lr": 3e-5}),
        model_id="distilbert-base-uncased",
        model_revision="d" * 40,
        training_pairs=4096,
    )


def fake_writer(path: Path) -> None:
    (path / "config.json").write_text('{"model":"fake"}', encoding="utf-8")
    weights = path / "weights"
    weights.mkdir()
    (weights / "model.bin").write_bytes(b"model weights")


def test_checkpoint_round_trip_and_exact_resume(tmp_path: Path) -> None:
    run = contract()
    directory = tmp_path / "checkpoint"
    write_checkpoint_atomic(
        directory,
        run,
        fake_writer,
        evidence={"history": [{"epoch": 1, "loss": 0.5}]},
    )
    payload = validate_checkpoint(directory, run)
    assert set(payload["files"]) == {"config.json", "weights/model.bin"}
    state, resumed = checkpoint_resume_state(directory, run)
    assert state == "complete"
    assert resumed == payload
    with pytest.raises(FileExistsError):
        write_checkpoint_atomic(directory, run, fake_writer, evidence={})


def test_checkpoint_rejects_contract_or_file_changes(tmp_path: Path) -> None:
    run = contract()
    directory = tmp_path / "checkpoint"
    write_checkpoint_atomic(directory, run, fake_writer, evidence={})
    changed = replace(
        run,
        scientific_parameters_sha256=canonical_sha256({"lr": 5e-5}),
    )
    with pytest.raises(ValueError, match="different training contract"):
        validate_checkpoint(directory, changed)
    (directory / "weights" / "model.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        validate_checkpoint(directory, run)
    with pytest.raises(RuntimeError, match="corrupt"):
        checkpoint_resume_state(directory, run)
