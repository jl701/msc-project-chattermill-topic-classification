"""Frozen validation-selection artifacts transferred unchanged to test."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from msc_project.experiments.taxonomy_analysis import assert_shared_seen_threshold
from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_protocol import (
    StrictThresholdSelection,
    TaxonomyFold,
)


@dataclass(frozen=True)
class ThresholdTransferArtifact:
    schema_version: str
    protocol_id: str
    method_id: str
    fold_id: str
    level: str
    conditions: tuple[str, ...]
    threshold: float
    calibration_aspects: tuple[str, ...]
    heldout_aspects: tuple[str, ...]
    scientific_parameters_sha256: str
    training_contract_sha256: str
    validation_score_contracts: dict[str, str]
    validation_pair_identity_sha256: str
    selection_metric: str
    tie_breakers: tuple[str, ...]

    def validate(self) -> None:
        if self.schema_version != "taxonomy_threshold_transfer_v1":
            raise ValueError("Unexpected threshold-transfer schema version.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Transferred threshold must lie in [0,1].")
        if not self.conditions or set(self.validation_score_contracts) != set(
            self.conditions
        ):
            raise ValueError("Threshold artifact lacks per-condition score contracts.")
        if set(self.calibration_aspects) & set(self.heldout_aspects):
            raise ValueError("Held-out aspects entered threshold calibration.")
        if not self.calibration_aspects or not self.heldout_aspects:
            raise ValueError("Threshold calibration partitions must be non-empty.")
        for value in (
            self.scientific_parameters_sha256,
            self.training_contract_sha256,
            self.validation_pair_identity_sha256,
            *self.validation_score_contracts.values(),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("Threshold artifact contains an invalid SHA-256.")

    @property
    def artifact_sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "artifact_sha256": self.artifact_sha256}


def build_threshold_transfer_artifact(
    fold: TaxonomyFold,
    method_id: str,
    selections: Mapping[str, StrictThresholdSelection],
    *,
    protocol_id: str,
    scientific_parameters_sha256: str,
    training_contract_sha256: str,
    validation_score_contracts: Mapping[str, str],
    validation_pair_identity_sha256: str,
) -> ThresholdTransferArtifact:
    if set(selections) != set(fold.conditions):
        raise ValueError("Threshold selections do not cover every registered condition.")
    if set(validation_score_contracts) != set(fold.conditions):
        raise ValueError("Validation score contracts do not cover every condition.")
    threshold = assert_shared_seen_threshold(selections)
    calibration_sets = {
        tuple(selection.calibration_aspects)
        for selection in selections.values()
    }
    if calibration_sets != {tuple(fold.seen_aspects)}:
        raise ValueError("Threshold selections do not use the exact seen-aspect set.")
    artifact = ThresholdTransferArtifact(
        schema_version="taxonomy_threshold_transfer_v1",
        protocol_id=protocol_id,
        method_id=method_id,
        fold_id=fold.fold_id,
        level=fold.level,
        conditions=tuple(fold.conditions),
        threshold=float(threshold),
        calibration_aspects=tuple(fold.seen_aspects),
        heldout_aspects=tuple(fold.heldout_aspects),
        scientific_parameters_sha256=scientific_parameters_sha256,
        training_contract_sha256=training_contract_sha256,
        validation_score_contracts={
            str(key): str(value)
            for key, value in validation_score_contracts.items()
        },
        validation_pair_identity_sha256=validation_pair_identity_sha256,
        selection_metric="pair_micro_f1",
        tie_breakers=(
            "pair_samples_f1",
            "pair_micro_precision",
            "lower_false_positive_rows_per_100",
            "higher_threshold",
        ),
    )
    artifact.validate()
    return artifact


def write_threshold_transfer_artifact(
    path: Path,
    artifact: ThresholdTransferArtifact,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite threshold artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale threshold temporary file: {temporary}")
    temporary.write_text(
        json.dumps(artifact.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def load_threshold_transfer_artifact(
    path: Path,
    *,
    fold: TaxonomyFold,
    method_id: str,
    scientific_parameters_sha256: str,
    training_contract_sha256: str,
) -> ThresholdTransferArtifact:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Threshold artifact must contain an object.")
    declared_hash = value.pop("artifact_sha256", None)
    value["conditions"] = tuple(value["conditions"])
    value["calibration_aspects"] = tuple(value["calibration_aspects"])
    value["heldout_aspects"] = tuple(value["heldout_aspects"])
    value["tie_breakers"] = tuple(value["tie_breakers"])
    artifact = ThresholdTransferArtifact(**value)
    if declared_hash != artifact.artifact_sha256:
        raise ValueError("Threshold artifact content hash mismatch.")
    expected = {
        "fold_id": fold.fold_id,
        "level": fold.level,
        "conditions": tuple(fold.conditions),
        "calibration_aspects": tuple(fold.seen_aspects),
        "heldout_aspects": tuple(fold.heldout_aspects),
        "method_id": method_id,
        "scientific_parameters_sha256": scientific_parameters_sha256,
        "training_contract_sha256": training_contract_sha256,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": getattr(artifact, key)}
        for key, expected_value in expected.items()
        if getattr(artifact, key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Threshold artifact is incompatible with test: {mismatches}")
    return artifact
