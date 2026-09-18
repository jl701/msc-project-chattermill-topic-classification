"""Content-addressed training checkpoints for cloud validation/test separation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

from msc_project.experiments.taxonomy_execution import canonical_sha256


CHECKPOINT_MANIFEST = "checkpoint.manifest.json"


@dataclass(frozen=True)
class TrainingContract:
    protocol_id: str
    scientific_protocol_sha256: str
    method_id: str
    method_spec_sha256: str
    method_registry_sha256: str
    training_scope_id: str
    seed: int
    training_manifest_sha256: str
    scientific_parameters_sha256: str
    model_id: str | None
    model_revision: str | None
    training_pairs: int

    def validate(self) -> None:
        for field in (
            "protocol_id",
            "scientific_protocol_sha256",
            "method_id",
            "method_spec_sha256",
            "scientific_protocol_sha256",
            "method_registry_sha256",
            "training_scope_id",
            "training_manifest_sha256",
            "scientific_parameters_sha256",
        ):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"Training contract {field} must be non-empty.")
        for field in (
            "method_spec_sha256",
            "method_registry_sha256",
            "training_manifest_sha256",
            "scientific_parameters_sha256",
        ):
            value = str(getattr(self, field))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"Training contract {field} must be a lowercase SHA-256.")
        if self.seed < 0 or self.training_pairs < 1:
            raise ValueError("Training seed and pair count are invalid.")

    @property
    def contract_sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "contract_sha256": self.contract_sha256}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_hashes(checkpoint_dir: Path) -> dict[str, str]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    files = sorted(
        path
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and path.name != CHECKPOINT_MANIFEST
    )
    if not files:
        raise ValueError("Checkpoint directory contains no model artifacts.")
    return {
        path.relative_to(checkpoint_dir).as_posix(): _file_sha256(path)
        for path in files
    }


def _write_manifest(
    checkpoint_dir: Path,
    contract: TrainingContract,
    evidence: Mapping[str, object],
) -> Path:
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST
    if manifest_path.exists():
        raise FileExistsError(f"Checkpoint manifest already exists: {manifest_path}")
    payload = {
        "schema_version": "taxonomy_training_checkpoint_v1",
        "training_contract": contract.to_dict(),
        "files": checkpoint_file_hashes(checkpoint_dir),
        "evidence": dict(evidence),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def write_checkpoint_atomic(
    checkpoint_dir: Path,
    contract: TrainingContract,
    writer: Callable[[Path], None],
    *,
    evidence: Mapping[str, object],
) -> Path:
    """Write to a sibling directory and publish only after all hashes exist."""

    contract.validate()
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_dir.with_name(
        f".{checkpoint_dir.name}.tmp-{os.getpid()}-{contract.contract_sha256[:8]}"
    )
    if temporary.exists():
        raise FileExistsError(f"Stale checkpoint temporary directory: {temporary}")
    temporary.mkdir()
    writer(temporary)
    _write_manifest(temporary, contract, evidence)
    os.replace(temporary, checkpoint_dir)
    return checkpoint_dir / CHECKPOINT_MANIFEST


def validate_checkpoint(
    checkpoint_dir: Path,
    contract: TrainingContract,
) -> dict[str, object]:
    contract.validate()
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint manifest must contain an object.")
    if payload.get("schema_version") != "taxonomy_training_checkpoint_v1":
        raise ValueError("Unexpected checkpoint schema version.")
    if payload.get("training_contract") != contract.to_dict():
        raise ValueError("Checkpoint belongs to a different training contract.")
    expected_files = payload.get("files")
    observed_files = checkpoint_file_hashes(checkpoint_dir)
    if expected_files != observed_files:
        raise ValueError("Checkpoint file set or content hash mismatch.")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Checkpoint evidence must be an object.")
    return payload


def checkpoint_resume_state(
    checkpoint_dir: Path,
    contract: TrainingContract,
) -> tuple[str, dict[str, object] | None]:
    if not checkpoint_dir.exists():
        return "missing", None
    try:
        payload = validate_checkpoint(checkpoint_dir, contract)
    except Exception as error:
        raise RuntimeError(
            f"Partial, corrupt, or incompatible checkpoint: {checkpoint_dir}"
        ) from error
    return "complete", payload
