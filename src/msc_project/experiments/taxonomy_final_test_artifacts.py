"""Resolve and verify frozen validation-era artifacts for the final test.

The inventory stores portable root aliases.  Callers supply the four concrete
roots at execution time; every selected file/tree is checked against the
pre-release inventory before it may influence a score job.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from msc_project.experiments.taxonomy_final_test import (
    FinalTestJob,
    canonical_sha256,
    file_sha256,
)


@dataclass(frozen=True)
class ArtifactRoots:
    project_root: Path
    local_validation_root: Path
    cloud_worker_distil_frozen: Path
    cloud_worker_qlora_a: Path
    cloud_worker_qlora_b: Path

    def aliases(self) -> dict[str, Path]:
        return {
            "repository": self.project_root,
            "project_root": self.project_root,
            "post_supervisor_local_v1": self.local_validation_root,
            "cloud_worker_distil_frozen": self.cloud_worker_distil_frozen,
            "cloud_worker_qlora_a": self.cloud_worker_qlora_a,
            "cloud_worker_qlora_b": self.cloud_worker_qlora_b,
        }


@dataclass(frozen=True)
class FrozenJobArtifacts:
    job: FinalTestJob
    selection_path: Path
    selection: Mapping[str, object]
    checkpoint_path: Path | None
    aspect_threshold: float
    runner_up_threshold: float
    source_artifact_sha256s: Mapping[str, str]


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = [value for value in root.rglob("*") if value.is_file()]
    files.sort(
        key=lambda path: (
            path.relative_to(root).as_posix().casefold(),
            path.relative_to(root).as_posix(),
        )
    )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    if not records:
        raise ValueError(f"Frozen checkpoint tree is empty: {root}")
    return canonical_sha256(records)


def _entry_path(entry: Mapping[str, object], roots: ArtifactRoots) -> Path:
    alias = str(entry.get("root_alias", ""))
    root = roots.aliases().get(alias)
    if root is None:
        raise ValueError(f"Unknown final-test artifact root alias: {alias!r}.")
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact inventory entry lacks a portable path.")
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Artifact inventory path escapes its registered root.") from error
    return path


def verify_inventory_entries(
    inventory: Mapping[str, object], roots: ArtifactRoots
) -> dict[str, object]:
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Final-test artifact inventory is empty.")
    failures: list[str] = []
    bytes_total = 0
    files_total = 0
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise TypeError("Artifact inventory entry is not an object.")
        path = _entry_path(raw, roots)
        kind = str(raw.get("kind", ""))
        observed = _tree_sha256(path) if kind == "selected_checkpoint_tree" else file_sha256(path)
        if observed != raw.get("sha256"):
            failures.append(f"{raw.get('root_alias')}:{raw.get('path')}")
        bytes_total += int(raw.get("bytes", 0))
        files_total += int(raw.get("file_count", 0))
    if failures:
        raise ValueError(f"Frozen artifact SHA-256 mismatch: {failures[:5]!r}.")
    return {
        "status": "pass",
        "entries": len(entries),
        "files": files_total,
        "bytes": bytes_total,
        "hash_mismatch_count": 0,
        "test_contract_count": 0,
    }


def _matching_entries(
    inventory: Mapping[str, object],
    *,
    kind: str,
    method_id: str,
    scope_id: str,
) -> list[Mapping[str, object]]:
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise TypeError("Artifact inventory has no entries.")
    return [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("kind") == kind
        and entry.get("method_id") == method_id
        and entry.get("scope_id") == scope_id
    ]


def _one_entry(
    inventory: Mapping[str, object],
    *,
    kind: str,
    method_id: str,
    scope_id: str,
) -> Mapping[str, object]:
    matches = _matching_entries(
        inventory, kind=kind, method_id=method_id, scope_id=scope_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {kind} artifact for {method_id}/{scope_id}; "
            f"observed {len(matches)}."
        )
    return matches[0]


def _read_verified_json(entry: Mapping[str, object], roots: ArtifactRoots) -> tuple[Path, dict[str, object]]:
    path = _entry_path(entry, roots)
    if file_sha256(path) != entry.get("sha256"):
        raise ValueError(f"Frozen JSON artifact hash mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Frozen JSON artifact is not an object: {path}")
    return path, value


def _local_thresholds(method_id: str, value: Mapping[str, object]) -> tuple[float, float]:
    if method_id == "description_to_classifier_weight_transfer":
        generators = value.get("generators")
        if not isinstance(generators, Mapping):
            raise ValueError("DCWT fold record lacks generators.")
        primary = generators.get("kernel_ridge")
        if not isinstance(primary, Mapping):
            raise ValueError("DCWT fold record lacks the frozen kernel-ridge primary.")
        selection = primary.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("DCWT primary lacks its frozen selection.")
        return float(selection["aspect_threshold"]), float(
            primary["second_sentiment_threshold"]
        )
    return float(value["aspect_threshold"]), float(value["second_sentiment_threshold"])


def resolve_job_artifacts(
    job: FinalTestJob,
    inventory: Mapping[str, object],
    roots: ArtifactRoots,
) -> FrozenJobArtifacts:
    if job.phase != "score":
        raise ValueError("Only base score jobs own frozen source artifacts.")
    local_methods = {
        "strict_train_only_tfidf",
        "e5_base_v2",
        "description_to_classifier_weight_transfer",
        "frozen_qwen_candidate_pair",
    }
    source_hashes: dict[str, str] = {}
    if job.method_id in local_methods:
        if job.level != "L2":
            raise ValueError("Local validation-only source methods are registered for L2 only.")
        entry = _one_entry(
            inventory,
            kind="local_fold_selection_and_result",
            method_id=job.method_id,
            scope_id=job.fold_id,
        )
        path, value = _read_verified_json(entry, roots)
        if int(value.get("failure_count", -1)) != 0 or int(
            value.get("test_contract_count", -1)
        ) != 0:
            raise ValueError("A local frozen selection reports a validation failure.")
        aspect, runner = _local_thresholds(job.method_id, value)
        source_hashes["selection"] = str(entry["sha256"])
        return FrozenJobArtifacts(
            job=job,
            selection_path=path,
            selection=value,
            checkpoint_path=None,
            aspect_threshold=aspect,
            runner_up_threshold=runner,
            source_artifact_sha256s=source_hashes,
        )

    entry = _one_entry(
        inventory,
        kind="frozen_selection",
        method_id=job.method_id,
        scope_id=job.training_scope_id,
    )
    path, value = _read_verified_json(entry, roots)
    if int(value.get("failure_count", -1)) != 0 or int(
        value.get("test_contract_count", -1)
    ) != 0:
        raise ValueError("A formal frozen selection reports a validation failure.")
    source_hashes["selection"] = str(entry["sha256"])
    checkpoint_path: Path | None = None
    if job.method_id == "frozen_qwen_few_shot":
        thresholds = value.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("Few-shot selection lacks frozen thresholds.")
    else:
        thresholds = value.get("selected_thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("Trainable selection lacks frozen thresholds.")
        checkpoint_entry = _one_entry(
            inventory,
            kind="selected_checkpoint_tree",
            method_id=job.method_id,
            scope_id=job.training_scope_id,
        )
        checkpoint_path = _entry_path(checkpoint_entry, roots)
        if _tree_sha256(checkpoint_path) != checkpoint_entry.get("sha256"):
            raise ValueError("Selected checkpoint tree hash mismatch.")
        source_hashes["checkpoint_tree"] = str(checkpoint_entry["sha256"])
    return FrozenJobArtifacts(
        job=job,
        selection_path=path,
        selection=value,
        checkpoint_path=checkpoint_path,
        aspect_threshold=float(thresholds["aspect"]),
        runner_up_threshold=float(thresholds["runner_up_sentiment"]),
        source_artifact_sha256s=source_hashes,
    )


def source_hash_union(values: Sequence[FrozenJobArtifacts]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        for key, digest in value.source_artifact_sha256s.items():
            output[f"{value.job.method_id}:{key}"] = digest
    return dict(sorted(output.items()))
