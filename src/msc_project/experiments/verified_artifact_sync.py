"""Immutable artifact-unit publication and fail-closed verified synchronisation.

The cloud writer publishes a manifest only after every file in a completed unit
exists and has been hashed.  A receiver stages that complete unit, verifies the
declared byte counts and SHA-256 values, and then publishes files locally.  It
never deletes remote data and never overwrites a different local file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence


SYNC_SCHEMA = "verified_artifact_unit_v1"
UNIT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Artifact paths must be non-empty strings.")
    if "\\" in value:
        raise ValueError("Artifact paths must use forward slashes.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe artifact path: {value!r}.")
    if path.parts[0] == "_sync":
        raise ValueError("Artifact units may not publish synchronisation metadata.")
    return path


def _validate_unit_id(unit_id: str) -> str:
    value = str(unit_id)
    if not UNIT_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Unsafe artifact unit ID: {value!r}.")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary path: {temporary}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_artifact_unit_manifest(
    root: Path,
    relative_files: Sequence[str | Path],
    *,
    unit_id: str,
    protocol_id: str,
    contract_sha256: str,
) -> dict[str, object]:
    """Build a manifest for already-complete files beneath ``root``."""

    identifier = _validate_unit_id(unit_id)
    if not str(protocol_id).strip():
        raise ValueError("protocol_id must be non-empty.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(contract_sha256)):
        raise ValueError("contract_sha256 must be a lowercase SHA-256.")
    resolved_root = root.resolve()
    records: list[dict[str, object]] = []
    observed: set[str] = set()
    for supplied in relative_files:
        supplied_text = supplied.as_posix() if isinstance(supplied, Path) else str(supplied)
        relative = _safe_relative_path(supplied_text)
        relative_text = relative.as_posix()
        if relative_text in observed:
            raise ValueError(f"Duplicate artifact path: {relative_text}")
        observed.add(relative_text)
        source = (resolved_root / Path(*relative.parts)).resolve()
        try:
            source.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"Artifact escapes the declared root: {relative_text}") from error
        if not source.is_file():
            raise FileNotFoundError(source)
        records.append(
            {
                "path": relative_text,
                "bytes": int(source.stat().st_size),
                "sha256": file_sha256(source),
            }
        )
    if not records:
        raise ValueError("An artifact unit must contain at least one file.")
    records.sort(key=lambda item: str(item["path"]))
    return {
        "schema_version": SYNC_SCHEMA,
        "unit_id": identifier,
        "protocol_id": str(protocol_id),
        "contract_sha256": str(contract_sha256),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }


def validate_artifact_unit_manifest(value: Mapping[str, object]) -> dict[str, object]:
    if value.get("schema_version") != SYNC_SCHEMA:
        raise ValueError("Unexpected artifact-unit schema version.")
    unit_id = _validate_unit_id(str(value.get("unit_id", "")))
    protocol_id = str(value.get("protocol_id", ""))
    contract_hash = str(value.get("contract_sha256", ""))
    if not protocol_id:
        raise ValueError("Artifact-unit protocol_id must be non-empty.")
    if not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
        raise ValueError("Artifact-unit contract SHA-256 is invalid.")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Artifact-unit files must be a non-empty list.")
    files: list[dict[str, object]] = []
    paths: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise ValueError("Artifact-unit file records must be objects.")
        path = _safe_relative_path(str(raw.get("path", ""))).as_posix()
        size = raw.get("bytes")
        sha256 = str(raw.get("sha256", ""))
        if path in paths:
            raise ValueError(f"Duplicate artifact-unit path: {path}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"Artifact-unit byte count is invalid for {path}.")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"Artifact-unit SHA-256 is invalid for {path}.")
        paths.add(path)
        files.append({"path": path, "bytes": size, "sha256": sha256})
    if [record["path"] for record in files] != sorted(paths):
        raise ValueError("Artifact-unit file records must be sorted by path.")
    return {
        **dict(value),
        "unit_id": unit_id,
        "protocol_id": protocol_id,
        "contract_sha256": contract_hash,
        "files": files,
    }


def publish_artifact_unit(
    root: Path,
    relative_files: Sequence[str | Path],
    *,
    unit_id: str,
    protocol_id: str,
    contract_sha256: str,
) -> Path:
    """Atomically publish the completion manifest for one immutable unit."""

    manifest = build_artifact_unit_manifest(
        root,
        relative_files,
        unit_id=unit_id,
        protocol_id=protocol_id,
        contract_sha256=contract_sha256,
    )
    path = root / "_sync" / "units" / f"{manifest['unit_id']}.json"
    if path.exists():
        observed = validate_artifact_unit_manifest(
            json.loads(path.read_text(encoding="utf-8"))
        )
        comparable = {key: value for key, value in observed.items() if key != "created_at"}
        proposed = {key: value for key, value in manifest.items() if key != "created_at"}
        if comparable != proposed:
            raise FileExistsError(f"Artifact unit ID is already bound to other content: {path}")
        return path
    _atomic_write_json(path, manifest)
    return path


def verify_staged_unit(staging_root: Path, manifest: Mapping[str, object]) -> None:
    value = validate_artifact_unit_manifest(manifest)
    for record in value["files"]:  # type: ignore[index]
        relative = _safe_relative_path(str(record["path"]))
        path = staging_root / Path(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Staged artifact byte count mismatch: {relative}")
        if file_sha256(path) != str(record["sha256"]):
            raise ValueError(f"Staged artifact SHA-256 mismatch: {relative}")


def receive_artifact_unit(
    manifest: Mapping[str, object],
    *,
    local_root: Path,
    fetch_file: Callable[[str, Path], None],
) -> str:
    """Fetch, verify and publish one unit; return ``copied`` or ``already_complete``."""

    value = validate_artifact_unit_manifest(manifest)
    unit_id = str(value["unit_id"])
    receipt = local_root / "_sync" / "received" / f"{unit_id}.json"
    if receipt.is_file():
        observed = validate_artifact_unit_manifest(
            json.loads(receipt.read_text(encoding="utf-8"))
        )
        if observed != value:
            raise RuntimeError(f"Local receipt conflicts with artifact unit {unit_id}.")
        for record in value["files"]:  # type: ignore[index]
            path = local_root / Path(*_safe_relative_path(str(record["path"])).parts)
            if not path.is_file() or file_sha256(path) != str(record["sha256"]):
                raise RuntimeError(f"Received artifact later changed or disappeared: {path}")
        return "already_complete"

    incoming_parent = local_root / "_sync" / "incoming"
    staging = incoming_parent / f".{unit_id}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Stale artifact staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        for record in value["files"]:  # type: ignore[index]
            relative = _safe_relative_path(str(record["path"]))
            target = staging / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            fetch_file(relative.as_posix(), target)
        verify_staged_unit(staging, value)
        for record in value["files"]:  # type: ignore[index]
            relative = _safe_relative_path(str(record["path"]))
            source = staging / Path(*relative.parts)
            target = local_root / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_file() or file_sha256(target) != str(record["sha256"]):
                    raise RuntimeError(f"Refusing to overwrite conflicting local artifact: {target}")
                continue
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{unit_id}")
            shutil.copyfile(source, temporary)
            if file_sha256(temporary) != str(record["sha256"]):
                raise RuntimeError(f"Local publication copy changed unexpectedly: {target}")
            os.replace(temporary, target)
        _atomic_write_json(receipt, value)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return "copied"


def sync_from_local_source(source_root: Path, local_root: Path) -> dict[str, int]:
    """Local transport used by the restore drill and offline transfers."""

    unit_dir = source_root / "_sync" / "units"
    manifests = sorted(unit_dir.glob("*.json")) if unit_dir.is_dir() else []
    counts = {"copied": 0, "already_complete": 0}
    for manifest_path in manifests:
        manifest = validate_artifact_unit_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )

        def fetch(relative: str, target: Path) -> None:
            source = source_root / Path(*_safe_relative_path(relative).parts)
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copyfile(source, target)

        status = receive_artifact_unit(manifest, local_root=local_root, fetch_file=fetch)
        counts[status] += 1
    return counts
