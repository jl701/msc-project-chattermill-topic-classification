from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.verified_artifact_sync import (
    build_artifact_unit_manifest,
    file_sha256,
    publish_artifact_unit,
    receive_artifact_unit,
    sync_from_local_source,
    validate_artifact_unit_manifest,
)


def _source_unit(root: Path, *, unit_id: str = "scope-a01-lr-5e-6") -> Path:
    checkpoint = root / "checkpoints" / "scope-a01" / "adapter.bin"
    score = root / "scores" / "scope-a01" / "shard-00000.csv"
    checkpoint.parent.mkdir(parents=True)
    score.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable checkpoint")
    score.write_text("row_uid,score\nvalidation:1,0.75\n", encoding="utf-8")
    return publish_artifact_unit(
        root,
        [
            checkpoint.relative_to(root),
            score.relative_to(root),
        ],
        unit_id=unit_id,
        protocol_id="taxonomy_two_stage_formal_v1",
        contract_sha256="a" * 64,
    )


def test_publish_sync_resume_and_hash_verify(tmp_path: Path) -> None:
    source = tmp_path / "remote"
    local = tmp_path / "local"
    manifest_path = _source_unit(source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert validate_artifact_unit_manifest(manifest)["unit_id"] == "scope-a01-lr-5e-6"

    first = sync_from_local_source(source, local)
    assert first == {"copied": 1, "already_complete": 0}
    assert file_sha256(local / "checkpoints/scope-a01/adapter.bin") == file_sha256(
        source / "checkpoints/scope-a01/adapter.bin"
    )
    second = sync_from_local_source(source, local)
    assert second == {"copied": 0, "already_complete": 1}


def test_corruption_is_rejected_before_publication(tmp_path: Path) -> None:
    source = tmp_path / "remote"
    local = tmp_path / "local"
    manifest_path = _source_unit(source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def corrupting_fetch(relative: str, target: Path) -> None:
        shutil.copyfile(source / relative, target)
        if relative.endswith("adapter.bin"):
            target.write_bytes(b"corrupt in transit")

    with pytest.raises(ValueError, match="mismatch"):
        receive_artifact_unit(
            manifest,
            local_root=local,
            fetch_file=corrupting_fetch,
        )
    assert not (local / "checkpoints/scope-a01/adapter.bin").exists()
    assert not (local / "_sync/received/scope-a01-lr-5e-6.json").exists()


def test_conflicting_local_file_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "remote"
    local = tmp_path / "local"
    _source_unit(source)
    conflict = local / "checkpoints/scope-a01/adapter.bin"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"user-owned different data")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        sync_from_local_source(source, local)
    assert conflict.read_bytes() == b"user-owned different data"


@pytest.mark.parametrize(
    "unsafe",
    ["../secret", "/absolute/file", "folder\\windows", "_sync/receipt.json"],
)
def test_manifest_rejects_unsafe_paths(tmp_path: Path, unsafe: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        build_artifact_unit_manifest(
            root,
            [unsafe],
            unit_id="safe-unit",
            protocol_id="taxonomy_two_stage_formal_v1",
            contract_sha256="b" * 64,
        )


def test_published_unit_id_cannot_be_rebound(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    _source_unit(root)
    changed = root / "other.bin"
    changed.write_bytes(b"other")
    with pytest.raises(FileExistsError, match="other content"):
        publish_artifact_unit(
            root,
            [changed.relative_to(root)],
            unit_id="scope-a01-lr-5e-6",
            protocol_id="taxonomy_two_stage_formal_v1",
            contract_sha256="a" * 64,
        )
