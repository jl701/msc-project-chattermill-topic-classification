from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/audit_taxonomy_two_stage_runpod_startup.py"
SPEC = importlib.util.spec_from_file_location("formal_runpod_startup", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_weight_index_requires_every_cached_shard(tmp_path: Path) -> None:
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="incomplete"):
        MODULE._snapshot_weight_files(tmp_path)


def test_weight_index_returns_the_complete_unique_shard_set(tmp_path: Path) -> None:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": first.name,
                    "b": second.name,
                    "c": second.name,
                }
            }
        ),
        encoding="utf-8",
    )

    assert MODULE._snapshot_weight_files(tmp_path) == [first, second]


def test_new_output_root_uses_nearest_existing_parent(tmp_path: Path) -> None:
    requested = tmp_path / "new" / "worker" / "output"

    assert MODULE._nearest_existing_parent(requested) == tmp_path.resolve()


def test_worker_manifest_dispatches_to_post_supervisor_v2_parser() -> None:
    manifest = MODULE._build_parallel_worker_manifest(
        PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_three_gpu_parallel_v2.json",
        "worker-qlora-b",
        output_root="/workspace/taxonomy_two_stage_formal_v2/worker-qlora-b",
        data_dir="/workspace/fabsa_data",
    )

    assert manifest["schema_version"] == "taxonomy_post_supervisor_worker_manifest_v2"
    assert manifest["protocol_id"] == "taxonomy_two_stage_formal_v2"
    assert manifest["failure_count"] == 0
    assert manifest["test_contract_count"] == 0


def test_worker_manifest_keeps_legacy_v1_compatibility() -> None:
    manifest = MODULE._build_parallel_worker_manifest(
        PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_three_gpu_parallel_v1.json",
        "worker-qlora-b",
        output_root="/workspace/taxonomy_two_stage_formal_v1/worker-qlora-b",
        data_dir="/workspace/fabsa_data",
    )

    assert manifest["schema_version"] == "taxonomy_two_stage_worker_manifest_v1"
    assert manifest["protocol_id"] == "taxonomy_two_stage_formal_v1"
    assert manifest["failure_count"] == 0
    assert manifest["test_contract_count"] == 0
