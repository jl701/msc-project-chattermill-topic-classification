from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_taxonomy_two_stage_cloud_gate.py"
SPEC = importlib.util.spec_from_file_location("two_stage_cloud_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    for name, content in (("train.csv", b"train"), ("validation.csv", b"validation")):
        (data / name).write_bytes(content)
    source = project / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    audit = project / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "failure_count": 0,
                "test_contract_count": 0,
            }
        ),
        encoding="utf-8",
    )
    config: dict[str, object] = {
        "gate_id": "taxonomy_two_stage_cloud_gate_v1",
        "status": "preregistered_before_cloud_execution",
        "parent_protocol_id": "taxonomy_two_stage_precloud_v2",
        "data_contract": {
            "allowed_splits": ["train", "validation"],
            "include_official_test": False,
            "upload_whitelist": {
                "train.csv": _sha256(data / "train.csv"),
                "validation.csv": _sha256(data / "validation.csv"),
            },
        },
        "required_source_files": ["source.py"],
        "required_local_artifacts": ["audit.json"],
        "benchmark_gate": {
            "fold_id": "l2-a01",
            "formal_result": False,
            "commands": [["python", "safe_smoke.py"]],
            "admission_criteria": ["pass"],
        },
        "full_validation_schedule_after_gate": {
            "release_status": "blocked_until_measured_gate_passes",
            "unique_training_scopes": 26,
        },
        "stop_conditions": ["failure"],
    }
    return config, project, data


def test_build_manifest_is_validation_only_and_complete(tmp_path: Path) -> None:
    config, project, data = _fixture(tmp_path)
    value = MODULE.build_manifest(config, project_root=project, data_dir=data)
    assert value["status"] == "pass"
    assert value["official_splits_permitted"] == ["train", "validation"]
    assert value["include_official_test"] is False
    assert value["test_contract_count"] == 0
    assert len(value["unique_training_scopes"]) == 26


def test_build_manifest_rejects_official_test_command(tmp_path: Path) -> None:
    config, project, data = _fixture(tmp_path)
    config["benchmark_gate"]["commands"] = [  # type: ignore[index]
        ["python", "unsafe.py", "--include-official-test"]
    ]
    with pytest.raises(ValueError, match="Official-test"):
        MODULE.build_manifest(config, project_root=project, data_dir=data)


def test_build_manifest_rejects_failed_prerequisite(tmp_path: Path) -> None:
    config, project, data = _fixture(tmp_path)
    (project / "audit.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "failure_count": 1,
                "test_contract_count": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not pass"):
        MODULE.build_manifest(config, project_root=project, data_dir=data)
