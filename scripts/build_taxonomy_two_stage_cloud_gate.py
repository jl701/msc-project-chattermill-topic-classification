from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir  # noqa: E402
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    registered_folds,
    training_scope_id,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_two_stage_cloud_gate_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "experimental"
    / "taxonomy_two_stage_precloud_v2"
    / "cloud_gate_manifest.json"
)
FORBIDDEN_COMMAND_TOKENS = {
    "--include-official-test",
    "score-test",
    "analyse-test",
    "formal-score-test",
    "formal-analyse-test",
}


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pass_artifact(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    if value.get("status") != "pass":
        raise ValueError(f"Required local artifact did not pass: {path}")
    if int(value.get("test_contract_count", 0)) != 0:
        raise ValueError(f"Required local artifact contains a test contract: {path}")
    if int(value.get("failure_count", 0)) != 0:
        raise ValueError(f"Required local artifact contains a failure: {path}")
    return value


def _validate_commands(commands: object) -> list[list[str]]:
    if not isinstance(commands, list) or not commands:
        raise ValueError("Cloud benchmark commands must be a non-empty list.")
    normalised: list[list[str]] = []
    for command in commands:
        if not isinstance(command, list) or not command:
            raise ValueError("Each cloud benchmark command must be a non-empty argv list.")
        argv = [str(value) for value in command]
        lowered = {value.lower() for value in argv}
        forbidden = sorted(lowered & FORBIDDEN_COMMAND_TOKENS)
        if forbidden or any(value.lower().endswith("test.csv") for value in argv):
            raise ValueError(f"Official-test work entered the cloud gate: {forbidden or argv}")
        normalised.append(argv)
    return normalised


def build_manifest(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    data_dir: Path,
) -> dict[str, Any]:
    if config.get("gate_id") != "taxonomy_two_stage_cloud_gate_v1":
        raise ValueError("Unexpected two-stage cloud gate ID.")
    if config.get("status") != "preregistered_before_cloud_execution":
        raise ValueError("Two-stage cloud gate is not preregistered.")
    contract = config.get("data_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Cloud gate lacks a data contract.")
    if list(contract.get("allowed_splits", [])) != ["train", "validation"]:
        raise ValueError("Cloud data contract must allow only train and validation.")
    if contract.get("include_official_test") is not False:
        raise ValueError("Cloud data contract must keep official test disabled.")

    whitelist = contract.get("upload_whitelist")
    if not isinstance(whitelist, Mapping) or set(whitelist) != {
        "train.csv",
        "validation.csv",
    }:
        raise ValueError("Cloud upload whitelist must contain only train and validation CSVs.")
    data_files: dict[str, dict[str, object]] = {}
    for name, expected_hash in whitelist.items():
        path = data_dir / str(name)
        observed_hash = _sha256(path)
        if observed_hash != str(expected_hash):
            raise ValueError(f"Cloud data hash mismatch for {name}.")
        data_files[str(name)] = {
            "sha256": observed_hash,
            "bytes": path.stat().st_size,
        }

    source_files: dict[str, str] = {}
    for relative in config.get("required_source_files", []):
        relative_path = Path(str(relative))
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        source_files[relative_path.as_posix()] = _sha256(path)

    local_artifacts: dict[str, str] = {}
    for relative in config.get("required_local_artifacts", []):
        relative_path = Path(str(relative))
        path = project_root / relative_path
        _assert_pass_artifact(path)
        local_artifacts[relative_path.as_posix()] = _sha256(path)

    gate = config.get("benchmark_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("Cloud benchmark gate is missing.")
    commands = _validate_commands(gate.get("commands"))

    folds = {
        level: [fold.fold_id for fold in registered_folds(level)]
        for level in ("L1", "L2", "L3", "L4")
    }
    scopes = sorted(
        {
            training_scope_id(fold)
            for level in folds
            for fold in registered_folds(level)
        }
    )
    schedule = config.get("full_validation_schedule_after_gate")
    if not isinstance(schedule, Mapping):
        raise ValueError("Full validation schedule is missing.")
    if int(schedule.get("unique_training_scopes", -1)) != len(scopes):
        raise ValueError("Registered training-scope count changed.")

    return {
        "schema_version": "taxonomy_two_stage_cloud_gate_manifest_v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_id": str(config["gate_id"]),
        "parent_protocol_id": str(config["parent_protocol_id"]),
        "official_splits_permitted": ["train", "validation"],
        "include_official_test": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "data_files": data_files,
        "source_files": source_files,
        "local_artifacts": local_artifacts,
        "benchmark": {
            "fold_id": str(gate["fold_id"]),
            "formal_result": bool(gate["formal_result"]),
            "commands": commands,
            "admission_criteria": list(gate["admission_criteria"]),
        },
        "registered_folds": folds,
        "unique_training_scopes": scopes,
        "full_validation_release_status": str(schedule["release_status"]),
        "stop_conditions": list(config["stop_conditions"]),
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the test-free genuine-two-stage cloud benchmark gate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        _read_object(args.config),
        project_root=PROJECT_ROOT,
        data_dir=args.data_dir,
    )
    _write_atomic(args.output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "test_contract_count": manifest["test_contract_count"],
                "unique_training_scopes": len(manifest["unique_training_scopes"]),
                "benchmark_commands": len(manifest["benchmark"]["commands"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
