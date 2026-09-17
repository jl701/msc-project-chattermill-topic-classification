"""Fail-closed local release audit for the long two-stage cloud campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.taxonomy_checkpoints import (  # noqa: E402
    TrainingContract,
    checkpoint_resume_state,
    write_checkpoint_atomic,
)
from msc_project.experiments.taxonomy_execution import (  # noqa: E402
    RunContract,
    canonical_sha256,
)
from msc_project.experiments.taxonomy_protocol import pair_identity_hash  # noqa: E402
from msc_project.experiments.taxonomy_two_stage_artifacts import (  # noqa: E402
    build_two_stage_score_artifact,
    two_stage_shard_resume_state,
    write_two_stage_score_shard,
)
from msc_project.experiments.taxonomy_two_stage_formal import (  # noqa: E402
    build_formal_jobs,
    scope_folds,
    validate_formal_job_graph,
)
from msc_project.experiments.verified_artifact_sync import (  # noqa: E402
    publish_artifact_unit,
    receive_artifact_unit,
    sync_from_local_source,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/taxonomy_two_stage_cloud_execution_safety_v1.json"
)
DEFAULT_PARENT = (
    PROJECT_ROOT / "configs/experiments/taxonomy_two_stage_precloud_v2.json"
)
DEFAULT_HARDWARE = (
    PROJECT_ROOT
    / "docs/experiments/taxonomy_two_stage_4090_hardware_gate_v1_result.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/experimental/taxonomy_two_stage_formal_preflight_v1/audit.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _synthetic_grid() -> pd.DataFrame:
    records = []
    for row_index, uid in enumerate(("validation:1", "validation:2")):
        for index, sentiment in enumerate(("negative", "neutral", "positive")):
            records.append(
                {
                    "fold_id": "l2-a01",
                    "condition": "D",
                    "row_index": row_index,
                    "row_uid": uid,
                    "candidate_aspect": "Quality",
                    "candidate_sentiment": sentiment,
                    "pair_label": f"Quality | {sentiment}",
                    "representation_variant": "name_and_description",
                    "is_seen": False,
                    "is_heldout": True,
                    "target": int(row_index == 0 and sentiment == "positive"),
                    "aspect_score": 0.8 - row_index * 0.2,
                    "sentiment_score": 0.15 + index * 0.25,
                }
            )
    return pd.DataFrame.from_records(records)


def _training_contract() -> TrainingContract:
    return TrainingContract(
        protocol_id="taxonomy_two_stage_formal_v1",
        scientific_protocol_sha256="0" * 64,
        method_id="qwen_candidate_pair_qlora",
        method_spec_sha256="1" * 64,
        method_registry_sha256="2" * 64,
        training_scope_id="heldout-a01",
        seed=13,
        training_manifest_sha256="3" * 64,
        scientific_parameters_sha256=canonical_sha256({"lr": 5e-6}),
        model_id="pinned/fake",
        model_revision="4" * 40,
        training_pairs=4096,
    )


def _run_contract(grid: pd.DataFrame) -> RunContract:
    return RunContract(
        protocol_id="taxonomy_two_stage_formal_v1",
        scientific_protocol_sha256="0" * 64,
        method_id="qwen_candidate_pair_qlora",
        method_spec_sha256="1" * 64,
        method_registry_sha256="2" * 64,
        description_resource_sha256="3" * 64,
        candidate_representation_sha256="4" * 64,
        level="L2",
        fold_id="l2-a01",
        condition="D",
        split="validation",
        seed=13,
        training_manifest_sha256="5" * 64,
        evaluation_data_sha256="6" * 64,
        evaluation_pair_identity_sha256=pair_identity_hash(grid),
        scientific_parameters_sha256=canonical_sha256({"decoder": "capped_two"}),
        shard_count=1,
        formal=True,
    )


def _artifact_restore_drill() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="taxonomy-formal-preflight-") as temporary:
        root = Path(temporary)
        remote = root / "remote"
        local = root / "local"
        interrupted = root / "interrupted"
        corrupt_local = root / "corrupt-local"
        contract = _training_contract()
        checkpoint = remote / "checkpoints/fake"

        def writer(path: Path) -> None:
            (path / "adapter.bin").write_bytes(b"verified fake adapter")

        write_checkpoint_atomic(
            checkpoint,
            contract,
            writer,
            evidence={"test_contract_count": 0},
        )
        checkpoint_files = sorted(
            path.relative_to(remote) for path in checkpoint.rglob("*") if path.is_file()
        )
        checkpoint_unit = publish_artifact_unit(
            remote,
            checkpoint_files,
            unit_id="checkpoint-interruption-drill",
            protocol_id="taxonomy_two_stage_formal_v1",
            contract_sha256=contract.contract_sha256,
        )

        grid = _synthetic_grid()
        run = _run_contract(grid)
        artifact = build_two_stage_score_artifact(grid, run, shard_index=0)
        csv_path, manifest_path = write_two_stage_score_shard(
            artifact, remote, run, shard_index=0
        )
        publish_artifact_unit(
            remote,
            [csv_path.relative_to(remote), manifest_path.relative_to(remote)],
            unit_id="score-interruption-drill",
            protocol_id="taxonomy_two_stage_formal_v1",
            contract_sha256=run.contract_sha256,
        )

        manifest = _read(checkpoint_unit)
        calls = 0

        def fail_after_first(relative: str, target: Path) -> None:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ConnectionError("simulated interrupted SCP")
            target.write_bytes((remote / relative).read_bytes())

        interrupted_rejected = False
        try:
            receive_artifact_unit(
                manifest,
                local_root=interrupted,
                fetch_file=fail_after_first,
            )
        except ConnectionError:
            interrupted_rejected = True
        if not interrupted_rejected or (interrupted / "_sync/received").exists():
            raise AssertionError("Interrupted transfer was incorrectly published.")

        first = sync_from_local_source(remote, local)
        second = sync_from_local_source(remote, local)
        checkpoint_resume_state(local / "checkpoints/fake", contract)
        two_stage_shard_resume_state(local, run, grid, shard_index=0)

        # Change a remote byte after publication.  A fresh receiver must reject
        # it before any local artifact or receipt is published.
        (remote / checkpoint_files[0]).write_bytes(b"tampered remote byte")
        corruption_rejected = False
        try:
            sync_from_local_source(remote, corrupt_local)
        except ValueError:
            corruption_rejected = True
        if not corruption_rejected:
            raise AssertionError("Corrupt transfer was not rejected.")
        return {
            "status": "pass",
            "interrupted_transfer_not_published": True,
            "retry_completed_units": first["copied"],
            "idempotent_resume_units": second["already_complete"],
            "checkpoint_exact_resume": True,
            "score_shard_exact_resume": True,
            "corrupt_transfer_rejected": True,
        }


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    config = _read(args.config)
    parent = _read(args.parent_config)
    hardware = _read(args.hardware_record)
    failures: list[str] = []
    checks: dict[str, object] = {}

    data = config.get("sealed_data_contract")
    checks["sealed_data_contract"] = bool(
        config.get("protocol_id") == "taxonomy_two_stage_formal_v1"
        and config.get("status") == "preregistered_before_formal_execution"
        and isinstance(data, Mapping)
        and list(data.get("allowed_splits", [])) == ["train", "validation"]
        and data.get("include_official_test") is False
        and data.get("test_contract_count_required") == 0
    )
    if not checks["sealed_data_contract"]:
        failures.append("sealed_data_contract")

    expected_hashes = parent.get("exact_input_hashes", {})
    data_dir = args.data_dir.resolve()
    observed_data = {
        "train_csv_sha256": _file_hash(data_dir / "train.csv"),
        "validation_csv_sha256": _file_hash(data_dir / "validation.csv"),
    }
    checks["input_hashes"] = all(
        observed_data[key] == expected_hashes.get(key) for key in observed_data
    )
    if not checks["input_hashes"]:
        failures.append("input_hashes")

    jobs = build_formal_jobs(output_root="/workspace/taxonomy_two_stage_formal_v1")
    job_counts = validate_formal_job_graph(jobs)
    checks["job_graph"] = bool(
        len(scope_folds()) == int(config["training"]["unique_outer_scopes"])
        and job_counts["train-candidate"]
        == int(config["training"]["candidate_checkpoint_count"])
    )
    if not checks["job_graph"]:
        failures.append("job_graph")

    checks["qualified_4090"] = bool(
        hardware.get("failure_count") == 0
        and hardware.get("test_contract_count") == 0
        and hardware.get("official_test_opened") is False
        and hardware.get("qualified_host", {}).get("gpu")
        == "NVIDIA GeForce RTX 4090"
        and hardware.get("hardware_qualification", {}).get("sustained_soak_audit_status")
        == "pass"
    )
    if not checks["qualified_4090"]:
        failures.append("qualified_4090")

    required_files = [
        "scripts/run_taxonomy_two_stage_trainable_validation.py",
        "scripts/run_taxonomy_two_stage_formal_campaign.py",
        "scripts/sync_runpod_taxonomy_artifacts.py",
        "scripts/audit_taxonomy_two_stage_runpod_startup.py",
        "src/msc_project/experiments/taxonomy_two_stage_artifacts.py",
        "src/msc_project/experiments/taxonomy_two_stage_formal.py",
        "src/msc_project/experiments/verified_artifact_sync.py",
    ]
    missing = [value for value in required_files if not (PROJECT_ROOT / value).is_file()]
    checks["required_source_files"] = not missing
    if missing:
        failures.append("required_source_files")

    restore = _artifact_restore_drill()
    checks["artifact_restore_drill"] = restore.get("status") == "pass"
    if not checks["artifact_restore_drill"]:
        failures.append("artifact_restore_drill")

    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    try:
        upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        remote_head = _git("rev-parse", "@{u}")
    except subprocess.CalledProcessError:
        upstream = ""
        remote_head = ""
    git_synced = bool(upstream and head == remote_head)
    checks["git_synced"] = git_synced
    if args.require_git_synced and not git_synced:
        failures.append("git_synced")
    status_lines = _git("status", "--porcelain").splitlines()
    scientific_prefixes = (
        "configs/",
        "scripts/",
        "src/",
        "tests/",
        "requirements",
    )
    scientific_dirty = [
        line
        for line in status_lines
        if line[3:].replace("\\", "/").startswith(scientific_prefixes)
    ]
    checks["scientific_worktree_clean"] = not scientific_dirty
    if args.require_git_synced and scientific_dirty:
        failures.append("scientific_worktree_clean")

    runtime_startup: dict[str, object] | None = None
    runtime_startup_ok = False
    if args.runtime_startup_artifact is not None:
        runtime_startup = _read(args.runtime_startup_artifact)
        runtime_checks = runtime_startup.get("checks", {})
        runtime_startup_ok = bool(
            runtime_startup.get("schema_version")
            == "taxonomy_two_stage_runpod_startup_v2"
            and runtime_startup.get("status") == "pass"
            and runtime_startup.get("formal_release_ready") is True
            and runtime_startup.get("failure_count") == 0
            and runtime_startup.get("test_contract_count") == 0
            and runtime_startup.get("observed_commit") == head
            and isinstance(runtime_checks, Mapping)
            and runtime_checks.get("pinned_model_cache") is True
        )
        if not runtime_startup_ok:
            failures.append("runtime_startup_gate")
        checks["runtime_startup_gate"] = runtime_startup_ok

    local_status = "pass" if not failures else "fail"
    return {
        "schema_version": "taxonomy_two_stage_formal_preflight_v1",
        "status": local_status,
        "formal_release_ready": bool(local_status == "pass" and runtime_startup_ok),
        "local_preflight_ready": local_status == "pass",
        "runtime_startup_gate": (
            "pass" if runtime_startup_ok else "pending_on_new_formal_pod"
        ),
        "protocol_id": "taxonomy_two_stage_formal_v1",
        "head_commit": head,
        "branch": branch,
        "upstream": upstream,
        "upstream_commit": remote_head,
        "checks": checks,
        "input_hashes": observed_data,
        "job_counts": job_counts,
        "unique_training_scopes": len(scope_folds()),
        "artifact_restore_drill": restore,
        "missing_required_files": missing,
        "scientific_dirty_paths": scientific_dirty,
        "unrelated_worktree_changes": [
            line for line in status_lines if line not in scientific_dirty
        ],
        "failure_count": len(failures),
        "failures": failures,
        "test_contract_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formal two-stage cloud readiness.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-config", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--hardware-record", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-git-synced", action="store_true")
    parser.add_argument("--runtime-startup-artifact", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = run_audit(args)
    _atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
