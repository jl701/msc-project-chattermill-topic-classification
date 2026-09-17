"""Fail-closed audit and receipt generation for isolated formal workers."""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_checkpoints import (
    TrainingContract,
    validate_checkpoint,
)
from msc_project.experiments.taxonomy_execution import (
    canonical_sha256,
    dataframe_sha256,
)
from msc_project.experiments.taxonomy_two_stage_formal import (
    PROTOCOL_ID,
    TRAINABLE_METHODS,
    candidate_result_path,
    formal_job_id,
    learning_rates,
    scope_folds,
    selection_path,
)
from msc_project.experiments.taxonomy_two_stage_parallel import (
    audit_worker_union,
    load_parallel_plan,
    trainable_worker_jobs,
    validate_worker_manifest,
    worker_record,
)
from msc_project.experiments.verified_artifact_sync import (
    file_sha256,
    validate_artifact_unit_manifest,
)


FROZEN_METHOD = "frozen_qwen_few_shot"
ALL_METHODS = (*TRAINABLE_METHODS, FROZEN_METHOD)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_sealed(value: Mapping[str, object], field: str, path: Path) -> None:
    expected = str(value.get(field, ""))
    payload = dict(value)
    payload.pop(field, None)
    if expected != canonical_sha256(payload):
        raise RuntimeError(f"Sealed payload hash mismatch: {path}")


def _reject_nonfinite(value: object, *, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite JSON value at {location}.")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, location=f"{location}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _reject_nonfinite(child, location=f"{location}[{index}]")


def _training_contract(value: Mapping[str, object]) -> TrainingContract:
    names = {field.name for field in fields(TrainingContract)}
    payload = {name: value[name] for name in names}
    contract = TrainingContract(**payload)  # type: ignore[arg-type]
    if value.get("contract_sha256") != contract.contract_sha256:
        raise ValueError("Candidate training contract hash mismatch.")
    return contract


def expected_result_assignments(
    plan: Mapping[str, object],
) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    """Map every registered result identity to exactly one worker."""

    assignments: dict[
        tuple[str, str, str, str], tuple[str, str]
    ] = {}
    for worker_id in (
        "worker-qlora-a",
        "worker-qlora-b",
        "worker-distil-frozen",
    ):
        worker = worker_record(plan, worker_id)
        method_id = str(worker["method_id"])
        for scope_id in worker["training_scope_ids"]:
            for fold in scope_folds()[str(scope_id)]:
                for condition in fold.conditions:
                    key = (method_id, fold.level, fold.fold_id, condition)
                    if key in assignments:
                        raise ValueError(f"Duplicate trainable result assignment: {key}")
                    assignments[key] = (worker_id, str(scope_id))
        for scope_id in worker.get("frozen_qwen_few_shot_scope_ids", []):
            for fold in scope_folds()[str(scope_id)]:
                for condition in fold.conditions:
                    key = (FROZEN_METHOD, fold.level, fold.fold_id, condition)
                    if key in assignments:
                        raise ValueError(f"Duplicate frozen result assignment: {key}")
                    assignments[key] = (worker_id, str(scope_id))
    expected = len(ALL_METHODS) * 99
    if len(assignments) != expected:
        raise ValueError(
            f"Parallel result assignment is incomplete: {len(assignments)} != {expected}."
        )
    return assignments


def _verify_sync_units(root: Path) -> dict[str, object]:
    unit_dir = root / "_sync" / "units"
    manifests = sorted(unit_dir.glob("*.json"))
    if not manifests:
        raise FileNotFoundError(f"No immutable sync units found beneath {root}.")
    unit_ids: set[str] = set()
    published: dict[str, str] = {}
    total_bytes = 0
    for manifest_path in manifests:
        manifest = validate_artifact_unit_manifest(_read_object(manifest_path))
        unit_id = str(manifest["unit_id"])
        if unit_id in unit_ids:
            raise ValueError(f"Duplicate sync unit ID: {unit_id}")
        if manifest.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"Sync unit belongs to another protocol: {manifest_path}")
        unit_ids.add(unit_id)
        for record in manifest["files"]:  # type: ignore[index]
            relative = str(record["path"])
            path = root / Path(*relative.split("/"))
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"Published artifact byte count mismatch: {path}")
            observed_hash = file_sha256(path)
            if observed_hash != str(record["sha256"]):
                raise ValueError(f"Published artifact SHA-256 mismatch: {path}")
            previous = published.get(relative)
            if previous is not None and previous != observed_hash:
                raise ValueError(f"One path was published with conflicting hashes: {relative}")
            published[relative] = observed_hash
            total_bytes += int(record["bytes"])
    return {
        "unit_count": len(manifests),
        "published_path_count": len(published),
        "declared_bytes": total_bytes,
        "published_paths": published,
    }


def _validate_score_manifest(
    manifest_path: Path, published_paths: Mapping[str, str], root: Path
) -> None:
    value = _read_object(manifest_path)
    if value.get("schema_version") != "taxonomy_two_stage_score_shard_v1":
        raise ValueError(f"Unexpected score manifest schema: {manifest_path}")
    if value.get("test_contract_count") != 0:
        raise ValueError(f"Test contract entered score manifest: {manifest_path}")
    contract = value.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"Score manifest lacks its run contract: {manifest_path}")
    if (
        contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("split") != "validation"
        or contract.get("formal") is not True
    ):
        raise ValueError(f"Invalid formal score contract: {manifest_path}")
    relative_manifest = manifest_path.relative_to(root).as_posix()
    csv_path = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".csv"))
    relative_csv = csv_path.relative_to(root).as_posix()
    if relative_manifest not in published_paths or relative_csv not in published_paths:
        raise ValueError(f"Score shard was not published atomically: {manifest_path}")
    if value.get("csv_sha256") != file_sha256(csv_path):
        raise ValueError(f"Score CSV hash mismatch: {csv_path}")
    frame = pd.read_csv(csv_path, float_precision="round_trip")
    if len(frame) != int(value.get("rows", -1)) or frame.empty:
        raise ValueError(f"Score shard row count is invalid: {csv_path}")
    scores = frame[["aspect_score", "sentiment_score"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite formal score: {csv_path}")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError(f"Formal score is outside [0, 1]: {csv_path}")
    if set(frame["split"].astype(str)) != {"validation"}:
        raise ValueError(f"Non-validation rows entered formal scores: {csv_path}")
    if int(value.get("finite_score_count", -1)) != scores.size:
        raise ValueError(f"Finite-score count mismatch: {manifest_path}")


def _validate_trainable_scope(
    root: Path,
    method_id: str,
    scope_id: str,
    published_paths: Mapping[str, str],
) -> None:
    candidates: list[dict[str, Any]] = []
    for rate in learning_rates(method_id):
        path = candidate_result_path(root, method_id, scope_id, rate)
        value = _read_object(path)
        _validate_sealed(value, "candidate_payload_sha256", path)
        _reject_nonfinite(value, location=str(path))
        if (
            value.get("protocol_id") != PROTOCOL_ID
            or value.get("method_id") != method_id
            or value.get("training_scope_id") != scope_id
            or float(value.get("learning_rate", -1)) != rate
            or value.get("failure_count") != 0
            or value.get("test_contract_count") != 0
        ):
            raise ValueError(f"Candidate identity or safety mismatch: {path}")
        if path.relative_to(root).as_posix() not in published_paths:
            raise ValueError(f"Candidate was not published as an immutable unit: {path}")
        training = value.get("training_contract")
        if not isinstance(training, Mapping):
            raise ValueError(f"Candidate lacks a training contract: {path}")
        contract = _training_contract(training)
        checkpoint = root / str(value["checkpoint_relative_path"])
        validate_checkpoint(checkpoint, contract)
        manifest = checkpoint / "checkpoint.manifest.json"
        if value.get("checkpoint_manifest_sha256") != file_sha256(manifest):
            raise ValueError(f"Candidate checkpoint-manifest hash mismatch: {path}")
        for checkpoint_file in checkpoint.rglob("*"):
            if checkpoint_file.is_file():
                relative = checkpoint_file.relative_to(root).as_posix()
                if relative not in published_paths:
                    raise ValueError(
                        f"Checkpoint file was not published atomically: {checkpoint_file}"
                    )
        candidates.append(value)
    selected_path = selection_path(root, method_id, scope_id)
    selected = _read_object(selected_path)
    _validate_sealed(selected, "selection_payload_sha256", selected_path)
    _reject_nonfinite(selected, location=str(selected_path))
    if (
        selected.get("protocol_id") != PROTOCOL_ID
        or selected.get("method_id") != method_id
        or selected.get("training_scope_id") != scope_id
        or selected.get("failure_count") != 0
        or selected.get("test_contract_count") != 0
    ):
        raise ValueError(f"Selection identity or safety mismatch: {selected_path}")
    candidate_hashes = {
        str(candidate["training_contract"]["contract_sha256"])
        for candidate in candidates
    }
    if set(selected.get("candidate_contract_sha256s", [])) != candidate_hashes:
        raise ValueError(f"Selection candidate set is incomplete: {selected_path}")
    if str(selected.get("selected_candidate_contract_sha256")) not in candidate_hashes:
        raise ValueError(f"Selection points outside its candidates: {selected_path}")
    if selected_path.relative_to(root).as_posix() not in published_paths:
        raise ValueError(f"Selection was not published atomically: {selected_path}")


def _validate_frozen_selection(
    root: Path, scope_id: str, published_paths: Mapping[str, str]
) -> None:
    path = selection_path(root, FROZEN_METHOD, scope_id)
    value = _read_object(path)
    _validate_sealed(value, "selection_payload_sha256", path)
    _reject_nonfinite(value, location=str(path))
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("method_id") != FROZEN_METHOD
        or value.get("training_scope_id") != scope_id
        or value.get("failure_count") != 0
        or value.get("test_contract_count") != 0
        or value.get("selection_partition") != "seen_validation_only"
    ):
        raise ValueError(f"Frozen selection identity or safety mismatch: {path}")
    if path.relative_to(root).as_posix() not in published_paths:
        raise ValueError(f"Frozen selection was not published atomically: {path}")


def _validate_result(
    root: Path,
    key: tuple[str, str, str, str],
    expected_scope_id: str,
    published_paths: Mapping[str, str],
) -> None:
    method_id, level, fold_id, condition = key
    path = root / "results" / method_id / level / fold_id / f"{condition}.json"
    value = _read_object(path)
    _validate_sealed(value, "result_payload_sha256", path)
    _reject_nonfinite(value, location=str(path))
    if (
        value.get("protocol_id") != PROTOCOL_ID
        or value.get("method_id") != method_id
        or value.get("level") != level
        or value.get("fold_id") != fold_id
        or value.get("condition") != condition
        or value.get("training_scope_id") != expected_scope_id
        or value.get("evaluation_partition") != "validation_only"
        or value.get("selection_partition") != "seen_validation_only"
        or value.get("failure_count") != 0
        or value.get("test_contract_count") != 0
        or int(value.get("maximum_sentiments_per_aspect", 99)) > 2
        or int(value.get("score_shards", 0)) < 1
        or int(value.get("score_rows", 0)) < 1
    ):
        raise ValueError(f"Formal result identity or safety mismatch: {path}")
    if PROTOCOL_ID == "taxonomy_two_stage_formal_v2" and level == "L2":
        views = value.get("post_supervisor_views")
        if not isinstance(views, Mapping) or set(views) != {
            "score_sha256",
            "L2_S",
            "L2_E",
        }:
            raise ValueError(f"Formal-v2 Level 2 views are incomplete: {path}")
    if path.relative_to(root).as_posix() not in published_paths:
        raise ValueError(f"Formal result was not published atomically: {path}")
    run_hash = str(value.get("run_contract_sha256", ""))
    score_dir = (
        root
        / "scores"
        / method_id
        / level
        / fold_id
        / condition
        / run_hash[:16]
    )
    manifests = sorted(score_dir.glob("shard-*.manifest.json"))
    expected_shards = int(value["score_shards"])
    if len(manifests) != expected_shards:
        raise ValueError(f"Result score-shard set is incomplete: {path}")
    frames: list[pd.DataFrame] = []
    shard_indices: set[int] = set()
    for manifest_path in manifests:
        manifest = _read_object(manifest_path)
        contract = manifest.get("contract")
        if not isinstance(contract, Mapping) or contract.get(
            "contract_sha256"
        ) != run_hash:
            raise ValueError(f"Result points to another score contract: {path}")
        shard_index = int(manifest.get("shard_index", -1))
        if shard_index in shard_indices:
            raise ValueError(f"Duplicate result score-shard index: {path}")
        shard_indices.add(shard_index)
        csv_path = manifest_path.with_name(
            manifest_path.name.replace(".manifest.json", ".csv")
        )
        frames.append(pd.read_csv(csv_path, float_precision="round_trip"))
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != int(value["score_rows"]):
        raise ValueError(f"Result score row count mismatch: {path}")
    if merged.duplicated(
        ["row_uid", "candidate_aspect", "candidate_sentiment"]
    ).any():
        raise ValueError(f"Result score shards contain duplicate identities: {path}")
    if (
        merged["aspect_score"].nunique() < 2
        or merged["sentiment_score"].nunique() < 2
    ):
        raise ValueError(f"Result score distribution collapsed: {path}")
    observed_score_hash = dataframe_sha256(
        merged,
        [
            "row_uid",
            "candidate_aspect",
            "candidate_sentiment",
            "aspect_score",
            "sentiment_score",
        ],
        sort_columns=["row_uid", "candidate_aspect", "candidate_sentiment"],
    )
    if value.get("score_sha256") != observed_score_hash:
        raise ValueError(f"Result score hash mismatch: {path}")


def audit_parallel_outputs(
    campaign_root: Path, plan_path: Path
) -> dict[str, object]:
    plan = load_parallel_plan(plan_path)
    assignments = expected_result_assignments(plan)
    worker_roots = {
        worker_id: campaign_root / worker_id
        for worker_id in (
            "worker-qlora-a",
            "worker-qlora-b",
            "worker-distil-frozen",
        )
    }
    manifests: list[dict[str, Any]] = []
    sync_audits: dict[str, dict[str, object]] = {}
    for worker_id, root in worker_roots.items():
        campaign_dir = root / "_campaign" / "workers" / worker_id
        manifest = _read_object(campaign_dir / "worker_manifest.json")
        validate_worker_manifest(manifest, plan)
        manifests.append(manifest)
        state = _read_object(campaign_dir / "state.json")
        expected_jobs = [
            formal_job_id(job)
            for job in trainable_worker_jobs(
                plan,
                worker_id,
                output_root=str(manifest["output_root"]),
                data_dir=str(manifest["data_dir"]),
            )
        ]
        if (
            state.get("schema_version")
            != "taxonomy_two_stage_parallel_campaign_state_v1"
            or state.get("status") != "complete"
            or state.get("failure_count") != 0
            or state.get("test_contract_count") != 0
            or state.get("worker_manifest_sha256")
            != manifest.get("worker_manifest_sha256")
            or list(state.get("completed_jobs", [])) != expected_jobs
            or int(state.get("completed_count", -1)) != len(expected_jobs)
        ):
            raise ValueError(f"Trainable worker state is incomplete: {worker_id}")
        sync_audits[worker_id] = _verify_sync_units(root)

    union = audit_worker_union(manifests, plan)
    frozen_root = worker_roots["worker-distil-frozen"]
    frozen_state = _read_object(
        frozen_root
        / "_campaign"
        / "workers"
        / "worker-distil-frozen"
        / FROZEN_METHOD
        / "state.json"
    )
    frozen_scopes = list(scope_folds())
    if (
        frozen_state.get("schema_version")
        != "taxonomy_frozen_qwen_few_shot_campaign_state_v1"
        or frozen_state.get("status") != "complete"
        or frozen_state.get("failure_count") != 0
        or frozen_state.get("test_contract_count") != 0
        or frozen_state.get("parallel_plan_sha256") != canonical_sha256(plan)
        or frozen_state.get("worker_manifest_sha256")
        != next(
            manifest.get("worker_manifest_sha256")
            for manifest in manifests
            if manifest.get("worker_id") == "worker-distil-frozen"
        )
        or list(frozen_state.get("completed_scopes", [])) != frozen_scopes
        or int(frozen_state.get("completed_count", -1)) != len(frozen_scopes)
    ):
        raise ValueError("Frozen-Qwen worker state is incomplete.")

    for worker_id, root in worker_roots.items():
        worker = worker_record(plan, worker_id)
        published = sync_audits[worker_id]["published_paths"]
        assert isinstance(published, Mapping)
        for scope_id in worker["training_scope_ids"]:
            _validate_trainable_scope(
                root, str(worker["method_id"]), str(scope_id), published
            )
        for scope_id in worker.get("frozen_qwen_few_shot_scope_ids", []):
            _validate_frozen_selection(root, str(scope_id), published)
        for manifest_path in sorted(root.glob("scores/**/*.manifest.json")):
            _validate_score_manifest(manifest_path, published, root)

    for key, (worker_id, scope_id) in assignments.items():
        root = worker_roots[worker_id]
        published = sync_audits[worker_id]["published_paths"]
        assert isinstance(published, Mapping)
        _validate_result(root, key, scope_id, published)

    receipt = {
        "schema_version": "taxonomy_two_stage_parallel_merge_audit_v1",
        "status": "pass",
        "protocol_id": PROTOCOL_ID,
        "parallel_plan_sha256": canonical_sha256(plan),
        "worker_union": union,
        "sync": {
            worker_id: {
                key: value
                for key, value in audit.items()
                if key != "published_paths"
            }
            for worker_id, audit in sync_audits.items()
        },
        "methods": list(ALL_METHODS),
        "training_scope_count_per_method": len(scope_folds()),
        "result_count_per_method": 99,
        "result_count": len(assignments),
        "maximum_sentiments_per_aspect": 2,
        "missing_result_count": 0,
        "duplicate_result_count": 0,
        "nonfinite_score_count": 0,
        "failure_count": 0,
        "test_contract_count": 0,
    }
    receipt["merge_audit_sha256"] = canonical_sha256(receipt)
    return receipt
