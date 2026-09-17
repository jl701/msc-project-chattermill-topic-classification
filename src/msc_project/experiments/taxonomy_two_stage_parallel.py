"""Fail-closed worker partitioning for the three-GPU formal campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_two_stage_formal import (
    PROTOCOL_ID,
    TRAINABLE_METHODS,
    FormalJob,
    build_formal_jobs,
    formal_job_id,
    scope_folds,
    validate_formal_job_subset,
)


PARALLEL_PLAN_SCHEMA = "taxonomy_two_stage_parallel_plan_v1"
WORKER_MANIFEST_SCHEMA = "taxonomy_two_stage_worker_manifest_v1"


def load_parallel_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Parallel execution plan must be a JSON object.")
    validate_parallel_plan(value)
    return value


def _workers(value: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = value.get("workers")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("The registered parallel plan must contain three workers.")
    if not all(isinstance(worker, Mapping) for worker in raw):
        raise ValueError("Parallel worker records must be objects.")
    return list(raw)


def validate_parallel_plan(value: Mapping[str, object]) -> dict[str, object]:
    if value.get("schema_version") != PARALLEL_PLAN_SCHEMA:
        raise ValueError("Unexpected parallel-plan schema.")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Parallel plan belongs to another protocol.")
    if value.get("status") != "preregistered_before_parallel_execution":
        raise ValueError("Parallel plan was not frozen before execution.")
    if list(value.get("allowed_splits", [])) != ["train", "validation"]:
        raise ValueError("Parallel workers may load only train and validation.")
    if value.get("include_official_test") is not False:
        raise ValueError("Official test entered the parallel plan.")
    if value.get("test_contract_count") != 0:
        raise ValueError("Parallel plan contains a test contract.")

    registered = tuple(scope_folds())
    registered_set = set(registered)
    workers = _workers(value)
    worker_ids = [str(worker.get("worker_id", "")) for worker in workers]
    if any(not worker_id for worker_id in worker_ids) or len(set(worker_ids)) != 3:
        raise ValueError("Parallel worker IDs must be non-empty and unique.")

    assignments: dict[str, list[str]] = {method: [] for method in TRAINABLE_METHODS}
    frozen_assignments: list[str] = []
    for worker in workers:
        method_id = str(worker.get("method_id", ""))
        if method_id not in TRAINABLE_METHODS:
            raise ValueError(f"Unknown parallel worker method: {method_id!r}")
        raw_scopes = worker.get("training_scope_ids")
        if not isinstance(raw_scopes, list) or not raw_scopes:
            raise ValueError("Every trainable worker needs assigned scopes.")
        scopes = [str(scope) for scope in raw_scopes]
        if len(scopes) != len(set(scopes)) or set(scopes) - registered_set:
            raise ValueError(f"Invalid scope assignment for {worker['worker_id']}.")
        assignments[method_id].extend(scopes)
        raw_frozen = worker.get("frozen_qwen_few_shot_scope_ids", [])
        if not isinstance(raw_frozen, list):
            raise ValueError("Frozen-Qwen scope assignment must be a list.")
        frozen = [str(scope) for scope in raw_frozen]
        if len(frozen) != len(set(frozen)) or set(frozen) - registered_set:
            raise ValueError("Frozen-Qwen worker scope assignment is invalid.")
        frozen_assignments.extend(frozen)

    for method_id, scopes in assignments.items():
        if len(scopes) != len(set(scopes)) or set(scopes) != registered_set:
            raise ValueError(
                f"Parallel workers do not form an exact scope partition for {method_id}."
            )
    if len(frozen_assignments) != len(set(frozen_assignments)) or set(
        frozen_assignments
    ) != registered_set:
        raise ValueError("Frozen-Qwen workers do not cover every scope exactly once.")

    return {
        "worker_count": 3,
        "unique_training_scopes": len(registered),
        "trainable_method_count": len(assignments),
        "frozen_scope_count": len(frozen_assignments),
        "test_contract_count": 0,
        "parallel_plan_sha256": canonical_sha256(value),
    }


def worker_record(
    plan: Mapping[str, object], worker_id: str
) -> Mapping[str, object]:
    validate_parallel_plan(plan)
    matches = [
        worker
        for worker in _workers(plan)
        if str(worker.get("worker_id")) == str(worker_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown parallel worker ID: {worker_id!r}")
    return matches[0]


def trainable_worker_jobs(
    plan: Mapping[str, object],
    worker_id: str,
    *,
    output_root: str,
    data_dir: str,
) -> list[FormalJob]:
    worker = worker_record(plan, worker_id)
    jobs = build_formal_jobs(
        output_root=output_root,
        data_dir=data_dir,
        methods=(str(worker["method_id"]),),
        training_scope_ids=tuple(str(value) for value in worker["training_scope_ids"]),
    )
    validate_formal_job_subset(jobs)
    return jobs


def build_worker_manifest(
    plan: Mapping[str, object],
    worker_id: str,
    *,
    output_root: str,
    data_dir: str,
) -> dict[str, object]:
    worker = worker_record(plan, worker_id)
    jobs = trainable_worker_jobs(
        plan,
        worker_id,
        output_root=output_root,
        data_dir=data_dir,
    )
    payload: dict[str, object] = {
        "schema_version": WORKER_MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "parallel_plan_sha256": canonical_sha256(plan),
        "worker_id": worker_id,
        "role": str(worker["role"]),
        "method_id": str(worker["method_id"]),
        "training_scope_ids": [str(value) for value in worker["training_scope_ids"]],
        "frozen_qwen_few_shot_scope_ids": [
            str(value) for value in worker.get("frozen_qwen_few_shot_scope_ids", [])
        ],
        "output_root": str(output_root),
        "data_dir": str(data_dir),
        "job_ids": [formal_job_id(job) for job in jobs],
        "job_count": len(jobs),
        "failure_count": 0,
        "test_contract_count": 0,
    }
    payload["worker_manifest_sha256"] = canonical_sha256(payload)
    return payload


def validate_worker_manifest(
    manifest: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    if manifest.get("schema_version") != WORKER_MANIFEST_SCHEMA:
        raise ValueError("Unexpected worker-manifest schema.")
    expected_hash = str(manifest.get("worker_manifest_sha256", ""))
    unhashed = dict(manifest)
    unhashed.pop("worker_manifest_sha256", None)
    if expected_hash != canonical_sha256(unhashed):
        raise ValueError("Worker manifest content hash mismatch.")
    if manifest.get("parallel_plan_sha256") != canonical_sha256(plan):
        raise ValueError("Worker manifest belongs to another parallel plan.")
    if manifest.get("failure_count") != 0 or manifest.get("test_contract_count") != 0:
        raise ValueError("Worker manifest contains a failure or test contract.")
    expected = build_worker_manifest(
        plan,
        str(manifest.get("worker_id", "")),
        output_root=str(manifest.get("output_root", "")),
        data_dir=str(manifest.get("data_dir", "")),
    )
    if dict(manifest) != expected:
        raise ValueError("Worker manifest differs from its registered assignment.")
    return expected


def audit_worker_union(
    manifests: Sequence[Mapping[str, object]],
    plan: Mapping[str, object],
) -> dict[str, object]:
    validate_parallel_plan(plan)
    validated = [validate_worker_manifest(manifest, plan) for manifest in manifests]
    expected_ids = {str(worker["worker_id"]) for worker in _workers(plan)}
    observed_ids = {str(manifest["worker_id"]) for manifest in validated}
    if len(validated) != len(observed_ids) or observed_ids != expected_ids:
        raise ValueError("Worker manifest set is incomplete or duplicated.")
    job_ids = [
        str(job_id)
        for manifest in validated
        for job_id in manifest["job_ids"]  # type: ignore[index]
    ]
    full_jobs = build_formal_jobs(output_root="<root>", data_dir="<data>")
    expected_job_ids = {formal_job_id(job) for job in full_jobs}
    if len(job_ids) != len(set(job_ids)) or set(job_ids) != expected_job_ids:
        raise ValueError("Worker jobs do not exactly equal the frozen formal graph.")
    return {
        "schema_version": "taxonomy_two_stage_worker_union_audit_v1",
        "status": "pass",
        "parallel_plan_sha256": canonical_sha256(plan),
        "worker_count": len(validated),
        "trainable_job_count": len(job_ids),
        "unique_trainable_job_count": len(set(job_ids)),
        "missing_job_count": 0,
        "duplicate_job_count": 0,
        "failure_count": 0,
        "test_contract_count": 0,
    }
