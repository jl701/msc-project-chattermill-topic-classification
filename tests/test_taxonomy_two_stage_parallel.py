from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_two_stage_parallel import (
    audit_worker_union,
    build_worker_manifest,
    load_parallel_plan,
    validate_parallel_plan,
    validate_worker_manifest,
)


PLAN_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_two_stage_three_gpu_parallel_v1.json"
)


def test_three_gpu_plan_is_an_exact_dependency_closed_union() -> None:
    plan = load_parallel_plan(PLAN_PATH)
    audit = validate_parallel_plan(plan)
    assert audit["worker_count"] == 3
    assert audit["unique_training_scopes"] == 26
    manifests = [
        build_worker_manifest(
            plan,
            worker["worker_id"],
            output_root=f"/workspace/formal/{worker['worker_id']}",
            data_dir="/workspace/fabsa_data",
        )
        for worker in plan["workers"]
    ]
    result = audit_worker_union(manifests, plan)
    assert result["status"] == "pass"
    assert result["trainable_job_count"] == 260
    assert result["missing_job_count"] == 0
    assert result["duplicate_job_count"] == 0
    assert result["test_contract_count"] == 0


def test_parallel_plan_rejects_overlap_and_missing_scope() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(plan)
    changed["workers"][1]["training_scope_ids"][0] = changed["workers"][0][
        "training_scope_ids"
    ][0]
    with pytest.raises(ValueError, match="exact scope partition"):
        validate_parallel_plan(changed)


def test_worker_manifest_is_bound_to_paths_and_assignment() -> None:
    plan = load_parallel_plan(PLAN_PATH)
    worker = plan["workers"][0]
    manifest = build_worker_manifest(
        plan,
        worker["worker_id"],
        output_root="/workspace/formal/worker-qlora-a",
        data_dir="/workspace/fabsa_data",
    )
    assert validate_worker_manifest(manifest, plan) == manifest
    changed = copy.deepcopy(manifest)
    changed["output_root"] = "/workspace/formal/conflict"
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_worker_manifest(changed, plan)
