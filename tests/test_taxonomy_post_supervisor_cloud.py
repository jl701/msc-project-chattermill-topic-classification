from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from msc_project.experiments.taxonomy_post_supervisor_cloud import (
    audit_worker_union,
    build_formal_jobs,
    build_worker_manifest,
    expected_result_assignments,
    load_parallel_plan,
    plan_payload,
    scope_folds,
    runtime_cost_forecast,
    validate_parallel_plan,
    validate_safety_config,
    validate_worker_manifest,
    workload_contract_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "configs/experiments/taxonomy_two_stage_three_gpu_parallel_v2.json"
SAFETY_PATH = PROJECT_ROOT / "configs/experiments/taxonomy_two_stage_cloud_execution_safety_v2.json"


def test_formal_v2_graph_has_exact_reduced_counts_and_no_test_work() -> None:
    payload = plan_payload(build_formal_jobs())
    assert len(scope_folds()) == 15
    assert payload["job_counts"] == {
        "train-candidate": 90,
        "select-scope": 30,
        "score-selected": 30,
    }
    assert len(payload["jobs"]) == 150
    assert payload["fold_condition_count_per_method"] == 27
    assert payload["all_method_result_payload_count"] == 81
    assert payload["include_official_test"] is False
    assert payload["test_contract_count"] == 0
    assert not any(
        "test.csv" in token.lower() or token == "--include-official-test"
        for job in payload["jobs"]
        for token in job["command"]
    )


def test_parallel_v2_plan_is_an_exact_dependency_closed_union() -> None:
    plan = load_parallel_plan(PLAN_PATH)
    assert validate_parallel_plan(plan)["unique_training_scopes"] == 15
    manifests = [
        build_worker_manifest(
            plan,
            worker["worker_id"],
            output_root=f"/workspace/formal/{worker['worker_id']}",
            data_dir="/workspace/data/fabsa",
        )
        for worker in plan["workers"]
    ]
    audit = audit_worker_union(manifests, plan)
    assert audit["status"] == "pass"
    assert audit["trainable_job_count"] == 150
    assert audit["result_payload_count"] == 81
    assert all(validate_worker_manifest(manifest, plan) for manifest in manifests)


def test_parallel_v2_result_assignment_is_27_per_method() -> None:
    assignments = expected_result_assignments(load_parallel_plan(PLAN_PATH))
    assert Counter(key[0] for key in assignments) == {
        "distilbert_review_candidate_cross_encoder": 27,
        "qwen_candidate_pair_qlora": 27,
        "frozen_qwen_few_shot": 27,
    }


def test_parallel_v2_rejects_overlap() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(plan)
    changed["workers"][1]["training_scope_ids"][0] = changed["workers"][0]["training_scope_ids"][0]
    with pytest.raises(ValueError, match="exact partition"):
        validate_parallel_plan(changed)


def test_safety_v2_counts_match_frozen_graph() -> None:
    value = json.loads(SAFETY_PATH.read_text(encoding="utf-8"))
    audit = validate_safety_config(value)
    assert audit["status"] == "pass"
    assert audit["trainable_job_count"] == 150
    assert audit["result_payload_count"] == 81
    assert audit["test_contract_count"] == 0


def test_formal_v2_workload_and_forecast_are_exact() -> None:
    counts = workload_contract_counts()
    assert counts == {
        "unique_training_scopes": 15,
        "full_unique_candidate_contracts": 192,
        "seen_selection_candidate_contracts": 160,
        "selected_model_additional_candidate_contracts": 32,
    }
    forecast = runtime_cost_forecast(
        counts,
        unique_validation_texts=1042,
        frozen_prompts_per_second=26.61106371747323,
        qlora_prompts_per_second=57.32711937108574,
        qlora_training_examples_per_second=5.562192183691083,
        distilbert_prompts_per_second=1159.6473522899469,
        distilbert_training_examples_per_second=176.80911045647082,
        price_per_hour_usd=0.75,
    )
    assert forecast["workload"]["qlora_training_examples"] == 184320
    assert 24.5 < forecast["total_planned_gpu_hours"] < 25.0
    assert 18.0 < forecast["total_planned_cost_usd"] < 20.0
