from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SCRIPT = PROJECT_ROOT / "scripts" / "build_taxonomy_cloud_plan.py"
SPEC = importlib.util.spec_from_file_location("build_taxonomy_cloud_plan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cloud_plan_is_blocked_complete_and_compute_deduplicated() -> None:
    plan = MODULE.build_plan()
    assert plan["formal_execution_blocked"] is True
    assert plan["jobs_total"] == 2611
    compute = plan["compute_summary"]
    assert compute["core_unique_training_scopes_per_method"] == 26
    assert compute["qlora_task_specific_training_runs_total"] == 102
    assert (
        compute["unique_pair_scores"]["qlora_total_including_tuning_and_extra_seeds"]
        == 5_436_984
    )
    assert (
        compute["unique_pair_scores"]["frozen_qwen_plus_qlora_total"]
        == 8_095_956
    )
    assert plan["job_counts_by_executor"] == {
        "cloud_control": 230,
        "cloud_gpu": 294,
        "local_cpu": 1733,
        "local_gpu": 354,
    }
    scores = compute["unique_pair_scores"]
    assert scores["frozen_qwen_uncached_core_total"] == 2_658_972
    assert scores["frozen_qwen_cached_unique_validation_inputs"] == 38_052
    assert scores["frozen_qwen_cached_unique_test_inputs"] == 114_264
    assert scores["frozen_qwen_cached_unique_total_inputs"] == 152_316
    assert scores["frozen_qwen_cache_inference_reduction_fraction"] > 0.94
    assert {
        value["executor"]
        for value in plan["jobs"]
        if "--method" in value["argv"]
        and value["argv"][value["argv"].index("--method") + 1]
        == "qwen_candidate_pair_qlora"
    } == {"cloud_gpu", "cloud_control"}

    jobs = plan["jobs"]
    formal_l2_validation_scores = [
        value
        for value in jobs
        if value["stage"] == "formal-score-validation"
        and "-l2-" in value["job_id"]
    ]
    assert formal_l2_validation_scores == []
    train_jobs = [value for value in jobs if value["stage"] == "formal-train"]
    assert all(value["official_splits_opened"] == ["train"] for value in train_jobs)
