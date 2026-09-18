from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments import taxonomy_two_stage_formal as formal_module
from msc_project.experiments.taxonomy_two_stage_formal import (
    TRAINABLE_METHODS,
    build_formal_jobs,
    formal_job_id,
    formal_storage_budget_report,
    plan_payload,
    scope_folds,
    validate_formal_job_graph,
    validate_formal_job_subset,
    variant_map,
)


def test_scope_registry_loads_the_frozen_protocol_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = formal_module.load_precloud_config
    calls = 0

    def counted_load() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(formal_module, "load_precloud_config", counted_load)
    assert len(formal_module.scope_folds()) == 26
    assert calls == 1


def test_formal_job_graph_has_exact_registered_counts_and_no_test_work() -> None:
    jobs = build_formal_jobs()
    assert validate_formal_job_graph(jobs) == {
        "train-candidate": 156,
        "select-scope": 52,
        "score-selected": 52,
    }
    assert len(scope_folds()) == 26
    assert set(job.method_id for job in jobs) == set(TRAINABLE_METHODS)
    assert all("--include-official-test" not in job.command for job in jobs)
    assert all("test.csv" not in " ".join(job.command).lower() for job in jobs)
    payload = plan_payload(jobs)
    assert payload["include_official_test"] is False
    assert payload["test_contract_count"] == 0
    assert payload["fold_condition_count_per_method"] == 99


def test_dependency_closed_worker_subset_keeps_all_scope_jobs() -> None:
    scopes = tuple(scope_folds())[:2]
    jobs = build_formal_jobs(
        methods=("qwen_candidate_pair_qlora",),
        training_scope_ids=scopes,
    )
    assert validate_formal_job_subset(jobs) == {
        "train-candidate": 6,
        "select-scope": 2,
        "score-selected": 2,
    }
    assert len({formal_job_id(job) for job in jobs}) == 10


def test_worker_subset_rejects_a_partial_scope_dependency_chain() -> None:
    jobs = build_formal_jobs(
        methods=("qwen_candidate_pair_qlora",),
        training_scope_ids=(next(iter(scope_folds())),),
    )
    with pytest.raises(ValueError, match="dependency-complete"):
        validate_formal_job_subset(jobs[:-1])


def test_shared_scopes_really_share_heldout_and_seen_training_evidence() -> None:
    for folds in scope_folds().values():
        assert len({frozenset(fold.heldout_aspects) for fold in folds}) == 1
        assert len({frozenset(fold.seen_aspects) for fold in folds}) == 1


def test_l3_variant_map_changes_only_the_two_heldout_candidates() -> None:
    fold = next(
        fold
        for folds in scope_folds().values()
        for fold in folds
        if fold.level == "L3"
    )
    expected = {
        "NN": ("name_only", "name_only"),
        "DN": ("name_and_description", "name_only"),
        "ND": ("name_only", "name_and_description"),
        "DD": ("name_and_description", "name_and_description"),
        "RR": ("rich", "rich"),
    }
    for condition, heldout_values in expected.items():
        values = variant_map(fold, condition)
        assert tuple(values[aspect] for aspect in fold.heldout_aspects) == heldout_values
        assert {values[aspect] for aspect in fold.seen_aspects} == {
            "name_and_description"
        }


def test_formal_storage_budget_uses_declared_quota_and_conservative_growth() -> None:
    config = {
        "storage_budget": {
            "network_volume_capacity_gib": 100,
            "campaign_growth_upper_bound_gib": 30,
            "minimum_final_headroom_gib": 20,
        }
    }
    report = formal_storage_budget_report(config, observed_used_gib=20)
    assert report["status"] == "pass"
    assert report["projected_final_headroom_gib"] == 50
    assert report["source"] == "runpod_declared_quota_not_shared_backend_df"


def test_formal_storage_budget_fails_when_quota_headroom_is_insufficient() -> None:
    config = {
        "storage_budget": {
            "network_volume_capacity_gib": 100,
            "campaign_growth_upper_bound_gib": 30,
            "minimum_final_headroom_gib": 20,
        }
    }
    report = formal_storage_budget_report(config, observed_used_gib=55)
    assert report["status"] == "fail"
    assert report["projected_final_headroom_gib"] == 15
