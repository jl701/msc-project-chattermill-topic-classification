from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_taxonomy_two_stage_cloud_throughput.py"
SPEC = importlib.util.spec_from_file_location("cloud_throughput", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_contract_counts_deduplicate_shared_scopes() -> None:
    assert MODULE.candidate_contract_counts() == {
        "full_unique_candidate_contracts": 372,
        "seen_selection_candidate_contracts": 270,
        "selected_model_additional_candidate_contracts": 102,
        "unique_training_scopes": 26,
    }


def test_formal_workload_uses_exact_prompt_hash_deduplication() -> None:
    workload = MODULE.formal_workload(
        validation_rows=1057,
        unique_validation_texts=1042,
    )
    assert workload["frozen_prompts_per_arm"] == 775_248
    assert workload["trainable_tuning_prompts"] == 1_688_040
    assert workload["trainable_selected_model_additional_prompts"] == 212_568
    assert workload["trainable_total_prompts_per_method"] == 1_900_608
    assert workload["qlora_training_examples"] == 319_488
    assert workload["qlora_optimizer_steps"] == 39_936
    assert workload["distilbert_training_examples_across_three_epochs"] == 958_464
    assert workload["distilbert_optimizer_steps"] == 29_952


def test_duration_and_cost_estimate_has_planning_margin() -> None:
    workload = MODULE.formal_workload(
        validation_rows=1057,
        unique_validation_texts=1042,
    )
    estimate = MODULE.duration_and_cost_estimate(
        workload,
        frozen_prompts_per_second=10.0,
        qlora_prompts_per_second=8.0,
        qlora_training_examples_per_second=2.0,
        distilbert_prompts_per_second=100.0,
        distilbert_training_examples_per_second=50.0,
    )
    frozen = estimate["methods"]["frozen_few_shot"]
    assert frozen["raw_seconds"] == pytest.approx(77_524.8)
    assert frozen["planned_hours_with_25pct_margin"] == pytest.approx(
        77_524.8 * 1.25 / 3600
    )
    assert frozen["planned_cost_usd"] == pytest.approx(
        77_524.8 * 1.25 / 3600 * 0.24
    )


def test_duration_and_cost_estimate_uses_observed_instance_price() -> None:
    workload = MODULE.formal_workload(
        validation_rows=1057,
        unique_validation_texts=1042,
    )
    estimate = MODULE.duration_and_cost_estimate(
        workload,
        frozen_prompts_per_second=10.0,
        qlora_prompts_per_second=8.0,
        qlora_training_examples_per_second=2.0,
        distilbert_prompts_per_second=100.0,
        distilbert_training_examples_per_second=50.0,
        price_per_hour_usd=0.75,
    )
    frozen = estimate["methods"]["frozen_few_shot"]
    assert estimate["hourly_price_usd"] == pytest.approx(0.75)
    assert frozen["planned_cost_usd"] == pytest.approx(
        77_524.8 * 1.25 / 3600 * 0.75
    )


@pytest.mark.parametrize(
    ("rows", "texts"),
    [(0, 0), (10, 0), (10, 11)],
)
def test_formal_workload_rejects_invalid_validation_counts(
    rows: int, texts: int
) -> None:
    with pytest.raises(ValueError):
        MODULE.formal_workload(
            validation_rows=rows,
            unique_validation_texts=texts,
        )
