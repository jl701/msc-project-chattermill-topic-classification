from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_taxonomy_two_stage_cloud_throughput.py"
SPEC = importlib.util.spec_from_file_location("throughput_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _trial(batch: int, rate: float, peak: int) -> dict[str, object]:
    return {
        "batch_size": batch,
        "prompts": 96,
        "seconds": 96 / rate,
        "prompts_per_second": rate,
        "cuda_peak_memory_bytes": peak,
        "aspect_probability_std": 0.2,
        "sentiment_probability_std": 0.3,
    }


def _benchmark() -> dict[str, object]:
    qwen = {
        "inference": {
            "trials": [_trial(4, 10.0, 4_000), _trial(8, 12.0, 5_000)],
            "recommended_batch_size": 8,
        }
    }
    return {
        "schema_version": "taxonomy_two_stage_cloud_throughput_benchmark_v1",
        "status": "pass",
        "official_splits_opened": ["train", "validation"],
        "include_official_test": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "formal_result": False,
        "workload": dict(MODULE.EXPECTED_WORKLOAD),
        "gpu": {"total_memory_bytes": 24_000},
        "benchmarks": {
            "frozen_qwen_few_shot": qwen,
            "qlora": {
                **qwen,
                "training_examples_per_second": 3.0,
                "history": [{"train_loss": 2.0}],
            },
            "distilbert": {
                "training_examples_per_second": 300.0,
                "scoring_prompts_per_second": 500.0,
                "training_cuda_peak_memory_bytes": 3_000,
                "scoring_cuda_peak_memory_bytes": 2_000,
            },
        },
        "formal_estimates": {
            "methods": {
                method: {
                    "raw_seconds": 1.0,
                    "planned_hours_with_25pct_margin": 1.0,
                    "planned_cost_usd": 1.0,
                }
                for method in ("frozen_few_shot", "distilbert", "qlora")
            }
        },
    }


def test_audit_benchmark_accepts_complete_finite_evidence() -> None:
    result = MODULE.audit_benchmark(_benchmark())
    assert result["frozen_qwen_few_shot"]["recommended_batch_size"] == 8
    assert result["maximum_cuda_memory_fraction"] < 0.9


def test_audit_benchmark_rejects_test_access() -> None:
    value = _benchmark()
    value["official_splits_opened"] = ["train", "validation", "test"]
    with pytest.raises(ValueError, match="top-level contract"):
        MODULE.audit_benchmark(value)


def test_audit_telemetry_rejects_active_thermal_reason(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    path.write_text(
        "timestamp, utilization.gpu [%], memory.used [MiB], temperature.gpu, "
        "power.draw [W], clocks_event_reasons.sw_thermal_slowdown, "
        "clocks_event_reasons.hw_thermal_slowdown, "
        "clocks_event_reasons.hw_power_brake_slowdown\n"
        "now, 100, 1000, 80, 300, Active, Not Active, Not Active\n",
        encoding="utf-8",
    )
    result = MODULE.audit_telemetry(path)
    assert result["thermal_ok"] is False
    assert result["thermal_active_samples"][
        "clocks_event_reasons.sw_thermal_slowdown"
    ] == 1


def test_build_audit_hashes_inputs(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(_benchmark()), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.csv"
    telemetry_path.write_text(
        "timestamp, utilization.gpu [%], memory.used [MiB], temperature.gpu, "
        "power.draw [W], clocks_event_reasons.sw_thermal_slowdown, "
        "clocks_event_reasons.hw_thermal_slowdown, "
        "clocks_event_reasons.hw_power_brake_slowdown\n"
        "now, 100, 1000, 80, 300, Not Active, Not Active, Not Active\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "benchmark.log"
    log_path.write_text("complete\n", encoding="utf-8")
    result = MODULE.build_audit(benchmark_path, telemetry_path, log_path)
    assert result["status"] == "pass"
    assert result["failure_count"] == 0
    assert len(result["inputs"]["benchmark"]["sha256"]) == 64
