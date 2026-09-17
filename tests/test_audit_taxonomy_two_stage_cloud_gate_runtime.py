from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_taxonomy_two_stage_cloud_gate_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("cloud_gate_runtime_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_runtime_audit_passes_safe_finite_gate(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    artifacts = tmp_path / "artifacts"
    logs = tmp_path / "logs"
    telemetry = tmp_path / "telemetry.csv"
    _write(
        manifest,
        {
            "status": "pass",
            "include_official_test": False,
            "test_contract_count": 0,
            "failure_count": 0,
            "unique_training_scopes": list(range(26)),
            "benchmark": {"commands": [[], [], []], "formal_result": False},
            "data_files": {"train.csv": {}, "validation.csv": {}},
        },
    )
    common = {
        "status": "pass",
        "test_contract_count": 0,
        "official_splits_opened": ["train"],
        "completed_at": "2026-08-17T12:00:10+00:00",
        "elapsed_seconds": 5.0,
        "cuda_peak_memory_bytes": 100,
    }
    _write(
        artifacts / "frozen_qwen_few_shot.json",
        {**common, "score_checks": {"aspect_min": 0.1, "aspect_max": 0.9, "sentiment_min": 0.1, "sentiment_max": 0.8}},
    )
    _write(
        artifacts / "distilbert_true_two_stage.json",
        {**common, "score_checks": {"aspect_min": 0.4, "aspect_max": 0.6}},
    )
    _write(
        artifacts / "qwen_true_two_stage_qlora.json",
        {**common, "formal_result": False, "post_update_score_shapes": {"aspect": [1, 2], "sentiment": [1, 3]}},
    )
    logs.mkdir()
    for name in MODULE.LOGS:
        (logs / name).write_text("completed\n", encoding="utf-8")
    telemetry.write_text(
        "timestamp,name,driver_version,pstate,gpu_util_pct,memory_util_pct,memory_used_mib,memory_total_mib,temp_c,power_w,power_limit_w,graphics_clock_mhz,memory_clock_mhz,clock_event_reasons_active\n"
        "2026/08/17 12:00:07.000,GPU,1,P0,90,50,4000,24576,60,200,350,1800,9000,0x0000000000000000\n",
        encoding="utf-8",
    )
    result = MODULE.audit_gate(
        manifest_path=manifest,
        artifact_dir=artifacts,
        telemetry_path=telemetry,
        log_dir=logs,
    )
    assert result["status"] == "pass"
    assert result["thermal_ok"] is True
    assert result["artifacts_ok"] is True


def test_runtime_audit_fails_test_contract_and_thermal_slowdown(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    artifacts = tmp_path / "artifacts"
    logs = tmp_path / "logs"
    telemetry = tmp_path / "telemetry.csv"
    _write(
        manifest,
        {
            "status": "pass",
            "include_official_test": False,
            "test_contract_count": 0,
            "failure_count": 0,
            "unique_training_scopes": list(range(26)),
            "benchmark": {"commands": [[], [], []], "formal_result": False},
            "data_files": {"train.csv": {}, "validation.csv": {}},
        },
    )
    common = {
        "status": "pass",
        "test_contract_count": 0,
        "official_splits_opened": ["train"],
        "completed_at": "2026-08-17T12:00:10+00:00",
        "elapsed_seconds": 5.0,
        "cuda_peak_memory_bytes": 100,
    }
    _write(
        artifacts / "frozen_qwen_few_shot.json",
        {**common, "test_contract_count": 1, "score_checks": {"aspect_min": 0.1, "aspect_max": 0.9, "sentiment_min": 0.1, "sentiment_max": 0.8}},
    )
    _write(
        artifacts / "distilbert_true_two_stage.json",
        {**common, "score_checks": {"aspect_min": 0.4, "aspect_max": 0.6}},
    )
    _write(
        artifacts / "qwen_true_two_stage_qlora.json",
        {**common, "post_update_score_shapes": {"aspect": [1, 2], "sentiment": [1, 3]}},
    )
    logs.mkdir()
    for name in MODULE.LOGS:
        (logs / name).write_text("completed\n", encoding="utf-8")
    telemetry.write_text(
        "timestamp,name,driver_version,pstate,gpu_util_pct,memory_util_pct,memory_used_mib,memory_total_mib,temp_c,power_w,power_limit_w,graphics_clock_mhz,memory_clock_mhz,clock_event_reasons_active\n"
        "2026/08/17 12:00:07.000,GPU,1,P0,90,50,4000,24576,90,300,350,1500,9000,0x0000000000000020\n",
        encoding="utf-8",
    )
    result = MODULE.audit_gate(
        manifest_path=manifest,
        artifact_dir=artifacts,
        telemetry_path=telemetry,
        log_dir=logs,
    )
    assert result["status"] == "fail"
    assert result["thermal_ok"] is False
    assert result["artifacts_ok"] is False
