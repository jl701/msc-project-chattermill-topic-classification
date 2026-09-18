"""Fail-closed audit for the representative two-stage cloud benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT / "outputs/experimental/taxonomy_two_stage_cloud_v1"
)
EXPECTED_WORKLOAD = {
    "full_unique_candidate_contracts": 372,
    "seen_selection_candidate_contracts": 270,
    "selected_model_additional_candidate_contracts": 102,
    "unique_training_scopes": 26,
    "validation_rows": 1057,
    "unique_validation_texts": 1042,
    "frozen_prompts_per_arm": 775_248,
    "trainable_tuning_prompts": 1_688_040,
    "trainable_selected_model_additional_prompts": 212_568,
    "trainable_total_prompts_per_method": 1_900_608,
    "qlora_training_examples": 319_488,
    "qlora_optimizer_steps": 39_936,
    "distilbert_training_examples_across_three_epochs": 958_464,
    "distilbert_optimizer_steps": 29_952,
}
THERMAL_FIELDS = (
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _field(row: Mapping[str, str], prefix: str) -> str:
    matches = [
        value for key, value in row.items() if str(key).strip().startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Telemetry field {prefix!r} is missing or ambiguous.")
    return str(matches[0]).strip()


def _number(row: Mapping[str, str], prefix: str) -> float:
    value = _field(row, prefix)
    try:
        return float(value.split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"Telemetry field {prefix!r} is not numeric: {value!r}."
        ) from error


def audit_telemetry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Throughput telemetry contains no samples.")
    thermal_active = {
        field: sum(
            _field(row, field).casefold() not in {"not active", "n/a", "[n/a]"}
            for row in rows
        )
        for field in THERMAL_FIELDS
    }
    return {
        "samples": len(rows),
        "max_gpu_utilization_pct": max(
            _number(row, "utilization.gpu") for row in rows
        ),
        "max_memory_used_mib": max(
            _number(row, "memory.used") for row in rows
        ),
        "max_temperature_c": max(
            _number(row, "temperature.gpu") for row in rows
        ),
        "max_power_w": max(_number(row, "power.draw") for row in rows),
        "thermal_active_samples": thermal_active,
        "thermal_ok": not any(thermal_active.values()),
    }


def _audit_qwen(value: Mapping[str, Any]) -> dict[str, Any]:
    inference = value.get("inference")
    if not isinstance(inference, Mapping):
        raise ValueError("Qwen benchmark lacks inference evidence.")
    trials = inference.get("trials")
    if not isinstance(trials, list) or len(trials) < 2:
        raise ValueError("Qwen benchmark requires multiple batch-size trials.")
    rates: list[float] = []
    peaks: list[int] = []
    for raw in trials:
        if not isinstance(raw, Mapping):
            raise ValueError("Qwen trial is malformed.")
        required = (
            "batch_size",
            "prompts",
            "seconds",
            "prompts_per_second",
            "cuda_peak_memory_bytes",
            "aspect_probability_std",
            "sentiment_probability_std",
        )
        if any(key not in raw for key in required):
            raise ValueError("Qwen trial lacks required evidence.")
        if not all(
            _finite_positive(raw[key])
            for key in (
                "batch_size",
                "prompts",
                "seconds",
                "prompts_per_second",
                "cuda_peak_memory_bytes",
                "aspect_probability_std",
                "sentiment_probability_std",
            )
        ):
            raise ValueError("Qwen trial contains invalid or collapsed evidence.")
        rates.append(float(raw["prompts_per_second"]))
        peaks.append(int(raw["cuda_peak_memory_bytes"]))
    recommended = int(inference["recommended_batch_size"])
    best = trials[rates.index(max(rates))]
    if recommended != int(best["batch_size"]):
        raise ValueError("Recommended Qwen batch size is not the measured optimum.")
    return {
        "recommended_batch_size": recommended,
        "recommended_prompts_per_second": max(rates),
        "maximum_cuda_peak_memory_bytes": max(peaks),
        "trials": len(trials),
    }


def audit_benchmark(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version": "taxonomy_two_stage_cloud_throughput_benchmark_v1",
        "status": "pass",
        "include_official_test": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "formal_result": False,
        "official_splits_opened": ["train", "validation"],
    }
    mismatches = {
        key: {"expected": expected, "observed": value.get(key)}
        for key, expected in expected_top.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Benchmark top-level contract mismatch: {mismatches}")
    workload = value.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("Benchmark workload is missing.")
    changed = {
        key: {"expected": expected, "observed": workload.get(key)}
        for key, expected in EXPECTED_WORKLOAD.items()
        if workload.get(key) != expected
    }
    if changed:
        raise ValueError(f"Formal workload contract changed: {changed}")
    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != {
        "frozen_qwen_few_shot",
        "distilbert",
        "qlora",
    }:
        raise ValueError("Benchmark method set is incomplete.")
    frozen = _audit_qwen(benchmarks["frozen_qwen_few_shot"])
    qlora = _audit_qwen(benchmarks["qlora"])
    qlora_value = benchmarks["qlora"]
    distilbert = benchmarks["distilbert"]
    if not isinstance(qlora_value, Mapping) or not isinstance(distilbert, Mapping):
        raise ValueError("Trainable benchmark evidence is malformed.")
    if not _finite_positive(qlora_value.get("training_examples_per_second")):
        raise ValueError("QLoRA training throughput is invalid.")
    history = qlora_value.get("history")
    if not isinstance(history, list) or not history or any(
        not isinstance(record, Mapping)
        or not _finite_positive(record.get("train_loss"))
        for record in history
    ):
        raise ValueError("QLoRA training history is empty or non-finite.")
    for key in (
        "training_examples_per_second",
        "scoring_prompts_per_second",
        "training_cuda_peak_memory_bytes",
        "scoring_cuda_peak_memory_bytes",
    ):
        if not _finite_positive(distilbert.get(key)):
            raise ValueError(f"DistilBERT benchmark field is invalid: {key}")
    estimates = value.get("formal_estimates")
    methods = estimates.get("methods") if isinstance(estimates, Mapping) else None
    if not isinstance(methods, Mapping) or set(methods) != {
        "frozen_few_shot",
        "distilbert",
        "qlora",
    }:
        raise ValueError("Formal duration estimates are incomplete.")
    for method, estimate in methods.items():
        if not isinstance(estimate, Mapping) or not all(
            _finite_positive(estimate.get(key))
            for key in (
                "raw_seconds",
                "planned_hours_with_25pct_margin",
                "planned_cost_usd",
            )
        ):
            raise ValueError(f"Formal estimate is invalid for {method}.")
    total_memory = int(value["gpu"]["total_memory_bytes"])
    maximum_peak = max(
        frozen["maximum_cuda_peak_memory_bytes"],
        qlora["maximum_cuda_peak_memory_bytes"],
        int(distilbert["training_cuda_peak_memory_bytes"]),
        int(distilbert["scoring_cuda_peak_memory_bytes"]),
    )
    memory_fraction = maximum_peak / total_memory
    if memory_fraction >= 0.9:
        raise ValueError("Representative benchmark exceeded the 90% memory gate.")
    return {
        "workload": dict(workload),
        "frozen_qwen_few_shot": frozen,
        "qlora": qlora,
        "distilbert": {
            "training_examples_per_second": float(
                distilbert["training_examples_per_second"]
            ),
            "scoring_prompts_per_second": float(
                distilbert["scoring_prompts_per_second"]
            ),
        },
        "maximum_cuda_peak_memory_bytes": maximum_peak,
        "maximum_cuda_memory_fraction": memory_fraction,
        "formal_estimates": dict(methods),
    }


def audit_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fatal_tokens: Sequence[str] = (
        "traceback (most recent call last)",
        "cuda out of memory",
        "cuda error",
        "non-finite",
    )
    hits = [token for token in fatal_tokens if token in text.casefold()]
    return {"fatal_tokens": hits, "ok": not hits, "bytes": path.stat().st_size}


def build_audit(
    benchmark_path: Path,
    telemetry_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    benchmark = audit_benchmark(_read_object(benchmark_path))
    telemetry = audit_telemetry(telemetry_path)
    log = audit_log(log_path)
    checks = {
        "benchmark_ok": True,
        "telemetry_ok": bool(telemetry["thermal_ok"]),
        "log_ok": bool(log["ok"]),
        "test_contract_ok": True,
    }
    failures = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "taxonomy_two_stage_cloud_throughput_audit_v1",
        "status": "pass" if not failures else "fail",
        "failure_count": len(failures),
        "failures": failures,
        "test_contract_count": 0,
        "include_official_test": False,
        "inputs": {
            "benchmark": {
                "path": benchmark_path.as_posix(),
                "sha256": _sha256(benchmark_path),
            },
            "telemetry": {
                "path": telemetry_path.as_posix(),
                "sha256": _sha256(telemetry_path),
            },
            "log": {"path": log_path.as_posix(), "sha256": _sha256(log_path)},
        },
        "checks": checks,
        "benchmark": benchmark,
        "telemetry": telemetry,
        "log": log,
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_ROOT / "representative_throughput_benchmark.json",
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=DEFAULT_ROOT / "runtime_logs/throughput_gpu_telemetry.csv",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_ROOT / "runtime_logs/representative_throughput_benchmark.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "throughput_benchmark_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(args.benchmark, args.telemetry, args.log)
    _write_atomic(args.output, audit)
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
