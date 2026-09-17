"""Fail-closed audit for the sustained RTX 4090 hardware soak."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping


STOP_FIELDS = (
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)
INFORMATIONAL_FIELDS = (
    "clocks_event_reasons.gpu_idle",
    "clocks_event_reasons.sw_power_cap",
)
INACTIVE_VALUES = {"not active", "n/a", "[n/a]"}
FATAL_PATTERN = re.compile(
    r"out of memory|cuda error|traceback|\bnan\b|\binf(?:inity)?\b",
    re.IGNORECASE,
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


def _finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


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


def audit_telemetry(path: Path, *, minimum_samples: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < minimum_samples:
        raise ValueError(
            f"Sustained telemetry has {len(rows)} rows; expected at least "
            f"{minimum_samples}."
        )
    active_samples = {
        field: sum(
            _field(row, field).casefold() not in INACTIVE_VALUES for row in rows
        )
        for field in STOP_FIELDS + INFORMATIONAL_FIELDS
    }
    return {
        "samples": len(rows),
        "minimum_samples": minimum_samples,
        "max_gpu_utilization_pct": max(
            _number(row, "utilization.gpu") for row in rows
        ),
        "average_gpu_utilization_pct": sum(
            _number(row, "utilization.gpu") for row in rows
        )
        / len(rows),
        "max_memory_used_mib": max(_number(row, "memory.used") for row in rows),
        "max_temperature_c": max(
            _number(row, "temperature.gpu") for row in rows
        ),
        "max_power_w": max(_number(row, "power.draw") for row in rows),
        "active_samples": active_samples,
        "stop_fields_ok": not any(active_samples[field] for field in STOP_FIELDS),
    }


def build_audit(
    artifact_path: Path,
    telemetry_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    artifact = _read_object(artifact_path)
    frozen = artifact.get("frozen_scoring", {})
    qlora = artifact.get("qlora_training", {})
    history = qlora.get("history", []) if isinstance(qlora, Mapping) else []
    history_row = history[0] if isinstance(history, list) and len(history) == 1 else {}
    elapsed = float(artifact.get("elapsed_seconds", 0.0))
    telemetry = audit_telemetry(
        telemetry_path,
        minimum_samples=max(1, int(elapsed / 2.5)),
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    fatal_tokens = sorted(
        {match.group(0).casefold() for match in FATAL_PATTERN.finditer(log_text)}
    )

    contract_ok = (
        artifact.get("status") == "pass"
        and artifact.get("official_splits_opened") == ["train", "validation"]
        and artifact.get("include_official_test") is False
        and artifact.get("test_contract_count") == 0
        and artifact.get("failure_count") == 0
        and artifact.get("formal_result") is False
        and artifact.get("fold_id") == "l2-a01"
        and artifact.get("training_scope_id") == "heldout-a01"
        and artifact.get("gpu", {}).get("name") == "NVIDIA GeForce RTX 4090"
        and _finite_tree(artifact)
    )
    frozen_ok = (
        isinstance(frozen, Mapping)
        and frozen.get("batch_size") == 8
        and frozen.get("max_length") == 1024
        and int(frozen.get("requested_seconds", 0)) == 600
        and float(frozen.get("elapsed_seconds", 0.0)) >= 600.0
        and int(frozen.get("loops", 0)) > 0
        and int(frozen.get("prompts", 0)) > 0
        and math.isfinite(float(frozen.get("prompts_per_second", math.nan)))
        and float(frozen.get("aspect_probability_std_max", 0.0)) > 1e-8
        and float(frozen.get("sentiment_probability_std_max", 0.0)) > 1e-8
    )
    qlora_ok = (
        isinstance(qlora, Mapping)
        and qlora.get("training_examples") == 4096
        and qlora.get("max_length") == 384
        and qlora.get("batch_size") == 1
        and qlora.get("gradient_accumulation_steps") == 8
        and qlora.get("epochs") == 1
        and float(qlora.get("learning_rate", 0.0)) == 5e-6
        and qlora.get("lora_r") == 4
        and qlora.get("lora_alpha") == 8
        and float(qlora.get("lora_dropout", -1.0)) == 0.05
        and qlora.get("checkpoint_retained") is False
        and qlora.get("checkpoint_path") is None
        and isinstance(history_row, Mapping)
        and history_row.get("global_step") == 512
        and history_row.get("planned_steps") == 512
        and math.isfinite(float(history_row.get("train_loss", math.nan)))
        and float(history_row.get("train_loss", -1.0)) >= 0.0
    )
    telemetry_ok = bool(telemetry["stop_fields_ok"])
    log_ok = not fatal_tokens
    status = "pass" if all((contract_ok, frozen_ok, qlora_ok, telemetry_ok, log_ok)) else "fail"
    failures = [
        name
        for name, ok in (
            ("contract", contract_ok),
            ("frozen_scoring", frozen_ok),
            ("qlora_training", qlora_ok),
            ("telemetry", telemetry_ok),
            ("log", log_ok),
        )
        if not ok
    ]
    return {
        "schema_version": "taxonomy_two_stage_4090_soak_audit_v1",
        "status": status,
        "failure_count": len(failures),
        "failures": failures,
        "test_contract_count": 0 if contract_ok else 1,
        "include_official_test": False,
        "checks": {
            "contract_ok": contract_ok,
            "frozen_scoring_ok": frozen_ok,
            "qlora_training_ok": qlora_ok,
            "telemetry_ok": telemetry_ok,
            "log_ok": log_ok,
        },
        "inputs": {
            "artifact": {"path": artifact_path.as_posix(), "sha256": _sha256(artifact_path)},
            "telemetry": {"path": telemetry_path.as_posix(), "sha256": _sha256(telemetry_path)},
            "log": {"path": log_path.as_posix(), "sha256": _sha256(log_path)},
        },
        "telemetry": telemetry,
        "fatal_log_tokens": fatal_tokens,
        "observed": {
            "frozen_prompts_per_second": frozen.get("prompts_per_second"),
            "frozen_cuda_peak_memory_bytes": frozen.get("cuda_peak_memory_bytes"),
            "qlora_training_examples_per_second": qlora.get(
                "training_examples_per_second"
            ),
            "qlora_cuda_peak_memory_bytes": qlora.get("cuda_peak_memory_bytes"),
            "qlora_train_loss": history_row.get("train_loss"),
        },
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(args.artifact, args.telemetry, args.log)
    _write_atomic(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
