"""Audit measured cloud-gate artifacts without opening the official test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ARTIFACTS = {
    "frozen_qwen_few_shot": "frozen_qwen_few_shot.json",
    "distilbert_true_two_stage": "distilbert_true_two_stage.json",
    "qwen_true_two_stage_qlora": "qwen_true_two_stage_qlora.json",
}
LOGS = (
    "frozen_qwen_few_shot_attempt2.log",
    "distilbert_true_two_stage.log",
    "qwen_true_two_stage_qlora.log",
)
ANOMALY_PATTERN = re.compile(
    r"out of memory|cuda error|traceback|\bnan\b|\binf(?:inity)?\b",
    re.IGNORECASE,
)
ALLOWED_SPLITS = {"train", "validation"}
THERMAL_BITS = 0x20 | 0x40
HARDWARE_SLOWDOWN_BIT = 0x8
POWER_BRAKE_BIT = 0x80


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


def _telemetry_rows(path: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            row = {key: value.strip() for key, value in source.items()}
            parsed: dict[str, object] = dict(row)
            parsed["dt"] = datetime.strptime(
                row["timestamp"], "%Y/%m/%d %H:%M:%S.%f"
            ).replace(tzinfo=timezone.utc)
            for key in (
                "gpu_util_pct",
                "memory_util_pct",
                "memory_used_mib",
                "memory_total_mib",
                "temp_c",
                "power_w",
                "power_limit_w",
                "graphics_clock_mhz",
                "memory_clock_mhz",
            ):
                parsed[key] = float(row[key])
            parsed["mask"] = int(row["clock_event_reasons_active"], 16)
            output.append(parsed)
    if not output:
        raise ValueError("GPU telemetry is empty.")
    return output


def _telemetry_stats(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    values = list(rows)
    masks = [int(row["mask"]) for row in values]
    util = [float(row["gpu_util_pct"]) for row in values]
    return {
        "samples": len(values),
        "gpu_util_avg_pct": round(sum(util) / len(util), 3) if util else None,
        "gpu_util_max_pct": max(util, default=None),
        "memory_used_max_mib": max(
            (float(row["memory_used_mib"]) for row in values), default=None
        ),
        "temperature_max_c": max(
            (float(row["temp_c"]) for row in values), default=None
        ),
        "power_max_w": max(
            (float(row["power_w"]) for row in values), default=None
        ),
        "thermal_slowdown_samples": sum(
            bool(mask & THERMAL_BITS) for mask in masks
        ),
        "hw_slowdown_samples": sum(
            bool(mask & HARDWARE_SLOWDOWN_BIT) for mask in masks
        ),
        "power_brake_samples": sum(bool(mask & POWER_BRAKE_BIT) for mask in masks),
        "event_masks": sorted({f"0x{mask:016x}" for mask in masks}),
    }


def audit_gate(
    *,
    manifest_path: Path,
    artifact_dir: Path,
    telemetry_path: Path,
    log_dir: Path,
) -> dict[str, object]:
    manifest = _read_object(manifest_path)
    artifact_paths = {
        name: artifact_dir / filename for name, filename in ARTIFACTS.items()
    }
    artifacts = {name: _read_object(path) for name, path in artifact_paths.items()}

    manifest_check = {
        "status": manifest.get("status"),
        "include_official_test": manifest.get("include_official_test"),
        "test_contract_count": manifest.get("test_contract_count"),
        "failure_count": manifest.get("failure_count"),
        "unique_training_scopes": len(manifest.get("unique_training_scopes", [])),
        "benchmark_commands": len(
            manifest.get("benchmark", {}).get("commands", [])
        ),
        "formal_result": manifest.get("benchmark", {}).get("formal_result"),
        "data_files": sorted(manifest.get("data_files", {})),
        "sha256": _sha256(manifest_path),
    }
    artifact_checks: dict[str, dict[str, object]] = {}
    for name, value in artifacts.items():
        splits = list(value.get("official_splits_opened", []))
        artifact_checks[name] = {
            "status": value.get("status"),
            "test_contract_count": value.get("test_contract_count", 0),
            "failure_count": value.get("failure_count", 0),
            "finite_json_numbers": _finite_tree(value),
            "official_splits_opened": splits,
            "official_splits_safe": set(splits) <= ALLOWED_SPLITS,
            "sha256": _sha256(artifact_paths[name]),
            "elapsed_seconds": value.get("elapsed_seconds"),
            "cuda_peak_memory_bytes": value.get("cuda_peak_memory_bytes"),
        }

    frozen_scores = artifacts["frozen_qwen_few_shot"]["score_checks"]
    distilbert_scores = artifacts["distilbert_true_two_stage"]["score_checks"]
    collapse_checks = {
        "frozen_aspect_nonconstant": (
            float(frozen_scores["aspect_max"])
            - float(frozen_scores["aspect_min"])
            > 1e-8
        ),
        "frozen_sentiment_nonconstant": (
            float(frozen_scores["sentiment_max"])
            - float(frozen_scores["sentiment_min"])
            > 1e-8
        ),
        "distilbert_aspect_nonconstant": (
            float(distilbert_scores["aspect_max"])
            - float(distilbert_scores["aspect_min"])
            > 1e-8
        ),
        "qlora_post_update_shapes": artifacts["qwen_true_two_stage_qlora"].get(
            "post_update_score_shapes"
        ),
        "qlora_scope_note": (
            "One post-update prompt per head checks finite execution; population "
            "collapse remains a stop condition for formal runs."
        ),
    }

    telemetry = _telemetry_rows(telemetry_path)
    run_stats: dict[str, dict[str, object]] = {}
    for name, artifact in artifacts.items():
        completed_at = datetime.fromisoformat(str(artifact["completed_at"]))
        started_at = completed_at - timedelta(
            seconds=float(artifact["elapsed_seconds"])
        )
        samples = [
            row
            for row in telemetry
            if started_at <= row["dt"] <= completed_at
        ]
        run_stats[name] = {
            "start_utc": started_at.isoformat(),
            "end_utc": completed_at.isoformat(),
            **_telemetry_stats(samples),
        }
    telemetry_check = {
        "all": _telemetry_stats(telemetry),
        "runs": run_stats,
    }
    run_thermal_samples = sum(
        int(value["thermal_slowdown_samples"]) for value in run_stats.values()
    )
    telemetry_check["thermal_slowdown_samples_during_runs"] = run_thermal_samples
    telemetry_check["thermal_slowdown_samples_outside_runs"] = max(
        0,
        int(telemetry_check["all"]["thermal_slowdown_samples"])
        - run_thermal_samples,
    )
    telemetry_check["thermal_policy"] = (
        "Admission is based on measured model-run windows; isolated post-run "
        "clock-transition masks remain visible but do not count as repeated "
        "thermal throttling."
    )

    log_hits = {}
    for name in LOGS:
        text = (log_dir / name).read_text(encoding="utf-8", errors="replace")
        log_hits[name] = sorted({match.group(0) for match in ANOMALY_PATTERN.finditer(text)})

    manifest_ok = (
        manifest_check["status"] == "pass"
        and manifest_check["include_official_test"] is False
        and manifest_check["test_contract_count"] == 0
        and manifest_check["failure_count"] == 0
        and manifest_check["unique_training_scopes"] == 26
        and manifest_check["benchmark_commands"] == 3
        and manifest_check["formal_result"] is False
        and manifest_check["data_files"] == ["train.csv", "validation.csv"]
    )
    artifacts_ok = all(
        value["status"] == "pass"
        and value["test_contract_count"] == 0
        and value["failure_count"] == 0
        and value["finite_json_numbers"] is True
        and value["official_splits_safe"] is True
        for value in artifact_checks.values()
    )
    aggregate_telemetry = telemetry_check["all"]
    thermal_ok = (
        telemetry_check["thermal_slowdown_samples_during_runs"] == 0
        and aggregate_telemetry["hw_slowdown_samples"] == 0
        and aggregate_telemetry["power_brake_samples"] == 0
    )
    logs_ok = not any(log_hits.values())
    collapse_ok = all(
        collapse_checks[key] is True
        for key in (
            "frozen_aspect_nonconstant",
            "frozen_sentiment_nonconstant",
            "distilbert_aspect_nonconstant",
        )
    )
    status = (
        "pass"
        if all((manifest_ok, artifacts_ok, thermal_ok, logs_ok, collapse_ok))
        else "fail"
    )
    return {
        "schema_version": "taxonomy_two_stage_cloud_gate_runtime_audit_v1",
        "status": status,
        "manifest_ok": manifest_ok,
        "artifacts_ok": artifacts_ok,
        "thermal_ok": thermal_ok,
        "logs_ok": logs_ok,
        "collapse_checks_ok": collapse_ok,
        "checks": {
            "manifest": manifest_check,
            "artifacts": artifact_checks,
            "collapse_checks": collapse_checks,
            "telemetry": telemetry_check,
            "log_anomaly_hits": log_hits,
        },
    }


def _write_atomic(path: Path, value: Mapping[str, object]) -> None:
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_gate(
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        telemetry_path=args.telemetry,
        log_dir=args.log_dir,
    )
    _write_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
