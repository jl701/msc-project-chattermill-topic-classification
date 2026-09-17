"""Live GPU, storage, and replication guard for formal cloud jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, TextIO


GPU_FIELDS = (
    "timestamp",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()


def gpu_snapshot() -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected exactly one formal GPU; observed {len(lines)}.")
    values = [value.strip() for value in lines[0].split(",")]
    if len(values) != len(GPU_FIELDS):
        raise RuntimeError("NVIDIA telemetry field count changed.")
    return dict(zip(GPU_FIELDS, values))


def _active(value: object) -> bool:
    return str(value).strip().lower() in {"active", "yes", "true", "1"}


def validate_replication_heartbeat(path: Path, *, maximum_age_seconds: float) -> float:
    if maximum_age_seconds <= 0:
        raise ValueError("Replication heartbeat age must be positive.")
    if not path.is_file():
        raise FileNotFoundError(f"Replication receiver heartbeat is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "pass":
        raise ValueError("Replication receiver heartbeat is invalid.")
    age = time.time() - path.stat().st_mtime
    if age < -5 or age > maximum_age_seconds:
        raise RuntimeError(
            f"Replication receiver heartbeat is stale ({age:.1f} seconds)."
        )
    return age


def _stop_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def run_guarded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_stream: TextIO,
    telemetry_path: Path,
    output_root: Path,
    replication_heartbeat: Path,
    poll_interval_seconds: float = 30.0,
    maximum_temperature_c: float = 88.0,
    minimum_free_gib: float = 20.0,
    maximum_heartbeat_age_seconds: float = 900.0,
) -> dict[str, object]:
    """Run one child and terminate it on a registered infrastructure stop."""

    if poll_interval_seconds <= 0:
        raise ValueError("GPU guard polling interval must be positive.")
    validate_replication_heartbeat(
        replication_heartbeat,
        maximum_age_seconds=maximum_heartbeat_age_seconds,
    )
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        start_new_session=True,
    )
    software_thermal_samples = 0
    telemetry_errors = 0
    samples = 0
    stop_reason: str | None = None
    while process.poll() is None:
        sample: dict[str, object] = {
            "event": "guard_sample",
            "at": datetime.now(timezone.utc).isoformat(),
            "pid": process.pid,
        }
        try:
            gpu = gpu_snapshot()
            sample["gpu"] = gpu
            telemetry_errors = 0
            temperature = float(gpu["temperature.gpu"])
            if temperature >= maximum_temperature_c:
                stop_reason = f"gpu_temperature_{temperature:.1f}c"
            if _active(gpu["clocks_event_reasons.sw_thermal_slowdown"]):
                software_thermal_samples += 1
                if software_thermal_samples >= 2:
                    stop_reason = "repeated_software_thermal_slowdown"
            for field in (
                "clocks_event_reasons.hw_thermal_slowdown",
                "clocks_event_reasons.hw_slowdown",
                "clocks_event_reasons.hw_power_brake_slowdown",
            ):
                if _active(gpu[field]):
                    stop_reason = field
                    break
        except Exception as error:
            telemetry_errors += 1
            sample["gpu_error"] = repr(error)
            if telemetry_errors >= 2:
                stop_reason = "repeated_gpu_telemetry_failure"
        try:
            free_gib = shutil.disk_usage(output_root).free / (1024**3)
            sample["output_free_gib"] = free_gib
            if free_gib < minimum_free_gib:
                stop_reason = f"storage_headroom_{free_gib:.2f}gib"
        except Exception as error:
            sample["storage_error"] = repr(error)
            stop_reason = "storage_telemetry_failure"
        try:
            sample["replication_heartbeat_age_seconds"] = (
                validate_replication_heartbeat(
                    replication_heartbeat,
                    maximum_age_seconds=maximum_heartbeat_age_seconds,
                )
            )
        except Exception as error:
            sample["replication_error"] = repr(error)
            stop_reason = "replication_receiver_unhealthy"
        samples += 1
        sample["software_thermal_samples"] = software_thermal_samples
        sample["stop_reason"] = stop_reason
        _append_jsonl(telemetry_path, sample)
        if stop_reason is not None:
            _stop_process(process)
            break
        deadline = time.monotonic() + poll_interval_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return {
        "returncode": int(process.wait()),
        "guard_stop_reason": stop_reason,
        "telemetry_samples": samples,
        "software_thermal_samples": software_thermal_samples,
    }
