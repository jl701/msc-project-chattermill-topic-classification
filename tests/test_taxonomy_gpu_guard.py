from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments import taxonomy_gpu_guard as guard


def _heartbeat(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "taxonomy_sync_receiver_heartbeat_v1",
                "status": "pass",
            }
        ),
        encoding="utf-8",
    )


def test_guard_runs_short_process_with_live_replication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    _heartbeat(heartbeat)
    monkeypatch.setattr(
        guard,
        "gpu_snapshot",
        lambda: {
            "timestamp": "now",
            "name": "NVIDIA GeForce RTX 4090",
            "utilization.gpu": "100",
            "memory.used": "1000",
            "memory.total": "24564",
            "temperature.gpu": "70",
            "power.draw": "400",
            "power.limit": "450",
            "clocks_event_reasons.sw_thermal_slowdown": "Not Active",
            "clocks_event_reasons.hw_thermal_slowdown": "Not Active",
            "clocks_event_reasons.hw_slowdown": "Not Active",
            "clocks_event_reasons.hw_power_brake_slowdown": "Not Active",
        },
    )
    telemetry = tmp_path / "telemetry.jsonl"
    log = tmp_path / "child.log"
    with log.open("w", encoding="utf-8") as stream:
        result = guard.run_guarded_command(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_stream=stream,
            telemetry_path=telemetry,
            output_root=tmp_path,
            replication_heartbeat=heartbeat,
            poll_interval_seconds=0.05,
            minimum_free_gib=0.0,
        )

    assert result["returncode"] == 0
    assert result["guard_stop_reason"] is None
    assert result["telemetry_samples"] >= 1
    assert telemetry.is_file()


def test_stale_replication_heartbeat_is_rejected(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    _heartbeat(heartbeat)
    stale = time.time() - 1000
    os.utime(heartbeat, (stale, stale))

    with pytest.raises(RuntimeError, match="stale"):
        guard.validate_replication_heartbeat(
            heartbeat, maximum_age_seconds=10.0
        )
