from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_taxonomy_two_stage_4090_soak.py"
SPEC = importlib.util.spec_from_file_location("taxonomy_4090_soak_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


HEADER = (
    "timestamp,index,utilization.gpu [%],memory.used [MiB],temperature.gpu,"
    "power.draw [W],clocks_event_reasons.gpu_idle,"
    "clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,"
    "clocks_event_reasons.sw_thermal_slowdown,"
    "clocks_event_reasons.hw_thermal_slowdown,"
    "clocks_event_reasons.hw_power_brake_slowdown\n"
)


def _row(*, hardware_slowdown: str = "Not Active") -> str:
    return (
        "2026/08/17 21:00:00.000,0,98 %,6930 MiB,72,450 W,Not Active,"
        f"Active,{hardware_slowdown},Not Active,Not Active,Not Active\n"
    )


def test_telemetry_accepts_power_cap_but_rejects_hardware_slowdown(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe.csv"
    safe.write_text(HEADER + _row(), encoding="utf-8")
    result = MODULE.audit_telemetry(safe, minimum_samples=1)
    assert result["stop_fields_ok"] is True
    assert result["active_samples"]["clocks_event_reasons.sw_power_cap"] == 1

    unsafe = tmp_path / "unsafe.csv"
    unsafe.write_text(
        HEADER + _row(hardware_slowdown="Active"), encoding="utf-8"
    )
    result = MODULE.audit_telemetry(unsafe, minimum_samples=1)
    assert result["stop_fields_ok"] is False
    assert result["active_samples"]["clocks_event_reasons.hw_slowdown"] == 1


def test_telemetry_requires_registered_sample_coverage(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    path.write_text(HEADER + _row(), encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.audit_telemetry(path, minimum_samples=2)
