from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/sync_runpod_taxonomy_artifacts.py"
SPEC = importlib.util.spec_from_file_location("runpod_artifact_sync", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_long_remote_cycle_refreshes_guard_heartbeat(
    tmp_path: Path, monkeypatch
) -> None:
    args = argparse.Namespace(
        local_root=tmp_path,
        heartbeat_refresh_seconds=60.0,
    )
    heartbeat_calls: list[tuple[int, dict[str, int]]] = []

    monkeypatch.setattr(
        MODULE, "_remote_manifest_names", lambda _args: ["a.json", "b.json"]
    )

    def fake_scp(_args, _remote_relative, target: Path) -> None:
        target.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(MODULE, "_scp", fake_scp)
    monkeypatch.setattr(MODULE, "validate_artifact_unit_manifest", lambda _value: {})
    monkeypatch.setattr(
        MODULE, "receive_artifact_unit", lambda *_args, **_kwargs: "already_complete"
    )
    monotonic_values = iter([0.0, 30.0, 61.0])
    monkeypatch.setattr(MODULE.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        MODULE,
        "_write_remote_heartbeat",
        lambda _args, *, cycle, counts: heartbeat_calls.append((cycle, dict(counts))),
    )

    counts = MODULE.sync_remote_once(args, heartbeat_cycle=7)

    assert counts == {"copied": 0, "already_complete": 2}
    assert heartbeat_calls == [
        (7, {"copied": 0, "already_complete": 0}),
        (7, {"copied": 0, "already_complete": 2}),
    ]
