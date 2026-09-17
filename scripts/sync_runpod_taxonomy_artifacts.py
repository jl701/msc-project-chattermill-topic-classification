"""Continuously mirror completed RunPod artifact units with SHA-256 verification."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.verified_artifact_sync import (
    receive_artifact_unit,
    sync_from_local_source,
    validate_artifact_unit_manifest,
)


def _ssh_base(args: argparse.Namespace) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-p",
        str(args.ssh_port),
    ]
    if args.identity_file is not None:
        command.extend(["-i", str(args.identity_file.resolve())])
    command.append(args.ssh_host)
    return command


def _scp_base(args: argparse.Namespace) -> list[str]:
    command = [
        "scp",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-P",
        str(args.ssh_port),
    ]
    if args.identity_file is not None:
        command.extend(["-i", str(args.identity_file.resolve())])
    return command


def _remote_root(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("--remote-root must be an absolute normalised POSIX path.")
    return path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _remote_manifest_names(args: argparse.Namespace) -> list[str]:
    root = _remote_root(args.remote_root)
    directory = (root / "_sync" / "units").as_posix()
    remote_command = (
        f"if [ -d {shlex.quote(directory)} ]; then "
        f"find {shlex.quote(directory)} -maxdepth 1 -type f -name '*.json' -print; fi"
    )
    result = _run([*_ssh_base(args), remote_command])
    names: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = PurePosixPath(line.strip())
        if path.parent.as_posix() != directory or path.suffix != ".json":
            raise RuntimeError(f"Unexpected remote manifest path: {line!r}")
        names.append(path.name)
    return sorted(set(names))


def _scp(
    args: argparse.Namespace, remote_relative: PurePosixPath, target: Path
) -> None:
    root = _remote_root(args.remote_root)
    remote_path = (root / remote_relative).as_posix()
    target.parent.mkdir(parents=True, exist_ok=True)
    _run([*_scp_base(args), f"{args.ssh_host}:{remote_path}", str(target)])


def sync_remote_once(
    args: argparse.Namespace, *, heartbeat_cycle: int | None = None
) -> dict[str, int]:
    counts = {"copied": 0, "already_complete": 0}
    last_heartbeat = time.monotonic()
    if heartbeat_cycle is not None:
        _write_remote_heartbeat(args, cycle=heartbeat_cycle, counts=counts)
    with tempfile.TemporaryDirectory(prefix="taxonomy-sync-manifest-") as temporary:
        temporary_root = Path(temporary)
        for name in _remote_manifest_names(args):
            manifest_target = temporary_root / name
            _scp(args, PurePosixPath("_sync") / "units" / name, manifest_target)
            manifest = validate_artifact_unit_manifest(
                json.loads(manifest_target.read_text(encoding="utf-8"))
            )

            def fetch(relative: str, target: Path) -> None:
                _scp(args, PurePosixPath(relative), target)

            status = receive_artifact_unit(
                manifest,
                local_root=args.local_root.resolve(),
                fetch_file=fetch,
            )
            counts[status] += 1
            now = time.monotonic()
            if (
                heartbeat_cycle is not None
                and now - last_heartbeat >= args.heartbeat_refresh_seconds
            ):
                _write_remote_heartbeat(
                    args,
                    cycle=heartbeat_cycle,
                    counts=counts,
                )
                last_heartbeat = now
    return counts


def _write_remote_heartbeat(
    args: argparse.Namespace, *, cycle: int, counts: dict[str, int]
) -> None:
    root = _remote_root(args.remote_root)
    relative = PurePosixPath(args.remote_heartbeat_relative)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or not relative.parts
        or relative.parts[0] != "_sync"
    ):
        raise ValueError("Remote heartbeat must be a safe path beneath _sync.")
    path = root / relative
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(
        {
            "schema_version": "taxonomy_sync_receiver_heartbeat_v1",
            "status": "pass",
            "cycle": cycle,
            "at": datetime.now(timezone.utc).isoformat(),
            "copied_units": counts["copied"],
            "already_complete_units": counts["already_complete"],
        },
        sort_keys=True,
    )
    remote_command = (
        f"mkdir -p {shlex.quote(path.parent.as_posix())} && "
        f"printf %s {shlex.quote(payload)} > {shlex.quote(temporary.as_posix())} && "
        f"mv {shlex.quote(temporary.as_posix())} {shlex.quote(path.as_posix())}"
    )
    _run([*_ssh_base(args), remote_command])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror only completed immutable taxonomy units from RunPod and "
            "verify every byte before local publication."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-root",
        type=Path,
        help="Local source root for a restore drill; no SSH is used.",
    )
    source.add_argument(
        "--ssh-host",
        help="OpenSSH destination such as root@203.0.113.10.",
    )
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument(
        "--remote-root",
        default="/workspace/taxonomy_two_stage_formal_v1",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        required=True,
        help="Local destination root outside Git-tracked source files.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Polling interval; zero performs one cycle.",
    )
    parser.add_argument(
        "--remote-heartbeat-relative",
        default="_sync/receiver_heartbeat.json",
        help="Remote health heartbeat checked by the formal executor.",
    )
    parser.add_argument(
        "--heartbeat-refresh-seconds",
        type=float,
        default=120.0,
        help=(
            "Refresh the remote heartbeat during long verification cycles so "
            "the guard can distinguish active SHA-256 verification from a "
            "failed receiver."
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        help="Optional bounded cycle count for supervised runs and tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.ssh_port <= 65535:
        raise ValueError("--ssh-port is invalid.")
    if args.interval_seconds < 0:
        raise ValueError("--interval-seconds must be non-negative.")
    if args.heartbeat_refresh_seconds <= 0:
        raise ValueError("--heartbeat-refresh-seconds must be positive.")
    if args.max_cycles is not None and args.max_cycles < 1:
        raise ValueError("--max-cycles must be positive.")
    if args.ssh_host:
        _remote_root(args.remote_root)
    cycles = 0
    while True:
        started = datetime.now(timezone.utc).isoformat()
        next_cycle = cycles + 1
        counts = (
            sync_from_local_source(
                args.source_root.resolve(), args.local_root.resolve()
            )
            if args.source_root is not None
            else sync_remote_once(args, heartbeat_cycle=next_cycle)
        )
        cycles = next_cycle
        if args.ssh_host:
            _write_remote_heartbeat(args, cycle=cycles, counts=counts)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "cycle": cycles,
                    "started_at": started,
                    "copied_units": counts["copied"],
                    "already_complete_units": counts["already_complete"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.interval_seconds == 0 or (
            args.max_cycles is not None and cycles >= args.max_cycles
        ):
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
