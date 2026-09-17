"""Continuously receive immutable validation-only QLoRA units over SSH.

Uses the existing verified_artifact_unit_v1 publication/receipt protocol. This
receiver never runs a model, opens dataset labels, changes Pod billing, or
restarts a failed experiment. Only its scoped receiver heartbeat is written
remotely. READY_TO_STOP requires exited workers and exact verified backups.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.experiments.verified_artifact_sync import (
    file_sha256, receive_artifact_unit, validate_artifact_unit_manifest,
)

PROTOCOL = "taxonomy_policy_qlora_completion_v1"
ALLOWED_ROOTS = {"checkpoints", "selection_candidates", "selections", "scores",
                 "results", "policy_provenance", "calibration_scores"}
RUNTIME_NAMES = {"state.json", "COMPLETE.json", "FAILED.json", "environment.json",
                 "worker.log", "stdout.log", "stderr.log", "telemetry.jsonl"}
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10))}
SCOPE = re.compile(r"heldout-a(?:0[1-9]|1[0-2])")
MAX_FILE_BYTES = 8 * 1024 ** 3

INVENTORY_CODE = r'''
import hashlib,json,pathlib,sys,time
result=pathlib.Path(sys.argv[1]);runtime=pathlib.Path(sys.argv[2]);transport=pathlib.Path(sys.argv[3])
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
def read(p): return json.loads(p.read_text()) if p.is_file() else None
units={}
for p in sorted((result/'_sync/units').glob('*.json')):
    if p.is_symlink(): raise ValueError('Symlink publication manifest')
    raw=p.read_bytes()
    units[p.name]={'sha256':hashlib.sha256(raw).hexdigest(),'manifest':json.loads(raw)}
alive=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        cmd=p.read_bytes().replace(b'\0',b' ')
        if b'python' in cmd and any(n in cmd for n in [b'policy_qlora_cloud_worker.py',b'run_taxonomy_policy_qlora_completion.py']):
            alive.append(int(p.parent.name))
    except (OSError,ProcessLookupError): pass
locks=[p.name for p in runtime.glob('*.lock')]
complete=read(runtime/'COMPLETE.json');failure=read(runtime/'FAILED.json')
terminal=bool(complete and not alive and not locks)
files={}
if terminal or failure:
    for name in ['state.json','COMPLETE.json','FAILED.json','environment.json','worker.log','stdout.log','stderr.log','telemetry.jsonl']:
        p=runtime/name
        if p.is_file():
            if p.is_symlink(): raise ValueError('Symlink runtime file')
            files[name]={'bytes':p.stat().st_size,'sha256':digest(p)}
print(json.dumps({'at_unix':time.time(),'units':units,'runtime_state':read(runtime/'state.json'),
    'complete':complete,'failure':failure,'terminal':terminal,'alive_pids':alive,
    'locks':locks,'runtime_files':files,'transport_sha256':digest(transport)}))
'''


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def safe_artifact_path(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError("Unsafe artifact path characters")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or len(path.parts) < 2:
        raise ValueError("Artifact path is not canonical and relative")
    if path.parts[0] not in ALLOWED_ROOTS:
        raise ValueError("Artifact outside completion-study allowlist")
    for part in path.parts:
        if part in {".", ".."} or part.endswith(".") or part.split(".")[0].upper() in RESERVED:
            raise ValueError("Unsafe Windows artifact path")
        if any(token in part.lower() for token in ("official_test", "official-test", "label_vault", "test_labels")):
            raise ValueError("Official-test material is forbidden")
    return value


def contained_target(root: Path, relative: str) -> Path:
    base = root.resolve()
    target = base.joinpath(*PurePosixPath(relative).parts)
    try:
        target.resolve().relative_to(base)
    except ValueError as error:
        raise ValueError("Local artifact path escapes destination") from error
    cursor = target
    while cursor != base:
        if cursor.is_symlink():
            raise ValueError("Local artifact paths may not contain symlinks")
        cursor = cursor.parent
    return target


def validate_unit(value: dict, filename: str) -> dict:
    manifest = validate_artifact_unit_manifest(value)
    if manifest["protocol_id"] != PROTOCOL or filename != manifest["unit_id"] + ".json":
        raise ValueError("Published unit identity/protocol mismatch")
    lower_paths = set()
    for record in manifest["files"]:
        path = safe_artifact_path(record["path"])
        if path.casefold() in lower_paths:
            raise ValueError("Case-insensitive artifact path collision")
        lower_paths.add(path.casefold())
        if record["bytes"] > MAX_FILE_BYTES:
            raise ValueError("Artifact exceeds receiver size limit")
    return manifest


def validate_inventory(inventory: dict, transport_sha256: str) -> dict[str, dict]:
    if inventory.get("transport_sha256") != transport_sha256:
        raise ValueError("Remote transport release identity changed")
    units = inventory.get("units")
    if not isinstance(units, dict):
        raise ValueError("Missing unit inventory")
    manifests = {}
    all_paths = {}
    for filename, entry in units.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.json", filename):
            raise ValueError("Unsafe remote unit filename")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ValueError("Invalid remote manifest hash")
        manifest = validate_unit(entry["manifest"], filename)
        for record in manifest["files"]:
            identity = (record["path"], record["bytes"], record["sha256"])
            prior = all_paths.setdefault(record["path"].casefold(), identity)
            if prior != identity:
                raise ValueError("Published units conflict on shared artifact path")
        manifests[filename] = manifest
    return manifests


def verify_received_set(root: Path, manifests: dict[str, dict], *, exact: bool = False) -> None:
    receipts = root / "_sync" / "received"
    names = {p.name for p in receipts.glob("*.json")}
    if not names.issubset(manifests) or exact and names != set(manifests):
        raise ValueError("Remote unit/local receipt identity set mismatch")
    for name in names:
        local = json.loads((receipts / name).read_text(encoding="utf-8"))
        if local != manifests[name]:
            raise ValueError("Immutable receipt differs from published unit")
        for record in local["files"]:
            path = contained_target(root, record["path"])
            if not path.is_file() or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
                raise ValueError("Previously received artifact changed or disappeared")


def receive_snapshot(inventory: dict, *, transport_sha256: str, local_root: Path,
                     backup_root: Path, fetch_file, heartbeat=lambda: None) -> dict:
    manifests = validate_inventory(inventory, transport_sha256)
    verify_received_set(local_root, manifests)
    verify_received_set(backup_root, manifests)
    counts = {"copied": 0, "already_complete": 0}
    for name, manifest in sorted(manifests.items()):
        for record in manifest["files"]:
            contained_target(local_root, record["path"])
            contained_target(backup_root, record["path"])
        heartbeat()
        status = receive_artifact_unit(manifest, local_root=local_root, fetch_file=fetch_file)
        counts[status] += 1

        def copy_verified(relative, target):
            shutil.copyfile(contained_target(local_root, relative), target)

        receive_artifact_unit(manifest, local_root=backup_root, fetch_file=copy_verified)
    counts["receipts"] = len(manifests)
    counts["files"] = len({r["path"] for m in manifests.values() for r in m["files"]})
    return counts


def completion_counts(manifests: dict[str, dict]) -> dict[str, int]:
    counts = {key: 0 for key in ("checkpoints", "candidates", "selections", "score_shards", "results")}
    prefixes = {"checkpoint-": "checkpoints", "candidate-": "candidates", "selection-": "selections",
                "score-": "score_shards", "result-": "results"}
    for manifest in manifests.values():
        for prefix, key in prefixes.items():
            if manifest["unit_id"].startswith(prefix):
                counts[key] += 1
                break
    return counts


def check_complete(inventory: dict, manifests: dict[str, dict], expected_scopes: list[str]) -> dict:
    complete = inventory.get("complete") or {}
    if not inventory.get("terminal") or inventory.get("alive_pids") or inventory.get("locks"):
        raise ValueError("Worker process or lock has not exited")
    if inventory.get("failure") or complete.get("failure_count") != 0 or complete.get("test_contract_count") != 0:
        raise ValueError("Worker failure/test boundary violation")
    if complete.get("status") != "READY_FOR_VERIFIED_LOCAL_BACKUP":
        raise ValueError("Worker is not ready for verified backup")
    planned = complete.get("planned_scopes")
    completed = complete.get("completed_scopes")
    if isinstance(planned, list):
        if sorted(planned) != sorted(expected_scopes):
            raise ValueError("Planned scope identity mismatch")
        planned = len(planned)
    if isinstance(completed, list):
        if sorted(completed) != sorted(expected_scopes):
            raise ValueError("Completed scope identity mismatch")
        completed = len(completed)
    if planned != len(expected_scopes) or completed != planned:
        raise ValueError("Incomplete planned scope boundary")
    counts = completion_counts(manifests)
    n = len(expected_scopes)
    if counts != {"checkpoints": 3*n, "candidates": 3*n, "selections": n, "score_shards": 16*n, "results": 2*n}:
        raise ValueError("Incomplete candidate/checkpoint/selection/score/result boundary")
    return counts


class SSHTransport:
    def __init__(self, args):
        self.args = args
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.common = ["-i", str(args.identity_file.resolve()), "-o", "BatchMode=yes",
                       "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20",
                       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]

    def ssh(self, command: str, *, code: str | None = None):
        return subprocess.run(["ssh", *self.common, "-p", str(self.args.ssh_port), self.args.ssh_host, command],
                              input=code, text=True, encoding="utf-8", capture_output=True, check=True,
                              timeout=90, creationflags=self.flags)

    def inventory(self):
        command = "python3 - " + " ".join(shlex.quote(v) for v in
                    (self.args.remote_root, self.args.runtime_root, self.args.remote_transport))
        return json.loads(self.ssh(command, code=INVENTORY_CODE).stdout)

    def fetch_absolute(self, remote: str, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-q", *self.common, "-P", str(self.args.ssh_port),
                        self.args.ssh_host + ":" + remote, str(target)], capture_output=True,
                       check=True, timeout=600, creationflags=self.flags)

    def fetch(self, relative: str, target: Path):
        safe_artifact_path(relative)
        self.fetch_absolute(self.args.remote_root + "/" + relative, target)

    def heartbeat(self, cycle: int, counts: dict):
        # Only this designated operational heartbeat is written remotely.
        path = self.args.remote_root + "/_sync/receiver_heartbeat.json"
        temporary = path + f".tmp-{os.getpid()}"
        payload = json.dumps({"schema_version": "taxonomy_sync_receiver_heartbeat_v1", "status": "pass",
                              "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                              "cycle": cycle, "pid": os.getpid(), "receipts": counts.get("receipts", 0)})
        code = "import json,os,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); t=pathlib.Path(sys.argv[2]); t.write_text(sys.argv[3]); os.replace(t,p)"
        self.ssh("python3 -c " + shlex.quote(code) + " " + " ".join(shlex.quote(x) for x in (path, temporary, payload)))


def save_runtime(inventory: dict, transport: SSHTransport, roots: list[Path]) -> None:
    files = inventory.get("runtime_files", {})
    if not set(files).issubset(RUNTIME_NAMES):
        raise ValueError("Unexpected remote runtime filename")
    with tempfile.TemporaryDirectory(prefix="policy-qlora-runtime-") as directory:
        for name, record in files.items():
            if not isinstance(record.get("bytes"), int) or not 0 <= record["bytes"] <= MAX_FILE_BYTES:
                raise ValueError("Invalid runtime file size")
            source = Path(directory) / name
            transport.fetch_absolute(transport.args.runtime_root + "/" + name, source)
            if source.stat().st_size != record["bytes"] or file_sha256(source) != record["sha256"]:
                raise ValueError("Runtime provenance changed during transfer")
            for root in roots:
                target = root / "_sync" / "runtime_final" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if file_sha256(target) != record["sha256"]:
                        raise ValueError("Existing final runtime provenance conflict")
                else:
                    shutil.copyfile(source, target)
                if file_sha256(target) != record["sha256"]:
                    raise ValueError("Runtime provenance backup mismatch")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--remote-transport", required=True)
    parser.add_argument("--transport-manifest", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--scopes", nargs="+", required=True)
    parser.add_argument("--interval-seconds", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if not args.ssh_host.startswith("root@"):
        raise ValueError("Expected root@IP SSH destination")
    ipaddress.ip_address(args.ssh_host[5:])
    if not 1 <= args.ssh_port <= 65535 or not re.fullmatch(r"[a-z0-9]+", args.pod_id):
        raise ValueError("Invalid Pod/SSH identity")
    if not args.identity_file.is_file() or not args.transport_manifest.is_file():
        raise ValueError("Missing SSH identity or local transport manifest")
    for remote in (args.remote_root, args.runtime_root, args.remote_transport):
        path = PurePosixPath(remote)
        if not re.fullmatch(r"/workspace/[A-Za-z0-9_./-]+", remote) or path.as_posix() != remote or ".." in path.parts:
            raise ValueError("Remote paths must be canonical paths under /workspace")
    parent = PurePosixPath(args.remote_root).parent
    if PurePosixPath(args.runtime_root) != parent / "runtime" or PurePosixPath(args.remote_transport) != parent / "transport.json":
        raise ValueError("Runtime/transport paths differ from declared release root")
    if args.interval_seconds <= 0 or len(args.scopes) != len(set(args.scopes)) or not all(SCOPE.fullmatch(s) for s in args.scopes):
        raise ValueError("Invalid interval or expected scope set")
    roots = [args.local_root.resolve(), args.backup_root.resolve()]
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        raise ValueError("Primary and backup destinations must be separate non-nested directories")


def main() -> int:
    args = parse_args()
    validate_args(args)
    transport_sha = file_sha256(args.transport_manifest)
    transport = SSHTransport(args)
    sync = args.local_root / "_sync"
    sync.mkdir(parents=True, exist_ok=True)
    if any((sync / name).exists() for name in ("receiver_FAILED.json", "READY_TO_STOP.json")):
        raise RuntimeError("Existing terminal receiver evidence requires explicit review")
    lock = sync / "receiver.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = {"pod_id": args.pod_id, "pid": os.getpid(), "protocol_id": PROTOCOL,
             "transport_sha256": transport_sha, "receiver_sha256": file_sha256(Path(__file__)),
             "planned_scopes": args.scopes, "test_contract_count": 0, "failure_count": 0,
             "started_unix": time.time()}
    cycle = 0
    connection_errors = 0
    counts = {}
    try:
        while True:
            cycle += 1
            try:
                inventory = transport.inventory()
                manifests = validate_inventory(inventory, transport_sha)
                write_json(sync / "last_remote_inventory.json", inventory)
                last_heartbeat = [0.0]

                def heartbeat():
                    if time.monotonic() - last_heartbeat[0] >= 60:
                        transport.heartbeat(cycle, counts)
                        last_heartbeat[0] = time.monotonic()

                counts = receive_snapshot(inventory, transport_sha256=transport_sha,
                    local_root=args.local_root, backup_root=args.backup_root,
                    fetch_file=transport.fetch, heartbeat=heartbeat)
                transport.heartbeat(cycle, counts)
                connection_errors = 0
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                connection_errors += 1
                state.update(status="connection_retry", cycle=cycle, connection_errors=connection_errors,
                             updated_unix=time.time(), error_type=type(error).__name__)
                write_json(sync / "receiver_state.json", state)
                if connection_errors >= 5 or args.once:
                    raise
                time.sleep(30)
                continue
            state.update(status="receiving", cycle=cycle, updated_unix=time.time(), counts=counts,
                         remote_state=inventory.get("runtime_state"), alive_pids=inventory.get("alive_pids"))
            write_json(sync / "receiver_state.json", state)
            print(json.dumps({"status": state["status"], "cycle": cycle, **counts}), flush=True)
            if inventory.get("failure"):
                save_runtime(inventory, transport, [args.local_root, args.backup_root])
                raise RuntimeError("Cloud worker FAILED: evidence preserved, no restart")
            if inventory.get("terminal"):
                boundary = check_complete(inventory, manifests, args.scopes)
                verify_received_set(args.local_root, manifests, exact=True)
                verify_received_set(args.backup_root, manifests, exact=True)
                save_runtime(inventory, transport, [args.local_root, args.backup_root])
                # Re-observe after transfer to prevent an old terminal snapshot
                # from authorising Stop while a worker has resumed or changed.
                final = transport.inventory()
                final_manifests = validate_inventory(final, transport_sha)
                check_complete(final, final_manifests, args.scopes)
                if final_manifests != manifests or final.get("complete") != inventory.get("complete"):
                    raise ValueError("Remote completion changed during final verification")
                ready = {**state, "status": "READY_TO_STOP", "boundary": boundary,
                         "backup_verified": True, "remote_processes_exited": True,
                         "completed_unix": time.time(), "pod_stop_performed": False}
                for root in (args.local_root, args.backup_root):
                    write_json(root / "_sync" / "READY_TO_STOP.json", ready)
                write_json(sync / "receiver_state.json", ready)
                print("LOCAL_BACKUP_VERIFIED_READY_TO_STOP " + args.pod_id, flush=True)
                return 0
            if args.once:
                return 0
            time.sleep(args.interval_seconds)
    except BaseException as error:
        state.update(status="stopped_failure", failure_count=1, updated_unix=time.time(), error=repr(error))
        write_json(sync / "receiver_FAILED.json", state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
