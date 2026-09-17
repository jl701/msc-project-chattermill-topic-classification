"""Serial fail-closed orchestrator for the formal trainable cloud campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.taxonomy_two_stage_formal import (  # noqa: E402
    TRAINABLE_METHODS,
    FormalJob,
    build_formal_jobs,
    formal_job_id,
    validate_formal_job_graph,
    validate_formal_job_subset,
)
from msc_project.experiments.taxonomy_gpu_guard import (  # noqa: E402
    run_guarded_command,
)
from msc_project.experiments.taxonomy_two_stage_parallel import (  # noqa: E402
    build_worker_manifest,
    load_parallel_plan,
    trainable_worker_jobs,
)
from msc_project.experiments.verified_artifact_sync import (  # noqa: E402
    publish_artifact_unit,
)


FATAL_TOKENS = (
    "out of memory",
    "cuda oom",
    "non-finite",
    "nan loss",
    "prediction collapse",
    "third sentiment",
    "corrupt",
    "incompatible",
    "official-test",
)
DEFAULT_PARALLEL_PLAN = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_two_stage_three_gpu_parallel_v1.json"
)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _job_id(job: FormalJob) -> str:
    return formal_job_id(job)


def _gpu_snapshot() -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return {"raw": result.stdout.strip()}
    except Exception as error:
        return {"error": repr(error)}


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()


def _materialize_job_command(
    job: FormalJob, *, local_files_only: bool
) -> list[str]:
    """Bind a frozen portable job to the orchestrator's exact interpreter."""

    command = list(job.command)
    if not command or command[0] != "python":
        raise ValueError("Formal jobs must declare the portable 'python' launcher.")
    command[0] = sys.executable
    if local_files_only:
        command.append("--local-files-only")
    return command


def _materialize_job_environment(
    *, hf_home: Path, local_files_only: bool
) -> dict[str, str]:
    """Bind every child to the audited persistent Hugging Face cache."""

    resolved_home = hf_home.resolve()
    hub_cache = resolved_home / "hub"
    if not resolved_home.is_dir() or not hub_cache.is_dir():
        raise FileNotFoundError(
            f"Formal Hugging Face cache is incomplete: {resolved_home}"
        )
    environment = os.environ.copy()
    environment["HF_HOME"] = str(resolved_home)
    environment["HF_HUB_CACHE"] = str(hub_cache)
    if local_files_only:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def run(args: argparse.Namespace) -> dict[str, object]:
    resolved_output_root = str(args.output_root.resolve())
    resolved_data_dir = str(args.data_dir.resolve())
    worker_manifest: dict[str, object] | None = None
    if args.worker_id:
        plan = load_parallel_plan(args.parallel_plan)
        worker_manifest = build_worker_manifest(
            plan,
            args.worker_id,
            output_root=resolved_output_root,
            data_dir=resolved_data_dir,
        )
        jobs = trainable_worker_jobs(
            plan,
            args.worker_id,
            output_root=resolved_output_root,
            data_dir=resolved_data_dir,
        )
        validate_formal_job_subset(jobs)
    else:
        jobs = build_formal_jobs(
            output_root=resolved_output_root,
            data_dir=resolved_data_dir,
        )
        validate_formal_job_graph(jobs)
        if args.method:
            jobs = [job for job in jobs if job.method_id == args.method]
    if args.start_after:
        identifiers = [_job_id(job) for job in jobs]
        if args.start_after not in identifiers:
            raise ValueError(f"Unknown --start-after job: {args.start_after}")
        jobs = jobs[identifiers.index(args.start_after) + 1 :]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    if args.dry_run:
        return {
            "schema_version": "taxonomy_two_stage_formal_campaign_dry_run_v1",
            "status": "pass",
            "planned_count": len(jobs),
            "first_job": _job_id(jobs[0]) if jobs else None,
            "last_job": _job_id(jobs[-1]) if jobs else None,
            "failure_count": 0,
            "test_contract_count": 0,
            "worker_id": args.worker_id,
            "worker_manifest_sha256": (
                worker_manifest.get("worker_manifest_sha256")
                if worker_manifest is not None
                else None
            ),
        }
    if args.sync_heartbeat is None:
        raise ValueError(
            "--sync-heartbeat is required so replication failure stops formal work."
        )
    child_environment = _materialize_job_environment(
        hf_home=args.hf_home,
        local_files_only=args.local_files_only,
    )
    resolved_hf_home = str(args.hf_home.resolve())
    campaign_root = (
        args.output_root / "_campaign" / "workers" / args.worker_id
        if args.worker_id
        else args.output_root / "_campaign" / (args.method or "all_methods")
    )
    if worker_manifest is not None:
        manifest_path = campaign_root / "worker_manifest.json"
        if manifest_path.is_file():
            observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if observed_manifest != worker_manifest:
                raise RuntimeError("Existing worker manifest conflicts with this launch.")
        else:
            _atomic_json(manifest_path, worker_manifest)
    state_path = campaign_root / "state.json"
    telemetry_path = campaign_root / "telemetry.jsonl"
    completed: list[str] = []
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        expected_schema = (
            "taxonomy_two_stage_parallel_campaign_state_v1"
            if worker_manifest is not None
            else "taxonomy_two_stage_formal_campaign_state_v2"
        )
        if previous.get("schema_version") != expected_schema:
            raise ValueError("Unexpected formal campaign state schema.")
        if previous.get("failure_count", 0):
            raise RuntimeError("Previous formal campaign state contains a failure; audit before resume.")
        if previous.get("hf_home") != resolved_hf_home:
            raise RuntimeError(
                "Previous formal campaign state was created with a different Hugging Face cache."
            )
        if worker_manifest is not None and previous.get(
            "worker_manifest_sha256"
        ) != worker_manifest.get("worker_manifest_sha256"):
            raise RuntimeError("Previous campaign used a different worker manifest.")
        completed = [str(value) for value in previous.get("completed_jobs", [])]
        current_ids = [_job_id(job) for job in jobs]
        if previous.get("status") == "complete" and all(
            identifier in completed for identifier in current_ids
        ):
            return previous

    logs = campaign_root / "logs"
    for job in jobs:
        identifier = _job_id(job)
        if identifier in completed:
            continue
        state = {
            "schema_version": (
                "taxonomy_two_stage_parallel_campaign_state_v1"
                if worker_manifest is not None
                else "taxonomy_two_stage_formal_campaign_state_v2"
            ),
            "status": "running",
            "hf_home": resolved_hf_home,
            "worker_id": args.worker_id,
            "worker_manifest_sha256": (
                worker_manifest.get("worker_manifest_sha256")
                if worker_manifest is not None
                else None
            ),
            "current_job": identifier,
            "completed_jobs": completed,
            "completed_count": len(completed),
            "planned_count": len(jobs),
            "failure_count": 0,
            "test_contract_count": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(state_path, state)
        _append_jsonl(
            telemetry_path,
            {
                "event": "job_start",
                "job_id": identifier,
                "at": datetime.now(timezone.utc).isoformat(),
                "gpu": _gpu_snapshot(),
            },
        )
        log_path = logs / f"{identifier}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _materialize_job_command(
            job, local_files_only=args.local_files_only
        )
        with log_path.open("w", encoding="utf-8", newline="\n") as stream:
            guarded = run_guarded_command(
                command,
                cwd=PROJECT_ROOT,
                environment=child_environment,
                log_stream=stream,
                telemetry_path=telemetry_path,
                output_root=args.output_root,
                replication_heartbeat=args.sync_heartbeat,
                poll_interval_seconds=args.guard_interval_seconds,
                maximum_heartbeat_age_seconds=args.maximum_heartbeat_age_seconds,
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        fatal = sorted(token for token in FATAL_TOKENS if token in log_text)
        returncode = int(guarded["returncode"])
        guard_stop_reason = guarded["guard_stop_reason"]
        if returncode != 0 or fatal or guard_stop_reason:
            state.update(
                {
                    "status": "failed",
                    "failure_count": 1,
                    "failed_job": identifier,
                    "exit_code": returncode,
                    "fatal_log_tokens": fatal,
                    "guard_stop_reason": guard_stop_reason,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            _atomic_json(state_path, state)
            raise RuntimeError(
                f"Formal campaign stopped at {identifier}; exit={returncode}, "
                f"fatal={fatal}, guard={guard_stop_reason}."
            )
        if worker_manifest is not None:
            publish_artifact_unit(
                args.output_root,
                [log_path.relative_to(args.output_root)],
                unit_id=f"log-{identifier}",
                protocol_id=str(worker_manifest["protocol_id"]),
                contract_sha256=str(worker_manifest["worker_manifest_sha256"]),
            )
        completed.append(identifier)
        _append_jsonl(
            telemetry_path,
            {
                "event": "job_complete",
                "job_id": identifier,
                "at": datetime.now(timezone.utc).isoformat(),
                "gpu": _gpu_snapshot(),
            },
        )
    result = {
        "schema_version": (
            "taxonomy_two_stage_parallel_campaign_state_v1"
            if worker_manifest is not None
            else "taxonomy_two_stage_formal_campaign_state_v2"
        ),
        "status": "complete",
        "hf_home": resolved_hf_home,
        "worker_id": args.worker_id,
        "worker_manifest_sha256": (
            worker_manifest.get("worker_manifest_sha256")
            if worker_manifest is not None
            else None
        ),
        "current_job": None,
        "completed_jobs": completed,
        "completed_count": len(completed),
        "planned_count": len(jobs),
        "failure_count": 0,
        "test_contract_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(state_path, result)
    if worker_manifest is not None and completed == list(worker_manifest["job_ids"]):
        publish_artifact_unit(
            args.output_root,
            [
                manifest_path.relative_to(args.output_root),
                state_path.relative_to(args.output_root),
            ],
            unit_id=f"campaign-state-{args.worker_id}",
            protocol_id=str(worker_manifest["protocol_id"]),
            contract_sha256=str(worker_manifest["worker_manifest_sha256"]),
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal two-stage job graph serially.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/taxonomy_two_stage_formal_v1"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/data/fabsa"),
    )
    parser.add_argument("--method", choices=TRAINABLE_METHODS)
    parser.add_argument("--worker-id")
    parser.add_argument(
        "--parallel-plan",
        type=Path,
        default=DEFAULT_PARALLEL_PLAN,
    )
    parser.add_argument("--start-after")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument(
        "--hf-home",
        type=Path,
        required=True,
        help="Audited persistent Hugging Face home containing the hub cache.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--sync-heartbeat",
        type=Path,
        help="Remote heartbeat written by the verified local sync receiver.",
    )
    parser.add_argument("--guard-interval-seconds", type=float, default=30.0)
    parser.add_argument(
        "--maximum-heartbeat-age-seconds", type=float, default=900.0
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_id and args.method:
        raise ValueError("--worker-id and --method are mutually exclusive.")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("--max-jobs must be positive.")
    if args.guard_interval_seconds <= 0 or args.maximum_heartbeat_age_seconds <= 0:
        raise ValueError("Guard and heartbeat intervals must be positive.")
    value = run(args)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
