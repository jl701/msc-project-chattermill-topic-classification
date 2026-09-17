"""Fail-closed campaign runner for the 26 frozen-Qwen few-shot scopes."""

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

from msc_project.experiments.taxonomy_execution import canonical_sha256  # noqa: E402
from msc_project.experiments.taxonomy_gpu_guard import (  # noqa: E402
    run_guarded_command,
)
from msc_project.experiments.taxonomy_two_stage_formal import (  # noqa: E402
    PROTOCOL_ID,
    scope_folds,
)
from msc_project.experiments.taxonomy_two_stage_parallel import (  # noqa: E402
    build_worker_manifest,
    load_parallel_plan,
    worker_record,
)
from msc_project.experiments.verified_artifact_sync import (  # noqa: E402
    publish_artifact_unit,
)


METHOD_ID = "frozen_qwen_few_shot"
WORKER_ID = "worker-distil-frozen"
STATE_SCHEMA = "taxonomy_frozen_qwen_few_shot_campaign_state_v1"
FATAL_TOKENS = (
    "out of memory",
    "cuda oom",
    "non-finite",
    "nan loss",
    "prediction collapse",
    "third sentiment",
    "resume conflict",
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
    if temporary.exists():
        raise FileExistsError(f"Stale campaign state temporary file: {temporary}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()


def _gpu_snapshot() -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,utilization.gpu,memory.used,"
                "memory.total,temperature.gpu,power.draw,power.limit,"
                "clocks_throttle_reasons.active",
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


def _child_environment(hf_home: Path, *, local_files_only: bool) -> dict[str, str]:
    resolved = hf_home.resolve()
    if not resolved.is_dir() or not (resolved / "hub").is_dir():
        raise FileNotFoundError(f"Formal Hugging Face cache is incomplete: {resolved}")
    environment = os.environ.copy()
    environment["HF_HOME"] = str(resolved)
    environment["HF_HUB_CACHE"] = str(resolved / "hub")
    if local_files_only:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _scope_command(args: argparse.Namespace, scope_id: str) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_taxonomy_two_stage_frozen_few_shot_validation.py"),
        "--scope-id",
        scope_id,
        "--config",
        str(args.config.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--resume",
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def run(args: argparse.Namespace) -> dict[str, object]:
    plan = load_parallel_plan(args.parallel_plan)
    worker = worker_record(plan, WORKER_ID)
    scopes = [str(value) for value in worker["frozen_qwen_few_shot_scope_ids"]]
    if scopes != list(scope_folds()):
        raise RuntimeError(
            "Frozen-Qwen scopes must follow the complete registered scope order."
        )
    worker_manifest = build_worker_manifest(
        plan,
        WORKER_ID,
        output_root=str(args.output_root.resolve()),
        data_dir=str(args.data_dir.resolve()),
    )
    plan_hash = canonical_sha256(plan)
    if args.start_after:
        if args.start_after not in scopes:
            raise ValueError(f"Unknown --start-after scope: {args.start_after}")
        scopes = scopes[scopes.index(args.start_after) + 1 :]
    if args.max_scopes is not None:
        scopes = scopes[: args.max_scopes]
    if args.dry_run:
        return {
            "schema_version": "taxonomy_frozen_qwen_few_shot_campaign_dry_run_v1",
            "status": "pass",
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "worker_id": WORKER_ID,
            "parallel_plan_sha256": plan_hash,
            "worker_manifest_sha256": worker_manifest["worker_manifest_sha256"],
            "planned_count": len(scopes),
            "first_scope": scopes[0] if scopes else None,
            "last_scope": scopes[-1] if scopes else None,
            "failure_count": 0,
            "test_contract_count": 0,
        }
    if args.sync_heartbeat is None:
        raise ValueError(
            "--sync-heartbeat is required so replication failure stops formal work."
        )

    environment = _child_environment(
        args.hf_home, local_files_only=args.local_files_only
    )
    resolved_hf_home = str(args.hf_home.resolve())
    campaign_root = (
        args.output_root
        / "_campaign"
        / "workers"
        / WORKER_ID
        / METHOD_ID
    )
    state_path = campaign_root / "state.json"
    manifest_path = campaign_root / "worker_manifest.json"
    if manifest_path.is_file():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != worker_manifest:
            raise RuntimeError("Frozen campaign worker manifest resume conflict.")
    else:
        _atomic_json(manifest_path, worker_manifest)

    completed: list[str] = []
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("schema_version") != STATE_SCHEMA:
            raise ValueError("Unexpected Frozen-Qwen campaign state schema.")
        if previous.get("failure_count") != 0:
            raise RuntimeError("Frozen-Qwen campaign state contains a failure.")
        if previous.get("test_contract_count") != 0:
            raise RuntimeError("Frozen-Qwen campaign state contains a test contract.")
        if previous.get("parallel_plan_sha256") != plan_hash:
            raise RuntimeError("Frozen-Qwen campaign parallel-plan resume conflict.")
        if previous.get("worker_manifest_sha256") != worker_manifest.get(
            "worker_manifest_sha256"
        ):
            raise RuntimeError("Frozen-Qwen campaign worker resume conflict.")
        if previous.get("hf_home") != resolved_hf_home:
            raise RuntimeError("Frozen-Qwen campaign Hugging Face cache changed.")
        completed = [str(value) for value in previous.get("completed_scopes", [])]
        if previous.get("status") == "complete" and all(
            scope_id in completed for scope_id in scopes
        ):
            return previous

    logs = campaign_root / "logs"
    telemetry = campaign_root / "telemetry.jsonl"
    for scope_id in scopes:
        if scope_id in completed:
            continue
        state: dict[str, object] = {
            "schema_version": STATE_SCHEMA,
            "status": "running",
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "worker_id": WORKER_ID,
            "parallel_plan_sha256": plan_hash,
            "worker_manifest_sha256": worker_manifest["worker_manifest_sha256"],
            "hf_home": resolved_hf_home,
            "current_scope": scope_id,
            "completed_scopes": completed,
            "completed_count": len(completed),
            "planned_count": len(scopes),
            "failure_count": 0,
            "test_contract_count": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(state_path, state)
        _append_jsonl(
            telemetry,
            {
                "event": "scope_start",
                "scope_id": scope_id,
                "at": datetime.now(timezone.utc).isoformat(),
                "gpu": _gpu_snapshot(),
            },
        )
        log_path = logs / f"{scope_id}.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as stream:
            guarded = run_guarded_command(
                _scope_command(args, scope_id),
                cwd=PROJECT_ROOT,
                environment=environment,
                log_stream=stream,
                telemetry_path=telemetry,
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
                    "failed_scope": scope_id,
                    "exit_code": returncode,
                    "fatal_log_tokens": fatal,
                    "guard_stop_reason": guard_stop_reason,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            _atomic_json(state_path, state)
            raise RuntimeError(
                f"Frozen-Qwen campaign stopped at {scope_id}; "
                f"exit={returncode}, fatal={fatal}, guard={guard_stop_reason}."
            )
        publish_artifact_unit(
            args.output_root,
            [log_path.relative_to(args.output_root)],
            unit_id=f"log-{METHOD_ID}-{scope_id}",
            protocol_id=PROTOCOL_ID,
            contract_sha256=str(worker_manifest["worker_manifest_sha256"]),
        )
        completed.append(scope_id)
        _append_jsonl(
            telemetry,
            {
                "event": "scope_complete",
                "scope_id": scope_id,
                "at": datetime.now(timezone.utc).isoformat(),
                "gpu": _gpu_snapshot(),
            },
        )

    result = {
        "schema_version": STATE_SCHEMA,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "worker_id": WORKER_ID,
        "parallel_plan_sha256": plan_hash,
        "worker_manifest_sha256": worker_manifest["worker_manifest_sha256"],
        "hf_home": resolved_hf_home,
        "current_scope": None,
        "completed_scopes": completed,
        "completed_count": len(completed),
        "planned_count": len(scopes),
        "failure_count": 0,
        "test_contract_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(state_path, result)
    if completed == list(scope_folds()):
        publish_artifact_unit(
            args.output_root,
            [
                manifest_path.relative_to(args.output_root),
                state_path.relative_to(args.output_root),
            ],
            unit_id=f"campaign-state-{WORKER_ID}-{METHOD_ID}",
            protocol_id=PROTOCOL_ID,
            contract_sha256=str(worker_manifest["worker_manifest_sha256"]),
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all registered Frozen-Qwen few-shot validation scopes."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/taxonomy_two_stage_formal_v1/worker-distil-frozen"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/workspace/data/fabsa")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "taxonomy_two_stage_cloud_execution_safety_v1.json"
        ),
    )
    parser.add_argument(
        "--parallel-plan", type=Path, default=DEFAULT_PARALLEL_PLAN
    )
    parser.add_argument("--start-after")
    parser.add_argument("--max-scopes", type=int)
    parser.add_argument("--hf-home", type=Path, required=True)
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
    if args.max_scopes is not None and args.max_scopes < 1:
        raise ValueError("--max-scopes must be positive.")
    if args.guard_interval_seconds <= 0 or args.maximum_heartbeat_age_seconds <= 0:
        raise ValueError("Guard and heartbeat intervals must be positive.")
    value = run(args)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
