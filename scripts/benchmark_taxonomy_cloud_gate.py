"""Run and measure the first sealed QLoRA validation-selection scope.

This is an administrative runtime gate. It delegates every scientific job to
``execute_taxonomy_plan.py`` and never enables the official-test flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = PROJECT_ROOT / "scripts" / "execute_taxonomy_plan.py"
METHOD_ID = "qwen_candidate_pair_qlora"
TARGET_JOB_ID = "select-tuned-qwen_candidate_pair_qlora-heldout-a01"
TEST_STAGES = {
    "formal-score-test",
    "formal-analyse-test",
    "formal-primary-comparisons",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _job_method(job: dict[str, object]) -> str | None:
    argv = [str(value) for value in job["argv"]]
    if "--method" not in argv:
        return None
    return argv[argv.index("--method") + 1]


def validation_jobs(
    plan: dict[str, object],
    *,
    method_id: str = METHOD_ID,
) -> list[dict[str, object]]:
    jobs = []
    for raw_job in plan["jobs"]:
        job = dict(raw_job)
        if _job_method(job) != method_id:
            continue
        if (
            "test" in job["official_splits_opened"]
            or str(job["stage"]) in TEST_STAGES
        ):
            continue
        jobs.append(job)
    if not jobs:
        raise ValueError(f"No guarded validation jobs found for {method_id}.")
    selected_ids = {str(job["job_id"]) for job in jobs}
    omitted = {
        str(dependency)
        for job in jobs
        for dependency in job["depends_on"]
        if str(dependency) not in selected_ids
    }
    if omitted:
        raise ValueError(
            "Guarded QLoRA validation jobs omit dependencies: "
            f"{sorted(omitted)[:5]}"
        )
    return jobs


def first_scope_jobs(
    jobs: Sequence[dict[str, object]],
    *,
    target_job_id: str = TARGET_JOB_ID,
) -> list[dict[str, object]]:
    by_id = {str(job["job_id"]): job for job in jobs}
    if target_job_id not in by_id:
        raise ValueError(f"First-scope target job is missing: {target_job_id}")
    required = {target_job_id}
    frontier = [target_job_id]
    while frontier:
        job_id = frontier.pop()
        for dependency in by_id[job_id]["depends_on"]:
            dependency_id = str(dependency)
            if dependency_id not in by_id:
                raise ValueError(
                    f"First-scope dependency is outside the guarded plan: "
                    f"{dependency_id}"
                )
            if dependency_id not in required:
                required.add(dependency_id)
                frontier.append(dependency_id)
    scope = [job for job in jobs if str(job["job_id"]) in required]
    if str(scope[-1]["job_id"]) != target_job_id:
        raise ValueError("First-scope target is not last in plan order.")
    if any(
        "test" in job["official_splits_opened"]
        or str(job["stage"]) in TEST_STAGES
        for job in scope
    ):
        raise AssertionError("Official-test work entered the cloud runtime gate.")
    return scope


def default_state_path(plan: dict[str, object], plan_path: Path) -> Path:
    return (
        PROJECT_ROOT
        / str(plan["output_root"])
        / "_plan_state"
        / f"{plan_path.stem}.{METHOD_ID}.through-validation.json"
    )


def default_telemetry_path(plan: dict[str, object]) -> Path:
    return (
        PROJECT_ROOT
        / str(plan["output_root"])
        / "_runtime_telemetry"
        / METHOD_ID
        / "first_validation_gate.json"
    )


def executor_command(plan_path: Path, state_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(EXECUTOR),
        "--plan",
        str(plan_path),
        "--method",
        METHOD_ID,
        "--state-path",
        str(state_path),
        "--stop-after",
        "1",
        "--max-workers",
        "1",
    ]
    if "--include-official-test" in command:
        raise AssertionError("Cloud runtime gate must not enable official test.")
    return command


def _run_nvidia_smi(query: str) -> list[list[str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def gpu_inventory() -> list[dict[str, object]]:
    rows = _run_nvidia_smi("index,name,uuid,driver_version,memory.total")
    inventory = []
    for row in rows:
        if len(row) != 5:
            continue
        try:
            memory_total = float(row[4])
        except ValueError:
            continue
        inventory.append(
            {
                "index": row[0],
                "name": row[1],
                "uuid": row[2],
                "driver_version": row[3],
                "memory_total_mib": memory_total,
            }
        )
    return inventory


def sample_gpu() -> tuple[float, float] | None:
    rows = _run_nvidia_smi("memory.used,utilization.gpu")
    parsed = []
    for row in rows:
        if len(row) != 2:
            continue
        try:
            parsed.append((float(row[0]), float(row[1])))
        except ValueError:
            continue
    if not parsed:
        return None
    return max(value[0] for value in parsed), max(value[1] for value in parsed)


def _artifact_snapshot(output_root: Path) -> tuple[set[Path], set[Path]]:
    checkpoints = set(
        (
            output_root
            / "checkpoints"
            / METHOD_ID
        ).rglob("checkpoint.manifest.json")
    )
    score_manifests = set(
        (
            output_root
            / "taxonomy_generalisation_precloud_v1"
            / METHOD_ID
        ).rglob("*.manifest.json")
    )
    return checkpoints, score_manifests


def _artifact_delta(
    before: tuple[set[Path], set[Path]],
    after: tuple[set[Path], set[Path]],
) -> dict[str, object]:
    new_checkpoints = sorted(after[0] - before[0])
    new_score_manifests = sorted(after[1] - before[1])
    training_pairs = 0
    checkpoint_bytes = 0
    for manifest_path in new_checkpoints:
        manifest = _read_json(manifest_path)
        contract = dict(manifest["training_contract"])
        training_pairs += int(contract["training_pairs"])
        checkpoint_bytes += sum(
            path.stat().st_size
            for path in manifest_path.parent.rglob("*")
            if path.is_file()
        )
    score_pairs = 0
    score_csv_bytes = 0
    for manifest_path in new_score_manifests:
        manifest = _read_json(manifest_path)
        score_pairs += int(manifest["rows"])
        csv_path = manifest_path.with_name(
            manifest_path.name.removesuffix(".manifest.json") + ".csv"
        )
        score_csv_bytes += csv_path.stat().st_size
    return {
        "new_checkpoint_manifests": [
            path.relative_to(PROJECT_ROOT).as_posix() for path in new_checkpoints
        ],
        "new_score_manifests": len(new_score_manifests),
        "measured_training_pairs": training_pairs,
        "measured_score_pairs": score_pairs,
        "checkpoint_bytes": checkpoint_bytes,
        "score_csv_bytes": score_csv_bytes,
    }


def _completed_ids(state_path: Path) -> set[str]:
    if not state_path.is_file():
        return set()
    state = _read_json(state_path)
    return {str(value) for value in state.get("completed_job_ids", [])}


def _state_failures(state_path: Path) -> list[dict[str, object]]:
    if not state_path.is_file():
        return []
    return list(_read_json(state_path).get("failed_jobs", []))


def _next_scope_job(
    scope: Sequence[dict[str, object]],
    completed: set[str],
) -> dict[str, object] | None:
    for job in scope:
        job_id = str(job["job_id"])
        if job_id in completed:
            continue
        if all(str(value) in completed for value in job["depends_on"]):
            return job
    return None


def summarise_attempts(
    attempts: Iterable[dict[str, object]],
    *,
    target_complete: bool,
) -> dict[str, object]:
    completed = [
        attempt
        for attempt in attempts
        if attempt.get("status") == "complete"
    ]
    training = [
        attempt for attempt in completed if attempt.get("stage") == "tuning-train"
    ]
    scoring = [
        attempt
        for attempt in completed
        if attempt.get("stage") == "tuning-score-validation"
    ]
    training_seconds = sum(float(value["wall_seconds"]) for value in training)
    scoring_seconds = sum(float(value["wall_seconds"]) for value in scoring)
    score_pairs = sum(
        int(dict(value.get("artifact_delta", {})).get("measured_score_pairs", 0))
        for value in scoring
    )
    peaks = [
        float(value["peak_device_memory_used_mib"])
        for value in completed
        if value.get("peak_device_memory_used_mib") is not None
    ]
    return {
        "target_complete": target_complete,
        "completed_measured_jobs": len(completed),
        "total_measured_wall_seconds": sum(
            float(value["wall_seconds"]) for value in completed
        ),
        "training_wall_seconds": training_seconds,
        "scoring_wall_seconds": scoring_seconds,
        "measured_training_pairs": sum(
            int(
                dict(value.get("artifact_delta", {})).get(
                    "measured_training_pairs", 0
                )
            )
            for value in training
        ),
        "measured_score_pairs": score_pairs,
        "score_pairs_per_wall_second": (
            score_pairs / scoring_seconds if scoring_seconds else None
        ),
        "peak_device_memory_used_mib": max(peaks) if peaks else None,
        "checkpoint_bytes": sum(
            int(dict(value.get("artifact_delta", {})).get("checkpoint_bytes", 0))
            for value in completed
        ),
        "score_csv_bytes": sum(
            int(dict(value.get("artifact_delta", {})).get("score_csv_bytes", 0))
            for value in completed
        ),
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_gate(args: argparse.Namespace) -> int:
    plan_path = args.plan.resolve()
    plan = _read_json(plan_path)
    if plan.get("schema_version") != "taxonomy_cloud_execution_plan_v2":
        raise ValueError("Expected taxonomy_cloud_execution_plan_v2.")
    if plan.get("formal_execution_blocked") is not False:
        raise ValueError("The execution plan is still formally blocked.")
    jobs = validation_jobs(plan)
    scope = first_scope_jobs(jobs)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "event": "cloud_gate_dry_run",
                    "official_test_included": False,
                    "target_job_id": TARGET_JOB_ID,
                    "jobs": [
                        {
                            "job_id": job["job_id"],
                            "stage": job["stage"],
                            "executor": job["executor"],
                            "official_splits_opened": job[
                                "official_splits_opened"
                            ],
                            "argv": job["argv"],
                        }
                        for job in scope
                    ],
                },
                indent=2,
            )
        )
        return 0

    state_path = (
        args.state_path.resolve()
        if args.state_path
        else default_state_path(plan, plan_path)
    )
    telemetry_path = (
        args.telemetry_output.resolve()
        if args.telemetry_output
        else default_telemetry_path(plan)
    )
    plan_hash = _sha256(plan_path)
    scope_ids = {str(job["job_id"]) for job in scope}
    preexisting_scope_ids = _completed_ids(state_path) & scope_ids
    if telemetry_path.is_file():
        telemetry = _read_json(telemetry_path)
        expected = {
            "schema_version": "taxonomy_cloud_runtime_gate_v1",
            "plan_sha256": plan_hash,
            "method_id": METHOD_ID,
            "target_job_id": TARGET_JOB_ID,
            "official_test_included": False,
        }
        for key, expected_value in expected.items():
            if telemetry.get(key) != expected_value:
                raise ValueError(
                    f"Telemetry contract mismatch for {key}: "
                    f"{telemetry.get(key)!r} != {expected_value!r}"
                )
        for attempt in telemetry.get("attempts", []):
            if attempt.get("status") == "running":
                attempt["status"] = "interrupted"
                attempt["interruption_recorded_at"] = _utc_now()
    else:
        if preexisting_scope_ids:
            raise RuntimeError(
                "The first QLoRA scope already has completed jobs but no "
                "matching telemetry artifact. Use a fresh state/output root "
                "for the measured cloud gate."
            )
        telemetry = {
            "schema_version": "taxonomy_cloud_runtime_gate_v1",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "status": "pending",
            "plan_path": plan_path.as_posix(),
            "plan_sha256": plan_hash,
            "git_commit": _git_commit(),
            "method_id": METHOD_ID,
            "target_job_id": TARGET_JOB_ID,
            "official_test_included": False,
            "sample_interval_seconds": args.sample_interval_seconds,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "gpu_inventory": gpu_inventory(),
            },
            "attempts": [],
            "summary": {},
        }

    output_root = PROJECT_ROOT / str(plan["output_root"])
    command = executor_command(plan_path, state_path)
    jobs_run = 0
    while TARGET_JOB_ID not in _completed_ids(state_path):
        if jobs_run >= args.max_jobs:
            raise RuntimeError(
                f"Cloud gate exceeded the {args.max_jobs}-job safety bound."
            )
        before_completed = _completed_ids(state_path)
        expected_job = _next_scope_job(scope, before_completed)
        if expected_job is None:
            raise RuntimeError("First-scope dependency graph cannot make progress.")
        before_failures = _state_failures(state_path)
        before_artifacts = _artifact_snapshot(output_root)
        attempt: dict[str, object] = {
            "attempt_index": len(telemetry["attempts"]) + 1,
            "expected_job_id": expected_job["job_id"],
            "stage": expected_job["stage"],
            "executor": expected_job["executor"],
            "started_at": _utc_now(),
            "status": "running",
            "command": command,
            "gpu_samples": 0,
            "peak_device_memory_used_mib": None,
            "peak_gpu_utilization_pct": None,
        }
        telemetry["attempts"].append(attempt)
        telemetry["status"] = "running"
        telemetry["updated_at"] = _utc_now()
        _write_json_atomic(telemetry_path, telemetry)

        started = time.monotonic()
        process = subprocess.Popen(command, cwd=PROJECT_ROOT)
        peak_memory: float | None = None
        peak_utilisation: float | None = None
        samples = 0
        while process.poll() is None:
            sample = sample_gpu()
            if sample is not None:
                memory, utilisation = sample
                peak_memory = (
                    memory if peak_memory is None else max(peak_memory, memory)
                )
                peak_utilisation = (
                    utilisation
                    if peak_utilisation is None
                    else max(peak_utilisation, utilisation)
                )
                samples += 1
            time.sleep(args.sample_interval_seconds)
        returncode = int(process.wait())
        wall_seconds = time.monotonic() - started
        after_completed = _completed_ids(state_path)
        new_completed = sorted(after_completed - before_completed)
        after_failures = _state_failures(state_path)
        attempt.update(
            {
                "finished_at": _utc_now(),
                "wall_seconds": wall_seconds,
                "returncode": returncode,
                "gpu_samples": samples,
                "peak_device_memory_used_mib": peak_memory,
                "peak_gpu_utilization_pct": peak_utilisation,
                "completed_job_ids": new_completed,
                "new_failures": after_failures[len(before_failures) :],
                "artifact_delta": _artifact_delta(
                    before_artifacts,
                    _artifact_snapshot(output_root),
                ),
            }
        )
        if returncode == 0 and new_completed == [str(expected_job["job_id"])]:
            attempt["status"] = "complete"
        else:
            attempt["status"] = "failed"
            telemetry["status"] = "failed"
            telemetry["updated_at"] = _utc_now()
            telemetry["summary"] = summarise_attempts(
                telemetry["attempts"],
                target_complete=TARGET_JOB_ID in after_completed,
            )
            _write_json_atomic(telemetry_path, telemetry)
            return returncode or 1
        jobs_run += 1
        telemetry["updated_at"] = _utc_now()
        telemetry["summary"] = summarise_attempts(
            telemetry["attempts"],
            target_complete=TARGET_JOB_ID in after_completed,
        )
        _write_json_atomic(telemetry_path, telemetry)

    telemetry["status"] = "complete"
    telemetry["completed_at"] = _utc_now()
    telemetry["updated_at"] = _utc_now()
    telemetry["summary"] = summarise_attempts(
        telemetry["attempts"],
        target_complete=True,
    )
    measured_ids = {
        str(job_id)
        for attempt in telemetry["attempts"]
        if attempt.get("status") == "complete"
        for job_id in attempt.get("completed_job_ids", [])
    }
    telemetry["summary"]["expected_scope_jobs"] = len(scope_ids)
    telemetry["summary"]["measured_scope_jobs"] = len(measured_ids & scope_ids)
    telemetry["summary"]["complete_scope_coverage"] = (
        measured_ids & scope_ids
    ) == scope_ids
    if not telemetry["summary"]["complete_scope_coverage"]:
        telemetry["status"] = "measurement_incomplete"
    _write_json_atomic(telemetry_path, telemetry)
    print(json.dumps(telemetry["summary"], indent=2))
    print(f"Telemetry: {telemetry_path}")
    return 0 if telemetry["status"] == "complete" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first QLoRA validation-selection scope one resumable job "
            "at a time while persisting cloud runtime telemetry."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--telemetry-output", type=Path)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_interval_seconds <= 0:
        raise ValueError("--sample-interval-seconds must be positive.")
    if args.max_jobs < 1:
        raise ValueError("--max-jobs must be positive.")
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
