from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_IDS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_candidate_pair",
    "qwen_candidate_pair_qlora",
)
TEST_STAGES = {
    "formal-score-test",
    "formal-analyse-test",
    "formal-primary-comparisons",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job_method(job: dict[str, object]) -> str | None:
    argv = list(job["argv"])
    if "--method" not in argv:
        return None
    index = argv.index("--method")
    return str(argv[index + 1])


def load_plan(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "taxonomy_cloud_execution_plan_v2":
        raise ValueError("Expected a taxonomy_cloud_execution_plan_v2 plan.")
    if value.get("formal_execution_blocked") is not False:
        raise ValueError("The execution plan is still formally blocked.")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("The execution plan contains no jobs.")
    job_ids = [str(job["job_id"]) for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("The execution plan contains duplicate job IDs.")
    return value


def select_jobs(
    plan: dict[str, object],
    *,
    methods: Iterable[str],
    include_official_test: bool = False,
) -> list[dict[str, object]]:
    selected_methods = set(methods)
    jobs = []
    for raw_job in plan["jobs"]:
        job = dict(raw_job)
        if _job_method(job) not in selected_methods:
            continue
        opens_test = "test" in job["official_splits_opened"]
        is_test_stage = str(job["stage"]) in TEST_STAGES
        if (opens_test or is_test_stage) and not include_official_test:
            continue
        jobs.append(job)

    if not jobs:
        raise ValueError("No jobs matched the requested methods and split guard.")

    selected_ids = {str(job["job_id"]) for job in jobs}
    all_ids = {str(job["job_id"]) for job in plan["jobs"]}
    missing = {
        str(dependency)
        for job in jobs
        for dependency in job["depends_on"]
        if str(dependency) not in all_ids
    }
    if missing:
        raise ValueError(f"Plan references unknown dependencies: {sorted(missing)}")

    if not include_official_test:
        forbidden = [
            str(job["job_id"])
            for job in jobs
            if "test" in job["official_splits_opened"]
            or str(job["stage"]) in TEST_STAGES
        ]
        if forbidden:
            raise AssertionError(f"Official-test guard failed: {forbidden}")

    unresolved_inside_plan = {
        str(dependency)
        for job in jobs
        for dependency in job["depends_on"]
        if str(dependency) not in selected_ids
    }
    if unresolved_inside_plan:
        raise ValueError(
            "The filtered run omits required dependencies. Select every method "
            f"needed by the requested jobs: {sorted(unresolved_inside_plan)[:5]}"
        )
    return jobs


def default_state_path(
    plan: dict[str, object],
    plan_path: Path,
    methods: Sequence[str],
    *,
    include_official_test: bool,
) -> Path:
    scope = "-".join(methods)
    split_scope = "through-test" if include_official_test else "through-validation"
    return (
        PROJECT_ROOT
        / str(plan["output_root"])
        / "_plan_state"
        / f"{plan_path.stem}.{scope}.{split_scope}.json"
    )


def _empty_state(
    *,
    plan_path: Path,
    plan_sha256: str,
    methods: Sequence[str],
    include_official_test: bool,
) -> dict[str, object]:
    return {
        "schema_version": "taxonomy_execution_state_v1",
        "plan_path": plan_path.as_posix(),
        "plan_sha256": plan_sha256,
        "methods": list(methods),
        "include_official_test": include_official_test,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "current_job_id": None,
        "running_job_ids": [],
        "completed_job_ids": [],
        "failed_jobs": [],
    }


def load_or_create_state(
    path: Path,
    *,
    plan_path: Path,
    plan_sha256: str,
    methods: Sequence[str],
    include_official_test: bool,
) -> dict[str, object]:
    if not path.exists():
        return _empty_state(
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            methods=methods,
            include_official_test=include_official_test,
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "taxonomy_execution_state_v1",
        "plan_sha256": plan_sha256,
        "methods": list(methods),
        "include_official_test": include_official_test,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"State contract mismatch for {key}: "
                f"{value.get(key)!r} != {expected_value!r}"
            )
    return value


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _utc_now()
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        for attempt in range(20):
            try:
                os.replace(temporary_name, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def execute_jobs(
    jobs: Sequence[dict[str, object]],
    *,
    state: dict[str, object],
    state_path: Path,
    dry_run: bool,
    stop_after: int | None,
    max_workers: int = 1,
) -> int:
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")
    completed = set(str(value) for value in state["completed_job_ids"])
    pending = [
        job for job in jobs if str(job["job_id"]) not in completed
    ]

    def emit_start(job: dict[str, object]) -> list[str]:
        job_id = str(job["job_id"])
        argv = [str(value) for value in job["argv"]]
        if (
            str(job["stage"])
            in {"tuning-select-threshold", "formal-select-threshold"}
            and "--resume" not in argv
        ):
            argv.append("--resume")
        print(
            json.dumps(
                {
                    "event": "job_start",
                    "job_id": job_id,
                    "stage": job["stage"],
                    "executor": job["executor"],
                    "official_splits_opened": job["official_splits_opened"],
                    "argv": argv,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return argv

    if dry_run:
        executed = 0
        while pending:
            ready = next(
                (
                    job
                    for job in pending
                    if all(
                        str(value) in completed
                        for value in job["depends_on"]
                    )
                ),
                None,
            )
            if ready is None:
                raise RuntimeError("Dry-run dependency graph cannot make progress.")
            if stop_after is not None and executed >= stop_after:
                break
            emit_start(ready)
            job_id = str(ready["job_id"])
            completed.add(job_id)
            pending.remove(ready)
            executed += 1
        return 0

    state["current_job_id"] = None
    state["running_job_ids"] = []
    write_state(state_path, state)
    executed = 0
    scheduled = 0
    failure_code: int | None = None
    in_flight: dict[Future[int], dict[str, object]] = {}

    def run(argv: Sequence[str]) -> int:
        return subprocess.run(
            list(argv), cwd=PROJECT_ROOT, check=False
        ).returncode

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or in_flight:
            if failure_code is None:
                for job in list(pending):
                    if len(in_flight) >= max_workers:
                        break
                    if stop_after is not None and scheduled >= stop_after:
                        break
                    if not all(
                        str(value) in completed
                        for value in job["depends_on"]
                    ):
                        continue
                    job_id = str(job["job_id"])
                    argv = emit_start(job)
                    pending.remove(job)
                    state["running_job_ids"].append(job_id)
                    write_state(state_path, state)
                    in_flight[pool.submit(run, argv)] = job
                    scheduled += 1

            if not in_flight:
                if stop_after is not None and scheduled >= stop_after:
                    break
                unresolved = {
                    str(job["job_id"]): [
                        str(value)
                        for value in job["depends_on"]
                        if str(value) not in completed
                    ]
                    for job in pending
                }
                raise RuntimeError(
                    "Dependency graph cannot make progress: "
                    f"{list(unresolved.items())[:3]}"
                )

            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                job = in_flight.pop(future)
                job_id = str(job["job_id"])
                state["running_job_ids"].remove(job_id)
                try:
                    returncode = int(future.result())
                    error = None
                except BaseException as exc:
                    returncode = 1
                    error = f"{type(exc).__name__}: {exc}"
                if returncode != 0:
                    failure_code = failure_code or returncode
                    state["failed_jobs"].append(
                        {
                            "job_id": job_id,
                            "returncode": returncode,
                            "error": error,
                            "failed_at": _utc_now(),
                        }
                    )
                else:
                    state["completed_job_ids"].append(job_id)
                    completed.add(job_id)
                    executed += 1
                    print(
                        json.dumps(
                            {
                                "event": "job_complete",
                                "job_id": job_id,
                                "completed_in_this_invocation": executed,
                                "completed_in_state": len(completed),
                            }
                        ),
                        flush=True,
                    )
                write_state(state_path, state)

    if failure_code is not None:
        return failure_code
    if state["running_job_ids"]:
        raise AssertionError("Execution finished with jobs still marked running.")
    state["current_job_id"] = None
    write_state(state_path, state)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a dependency-ordered taxonomy plan with an official-test "
            "guard and resumable atomic state."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        choices=METHOD_IDS,
        required=True,
        dest="methods",
    )
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Maximum independent plan jobs to run concurrently. Keep this at "
            "one for GPU methods; four is the validated local TF-IDF setting."
        ),
    )
    parser.add_argument(
        "--include-official-test",
        action="store_true",
        help=(
            "Explicitly include test-scoring and test-analysis jobs. Omit this "
            "flag throughout validation and model selection."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("--stop-after must be positive.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive.")
    methods = tuple(dict.fromkeys(args.methods))
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    jobs = select_jobs(
        plan,
        methods=methods,
        include_official_test=args.include_official_test,
    )
    plan_sha256 = _sha256(plan_path)
    state_path = (
        args.state_path.resolve()
        if args.state_path
        else default_state_path(
            plan,
            plan_path,
            methods,
            include_official_test=args.include_official_test,
        )
    )
    state = load_or_create_state(
        state_path,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        methods=methods,
        include_official_test=args.include_official_test,
    )
    print(
        json.dumps(
            {
                "event": "execution_scope",
                "plan_sha256": plan_sha256,
                "methods": methods,
                "include_official_test": args.include_official_test,
                "selected_jobs": len(jobs),
                "already_completed": len(state["completed_job_ids"]),
                "state_path": state_path.as_posix(),
                "dry_run": args.dry_run,
                "max_workers": args.max_workers,
            }
        ),
        flush=True,
    )
    return execute_jobs(
        jobs,
        state=state,
        state_path=state_path,
        dry_run=args.dry_run,
        stop_after=args.stop_after,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
