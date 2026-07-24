from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "execute_taxonomy_plan.py"
SPEC = importlib.util.spec_from_file_location("execute_taxonomy_plan", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _plan() -> dict[str, object]:
    return {
        "schema_version": "taxonomy_cloud_execution_plan_v2",
        "formal_execution_blocked": False,
        "output_root": "outputs/example",
        "jobs": [
            {
                "job_id": "train-a",
                "stage": "formal-train",
                "depends_on": [],
                "official_splits_opened": ["train"],
                "argv": ["python", "runner.py", "--method", "e5_base_v2"],
                "executor": "local_gpu",
            },
            {
                "job_id": "validation-a",
                "stage": "formal-score-validation",
                "depends_on": ["train-a"],
                "official_splits_opened": ["validation"],
                "argv": ["python", "runner.py", "--method", "e5_base_v2"],
                "executor": "local_gpu",
            },
            {
                "job_id": "test-a",
                "stage": "formal-score-test",
                "depends_on": ["validation-a"],
                "official_splits_opened": ["test"],
                "argv": ["python", "runner.py", "--method", "e5_base_v2"],
                "executor": "local_gpu",
            },
            {
                "job_id": "train-b",
                "stage": "formal-train",
                "depends_on": [],
                "official_splits_opened": ["train"],
                "argv": [
                    "python",
                    "runner.py",
                    "--method",
                    "strict_train_only_tfidf",
                ],
                "executor": "local_cpu",
            },
        ],
    }


def test_validation_default_excludes_every_test_job() -> None:
    selected = MODULE.select_jobs(_plan(), methods=["e5_base_v2"])
    assert [job["job_id"] for job in selected] == ["train-a", "validation-a"]
    assert all("test" not in job["official_splits_opened"] for job in selected)


def test_test_requires_explicit_opt_in() -> None:
    selected = MODULE.select_jobs(
        _plan(), methods=["e5_base_v2"], include_official_test=True
    )
    assert [job["job_id"] for job in selected] == [
        "train-a",
        "validation-a",
        "test-a",
    ]


def test_method_filter_cannot_omit_an_internal_dependency() -> None:
    plan = _plan()
    plan["jobs"][1]["argv"][-1] = "strict_train_only_tfidf"
    with pytest.raises(ValueError, match="omits required dependencies"):
        MODULE.select_jobs(plan, methods=["strict_train_only_tfidf"])


def test_state_contract_rejects_a_different_plan_hash(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = MODULE._empty_state(
        plan_path=tmp_path / "plan.json",
        plan_sha256="first",
        methods=("e5_base_v2",),
        include_official_test=False,
    )
    MODULE.write_state(state_path, state)
    with pytest.raises(ValueError, match="plan_sha256"):
        MODULE.load_or_create_state(
            state_path,
            plan_path=tmp_path / "plan.json",
            plan_sha256="second",
            methods=("e5_base_v2",),
            include_official_test=False,
        )


def test_dry_run_simulates_completed_dependencies_without_writing(
    tmp_path: Path,
) -> None:
    jobs = MODULE.select_jobs(_plan(), methods=["e5_base_v2"])
    state = MODULE._empty_state(
        plan_path=tmp_path / "plan.json",
        plan_sha256="plan",
        methods=("e5_base_v2",),
        include_official_test=False,
    )
    state_path = tmp_path / "state.json"
    assert (
        MODULE.execute_jobs(
            jobs,
            state=state,
            state_path=state_path,
            dry_run=True,
            stop_after=None,
        )
        == 0
    )
    assert not state_path.exists()


def test_resumable_executor_adds_only_administrative_threshold_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> SimpleNamespace:
        observed.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    jobs = [
        {
            "job_id": "threshold",
            "stage": "tuning-select-threshold",
            "depends_on": [],
            "official_splits_opened": ["validation"],
            "argv": ["python", "runner.py", "--phase", "select-threshold"],
            "executor": "local_cpu",
        }
    ]
    state = MODULE._empty_state(
        plan_path=tmp_path / "plan.json",
        plan_sha256="plan",
        methods=("strict_train_only_tfidf",),
        include_official_test=False,
    )
    assert (
        MODULE.execute_jobs(
            jobs,
            state=state,
            state_path=tmp_path / "state.json",
            dry_run=False,
            stop_after=None,
        )
        == 0
    )
    assert observed == [
        ["python", "runner.py", "--phase", "select-threshold", "--resume"]
    ]


def test_parallel_executor_runs_only_ready_jobs_and_records_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [
        {
            "job_id": "first",
            "stage": "tuning-train",
            "depends_on": [],
            "official_splits_opened": ["train"],
            "argv": ["python", "first.py"],
            "executor": "local_cpu",
        },
        {
            "job_id": "second",
            "stage": "tuning-train",
            "depends_on": [],
            "official_splits_opened": ["train"],
            "argv": ["python", "second.py"],
            "executor": "local_cpu",
        },
        {
            "job_id": "final",
            "stage": "tuning-score-validation",
            "depends_on": ["first", "second"],
            "official_splits_opened": ["validation"],
            "argv": ["python", "final.py"],
            "executor": "local_cpu",
        },
    ]
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    finished: set[str] = set()

    def fake_run(argv: list[str], **_: object) -> SimpleNamespace:
        nonlocal active, maximum_active
        name = Path(argv[1]).stem
        with lock:
            if name == "final":
                assert finished == {"first", "second"}
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
            finished.add(name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    state = MODULE._empty_state(
        plan_path=tmp_path / "plan.json",
        plan_sha256="plan",
        methods=("strict_train_only_tfidf",),
        include_official_test=False,
    )
    state_path = tmp_path / "state.json"
    assert (
        MODULE.execute_jobs(
            jobs,
            state=state,
            state_path=state_path,
            dry_run=False,
            stop_after=None,
            max_workers=2,
        )
        == 0
    )
    assert maximum_active == 2
    assert set(state["completed_job_ids"]) == {"first", "second", "final"}
    assert state["running_job_ids"] == []


def test_atomic_state_write_retries_a_transient_windows_reader_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    real_replace = MODULE.os.replace
    calls = 0

    def transiently_locked(source: str, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("simulated Windows reader lock")
        real_replace(source, target)

    monkeypatch.setattr(MODULE.os, "replace", transiently_locked)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    state = MODULE._empty_state(
        plan_path=tmp_path / "plan.json",
        plan_sha256="plan",
        methods=("strict_train_only_tfidf",),
        include_official_test=False,
    )
    MODULE.write_state(state_path, state)
    assert calls == 2
    assert state_path.exists()
