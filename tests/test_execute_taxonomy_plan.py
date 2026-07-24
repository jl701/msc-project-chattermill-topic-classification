from __future__ import annotations

import importlib.util
from pathlib import Path

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
