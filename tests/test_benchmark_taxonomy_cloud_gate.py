from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_taxonomy_cloud_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_taxonomy_cloud_gate", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _job(
    job_id: str,
    stage: str,
    depends_on: list[str],
    *,
    splits: list[str],
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "stage": stage,
        "depends_on": depends_on,
        "official_splits_opened": splits,
        "argv": [
            "python",
            "runner.py",
            "--method",
            MODULE.METHOD_ID,
        ],
        "executor": (
            "cloud_control"
            if stage in {"tuning-select-threshold", "parameter-selection"}
            else "cloud_gpu"
        ),
    }


def _plan() -> dict[str, object]:
    jobs = []
    thresholds = []
    for candidate in range(1, 4):
        prefix = f"candidate-{candidate}"
        train = f"{prefix}-train"
        score = f"{prefix}-score"
        threshold = f"{prefix}-threshold"
        jobs.extend(
            [
                _job(train, "tuning-train", [], splits=["train"]),
                _job(
                    score,
                    "tuning-score-validation",
                    [train],
                    splits=["train", "validation"],
                ),
                _job(
                    threshold,
                    "tuning-select-threshold",
                    [score],
                    splits=["train", "validation"],
                ),
            ]
        )
        thresholds.append(threshold)
    jobs.append(
        _job(
            MODULE.TARGET_JOB_ID,
            "parameter-selection",
            thresholds,
            splits=[],
        )
    )
    return {"jobs": jobs}


def test_first_cloud_gate_is_exactly_one_sealed_tuning_scope() -> None:
    jobs = MODULE.validation_jobs(_plan())
    scope = MODULE.first_scope_jobs(jobs)
    assert len(scope) == 10
    assert scope[-1]["job_id"] == MODULE.TARGET_JOB_ID
    assert [job["stage"] for job in scope] == [
        "tuning-train",
        "tuning-score-validation",
        "tuning-select-threshold",
        "tuning-train",
        "tuning-score-validation",
        "tuning-select-threshold",
        "tuning-train",
        "tuning-score-validation",
        "tuning-select-threshold",
        "parameter-selection",
    ]
    assert all("test" not in job["official_splits_opened"] for job in scope)


def test_executor_command_cannot_open_official_test(tmp_path: Path) -> None:
    command = MODULE.executor_command(
        tmp_path / "plan.json", tmp_path / "state.json"
    )
    assert "--include-official-test" not in command
    assert command[-4:] == ["--stop-after", "1", "--max-workers", "1"]


def test_runtime_summary_separates_training_and_scoring() -> None:
    attempts = [
        {
            "status": "complete",
            "stage": "tuning-train",
            "wall_seconds": 20.0,
            "peak_device_memory_used_mib": 10000.0,
            "artifact_delta": {
                "measured_training_pairs": 4096,
                "checkpoint_bytes": 100,
            },
        },
        {
            "status": "complete",
            "stage": "tuning-score-validation",
            "wall_seconds": 10.0,
            "peak_device_memory_used_mib": 8000.0,
            "artifact_delta": {
                "measured_score_pairs": 2500,
                "score_csv_bytes": 200,
            },
        },
        {
            "status": "interrupted",
            "stage": "tuning-train",
            "wall_seconds": 99.0,
        },
    ]
    summary = MODULE.summarise_attempts(attempts, target_complete=False)
    assert summary["completed_measured_jobs"] == 2
    assert summary["training_wall_seconds"] == 20.0
    assert summary["scoring_wall_seconds"] == 10.0
    assert summary["measured_training_pairs"] == 4096
    assert summary["measured_score_pairs"] == 2500
    assert summary["score_pairs_per_wall_second"] == 250.0
    assert summary["peak_device_memory_used_mib"] == 10000.0
    assert summary["checkpoint_bytes"] == 100
    assert summary["score_csv_bytes"] == 200
