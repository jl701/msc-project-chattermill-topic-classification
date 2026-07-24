from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_artifacts import summary_path_for
from msc_project.experiments.taxonomy_methods import METHOD_IDS
from msc_project.experiments.taxonomy_protocol import (
    load_precloud_config,
    registered_folds,
    scientific_protocol_sha256,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions
from msc_project.experiments.taxonomy_tuning import (
    registered_tuning_candidates,
)


DEFAULT_OUTPUT_ROOT = Path(
    "outputs/experimental/taxonomy_generalisation_formal_v1"
)
DEFAULT_SELECTION_DIR = DEFAULT_OUTPUT_ROOT / "_parameter_selections"
DEFAULT_SHARD_COUNT = 8
OFFICIAL_ROWS = {"validation": 1057, "test": 1587}


def _posix(path: Path) -> str:
    return path.as_posix()


def _training_scope_representatives() -> dict[str, object]:
    representatives = {}
    for level in ("L1", "L2", "L3", "L4"):
        for fold in registered_folds(level):
            representatives.setdefault(training_scope_id(fold), fold)
    return representatives


def _selection_path(
    selection_dir: Path,
    method_id: str,
    scope: str,
) -> Path:
    if len(registered_tuning_candidates(method_id)) == 1:
        return selection_dir / f"{method_id}.json"
    return selection_dir / method_id / f"{scope}.json"


def _job(
    jobs: list[dict[str, object]],
    job_id: str,
    stage: str,
    argv: list[str],
    *,
    depends_on: tuple[str, ...] = (),
    official_splits: tuple[str, ...] = (),
) -> str:
    if any(value["job_id"] == job_id for value in jobs):
        raise ValueError(f"Duplicate cloud job ID: {job_id}")
    jobs.append(
        {
            "job_id": job_id,
            "stage": stage,
            "depends_on": list(depends_on),
            "official_splits_opened": list(official_splits),
            "argv": argv,
            "posix_command": shlex.join(argv),
        }
    )
    return job_id


def _runner_argv(
    *,
    phase: str,
    purpose: str,
    method_id: str,
    level: str,
    fold_id: str,
    output_root: Path,
    seed: int,
    shard_count: int,
    candidate_id: str | None = None,
    parameter_selection: Path | None = None,
) -> list[str]:
    argv = [
        "python",
        "scripts/run_taxonomy_generalisation.py",
        "--phase",
        phase,
        "--purpose",
        purpose,
        "--method",
        method_id,
        "--level",
        level,
        "--fold-id",
        fold_id,
        "--output-root",
        _posix(output_root),
        "--seed",
        str(seed),
        "--shard-count",
        str(shard_count),
    ]
    if candidate_id is not None:
        argv.extend(["--candidate-id", candidate_id])
    if parameter_selection is not None:
        argv.extend(["--parameter-selection", _posix(parameter_selection)])
    if phase in {"train", "score-validation", "score-test"}:
        argv.append("--resume")
    if phase in {"score-validation", "score-test"}:
        argv.append("--all-shards")
    return argv


def build_plan(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    selection_dir: Path = DEFAULT_SELECTION_DIR,
    shard_count: int = DEFAULT_SHARD_COUNT,
) -> dict[str, object]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive.")
    resource = load_minimal_descriptions(require_approved=False)
    config = load_precloud_config()
    jobs: list[dict[str, object]] = []
    selection_jobs: dict[tuple[str, str], str] = {}
    core_representatives = _training_scope_representatives()

    # Nested seen-only tuning per independent outer training scope.
    for method_id in METHOD_IDS:
        candidates = registered_tuning_candidates(method_id)
        if len(candidates) == 1:
            selection_path = _selection_path(
                selection_dir,
                method_id,
                "global_fixed_recipe",
            )
            selection_jobs[(method_id, "global_fixed_recipe")] = _job(
                jobs,
                f"select-fixed-{method_id}",
                "parameter-selection",
                [
                    "python",
                    "scripts/summarise_taxonomy_tuning.py",
                    "--method",
                    method_id,
                    "--fixed",
                    "--output",
                    _posix(selection_path),
                ],
            )
            continue

        for scope, fold in core_representatives.items():
            summaries = []
            selection_dependencies = []
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                digest = str(candidate["parameters_sha256"])
                prefix = f"tune-{method_id}-{candidate_id}-{fold.fold_id}"
                train = _job(
                    jobs,
                    f"{prefix}-train",
                    "tuning-train",
                    _runner_argv(
                        phase="train",
                        purpose="tuning",
                        method_id=method_id,
                        level=fold.level,
                        fold_id=fold.fold_id,
                        output_root=output_root,
                        seed=13,
                        shard_count=shard_count,
                        candidate_id=candidate_id,
                    ),
                    official_splits=("train",),
                )
                score = _job(
                    jobs,
                    f"{prefix}-score-validation",
                    "tuning-score-validation",
                    _runner_argv(
                        phase="score-validation",
                        purpose="tuning",
                        method_id=method_id,
                        level=fold.level,
                        fold_id=fold.fold_id,
                        output_root=output_root,
                        seed=13,
                        shard_count=shard_count,
                        candidate_id=candidate_id,
                    ),
                    depends_on=(train,),
                    official_splits=("train", "validation"),
                )
                select = _job(
                    jobs,
                    f"{prefix}-select-threshold",
                    "tuning-select-threshold",
                    _runner_argv(
                        phase="select-threshold",
                        purpose="tuning",
                        method_id=method_id,
                        level=fold.level,
                        fold_id=fold.fold_id,
                        output_root=output_root,
                        seed=13,
                        shard_count=shard_count,
                        candidate_id=candidate_id,
                    ),
                    depends_on=(score,),
                    official_splits=("train", "validation"),
                )
                summaries.append(
                    summary_path_for(
                        output_root,
                        "tuning",
                        "validation",
                        method_id,
                        fold,
                        digest,
                        13,
                    )
                )
                selection_dependencies.append(select)
            selection_path = _selection_path(selection_dir, method_id, scope)
            argv = [
                "python",
                "scripts/summarise_taxonomy_tuning.py",
                "--method",
                method_id,
                "--fold-id",
                fold.fold_id,
                "--output",
                _posix(selection_path),
            ]
            for summary in summaries:
                argv.extend(["--summary", _posix(summary)])
            selection_jobs[(method_id, scope)] = _job(
                jobs,
                f"select-tuned-{method_id}-{scope}",
                "parameter-selection",
                argv,
                depends_on=tuple(selection_dependencies),
            )

    final_analysis_jobs = []
    training_jobs: dict[tuple[str, int, str], str] = {}
    validation_jobs: dict[tuple[str, int, str], str] = {}

    def add_formal_suite(
        method_id: str,
        seed: int,
        levels: tuple[str, ...],
    ) -> None:
        candidates = registered_tuning_candidates(method_id)
        is_nested = len(candidates) > 1
        folds = [
            fold
            for level in levels
            for fold in registered_folds(level)
        ]
        representatives = {}
        for fold in folds:
            representatives.setdefault(training_scope_id(fold), fold)
        for scope, fold in representatives.items():
            selection_scope = scope if is_nested else "global_fixed_recipe"
            selection_path = _selection_path(
                selection_dir,
                method_id,
                selection_scope,
            )
            job_id = _job(
                jobs,
                f"formal-{method_id}-seed{seed:04d}-{scope}-train",
                "formal-train",
                _runner_argv(
                    phase="train",
                    purpose="final",
                    method_id=method_id,
                    level=fold.level,
                    fold_id=fold.fold_id,
                    output_root=output_root,
                    seed=seed,
                    shard_count=shard_count,
                    parameter_selection=selection_path,
                ),
                depends_on=(selection_jobs[(method_id, selection_scope)],),
                official_splits=("train",),
            )
            training_jobs[(method_id, seed, scope)] = job_id

        # L1 validation creates the exact matched L2-D seen-calibration scores,
        # so a separate L2 validation model load is intentionally omitted.
        for fold in folds:
            if fold.level == "L2":
                continue
            scope = training_scope_id(fold)
            selection_scope = scope if is_nested else "global_fixed_recipe"
            selection_path = _selection_path(
                selection_dir,
                method_id,
                selection_scope,
            )
            if (
                seed == 13
                and is_nested
                and core_representatives[scope].fold_id == fold.fold_id
            ):
                # Nested tuning already produced the selected checkpoint and
                # exact seen-calibration score contract for this representative.
                validation_jobs[(method_id, seed, fold.fold_id)] = (
                    training_jobs[(method_id, seed, scope)]
                )
                continue
            job_id = _job(
                jobs,
                f"formal-{method_id}-seed{seed:04d}-{fold.fold_id}-score-validation",
                "formal-score-validation",
                _runner_argv(
                    phase="score-validation",
                    purpose="final",
                    method_id=method_id,
                    level=fold.level,
                    fold_id=fold.fold_id,
                    output_root=output_root,
                    seed=seed,
                    shard_count=shard_count,
                    parameter_selection=selection_path,
                ),
                depends_on=(training_jobs[(method_id, seed, scope)],),
                official_splits=("train", "validation"),
            )
            validation_jobs[(method_id, seed, fold.fold_id)] = job_id

        for fold in folds:
            scope = training_scope_id(fold)
            selection_scope = scope if is_nested else "global_fixed_recipe"
            selection_path = _selection_path(
                selection_dir,
                method_id,
                selection_scope,
            )
            if fold.level == "L2":
                l1_fold = next(
                    value
                    for value in registered_folds("L1")
                    if value.heldout_aspects == fold.heldout_aspects
                )
                validation_dependency = validation_jobs[
                    (method_id, seed, l1_fold.fold_id)
                ]
            else:
                validation_dependency = validation_jobs[
                    (method_id, seed, fold.fold_id)
                ]
            select = _job(
                jobs,
                f"formal-{method_id}-seed{seed:04d}-{fold.fold_id}-select-threshold",
                "formal-select-threshold",
                _runner_argv(
                    phase="select-threshold",
                    purpose="final",
                    method_id=method_id,
                    level=fold.level,
                    fold_id=fold.fold_id,
                    output_root=output_root,
                    seed=seed,
                    shard_count=shard_count,
                    parameter_selection=selection_path,
                ),
                depends_on=(validation_dependency,),
                official_splits=("train", "validation"),
            )
            test = _job(
                jobs,
                f"formal-{method_id}-seed{seed:04d}-{fold.fold_id}-score-test",
                "formal-score-test",
                _runner_argv(
                    phase="score-test",
                    purpose="final",
                    method_id=method_id,
                    level=fold.level,
                    fold_id=fold.fold_id,
                    output_root=output_root,
                    seed=seed,
                    shard_count=shard_count,
                    parameter_selection=selection_path,
                ),
                depends_on=(select,),
                official_splits=("train", "test"),
            )
            analysis = _job(
                jobs,
                f"formal-{method_id}-seed{seed:04d}-{fold.fold_id}-analyse-test",
                "formal-analyse-test",
                _runner_argv(
                    phase="analyse-test",
                    purpose="final",
                    method_id=method_id,
                    level=fold.level,
                    fold_id=fold.fold_id,
                    output_root=output_root,
                    seed=seed,
                    shard_count=shard_count,
                    parameter_selection=selection_path,
                ),
                depends_on=(test,),
                official_splits=("train", "test"),
            )
            final_analysis_jobs.append(analysis)

    for method_id in METHOD_IDS:
        add_formal_suite(method_id, 13, ("L1", "L2", "L3", "L4"))
    for seed in (23, 42):
        add_formal_suite("qwen_candidate_pair_qlora", seed, ("L1",))

    primary_analysis = _job(
        jobs,
        "formal-primary-comparisons-seed0013",
        "formal-primary-comparisons",
        [
            "python",
            "scripts/analyse_taxonomy_primary_comparisons.py",
            "--output-root",
            _posix(output_root),
            "--parameter-selection-dir",
            _posix(selection_dir),
            "--output",
            _posix(output_root / "final" / "primary_comparisons_seed0013.json"),
            "--shard-count",
            str(shard_count),
            "--seed",
            "13",
        ],
        depends_on=tuple(
            value
            for value in final_analysis_jobs
        ),
        official_splits=("train", "test"),
    )

    stage_counts = Counter(str(value["stage"]) for value in jobs)
    core_training_scopes = len(core_representatives)
    tunable_training_runs = {
        method_id: (
            len(registered_tuning_candidates(method_id))
            * core_training_scopes
            if len(registered_tuning_candidates(method_id)) > 1
            else 0
        )
        for method_id in METHOD_IDS
    }
    formal_training_scopes = {
        "strict_train_only_tfidf": core_training_scopes,
        "e5_base_v2": 0,
        "distilbert_review_candidate_cross_encoder": core_training_scopes,
        "frozen_qwen_candidate_pair": 0,
        "qwen_candidate_pair_qlora": core_training_scopes + 24,
    }
    tuning_reused_in_formal = {
        method_id: (
            core_training_scopes
            if len(registered_tuning_candidates(method_id)) > 1
            else 0
        )
        for method_id in METHOD_IDS
    }
    actual_training_executions = {
        method_id: (
            tunable_training_runs[method_id]
            + formal_training_scopes[method_id]
            - tuning_reused_in_formal[method_id]
        )
        for method_id in METHOD_IDS
    }
    qlora_training_runs = actual_training_executions[
        "qwen_candidate_pair_qlora"
    ]
    validation_claims_per_review = (
        12 * 11 * 3
        + 12 * 10 * 3
        + (9 + 9 + 10) * 3
    )
    validation_core_per_method = (
        OFFICIAL_ROWS["validation"] * validation_claims_per_review
    )
    test_core_per_method = OFFICIAL_ROWS["test"] * 1116
    core_per_method = validation_core_per_method + test_core_per_method
    qlora_extra_seed_pairs = 2 * (
        OFFICIAL_ROWS["validation"] * (12 * 11 * 3)
        + OFFICIAL_ROWS["test"] * 72
    )
    tuning_claims_per_candidate = OFFICIAL_ROWS["validation"] * sum(
        len(fold.seen_aspects) * 3
        for fold in core_representatives.values()
    )
    tuning_pairs = {
        "strict_train_only_tfidf": (
            len(registered_tuning_candidates("strict_train_only_tfidf"))
            * tuning_claims_per_candidate
        ),
        "distilbert_review_candidate_cross_encoder": (
            len(
                registered_tuning_candidates(
                    "distilbert_review_candidate_cross_encoder"
                )
            )
            * tuning_claims_per_candidate
        ),
        "qwen_candidate_pair_qlora": (
            len(registered_tuning_candidates("qwen_candidate_pair_qlora"))
            * tuning_claims_per_candidate
        ),
    }
    selected_tuning_scores_reused = {
        method_id: (
            tuning_claims_per_candidate
            if len(registered_tuning_candidates(method_id)) > 1
            else 0
        )
        for method_id in METHOD_IDS
    }

    return {
        "schema_version": "taxonomy_cloud_execution_plan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_protocol_sha256": scientific_protocol_sha256(config),
        "formal_execution_blocked": not (
            resource["status"] == "approved_and_frozen"
            and config["statistics"]["status"] == "approved_and_frozen"
        ),
        "remaining_gates": {
            "description_status": resource["status"],
            "required_description_status": "approved_and_frozen",
            "statistics_status": config["statistics"]["status"],
            "required_statistics_status": "approved_and_frozen",
        },
        "output_root": _posix(output_root),
        "parameter_selection_dir": _posix(selection_dir),
        "shard_count": shard_count,
        "jobs_total": len(jobs),
        "job_counts_by_stage": dict(sorted(stage_counts.items())),
        "primary_analysis_job": primary_analysis,
        "compute_summary": {
            "registered_tuning_training_runs": tunable_training_runs,
            "core_unique_training_scopes_per_method": core_training_scopes,
            "formal_task_specific_training_scopes": formal_training_scopes,
            "selected_tuning_scopes_reused_in_formal": tuning_reused_in_formal,
            "actual_task_specific_training_executions": actual_training_executions,
            "qlora_task_specific_training_runs_total": qlora_training_runs,
            "frozen_method_train_jobs_create_markers_without_model_loading": True,
            "unique_pair_scores": {
                "core_per_method_validation": validation_core_per_method,
                "core_per_method_test": test_core_per_method,
                "core_per_method_total": core_per_method,
                "qlora_extra_level1_seeds": qlora_extra_seed_pairs,
                "tuning_by_method": tuning_pairs,
                "selected_tuning_scores_reused_in_formal": (
                    selected_tuning_scores_reused
                ),
                "qlora_total_including_tuning_and_extra_seeds": (
                    core_per_method
                    + qlora_extra_seed_pairs
                    + tuning_pairs["qwen_candidate_pair_qlora"]
                    - selected_tuning_scores_reused[
                        "qwen_candidate_pair_qlora"
                    ]
                ),
                "frozen_qwen_plus_qlora_total": (
                    2 * core_per_method
                    + qlora_extra_seed_pairs
                    + tuning_pairs["qwen_candidate_pair_qlora"]
                    - selected_tuning_scores_reused[
                        "qwen_candidate_pair_qlora"
                    ]
                ),
                "all_methods_all_registered_scores": (
                    5 * core_per_method
                    + qlora_extra_seed_pairs
                    + sum(tuning_pairs.values())
                    - sum(selected_tuning_scores_reused.values())
                ),
            },
            "scoring_optimisations": [
                "L3 scores 42 unique rendered claims per review instead of 144 logical condition claims.",
                "L1 validation scores only the matched L2-D seen-aspect calibration grid; held-out validation targets are never scored.",
                "Every level scores seen candidates only during validation; held-out validation candidates are never rendered or scored.",
                "The L1 calibration score artifacts are reused by L2 threshold selection.",
                "Nested per-scope tuning checkpoints and seen-calibration scores are reused by the final seed-13 protocol.",
                "All shards are processed after one model load while remaining individually resumable.",
            ],
        },
        "jobs": jobs,
    }


def _write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cloud plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the blocked, dependency-aware taxonomy cloud plan."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    args = parser.parse_args()
    plan = build_plan(
        output_root=args.output_root,
        selection_dir=args.selection_dir,
        shard_count=args.shard_count,
    )
    _write_new(args.output, plan)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "jobs_total": plan["jobs_total"],
                "formal_execution_blocked": plan["formal_execution_blocked"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
