"""Frozen formal job graph for genuine trainable two-stage validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from msc_project.experiments.taxonomy_protocol import (
    LEVELS,
    TaxonomyFold,
    canonical_aspects,
    load_precloud_config,
    registered_folds,
    training_scope_id,
)


PROTOCOL_ID = "taxonomy_two_stage_formal_v1"
TRAINABLE_METHODS = (
    "distilbert_review_candidate_cross_encoder",
    "qwen_candidate_pair_qlora",
)
LEARNING_RATES = {
    "distilbert_review_candidate_cross_encoder": (2e-5, 3e-5, 5e-5),
    "qwen_candidate_pair_qlora": (2e-6, 5e-6, 1e-5),
}


def formal_storage_budget_report(
    execution_config: Mapping[str, object],
    *,
    observed_used_gib: float,
) -> dict[str, object]:
    """Validate the explicit RunPod volume quota instead of shared-backend ``df``.

    RunPod network volumes may report the capacity of their shared backing
    filesystem to ``shutil.disk_usage``.  The formal release gate therefore
    uses the quota shown by RunPod together with a conservative campaign-growth
    allowance registered before execution.
    """

    raw = execution_config.get("storage_budget")
    if not isinstance(raw, Mapping):
        raise ValueError("Formal execution config lacks a storage_budget object.")
    try:
        capacity = float(raw["network_volume_capacity_gib"])
        growth = float(raw["campaign_growth_upper_bound_gib"])
        minimum_headroom = float(raw["minimum_final_headroom_gib"])
        used = float(observed_used_gib)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Formal storage budget values are invalid.") from error
    if capacity <= 0 or growth < 0 or minimum_headroom < 0 or used < 0:
        raise ValueError("Formal storage budget values must be non-negative.")
    if used > capacity:
        raise ValueError("Observed RunPod volume use exceeds its declared quota.")
    projected_headroom = capacity - used - growth
    return {
        "network_volume_capacity_gib": capacity,
        "observed_used_gib_upper_bound": used,
        "campaign_growth_upper_bound_gib": growth,
        "minimum_final_headroom_gib": minimum_headroom,
        "projected_final_headroom_gib": projected_headroom,
        "status": "pass" if projected_headroom >= minimum_headroom else "fail",
        "source": "runpod_declared_quota_not_shared_backend_df",
    }


@dataclass(frozen=True)
class FormalJob:
    phase: str
    method_id: str
    training_scope_id: str
    learning_rate: float | None
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["command"] = list(self.command)
        return value


def formal_job_id(job: FormalJob) -> str:
    """Return the immutable identity used by campaign and worker manifests."""

    rate = "selected" if job.learning_rate is None else f"{job.learning_rate:.0e}"
    return f"{job.phase}__{job.method_id}__{job.training_scope_id}__{rate}"


def scope_folds() -> dict[str, tuple[TaxonomyFold, ...]]:
    config = load_precloud_config()
    aspects = canonical_aspects(config)
    grouped: dict[str, list[TaxonomyFold]] = {}
    for level in LEVELS:
        for fold in registered_folds(level, config):
            grouped.setdefault(training_scope_id(fold, aspects), []).append(fold)
    result = {
        scope: tuple(sorted(folds, key=lambda fold: (fold.level, fold.fold_id)))
        for scope, folds in sorted(grouped.items())
    }
    if len(result) != 26:
        raise AssertionError(f"Formal training-scope count changed: {len(result)}")
    if sum(len(fold.conditions) for folds in result.values() for fold in folds) != 99:
        raise AssertionError("Formal fold-condition count changed.")
    return result


def representative_fold(folds: Sequence[TaxonomyFold]) -> TaxonomyFold:
    if not folds:
        raise ValueError("A training scope must contain a registered fold.")
    first = folds[0]
    heldout = set(first.heldout_aspects)
    for fold in folds:
        if set(fold.heldout_aspects) != heldout or set(fold.seen_aspects) != set(
            first.seen_aspects
        ):
            raise AssertionError("One training scope contains different held-out evidence.")
    return sorted(
        folds,
        key=lambda fold: (
            0 if fold.level == "L2" else 1,
            0 if fold.level == "L3" else 1,
            fold.fold_id,
        ),
    )[0]


def learning_rates(method_id: str) -> tuple[float, ...]:
    try:
        return LEARNING_RATES[method_id]
    except KeyError as error:
        raise ValueError(f"Unsupported formal trainable method: {method_id!r}") from error


def learning_rate_token(value: float) -> str:
    if value <= 0:
        raise ValueError("Learning rate must be positive.")
    return f"{value:.0e}".replace("+", "").replace("-0", "-")


def candidate_result_path(
    output_root: Path,
    method_id: str,
    scope_id: str,
    learning_rate: float,
) -> Path:
    return (
        output_root
        / "selection_candidates"
        / method_id
        / scope_id
        / f"lr-{learning_rate_token(learning_rate)}.json"
    )


def selection_path(output_root: Path, method_id: str, scope_id: str) -> Path:
    return output_root / "selections" / method_id / f"{scope_id}.json"


def build_formal_jobs(
    *,
    output_root: str = "/workspace/taxonomy_two_stage_formal_v1",
    data_dir: str = "/workspace/data/fabsa",
    methods: Iterable[str] | None = None,
    training_scope_ids: Iterable[str] | None = None,
) -> list[FormalJob]:
    selected_methods = tuple(TRAINABLE_METHODS if methods is None else methods)
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("Formal methods must be a non-empty unique sequence.")
    unknown_methods = sorted(set(selected_methods) - set(TRAINABLE_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown formal methods: {unknown_methods}")
    registered_scopes = scope_folds()
    selected_scopes = tuple(
        registered_scopes if training_scope_ids is None else training_scope_ids
    )
    if not selected_scopes or len(set(selected_scopes)) != len(selected_scopes):
        raise ValueError("Formal training scopes must be a non-empty unique sequence.")
    unknown_scopes = sorted(set(selected_scopes) - set(registered_scopes))
    if unknown_scopes:
        raise ValueError(f"Unknown formal training scopes: {unknown_scopes}")
    script = "scripts/run_taxonomy_two_stage_trainable_validation.py"
    jobs: list[FormalJob] = []
    for method_id in selected_methods:
        for scope_id in selected_scopes:
            for rate in learning_rates(method_id):
                command = (
                    "python",
                    script,
                    "--phase",
                    "train-candidate",
                    "--method",
                    method_id,
                    "--scope-id",
                    scope_id,
                    "--learning-rate",
                    str(rate),
                    "--output-root",
                    output_root,
                    "--data-dir",
                    data_dir,
                    "--resume",
                )
                jobs.append(FormalJob("train-candidate", method_id, scope_id, rate, command))
            jobs.append(
                FormalJob(
                    "select-scope",
                    method_id,
                    scope_id,
                    None,
                    (
                        "python",
                        script,
                        "--phase",
                        "select-scope",
                        "--method",
                        method_id,
                        "--scope-id",
                        scope_id,
                        "--output-root",
                        output_root,
                        "--data-dir",
                        data_dir,
                    ),
                )
            )
            jobs.append(
                FormalJob(
                    "score-selected",
                    method_id,
                    scope_id,
                    None,
                    (
                        "python",
                        script,
                        "--phase",
                        "score-selected",
                        "--method",
                        method_id,
                        "--scope-id",
                        scope_id,
                        "--output-root",
                        output_root,
                        "--data-dir",
                        data_dir,
                        "--resume",
                    ),
                )
            )
    return jobs


def validate_formal_job_graph(jobs: Sequence[FormalJob]) -> dict[str, int]:
    forbidden = {"--include-official-test", "score-test", "analyse-test", "test.csv"}
    for job in jobs:
        lowered = {value.lower() for value in job.command}
        if lowered & forbidden or any("test.csv" in value.lower() for value in job.command):
            raise ValueError("Official-test work entered the formal validation job graph.")
    counts = {
        phase: sum(job.phase == phase for job in jobs)
        for phase in ("train-candidate", "select-scope", "score-selected")
    }
    if counts != {"train-candidate": 156, "select-scope": 52, "score-selected": 52}:
        raise AssertionError(f"Formal job counts changed: {counts}")
    return counts


def validate_formal_job_subset(jobs: Sequence[FormalJob]) -> dict[str, int]:
    """Validate a complete, dependency-closed subset of method/scope units."""

    if not jobs:
        raise ValueError("A formal worker job subset must not be empty.")
    forbidden = {"--include-official-test", "score-test", "analyse-test", "test.csv"}
    grouped: dict[tuple[str, str], list[FormalJob]] = {}
    identities: set[str] = set()
    registered_scope_ids = set(scope_folds())
    for job in jobs:
        lowered = {value.lower() for value in job.command}
        if lowered & forbidden or any("test.csv" in value.lower() for value in job.command):
            raise ValueError("Official-test work entered a formal worker graph.")
        if job.method_id not in TRAINABLE_METHODS:
            raise ValueError(f"Unknown worker method: {job.method_id!r}")
        if job.training_scope_id not in registered_scope_ids:
            raise ValueError(f"Unknown worker scope: {job.training_scope_id!r}")
        identity = formal_job_id(job)
        if identity in identities:
            raise ValueError(f"Duplicate formal worker job: {identity}")
        identities.add(identity)
        grouped.setdefault((job.method_id, job.training_scope_id), []).append(job)

    expected_phases = {
        "train-candidate": 3,
        "select-scope": 1,
        "score-selected": 1,
    }
    for (method_id, scope_id), values in grouped.items():
        phase_counts = {
            phase: sum(value.phase == phase for value in values)
            for phase in expected_phases
        }
        if phase_counts != expected_phases:
            raise ValueError(
                f"Worker unit is not dependency-complete for {method_id}/{scope_id}: "
                f"{phase_counts}"
            )
        observed_rates = sorted(
            value.learning_rate
            for value in values
            if value.phase == "train-candidate"
        )
        if observed_rates != sorted(learning_rates(method_id)):
            raise ValueError(
                f"Worker learning-rate set changed for {method_id}/{scope_id}."
            )
    return {
        "train-candidate": sum(job.phase == "train-candidate" for job in jobs),
        "select-scope": sum(job.phase == "select-scope" for job in jobs),
        "score-selected": sum(job.phase == "score-selected" for job in jobs),
    }


def variant_map(fold: TaxonomyFold, condition: str) -> dict[str, str]:
    if condition not in fold.conditions:
        raise ValueError(f"Condition {condition!r} is not registered for {fold.fold_id}.")
    values = {aspect: "name_and_description" for aspect in fold.evaluation_aspects}
    if fold.level == "L3":
        mapping = {"N": "name_only", "D": "name_and_description", "R": "rich"}
        for aspect, marker in zip(fold.heldout_aspects, condition):
            values[aspect] = mapping[marker]
    else:
        heldout_variant = "name_and_description" if condition == "D" else "name_only"
        for aspect in fold.heldout_aspects:
            values[aspect] = heldout_variant
    return values


def plan_payload(jobs: Sequence[FormalJob]) -> dict[str, object]:
    counts = validate_formal_job_graph(jobs)
    return {
        "schema_version": "taxonomy_two_stage_formal_job_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "allowed_splits": ["train", "validation"],
        "include_official_test": False,
        "test_contract_count": 0,
        "unique_training_scopes": len(scope_folds()),
        "fold_condition_count_per_method": 99,
        "job_counts": counts,
        "jobs": [job.to_dict() for job in jobs],
    }
