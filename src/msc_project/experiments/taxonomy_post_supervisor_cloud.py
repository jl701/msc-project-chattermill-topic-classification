"""Frozen job graph and three-worker partition for formal campaign v2.

The graph was amended before launch after the pre-registered rich-description
gate.  Training remains dependency-identical, while formal scoring is limited
to Level 2 N/D and Level 4 D.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_post_supervisor import post_supervisor_l2_folds
from msc_project.experiments.taxonomy_protocol import (
    TaxonomyFold,
    registered_folds,
    training_scope_id,
)
from msc_project.experiments.taxonomy_two_stage_formal import (
    FormalJob,
    LEARNING_RATES,
    TRAINABLE_METHODS,
    formal_job_id,
    learning_rates,
    representative_fold,
)


PROTOCOL_ID = "taxonomy_two_stage_formal_v2"
PARALLEL_PLAN_SCHEMA = "taxonomy_post_supervisor_parallel_plan_v2"
WORKER_MANIFEST_SCHEMA = "taxonomy_post_supervisor_worker_manifest_v2"
FROZEN_METHOD = "frozen_qwen_few_shot"
TRAINING_EXAMPLES_PER_SCOPE = 4096
LEARNING_RATE_COUNT = 3
STAGES_PER_CANDIDATE = 2
FORMAL_L2_CONDITIONS = ("N", "D")
FORMAL_L4_CONDITIONS = ("D",)


def formal_conditions(fold: TaxonomyFold) -> tuple[str, ...]:
    """Return the frozen post-gate conditions that may be formally scored."""

    if fold.level == "L2":
        expected = FORMAL_L2_CONDITIONS
    elif fold.level == "L4":
        expected = FORMAL_L4_CONDITIONS
    else:
        raise ValueError(f"Formal v2 does not register level {fold.level!r}.")
    if not set(expected).issubset(fold.conditions):
        raise ValueError(f"Fold {fold.fold_id} is missing a frozen condition.")
    return expected


def validate_safety_config(value: Mapping[str, object]) -> dict[str, object]:
    """Cross-check the frozen v2 execution contract against the job graph."""

    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected formal-v2 safety protocol ID.")
    if value.get("status") != "preregistered_after_rich_gate_before_formal_execution":
        raise ValueError("Formal-v2 safety config is not preregistered.")
    data = value.get("sealed_data_contract")
    training = value.get("training")
    evaluation = value.get("evaluation")
    if not all(isinstance(item, Mapping) for item in (data, training, evaluation)):
        raise ValueError("Formal-v2 safety config is structurally incomplete.")
    assert isinstance(data, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(evaluation, Mapping)
    if (
        list(data.get("allowed_splits", [])) != ["train", "validation"]
        or data.get("include_official_test") is not False
        or data.get("test_contract_count_required") != 0
    ):
        raise ValueError("Formal-v2 safety data boundary is invalid.")
    expected_training = {
        "unique_outer_scopes": 15,
        "candidate_checkpoint_count": 90,
        "trainable_work_unit_count": 150,
    }
    if any(training.get(key) != expected for key, expected in expected_training.items()):
        raise ValueError("Formal-v2 training counts do not match the frozen graph.")
    expected_evaluation = {
        "level_2_folds": 12,
        "level_4_folds": 3,
        "fold_condition_count_per_method": 27,
        "result_payload_count_all_three_methods": 81,
        "maximum_sentiments_per_aspect": 2,
        "third_sentiment_permitted": False,
    }
    if any(evaluation.get(key) != expected for key, expected in expected_evaluation.items()):
        raise ValueError("Formal-v2 evaluation counts do not match the frozen graph.")
    validate_formal_job_graph(build_formal_jobs())
    return {
        "status": "pass",
        "protocol_id": PROTOCOL_ID,
        "unique_training_scopes": 15,
        "trainable_job_count": 150,
        "result_payload_count": 81,
        "failure_count": 0,
        "test_contract_count": 0,
        "safety_config_sha256": canonical_sha256(value),
    }


def scope_folds() -> dict[str, tuple[TaxonomyFold, ...]]:
    folds = (*post_supervisor_l2_folds(), *registered_folds("L4"))
    grouped: dict[str, list[TaxonomyFold]] = {}
    for fold in folds:
        grouped.setdefault(training_scope_id(fold), []).append(fold)
    result = {
        scope: tuple(sorted(values, key=lambda fold: (fold.level, fold.fold_id)))
        for scope, values in sorted(grouped.items())
    }
    if len(result) != 15:
        raise AssertionError(f"Post-supervisor core scope count changed: {len(result)}.")
    if sum(len(formal_conditions(fold)) for values in result.values() for fold in values) != 27:
        raise AssertionError("Post-supervisor fold-condition count changed.")
    return result


def variant_map(fold: TaxonomyFold, condition: str) -> dict[str, str]:
    if condition not in fold.conditions:
        raise ValueError(f"Condition {condition!r} is not registered for {fold.fold_id}.")
    variants = {
        aspect: "name_and_description" for aspect in fold.evaluation_aspects
    }
    mapping = {"N": "name_only", "D": "name_and_description", "R": "rich"}
    if fold.level == "L2":
        for aspect in fold.heldout_aspects:
            variants[aspect] = mapping[condition]
    elif fold.level == "L4" and condition != "D":
        raise ValueError("Level 4 formal v2 permits only condition D.")
    return variants


def workload_contract_counts(
    training_scope_ids: Iterable[str] | None = None,
) -> dict[str, int]:
    """Count deduplicated prompt and training contracts in formal v2."""

    scopes = scope_folds()
    selected = tuple(scopes if training_scope_ids is None else training_scope_ids)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(scopes):
        raise ValueError("Workload scope subset is empty, duplicated or unknown.")
    full: set[tuple[str, str, str]] = set()
    selection: set[tuple[str, str, str]] = set()
    for scope_id in selected:
        folds = scopes[scope_id]
        fold = representative_fold(folds)
        for aspect in fold.seen_aspects:
            selection.add((scope_id, str(aspect), "name_and_description"))
        for current in folds:
            for condition in formal_conditions(current):
                for aspect, representation in variant_map(current, condition).items():
                    full.add((scope_id, str(aspect), representation))
    full |= selection
    return {
        "unique_training_scopes": len(selected),
        "full_unique_candidate_contracts": len(full),
        "seen_selection_candidate_contracts": len(selection),
        "selected_model_additional_candidate_contracts": len(full - selection),
    }


def runtime_cost_forecast(
    counts: Mapping[str, int],
    *,
    unique_validation_texts: int,
    frozen_prompts_per_second: float,
    qlora_prompts_per_second: float,
    qlora_training_examples_per_second: float,
    distilbert_prompts_per_second: float,
    distilbert_training_examples_per_second: float,
    price_per_hour_usd: float,
) -> dict[str, object]:
    """Apply the qualified-host rates to an exact v2 workload subset."""

    rates = {
        "frozen_prompts_per_second": frozen_prompts_per_second,
        "qlora_prompts_per_second": qlora_prompts_per_second,
        "qlora_training_examples_per_second": qlora_training_examples_per_second,
        "distilbert_prompts_per_second": distilbert_prompts_per_second,
        "distilbert_training_examples_per_second": distilbert_training_examples_per_second,
    }
    if unique_validation_texts < 1 or price_per_hour_usd <= 0:
        raise ValueError("Forecast population and price must be positive.")
    if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
        raise ValueError("Forecast rates must be finite and positive.")
    full = int(counts["full_unique_candidate_contracts"])
    selection = int(counts["seen_selection_candidate_contracts"])
    additional = int(counts["selected_model_additional_candidate_contracts"])
    scope_count = int(counts["unique_training_scopes"])
    frozen_prompts = full * unique_validation_texts * STAGES_PER_CANDIDATE
    tuning_prompts = (
        selection
        * unique_validation_texts
        * STAGES_PER_CANDIDATE
        * LEARNING_RATE_COUNT
    )
    selected_prompts = additional * unique_validation_texts * STAGES_PER_CANDIDATE
    trainable_prompts = tuning_prompts + selected_prompts
    qlora_training = scope_count * LEARNING_RATE_COUNT * TRAINING_EXAMPLES_PER_SCOPE
    distilbert_training = qlora_training * 3
    raw_seconds = {
        "frozen_few_shot": frozen_prompts / frozen_prompts_per_second,
        "qlora": qlora_training / qlora_training_examples_per_second
        + trainable_prompts / qlora_prompts_per_second,
        "distilbert": distilbert_training / distilbert_training_examples_per_second
        + trainable_prompts / distilbert_prompts_per_second,
    }
    methods = {
        method: {
            "raw_seconds": seconds,
            "planned_hours_with_25pct_margin": seconds * 1.25 / 3600.0,
            "planned_cost_usd": seconds * 1.25 / 3600.0 * price_per_hour_usd,
        }
        for method, seconds in raw_seconds.items()
    }
    return {
        "counts": dict(counts),
        "unique_validation_texts": unique_validation_texts,
        "workload": {
            "frozen_prompts": frozen_prompts,
            "trainable_tuning_prompts": tuning_prompts,
            "trainable_selected_model_additional_prompts": selected_prompts,
            "trainable_total_prompts_per_method": trainable_prompts,
            "qlora_training_examples": qlora_training,
            "distilbert_training_examples_across_three_epochs": distilbert_training,
        },
        "rates": rates,
        "overhead_multiplier": 1.25,
        "hourly_price_usd": price_per_hour_usd,
        "methods": methods,
        "total_planned_gpu_hours": sum(
            float(value["planned_hours_with_25pct_margin"])
            for value in methods.values()
        ),
        "total_planned_cost_usd": sum(
            float(value["planned_cost_usd"]) for value in methods.values()
        ),
    }


def candidate_result_path(
    output_root: Path,
    method_id: str,
    scope_id: str,
    learning_rate: float,
) -> Path:
    token = f"{learning_rate:.0e}".replace("+", "").replace("-0", "-")
    return output_root / "selection_candidates" / method_id / scope_id / f"lr-{token}.json"


def selection_path(output_root: Path, method_id: str, scope_id: str) -> Path:
    return output_root / "selections" / method_id / f"{scope_id}.json"


def build_formal_jobs(
    *,
    output_root: str = "/workspace/taxonomy_two_stage_formal_v2",
    data_dir: str = "/workspace/data/fabsa",
    methods: Iterable[str] | None = None,
    training_scope_ids: Iterable[str] | None = None,
) -> list[FormalJob]:
    selected_methods = tuple(TRAINABLE_METHODS if methods is None else methods)
    if not selected_methods or len(selected_methods) != len(set(selected_methods)):
        raise ValueError("Formal-v2 methods must be a non-empty unique sequence.")
    if set(selected_methods) - set(TRAINABLE_METHODS):
        raise ValueError("Formal-v2 method set contains an unsupported method.")
    scopes = scope_folds()
    selected_scopes = tuple(scopes if training_scope_ids is None else training_scope_ids)
    if not selected_scopes or len(selected_scopes) != len(set(selected_scopes)):
        raise ValueError("Formal-v2 scopes must be a non-empty unique sequence.")
    if set(selected_scopes) - set(scopes):
        raise ValueError("Formal-v2 scope set contains an unregistered scope.")
    script = "scripts/run_taxonomy_post_supervisor_trainable_validation.py"
    jobs: list[FormalJob] = []
    for method_id in selected_methods:
        for scope_id in selected_scopes:
            for rate in learning_rates(method_id):
                jobs.append(
                    FormalJob(
                        "train-candidate",
                        method_id,
                        scope_id,
                        rate,
                        (
                            "python", script, "--phase", "train-candidate",
                            "--method", method_id, "--scope-id", scope_id,
                            "--learning-rate", str(rate), "--output-root", output_root,
                            "--data-dir", data_dir, "--resume",
                        ),
                    )
                )
            jobs.append(
                FormalJob(
                    "select-scope", method_id, scope_id, None,
                    (
                        "python", script, "--phase", "select-scope", "--method",
                        method_id, "--scope-id", scope_id, "--output-root", output_root,
                        "--data-dir", data_dir,
                    ),
                )
            )
            jobs.append(
                FormalJob(
                    "score-selected", method_id, scope_id, None,
                    (
                        "python", script, "--phase", "score-selected", "--method",
                        method_id, "--scope-id", scope_id, "--output-root", output_root,
                        "--data-dir", data_dir, "--resume",
                    ),
                )
            )
    return jobs


def validate_formal_job_subset(jobs: Sequence[FormalJob]) -> dict[str, int]:
    if not jobs:
        raise ValueError("Formal-v2 worker job subset must not be empty.")
    forbidden = {"--include-official-test", "score-test", "analyse-test", "test.csv"}
    grouped: dict[tuple[str, str], list[FormalJob]] = {}
    identities: set[str] = set()
    for job in jobs:
        if any(token.lower() in forbidden or "test.csv" in token.lower() for token in job.command):
            raise ValueError("Official-test work entered formal-v2 jobs.")
        identity = formal_job_id(job)
        if identity in identities:
            raise ValueError(f"Duplicate formal-v2 job: {identity}.")
        identities.add(identity)
        grouped.setdefault((job.method_id, job.training_scope_id), []).append(job)
    for (method_id, scope_id), values in grouped.items():
        counts = {
            phase: sum(value.phase == phase for value in values)
            for phase in ("train-candidate", "select-scope", "score-selected")
        }
        if counts != {"train-candidate": 3, "select-scope": 1, "score-selected": 1}:
            raise ValueError(f"Incomplete formal-v2 unit for {method_id}/{scope_id}: {counts}.")
        rates = sorted(value.learning_rate for value in values if value.learning_rate is not None)
        if rates != sorted(LEARNING_RATES[method_id]):
            raise ValueError("Formal-v2 learning-rate set changed.")
    return {
        phase: sum(job.phase == phase for job in jobs)
        for phase in ("train-candidate", "select-scope", "score-selected")
    }


def validate_formal_job_graph(jobs: Sequence[FormalJob]) -> dict[str, int]:
    counts = validate_formal_job_subset(jobs)
    expected = {"train-candidate": 90, "select-scope": 30, "score-selected": 30}
    if counts != expected:
        raise AssertionError(f"Formal-v2 job counts changed: {counts}.")
    if len(jobs) != 150:
        raise AssertionError("Formal-v2 total job count changed.")
    return counts


def plan_payload(jobs: Sequence[FormalJob]) -> dict[str, object]:
    return {
        "schema_version": "taxonomy_post_supervisor_formal_job_plan_v2",
        "protocol_id": PROTOCOL_ID,
        "allowed_splits": ["train", "validation"],
        "include_official_test": False,
        "test_contract_count": 0,
        "unique_training_scopes": 15,
        "fold_condition_count_per_method": 27,
        "trainable_result_payload_count": 54,
        "all_method_result_payload_count": 81,
        "job_counts": validate_formal_job_graph(jobs),
        "jobs": [job.to_dict() for job in jobs],
    }


def load_parallel_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Formal-v2 parallel plan must be an object.")
    validate_parallel_plan(value)
    return value


def _workers(plan: Mapping[str, object]) -> list[Mapping[str, object]]:
    workers = plan.get("workers")
    if not isinstance(workers, list) or len(workers) != 3 or not all(isinstance(value, Mapping) for value in workers):
        raise ValueError("Formal-v2 plan must contain exactly three workers.")
    return list(workers)


def validate_parallel_plan(plan: Mapping[str, object]) -> dict[str, object]:
    if plan.get("schema_version") != PARALLEL_PLAN_SCHEMA or plan.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected formal-v2 parallel-plan identity.")
    if plan.get("status") != "preregistered_before_parallel_execution":
        raise ValueError("Formal-v2 plan is not preregistered.")
    if list(plan.get("allowed_splits", [])) != ["train", "validation"] or plan.get("include_official_test") is not False or plan.get("test_contract_count") != 0:
        raise ValueError("Formal-v2 data boundary is invalid.")
    scopes = set(scope_folds())
    assignments = {method: [] for method in TRAINABLE_METHODS}
    frozen: list[str] = []
    ids: list[str] = []
    for worker in _workers(plan):
        worker_id = str(worker.get("worker_id", ""))
        ids.append(worker_id)
        method = str(worker.get("method_id", ""))
        if method not in assignments:
            raise ValueError("Formal-v2 worker method is invalid.")
        values = worker.get("training_scope_ids")
        if not isinstance(values, list) or not values:
            raise ValueError("Formal-v2 worker scopes are invalid.")
        assigned = [str(value) for value in values]
        if len(assigned) != len(set(assigned)) or set(assigned) - scopes:
            raise ValueError("Formal-v2 worker has duplicate or unknown scopes.")
        assignments[method].extend(assigned)
        frozen_values = worker.get("frozen_qwen_few_shot_scope_ids", [])
        if not isinstance(frozen_values, list):
            raise ValueError("Formal-v2 frozen scopes are invalid.")
        frozen.extend(str(value) for value in frozen_values)
    if any(not value for value in ids) or len(set(ids)) != 3:
        raise ValueError("Formal-v2 worker IDs must be unique.")
    for method, assigned in assignments.items():
        if len(assigned) != len(set(assigned)) or set(assigned) != scopes:
            raise ValueError(f"Formal-v2 {method} scopes are not an exact partition.")
    if len(frozen) != len(set(frozen)) or set(frozen) != scopes:
        raise ValueError("Formal-v2 frozen scopes are not an exact partition.")
    return {
        "worker_count": 3,
        "unique_training_scopes": 15,
        "trainable_job_count": 150,
        "result_payload_count": 81,
        "test_contract_count": 0,
        "parallel_plan_sha256": canonical_sha256(plan),
    }


def worker_record(plan: Mapping[str, object], worker_id: str) -> Mapping[str, object]:
    validate_parallel_plan(plan)
    matches = [worker for worker in _workers(plan) if str(worker.get("worker_id")) == worker_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown formal-v2 worker: {worker_id!r}.")
    return matches[0]


def trainable_worker_jobs(
    plan: Mapping[str, object],
    worker_id: str,
    *,
    output_root: str,
    data_dir: str,
) -> list[FormalJob]:
    worker = worker_record(plan, worker_id)
    jobs = build_formal_jobs(
        output_root=output_root,
        data_dir=data_dir,
        methods=(str(worker["method_id"]),),
        training_scope_ids=tuple(str(value) for value in worker["training_scope_ids"]),
    )
    validate_formal_job_subset(jobs)
    return jobs


def build_worker_manifest(
    plan: Mapping[str, object],
    worker_id: str,
    *,
    output_root: str,
    data_dir: str,
) -> dict[str, object]:
    worker = worker_record(plan, worker_id)
    jobs = trainable_worker_jobs(
        plan, worker_id, output_root=output_root, data_dir=data_dir
    )
    payload: dict[str, object] = {
        "schema_version": WORKER_MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "parallel_plan_sha256": canonical_sha256(plan),
        "worker_id": worker_id,
        "role": str(worker["role"]),
        "method_id": str(worker["method_id"]),
        "training_scope_ids": [str(value) for value in worker["training_scope_ids"]],
        "frozen_qwen_few_shot_scope_ids": [str(value) for value in worker.get("frozen_qwen_few_shot_scope_ids", [])],
        "output_root": output_root,
        "data_dir": data_dir,
        "job_ids": [formal_job_id(job) for job in jobs],
        "job_count": len(jobs),
        "failure_count": 0,
        "test_contract_count": 0,
    }
    payload["worker_manifest_sha256"] = canonical_sha256(payload)
    return payload


def validate_worker_manifest(
    manifest: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    if manifest.get("schema_version") != WORKER_MANIFEST_SCHEMA:
        raise ValueError("Unexpected formal-v2 worker-manifest schema.")
    expected_hash = str(manifest.get("worker_manifest_sha256", ""))
    unhashed = dict(manifest)
    unhashed.pop("worker_manifest_sha256", None)
    if expected_hash != canonical_sha256(unhashed):
        raise ValueError("Formal-v2 worker-manifest content hash mismatch.")
    if manifest.get("parallel_plan_sha256") != canonical_sha256(plan):
        raise ValueError("Formal-v2 worker manifest belongs to another plan.")
    if manifest.get("failure_count") != 0 or manifest.get("test_contract_count") != 0:
        raise ValueError("Formal-v2 worker manifest contains failure/test evidence.")
    expected = build_worker_manifest(
        plan,
        str(manifest.get("worker_id", "")),
        output_root=str(manifest.get("output_root", "")),
        data_dir=str(manifest.get("data_dir", "")),
    )
    if dict(manifest) != expected:
        raise ValueError("Formal-v2 worker manifest differs from its assignment.")
    return expected


def expected_result_assignments(plan: Mapping[str, object]) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    validate_parallel_plan(plan)
    assignments: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    for worker in _workers(plan):
        worker_id = str(worker["worker_id"])
        method = str(worker["method_id"])
        for scope in worker["training_scope_ids"]:
            for fold in scope_folds()[str(scope)]:
                for condition in formal_conditions(fold):
                    key = (method, fold.level, fold.fold_id, condition)
                    if key in assignments:
                        raise ValueError("Duplicate formal-v2 trainable result assignment.")
                    assignments[key] = (worker_id, str(scope))
        for scope in worker.get("frozen_qwen_few_shot_scope_ids", []):
            for fold in scope_folds()[str(scope)]:
                for condition in formal_conditions(fold):
                    key = (FROZEN_METHOD, fold.level, fold.fold_id, condition)
                    if key in assignments:
                        raise ValueError("Duplicate formal-v2 frozen result assignment.")
                    assignments[key] = (worker_id, str(scope))
    if len(assignments) != 81:
        raise ValueError(f"Formal-v2 result assignment count changed: {len(assignments)}.")
    return assignments


def audit_worker_union(manifests: Sequence[Mapping[str, object]], plan: Mapping[str, object]) -> dict[str, object]:
    validate_parallel_plan(plan)
    expected_workers = {str(worker["worker_id"]) for worker in _workers(plan)}
    observed_workers = {str(manifest.get("worker_id")) for manifest in manifests}
    if len(manifests) != 3 or observed_workers != expected_workers:
        raise ValueError("Formal-v2 worker manifest set is incomplete.")
    job_ids: list[str] = []
    for manifest in manifests:
        validate_worker_manifest(manifest, plan)
        worker_id = str(manifest.get("worker_id"))
        expected = build_worker_manifest(
            plan,
            worker_id,
            output_root=str(manifest.get("output_root")),
            data_dir=str(manifest.get("data_dir")),
        )
        if dict(manifest) != expected:
            raise ValueError("Formal-v2 worker manifest content mismatch.")
        job_ids.extend(str(value) for value in manifest["job_ids"])  # type: ignore[index]
    expected_jobs = {formal_job_id(job) for job in build_formal_jobs(output_root="<root>", data_dir="<data>")}
    if len(job_ids) != len(set(job_ids)) or set(job_ids) != expected_jobs:
        raise ValueError("Formal-v2 worker job union is incomplete or duplicated.")
    return {
        "schema_version": "taxonomy_post_supervisor_worker_union_audit_v2",
        "status": "pass",
        "worker_count": 3,
        "trainable_job_count": 150,
        "result_payload_count": len(expected_result_assignments(plan)),
        "missing_job_count": 0,
        "duplicate_job_count": 0,
        "failure_count": 0,
        "test_contract_count": 0,
    }
