"""Fail-closed contract utilities for the final revised-protocol test.

The module deliberately separates a scientific preregistration from a later,
one-time release record.  Loading or validating the preregistration never
requires a test path.  Any operation that may touch the official test must
first validate a release record whose hash is bound to the preregistration,
the execution commit, and the test bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

PROTOCOL_ID = "taxonomy_two_stage_final_test_v1"
PREREGISTRATION_STATUS = "frozen_pre_release"
RELEASE_STATUS = "authorised_once"

L2_FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))
L2_SCOPES = tuple(f"heldout-a{index:02d}" for index in range(1, 13))
L4_FOLDS = ("l4-g01", "l4-g02", "l4-g03")
L4_SCOPES = (
    "heldout-a02-a03-a04",
    "heldout-a08-a09-a10",
    "heldout-a11-a12",
)

L2_BASE_METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "description_to_classifier_weight_transfer",
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_candidate_pair",
    "frozen_qwen_few_shot",
    "qwen_candidate_pair_qlora",
)
FIXED_COMPOSITION_METHOD = "frozen_qwen_few_shot_stage1__qlora_stage2"
L4_METHODS = (
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_few_shot",
    "qwen_candidate_pair_qlora",
)
CONFIRMATORY_SYSTEMS = (
    FIXED_COMPOSITION_METHOD,
    "frozen_qwen_few_shot",
    "qwen_candidate_pair_qlora",
)
RELEASE_BINDINGS = (
    "preregistration_sha256",
    "execution_commit",
    "official_test_file_sha256",
    "official_test_row_uid_sha256",
    "official_test_rows",
    "official_test_relative_path",
    "backup_root_id",
    "batch_fallback_policy",
    "explicit_user_authorisation",
    "authorisation_id",
    "authorised_at",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preregistration_sha256(config: Mapping[str, object]) -> str:
    """Hash the complete immutable preregistration object."""

    return canonical_sha256(dict(config))


def _require_exact_sequence(
    value: object,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, list) or tuple(str(item) for item in value) != tuple(
        expected
    ):
        raise ValueError(f"{label} must exactly equal {list(expected)!r}.")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return value


def validate_preregistration(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the frozen, pre-release scientific and operational contract."""

    if config.get("schema_version") != "taxonomy_final_test_preregistration_v1":
        raise ValueError("Unexpected final-test preregistration schema.")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected final-test protocol_id.")
    if config.get("status") != PREREGISTRATION_STATUS:
        raise ValueError("Final-test preregistration is not frozen pre-release.")
    if config.get("include_official_test") is not False:
        raise ValueError("Pre-release preregistration must keep official test disabled.")
    if int(config.get("test_contract_count", -1)) != 0:
        raise ValueError("Pre-release preregistration must have zero test contracts.")
    _require_exact_sequence(
        config.get("allowed_splits_pre_release"),
        ("train", "validation"),
        label="allowed_splits_pre_release",
    )

    levels = _mapping(config.get("levels"), label="levels")
    l2 = _mapping(levels.get("L2"), label="levels.L2")
    l4 = _mapping(levels.get("L4"), label="levels.L4")
    _require_exact_sequence(l2.get("folds"), L2_FOLDS, label="levels.L2.folds")
    _require_exact_sequence(
        l2.get("training_scopes"), L2_SCOPES, label="levels.L2.training_scopes"
    )
    _require_exact_sequence(l2.get("conditions"), ("D", "N"), label="L2 conditions")
    if l2.get("primary_condition") != "D" or l2.get("n_role") != "descriptive":
        raise ValueError("L2 must freeze D as primary and N as descriptive.")
    _require_exact_sequence(l4.get("folds"), L4_FOLDS, label="levels.L4.folds")
    _require_exact_sequence(
        l4.get("training_scopes"), L4_SCOPES, label="levels.L4.training_scopes"
    )
    _require_exact_sequence(l4.get("conditions"), ("D",), label="L4 conditions")
    if l4.get("role") != "descriptive_structured_shift":
        raise ValueError("L4 must remain a descriptive structured-shift stress test.")

    roster = _mapping(config.get("roster"), label="roster")
    _require_exact_sequence(
        roster.get("confirmatory_L2_D"),
        CONFIRMATORY_SYSTEMS,
        label="confirmatory L2-D roster",
    )
    _require_exact_sequence(
        roster.get("descriptive_L2_base_methods"),
        L2_BASE_METHODS,
        label="descriptive L2 base roster",
    )
    _require_exact_sequence(
        roster.get("descriptive_L4_D"), L4_METHODS, label="descriptive L4-D roster"
    )
    excluded = roster.get("excluded")
    if not isinstance(excluded, list) or not excluded:
        raise ValueError("The final-test exclusion roster must be non-empty.")
    forbidden_tokens = {
        "global_router",
        "relative_calibrator",
        "smooth_stage1_fusion",
        "retrieval_few_shot",
        "reverse_composition",
        "rich_description_R",
        "Level_3",
    }
    if not forbidden_tokens.issubset({str(item) for item in excluded}):
        raise ValueError("The exclusion roster is incomplete.")

    composition = _mapping(config.get("fixed_composition"), label="fixed_composition")
    expected_composition = {
        "method_id": FIXED_COMPOSITION_METHOD,
        "stage_1": "frozen_qwen_few_shot",
        "stage_2": "qwen_candidate_pair_qlora",
        "retuning": False,
        "retraining": False,
        "maximum_sentiments_per_selected_aspect": 2,
    }
    for key, expected in expected_composition.items():
        if composition.get(key) != expected:
            raise ValueError(f"fixed_composition.{key} must equal {expected!r}.")

    hypotheses = _mapping(config.get("hypotheses"), label="hypotheses")
    if hypotheses.get("multiplicity") != "hierarchical_gatekeeping_H1_then_H2":
        raise ValueError("Confirmatory multiplicity must use hierarchical gatekeeping.")
    h1 = _mapping(hypotheses.get("H1"), label="hypotheses.H1")
    h2 = _mapping(hypotheses.get("H2"), label="hypotheses.H2")
    if (
        h1.get("candidate") != FIXED_COMPOSITION_METHOD
        or h1.get("comparator") != "frozen_qwen_few_shot"
        or h2.get("candidate") != FIXED_COMPOSITION_METHOD
        or h2.get("comparator") != "qwen_candidate_pair_qlora"
    ):
        raise ValueError("The two confirmatory comparisons are not the frozen H1/H2 family.")

    statistics = _mapping(config.get("statistics"), label="statistics")
    if (
        statistics.get("primary_endpoint")
        != "L2_D_aspect_balanced_mean_heldout_pair_micro_f1"
        or int(statistics.get("bootstrap_draws", -1)) != 20000
        or int(statistics.get("bootstrap_seed", -1)) != 13
        or statistics.get("bootstrap_unit") != "synchronised_row_uid_review_cluster"
    ):
        raise ValueError("Primary endpoint or paired-bootstrap contract changed.")

    governance = _mapping(config.get("governance"), label="governance")
    required_false = (
        "train_plus_validation_retraining",
        "interim_outcome_inspection",
        "test_informed_retuning",
        "post_test_candidate_promotion",
    )
    if any(governance.get(key) is not False for key in required_false):
        raise ValueError("Final-test governance contains a prohibited permission.")
    if governance.get("score_before_label_reveal") is not True:
        raise ValueError("Scores must be sealed before labels are revealed to analysis.")
    if governance.get("canonical_seen_scores_shared_across_N_D") is not True:
        raise ValueError("N/D unchanged seen scores must be shared by construction.")

    reuse = _mapping(config.get("selection_and_reuse"), label="selection_and_reuse")
    if (
        reuse.get("train_plus_validation_retraining") is not False
        or reuse.get("new_parameter_selection_on_test") is not False
        or reuse.get("checkpoint_source")
        != "formal_v2_train_only_selected_checkpoints"
        or reuse.get("few_shot_demonstrations")
        != "formal_v2_deterministic_train_only_contract"
        or reuse.get("threshold_source")
        != "formal_v2_seen_validation_only selections"
    ):
        raise ValueError("The no-retraining / frozen-selection contract changed.")

    failure = _mapping(config.get("failure_policy"), label="failure_policy")
    if (
        failure.get("batch_size_change")
        != "forbidden_after_release_no_fallback"
        or failure.get("low_performance_or_failed_hypothesis")
        != "scientific_outcome_never_rerun"
        or failure.get("resume_conflict") != "abort_without_overwrite"
    ):
        raise ValueError("The fail-closed final-test failure policy changed.")

    release = _mapping(config.get("release"), label="release")
    if (
        release.get("required") is not True
        or release.get("currently_authorised") is not False
        or release.get("release_manifest_schema")
        != "taxonomy_final_test_release_v1"
    ):
        raise ValueError("Pre-release state must require a separate one-time release record.")
    _require_exact_sequence(
        release.get("must_bind"), RELEASE_BINDINGS, label="release.must_bind"
    )

    return {
        "status": "pass",
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": preregistration_sha256(config),
        "l2_fold_count": len(L2_FOLDS),
        "l4_group_count": len(L4_FOLDS),
        "confirmatory_system_count": len(CONFIRMATORY_SYSTEMS),
        "descriptive_l2_base_method_count": len(L2_BASE_METHODS),
        "test_contract_count": 0,
    }


@dataclass(frozen=True)
class FinalTestJob:
    job_id: str
    phase: str
    role: str
    method_id: str
    level: str
    fold_id: str
    training_scope_id: str
    conditions: tuple[str, ...]
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["conditions"] = list(self.conditions)
        value["depends_on"] = list(self.depends_on)
        return value


def build_final_test_jobs(
    config: Mapping[str, object],
) -> tuple[FinalTestJob, ...]:
    """Build the exact frozen job graph without touching any dataset."""

    validate_preregistration(config)
    jobs: list[FinalTestJob] = []
    for fold_id, scope_id in zip(L2_FOLDS, L2_SCOPES):
        source_ids: dict[str, str] = {}
        for method_id in L2_BASE_METHODS:
            job_id = f"score-L2-{fold_id}-{method_id}"
            source_ids[method_id] = job_id
            jobs.append(
                FinalTestJob(
                    job_id=job_id,
                    phase="score",
                    role="descriptive_base_score",
                    method_id=method_id,
                    level="L2",
                    fold_id=fold_id,
                    training_scope_id=scope_id,
                    conditions=("D", "N"),
                )
            )
        jobs.append(
            FinalTestJob(
                job_id=f"compose-L2-{fold_id}-{FIXED_COMPOSITION_METHOD}",
                phase="compose",
                role="confirmatory_fixed_composition",
                method_id=FIXED_COMPOSITION_METHOD,
                level="L2",
                fold_id=fold_id,
                training_scope_id=scope_id,
                conditions=("D", "N"),
                depends_on=(
                    source_ids["frozen_qwen_few_shot"],
                    source_ids["qwen_candidate_pair_qlora"],
                ),
            )
        )
    for fold_id, scope_id in zip(L4_FOLDS, L4_SCOPES):
        for method_id in L4_METHODS:
            jobs.append(
                FinalTestJob(
                    job_id=f"score-L4-{fold_id}-{method_id}",
                    phase="score",
                    role="descriptive_structured_shift",
                    method_id=method_id,
                    level="L4",
                    fold_id=fold_id,
                    training_scope_id=scope_id,
                    conditions=("D",),
                )
            )
    dependencies = tuple(job.job_id for job in jobs)
    jobs.append(
        FinalTestJob(
            job_id="seal-all-score-grids",
            phase="seal",
            role="integrity_and_backup_barrier",
            method_id="all_registered_systems",
            level="L2_L4",
            fold_id="all",
            training_scope_id="all",
            conditions=("D", "N"),
            depends_on=dependencies,
        )
    )
    jobs.append(
        FinalTestJob(
            job_id="analyse-once-after-seal",
            phase="analyse",
            role="single_outcome_reveal",
            method_id="all_registered_systems",
            level="L2_L4",
            fold_id="all",
            training_scope_id="all",
            conditions=("D", "N"),
            depends_on=("seal-all-score-grids",),
        )
    )
    validate_job_graph(jobs)
    return tuple(jobs)


def validate_job_graph(jobs: Sequence[FinalTestJob]) -> dict[str, int]:
    ids = [job.job_id for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Final-test job IDs are not unique.")
    known = set(ids)
    for job in jobs:
        missing = set(job.depends_on) - known
        if missing:
            raise ValueError(f"Job {job.job_id} has missing dependencies: {sorted(missing)}")
        if job.level == "L2" and set(job.conditions) != {"D", "N"}:
            raise ValueError("Every L2 score/composition job must produce D and N together.")
        if job.level == "L4" and job.conditions != ("D",):
            raise ValueError("Every L4 job must be D-only.")
    counts = {
        phase: sum(job.phase == phase for job in jobs)
        for phase in ("score", "compose", "seal", "analyse")
    }
    expected = {"score": 93, "compose": 12, "seal": 1, "analyse": 1}
    if counts != expected:
        raise ValueError(f"Unexpected final-test job graph: {counts!r}.")
    return counts


def job_plan(config: Mapping[str, object]) -> dict[str, object]:
    jobs = build_final_test_jobs(config)
    counts = validate_job_graph(jobs)
    return {
        "schema_version": "taxonomy_final_test_job_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "pre_release_no_test_access",
        "preregistration_sha256": preregistration_sha256(config),
        "job_counts": counts,
        "jobs": [job.to_dict() for job in jobs],
        "include_official_test": False,
        "test_contract_count": 0,
    }


def load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def validate_release_manifest(
    release: Mapping[str, object],
    config: Mapping[str, object],
    *,
    execution_commit: str,
    test_path: Path | None = None,
) -> dict[str, object]:
    """Validate the one-time release record.

    Supplying ``test_path`` reads only the bytes required for a SHA-256 check.
    Callers must not invoke this function with a test path before the user has
    granted the explicit one-time release.
    """

    validate_preregistration(config)
    if release.get("schema_version") != "taxonomy_final_test_release_v1":
        raise ValueError("Unexpected final-test release schema.")
    if release.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Release protocol_id mismatch.")
    if release.get("status") != RELEASE_STATUS:
        raise PermissionError("The official test has not received a one-time release.")
    if release.get("include_official_test") is not True:
        raise PermissionError("Release record does not enable the official test.")
    if release.get("explicit_user_authorisation") is not True:
        raise PermissionError("Explicit user authorisation is absent.")
    if release.get("preregistration_sha256") != preregistration_sha256(config):
        raise ValueError("Release record is bound to another preregistration.")
    if release.get("execution_commit") != execution_commit:
        raise ValueError("Release record is bound to another execution commit.")
    if not isinstance(release.get("authorised_at"), str) or not release.get(
        "authorised_at"
    ):
        raise ValueError("Release record lacks an authorisation timestamp.")
    if not isinstance(release.get("authorisation_id"), str) or not release.get(
        "authorisation_id"
    ):
        raise ValueError("Release record lacks a unique authorisation ID.")
    expected_hash = release.get("official_test_file_sha256")
    expected_row_uid_hash = release.get("official_test_row_uid_sha256")
    expected_rows = release.get("official_test_rows")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Release record lacks a valid official-test SHA-256.")
    if int(expected_rows or -1) != 1587:
        raise ValueError("Release record must bind the registered 1,587-row test split.")
    if not isinstance(expected_row_uid_hash, str) or len(expected_row_uid_hash) != 64:
        raise ValueError("Release record lacks a valid official-test row-UID SHA-256.")
    if release.get("official_test_relative_path") != "test.csv":
        raise ValueError("Release record must bind the canonical test.csv filename.")
    backup_root_id = release.get("backup_root_id")
    if not isinstance(backup_root_id, str) or not backup_root_id:
        raise ValueError("Release record lacks the immutable backup-root identifier.")
    if release.get("batch_fallback_policy") != "none_no_post_release_batch_change":
        raise ValueError("Release record must prohibit post-release batch-size changes.")
    if test_path is not None:
        if not test_path.is_file():
            raise FileNotFoundError(test_path)
        observed_hash = file_sha256(test_path)
        if observed_hash != expected_hash:
            raise ValueError("Official-test file hash does not match the release record.")
    return {
        "status": "pass",
        "protocol_id": PROTOCOL_ID,
        "authorisation_id": str(release["authorisation_id"]),
        "execution_commit": execution_commit,
        "official_test_file_sha256": expected_hash,
        "official_test_row_uid_sha256": expected_row_uid_hash,
        "official_test_rows": 1587,
        "backup_root_id": backup_root_id,
        "include_official_test": True,
        "test_contract_count": 1,
    }


def assert_pre_release_cannot_execute(
    config: Mapping[str, object],
    release: Mapping[str, object] | None,
) -> None:
    """Make accidental execution impossible in the frozen pre-release state."""

    validate_preregistration(config)
    if release is None:
        raise PermissionError(
            "Official-test execution is locked: no one-time release manifest was supplied."
        )
    if release.get("status") != RELEASE_STATUS:
        raise PermissionError("Official-test execution is locked: release is not authorised.")


def validate_artifact_inventory(
    inventory: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    if inventory.get("schema_version") != "taxonomy_final_test_artifact_inventory_v1":
        raise ValueError("Unexpected final-test artifact inventory schema.")
    if inventory.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Artifact inventory protocol mismatch.")
    if inventory.get("preregistration_sha256") != preregistration_sha256(config):
        raise ValueError("Artifact inventory is bound to another preregistration.")
    if inventory.get("status") != "pass":
        raise ValueError("Artifact inventory has not passed.")
    expected = {
        "distilbert_selected_scopes": 15,
        "qlora_selected_scopes": 15,
        "few_shot_selected_scopes": 15,
        "local_L2_folds_per_method": 12,
        "local_L2_method_count": 4,
        "failure_count": 0,
        "test_contract_count": 0,
    }
    for key, value in expected.items():
        if int(inventory.get(key, -1)) != value:
            raise ValueError(f"Artifact inventory {key} must equal {value}.")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Artifact inventory contains no entries.")
    hashes = [entry.get("sha256") for entry in entries if isinstance(entry, Mapping)]
    if len(hashes) != len(entries) or any(
        not isinstance(value, str) or len(value) != 64 for value in hashes
    ):
        raise ValueError("Artifact inventory contains an invalid SHA-256 entry.")
    return {
        "status": "pass",
        "entries": len(entries),
        **expected,
    }


def ordered_job_ids(jobs: Iterable[FinalTestJob]) -> tuple[str, ...]:
    """Return a deterministic topological ordering for the fixed shallow DAG."""

    pending = {job.job_id: job for job in jobs}
    emitted: list[str] = []
    while pending:
        ready = sorted(
            job_id
            for job_id, job in pending.items()
            if set(job.depends_on).issubset(emitted)
        )
        if not ready:
            raise ValueError("Final-test job graph contains a dependency cycle.")
        for job_id in ready:
            emitted.append(job_id)
            del pending[job_id]
    return tuple(emitted)
