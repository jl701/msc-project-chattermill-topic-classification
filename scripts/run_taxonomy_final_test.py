"""Manifest-driven, fail-closed runner for the revised-protocol final test.

Pre-release commands (plan, preflight, dry-run, status) cannot open the
official test.  All commands that can touch it require a separately authored,
one-time release record bound to a clean execution commit and explicit user
authorisation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.experiments.taxonomy_final_test import (
    FinalTestJob,
    assert_pre_release_cannot_execute,
    build_final_test_jobs,
    canonical_sha256,
    file_sha256,
    job_plan,
    load_json_object,
    preregistration_sha256,
    validate_artifact_inventory,
    validate_preregistration,
    validate_release_manifest,
)
from msc_project.experiments.taxonomy_final_test_analysis import (
    analyse_sealed_graph_once,
    build_label_vault,
)
from msc_project.experiments.taxonomy_final_test_artifacts import (
    ArtifactRoots,
    resolve_job_artifacts,
    source_hash_union,
    verify_inventory_entries,
)
from msc_project.experiments.taxonomy_final_test_scoring import (
    final_fold,
    score_base_job,
)
from msc_project.experiments.taxonomy_final_test_workflow import (
    atomic_csv,
    atomic_json,
    build_unlabelled_grids,
    canonical_l2_nd_scores,
    compose_fixed_stage_scores,
    copy_verified_file,
    read_score_bundle,
    score_bundle_paths,
    score_frame_from_grids,
    seal_score_graph,
    write_score_bundle,
)
from msc_project.experiments.taxonomy_resources import (
    load_description_bundle,
)

DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT / "configs" / "experiments" / "taxonomy_final_test_v1.json"
)
DEFAULT_RELEASE_TEMPLATE = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_final_test_release_TEMPLATE.json"
)
DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "taxonomy_final_test_v1_artifact_inventory.json"
)
DEFAULT_LOCAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "experimental"
    / "taxonomy_post_supervisor_local_v1"
)
DEFAULT_CLOUD_ROOT = (
    PROJECT_ROOT.parent / "cloud_backups" / "taxonomy_two_stage_formal_v2_r2"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "experimental" / "taxonomy_two_stage_final_test_v1"
)
DEFAULT_DRY_RUN_ROOT = (
    PROJECT_ROOT / "outputs" / "experimental" / "taxonomy_final_test_v1_dry_run"
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _execution_commit() -> str:
    return _git("rev-parse", "HEAD")


def _assert_clean_and_synced() -> str:
    if _git("status", "--porcelain"):
        raise PermissionError("Release creation requires a clean Git worktree.")
    commit = _execution_commit()
    upstream = _git("rev-parse", "@{upstream}")
    if upstream != commit:
        raise PermissionError("Release creation requires HEAD to equal its upstream.")
    return commit


def _roots(args: argparse.Namespace) -> ArtifactRoots:
    cloud = args.cloud_root.resolve()
    return ArtifactRoots(
        project_root=PROJECT_ROOT,
        local_validation_root=args.local_root.resolve(),
        cloud_worker_distil_frozen=cloud / "worker-distil-frozen",
        cloud_worker_qlora_a=cloud / "worker-qlora-a",
        cloud_worker_qlora_b=cloud / "worker-qlora-b",
    )


def _load_contracts(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    config = load_json_object(args.preregistration)
    inventory = load_json_object(args.inventory)
    validate_preregistration(config)
    validate_artifact_inventory(inventory, config)
    return config, inventory


def _job_by_id(config: Mapping[str, object], job_id: str) -> FinalTestJob:
    matches = [job for job in build_final_test_jobs(config) if job.job_id == job_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown final-test job ID: {job_id!r}.")
    return matches[0]


def _validate_release(args: argparse.Namespace, config: Mapping[str, object]) -> dict[str, object]:
    if args.release is None:
        assert_pre_release_cannot_execute(config, None)
    release = load_json_object(args.release)
    return validate_release_manifest(
        release,
        config,
        execution_commit=_execution_commit(),
    )


def _materialised_paths(output_root: Path) -> dict[str, Path]:
    root = output_root / "private_release_data"
    return {
        "unlabelled": root / "unlabelled_reviews.csv",
        "vault": root / "label_vault.csv",
        "manifest": root / "data_manifest.json",
        "release": root / "authorised_release.json",
    }


def _load_materialised(
    output_root: Path,
    release: Mapping[str, object],
    *,
    require_label_vault: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    paths = _materialised_paths(output_root)
    manifest = load_json_object(paths["manifest"])
    expected = {
        "schema_version": "taxonomy_final_test_materialised_data_v1",
        "protocol_id": "taxonomy_two_stage_final_test_v1",
        "authorisation_id": release["authorisation_id"],
        "unlabelled_reviews_sha256": file_sha256(paths["unlabelled"]),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Materialised final-test data {key} mismatch.")
    label_vault_sha256 = manifest.get("label_vault_sha256")
    if not isinstance(label_vault_sha256, str) or (
        len(label_vault_sha256) != 64
        or any(character not in "0123456789abcdef" for character in label_vault_sha256)
    ):
        raise ValueError("Materialised final-test label-vault hash is invalid.")
    if require_label_vault:
        if not paths["vault"].is_file():
            raise FileNotFoundError("Private label vault is unavailable for final analysis.")
        if file_sha256(paths["vault"]) != label_vault_sha256:
            raise ValueError("Materialised final-test label-vault hash mismatch.")
    reviews = pd.read_csv(paths["unlabelled"])
    if tuple(reviews.columns) != ("row_uid", "text"):
        raise ValueError("Materialised worker file is not strictly unlabelled.")
    uids = reviews["row_uid"].astype(str).tolist()
    if canonical_sha256(uids) != release["official_test_row_uid_sha256"]:
        raise ValueError("Materialised row-UID order differs from the release.")
    return reviews, uids, manifest


def command_plan(args: argparse.Namespace) -> dict[str, object]:
    config, inventory = _load_contracts(args)
    value = job_plan(config)
    value["artifact_inventory_entries"] = len(inventory["entries"])
    value["artifact_inventory_manifest_sha256"] = inventory[
        "entry_manifest_sha256"
    ]
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        atomic_json(args.output, value)
    return value


def command_preflight(args: argparse.Namespace) -> dict[str, object]:
    config, inventory = _load_contracts(args)
    template = load_json_object(args.release_template)
    try:
        assert_pre_release_cannot_execute(config, template)
    except PermissionError:
        template_locked = True
    else:
        template_locked = False
    if not template_locked:
        raise AssertionError("The inert release template unexpectedly enables execution.")
    artifact_audit = verify_inventory_entries(inventory, _roots(args))
    result: dict[str, object] = {
        "schema_version": "taxonomy_final_test_preflight_v1",
        "status": "pass",
        "protocol_id": config["protocol_id"],
        "preregistration_sha256": preregistration_sha256(config),
        "job_counts": job_plan(config)["job_counts"],
        "artifact_audit": artifact_audit,
        "release_template_locked": True,
        "official_test_accessed": False,
        "include_official_test": False,
        "test_contract_count": 0,
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        atomic_json(args.output, result)
    return result


def command_create_release(args: argparse.Namespace) -> dict[str, object]:
    if not args.i_have_explicit_user_authorisation:
        raise PermissionError("Release creation requires explicit one-time user authorisation.")
    if not args.authorisation_id or not args.authorised_at:
        raise ValueError("Release creation requires an authorisation ID and timestamp.")
    config, _ = _load_contracts(args)
    commit = _assert_clean_and_synced()
    if args.release is None or args.release.exists():
        raise FileExistsError("Authorised release output must be a new file.")
    test_path = args.data_dir.resolve() / "test.csv"
    # This is the first operation in the program that may read official-test bytes.
    test = load_official_fabsa_splits(args.data_dir.resolve(), ("test",))
    uids = test["row_uid"].astype(str).tolist()
    release: dict[str, object] = {
        "schema_version": "taxonomy_final_test_release_v1",
        "protocol_id": config["protocol_id"],
        "status": "authorised_once",
        "include_official_test": True,
        "explicit_user_authorisation": True,
        "preregistration_sha256": preregistration_sha256(config),
        "execution_commit": commit,
        "authorisation_id": args.authorisation_id,
        "authorised_at": args.authorised_at,
        "official_test_relative_path": "test.csv",
        "official_test_file_sha256": file_sha256(test_path),
        "official_test_row_uid_sha256": canonical_sha256(uids),
        "official_test_rows": len(test),
        "backup_root_id": args.backup_root_id,
        "batch_fallback_policy": "none_no_post_release_batch_change",
    }
    validate_release_manifest(
        release,
        config,
        execution_commit=commit,
        test_path=test_path,
    )
    atomic_json(args.release, release)
    return {
        "status": "authorised_once",
        "release": str(args.release),
        "execution_commit": commit,
        "official_test_rows": len(test),
        "outcomes_inspected": False,
    }


def command_materialise(args: argparse.Namespace) -> dict[str, object]:
    config, _ = _load_contracts(args)
    release = load_json_object(args.release) if args.release else None
    assert_pre_release_cannot_execute(config, release)
    if release is None:
        raise PermissionError("Missing one-time release.")
    test_path = args.data_dir.resolve() / "test.csv"
    validate_release_manifest(
        release,
        config,
        execution_commit=_execution_commit(),
        test_path=test_path,
    )
    paths = _materialised_paths(args.output_root.resolve())
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Final-test data materialisation is immutable and already exists.")
    test = load_official_fabsa_splits(args.data_dir.resolve(), ("test",))
    uids = test["row_uid"].astype(str).tolist()
    if (
        len(test) != int(release["official_test_rows"])
        or canonical_sha256(uids) != release["official_test_row_uid_sha256"]
    ):
        raise ValueError("Official-test rows differ from the authorised release.")
    unlabelled = test[["row_uid", "text"]].copy()
    labels = build_label_vault(
        test, load_description_bundle(require_approved=True)["canonical_order"]
    )
    atomic_csv(paths["unlabelled"], unlabelled)
    atomic_csv(paths["vault"], labels)
    atomic_json(paths["release"], release)
    manifest: dict[str, object] = {
        "schema_version": "taxonomy_final_test_materialised_data_v1",
        "protocol_id": config["protocol_id"],
        "authorisation_id": release["authorisation_id"],
        "official_test_rows": len(test),
        "row_uid_sha256": canonical_sha256(uids),
        "unlabelled_reviews_sha256": file_sha256(paths["unlabelled"]),
        "label_vault_sha256": file_sha256(paths["vault"]),
        "worker_label_columns": [],
        "outcomes_revealed": False,
        "failure_count": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_json(paths["manifest"], manifest)
    for path in paths.values():
        copy_verified_file(
            path,
            args.backup_root.resolve() / path.relative_to(args.output_root.resolve()),
        )
    return {
        "status": "materialised_and_hidden",
        "review_rows": len(test),
        "worker_columns": ["row_uid", "text"],
        "outcomes_revealed": False,
    }


def command_score_job(args: argparse.Namespace) -> dict[str, object]:
    config, inventory = _load_contracts(args)
    release = load_json_object(args.release) if args.release else None
    assert_pre_release_cannot_execute(config, release)
    if release is None:
        raise PermissionError("Missing one-time release.")
    validate_release_manifest(release, config, execution_commit=_execution_commit())
    job = _job_by_id(config, args.job_id)
    if job.phase != "score":
        raise ValueError("score-job accepts only a registered base score job.")
    reviews, uids, _ = _load_materialised(args.output_root.resolve(), release)
    # Explicitly load train only; workers never open the test split or label vault.
    train = load_official_fabsa_splits(args.data_dir.resolve(), ("train",))
    resource = load_description_bundle(require_approved=True)
    artifacts = resolve_job_artifacts(job, inventory, _roots(args))
    scores = score_base_job(
        job,
        artifacts,
        train,
        reviews,
        resource,
        local_files_only=args.local_files_only,
    )
    manifest = write_score_bundle(
        args.output_root.resolve(),
        job,
        scores,
        preregistration_sha256=preregistration_sha256(config),
        execution_commit=_execution_commit(),
        authorisation_id=str(release["authorisation_id"]),
        aspect_threshold=artifacts.aspect_threshold,
        runner_up_threshold=artifacts.runner_up_threshold,
        source_artifact_sha256s=artifacts.source_artifact_sha256s,
        expected_row_uids=uids,
        test_contract_count=1,
    )
    return {
        "status": "score_bundle_complete",
        "job_id": job.job_id,
        "score_file_sha256": manifest["score_file_sha256"],
        "outcomes_revealed": False,
    }


def command_compose_job(args: argparse.Namespace) -> dict[str, object]:
    config, inventory = _load_contracts(args)
    release = load_json_object(args.release) if args.release else None
    assert_pre_release_cannot_execute(config, release)
    if release is None:
        raise PermissionError("Missing one-time release.")
    validate_release_manifest(release, config, execution_commit=_execution_commit())
    job = _job_by_id(config, args.job_id)
    if job.phase != "compose" or len(job.depends_on) != 2:
        raise ValueError("compose-job accepts only a registered fixed-composition job.")
    _, uids, _ = _load_materialised(args.output_root.resolve(), release)
    source_jobs = [_job_by_id(config, value) for value in job.depends_on]
    loaded = [
        read_score_bundle(
            args.output_root.resolve(),
            source,
            preregistration_sha256=preregistration_sha256(config),
            execution_commit=_execution_commit(),
            authorisation_id=str(release["authorisation_id"]),
            expected_row_uids=uids,
        )
        for source in source_jobs
    ]
    by_method = {
        source.method_id: value for source, value in zip(source_jobs, loaded)
    }
    few_frame, few_manifest = by_method["frozen_qwen_few_shot"]
    qlora_frame, qlora_manifest = by_method["qwen_candidate_pair_qlora"]
    composed = compose_fixed_stage_scores(job, few_frame, qlora_frame)
    source_artifacts = [
        resolve_job_artifacts(source, inventory, _roots(args))
        for source in source_jobs
    ]
    manifest = write_score_bundle(
        args.output_root.resolve(),
        job,
        composed,
        preregistration_sha256=preregistration_sha256(config),
        execution_commit=_execution_commit(),
        authorisation_id=str(release["authorisation_id"]),
        aspect_threshold=float(few_manifest["aspect_threshold"]),
        runner_up_threshold=float(qlora_manifest["runner_up_threshold"]),
        source_artifact_sha256s=source_hash_union(source_artifacts),
        expected_row_uids=uids,
        test_contract_count=1,
    )
    return {
        "status": "composition_bundle_complete",
        "job_id": job.job_id,
        "score_file_sha256": manifest["score_file_sha256"],
        "outcomes_revealed": False,
    }


def command_seal(args: argparse.Namespace) -> dict[str, object]:
    config, _ = _load_contracts(args)
    release = load_json_object(args.release) if args.release else None
    assert_pre_release_cannot_execute(config, release)
    if release is None:
        raise PermissionError("Missing one-time release.")
    validate_release_manifest(release, config, execution_commit=_execution_commit())
    _, uids, _ = _load_materialised(args.output_root.resolve(), release)
    seal = seal_score_graph(
        args.output_root.resolve(),
        args.backup_root.resolve(),
        build_final_test_jobs(config),
        preregistration_sha256=preregistration_sha256(config),
        execution_commit=_execution_commit(),
        authorisation_id=str(release["authorisation_id"]),
        expected_row_uids=uids,
        test_contract_count=1,
    )
    return {
        "status": "sealed_and_backed_up",
        "score_bundle_count": seal["score_bundle_count"],
        "backup_receipt_count": seal["backup_receipt_count"],
        "outcomes_revealed": False,
    }


def command_analyse(args: argparse.Namespace) -> dict[str, object]:
    config, _ = _load_contracts(args)
    release = load_json_object(args.release) if args.release else None
    assert_pre_release_cannot_execute(config, release)
    if release is None:
        raise PermissionError("Missing one-time release.")
    validate_release_manifest(release, config, execution_commit=_execution_commit())
    _, uids, _ = _load_materialised(
        args.output_root.resolve(),
        release,
        require_label_vault=True,
    )
    resource = load_description_bundle(require_approved=True)
    stats = config["statistics"]
    return analyse_sealed_graph_once(
        output_root=args.output_root.resolve(),
        backup_root=args.backup_root.resolve(),
        jobs=build_final_test_jobs(config),
        label_vault_path=_materialised_paths(args.output_root.resolve())["vault"],
        expected_row_uids=uids,
        expected_aspects=resource["canonical_order"],
        preregistration_sha256=preregistration_sha256(config),
        execution_commit=_execution_commit(),
        authorisation_id=str(release["authorisation_id"]),
        bootstrap_replicates=int(stats["bootstrap_draws"]),
        bootstrap_seed=int(stats["bootstrap_seed"]),
        test_contract_count=1,
    )


def _synthetic_values(grid: pd.DataFrame, method_id: str, field: str) -> np.ndarray:
    values = []
    for row in grid.itertuples(index=False):
        identity = "|".join(
            [
                method_id,
                field,
                str(row.row_uid),
                str(row.candidate_aspect),
                str(getattr(row, "candidate_sentiment", "")),
                str(row.representation_variant),
            ]
        )
        value = int(canonical_sha256(identity)[:12], 16) / float(16**12 - 1)
        values.append(0.02 + 0.96 * value)
    return np.asarray(values, dtype=float)


def command_dry_run(args: argparse.Namespace) -> dict[str, object]:
    config, _ = _load_contracts(args)
    root = args.output_root.resolve()
    backup = args.backup_root.resolve()
    if root.exists() or backup.exists():
        raise FileExistsError("Synthetic dry-run roots must be new and immutable.")
    resource = load_description_bundle(require_approved=True)
    aspects = tuple(str(value) for value in resource["canonical_order"])
    reviews = pd.DataFrame(
        {
            "row_uid": ["synthetic:1", "synthetic:2", "synthetic:3", "synthetic:4"],
            "text": [
                "The app worked well and support was helpful.",
                "The price was poor but the booking flow was easy.",
                "I could not access my account.",
                "Neutral synthetic row for pipeline verification.",
            ],
        }
    )
    labels_source = reviews.copy()
    labels_source["labels"] = [
        [(aspects[0], "positive"), (aspects[8], "positive")],
        [(aspects[10], "negative"), (aspects[5], "positive")],
        [(aspects[0], "negative")],
        [],
    ]
    labels = build_label_vault(labels_source, aspects)
    label_path = root / "private_release_data" / "label_vault.csv"
    atomic_csv(label_path, labels)
    prereg_hash = preregistration_sha256(config)
    execution_commit = "synthetic-dry-run-no-official-test"
    authorisation_id = "synthetic-dry-run-not-a-release"
    jobs = build_final_test_jobs(config)
    base_frames: dict[str, pd.DataFrame] = {}
    for job in [value for value in jobs if value.phase == "score"]:
        fold = final_fold(job)
        d_aspect, d_sentiment = build_unlabelled_grids(reviews, fold, "D", resource)
        d = score_frame_from_grids(
            job,
            "D",
            d_aspect,
            d_sentiment,
            _synthetic_values(d_aspect, job.method_id, "aspect"),
            _synthetic_values(d_sentiment, job.method_id, "sentiment"),
        )
        if job.level == "L2":
            n_aspect, n_sentiment = build_unlabelled_grids(reviews, fold, "N", resource)
            n_aspect = n_aspect[n_aspect["is_heldout"]].reset_index(drop=True)
            n_sentiment = n_sentiment[n_sentiment["is_heldout"]].reset_index(drop=True)
            n = score_frame_from_grids(
                job,
                "N",
                n_aspect,
                n_sentiment,
                _synthetic_values(n_aspect, job.method_id, "aspect"),
                _synthetic_values(n_sentiment, job.method_id, "sentiment"),
            )
            frame = canonical_l2_nd_scores(job, d, n)
        else:
            frame = d
        write_score_bundle(
            root,
            job,
            frame,
            preregistration_sha256=prereg_hash,
            execution_commit=execution_commit,
            authorisation_id=authorisation_id,
            aspect_threshold=0.45,
            runner_up_threshold=0.66,
            source_artifact_sha256s={"synthetic": canonical_sha256(job.to_dict())},
            expected_row_uids=reviews["row_uid"].tolist(),
            test_contract_count=0,
        )
        base_frames[job.job_id] = frame
    for job in [value for value in jobs if value.phase == "compose"]:
        sources = [
            read_score_bundle(
                root,
                _job_by_id(config, dependency),
                preregistration_sha256=prereg_hash,
                execution_commit=execution_commit,
                authorisation_id=authorisation_id,
                expected_row_uids=reviews["row_uid"].tolist(),
                expected_test_contract_count=0,
            )
            for dependency in job.depends_on
        ]
        by_method = {
            _job_by_id(config, dependency).method_id: value[0]
            for dependency, value in zip(job.depends_on, sources)
        }
        composed = compose_fixed_stage_scores(
            job,
            by_method["frozen_qwen_few_shot"],
            by_method["qwen_candidate_pair_qlora"],
        )
        write_score_bundle(
            root,
            job,
            composed,
            preregistration_sha256=prereg_hash,
            execution_commit=execution_commit,
            authorisation_id=authorisation_id,
            aspect_threshold=0.45,
            runner_up_threshold=0.66,
            source_artifact_sha256s={"synthetic_composition": canonical_sha256(job.to_dict())},
            expected_row_uids=reviews["row_uid"].tolist(),
            test_contract_count=0,
        )
    seal = seal_score_graph(
        root,
        backup,
        jobs,
        preregistration_sha256=prereg_hash,
        execution_commit=execution_commit,
        authorisation_id=authorisation_id,
        expected_row_uids=reviews["row_uid"].tolist(),
        test_contract_count=0,
    )
    analysis = analyse_sealed_graph_once(
        output_root=root,
        backup_root=backup,
        jobs=jobs,
        label_vault_path=label_path,
        expected_row_uids=reviews["row_uid"].tolist(),
        expected_aspects=aspects,
        preregistration_sha256=prereg_hash,
        execution_commit=execution_commit,
        authorisation_id=authorisation_id,
        bootstrap_replicates=2_000,
        bootstrap_seed=13,
        test_contract_count=0,
    )
    summary = {
        "schema_version": "taxonomy_final_test_synthetic_dry_run_v1",
        "status": "pass",
        "score_bundle_count": seal["score_bundle_count"],
        "backup_receipt_count": seal["backup_receipt_count"],
        "analysis_manifest_sha256": analysis["analysis_manifest_sha256"],
        "official_test_accessed": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "failure_count": 0,
    }
    atomic_json(root / "dry_run_summary.json", summary)
    return summary


def command_status(args: argparse.Namespace) -> dict[str, object]:
    config, _ = _load_contracts(args)
    root = args.output_root.resolve()
    jobs = build_final_test_jobs(config)
    completed = [
        job.job_id
        for job in jobs
        if job.phase in {"score", "compose"}
        and all(path.is_file() for path in score_bundle_paths(root, job.job_id))
    ]
    return {
        "schema_version": "taxonomy_final_test_status_v1",
        "status": "pre_release" if args.release is None else "released",
        "score_bundles_complete": len(completed),
        "score_bundles_planned": 105,
        "score_sealed": (root / "sealed_scores" / "score_seal.json").is_file(),
        "analysis_complete": (
            root / "single_reveal_analysis" / "analysis_manifest.json"
        ).is_file(),
        "outcomes_displayed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "preflight",
            "create-release",
            "materialise",
            "score-job",
            "compose-job",
            "seal",
            "analyse",
            "dry-run",
            "status",
        ),
    )
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--release-template", type=Path, default=DEFAULT_RELEASE_TEMPLATE)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--cloud-root", type=Path, default=DEFAULT_CLOUD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--i-have-explicit-user-authorisation", action="store_true")
    parser.add_argument("--authorisation-id")
    parser.add_argument("--authorised-at")
    parser.add_argument("--backup-root-id")
    args = parser.parse_args()
    if args.command == "dry-run":
        if args.output_root == DEFAULT_OUTPUT_ROOT:
            args.output_root = DEFAULT_DRY_RUN_ROOT
        if args.backup_root is None:
            args.backup_root = Path(str(args.output_root) + "_backup")
    if args.command in {"materialise", "seal", "analyse"} and args.backup_root is None:
        parser.error(f"{args.command} requires --backup-root")
    if args.command in {"score-job", "compose-job"} and not args.job_id:
        parser.error(f"{args.command} requires --job-id")
    if args.command == "create-release" and (
        args.release is None or not args.backup_root_id
    ):
        parser.error("create-release requires --release and --backup-root-id")
    return args


def main() -> int:
    args = parse_args()
    commands = {
        "plan": command_plan,
        "preflight": command_preflight,
        "create-release": command_create_release,
        "materialise": command_materialise,
        "score-job": command_score_job,
        "compose-job": command_compose_job,
        "seal": command_seal,
        "analyse": command_analyse,
        "dry-run": command_dry_run,
        "status": command_status,
    }
    result = commands[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
