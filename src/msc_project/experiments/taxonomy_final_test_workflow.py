"""Score-seal-reveal workflow for the locked taxonomy final evaluation.

This module contains no default test path and never selects a model or
threshold.  Its score artifacts deliberately exclude review text and target
labels.  Labels can be joined only after the complete score graph has been
sealed and copied to the registered backup root.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_final_test import (
    FIXED_COMPOSITION_METHOD,
    PROTOCOL_ID,
    FinalTestJob,
    canonical_sha256,
    file_sha256,
)
from msc_project.experiments.taxonomy_protocol import TaxonomyFold
from msc_project.experiments.taxonomy_two_stage_formal import variant_map
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS

SCORE_COLUMNS = (
    "job_id",
    "method_id",
    "level",
    "fold_id",
    "training_scope_id",
    "condition",
    "row_uid",
    "candidate_aspect",
    "candidate_sentiment",
    "representation_variant",
    "is_seen",
    "is_heldout",
    "aspect_score",
    "sentiment_score",
)
FORBIDDEN_SCORE_COLUMNS = {
    "text",
    "target",
    "labels",
    "labels_json",
    "label_codes",
    "pair_labels",
    "supervision_labels",
}
SENTIMENTS = tuple(str(value) for value in CANDIDATE_SENTIMENTS)


def _render_aspect_candidate(
    aspect: str,
    variant: str,
    resource: Mapping[str, object],
) -> str:
    aspects = resource.get("minimal_aspects", resource.get("aspects"))
    if not isinstance(aspects, Mapping) or aspect not in aspects:
        raise ValueError(f"Missing frozen minimal description for {aspect!r}.")
    if variant == "name_only":
        return f"Aspect: {aspect}."
    if variant == "name_and_description":
        return f"Aspect: {aspect}. Definition: {str(aspects[aspect]).strip()}"
    raise ValueError(f"Final-test candidate variant is not permitted: {variant!r}.")


def _render_sentiment_candidate(
    aspect: str,
    sentiment: str,
    variant: str,
    resource: Mapping[str, object],
) -> str:
    if sentiment not in SENTIMENTS:
        raise ValueError(f"Unknown sentiment: {sentiment!r}.")
    return (
        f"{_render_aspect_candidate(aspect, variant, resource)} "
        f"Candidate sentiment: {sentiment}."
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def dataframe_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    records = (
        frame[list(columns)]
        .astype({column: str for column in columns})
        .sort_values(list(columns), kind="stable")
        .to_dict(orient="records")
    )
    return canonical_sha256(records)


def validate_unlabelled_reviews(frame: pd.DataFrame) -> dict[str, object]:
    required = {"row_uid", "text"}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Unlabelled review frame is invalid; missing={missing}.")
    forbidden = sorted(FORBIDDEN_SCORE_COLUMNS.intersection(frame.columns) - {"text"})
    if forbidden:
        raise ValueError(f"Unlabelled review frame contains label columns: {forbidden}.")
    if frame["row_uid"].astype(str).duplicated().any():
        raise ValueError("Unlabelled review row_uid values must be unique.")
    return {
        "rows": len(frame),
        "row_uid_sha256": canonical_sha256(sorted(frame["row_uid"].astype(str))),
    }


def build_unlabelled_grids(
    reviews: pd.DataFrame,
    fold: TaxonomyFold,
    condition: str,
    resource: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build model inputs without consulting any target label."""

    validate_unlabelled_reviews(reviews)
    variants = variant_map(fold, condition)
    aspect_records: list[dict[str, object]] = []
    sentiment_records: list[dict[str, object]] = []
    ordered = reviews.assign(_uid=reviews["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    )
    for _, row in ordered.iterrows():
        uid = str(row["row_uid"])
        text = "" if pd.isna(row["text"]) else str(row["text"])
        for aspect in fold.evaluation_aspects:
            variant = variants[str(aspect)]
            common = {
                "row_uid": uid,
                "text": text,
                "candidate_aspect": str(aspect),
                "representation_variant": variant,
                "is_seen": bool(aspect in fold.seen_aspects),
                "is_heldout": bool(aspect in fold.heldout_aspects),
            }
            aspect_records.append(
                {
                    **common,
                    "candidate_text": _render_aspect_candidate(
                        str(aspect), variant, resource
                    ),
                }
            )
            for sentiment in SENTIMENTS:
                sentiment_records.append(
                    {
                        **common,
                        "candidate_sentiment": sentiment,
                        "candidate_text": _render_sentiment_candidate(
                            str(aspect), sentiment, variant, resource
                        ),
                        "negative_type": "eval_candidate",
                    }
                )
    aspect_grid = pd.DataFrame.from_records(aspect_records)
    sentiment_grid = pd.DataFrame.from_records(sentiment_records)
    if aspect_grid.duplicated(["row_uid", "candidate_aspect"]).any():
        raise AssertionError("Unlabelled aspect identities are not unique.")
    if sentiment_grid.duplicated(
        ["row_uid", "candidate_aspect", "candidate_sentiment"]
    ).any():
        raise AssertionError("Unlabelled sentiment identities are not unique.")
    if len(sentiment_grid) != len(reviews) * len(fold.evaluation_aspects) * 3:
        raise AssertionError("Unlabelled sentiment grid is incomplete.")
    return aspect_grid, sentiment_grid


def score_frame_from_grids(
    job: FinalTestJob,
    condition: str,
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    aspect_scores: Sequence[float],
    sentiment_scores: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    aspects = aspect_grid.copy()
    sentiments = sentiment_grid.copy()
    aspect_values = np.asarray(aspect_scores, dtype=float).reshape(-1)
    sentiment_values = np.asarray(sentiment_scores, dtype=float).reshape(-1)
    if len(aspects) != len(aspect_values) or len(sentiments) != len(sentiment_values):
        raise ValueError("Model score lengths do not match the unlabelled grids.")
    if not np.isfinite(aspect_values).all() or not np.isfinite(sentiment_values).all():
        raise ValueError("Model scores contain a non-finite value.")
    if (
        (aspect_values < 0).any()
        or (aspect_values > 1).any()
        or (sentiment_values < 0).any()
        or (sentiment_values > 1).any()
    ):
        raise ValueError("Model scores must remain inside [0, 1].")
    aspects["aspect_score"] = aspect_values
    sentiments["sentiment_score"] = sentiment_values
    frame = sentiments.merge(
        aspects[["row_uid", "candidate_aspect", "aspect_score"]],
        on=["row_uid", "candidate_aspect"],
        how="left",
        validate="many_to_one",
    )
    frame.insert(0, "condition", str(condition))
    frame.insert(0, "training_scope_id", job.training_scope_id)
    frame.insert(0, "fold_id", job.fold_id)
    frame.insert(0, "level", job.level)
    frame.insert(0, "method_id", job.method_id)
    frame.insert(0, "job_id", job.job_id)
    return frame[list(SCORE_COLUMNS)].reset_index(drop=True)


def canonical_l2_nd_scores(
    job: FinalTestJob,
    d_scores: pd.DataFrame,
    n_heldout_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Share unchanged seen scores across N/D by construction."""

    if job.level != "L2" or tuple(job.conditions) != ("D", "N"):
        raise ValueError("Canonical N/D construction is L2-only.")
    if set(d_scores["condition"].astype(str)) != {"D"}:
        raise ValueError("D source contains another condition.")
    if set(n_heldout_scores["condition"].astype(str)) != {"N"}:
        raise ValueError("N held-out source contains another condition.")
    if not n_heldout_scores["is_heldout"].astype(bool).all():
        raise ValueError("N source may contain only held-out candidates.")
    seen = d_scores[d_scores["is_seen"].astype(bool)].copy()
    seen["condition"] = "N"
    n_scores = pd.concat([seen, n_heldout_scores], ignore_index=True)
    return pd.concat([d_scores, n_scores], ignore_index=True)[list(SCORE_COLUMNS)]


def _score_identity(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[["condition", "row_uid", "candidate_aspect", "candidate_sentiment"]]
        .astype(str)
    )


def validate_score_frame(
    frame: pd.DataFrame,
    job: FinalTestJob,
    *,
    expected_row_uids: Sequence[str] | None = None,
) -> dict[str, object]:
    missing = sorted(set(SCORE_COLUMNS) - set(frame.columns))
    forbidden = sorted(FORBIDDEN_SCORE_COLUMNS.intersection(frame.columns))
    if missing or forbidden or frame.empty:
        raise ValueError(
            f"Score frame violates its label-free schema; missing={missing}, "
            f"forbidden={forbidden}."
        )
    if tuple(frame.columns) != SCORE_COLUMNS:
        raise ValueError("Score-frame column order differs from the frozen contract.")
    identities = _score_identity(frame)
    if identities.duplicated().any():
        raise ValueError("Score frame contains duplicate candidate identities.")
    scalar_expectations = {
        "job_id": job.job_id,
        "method_id": job.method_id,
        "level": job.level,
        "fold_id": job.fold_id,
        "training_scope_id": job.training_scope_id,
    }
    for column, expected in scalar_expectations.items():
        if set(frame[column].astype(str)) != {str(expected)}:
            raise ValueError(f"Score frame {column} does not equal the frozen job.")
    if tuple(sorted(frame["condition"].astype(str).unique())) != tuple(
        sorted(job.conditions)
    ):
        raise ValueError("Score frame condition set differs from the frozen job.")
    values = frame[["aspect_score", "sentiment_score"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Score frame contains an invalid probability-like score.")
    if frame["aspect_score"].nunique() < 2 or frame["sentiment_score"].nunique() < 2:
        raise ValueError("Score frame failed the prediction-collapse check.")
    grouped = frame.groupby(
        ["condition", "row_uid", "candidate_aspect"], sort=False
    )
    if not grouped.size().eq(3).all():
        raise ValueError("Every review-aspect identity must contain three sentiments.")
    sentiment_sets = grouped["candidate_sentiment"].agg(lambda values: tuple(sorted(values)))
    expected_sentiments = tuple(sorted(SENTIMENTS))
    if not sentiment_sets.map(lambda value: value == expected_sentiments).all():
        raise ValueError("A review-aspect identity has the wrong sentiment set.")
    score_span = grouped["aspect_score"].agg(["min", "max"])
    if not np.allclose(score_span["min"], score_span["max"], atol=0.0, rtol=0.0):
        raise ValueError("Aspect scores differ across sentiments.")
    per_review = frame.groupby(["condition", "row_uid"], sort=False).size()
    if not per_review.eq(36).all():
        raise ValueError("Every condition must contain the complete 36-pair grid.")
    observed_uids = sorted(frame["row_uid"].astype(str).unique())
    if expected_row_uids is not None and observed_uids != sorted(
        str(value) for value in expected_row_uids
    ):
        raise ValueError("Score frame review identities differ from the release manifest.")
    if job.level == "L2":
        seen = frame[frame["is_seen"].astype(bool)]
        d = seen[seen["condition"].eq("D")].sort_values(
            ["row_uid", "candidate_aspect", "candidate_sentiment"], kind="stable"
        )
        n = seen[seen["condition"].eq("N")].sort_values(
            ["row_uid", "candidate_aspect", "candidate_sentiment"], kind="stable"
        )
        keys = ["row_uid", "candidate_aspect", "candidate_sentiment"]
        if not np.array_equal(d[keys].to_numpy(dtype=str), n[keys].to_numpy(dtype=str)):
            raise ValueError("N/D seen identities are not exactly aligned.")
        if not np.array_equal(
            d[["aspect_score", "sentiment_score"]].to_numpy(dtype=float),
            n[["aspect_score", "sentiment_score"]].to_numpy(dtype=float),
        ):
            raise ValueError("N/D seen scores are not bitwise shared by construction.")
    if (frame["is_seen"].astype(bool) & frame["is_heldout"].astype(bool)).any():
        raise ValueError("A candidate cannot be both seen and held out.")
    return {
        "status": "pass",
        "rows": len(frame),
        "review_rows": len(observed_uids),
        "conditions": list(job.conditions),
        "score_sha256": dataframe_sha256(frame, SCORE_COLUMNS),
        "non_finite_value_count": 0,
        "duplicate_identity_count": 0,
        "test_contract_count": 1,
    }


def decode_capped_two(
    frame: pd.DataFrame,
    *,
    aspect_threshold: float,
    runner_up_threshold: float,
) -> pd.Series:
    """Decode a label-free complete grid with the frozen capped-two rule."""

    ordered = frame.copy()
    sentiment_order = {value: index for index, value in enumerate(SENTIMENTS)}
    ordered["_sentiment_order"] = ordered["candidate_sentiment"].map(sentiment_order)
    ordered["_position"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(
        ["condition", "row_uid", "candidate_aspect", "_sentiment_order"],
        kind="stable",
    )
    width = 3
    if len(ordered) % width:
        raise ValueError("Score grid cannot be reshaped into three sentiments.")
    aspect = ordered["aspect_score"].to_numpy(dtype=float).reshape(-1, width)
    if not np.allclose(aspect, aspect[:, [0]], atol=0.0, rtol=0.0):
        raise ValueError("Aspect scores differ within a sentiment group.")
    sentiment = ordered["sentiment_score"].to_numpy(dtype=float).reshape(-1, width)
    selected = aspect[:, 0] >= float(aspect_threshold)
    rank = np.argsort(-sentiment, axis=1, kind="stable")
    mask = np.zeros_like(sentiment, dtype=bool)
    selected_rows = np.flatnonzero(selected)
    mask[selected_rows, rank[selected_rows, 0]] = True
    runner = sentiment[np.arange(len(sentiment)), rank[:, 1]]
    doubled = np.flatnonzero(selected & (runner >= float(runner_up_threshold)))
    mask[doubled, rank[doubled, 1]] = True
    if int(mask.sum(axis=1).max(initial=0)) > 2:
        raise AssertionError("The frozen decoder emitted a third sentiment.")
    output = np.zeros(len(frame), dtype=bool)
    output[ordered["_position"].to_numpy(dtype=int)] = mask.reshape(-1)
    return pd.Series(output, index=frame.index, dtype=bool)


def compose_fixed_stage_scores(
    job: FinalTestJob,
    few_shot: pd.DataFrame,
    qlora: pd.DataFrame,
) -> pd.DataFrame:
    if job.method_id != FIXED_COMPOSITION_METHOD:
        raise ValueError("The requested job is not the frozen fixed composition.")
    keys = ["condition", "row_uid", "candidate_aspect", "candidate_sentiment"]
    left = few_shot.sort_values(keys, kind="stable").reset_index(drop=True)
    right = qlora.sort_values(keys, kind="stable").reset_index(drop=True)
    if not np.array_equal(left[keys].to_numpy(dtype=str), right[keys].to_numpy(dtype=str)):
        raise ValueError("Fixed-composition source identities are not aligned.")
    invariant = [
        "level",
        "fold_id",
        "training_scope_id",
        "representation_variant",
        "is_seen",
        "is_heldout",
    ]
    for column in invariant:
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise ValueError(f"Fixed-composition source {column} values differ.")
    output = left.copy()
    output["job_id"] = job.job_id
    output["method_id"] = job.method_id
    output["sentiment_score"] = right["sentiment_score"].to_numpy(dtype=float)
    return output[list(SCORE_COLUMNS)]


def score_bundle_paths(output_root: Path, job_id: str) -> tuple[Path, Path]:
    root = output_root / "sealed_scores" / job_id
    return root / "scores.csv", root / "manifest.json"


def write_score_bundle(
    output_root: Path,
    job: FinalTestJob,
    frame: pd.DataFrame,
    *,
    preregistration_sha256: str,
    execution_commit: str,
    authorisation_id: str,
    aspect_threshold: float,
    runner_up_threshold: float,
    source_artifact_sha256s: Mapping[str, str],
    expected_row_uids: Sequence[str] | None = None,
    test_contract_count: int = 1,
) -> dict[str, object]:
    csv_path, manifest_path = score_bundle_paths(output_root, job.job_id)
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Immutable score bundle already exists: {job.job_id}")
    audit = validate_score_frame(frame, job, expected_row_uids=expected_row_uids)
    prediction = decode_capped_two(
        frame,
        aspect_threshold=aspect_threshold,
        runner_up_threshold=runner_up_threshold,
    )
    if not prediction.any() or prediction.all():
        raise ValueError("Decoded prediction collapse detected before score sealing.")
    atomic_csv(csv_path, frame)
    persisted = pd.read_csv(csv_path)
    audit = validate_score_frame(
        persisted, job, expected_row_uids=expected_row_uids
    )
    manifest: dict[str, object] = {
        "schema_version": "taxonomy_final_test_score_bundle_v1",
        "protocol_id": PROTOCOL_ID,
        "job": job.to_dict(),
        "preregistration_sha256": preregistration_sha256,
        "execution_commit": execution_commit,
        "authorisation_id": authorisation_id,
        "score_file": csv_path.name,
        "score_file_sha256": file_sha256(csv_path),
        "score_frame_sha256": audit["score_sha256"],
        "rows": audit["rows"],
        "review_rows": audit["review_rows"],
        "aspect_threshold": float(aspect_threshold),
        "runner_up_threshold": float(runner_up_threshold),
        "maximum_sentiments_per_selected_aspect": 2,
        "predicted_pair_count": int(prediction.sum()),
        "source_artifact_sha256s": dict(sorted(source_artifact_sha256s.items())),
        "failure_count": 0,
        "non_finite_value_count": 0,
        "prediction_collapse_count": 0,
        "third_sentiment_violation_count": 0,
        "test_contract_count": int(test_contract_count),
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def read_score_bundle(
    output_root: Path,
    job: FinalTestJob,
    *,
    preregistration_sha256: str,
    execution_commit: str,
    authorisation_id: str,
    expected_row_uids: Sequence[str] | None = None,
    expected_test_contract_count: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    csv_path, manifest_path = score_bundle_paths(output_root, job.job_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed = dict(manifest)
    observed_seal = sealed.pop("manifest_payload_sha256", None)
    if observed_seal != canonical_sha256(sealed):
        raise ValueError(f"Score manifest payload hash mismatch: {job.job_id}")
    expected = {
        "schema_version": "taxonomy_final_test_score_bundle_v1",
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": preregistration_sha256,
        "execution_commit": execution_commit,
        "authorisation_id": authorisation_id,
        "score_file_sha256": file_sha256(csv_path),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Score manifest {key} mismatch: {job.job_id}")
    if manifest.get("job") != job.to_dict():
        raise ValueError(f"Score manifest job contract mismatch: {job.job_id}")
    if int(manifest.get("test_contract_count", -1)) != int(
        expected_test_contract_count
    ):
        raise ValueError(f"Score manifest test-contract count mismatch: {job.job_id}")
    if any(
        int(manifest.get(key, -1)) != 0
        for key in (
            "failure_count",
            "non_finite_value_count",
            "prediction_collapse_count",
            "third_sentiment_violation_count",
        )
    ):
        raise ValueError(f"Score bundle reports an integrity failure: {job.job_id}")
    frame = pd.read_csv(csv_path)
    audit = validate_score_frame(frame, job, expected_row_uids=expected_row_uids)
    if audit["score_sha256"] != manifest.get("score_frame_sha256"):
        raise ValueError(f"Score-frame hash mismatch: {job.job_id}")
    return frame, manifest


def copy_verified_file(source: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise FileExistsError(f"Conflicting backup file: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    observed = file_sha256(source)
    if file_sha256(destination) != observed:
        raise OSError(f"Backup SHA-256 verification failed: {destination}")
    return {
        "source": str(source),
        "destination": str(destination),
        "sha256": observed,
        "bytes": source.stat().st_size,
    }


def seal_score_graph(
    output_root: Path,
    backup_root: Path,
    jobs: Sequence[FinalTestJob],
    *,
    preregistration_sha256: str,
    execution_commit: str,
    authorisation_id: str,
    expected_row_uids: Sequence[str],
    test_contract_count: int = 1,
) -> dict[str, object]:
    score_jobs = [job for job in jobs if job.phase in {"score", "compose"}]
    receipts: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for job in score_jobs:
        _, manifest = read_score_bundle(
            output_root,
            job,
            preregistration_sha256=preregistration_sha256,
            execution_commit=execution_commit,
            authorisation_id=authorisation_id,
            expected_row_uids=expected_row_uids,
            expected_test_contract_count=test_contract_count,
        )
        manifests.append(manifest)
        for source in score_bundle_paths(output_root, job.job_id):
            relative = source.relative_to(output_root)
            receipts.append(copy_verified_file(source, backup_root / relative))
    seal: dict[str, object] = {
        "schema_version": "taxonomy_final_test_score_seal_v1",
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": preregistration_sha256,
        "execution_commit": execution_commit,
        "authorisation_id": authorisation_id,
        "score_bundle_count": len(manifests),
        "expected_score_bundle_count": 105,
        "all_score_bundles_complete": len(manifests) == 105,
        "score_manifest_payload_sha256": canonical_sha256(
            sorted(str(value["manifest_payload_sha256"]) for value in manifests)
        ),
        "backup_receipt_count": len(receipts),
        "backup_receipts": receipts,
        "outcomes_revealed": False,
        "failure_count": 0,
        "test_contract_count": int(test_contract_count),
    }
    if not seal["all_score_bundles_complete"]:
        raise ValueError("The final score graph is incomplete; analysis remains locked.")
    seal["seal_payload_sha256"] = canonical_sha256(seal)
    seal_path = output_root / "sealed_scores" / "score_seal.json"
    backup_path = backup_root / "sealed_scores" / "score_seal.json"
    if seal_path.exists() or backup_path.exists():
        raise FileExistsError("An immutable final score seal already exists.")
    atomic_json(seal_path, seal)
    copy_verified_file(seal_path, backup_path)
    return seal


def validate_score_seal(
    path: Path,
    *,
    preregistration_sha256: str,
    execution_commit: str,
    authorisation_id: str,
    expected_test_contract_count: int = 1,
) -> dict[str, object]:
    seal = json.loads(path.read_text(encoding="utf-8"))
    value = dict(seal)
    observed = value.pop("seal_payload_sha256", None)
    if observed != canonical_sha256(value):
        raise ValueError("Final score-seal payload hash mismatch.")
    if (
        seal.get("schema_version") != "taxonomy_final_test_score_seal_v1"
        or seal.get("protocol_id") != PROTOCOL_ID
        or seal.get("preregistration_sha256") != preregistration_sha256
        or seal.get("execution_commit") != execution_commit
        or seal.get("authorisation_id") != authorisation_id
        or seal.get("all_score_bundles_complete") is not True
        or int(seal.get("score_bundle_count", -1)) != 105
        or seal.get("outcomes_revealed") is not False
        or int(seal.get("failure_count", -1)) != 0
        or int(seal.get("test_contract_count", -1))
        != int(expected_test_contract_count)
    ):
        raise PermissionError("The complete score-and-backup barrier has not passed.")
    return seal
