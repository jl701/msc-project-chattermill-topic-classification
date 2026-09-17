"""Evaluate fixed stage hybrids and one leakage-safe QLoRA calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import analyse_taxonomy_post_supervisor_formal_v2 as legacy
import analyse_taxonomy_scientific_freeze_v1 as inference
import analyse_taxonomy_two_stage_global_router as router

from msc_project.experiments.taxonomy_global_router import (
    ComponentArrays,
    threshold_aligned_scores,
    validate_component_alignment,
)
from msc_project.experiments.taxonomy_post_supervisor import (
    evaluate_l2_condition,
    post_supervisor_l2_folds,
)
from msc_project.experiments.taxonomy_scientific_freeze import (
    row_pair_confusions,
)
from msc_project.experiments.taxonomy_two_stage import (
    capped_two_sentiment_prediction_mask,
)

STUDY_ID = "taxonomy_two_stage_stage_hybrid_calibration_v1"
FEWSHOT = "frozen_qwen_few_shot"
QLORA = "qwen_candidate_pair_qlora"
PRIMARY_HYBRID = "frozen_qwen_few_shot_stage1__qlora_stage2"
REVERSE_HYBRID = "qlora_stage1__frozen_qwen_few_shot_stage2"
CALIBRATOR = "qwen_candidate_pair_qlora_relative_calibrator_v1"
L2_FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))
CONDITIONS = ("D", "N")
INPUT_METHODS = (FEWSHOT, QLORA)
COUNT_COLUMNS = ("pair_tp", "pair_fp", "pair_fn")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write an empty study table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def non_finite_numeric_count(frame: pd.DataFrame) -> int:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return 0
    return int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())


def validate_config(config: Mapping[str, Any]) -> None:
    calibration = config.get("direction_2_qlora_relative_calibrator", {})
    if (
        config.get("schema_version") != STUDY_ID
        or config.get("protocol_id") != legacy.PROTOCOL_ID
        or config.get("evaluation_partition") != "validation_only"
        or config.get("official_test_permitted") is not False
        or config.get("include_official_test") is not False
        or int(config.get("test_contract_count", -1)) != 0
        or tuple(config.get("folds", ())) != L2_FOLDS
        or tuple(config.get("input_methods", ())) != INPUT_METHODS
        or config.get("primary_condition") != "D"
        or config.get("unchanged_transfer_condition") != "N"
        or int(calibration.get("review_crossfit_folds", -1)) != 5
        or calibration.get("selection_condition") != "D_only"
    ):
        raise ValueError("Stage-hybrid/calibration configuration violates the boundary.")
    features = tuple(calibration.get("features", ()))
    if features != (
        "clipped_logit_absolute_qlora_aspect_probability",
        "absolute_logit_minus_within_review_median_other_aspect_logit",
    ):
        raise ValueError("The calibrator feature family differs from preregistration.")


def review_block(row_uid: str, *, seed: int = 13, folds: int = 5) -> int:
    if folds < 2:
        raise ValueError("Review cross-fitting requires at least two blocks.")
    payload = f"{int(seed)}|{row_uid!s}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % int(folds)


def build_stage_hybrid_frame(
    stage_1: ComponentArrays,
    stage_2: ComponentArrays,
    *,
    method_id: str,
) -> pd.DataFrame:
    """Combine a frozen aspect gate with a frozen sentiment decoder exactly."""

    validate_component_alignment(stage_1, stage_2)
    frame = stage_1.ordered_frame.copy()
    frame["sentiment_score"] = stage_2.ordered_frame["sentiment_score"].to_numpy(
        dtype=float
    )
    frame["method_id"] = str(method_id)
    prediction = capped_two_sentiment_prediction_mask(
        frame,
        aspect_threshold=0.5,
        second_sentiment_threshold=0.5,
    ).to_numpy(dtype=bool)
    expected_aspect = stage_1.aspect_present
    observed_aspect = prediction.reshape(-1, 3).any(axis=1)
    if not np.array_equal(expected_aspect, observed_aspect):
        raise AssertionError("Stage hybrid changed the frozen Stage-1 decision.")
    return frame


def relative_feature_matrix(
    raw_scored: pd.DataFrame,
    component: ComponentArrays,
    *,
    epsilon: float,
) -> np.ndarray:
    """Return absolute and within-review-relative raw QLoRA logit features."""

    required = {"row_uid", "candidate_aspect", "target", "aspect_score"}
    missing = sorted(required - set(raw_scored.columns))
    if missing or raw_scored.empty:
        raise ValueError(f"Relative-feature score frame is invalid; missing={missing}.")
    grouped = (
        raw_scored.assign(
            row_uid=raw_scored["row_uid"].astype(str),
            candidate_aspect=raw_scored["candidate_aspect"].astype(str),
        )
        .groupby(["row_uid", "candidate_aspect"], sort=True)
        .agg(
            aspect_score=("aspect_score", "first"),
            aspect_score_min=("aspect_score", "min"),
            aspect_score_max=("aspect_score", "max"),
            presence_target=("target", "max"),
        )
        .reset_index()
    )
    if not np.allclose(
        grouped["aspect_score_min"], grouped["aspect_score_max"], atol=0.0, rtol=0.0
    ):
        raise ValueError("Aspect scores differ across sentiments within a candidate.")
    expected_keys = np.column_stack((component.row_uid, component.candidate_aspect))
    observed_keys = grouped[["row_uid", "candidate_aspect"]].to_numpy(dtype=str)
    if not np.array_equal(expected_keys.astype(str), observed_keys):
        raise ValueError("Relative-feature candidate order differs from the component.")
    counts = grouped.groupby("row_uid", sort=False).size().to_numpy(dtype=int)
    if set(counts) != {12}:
        raise ValueError("Relative calibration requires all twelve aspect scores.")
    probability = np.clip(
        grouped["aspect_score"].to_numpy(dtype=float),
        float(epsilon),
        1.0 - float(epsilon),
    )
    absolute = np.log(probability / (1.0 - probability)).reshape(-1, 12)
    other_median = np.empty_like(absolute)
    for aspect_index in range(absolute.shape[1]):
        other_median[:, aspect_index] = np.median(
            np.delete(absolute, aspect_index, axis=1), axis=1
        )
    features = np.column_stack(
        (absolute.reshape(-1), (absolute - other_median).reshape(-1))
    )
    if not np.isfinite(features).all() or features.shape != (len(component.row_uid), 2):
        raise ValueError("Relative calibration produced invalid features.")
    return features


def sentiment_only_mask(component: ComponentArrays) -> np.ndarray:
    """Return the frozen capped-two Stage-2 mask before applying Stage 1."""

    scores = np.asarray(component.sentiment_score, dtype=float)
    rank = np.argsort(-scores, axis=1, kind="stable")
    mask = np.zeros_like(scores, dtype=bool)
    rows = np.arange(len(scores))
    mask[rows, rank[:, 0]] = True
    runner = scores[rows, rank[:, 1]]
    doubled = rows[runner >= 0.5]
    mask[doubled, rank[doubled, 1]] = True
    if int(mask.sum(axis=1).max(initial=0)) > 2:
        raise AssertionError("The sentiment-only decoder emitted a third sentiment.")
    return mask


def threshold_metrics(
    probabilities: np.ndarray,
    sentiment_mask: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    selected = np.asarray(probabilities, dtype=float) >= float(threshold)
    prediction = selected[:, None] & np.asarray(sentiment_mask, dtype=bool)
    mask = np.asarray(eligible, dtype=bool)
    truth = np.asarray(target, dtype=bool)[mask]
    pred = prediction[mask]
    truth_presence = truth.any(axis=1)
    pred_presence = pred.any(axis=1)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    presence_tp = int(np.sum(truth_presence & pred_presence))
    presence_fp = int(np.sum(~truth_presence & pred_presence))
    presence_fn = int(np.sum(truth_presence & ~pred_presence))
    pair_denominator = 2 * tp + fp + fn
    presence_denominator = 2 * presence_tp + presence_fp + presence_fn
    return {
        "threshold": float(threshold),
        "pair_micro_f1": float(2 * tp / pair_denominator if pair_denominator else 0.0),
        "presence_f1": float(
            2 * presence_tp / presence_denominator if presence_denominator else 0.0
        ),
        "aspect_call_rate": float(pred_presence.mean()),
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "presence_tp": presence_tp,
        "presence_fp": presence_fp,
        "presence_fn": presence_fn,
    }


def select_calibrated_threshold(
    probabilities: np.ndarray,
    sentiment_mask: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, float | int]:
    records = [
        threshold_metrics(
            probabilities,
            sentiment_mask,
            target,
            eligible,
            float(threshold),
        )
        for threshold in thresholds
    ]
    return max(
        records,
        key=lambda row: (
            float(row["pair_micro_f1"]),
            float(row["presence_f1"]),
            -float(row["aspect_call_rate"]),
            float(row["threshold"]),
        ),
    )


def fit_calibrator(
    features: np.ndarray,
    target_presence: np.ndarray,
    eligible: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> Pipeline:
    mask = np.asarray(eligible, dtype=bool)
    labels = np.asarray(target_presence, dtype=int)[mask]
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibrator training requires both presence classes.")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(config["C"]),
                    class_weight=str(config["class_weight"]),
                    solver=str(config["solver"]),
                    max_iter=int(config["max_iter"]),
                    random_state=int(config["random_seed"]),
                    l1_ratio=(0.0 if str(config["penalty"]) == "l2" else None),
                ),
            ),
        ]
    )
    model.fit(np.asarray(features, dtype=float)[mask], labels)
    return model


def calibrated_score_frame(
    component: ComponentArrays,
    aligned_probabilities: np.ndarray,
    *,
    method_id: str,
) -> pd.DataFrame:
    probabilities = np.asarray(aligned_probabilities, dtype=float)
    if (
        probabilities.shape != component.aspect_score.shape
        or not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError("Calibrated probabilities are invalid.")
    frame = component.ordered_frame.copy()
    frame["aspect_score"] = np.repeat(probabilities, 3)
    frame["method_id"] = str(method_id)
    return frame


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    fold_id: str,
    condition: str,
    method_id: str,
    evidence_role: str,
    taxonomy_fold: Any,
) -> tuple[dict[str, Any], list[pd.DataFrame]]:
    result = evaluate_l2_condition(
        frame,
        taxonomy_fold,
        aspect_threshold=0.5,
        second_sentiment_threshold=0.5,
    )
    prediction = capped_two_sentiment_prediction_mask(
        frame,
        aspect_threshold=0.5,
        second_sentiment_threshold=0.5,
    )
    l2_s = result["L2_S"]
    l2_e = result["L2_E"]
    selected = prediction.to_numpy(dtype=bool).reshape(-1, 3).any(axis=1)
    record = {
        "method_id": method_id,
        "evidence_role": evidence_role,
        "fold_id": fold_id,
        "condition": condition,
        "score_sha256": result["score_sha256"],
        "heldout_pair_micro_f1": l2_e["partitions"]["heldout"]["pair_micro_f1"],
        "overall_pair_micro_f1": l2_e["partitions"]["overall"]["pair_micro_f1"],
        "seen_pair_micro_f1": l2_e["partitions"]["seen"]["pair_micro_f1"],
        "heldout_pair_precision": l2_e["partitions"]["heldout"]["pair_micro_precision"],
        "heldout_pair_recall": l2_e["partitions"]["heldout"]["pair_micro_recall"],
        "heldout_presence_f1": l2_s["aspect_presence"]["f1"],
        "heldout_presence_ap": l2_s["aspect_presence"]["average_precision"],
        "oracle_gated_sentiment_set_micro_f1": l2_s["oracle_aspect_gated_sentiment"]["capped_two_pair_metrics"]["pair_micro_f1"],
        "maximum_sentiments_per_selected_aspect": l2_e["maximum_sentiments_per_selected_aspect"],
        "aspect_call_rate": float(selected.mean()),
        "selected_aspect_instances": l2_e["selected_aspect_instances"],
        "test_contract_count": 0,
    }
    evidence_frames = []
    for partition, aspects in (
        ("heldout", taxonomy_fold.heldout_aspects),
        ("overall", taxonomy_fold.evaluation_aspects),
    ):
        evidence = row_pair_confusions(
            frame,
            prediction,
            aspects=aspects,
            method_id=method_id,
            fold_id=fold_id,
            condition=condition,
            level="L2",
        )
        evidence.insert(4, "evidence_role", evidence_role)
        evidence.insert(5, "partition", partition)
        evidence_frames.append(evidence)
    return record, evidence_frames


def _bootstrap_tables(
    study_evidence: pd.DataFrame,
    *,
    formal_evidence_root: Path,
    local_evidence_root: Path,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    existing = inference._load_l2_evidence(formal_evidence_root, local_evidence_root)
    identities = (
        study_evidence[["method_id", "evidence_role"]]
        .drop_duplicates()
        .sort_values(["method_id", "evidence_role"], kind="stable")
    )
    study_ids = [
        f"{method}__{role}"
        for method, role in identities.itertuples(index=False, name=None)
    ]
    transformed = study_evidence[study_evidence["partition"].eq("heldout")].copy()
    transformed["method_id"] = (
        transformed["method_id"].astype(str)
        + "__"
        + transformed["evidence_role"].astype(str)
    )
    combined = pd.concat([existing, transformed], ignore_index=True)
    methods = (*inference.METHODS, *study_ids)
    cube, row_uids = router._evidence_cube_for_methods(combined, methods)
    balanced_boot, pooled_boot = inference._synchronised_bootstrap(
        cube, draws=draws, seed=seed
    )
    summaries: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    points: dict[tuple[str, str], float] = {}
    for method_index, bootstrap_id in enumerate(methods):
        for condition_index, condition in enumerate(CONDITIONS):
            balanced, pooled, _ = inference._point_metrics(
                cube[method_index, condition_index]
            )
            points[(bootstrap_id, condition)] = balanced
            if bootstrap_id not in study_ids:
                continue
            low, high = inference._interval(
                balanced_boot[:, method_index, condition_index]
            )
            pooled_low, pooled_high = inference._interval(
                pooled_boot[:, method_index, condition_index]
            )
            method_id, evidence_role = bootstrap_id.rsplit("__", 1)
            summaries.append(
                {
                    "bootstrap_method_id": bootstrap_id,
                    "method_id": method_id,
                    "evidence_role": evidence_role,
                    "condition": condition,
                    "fold_count": 12,
                    "validation_review_clusters": len(row_uids),
                    "aspect_balanced_heldout_pair_micro_f1": balanced,
                    "aspect_balanced_review_bootstrap_95_ci_low": low,
                    "aspect_balanced_review_bootstrap_95_ci_high": high,
                    "pooled_heldout_pair_micro_f1_sensitivity": pooled,
                    "pooled_review_bootstrap_95_ci_low": pooled_low,
                    "pooled_review_bootstrap_95_ci_high": pooled_high,
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                }
            )
    d_index = CONDITIONS.index("D")
    reference_ids = (*inference.METHODS, *study_ids)
    for study_id in study_ids:
        left_index = methods.index(study_id)
        for reference_id in reference_ids:
            if reference_id == study_id:
                continue
            right_index = methods.index(reference_id)
            differences = (
                balanced_boot[:, left_index, d_index]
                - balanced_boot[:, right_index, d_index]
            )
            low, high = inference._interval(differences)
            pairwise.append(
                {
                    "condition": "D",
                    "left_bootstrap_method_id": study_id,
                    "right_bootstrap_method_id": reference_id,
                    "left_minus_right_aspect_balanced_heldout_pair_micro_f1": (
                        points[(study_id, "D")] - points[(reference_id, "D")]
                    ),
                    "shared_review_bootstrap_95_ci_low": low,
                    "shared_review_bootstrap_95_ci_high": high,
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                }
            )
    audit = {
        "validation_review_cluster_count": len(row_uids),
        "validation_row_uid_sha256": hashlib.sha256(
            ("\n".join(row_uids) + "\n").encode("utf-8")
        ).hexdigest(),
        "study_bootstrap_method_ids": study_ids,
    }
    return pd.DataFrame(summaries), pd.DataFrame(pairwise), audit


def _aggregate_fold_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "heldout_pair_micro_f1",
        "overall_pair_micro_f1",
        "seen_pair_micro_f1",
        "heldout_pair_precision",
        "heldout_pair_recall",
        "heldout_presence_f1",
        "heldout_presence_ap",
        "oracle_gated_sentiment_set_micro_f1",
        "aspect_call_rate",
    )
    records = []
    for identity, group in folds.groupby(
        ["method_id", "evidence_role", "condition"], sort=True
    ):
        method, role, condition = identity
        row: dict[str, Any] = {
            "method_id": method,
            "evidence_role": role,
            "condition": condition,
            "fold_count": int(group["fold_id"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
        records.append(row)
    return pd.DataFrame.from_records(records)


def _comparison_table(
    source: Path,
    study_aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    existing = pd.read_csv(source).copy()
    existing["evidence_role"] = "single_system_formal_validation"
    existing = existing.rename(
        columns={
            "heldout_presence_average_precision_mean": "heldout_presence_ap_mean"
        }
    )
    study = study_aggregate.copy()
    study["method"] = study["method_id"]
    boot = bootstrap[
        [
            "method_id",
            "evidence_role",
            "condition",
            "aspect_balanced_review_bootstrap_95_ci_low",
            "aspect_balanced_review_bootstrap_95_ci_high",
            "pooled_heldout_pair_micro_f1_sensitivity",
        ]
    ]
    study = study.merge(
        boot,
        on=["method_id", "evidence_role", "condition"],
        how="left",
        validate="one_to_one",
    )
    existing["aspect_balanced_review_bootstrap_95_ci_low"] = np.nan
    existing["aspect_balanced_review_bootstrap_95_ci_high"] = np.nan
    existing["pooled_heldout_pair_micro_f1_sensitivity"] = np.nan
    columns = sorted(set(existing.columns) | set(study.columns))
    return pd.concat(
        [existing.reindex(columns=columns), study.reindex(columns=columns)],
        ignore_index=True,
    ).sort_values(
        ["condition", "heldout_pair_micro_f1_mean", "evidence_role"],
        ascending=[True, False, True],
        kind="stable",
    )


def _calibrator_fit_record(
    model: Pipeline,
    *,
    fold_id: str,
    block_id: str,
    selection: Mapping[str, float | int],
    train_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    target_presence: np.ndarray,
    row_uid: np.ndarray,
) -> dict[str, Any]:
    scaler = model.named_steps["scale"]
    logistic = model.named_steps["logistic"]
    train_reviews = set(np.asarray(row_uid, dtype=str)[train_mask])
    evaluation_reviews = set(np.asarray(row_uid, dtype=str)[evaluation_mask])
    overlap = train_reviews & evaluation_reviews
    return {
        "fold_id": fold_id,
        "block_id": block_id,
        "training_aspect_instances": int(train_mask.sum()),
        "training_positive_instances": int(target_presence[train_mask].sum()),
        "training_review_count": len(train_reviews),
        "evaluation_review_count": len(evaluation_reviews),
        "train_evaluation_review_overlap_count": len(overlap),
        "heldout_aspect_training_instance_count": 0,
        "selected_threshold": float(selection["threshold"]),
        "selection_pair_micro_f1": float(selection["pair_micro_f1"]),
        "selection_presence_f1": float(selection["presence_f1"]),
        "selection_aspect_call_rate": float(selection["aspect_call_rate"]),
        "scaler_mean_absolute_logit": float(scaler.mean_[0]),
        "scaler_mean_relative_logit": float(scaler.mean_[1]),
        "scaler_scale_absolute_logit": float(scaler.scale_[0]),
        "scaler_scale_relative_logit": float(scaler.scale_[1]),
        "logistic_intercept": float(logistic.intercept_[0]),
        "logistic_coefficient_absolute_logit": float(logistic.coef_[0, 0]),
        "logistic_coefficient_relative_logit": float(logistic.coef_[0, 1]),
        "logistic_iterations": int(logistic.n_iter_[0]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = legacy.read_json(config_path)
    validate_config(config)
    calibration_config = config["direction_2_qlora_relative_calibrator"]
    thresholds = np.round(
        np.arange(
            float(calibration_config["threshold_grid_start"]),
            float(calibration_config["threshold_grid_stop"]) + 0.0001,
            float(calibration_config["threshold_grid_step"]),
        ),
        12,
    )
    if len(thresholds) != 101 or thresholds[0] != 0.0 or thresholds[-1] != 1.0:
        raise ValueError("The preregistered threshold grid is invalid.")

    backup_root = args.backup_root.resolve()
    output_dir = args.output_dir.resolve()
    public_dir = args.public_dir.resolve()
    states = legacy.audit_states(backup_root)
    receipt_summary, published = legacy.audit_receipts(backup_root)
    selections_audit = legacy.audit_selections(backup_root, published)
    selections = selections_audit["values"]
    entries = router._entry_map(backup_root)

    critical_paths: set[Path] = {config_path}
    for method in INPUT_METHODS:
        for fold in L2_FOLDS:
            selection = router._selection_for_fold(selections, method, fold)
            scope = str(selection["training_scope_id"])
            selection_candidates = list(
                backup_root.glob(f"worker-*/selections/{method}/{scope}.json")
            )
            if len(selection_candidates) != 1:
                raise ValueError(f"Selection path is not unique for {method}/{scope}.")
            selection_path = selection_candidates[0]
            for condition in CONDITIONS:
                worker_root, result_path, result = entries[(method, fold, condition)]
                critical_paths.update(
                    router._critical_score_paths(
                        worker_root, result_path, result, selection_path
                    )
                )
    critical_hashes: dict[str, str] = {}
    for path in sorted(critical_paths):
        resolved = path.resolve()
        if path != config_path and resolved not in published:
            raise ValueError(f"Study input lacks a verified receipt: {path}")
        observed = file_sha256(path)
        if path != config_path and published[resolved] != observed:
            raise ValueError(f"Study input receipt hash mismatch: {path}")
        critical_hashes[str(path)] = observed

    folds_by_id = {fold.fold_id: fold for fold in post_supervisor_l2_folds()}
    fold_records: list[dict[str, Any]] = []
    row_evidence_frames: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    repair_audit: dict[str, Any] = {}
    for fold_id in L2_FOLDS:
        taxonomy_fold = folds_by_id[fold_id]
        components: dict[str, dict[str, ComponentArrays]] = {
            condition: {
                method: router._component(
                    entries,
                    selections,
                    method=method,
                    fold=fold_id,
                    condition=condition,
                    repair_audit=repair_audit,
                )
                for method in INPUT_METHODS
            }
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            fewshot = components[condition][FEWSHOT]
            qlora = components[condition][QLORA]
            for method_id, role, stage_1, stage_2 in (
                (
                    PRIMARY_HYBRID,
                    "fixed_primary_stage_hybrid",
                    fewshot,
                    qlora,
                ),
                (
                    REVERSE_HYBRID,
                    "fixed_reverse_control",
                    qlora,
                    fewshot,
                ),
            ):
                frame = build_stage_hybrid_frame(
                    stage_1, stage_2, method_id=method_id
                )
                record, evidence = evaluate_frame(
                    frame,
                    fold_id=fold_id,
                    condition=condition,
                    method_id=method_id,
                    evidence_role=role,
                    taxonomy_fold=taxonomy_fold,
                )
                fold_records.append(record)
                row_evidence_frames.extend(evidence)

        raw_scores = {
            condition: router._load_frame(
                entries,
                method=QLORA,
                fold=fold_id,
                condition=condition,
                repair_audit=repair_audit,
            )
            for condition in CONDITIONS
        }
        qlora_components = {
            condition: components[condition][QLORA] for condition in CONDITIONS
        }
        features = {
            condition: relative_feature_matrix(
                raw_scores[condition],
                qlora_components[condition],
                epsilon=float(calibration_config["probability_clip_epsilon"]),
            )
            for condition in CONDITIONS
        }
        d_component = qlora_components["D"]
        for condition in CONDITIONS:
            validate_component_alignment(d_component, qlora_components[condition])
        target_presence = d_component.target.any(axis=1).astype(int)
        review_blocks = np.asarray(
            [
                review_block(
                    value,
                    seed=int(calibration_config["random_seed"]),
                    folds=int(calibration_config["review_crossfit_folds"]),
                )
                for value in d_component.row_uid
            ],
            dtype=int,
        )
        stage_2_mask = sentiment_only_mask(d_component)
        crossfit_aligned = {
            condition: np.full(len(d_component.row_uid), np.nan, dtype=float)
            for condition in CONDITIONS
        }
        for block_id in range(int(calibration_config["review_crossfit_folds"])):
            evaluation_mask = review_blocks == block_id
            train_mask = (~d_component.is_heldout) & (~evaluation_mask)
            model = fit_calibrator(
                features["D"],
                target_presence,
                train_mask,
                config=calibration_config,
            )
            train_probability = model.predict_proba(features["D"])[:, 1]
            selection = select_calibrated_threshold(
                train_probability,
                stage_2_mask,
                d_component.target,
                train_mask,
                thresholds,
            )
            fit_records.append(
                _calibrator_fit_record(
                    model,
                    fold_id=fold_id,
                    block_id=str(block_id),
                    selection=selection,
                    train_mask=train_mask,
                    evaluation_mask=evaluation_mask,
                    target_presence=target_presence,
                    row_uid=d_component.row_uid,
                )
            )
            for condition in CONDITIONS:
                probability = model.predict_proba(features[condition])[:, 1]
                crossfit_aligned[condition][evaluation_mask] = threshold_aligned_scores(
                    probability[evaluation_mask], float(selection["threshold"])
                )
        for condition in CONDITIONS:
            if not np.isfinite(crossfit_aligned[condition]).all():
                raise ValueError("Cross-fitted calibrator left unevaluated review rows.")
            frame = calibrated_score_frame(
                qlora_components[condition],
                crossfit_aligned[condition],
                method_id=CALIBRATOR,
            )
            record, evidence = evaluate_frame(
                frame,
                fold_id=fold_id,
                condition=condition,
                method_id=CALIBRATOR,
                evidence_role="review_cross_fitted_primary",
                taxonomy_fold=taxonomy_fold,
            )
            fold_records.append(record)
            row_evidence_frames.extend(evidence)

        final_train_mask = ~d_component.is_heldout
        final_model = fit_calibrator(
            features["D"],
            target_presence,
            final_train_mask,
            config=calibration_config,
        )
        final_train_probability = final_model.predict_proba(features["D"])[:, 1]
        final_selection = select_calibrated_threshold(
            final_train_probability,
            stage_2_mask,
            d_component.target,
            final_train_mask,
            thresholds,
        )
        fit_records.append(
            _calibrator_fit_record(
                final_model,
                fold_id=fold_id,
                block_id="all_validation_fit",
                selection=final_selection,
                train_mask=final_train_mask,
                evaluation_mask=np.ones_like(final_train_mask, dtype=bool),
                target_presence=target_presence,
                row_uid=d_component.row_uid,
            )
        )
        for condition in CONDITIONS:
            probability = final_model.predict_proba(features[condition])[:, 1]
            aligned = threshold_aligned_scores(
                probability, float(final_selection["threshold"])
            )
            frame = calibrated_score_frame(
                qlora_components[condition], aligned, method_id=CALIBRATOR
            )
            record, evidence = evaluate_frame(
                frame,
                fold_id=fold_id,
                condition=condition,
                method_id=CALIBRATOR,
                evidence_role="all_validation_fitted_diagnostic",
                taxonomy_fold=taxonomy_fold,
            )
            fold_records.append(record)
            row_evidence_frames.extend(evidence)

    fold_frame = pd.DataFrame.from_records(fold_records)
    row_evidence = pd.concat(row_evidence_frames, ignore_index=True)
    fit_frame = pd.DataFrame.from_records(fit_records)
    expected_fold_rows = 4 * len(CONDITIONS) * len(L2_FOLDS)
    if len(fold_frame) != expected_fold_rows:
        raise ValueError("Study fold table is incomplete.")
    crossfit_fit_frame = fit_frame[
        fit_frame["block_id"].ne("all_validation_fit")
    ]
    if (
        int(fold_frame["test_contract_count"].sum()) != 0
        or int(fold_frame["maximum_sentiments_per_selected_aspect"].max()) > 2
        or int(crossfit_fit_frame["train_evaluation_review_overlap_count"].max()) != 0
        or int(fit_frame["heldout_aspect_training_instance_count"].sum()) != 0
    ):
        raise ValueError("Study safety or capped-two audit failed.")
    duplicate_evidence_count = int(
        row_evidence.duplicated(
            [
                "method_id",
                "evidence_role",
                "partition",
                "fold_id",
                "condition",
                "row_uid",
            ]
        ).sum()
    )
    prediction_collapse_count = int(
        row_evidence[row_evidence["partition"].eq("heldout")]
        .groupby(["method_id", "evidence_role", "fold_id", "condition"], sort=False)[
            "pair_predicted_count"
        ]
        .sum()
        .eq(0)
        .sum()
    )
    if duplicate_evidence_count or prediction_collapse_count:
        raise ValueError("Study identity or prediction-collapse audit failed.")

    aggregate = _aggregate_fold_metrics(fold_frame)
    bootstrap, pairwise, bootstrap_audit = _bootstrap_tables(
        row_evidence,
        formal_evidence_root=args.formal_evidence_root.resolve(),
        local_evidence_root=args.local_evidence_root.resolve(),
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    comparison = _comparison_table(
        args.comparison_source.resolve(), aggregate, bootstrap
    )
    numeric_frames = (fold_frame, row_evidence, fit_frame, aggregate, bootstrap, pairwise)
    non_finite_count = sum(non_finite_numeric_count(frame) for frame in numeric_frames)
    if non_finite_count:
        raise ValueError("Study outputs contain non-finite numeric values.")

    primary_bootstrap_id = f"{PRIMARY_HYBRID}__fixed_primary_stage_hybrid"
    reverse_bootstrap_id = f"{REVERSE_HYBRID}__fixed_reverse_control"
    calibrator_bootstrap_id = f"{CALIBRATOR}__review_cross_fitted_primary"
    pairwise_lookup = pairwise.set_index(
        ["left_bootstrap_method_id", "right_bootstrap_method_id"]
    )
    stage_hybrid_pass = bool(
        pairwise_lookup.loc[
            (primary_bootstrap_id, FEWSHOT),
            "left_minus_right_aspect_balanced_heldout_pair_micro_f1",
        ]
        > 0
        and pairwise_lookup.loc[
            (primary_bootstrap_id, QLORA),
            "left_minus_right_aspect_balanced_heldout_pair_micro_f1",
        ]
        > 0
    )
    calibrator_pass = bool(
        pairwise_lookup.loc[
            (calibrator_bootstrap_id, QLORA),
            "left_minus_right_aspect_balanced_heldout_pair_micro_f1",
        ]
        > 0
    )

    output_files = {
        "study_fold_metrics.csv": fold_frame,
        "study_row_evidence.csv": row_evidence,
        "calibrator_fit_records.csv": fit_frame,
        "study_aggregate_metrics.csv": aggregate,
        "study_review_bootstrap.csv": bootstrap,
        "study_pairwise_review_bootstrap.csv": pairwise,
        "method_comparison.csv": comparison,
    }
    output_hashes: dict[str, str] = {}
    for name, frame in output_files.items():
        path = output_dir / name
        atomic_csv(path, frame)
        output_hashes[str(path)] = file_sha256(path)
    for name, frame in (
        ("stage_hybrid_calibration_method_comparison.csv", comparison),
        ("stage_hybrid_calibration_fold_metrics.csv", fold_frame),
        ("stage_hybrid_calibration_review_bootstrap.csv", bootstrap),
        ("stage_hybrid_calibration_pairwise_bootstrap.csv", pairwise),
        ("stage_hybrid_calibration_fit_summary.csv", fit_frame),
    ):
        path = public_dir / name
        atomic_csv(path, frame)
        output_hashes[str(path)] = file_sha256(path)

    study_selection = {
        "schema_version": "taxonomy_two_stage_stage_hybrid_calibration_selected_v1",
        "study_id": STUDY_ID,
        "protocol_id": legacy.PROTOCOL_ID,
        "evaluation_partition": "validation_only",
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "primary_stage_hybrid_method_id": PRIMARY_HYBRID,
        "reverse_control_method_id": REVERSE_HYBRID,
        "relative_calibrator_method_id": CALIBRATOR,
        "relative_calibrator_primary_evidence_role": "review_cross_fitted_primary",
        "D_selection_transferred_unchanged_to_N": True,
        "stage_hybrid_point_promotion_rule_passed": stage_hybrid_pass,
        "relative_calibrator_point_promotion_rule_passed": calibrator_pass,
        "config_sha256": file_sha256(config_path),
        "bootstrap_method_ids": {
            "primary_hybrid": primary_bootstrap_id,
            "reverse_control": reverse_bootstrap_id,
            "relative_calibrator": calibrator_bootstrap_id,
        },
    }
    study_selection["selected_payload_sha256"] = canonical_sha256(study_selection)
    selected_path = output_dir / "selected_study.json"
    atomic_json(selected_path, study_selection)
    output_hashes[str(selected_path)] = file_sha256(selected_path)

    audit = {
        "schema_version": "taxonomy_two_stage_stage_hybrid_calibration_audit_v1",
        "study_id": STUDY_ID,
        "protocol_id": legacy.PROTOCOL_ID,
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "config_sha256": file_sha256(config_path),
        "campaign_states": states,
        "receipt_summary": receipt_summary,
        "selection_count": selections_audit["selection_count"],
        "critical_verified_input_file_count": len(critical_hashes),
        "critical_verified_input_sha256": dict(sorted(critical_hashes.items())),
        "fold_metric_record_count": len(fold_frame),
        "row_evidence_record_count": len(row_evidence),
        "calibrator_fit_record_count": len(fit_frame),
        "calibrator_review_crossfit_block_count": int(
            fit_frame["block_id"].ne("all_validation_fit").sum()
        ),
        "crossfit_train_evaluation_review_overlap_count": int(
            crossfit_fit_frame["train_evaluation_review_overlap_count"].sum()
        ),
        "all_validation_fit_expected_review_overlap_count": int(
            fit_frame.loc[
                fit_frame["block_id"].eq("all_validation_fit"),
                "train_evaluation_review_overlap_count",
            ].sum()
        ),
        "heldout_aspect_training_instance_count": int(
            fit_frame["heldout_aspect_training_instance_count"].sum()
        ),
        "maximum_sentiments_per_selected_aspect": int(
            fold_frame["maximum_sentiments_per_selected_aspect"].max()
        ),
        "non_finite_value_count": non_finite_count,
        "duplicate_row_evidence_identity_count": duplicate_evidence_count,
        "prediction_collapse_count": prediction_collapse_count,
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "synchronisation": "same review resample for all existing and study systems",
            "evidence": bootstrap_audit,
        },
        "qlora_seen_invariant_repairs": repair_audit,
        "output_sha256": dict(sorted(output_hashes.items())),
        "selected_study_sha256": study_selection["selected_payload_sha256"],
        "stage_hybrid_point_promotion_rule_passed": stage_hybrid_pass,
        "relative_calibrator_point_promotion_rule_passed": calibrator_pass,
    }
    audit["audit_payload_sha256"] = canonical_sha256(audit)
    audit_path = output_dir / "audit.json"
    atomic_json(audit_path, audit)
    summary = {
        "selected_study": study_selection,
        "audit_path": str(audit_path),
        "comparison_path": str(public_dir / "stage_hybrid_calibration_method_comparison.csv"),
        "output_dir": str(output_dir),
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_stage_hybrid_calibration_v1.json",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path(
            "C:/Msc_DSML/Msc_Project/cloud_backups/taxonomy_two_stage_formal_v2_r2"
        ),
    )
    parser.add_argument(
        "--comparison-source",
        type=Path,
        default=PROJECT_ROOT
        / "docs/thesis_figure_data/taxonomy_scientific_freeze_v1/l2_complete_model_condition_metrics.csv",
    )
    parser.add_argument(
        "--formal-evidence-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_scientific_freeze_v1/formal_row_evidence",
    )
    parser.add_argument(
        "--local-evidence-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_scientific_freeze_v1/local_row_evidence",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_two_stage_stage_hybrid_calibration_v1",
    )
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=PROJECT_ROOT
        / "docs/thesis_figure_data/taxonomy_two_stage_stage_hybrid_calibration_v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["selected_study"], indent=2, sort_keys=True), flush=True)
