"""Evaluate the preregistered smooth Stage-1 fusion on validation only."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import analyse_taxonomy_post_supervisor_formal_v2 as legacy
import analyse_taxonomy_two_stage_global_router as router
import analyse_taxonomy_two_stage_stage_hybrid_calibration as hybrid

from msc_project.experiments.taxonomy_global_router import (
    ComponentArrays,
    threshold_aligned_scores,
    validate_component_alignment,
)
from msc_project.experiments.taxonomy_post_supervisor import post_supervisor_l2_folds


STUDY_ID = "taxonomy_two_stage_no_retraining_extensions_v1"
METHOD_ID = "smooth_fewshot_qlora_stage1_fusion__qlora_stage2"
REFERENCE_ID = "frozen_qwen_few_shot_stage1__qlora_stage2"
FEWSHOT = "frozen_qwen_few_shot"
QLORA = "qwen_candidate_pair_qlora"
CONDITIONS = ("D", "N")
FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))


def validate_config(config: Mapping[str, Any]) -> None:
    section = config.get("smooth_stage_1_fusion", {})
    fixed = config.get("fixed_stage_2", {})
    if (
        config.get("schema_version") != STUDY_ID
        or config.get("protocol_id") != legacy.PROTOCOL_ID
        or config.get("evaluation_partition") != "validation_only"
        or config.get("allowed_splits") != ["train", "validation"]
        or config.get("official_test_permitted") is not False
        or config.get("include_official_test") is not False
        or int(config.get("test_contract_count", -1)) != 0
        or tuple(config.get("folds", ())) != FOLDS
        or tuple(config.get("conditions", ())) != CONDITIONS
        or section.get("method_id") != METHOD_ID
        or tuple(section.get("sources", ())) != (FEWSHOT, QLORA)
        or int(section.get("review_crossfit_folds", -1)) != 5
        or section.get("selection_condition") != "D_only"
        or fixed.get("method_id") != QLORA
        or fixed.get("retraining") is not False
        or fixed.get("retuning") is not False
    ):
        raise ValueError("Smooth-fusion configuration violates the frozen boundary.")


def _grid(section: Mapping[str, Any], name: str) -> np.ndarray:
    value = section[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing grid {name}.")
    grid = np.round(
        np.arange(
            float(value["start"]),
            float(value["stop"]) + 0.0001,
            float(value["step"]),
        ),
        12,
    )
    if grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError(f"Grid {name} must span [0, 1].")
    return grid


def _raw_aspect_probabilities(
    scored: pd.DataFrame, component: ComponentArrays
) -> np.ndarray:
    required = {"row_uid", "candidate_aspect", "aspect_score"}
    missing = sorted(required - set(scored.columns))
    if missing or scored.empty:
        raise ValueError(f"Raw score frame is invalid; missing={missing}.")
    grouped = (
        scored.assign(
            row_uid=scored["row_uid"].astype(str),
            candidate_aspect=scored["candidate_aspect"].astype(str),
        )
        .groupby(["row_uid", "candidate_aspect"], sort=True)
        .agg(
            aspect_score=("aspect_score", "first"),
            aspect_score_min=("aspect_score", "min"),
            aspect_score_max=("aspect_score", "max"),
        )
        .reset_index()
    )
    if not np.allclose(
        grouped["aspect_score_min"], grouped["aspect_score_max"], atol=0.0, rtol=0.0
    ):
        raise ValueError("Aspect score differs across sentiments.")
    expected = np.column_stack((component.row_uid, component.candidate_aspect)).astype(str)
    observed = grouped[["row_uid", "candidate_aspect"]].to_numpy(dtype=str)
    if not np.array_equal(expected, observed):
        raise ValueError("Raw and threshold-aligned aspect identities differ.")
    probability = grouped["aspect_score"].to_numpy(dtype=float)
    if (
        probability.shape != component.aspect_score.shape
        or not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise ValueError("Raw aspect probabilities are invalid.")
    return probability


def _logit(values: np.ndarray | float, epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def source_evidence(
    probability: np.ndarray, threshold: float, *, epsilon: float
) -> np.ndarray:
    value = _logit(probability, epsilon) - _logit(float(threshold), epsilon)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Source evidence is invalid.")
    return value


def fused_probability(
    fewshot_evidence: np.ndarray, qlora_evidence: np.ndarray, alpha: float
) -> np.ndarray:
    if fewshot_evidence.shape != qlora_evidence.shape:
        raise ValueError("Fusion sources are not aligned.")
    evidence = float(alpha) * fewshot_evidence + (1.0 - float(alpha)) * qlora_evidence
    probability = np.empty_like(evidence, dtype=float)
    positive = evidence >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-evidence[positive]))
    exp_value = np.exp(evidence[~positive])
    probability[~positive] = exp_value / (1.0 + exp_value)
    if not np.isfinite(probability).all():
        raise ValueError("Fusion produced a non-finite probability.")
    return probability


def select_policy(
    fewshot_evidence: np.ndarray,
    qlora_evidence: np.ndarray,
    sentiment_mask: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    alphas: Sequence[float],
    thresholds: Sequence[float],
) -> dict[str, float | int]:
    records: list[dict[str, float | int]] = []
    for alpha in alphas:
        probability = fused_probability(fewshot_evidence, qlora_evidence, float(alpha))
        for threshold in thresholds:
            row = hybrid.threshold_metrics(
                probability,
                sentiment_mask,
                target,
                eligible,
                float(threshold),
            )
            row["alpha"] = float(alpha)
            records.append(row)
    return max(
        records,
        key=lambda row: (
            float(row["pair_micro_f1"]),
            float(row["presence_f1"]),
            -float(row["aspect_call_rate"]),
            -abs(float(row["alpha"]) - 1.0),
            -abs(float(row["threshold"]) - 0.5),
            float(row["threshold"]),
        ),
    )


def _fit_record(
    *,
    fold_id: str,
    block_id: str,
    selection: Mapping[str, float | int],
    train_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    component: ComponentArrays,
) -> dict[str, Any]:
    train_reviews = set(component.row_uid[train_mask].astype(str))
    evaluation_reviews = set(component.row_uid[evaluation_mask].astype(str))
    return {
        "fold_id": fold_id,
        "block_id": block_id,
        "selected_alpha": float(selection["alpha"]),
        "selected_threshold": float(selection["threshold"]),
        "selection_pair_micro_f1": float(selection["pair_micro_f1"]),
        "selection_presence_f1": float(selection["presence_f1"]),
        "selection_aspect_call_rate": float(selection["aspect_call_rate"]),
        "training_aspect_instances": int(train_mask.sum()),
        "evaluation_aspect_instances": int(evaluation_mask.sum()),
        "training_review_count": len(train_reviews),
        "evaluation_review_count": len(evaluation_reviews),
        "train_evaluation_review_overlap_count": len(train_reviews & evaluation_reviews),
        "heldout_aspect_selection_instance_count": int(
            np.sum(component.is_heldout & train_mask)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = legacy.read_json(config_path)
    validate_config(config)
    section = config["smooth_stage_1_fusion"]
    alphas = _grid(section, "alpha_grid")
    thresholds = _grid(section, "aspect_threshold_grid")
    epsilon = float(section["probability_clip_epsilon"])

    backup_root = args.backup_root.resolve()
    output_dir = args.output_dir.resolve()
    public_dir = args.public_dir.resolve()
    states = legacy.audit_states(backup_root)
    receipt_summary, published = legacy.audit_receipts(backup_root)
    selections_audit = legacy.audit_selections(backup_root, published)
    selections = selections_audit["values"]
    entries = router._entry_map(backup_root)

    critical_paths: set[Path] = {config_path}
    for method in (FEWSHOT, QLORA):
        for fold in FOLDS:
            selected = router._selection_for_fold(selections, method, fold)
            scope = str(selected["training_scope_id"])
            candidates = list(backup_root.glob(f"worker-*/selections/{method}/{scope}.json"))
            if len(candidates) != 1:
                raise ValueError(f"Selection path is not unique for {method}/{scope}.")
            for condition in CONDITIONS:
                worker_root, result_path, result = entries[(method, fold, condition)]
                critical_paths.update(
                    router._critical_score_paths(
                        worker_root, result_path, result, candidates[0]
                    )
                )
    critical_hashes: dict[str, str] = {}
    for path in sorted(critical_paths):
        resolved = path.resolve()
        if path != config_path and resolved not in published:
            raise ValueError(f"Fusion input lacks a verified receipt: {path}")
        observed = hybrid.file_sha256(path)
        if path != config_path and published[resolved] != observed:
            raise ValueError(f"Fusion input receipt hash mismatch: {path}")
        critical_hashes[str(path)] = observed

    folds_by_id = {value.fold_id: value for value in post_supervisor_l2_folds()}
    fold_records: list[dict[str, Any]] = []
    row_frames: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    repair_audit: dict[str, Any] = {}
    for fold_id in FOLDS:
        taxonomy_fold = folds_by_id[fold_id]
        components = {
            condition: {
                method: router._component(
                    entries,
                    selections,
                    method=method,
                    fold=fold_id,
                    condition=condition,
                    repair_audit=repair_audit,
                )
                for method in (FEWSHOT, QLORA)
            }
            for condition in CONDITIONS
        }
        raw = {
            condition: {
                method: router._load_frame(
                    entries,
                    method=method,
                    fold=fold_id,
                    condition=condition,
                    repair_audit=repair_audit,
                )
                for method in (FEWSHOT, QLORA)
            }
            for condition in CONDITIONS
        }
        source_thresholds = {
            method: float(
                router._selection_for_fold(selections, method, fold_id)[
                    "selected_thresholds"
                ]["aspect"]
            )
            for method in (FEWSHOT, QLORA)
        }
        evidence: dict[str, dict[str, np.ndarray]] = {}
        for condition in CONDITIONS:
            fewshot = components[condition][FEWSHOT]
            qlora = components[condition][QLORA]
            validate_component_alignment(fewshot, qlora)
            evidence[condition] = {
                FEWSHOT: source_evidence(
                    _raw_aspect_probabilities(raw[condition][FEWSHOT], fewshot),
                    source_thresholds[FEWSHOT],
                    epsilon=epsilon,
                ),
                QLORA: source_evidence(
                    _raw_aspect_probabilities(raw[condition][QLORA], qlora),
                    source_thresholds[QLORA],
                    epsilon=epsilon,
                ),
            }

        d_component = components["D"][QLORA]
        sentiment_mask = hybrid.sentiment_only_mask(d_component)
        blocks = np.asarray(
            [
                hybrid.review_block(
                    uid,
                    seed=int(section["review_crossfit_seed"]),
                    folds=int(section["review_crossfit_folds"]),
                )
                for uid in d_component.row_uid
            ],
            dtype=int,
        )
        crossfit_aligned = {
            condition: np.full(len(d_component.row_uid), np.nan, dtype=float)
            for condition in CONDITIONS
        }
        for block in range(int(section["review_crossfit_folds"])):
            evaluation_mask = blocks == block
            train_mask = (~d_component.is_heldout) & (~evaluation_mask)
            selected = select_policy(
                evidence["D"][FEWSHOT],
                evidence["D"][QLORA],
                sentiment_mask,
                d_component.target,
                train_mask,
                alphas,
                thresholds,
            )
            fit_records.append(
                _fit_record(
                    fold_id=fold_id,
                    block_id=str(block),
                    selection=selected,
                    train_mask=train_mask,
                    evaluation_mask=evaluation_mask,
                    component=d_component,
                )
            )
            for condition in CONDITIONS:
                probability = fused_probability(
                    evidence[condition][FEWSHOT],
                    evidence[condition][QLORA],
                    float(selected["alpha"]),
                )
                crossfit_aligned[condition][evaluation_mask] = threshold_aligned_scores(
                    probability[evaluation_mask], float(selected["threshold"])
                )
        for condition in CONDITIONS:
            if not np.isfinite(crossfit_aligned[condition]).all():
                raise ValueError("Cross-fitted fusion left unevaluated instances.")
            frame = hybrid.calibrated_score_frame(
                components[condition][QLORA],
                crossfit_aligned[condition],
                method_id=METHOD_ID,
            )
            record, evidence_frames = hybrid.evaluate_frame(
                frame,
                fold_id=fold_id,
                condition=condition,
                method_id=METHOD_ID,
                evidence_role="review_cross_fitted_primary",
                taxonomy_fold=taxonomy_fold,
            )
            fold_records.append(record)
            row_frames.extend(evidence_frames)

        all_mask = ~d_component.is_heldout
        selected = select_policy(
            evidence["D"][FEWSHOT],
            evidence["D"][QLORA],
            sentiment_mask,
            d_component.target,
            all_mask,
            alphas,
            thresholds,
        )
        fit_records.append(
            _fit_record(
                fold_id=fold_id,
                block_id="all_validation_fit",
                selection=selected,
                train_mask=all_mask,
                evaluation_mask=np.ones_like(all_mask, dtype=bool),
                component=d_component,
            )
        )
        for condition in CONDITIONS:
            probability = fused_probability(
                evidence[condition][FEWSHOT],
                evidence[condition][QLORA],
                float(selected["alpha"]),
            )
            aligned = threshold_aligned_scores(probability, float(selected["threshold"]))
            frame = hybrid.calibrated_score_frame(
                components[condition][QLORA], aligned, method_id=METHOD_ID
            )
            record, evidence_frames = hybrid.evaluate_frame(
                frame,
                fold_id=fold_id,
                condition=condition,
                method_id=METHOD_ID,
                evidence_role="all_validation_fitted_diagnostic",
                taxonomy_fold=taxonomy_fold,
            )
            fold_records.append(record)
            row_frames.extend(evidence_frames)

        for condition in CONDITIONS:
            frame = hybrid.build_stage_hybrid_frame(
                components[condition][FEWSHOT],
                components[condition][QLORA],
                method_id=REFERENCE_ID,
            )
            record, evidence_frames = hybrid.evaluate_frame(
                frame,
                fold_id=fold_id,
                condition=condition,
                method_id=REFERENCE_ID,
                evidence_role="fixed_reference_reproduction",
                taxonomy_fold=taxonomy_fold,
            )
            fold_records.append(record)
            row_frames.extend(evidence_frames)

    fold_frame = pd.DataFrame.from_records(fold_records)
    row_evidence = pd.concat(row_frames, ignore_index=True)
    fit_frame = pd.DataFrame.from_records(fit_records)
    if len(fold_frame) != 3 * len(CONDITIONS) * len(FOLDS):
        raise ValueError("Smooth-fusion fold table is incomplete.")
    crossfit_fits = fit_frame[fit_frame["block_id"].ne("all_validation_fit")]
    if (
        int(crossfit_fits["train_evaluation_review_overlap_count"].sum()) != 0
        or int(fit_frame["heldout_aspect_selection_instance_count"].sum()) != 0
        or int(fold_frame["test_contract_count"].sum()) != 0
        or int(fold_frame["maximum_sentiments_per_selected_aspect"].max()) > 2
    ):
        raise ValueError("Fusion leakage or decoder safety audit failed.")
    duplicate_count = int(
        row_evidence.duplicated(
            ["method_id", "evidence_role", "partition", "fold_id", "condition", "row_uid"]
        ).sum()
    )
    collapse_count = int(
        row_evidence[row_evidence["partition"].eq("heldout")]
        .groupby(["method_id", "evidence_role", "fold_id", "condition"])[
            "pair_predicted_count"
        ]
        .sum()
        .eq(0)
        .sum()
    )
    if duplicate_count or collapse_count:
        raise ValueError("Fusion identity or prediction-collapse audit failed.")

    aggregate = hybrid._aggregate_fold_metrics(fold_frame)
    bootstrap, pairwise, bootstrap_audit = hybrid._bootstrap_tables(
        row_evidence,
        formal_evidence_root=args.formal_evidence_root.resolve(),
        local_evidence_root=args.local_evidence_root.resolve(),
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    comparison = hybrid._comparison_table(
        args.comparison_source.resolve(), aggregate, bootstrap
    )
    numeric = (fold_frame, row_evidence, fit_frame, aggregate, bootstrap, pairwise)
    non_finite = sum(hybrid.non_finite_numeric_count(frame) for frame in numeric)
    if non_finite:
        raise ValueError("Fusion output contains non-finite numeric values.")

    primary_id = f"{METHOD_ID}__review_cross_fitted_primary"
    reference_id = f"{REFERENCE_ID}__fixed_reference_reproduction"
    comparison_row = pairwise[
        pairwise["left_bootstrap_method_id"].eq(primary_id)
        & pairwise["right_bootstrap_method_id"].eq(reference_id)
    ]
    if len(comparison_row) != 1:
        raise ValueError("Fusion-reference paired comparison is missing.")
    gain = float(
        comparison_row.iloc[0][
            "left_minus_right_aspect_balanced_heldout_pair_micro_f1"
        ]
    )
    ci_low = float(comparison_row.iloc[0]["shared_review_bootstrap_95_ci_low"])
    promoted = bool(gain > 0.0 and ci_low > 0.0)

    output_files = {
        "smooth_fusion_fold_metrics.csv": fold_frame,
        "smooth_fusion_row_evidence.csv": row_evidence,
        "smooth_fusion_fit_records.csv": fit_frame,
        "smooth_fusion_aggregate_metrics.csv": aggregate,
        "smooth_fusion_review_bootstrap.csv": bootstrap,
        "smooth_fusion_pairwise_bootstrap.csv": pairwise,
        "smooth_fusion_method_comparison.csv": comparison,
    }
    output_hashes: dict[str, str] = {}
    for name, frame in output_files.items():
        path = output_dir / name
        hybrid.atomic_csv(path, frame)
        output_hashes[str(path)] = hybrid.file_sha256(path)
    for name, frame in (
        ("smooth_fusion_method_comparison.csv", comparison),
        ("smooth_fusion_fold_metrics.csv", fold_frame),
        ("smooth_fusion_review_bootstrap.csv", bootstrap),
        ("smooth_fusion_pairwise_bootstrap.csv", pairwise),
        ("smooth_fusion_fit_summary.csv", fit_frame),
    ):
        path = public_dir / name
        hybrid.atomic_csv(path, frame)
        output_hashes[str(path)] = hybrid.file_sha256(path)

    selected = {
        "schema_version": "taxonomy_two_stage_smooth_fusion_selected_v1",
        "study_id": STUDY_ID,
        "method_id": METHOD_ID,
        "evidence_role": "review_cross_fitted_primary",
        "evaluation_partition": "validation_only",
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "D_selection_transferred_unchanged_to_N": True,
        "reference_method_id": REFERENCE_ID,
        "D_aspect_balanced_heldout_pair_f1_gain_vs_reference": gain,
        "paired_review_bootstrap_95_ci_low": ci_low,
        "paired_review_bootstrap_95_ci_high": float(
            comparison_row.iloc[0]["shared_review_bootstrap_95_ci_high"]
        ),
        "promotion_rule_passed": promoted,
        "config_sha256": hybrid.file_sha256(config_path),
    }
    selected["selected_payload_sha256"] = hybrid.canonical_sha256(selected)
    selected_path = output_dir / "smooth_fusion_selected.json"
    hybrid.atomic_json(selected_path, selected)
    output_hashes[str(selected_path)] = hybrid.file_sha256(selected_path)

    audit = {
        "schema_version": "taxonomy_two_stage_smooth_fusion_audit_v1",
        "study_id": STUDY_ID,
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "campaign_states": states,
        "receipt_summary": receipt_summary,
        "selection_count": selections_audit["selection_count"],
        "critical_verified_input_file_count": len(critical_hashes),
        "critical_verified_input_sha256": dict(sorted(critical_hashes.items())),
        "crossfit_fit_count": int(len(crossfit_fits)),
        "crossfit_train_evaluation_review_overlap_count": int(
            crossfit_fits["train_evaluation_review_overlap_count"].sum()
        ),
        "heldout_aspect_selection_instance_count": int(
            fit_frame["heldout_aspect_selection_instance_count"].sum()
        ),
        "maximum_sentiments_per_selected_aspect": int(
            fold_frame["maximum_sentiments_per_selected_aspect"].max()
        ),
        "duplicate_row_evidence_identity_count": duplicate_count,
        "prediction_collapse_count": collapse_count,
        "non_finite_value_count": non_finite,
        "bootstrap": bootstrap_audit,
        "qlora_seen_invariant_repairs": repair_audit,
        "output_sha256": dict(sorted(output_hashes.items())),
        "promotion_rule_passed": promoted,
    }
    audit["audit_payload_sha256"] = hybrid.canonical_sha256(audit)
    audit_path = output_dir / "smooth_fusion_audit.json"
    hybrid.atomic_json(audit_path, audit)
    summary = {
        "selected": selected,
        "audit_path": str(audit_path),
        "comparison_path": str(public_dir / "smooth_fusion_method_comparison.csv"),
    }
    hybrid.atomic_json(output_dir / "smooth_fusion_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_no_retraining_extensions_v1.json",
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
        / "outputs/experimental/taxonomy_two_stage_no_retraining_extensions_v1",
    )
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=PROJECT_ROOT
        / "docs/thesis_figure_data/taxonomy_two_stage_no_retraining_extensions_v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["selected"], indent=2, sort_keys=True), flush=True)
