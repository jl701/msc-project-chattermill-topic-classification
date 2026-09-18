"""Analyse the preregistered retrieval few-shot Stage 1 on validation only."""

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
import run_taxonomy_two_stage_retrieval_few_shot_validation as runner

from msc_project.experiments.taxonomy_global_router import (
    ComponentArrays,
    threshold_aligned_scores,
    validate_component_alignment,
)
from msc_project.experiments.taxonomy_post_supervisor import post_supervisor_l2_folds


STUDY_ID = runner.STUDY_ID
METHOD_ID = runner.METHOD_ID
REFERENCE_ID = "frozen_qwen_few_shot_stage1__qlora_stage2"
FEWSHOT = "frozen_qwen_few_shot"
QLORA = "qwen_candidate_pair_qlora"
CONDITIONS = runner.CONDITIONS
FOLDS = runner.FOLDS


def validate_config(config: Mapping[str, Any]) -> None:
    runner.validate_config(config)
    section = config["retrieval_few_shot_stage_1"]
    fixed = config["fixed_stage_2"]
    if (
        int(section.get("review_crossfit_folds", -1)) != 5
        or section.get("selection_condition") != "D_only"
        or section.get("selection_aspects") != "same_outer_fold_seen_aspects_only"
        or fixed.get("method_id") != QLORA
        or fixed.get("retraining") is not False
        or fixed.get("retuning") is not False
    ):
        raise ValueError("Retrieval analysis configuration violates the boundary.")


def _thresholds(section: Mapping[str, Any]) -> np.ndarray:
    value = section["aspect_threshold_grid"]
    grid = np.round(
        np.arange(
            float(value["start"]),
            float(value["stop"]) + 0.0001,
            float(value["step"]),
        ),
        12,
    )
    if len(grid) != 101 or grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("Retrieval threshold grid changed.")
    return grid


def retrieval_probability(
    scored: pd.DataFrame, component: ComponentArrays
) -> np.ndarray:
    required = {"row_uid", "candidate_aspect", "target", "aspect_score"}
    missing = sorted(required - set(scored.columns))
    if missing or scored.empty:
        raise ValueError(f"Retrieval score frame is invalid; missing={missing}.")
    ordered = scored.assign(
        row_uid=scored["row_uid"].astype(str),
        candidate_aspect=scored["candidate_aspect"].astype(str),
    ).sort_values(["row_uid", "candidate_aspect"], kind="stable")
    if ordered.duplicated(["row_uid", "candidate_aspect"]).any():
        raise ValueError("Retrieval score identities are duplicated.")
    expected = np.column_stack((component.row_uid, component.candidate_aspect)).astype(str)
    observed = ordered[["row_uid", "candidate_aspect"]].to_numpy(dtype=str)
    if not np.array_equal(expected, observed):
        raise ValueError("Retrieval and QLoRA candidate identities differ.")
    probability = ordered["aspect_score"].to_numpy(dtype=float)
    target = ordered["target"].to_numpy(dtype=int)
    if not np.array_equal(target.astype(bool), component.target.any(axis=1)):
        raise ValueError("Retrieval target labels differ from formal evidence.")
    if (
        probability.shape != component.aspect_score.shape
        or not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise ValueError("Retrieval probabilities are invalid.")
    return probability


def select_threshold(
    probabilities: np.ndarray,
    sentiment_mask: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, float | int]:
    records = [
        hybrid.threshold_metrics(
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
    training_reviews = set(component.row_uid[train_mask].astype(str))
    evaluation_reviews = set(component.row_uid[evaluation_mask].astype(str))
    return {
        "fold_id": fold_id,
        "block_id": block_id,
        "selected_threshold": float(selection["threshold"]),
        "selection_pair_micro_f1": float(selection["pair_micro_f1"]),
        "selection_presence_f1": float(selection["presence_f1"]),
        "selection_aspect_call_rate": float(selection["aspect_call_rate"]),
        "training_aspect_instances": int(train_mask.sum()),
        "evaluation_aspect_instances": int(evaluation_mask.sum()),
        "training_review_count": len(training_reviews),
        "evaluation_review_count": len(evaluation_reviews),
        "train_evaluation_review_overlap_count": len(
            training_reviews & evaluation_reviews
        ),
        "heldout_aspect_selection_instance_count": int(
            np.sum(component.is_heldout & train_mask)
        ),
    }


def _audit_retrieval_inputs(
    score_root: Path, config_sha256: str
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, Any], dict[str, str]]:
    scores: dict[tuple[str, str], pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    heldout_pool_count = 0
    similarity_target_read_count = 0
    test_contract_count = 0
    failure_count = 0
    non_finite_count = 0
    collapse_count = 0
    demo_train_split_violation = 0
    demo_heldout_aspect_count = 0
    within_answer_row_duplicate_count = 0
    d_n_seen_demo_mismatch_count = 0
    folds_by_id = {fold.fold_id: fold for fold in post_supervisor_l2_folds()}
    for fold_id in FOLDS:
        fold = folds_by_id[fold_id]
        manifests: dict[str, pd.DataFrame] = {}
        for condition in CONDITIONS:
            manifest_path = score_root / "retrieval_manifests" / fold_id / f"{condition}.csv"
            manifest_meta_path = manifest_path.with_suffix(".manifest.json")
            score_path = score_root / "retrieval_scores" / fold_id / f"{condition}.csv"
            score_meta_path = score_path.with_suffix(".manifest.json")
            contract_path = score_path.with_suffix(".contract.json")
            for path in (manifest_path, manifest_meta_path, score_path, score_meta_path, contract_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete retrieval artifact unit: {path}")
                hashes[str(path)] = hybrid.file_sha256(path)
            manifest_meta = legacy.read_json(manifest_meta_path)
            score_meta = legacy.read_json(score_meta_path)
            contract = legacy.read_json(contract_path)
            if (
                manifest_meta.get("config_sha256") != config_sha256
                or score_meta.get("config_sha256") != config_sha256
                or contract.get("config_sha256") != config_sha256
                or manifest_meta.get("manifest_sha256") != hybrid.file_sha256(manifest_path)
                or score_meta.get("score_sha256") != hybrid.file_sha256(score_path)
                or score_meta.get("status") != "complete"
            ):
                raise RuntimeError("Retrieval artifact contract or hash mismatch.")
            heldout_pool_count += int(manifest_meta["heldout_aspect_pool_instance_count"])
            similarity_target_read_count += int(manifest_meta["retrieval_similarity_target_read_count"])
            test_contract_count += int(manifest_meta["test_contract_count"])
            test_contract_count += int(score_meta["test_contract_count"])
            failure_count += int(score_meta["failure_count"])
            non_finite_count += int(score_meta["non_finite_score_count"])
            collapse_count += int(int(score_meta["prediction_score_unique_count"]) < 2)
            manifest = pd.read_csv(manifest_path)
            manifests[condition] = manifest
            for position in range(1, 5):
                demo_train_split_violation += int(
                    (~manifest[f"demo_{position}_row_uid"].astype(str).str.startswith("train:")).sum()
                )
                demo_heldout_aspect_count += int(
                    manifest[f"demo_{position}_aspect"].astype(str).isin(fold.heldout_aspects).sum()
                )
            within_answer_row_duplicate_count += int(
                manifest["demo_1_row_uid"].astype(str).eq(
                    manifest["demo_2_row_uid"].astype(str)
                ).sum()
            )
            within_answer_row_duplicate_count += int(
                manifest["demo_3_row_uid"].astype(str).eq(
                    manifest["demo_4_row_uid"].astype(str)
                ).sum()
            )
            scores[(fold_id, condition)] = pd.read_csv(score_path)
        d = manifests["D"]
        n = manifests["N"]
        seen_d = d[~d["candidate_aspect"].astype(str).isin(fold.heldout_aspects)]
        seen_n = n[~n["candidate_aspect"].astype(str).isin(fold.heldout_aspects)]
        merged = seen_d[["row_uid", "candidate_aspect", "demonstrations_sha256"]].merge(
            seen_n[["row_uid", "candidate_aspect", "demonstrations_sha256"]],
            on=["row_uid", "candidate_aspect"],
            suffixes=("_D", "_N"),
            validate="one_to_one",
        )
        d_n_seen_demo_mismatch_count += int(
            merged["demonstrations_sha256_D"].ne(
                merged["demonstrations_sha256_N"]
            ).sum()
        )
    audit = {
        "retrieval_manifest_count": len(FOLDS) * len(CONDITIONS),
        "retrieval_score_unit_count": len(scores),
        "heldout_aspect_pool_instance_count": heldout_pool_count,
        "retrieval_similarity_target_read_count": similarity_target_read_count,
        "demonstration_non_train_row_count": demo_train_split_violation,
        "demonstration_heldout_aspect_count": demo_heldout_aspect_count,
        "within_answer_duplicate_review_count": within_answer_row_duplicate_count,
        "D_N_seen_demonstration_mismatch_count": d_n_seen_demo_mismatch_count,
        "failure_count": failure_count,
        "non_finite_score_count": non_finite_count,
        "prediction_score_collapse_count": collapse_count,
        "test_contract_count": test_contract_count,
    }
    if any(audit.values()):
        nonzero_allowed = {"retrieval_manifest_count", "retrieval_score_unit_count"}
        violations = {
            key: value for key, value in audit.items() if key not in nonzero_allowed and value
        }
        if violations:
            raise RuntimeError(f"Retrieval input audit failed: {violations}")
    return scores, audit, hashes


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = legacy.read_json(config_path)
    validate_config(config)
    section = config["retrieval_few_shot_stage_1"]
    thresholds = _thresholds(section)
    config_sha256 = hybrid.file_sha256(config_path)
    score_root = args.score_root.resolve()
    summary = legacy.read_json(score_root / "retrieval_run_summary.json")
    if (
        summary.get("status") != "complete"
        or int(summary.get("failure_count", -1)) != 0
        or int(summary.get("non_finite_score_count", -1)) != 0
        or int(summary.get("test_contract_count", -1)) != 0
    ):
        raise RuntimeError("Retrieval run is not safely complete.")
    retrieval_scores, retrieval_audit, retrieval_hashes = _audit_retrieval_inputs(
        score_root, config_sha256
    )

    backup_root = args.backup_root.resolve()
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
    critical_hashes = dict(retrieval_hashes)
    for path in sorted(critical_paths):
        resolved = path.resolve()
        if path != config_path and resolved not in published:
            raise ValueError(f"Analysis input lacks a verified receipt: {path}")
        observed = hybrid.file_sha256(path)
        if path != config_path and published[resolved] != observed:
            raise ValueError(f"Analysis receipt hash mismatch: {path}")
        critical_hashes[str(path)] = observed

    folds_by_id = {fold.fold_id: fold for fold in post_supervisor_l2_folds()}
    fold_records: list[dict[str, Any]] = []
    row_frames: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    repair_audit: dict[str, Any] = {}
    for fold_id in FOLDS:
        fold = folds_by_id[fold_id]
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
        probabilities: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            validate_component_alignment(
                components[condition][FEWSHOT], components[condition][QLORA]
            )
            probabilities[condition] = retrieval_probability(
                retrieval_scores[(fold_id, condition)],
                components[condition][QLORA],
            )
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
        aligned = {
            condition: np.full(len(d_component.row_uid), np.nan, dtype=float)
            for condition in CONDITIONS
        }
        for block in range(int(section["review_crossfit_folds"])):
            evaluation_mask = blocks == block
            train_mask = (~d_component.is_heldout) & (~evaluation_mask)
            selected = select_threshold(
                probabilities["D"],
                sentiment_mask,
                d_component.target,
                train_mask,
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
                aligned[condition][evaluation_mask] = threshold_aligned_scores(
                    probabilities[condition][evaluation_mask],
                    float(selected["threshold"]),
                )
        for condition in CONDITIONS:
            if not np.isfinite(aligned[condition]).all():
                raise ValueError("Retrieval cross-fitting left unevaluated instances.")
            frame = hybrid.calibrated_score_frame(
                components[condition][QLORA], aligned[condition], method_id=METHOD_ID
            )
            record, evidence = hybrid.evaluate_frame(
                frame,
                fold_id=fold_id,
                condition=condition,
                method_id=METHOD_ID,
                evidence_role="review_cross_fitted_primary",
                taxonomy_fold=fold,
            )
            fold_records.append(record)
            row_frames.extend(evidence)

        all_mask = ~d_component.is_heldout
        selected = select_threshold(
            probabilities["D"],
            sentiment_mask,
            d_component.target,
            all_mask,
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
            final_aligned = threshold_aligned_scores(
                probabilities[condition], float(selected["threshold"])
            )
            frame = hybrid.calibrated_score_frame(
                components[condition][QLORA], final_aligned, method_id=METHOD_ID
            )
            record, evidence = hybrid.evaluate_frame(
                frame,
                fold_id=fold_id,
                condition=condition,
                method_id=METHOD_ID,
                evidence_role="all_validation_fitted_diagnostic",
                taxonomy_fold=fold,
            )
            fold_records.append(record)
            row_frames.extend(evidence)

        for condition in CONDITIONS:
            frame = hybrid.build_stage_hybrid_frame(
                components[condition][FEWSHOT],
                components[condition][QLORA],
                method_id=REFERENCE_ID,
            )
            record, evidence = hybrid.evaluate_frame(
                frame,
                fold_id=fold_id,
                condition=condition,
                method_id=REFERENCE_ID,
                evidence_role="fixed_reference_reproduction",
                taxonomy_fold=fold,
            )
            fold_records.append(record)
            row_frames.extend(evidence)

    fold_frame = pd.DataFrame.from_records(fold_records)
    row_evidence = pd.concat(row_frames, ignore_index=True)
    fit_frame = pd.DataFrame.from_records(fit_records)
    crossfit = fit_frame[fit_frame["block_id"].ne("all_validation_fit")]
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
    if (
        len(fold_frame) != 3 * len(CONDITIONS) * len(FOLDS)
        or int(crossfit["train_evaluation_review_overlap_count"].sum()) != 0
        or int(fit_frame["heldout_aspect_selection_instance_count"].sum()) != 0
        or int(fold_frame["test_contract_count"].sum()) != 0
        or int(fold_frame["maximum_sentiments_per_selected_aspect"].max()) > 2
        or duplicate_count
        or collapse_count
    ):
        raise RuntimeError("Retrieval analysis leakage, identity, or decoder audit failed.")

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
    non_finite = sum(
        hybrid.non_finite_numeric_count(frame)
        for frame in (fold_frame, row_evidence, fit_frame, aggregate, bootstrap, pairwise)
    )
    if non_finite:
        raise ValueError("Retrieval analysis outputs contain non-finite values.")
    primary_id = f"{METHOD_ID}__review_cross_fitted_primary"
    reference_id = f"{REFERENCE_ID}__fixed_reference_reproduction"
    paired = pairwise[
        pairwise["left_bootstrap_method_id"].eq(primary_id)
        & pairwise["right_bootstrap_method_id"].eq(reference_id)
    ]
    if len(paired) != 1:
        raise ValueError("Retrieval-reference paired comparison is missing.")
    row = paired.iloc[0]
    gain = float(row["left_minus_right_aspect_balanced_heldout_pair_micro_f1"])
    ci_low = float(row["shared_review_bootstrap_95_ci_low"])
    promoted = bool(gain > 0.0 and ci_low > 0.0)

    output_dir = args.output_dir.resolve()
    public_dir = args.public_dir.resolve()
    files = {
        "retrieval_fold_metrics.csv": fold_frame,
        "retrieval_row_evidence.csv": row_evidence,
        "retrieval_fit_records.csv": fit_frame,
        "retrieval_aggregate_metrics.csv": aggregate,
        "retrieval_review_bootstrap.csv": bootstrap,
        "retrieval_pairwise_bootstrap.csv": pairwise,
        "retrieval_method_comparison.csv": comparison,
    }
    output_hashes: dict[str, str] = {}
    for name, frame in files.items():
        path = output_dir / name
        hybrid.atomic_csv(path, frame)
        output_hashes[str(path)] = hybrid.file_sha256(path)
    for name, frame in (
        ("retrieval_method_comparison.csv", comparison),
        ("retrieval_fold_metrics.csv", fold_frame),
        ("retrieval_review_bootstrap.csv", bootstrap),
        ("retrieval_pairwise_bootstrap.csv", pairwise),
        ("retrieval_fit_summary.csv", fit_frame),
    ):
        path = public_dir / name
        hybrid.atomic_csv(path, frame)
        output_hashes[str(path)] = hybrid.file_sha256(path)

    selected = {
        "schema_version": "taxonomy_two_stage_retrieval_few_shot_selected_v1",
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
            row["shared_review_bootstrap_95_ci_high"]
        ),
        "promotion_rule_passed": promoted,
        "config_sha256": config_sha256,
    }
    selected["selected_payload_sha256"] = hybrid.canonical_sha256(selected)
    selected_path = output_dir / "retrieval_selected.json"
    hybrid.atomic_json(selected_path, selected)
    output_hashes[str(selected_path)] = hybrid.file_sha256(selected_path)
    audit = {
        "schema_version": "taxonomy_two_stage_retrieval_few_shot_audit_v1",
        "study_id": STUDY_ID,
        "official_test_opened": False,
        "include_official_test": False,
        "test_contract_count": 0,
        "campaign_states": states,
        "receipt_summary": receipt_summary,
        "selection_count": selections_audit["selection_count"],
        "critical_input_file_count": len(critical_hashes),
        "critical_input_sha256": dict(sorted(critical_hashes.items())),
        "retrieval_inputs": retrieval_audit,
        "crossfit_fit_count": len(crossfit),
        "crossfit_train_evaluation_review_overlap_count": int(
            crossfit["train_evaluation_review_overlap_count"].sum()
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
    audit_path = output_dir / "retrieval_audit.json"
    hybrid.atomic_json(audit_path, audit)
    result = {
        "selected": selected,
        "audit_path": str(audit_path),
        "comparison_path": str(public_dir / "retrieval_method_comparison.csv"),
    }
    hybrid.atomic_json(output_dir / "retrieval_analysis_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_no_retraining_extensions_v1.json",
    )
    parser.add_argument(
        "--score-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_two_stage_no_retraining_extensions_v1",
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
