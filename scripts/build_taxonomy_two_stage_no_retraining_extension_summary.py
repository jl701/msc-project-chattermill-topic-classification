"""Build the compact comparison tables for the no-retraining extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ID = "frozen_qwen_few_shot_stage1__qlora_stage2"
SMOOTH_ID = "smooth_fewshot_qlora_stage1_fusion__qlora_stage2"
RETRIEVAL_ID = "e5_retrieval_frozen_qwen_few_shot_stage1__qlora_stage2"

DISPLAY_NAMES = {
    REFERENCE_ID: "Fixed Frozen-Qwen few-shot Stage 1 + QLoRA Stage 2",
    SMOOTH_ID: "Smooth few-shot/QLoRA Stage-1 fusion + QLoRA Stage 2",
    RETRIEVAL_ID: "E5-retrieved Frozen-Qwen few-shot Stage 1 + QLoRA Stage 2",
}


def _one(
    frame: pd.DataFrame, *, method_id: str, evidence_role: str, condition: str
) -> pd.Series:
    selected = frame[
        frame["method_id"].eq(method_id)
        & frame["evidence_role"].eq(evidence_role)
        & frame["condition"].eq(condition)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {method_id}/{evidence_role}/{condition}, "
            f"found {len(selected)}."
        )
    return selected.iloc[0]


def _paired(
    frame: pd.DataFrame, *, left_id: str, left_role: str
) -> pd.Series:
    left = f"{left_id}__{left_role}"
    right = f"{REFERENCE_ID}__fixed_reference_reproduction"
    selected = frame[
        frame["condition"].eq("D")
        & frame["left_bootstrap_method_id"].eq(left)
        & frame["right_bootstrap_method_id"].eq(right)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one D paired-bootstrap row for {left}.")
    return selected.iloc[0]


def build_tables(
    *,
    smooth_aggregate: pd.DataFrame,
    retrieval_aggregate: pd.DataFrame,
    smooth_pairwise: pd.DataFrame,
    retrieval_pairwise: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_smooth = _one(
        smooth_aggregate,
        method_id=REFERENCE_ID,
        evidence_role="fixed_reference_reproduction",
        condition="D",
    )
    reference_retrieval = _one(
        retrieval_aggregate,
        method_id=REFERENCE_ID,
        evidence_role="fixed_reference_reproduction",
        condition="D",
    )
    metric_columns = [
        "heldout_pair_micro_f1_mean",
        "overall_pair_micro_f1_mean",
        "heldout_pair_precision_mean",
        "heldout_pair_recall_mean",
        "heldout_presence_f1_mean",
        "heldout_presence_ap_mean",
        "oracle_gated_sentiment_set_micro_f1_mean",
        "aspect_call_rate_mean",
    ]
    for column in metric_columns:
        if float(reference_smooth[column]) != float(reference_retrieval[column]):
            raise ValueError(f"Reference reproduction differs for {column}.")

    methods = [
        (REFERENCE_ID, "fixed_reference_reproduction", smooth_aggregate, None),
        (
            SMOOTH_ID,
            "review_cross_fitted_primary",
            smooth_aggregate,
            _paired(
                smooth_pairwise,
                left_id=SMOOTH_ID,
                left_role="review_cross_fitted_primary",
            ),
        ),
        (
            RETRIEVAL_ID,
            "review_cross_fitted_primary",
            retrieval_aggregate,
            _paired(
                retrieval_pairwise,
                left_id=RETRIEVAL_ID,
                left_role="review_cross_fitted_primary",
            ),
        ),
    ]

    primary_records: list[dict[str, object]] = []
    transfer_records: list[dict[str, object]] = []
    for method_id, role, aggregate, paired in methods:
        for condition in ("D", "N"):
            row = _one(
                aggregate,
                method_id=method_id,
                evidence_role=role,
                condition=condition,
            )
            transfer_records.append(
                {
                    "method_id": method_id,
                    "method": DISPLAY_NAMES[method_id],
                    "evidence_role": role,
                    "condition": condition,
                    **{column: float(row[column]) for column in metric_columns},
                    "official_test_opened": False,
                    "test_contract_count": 0,
                }
            )
        d_row = _one(
            aggregate,
            method_id=method_id,
            evidence_role=role,
            condition="D",
        )
        if paired is None:
            gain = 0.0
            ci_low = None
            ci_high = None
            promoted = None
            decision = "fixed_reference"
        else:
            gain = float(
                paired[
                    "left_minus_right_aspect_balanced_heldout_pair_micro_f1"
                ]
            )
            ci_low = float(paired["shared_review_bootstrap_95_ci_low"])
            ci_high = float(paired["shared_review_bootstrap_95_ci_high"])
            promoted = bool(gain > 0.0 and ci_low > 0.0)
            decision = "promote" if promoted else "do_not_promote"
        primary_records.append(
            {
                "rank_by_D_heldout_pair_f1": 0,
                "method_id": method_id,
                "method": DISPLAY_NAMES[method_id],
                "evidence_role": role,
                "D_heldout_pair_micro_f1_mean": float(
                    d_row["heldout_pair_micro_f1_mean"]
                ),
                "D_overall_pair_micro_f1_mean": float(
                    d_row["overall_pair_micro_f1_mean"]
                ),
                "D_heldout_pair_f1_gain_vs_fixed_reference": gain,
                "paired_review_bootstrap_95_ci_low": ci_low,
                "paired_review_bootstrap_95_ci_high": ci_high,
                "promotion_rule_passed": promoted,
                "decision": decision,
                "bootstrap_draws": 20_000,
                "bootstrap_seed": 13,
                "official_test_opened": False,
                "test_contract_count": 0,
            }
        )

    primary = pd.DataFrame.from_records(primary_records).sort_values(
        "D_heldout_pair_micro_f1_mean", ascending=False, kind="stable"
    )
    primary["rank_by_D_heldout_pair_f1"] = range(1, len(primary) + 1)
    transfer = pd.DataFrame.from_records(transfer_records).sort_values(
        ["condition", "heldout_pair_micro_f1_mean"],
        ascending=[True, False],
        kind="stable",
    )
    return primary.reset_index(drop=True), transfer.reset_index(drop=True)


def build_audit_table(
    smooth_audit: dict[str, object], retrieval_audit: dict[str, object]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for direction, audit in (
        ("smooth_stage_1_fusion", smooth_audit),
        ("retrieval_few_shot_stage_1", retrieval_audit),
    ):
        receipt_summary = audit["receipt_summary"]
        assert isinstance(receipt_summary, dict)
        retrieval = audit.get("retrieval_inputs", {})
        assert isinstance(retrieval, dict)
        records.append(
            {
                "direction": direction,
                "formal_receipt_count": sum(
                    int(value["receipt_count"])
                    for value in receipt_summary.values()
                ),
                "formal_unique_verified_file_count": sum(
                    int(value["unique_verified_file_count"])
                    for value in receipt_summary.values()
                ),
                "critical_input_file_count": int(
                    audit.get(
                        "critical_input_file_count",
                        audit.get("critical_verified_input_file_count", -1),
                    )
                ),
                "selection_count": int(audit["selection_count"]),
                "crossfit_fit_count": int(audit["crossfit_fit_count"]),
                "retrieval_manifest_count": int(
                    retrieval.get("retrieval_manifest_count", 0)
                ),
                "retrieval_score_unit_count": int(
                    retrieval.get("retrieval_score_unit_count", 0)
                ),
                "retrieval_contract_violation_count": sum(
                    int(retrieval.get(key, 0))
                    for key in (
                        "D_N_seen_demonstration_mismatch_count",
                        "demonstration_heldout_aspect_count",
                        "demonstration_non_train_row_count",
                        "heldout_aspect_pool_instance_count",
                        "retrieval_similarity_target_read_count",
                        "within_answer_duplicate_review_count",
                    )
                ),
                "crossfit_train_evaluation_review_overlap_count": int(
                    audit["crossfit_train_evaluation_review_overlap_count"]
                ),
                "heldout_aspect_selection_instance_count": int(
                    audit["heldout_aspect_selection_instance_count"]
                ),
                "maximum_sentiments_per_selected_aspect": int(
                    audit["maximum_sentiments_per_selected_aspect"]
                ),
                "duplicate_row_evidence_identity_count": int(
                    audit["duplicate_row_evidence_identity_count"]
                ),
                "prediction_collapse_count": int(audit["prediction_collapse_count"]),
                "non_finite_value_count": int(audit["non_finite_value_count"]),
                "official_test_opened": bool(audit["official_test_opened"]),
                "test_contract_count": int(audit["test_contract_count"]),
                "promotion_rule_passed": bool(audit["promotion_rule_passed"]),
                "audit_payload_sha256": str(audit["audit_payload_sha256"]),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if (
        int(frame["retrieval_contract_violation_count"].sum())
        or int(frame["crossfit_train_evaluation_review_overlap_count"].sum())
        or int(frame["heldout_aspect_selection_instance_count"].sum())
        or int(frame["duplicate_row_evidence_identity_count"].sum())
        or int(frame["prediction_collapse_count"].sum())
        or int(frame["non_finite_value_count"].sum())
        or bool(frame["official_test_opened"].any())
        or int(frame["test_contract_count"].sum())
        or int(frame["maximum_sentiments_per_selected_aspect"].max()) > 2
    ):
        raise ValueError("No-retraining extension audit contains a violation.")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = (
        PROJECT_ROOT
        / "outputs/experimental/taxonomy_two_stage_no_retraining_extensions_v1"
    )
    parser.add_argument(
        "--smooth-aggregate",
        type=Path,
        default=base / "smooth_fusion_aggregate_metrics.csv",
    )
    parser.add_argument(
        "--retrieval-aggregate",
        type=Path,
        default=base / "retrieval_aggregate_metrics.csv",
    )
    parser.add_argument(
        "--smooth-pairwise",
        type=Path,
        default=base / "smooth_fusion_pairwise_bootstrap.csv",
    )
    parser.add_argument(
        "--retrieval-pairwise",
        type=Path,
        default=base / "retrieval_pairwise_bootstrap.csv",
    )
    parser.add_argument(
        "--smooth-audit", type=Path, default=base / "smooth_fusion_audit.json"
    )
    parser.add_argument(
        "--retrieval-audit", type=Path, default=base / "retrieval_audit.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "docs/thesis_figure_data/taxonomy_two_stage_no_retraining_extensions_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary, transfer = build_tables(
        smooth_aggregate=pd.read_csv(args.smooth_aggregate),
        retrieval_aggregate=pd.read_csv(args.retrieval_aggregate),
        smooth_pairwise=pd.read_csv(args.smooth_pairwise),
        retrieval_pairwise=pd.read_csv(args.retrieval_pairwise),
    )
    with args.smooth_audit.open("r", encoding="utf-8") as handle:
        smooth_audit = json.load(handle)
    with args.retrieval_audit.open("r", encoding="utf-8") as handle:
        retrieval_audit = json.load(handle)
    audit = build_audit_table(smooth_audit, retrieval_audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(args.output_dir / "primary_extension_comparison.csv", index=False)
    transfer.to_csv(args.output_dir / "condition_transfer_comparison.csv", index=False)
    audit.to_csv(args.output_dir / "extension_audit_summary.csv", index=False)


if __name__ == "__main__":
    main()
