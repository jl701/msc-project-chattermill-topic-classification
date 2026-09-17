"""Build shareable aggregate tables for the sealed taxonomy final test.

This post-reveal script never changes predictions, thresholds, checkpoints or
the registered hypothesis family.  It verifies the immutable single-reveal
outputs and converts them into compact, review-text-free thesis tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.taxonomy_final_test import (  # noqa: E402
    canonical_sha256,
    file_sha256,
)


DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT / "outputs" / "experimental" / "taxonomy_two_stage_final_test_v1"
)
DEFAULT_BACKUP_ROOT = (
    PROJECT_ROOT.parent
    / "cloud_backups"
    / "taxonomy_final_test_v1_6a0b42d_final_graph"
)
DEFAULT_PUBLIC_ROOT = (
    PROJECT_ROOT / "docs" / "thesis_figure_data" / "taxonomy_final_test_v1"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "taxonomy_final_test_v1_post_reveal_audit_20260825.json"
)
DEFAULT_VALIDATION = (
    PROJECT_ROOT
    / "docs"
    / "thesis_figure_data"
    / "taxonomy_two_stage_stage_hybrid_calibration_v1"
    / "stage_hybrid_calibration_method_comparison.csv"
)

METHOD_NAMES = {
    "strict_train_only_tfidf": "TF-IDF",
    "e5_base_v2": "Frozen E5",
    "description_to_classifier_weight_transfer": "DCWT",
    "distilbert_review_candidate_cross_encoder": "DistilBERT",
    "frozen_qwen_candidate_pair": "Frozen Qwen zero-shot",
    "frozen_qwen_few_shot": "Frozen Qwen few-shot",
    "qwen_candidate_pair_qlora": "QLoRA",
    "frozen_qwen_few_shot_stage1__qlora_stage2": "Fixed stage-wise composition",
}

CONFIRMATORY_METHODS = (
    "frozen_qwen_few_shot_stage1__qlora_stage2",
    "frozen_qwen_few_shot",
    "qwen_candidate_pair_qlora",
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _pooled_binary_metrics(frame: pd.DataFrame) -> tuple[float, float, float]:
    tp = int(frame["pair_tp"].sum())
    fp = int(frame["pair_fp"].sum())
    fn = int(frame["pair_fn"].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return float(precision), float(recall), float(f1)


def build_l2_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source = metrics[metrics["level"].eq("L2")]
    for (method, condition), group in source.groupby(
        ["method_id", "condition"], sort=True
    ):
        heldout = group[group["partition"].eq("heldout")]
        seen = group[group["partition"].eq("seen")]
        overall = group[group["partition"].eq("overall")]
        pooled_precision, pooled_recall, pooled_f1 = _pooled_binary_metrics(heldout)
        rows.append(
            {
                "method_id": method,
                "method": METHOD_NAMES[str(method)],
                "condition": condition,
                "fold_count": int(heldout["fold_id"].nunique()),
                "heldout_pair_precision_fold_mean": heldout[
                    "pair_micro_precision"
                ].mean(),
                "heldout_pair_recall_fold_mean": heldout[
                    "pair_micro_recall"
                ].mean(),
                "heldout_pair_f1_fold_mean": heldout["pair_micro_f1"].mean(),
                "heldout_pair_precision_pooled": pooled_precision,
                "heldout_pair_recall_pooled": pooled_recall,
                "heldout_pair_f1_pooled_sensitivity": pooled_f1,
                "seen_pair_f1_fold_mean": seen["pair_micro_f1"].mean(),
                "overall_pair_f1_fold_mean": overall["pair_micro_f1"].mean(),
                "overall_exact_set_match_fold_mean": overall[
                    "exact_set_match"
                ].mean(),
                "overall_mean_prediction_set_size_fold_mean": overall[
                    "mean_prediction_set_size"
                ].mean(),
                "overall_aspect_call_rate_fold_mean": overall[
                    "aspect_call_rate"
                ].mean(),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["condition", "heldout_pair_f1_fold_mean"],
        ascending=[True, False],
        kind="stable",
    )


def build_stage_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source = diagnostics[diagnostics["level"].eq("L2")]
    for (method, condition), group in source.groupby(
        ["method_id", "condition"], sort=True
    ):
        tp = int(group["stage1_tp"].sum())
        fp = int(group["stage1_fp"].sum())
        fn = int(group["stage1_fn"].sum())
        pooled_presence_f1 = (
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        )
        rows.append(
            {
                "method_id": method,
                "method": METHOD_NAMES[str(method)],
                "condition": condition,
                "fold_count": int(group["fold_id"].nunique()),
                "heldout_presence_ap_fold_mean": group[
                    "heldout_presence_average_precision"
                ].mean(),
                "heldout_presence_f1_fold_mean": group[
                    "heldout_presence_f1"
                ].mean(),
                "heldout_presence_f1_pooled_sensitivity": pooled_presence_f1,
                "oracle_gated_stage2_pair_f1_fold_mean": group[
                    "oracle_gated_stage2_pair_micro_f1"
                ].mean(),
                "stage1_tp": tp,
                "stage1_fp": fp,
                "stage1_fn": fn,
                "third_sentiment_violation_count": int(
                    group["third_sentiment_violation_count"].sum()
                ),
                "prediction_collapse_count": int(
                    group["prediction_collapse_count"].sum()
                ),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["condition", "heldout_presence_f1_fold_mean"],
        ascending=[True, False],
        kind="stable",
    )


def build_confirmatory_folds(metrics: pd.DataFrame) -> pd.DataFrame:
    source = metrics[
        metrics["level"].eq("L2")
        & metrics["condition"].eq("D")
        & metrics["partition"].eq("heldout")
        & metrics["method_id"].isin(CONFIRMATORY_METHODS)
    ]
    pivot = source.pivot(
        index="fold_id", columns="method_id", values="pair_micro_f1"
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "frozen_qwen_few_shot_stage1__qlora_stage2": "fixed_composition_f1",
            "frozen_qwen_few_shot": "frozen_qwen_few_shot_f1",
            "qwen_candidate_pair_qlora": "qlora_f1",
        }
    )
    pivot["fixed_minus_few_shot"] = (
        pivot["fixed_composition_f1"] - pivot["frozen_qwen_few_shot_f1"]
    )
    pivot["fixed_minus_qlora"] = (
        pivot["fixed_composition_f1"] - pivot["qlora_f1"]
    )
    return pivot.sort_values("fold_id", kind="stable")


def build_description_effects(l2_summary: pd.DataFrame) -> pd.DataFrame:
    heldout = l2_summary.pivot(
        index=["method_id", "method"],
        columns="condition",
        values="heldout_pair_f1_fold_mean",
    ).reset_index()
    overall = l2_summary.pivot(
        index=["method_id", "method"],
        columns="condition",
        values="overall_pair_f1_fold_mean",
    ).reset_index()
    heldout = heldout.rename(columns={"N": "heldout_N", "D": "heldout_D"})
    overall = overall.rename(columns={"N": "overall_N", "D": "overall_D"})
    result = heldout.merge(overall, on=["method_id", "method"], validate="one_to_one")
    result["heldout_D_minus_N"] = result["heldout_D"] - result["heldout_N"]
    result["overall_D_minus_N"] = result["overall_D"] - result["overall_N"]
    return result.sort_values("heldout_D", ascending=False, kind="stable")


def build_l4_table(metrics: pd.DataFrame) -> pd.DataFrame:
    heldout = metrics[
        metrics["level"].eq("L4")
        & metrics["condition"].eq("D")
        & metrics["partition"].eq("heldout")
    ].copy()
    overall = metrics[
        metrics["level"].eq("L4")
        & metrics["condition"].eq("D")
        & metrics["partition"].eq("overall")
    ][["method_id", "fold_id", "pair_micro_f1"]].rename(
        columns={"pair_micro_f1": "overall_pair_f1"}
    )
    result = heldout.merge(
        overall, on=["method_id", "fold_id"], validate="one_to_one"
    )
    result.insert(1, "method", result["method_id"].map(METHOD_NAMES))
    result.insert(
        3,
        "parent_group",
        result["fold_id"].map(
            {
                "l4-g01": "Company brand",
                "l4-g02": "Staff support",
                "l4-g03": "Value",
            }
        ),
    )
    return result[
        [
            "method_id",
            "method",
            "fold_id",
            "parent_group",
            "pair_micro_precision",
            "pair_micro_recall",
            "pair_micro_f1",
            "overall_pair_f1",
            "gold_pair_count",
            "predicted_pair_count",
            "review_rows",
        ]
    ].sort_values(["method_id", "fold_id"], kind="stable")


def build_validation_test_comparison(
    l2_summary: pd.DataFrame, validation_path: Path
) -> pd.DataFrame:
    validation = pd.read_csv(validation_path)
    validation = validation[
        validation["condition"].eq("D")
        & validation["method_id"].isin(CONFIRMATORY_METHODS)
    ][["method_id", "heldout_pair_micro_f1_mean"]].rename(
        columns={"heldout_pair_micro_f1_mean": "validation_primary_f1"}
    )
    test = l2_summary[
        l2_summary["condition"].eq("D")
        & l2_summary["method_id"].isin(CONFIRMATORY_METHODS)
    ][["method_id", "method", "heldout_pair_f1_fold_mean"]].rename(
        columns={"heldout_pair_f1_fold_mean": "official_test_primary_f1"}
    )
    result = validation.merge(test, on="method_id", validate="one_to_one")
    result["test_minus_validation"] = (
        result["official_test_primary_f1"] - result["validation_primary_f1"]
    )
    return result.sort_values(
        "official_test_primary_f1", ascending=False, kind="stable"
    )


def _verify_single_reveal(
    result_root: Path, backup_root: Path
) -> tuple[dict[str, object], dict[str, str]]:
    analysis_root = result_root / "single_reveal_analysis"
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest = _load_json(manifest_path)
    sealed = dict(manifest)
    observed_payload_hash = sealed.pop("manifest_payload_sha256", None)
    if observed_payload_hash != canonical_sha256(sealed):
        raise ValueError("Analysis manifest payload hash mismatch.")
    required = {
        "outcomes_revealed_once": True,
        "score_bundle_count": 105,
        "failure_count": 0,
        "test_contract_count": 1,
        "post_test_tuning_permitted": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Analysis manifest {key} mismatch.")
    verified: dict[str, str] = {}
    for name, expected_hash in dict(manifest["output_sha256s"]).items():
        path = analysis_root / {
            "all_metrics": "all_registered_metrics.csv",
            "confirmatory": "confirmatory_hypotheses.json",
            "diagnostics": "diagnostic_metrics.csv",
            "headline": "primary_headline.csv",
            "heldout_predictions": "heldout_prediction_sets.jsonl",
        }[str(name)]
        observed = file_sha256(path)
        if observed != expected_hash:
            raise ValueError(f"Single-reveal output hash mismatch: {path.name}")
        backup = backup_root / path.relative_to(result_root)
        if not backup.is_file() or file_sha256(backup) != observed:
            raise ValueError(f"Single-reveal backup mismatch: {path.name}")
        verified[path.name] = observed
    backup_manifest = backup_root / manifest_path.relative_to(result_root)
    if not backup_manifest.is_file() or file_sha256(backup_manifest) != file_sha256(
        manifest_path
    ):
        raise ValueError("Analysis-manifest backup mismatch.")
    verified[manifest_path.name] = file_sha256(manifest_path)
    return manifest, verified


def _integrity_checks(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    result_root: Path,
) -> dict[str, object]:
    expected_metric_rows = 8 * 12 * 2 * 3 + 3 * 3 * 1 * 3
    expected_diagnostic_rows = 8 * 12 * 2 + 3 * 3
    numeric_metrics = metrics.select_dtypes(include=[np.number])
    numeric_diagnostics = diagnostics.select_dtypes(include=[np.number])
    score_manifests = list((result_root / "sealed_scores").glob("*/manifest.json"))
    manifest_zero_keys = (
        "failure_count",
        "non_finite_value_count",
        "prediction_collapse_count",
        "third_sentiment_violation_count",
    )
    manifest_integrity_failures = 0
    manifest_payload_failures = 0
    for path in score_manifests:
        value = _load_json(path)
        payload = dict(value)
        observed = payload.pop("manifest_payload_sha256", None)
        if observed != canonical_sha256(payload):
            manifest_payload_failures += 1
        if any(int(value.get(key, -1)) != 0 for key in manifest_zero_keys):
            manifest_integrity_failures += 1
        score_path = path.parent / str(value.get("score_file", "scores.csv"))
        if not score_path.is_file() or file_sha256(score_path) != value.get(
            "score_file_sha256"
        ):
            manifest_integrity_failures += 1
    result = {
        "score_bundle_count": len(score_manifests),
        "score_manifest_payload_failure_count": manifest_payload_failures,
        "score_manifest_or_file_integrity_failure_count": manifest_integrity_failures,
        "registered_metric_rows": len(metrics),
        "expected_registered_metric_rows": expected_metric_rows,
        "diagnostic_rows": len(diagnostics),
        "expected_diagnostic_rows": expected_diagnostic_rows,
        "non_finite_metric_value_count": int(
            (~np.isfinite(numeric_metrics.to_numpy(dtype=float))).sum()
        ),
        "non_finite_diagnostic_value_count": int(
            (~np.isfinite(numeric_diagnostics.to_numpy(dtype=float))).sum()
        ),
        "negative_count_value_count": int(
            (metrics[
                [
                    "pair_tp",
                    "pair_fp",
                    "pair_fn",
                    "gold_pair_count",
                    "predicted_pair_count",
                    "review_rows",
                ]
            ].to_numpy(dtype=float) < 0).sum()
        ),
        "third_sentiment_violation_count": int(
            diagnostics["third_sentiment_violation_count"].sum()
        ),
        "prediction_collapse_count": int(
            diagnostics["prediction_collapse_count"].sum()
        ),
        "empty_prediction_metric_row_count": int(
            metrics["predicted_pair_count"].eq(0).sum()
        ),
        "saturated_prediction_metric_row_count": int(
            metrics["pair_micro_recall"].eq(1).mul(
                metrics["pair_micro_precision"].lt(0.01)
            ).sum()
        ),
    }
    result["status"] = (
        "pass"
        if result["score_bundle_count"] == 105
        and result["registered_metric_rows"] == expected_metric_rows
        and result["diagnostic_rows"] == expected_diagnostic_rows
        and all(
            int(result[key]) == 0
            for key in (
                "score_manifest_payload_failure_count",
                "score_manifest_or_file_integrity_failure_count",
                "non_finite_metric_value_count",
                "non_finite_diagnostic_value_count",
                "negative_count_value_count",
                "third_sentiment_violation_count",
                "prediction_collapse_count",
            )
        )
        else "fail"
    )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    result_root = args.result_root.resolve()
    backup_root = args.backup_root.resolve()
    public_root = args.public_root.resolve()
    audit_path = args.audit.resolve()
    if (public_root.exists() and any(public_root.iterdir())) or audit_path.exists():
        raise FileExistsError("Post-reveal public outputs must be new and immutable.")
    manifest, verified_outputs = _verify_single_reveal(result_root, backup_root)
    analysis_root = result_root / "single_reveal_analysis"
    metrics = pd.read_csv(analysis_root / "all_registered_metrics.csv")
    diagnostics = pd.read_csv(analysis_root / "diagnostic_metrics.csv")
    confirmatory = _load_json(analysis_root / "confirmatory_hypotheses.json")

    l2_summary = build_l2_summary(metrics)
    stage_summary = build_stage_summary(diagnostics)
    confirmatory_folds = build_confirmatory_folds(metrics)
    description_effects = build_description_effects(l2_summary)
    l4 = build_l4_table(metrics)
    validation_test = build_validation_test_comparison(
        l2_summary, args.validation.resolve()
    )
    confirmatory_table = pd.DataFrame.from_records(confirmatory["results"])

    public_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "l2_model_condition_summary": public_root
        / "official_l2_model_condition_summary.csv",
        "l2_stage_diagnostics": public_root / "official_l2_stage_diagnostics.csv",
        "l2_confirmatory_fold_results": public_root
        / "official_l2_confirmatory_fold_results.csv",
        "l2_description_effects": public_root / "official_l2_description_effects.csv",
        "l4_group_results": public_root / "official_l4_group_results.csv",
        "validation_test_comparison": public_root
        / "official_validation_test_comparison.csv",
        "confirmatory_results": public_root / "official_confirmatory_results.csv",
    }
    for path, frame in (
        (outputs["l2_model_condition_summary"], l2_summary),
        (outputs["l2_stage_diagnostics"], stage_summary),
        (outputs["l2_confirmatory_fold_results"], confirmatory_folds),
        (outputs["l2_description_effects"], description_effects),
        (outputs["l4_group_results"], l4),
        (outputs["validation_test_comparison"], validation_test),
        (outputs["confirmatory_results"], confirmatory_table),
    ):
        _write_csv(path, frame)

    integrity = _integrity_checks(metrics, diagnostics, result_root)
    if integrity["status"] != "pass":
        raise ValueError(f"Post-reveal integrity audit failed: {integrity}")
    audit: dict[str, object] = {
        "schema_version": "taxonomy_final_test_post_reveal_audit_v1",
        "protocol_id": manifest["protocol_id"],
        "execution_commit": manifest["execution_commit"],
        "single_reveal_manifest_sha256": verified_outputs["analysis_manifest.json"],
        "single_reveal_outputs": verified_outputs,
        "public_output_sha256s": {
            key: file_sha256(path) for key, path in sorted(outputs.items())
        },
        "integrity": integrity,
        "scientific_failure_modes": {
            "empty_partition_prediction_records": metrics.loc[
                metrics["predicted_pair_count"].eq(0),
                [
                    "method_id",
                    "level",
                    "fold_id",
                    "condition",
                    "partition",
                    "gold_pair_count",
                ],
            ].to_dict(orient="records"),
            "interpretation": (
                "Partition-local empty predictions are reported as scientific "
                "failure modes, not infrastructure failures, when the complete "
                "registered score bundle passes its non-collapse contract."
            ),
        },
        "confirmatory_h1_superiority": bool(
            confirmatory_table.loc[
                confirmatory_table["hypothesis"].eq("H1"), "superiority"
            ].iloc[0]
        ),
        "confirmatory_h2_gate_passed": bool(
            confirmatory_table.loc[
                confirmatory_table["hypothesis"].eq("H2"), "gate_H1_passed"
            ].iloc[0]
        ),
        "confirmatory_h2_superiority": bool(
            confirmatory_table.loc[
                confirmatory_table["hypothesis"].eq("H2"), "superiority"
            ].iloc[0]
        ),
        "outcomes_revealed_once": True,
        "post_test_tuning_permitted": False,
        "raw_review_text_in_public_outputs": False,
        "failure_count": 0,
    }
    audit["audit_payload_sha256"] = canonical_sha256(audit)
    _write_json(audit_path, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
