"""Render the audited taxonomy scientific-freeze results as a thesis record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_LABELS = {
    "strict_train_only_tfidf": "TF-IDF",
    "e5_base_v2": "Frozen E5",
    "description_conditioned_weight_transfer_kernel_ridge": "Description-to-weight transfer",
    "distilbert_review_candidate_cross_encoder": "DistilBERT",
    "frozen_qwen_candidate_pair": "Frozen Qwen zero-shot",
    "frozen_qwen_few_shot": "Frozen Qwen few-shot",
    "qwen_candidate_pair_qlora": "QLoRA Qwen",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    labels: list[str],
    formats: dict[str, str] | None = None,
) -> str:
    formats = formats or {}
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" for _ in labels) + "|",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                rendered = "—"
            elif column in formats:
                rendered = formats[column].format(value)
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    table_dir = args.table_dir.resolve()
    summary = pd.read_csv(table_dir / "l2_model_condition_review_bootstrap.csv")
    effects = pd.read_csv(table_dir / "l2_description_effects_review_bootstrap.csv")
    metrics = pd.read_csv(table_dir / "l2_complete_model_condition_metrics.csv")
    diagnostics = pd.read_csv(table_dir / "l2_fold_support_and_filtering.csv")
    rich = pd.read_csv(table_dir / "selected_rich_description_confirmation.csv")
    controls = pd.read_csv(table_dir / "architecture_and_decoder_controls.csv")
    dcwt_controls = pd.read_csv(table_dir / "dcwt_generator_controls.csv")
    l4 = pd.read_csv(table_dir / "l4_parent_group_review_bootstrap.csv")
    l3 = pd.read_csv(table_dir / "l3_posthoc_targeted_robustness.csv")
    negative = pd.read_csv(table_dir / "negative_findings.csv")
    l1 = pd.read_csv(table_dir / "l1_closed_taxonomy_development_reference.csv")
    manifest = _read_json(args.manifest.resolve())

    for frame in (summary, effects, metrics, rich, controls, l4):
        if "method_id" in frame:
            frame.insert(
                0,
                "model",
                frame["method_id"].map(METHOD_LABELS).fillna(frame["method_id"]),
            )

    primary = effects[
        [
            "model",
            "N",
            "D",
            "D_minus_N",
            "paired_review_bootstrap_95_ci_low",
            "paired_review_bootstrap_95_ci_high",
            "pooled_N_sensitivity",
            "pooled_D_sensitivity",
            "pooled_D_minus_N_sensitivity",
            "improved_folds",
            "tied_folds",
            "worsened_folds",
            "enumerated_fold_sign_sensitivity_p",
            "holm_adjusted_across_all_7_methods",
        ]
    ].copy()
    primary["review_bootstrap_95_ci"] = primary.apply(
        lambda row: (
            f"[{row.paired_review_bootstrap_95_ci_low:.4f}, "
            f"{row.paired_review_bootstrap_95_ci_high:.4f}]"
        ),
        axis=1,
    )
    primary["fold_direction"] = primary.apply(
        lambda row: f"{int(row.improved_folds)}/{int(row.tied_folds)}/{int(row.worsened_folds)}",
        axis=1,
    )

    metrics_view = metrics[
        [
            "model",
            "condition",
            "heldout_pair_micro_f1_mean",
            "overall_pair_micro_f1_mean",
            "seen_pair_micro_f1_mean",
            "heldout_presence_average_precision_mean",
            "oracle_gated_sentiment_set_micro_f1_mean",
        ]
    ]

    lines = [
        "# Taxonomy two-stage scientific-freeze results",
        "",
        "Date: 2026-08-22  ",
        "Status: **PASS — validation-only scientific freeze**  ",
        "Formal execution boundary: `aa84212976a652d62cfca31ed8bf0516a216c485`  ",
        "Official test: **sealed and unopened** (`include_official_test=false`, `test_contract_count=0`).",
        "",
        "## What is primary",
        "",
        "The primary Level 2 estimand is the unweighted mean of the 12 held-out-aspect pair micro-F1 values. Uncertainty uses 20,000 synchronized `row_uid` review-cluster bootstrap draws (seed 13), recomputing F1 inside each draw. The pooled held-out score is a sensitivity estimand. Intervals condition on the observed trained realization of each neural system and the fixed 12-aspect taxonomy.",
        "",
        "## Primary Level 2 minimal-description comparison",
        "",
        _markdown_table(
            primary,
            columns=[
                "model",
                "N",
                "D",
                "D_minus_N",
                "review_bootstrap_95_ci",
                "pooled_N_sensitivity",
                "pooled_D_sensitivity",
                "pooled_D_minus_N_sensitivity",
                "fold_direction",
                "enumerated_fold_sign_sensitivity_p",
                "holm_adjusted_across_all_7_methods",
            ],
            labels=[
                "System",
                "N F1",
                "D F1",
                "D−N",
                "Review-bootstrap 95% CI",
                "Pooled N",
                "Pooled D",
                "Pooled D−N",
                "Better/tie/worse folds",
                "Fold-sign sensitivity p",
                "Holm-adjusted sensitivity p",
            ],
            formats={
                "N": "{:.4f}",
                "D": "{:.4f}",
                "D_minus_N": "{:+.4f}",
                "pooled_N_sensitivity": "{:.4f}",
                "pooled_D_sensitivity": "{:.4f}",
                "pooled_D_minus_N_sensitivity": "{:+.4f}",
                "enumerated_fold_sign_sensitivity_p": "{:.4f}",
                "holm_adjusted_across_all_7_methods": "{:.4f}",
            },
        ),
        "",
        "The fold-sign calculation is an enumerated sensitivity diagnostic, not a design-exact randomization test. Holm adjustment applies to that explicitly labelled seven-method sensitivity family.",
        "",
        "## Complete Level 2 evaluation views",
        "",
        _markdown_table(
            metrics_view,
            columns=list(metrics_view.columns),
            labels=[
                "System",
                "Card",
                "Held-out pair F1",
                "Overall pair F1",
                "Seen pair F1",
                "Held-out presence AP",
                "Oracle-gated sentiment-set F1",
            ],
            formats={
                "heldout_pair_micro_f1_mean": "{:.4f}",
                "overall_pair_micro_f1_mean": "{:.4f}",
                "seen_pair_micro_f1_mean": "{:.4f}",
                "heldout_presence_average_precision_mean": "{:.4f}",
                "oracle_gated_sentiment_set_micro_f1_mean": "{:.4f}",
            },
        ),
        "",
        "Held-out presence F1 is omitted from this display because it duplicates held-out aspect F1 by construction. Oracle-gated sentiment-set F1 is diagnostic: it supplies gold aspect gates and therefore is not end-to-end performance.",
        "The companion `l2_fold_pair_error_diagnostics.csv` preserves every fold's TP, FP, FN, precision, recall, F1, error-row rates and prediction-set size for appendix analysis.",
        "",
        "## Level 1 closed-taxonomy development reference",
        "",
        _markdown_table(
            l1,
            columns=list(l1.columns),
            labels=["System", "Validation reviews", "Pair micro-F1", "Pair macro-F1", "Aspect micro-F1", "Selected C", "Threshold", "Boundary", "Interpretation"],
            formats={"pair_micro_f1": "{:.4f}", "pair_macro_f1": "{:.4f}", "aspect_micro_f1": "{:.4f}", "selected_c": "{:.2f}", "selected_threshold": "{:.4f}"},
        ),
        "",
        "## Training filtering and held-out support",
        "",
        _markdown_table(
            diagnostics,
            columns=list(diagnostics.columns),
            labels=[
                "Fold",
                "Held-out aspect",
                "Original train",
                "Filtered train",
                "Removed train",
                "Validation reviews",
                "Positive review support",
            ],
        ),
        "",
        "## Selected rich-description confirmation",
        "",
        _markdown_table(
            rich,
            columns=[
                "model",
                "selected_rich_interface",
                "fold_count",
                "D_heldout_pair_micro_f1_mean",
                "R_heldout_pair_micro_f1_mean",
                "R_minus_D_heldout_pair_micro_f1_mean",
                "R_higher_folds",
                "equal_folds",
                "R_lower_folds",
                "R_minus_D_presence_average_precision_mean",
                "R_minus_D_presence_f1_mean",
                "R_minus_D_presence_precision_mean",
                "R_minus_D_presence_recall_mean",
                "R_minus_D_false_positive_rows_per_100_mean",
                "R_minus_D_overall_pair_micro_f1_mean",
                "D_exactly_matches_prior_control",
            ],
            labels=[
                "System",
                "Selected card",
                "Folds",
                "D held-out pair F1",
                "R held-out pair F1",
                "Held-out pair F1 R−D",
                "R better",
                "Tie",
                "R worse",
                "Presence AP R−D",
                "Presence F1 R−D",
                "Precision R−D",
                "Recall R−D",
                "FP rows/100 R−D",
                "Overall pair F1 R−D",
                "D exact",
            ],
            formats={
                "R_minus_D_heldout_pair_micro_f1_mean": "{:+.4f}",
                "D_heldout_pair_micro_f1_mean": "{:.4f}",
                "R_heldout_pair_micro_f1_mean": "{:.4f}",
                "R_minus_D_presence_average_precision_mean": "{:+.4f}",
                "R_minus_D_presence_f1_mean": "{:+.4f}",
                "R_minus_D_presence_precision_mean": "{:+.4f}",
                "R_minus_D_presence_recall_mean": "{:+.4f}",
                "R_minus_D_false_positive_rows_per_100_mean": "{:+.2f}",
                "R_minus_D_overall_pair_micro_f1_mean": "{:+.4f}",
            },
        ),
        "",
        "This is a bounded secondary confirmation after validation-only interface development, not a new primary endpoint.",
        "",
        "## Architecture and decoder controls",
        "",
        _markdown_table(
            controls,
            columns=["study", "model", "variant", "pair_micro_f1_mean", "delta_reference", "pair_micro_f1_delta"],
            labels=["Study", "System", "Variant", "Pair F1", "Reference", "Delta"],
            formats={"pair_micro_f1_mean": "{:.4f}", "pair_micro_f1_delta": "{:+.4f}"},
        ),
        "",
        "## Description-to-weight transfer generator controls",
        "",
        _markdown_table(
            dcwt_controls,
            columns=list(dcwt_controls.columns),
            labels=[
                "Generator",
                "Folds",
                "N held-out F1",
                "D held-out F1",
                "D−N",
                "D presence AP",
                "D presence F1",
                "D FP rows/100",
                "Evidence role",
            ],
            formats={
                "N_heldout_pair_micro_f1_mean": "{:.4f}",
                "D_heldout_pair_micro_f1_mean": "{:.4f}",
                "D_minus_N_heldout_pair_micro_f1_mean": "{:+.4f}",
                "D_presence_average_precision_mean": "{:.4f}",
                "D_presence_f1_mean": "{:.4f}",
                "D_false_positive_rows_per_100_mean": "{:.2f}",
            },
        ),
        "",
        "## Level 3 targeted robustness",
        "",
        _markdown_table(
            l3,
            columns=list(l3.columns),
            labels=[str(value) for value in l3.columns],
            formats={column: "{:.4f}" for column in l3.columns if "f1" in column.lower()},
        ),
        "",
        "Level 3 is post-hoc and appendix-only. All 12 historical dual-unseen folds are audited so that the selected canonical matching is shown beside the alternate cyclic matching. Artifact payloads do not contain the complete row identities and held-out score hashes required for outcome-blind exact reuse, so no inferential p-values are attached.",
        "",
        "## Level 4 parent-group stress tests",
        "",
        _markdown_table(
            l4,
            columns=["model", "parent_group_fold", "heldout_parent_group", "validation_reviews", "positive_review_support", "heldout_pair_micro_f1", "review_bootstrap_95_ci_low", "review_bootstrap_95_ci_high"],
            labels=["System", "Parent-group fold", "Held-out group", "Reviews", "Positive support", "Held-out pair F1", "95% CI low", "95% CI high"],
            formats={"heldout_pair_micro_f1": "{:.4f}", "review_bootstrap_95_ci_low": "{:.4f}", "review_bootstrap_95_ci_high": "{:.4f}"},
        ),
        "",
        "Parent groups are non-exchangeable stress cases and are reported separately; no cross-group significance claim is made.",
        "",
        "## Registered negative findings",
        "",
        _markdown_table(
            negative,
            columns=list(negative.columns),
            labels=["Finding", "System", "Estimate", "Evidence role"],
            formats={"estimate": "{:+.4f}"},
        ),
        "",
        "## Claim boundaries",
        "",
        "- `D−N` is matched within method and fold, but for trainable methods it also includes train–inference representation match.",
        "- Cross-method contrasts compare complete systems; QLoRA versus frozen Qwen is not a causal LoRA-only estimate.",
        "- Frozen few-shot versus zero-shot changes demonstrations, length/truncation and system-specific operating points.",
        "- L1 is a conventional development reference, not an unbiased taxonomy-generalisation-loss estimator.",
        "- Single-seed neural results remain a disclosed limitation.",
        "- No result in this record uses or opens the official test labels.",
        "",
        "## Freeze provenance",
        "",
        f"Manifest schema: `{manifest['schema_version']}`  ",
        f"Audited inputs: {len(manifest['registered_input_sha256'])}  ",
        f"Audited outputs: {len(manifest['reporting_output_sha256'])}  ",
        f"Audit files: {len(manifest['audit_sha256'])}  ",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    temporary.replace(args.output)
    print(json.dumps({"status": "pass", "output": str(args.output)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("docs/thesis_figure_data/taxonomy_scientific_freeze_v1"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/experiments/taxonomy_scientific_freeze_manifest_20260822.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/taxonomy_scientific_freeze_results_20260822.md"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
