"""Build final validation-only comparison tables, figures, and claim matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.data.fabsa import default_data_dir  # noqa: E402
from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_post_supervisor import (  # noqa: E402
    post_supervisor_l2_folds,
)
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _condition_view(value: dict[str, Any]) -> dict[str, Any]:
    return value["L2_E"]["partitions"]["heldout"]


def _stage_view(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = value["L2_S"]
    return (
        stage["aspect_presence"],
        stage["oracle_aspect_gated_sentiment"]["capped_two_pair_metrics"],
    )


def _local_fold_rows(
    local_root: Path, dcwt_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in (
        "strict_train_only_tfidf",
        "e5_base_v2",
        "frozen_qwen_candidate_pair",
    ):
        for path in sorted((local_root / method / "folds").glob("l2-a*.json")):
            payload = _read_json(path)
            for condition in ("N", "D"):
                value = payload["conditions"][condition]
                heldout = _condition_view(value)
                overall = value["L2_E"]["partitions"]["overall"]
                seen = value["L2_E"]["partitions"]["seen"]
                aspect, oracle = _stage_view(value)
                rows.append(
                    {
                        "method_id": method,
                        "fold_id": payload["fold_id"],
                        "condition": condition,
                        "heldout_aspects": payload["heldout_aspect"],
                        "heldout_pair_micro_f1": heldout["pair_micro_f1"],
                        "overall_pair_micro_f1": overall["pair_micro_f1"],
                        "seen_pair_micro_f1": seen["pair_micro_f1"],
                        "heldout_presence_f1": aspect["f1"],
                        "heldout_presence_ap": aspect["average_precision"],
                        "oracle_gated_sentiment_set_micro_f1": oracle[
                            "pair_micro_f1"
                        ],
                        "heldout_positive_review_support": aspect["positive_rows"],
                        "heldout_predicted_positive_reviews": aspect[
                            "predicted_positive_rows"
                        ],
                    }
                )
    for path in sorted((dcwt_root / "folds").glob("l2-a*.json")):
        payload = _read_json(path)
        generator = payload["generators"]["kernel_ridge"]
        for condition in ("N", "D"):
            value = generator["conditions"][condition]
            heldout = _condition_view(value)
            overall = value["L2_E"]["partitions"]["overall"]
            seen = value["L2_E"]["partitions"]["seen"]
            aspect, oracle = _stage_view(value)
            rows.append(
                {
                    "method_id": "description_conditioned_weight_transfer_kernel_ridge",
                    "fold_id": payload["fold_id"],
                    "condition": condition,
                    "heldout_aspects": payload["heldout_aspect"],
                    "heldout_pair_micro_f1": heldout["pair_micro_f1"],
                    "overall_pair_micro_f1": overall["pair_micro_f1"],
                    "seen_pair_micro_f1": seen["pair_micro_f1"],
                    "heldout_presence_f1": aspect["f1"],
                    "heldout_presence_ap": aspect["average_precision"],
                    "oracle_gated_sentiment_set_micro_f1": oracle[
                        "pair_micro_f1"
                    ],
                    "heldout_positive_review_support": aspect["positive_rows"],
                    "heldout_predicted_positive_reviews": aspect[
                        "predicted_positive_rows"
                    ],
                }
            )
    return rows


def _stage_table(
    formal_results: Path, local_root: Path, dcwt_root: Path
) -> pd.DataFrame:
    formal = pd.read_csv(formal_results)
    formal = formal[(formal["level"] == "L2") & formal["condition"].isin(["N", "D"])]
    formal_rows = formal[
        [
            "method_id",
            "fold_id",
            "condition",
            "heldout_aspects",
            "heldout_pair_micro_f1",
            "overall_pair_micro_f1",
            "seen_pair_micro_f1",
            "heldout_presence_f1",
            "heldout_presence_ap",
            "oracle_pair_micro_f1",
            "heldout_presence_positive_rows",
            "heldout_presence_predicted_positive_rows",
        ]
    ].rename(
        columns={
            "oracle_pair_micro_f1": "oracle_gated_sentiment_set_micro_f1",
            "heldout_presence_positive_rows": "heldout_positive_review_support",
            "heldout_presence_predicted_positive_rows": "heldout_predicted_positive_reviews",
        }
    )
    local_rows = pd.DataFrame(_local_fold_rows(local_root, dcwt_root))
    result = pd.concat([formal_rows, local_rows], ignore_index=True)
    result["method"] = result["method_id"].map(METHOD_LABELS)
    keys = ["method_id", "fold_id", "condition"]
    if result.duplicated(keys).any() or len(result) != 7 * 12 * 2:
        raise ValueError("Seven-method stage-decomposition evidence is incomplete.")
    if not np.isfinite(
        result[
            [
                "heldout_pair_micro_f1",
                "overall_pair_micro_f1",
                "seen_pair_micro_f1",
                "heldout_presence_f1",
                "heldout_presence_ap",
                "oracle_gated_sentiment_set_micro_f1",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Stage-decomposition table contains non-finite values.")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _fold_diagnostics(stage: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    frame = load_official_fabsa_splits(data_dir, ("train", "validation"))
    original_train = int(frame["original_split"].eq("train").sum())
    support = (
        stage[stage["condition"].eq("D")]
        .groupby(["fold_id", "heldout_aspects"], as_index=False)[
            "heldout_positive_review_support"
        ]
        .nunique()
    )
    if not support["heldout_positive_review_support"].eq(1).all():
        raise ValueError("Methods disagree on held-out support.")
    rows = []
    for fold in post_supervisor_l2_folds():
        splits = build_taxonomy_fold_splits(
            frame, fold, evaluation_splits=("validation",)
        )
        fold_stage = stage[
            stage["fold_id"].eq(fold.fold_id) & stage["condition"].eq("D")
        ]
        positive_support = int(fold_stage["heldout_positive_review_support"].iloc[0])
        rows.append(
            {
                "fold_id": fold.fold_id,
                "heldout_aspect": fold.heldout_aspects[0],
                "original_train_reviews": original_train,
                "filtered_train_reviews": len(splits["train"]),
                "removed_train_reviews": original_train - len(splits["train"]),
                "validation_reviews": len(splits["validation"]),
                "heldout_positive_review_support": positive_support,
            }
        )
    return pd.DataFrame(rows)


def _comparability_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contrast": "L2 held-out D minus N within method/fold",
                "methods": "all seven",
                "same_rows": True,
                "same_training_scope_and_parameters": True,
                "same_decoder_thresholds": True,
                "changed_component": "held-out aspect card gains frozen minimal definition",
                "status": "primary matched representation contrast",
                "claim_boundary": "includes train-inference format match for trainable systems",
            },
            {
                "contrast": "selected R2_positive_concat minus D",
                "methods": "TF-IDF; Frozen E5",
                "same_rows": True,
                "same_training_scope_and_parameters": True,
                "same_decoder_thresholds": True,
                "changed_component": "held-out card gains selected rich guidance",
                "status": "secondary confirmation after bounded development",
                "claim_boundary": "not a primary endpoint; no neural expansion",
            },
            {
                "contrast": "two-stage capped-two minus one-stage independent pairs",
                "methods": "TF-IDF; Frozen E5",
                "same_rows": True,
                "same_training_scope_and_parameters": False,
                "same_decoder_thresholds": False,
                "changed_component": "architecture and architecture-appropriate training/selection",
                "status": "matched systems architecture control",
                "claim_boundary": "not the isolated effect of one threshold",
            },
            {
                "contrast": "capped-two minus top-one sentiment",
                "methods": "TF-IDF; Frozen E5",
                "same_rows": True,
                "same_training_scope_and_parameters": True,
                "same_decoder_thresholds": "aspect threshold only",
                "changed_component": "thresholded runner-up is permitted",
                "status": "dataset-fidelity decoder control",
                "claim_boundary": "retained despite small validation F1 cost",
            },
            {
                "contrast": "QLoRA Qwen versus Frozen Qwen zero-shot",
                "methods": "Qwen",
                "same_rows": True,
                "same_training_scope_and_parameters": False,
                "same_decoder_thresholds": False,
                "changed_component": "task supervision/adaptation and operating point",
                "status": "descriptive systems comparison",
                "claim_boundary": "not a causal LoRA-only effect",
            },
            {
                "contrast": "Frozen Qwen few-shot versus zero-shot",
                "methods": "Qwen",
                "same_rows": True,
                "same_training_scope_and_parameters": False,
                "same_decoder_thresholds": False,
                "changed_component": "demonstrations, max length, truncation, operating point",
                "status": "descriptive prompting systems comparison",
                "claim_boundary": "not a causal demonstration-only effect",
            },
            {
                "contrast": "L1 to L2-D to L4-D",
                "methods": "within-system scenario profile",
                "same_rows": False,
                "same_training_scope_and_parameters": False,
                "same_decoder_thresholds": False,
                "changed_component": "taxonomy shift, filtered training, fold unit and support",
                "status": "descriptive stress-test profile",
                "claim_boundary": "not a common-grid causal curve",
            },
            {
                "contrast": "Historical L3 dual-unseen conditions and selected-versus-alternate matching",
                "methods": "TF-IDF; Frozen E5; Frozen Qwen",
                "same_rows": "not fully evidenced in payload",
                "same_training_scope_and_parameters": "partially evidenced",
                "same_decoder_thresholds": "recorded but no full score identities/hashes",
                "changed_component": "two held-out cards",
                "status": "all 12 historical folds audited; post-hoc targeted robustness; appendix only",
                "claim_boundary": "descriptive metrics; no inferential p-values",
            },
        ]
    )


def _selected_r_table(path: Path, stage: pd.DataFrame) -> pd.DataFrame:
    audit = _read_json(path)
    if (
        audit.get("status") != "pass"
        or audit.get("failure_count") != 0
        or audit.get("non_finite_value_count") != 0
        or audit.get("resume_conflict_count") != 0
        or audit.get("test_contract_count") != 0
        or audit.get("development", {}).get("selected_interface")
        != "R2_positive_concat"
        or set(audit.get("confirmation", {}))
        != {"strict_train_only_tfidf", "e5_base_v2"}
    ):
        raise ValueError("Unsafe or contaminated selected rich-card audit.")
    rows = []
    for method, value in audit["confirmation"].items():
        if value.get("fold_count") != 12 or not value.get(
            "D_exactly_matches_prior_control"
        ):
            raise ValueError(f"Incomplete selected rich-card confirmation: {method}")
        effects = value["R_minus_D"]
        effect = effects["heldout_pair_micro_f1"]
        d_heldout = float(
            stage[
                stage["method_id"].eq(method) & stage["condition"].eq("D")
            ]["heldout_pair_micro_f1"].mean()
        )
        rows.append(
            {
                "method_id": method,
                "method": METHOD_LABELS[method],
                "selected_rich_interface": audit["development"]["selected_interface"],
                "fold_count": value["fold_count"],
                "D_heldout_pair_micro_f1_mean": d_heldout,
                "R_heldout_pair_micro_f1_mean": d_heldout + effect["mean"],
                "R_minus_D_heldout_pair_micro_f1_mean": effect["mean"],
                "R_higher_folds": effect["R_higher_folds"],
                "equal_folds": effect["equal_folds"],
                "R_lower_folds": effect["R_lower_folds"],
                "R_minus_D_presence_average_precision_mean": effects[
                    "presence_average_precision"
                ]["mean"],
                "R_minus_D_presence_f1_mean": effects["presence_f1"]["mean"],
                "R_minus_D_presence_precision_mean": effects[
                    "presence_precision"
                ]["mean"],
                "R_minus_D_presence_recall_mean": effects["presence_recall"][
                    "mean"
                ],
                "R_minus_D_false_positive_rows_per_100_mean": effects[
                    "false_positive_rows_per_100"
                ]["mean"],
                "R_minus_D_overall_pair_micro_f1_mean": effects[
                    "overall_pair_micro_f1"
                ]["mean"],
                "D_exactly_matches_prior_control": value[
                    "D_exactly_matches_prior_control"
                ],
                "status": "selected secondary confirmation; not primary",
            }
        )
    return pd.DataFrame(rows)


def _controls_table(root: Path) -> pd.DataFrame:
    rows = []
    matched = root / "taxonomy_matched_one_vs_two_stage_validation_v1"
    capped = root / "taxonomy_capped_two_sentiment_validation_v2"
    for audit_path in (matched / "audit.json", capped / "audit.json"):
        audit = _read_json(audit_path)
        if any(
            int(audit.get(key, 0)) != 0
            for key in (
                "failure_count",
                "non_finite_value_count",
                "resume_conflict_count",
                "test_contract_count",
                "official_test_artifact_count",
            )
        ):
            raise ValueError(f"Unsafe architecture/decoder control audit: {audit_path}")
    for method in ("strict_train_only_tfidf", "e5_base_v2"):
        value = _read_json(matched / method / "summary.json")
        for record in value["aggregate"]:
            rows.append(
                {
                    "study": "matched one-stage versus two-stage",
                    "method_id": method,
                    "method": METHOD_LABELS[method],
                    "variant": record["decoder"],
                    "pair_micro_f1_mean": record["pair_micro_f1_mean"],
                    "delta_reference": "one_stage_independent_pairs",
                    "pair_micro_f1_delta": record[
                        "pair_micro_f1_delta_vs_one_stage_mean"
                    ],
                    "test_contract_count": 0,
                }
            )
        value = _read_json(capped / method / "summary.json")
        for record in value["aggregate"]:
            rows.append(
                {
                    "study": "capped-two versus top-one",
                    "method_id": method,
                    "method": METHOD_LABELS[method],
                    "variant": record["decoder"],
                    "pair_micro_f1_mean": record["pair_micro_f1_mean"],
                    "delta_reference": "argmax",
                    "pair_micro_f1_delta": record[
                        "pair_micro_f1_delta_vs_argmax_mean"
                    ],
                    "test_contract_count": 0,
                }
            )
    return pd.DataFrame(rows)


def _dcwt_controls_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "folds").glob("l2-a*.json")):
        payload = _read_json(path)
        if any(
            int(payload.get(key, 0)) != 0
            for key in (
                "failure_count",
                "non_finite_value_count",
                "resume_conflict_count",
                "test_contract_count",
            )
        ):
            raise ValueError(f"Unsafe DCWT fold payload: {path}")
        for generator, generator_payload in payload["generators"].items():
            for condition in ("N", "D"):
                value = generator_payload["conditions"][condition]
                heldout = _condition_view(value)
                presence, _ = _stage_view(value)
                rows.append(
                    {
                        "fold_id": payload["fold_id"],
                        "generator": generator,
                        "condition": condition,
                        "heldout_pair_micro_f1": heldout["pair_micro_f1"],
                        "presence_average_precision": presence["average_precision"],
                        "presence_f1": presence["f1"],
                        "false_positive_rows_per_100": presence[
                            "false_positive_rows_per_100"
                        ],
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 4 * 12 * 2:
        raise ValueError("DCWT generator-control evidence is incomplete.")
    aggregate = (
        frame.groupby(["generator", "condition"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            heldout_pair_micro_f1_mean=("heldout_pair_micro_f1", "mean"),
            presence_average_precision_mean=("presence_average_precision", "mean"),
            presence_f1_mean=("presence_f1", "mean"),
            false_positive_rows_per_100_mean=(
                "false_positive_rows_per_100",
                "mean",
            ),
        )
        .sort_values(["generator", "condition"], kind="stable")
    )
    result_rows = []
    for generator, group in aggregate.groupby("generator", sort=True):
        by_condition = group.set_index("condition")
        result_rows.append(
            {
                "generator": generator,
                "fold_count": int(by_condition.loc["D", "fold_count"]),
                "N_heldout_pair_micro_f1_mean": by_condition.loc[
                    "N", "heldout_pair_micro_f1_mean"
                ],
                "D_heldout_pair_micro_f1_mean": by_condition.loc[
                    "D", "heldout_pair_micro_f1_mean"
                ],
                "D_minus_N_heldout_pair_micro_f1_mean": (
                    by_condition.loc["D", "heldout_pair_micro_f1_mean"]
                    - by_condition.loc["N", "heldout_pair_micro_f1_mean"]
                ),
                "D_presence_average_precision_mean": by_condition.loc[
                    "D", "presence_average_precision_mean"
                ],
                "D_presence_f1_mean": by_condition.loc["D", "presence_f1_mean"],
                "D_false_positive_rows_per_100_mean": by_condition.loc[
                    "D", "false_positive_rows_per_100_mean"
                ],
                "evidence_role": "appendix generator diagnostic; no inferential p-value",
            }
        )
    return pd.DataFrame(result_rows)


def _negative_findings(
    effects: pd.DataFrame,
    rich: pd.DataFrame,
    controls: pd.DataFrame,
    dcwt_controls: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for value in effects.itertuples(index=False):
        if float(value.D_minus_N) <= 0:
            rows.append(
                {
                    "finding": "Minimal description did not improve the aspect-balanced held-out pair F1",
                    "method": METHOD_LABELS[str(value.method_id)],
                    "estimate": float(value.D_minus_N),
                    "evidence_role": "primary L2 matched contrast",
                }
            )
    for value in rich.itertuples(index=False):
        if float(value.R_minus_D_heldout_pair_micro_f1_mean) <= 0:
            rows.append(
                {
                    "finding": "Selected rich card did not improve over the minimal description",
                    "method": str(value.method),
                    "estimate": float(value.R_minus_D_heldout_pair_micro_f1_mean),
                    "evidence_role": "secondary confirmation",
                }
            )
    subset = controls[
        controls["study"].eq("capped-two versus top-one")
        & controls["variant"].eq("capped_two_threshold")
    ]
    for value in subset.itertuples(index=False):
        if float(value.pair_micro_f1_delta) <= 0:
            rows.append(
                {
                    "finding": "Dataset-faithful capped-two decoder had a small F1 cost",
                    "method": str(value.method),
                    "estimate": float(value.pair_micro_f1_delta),
                    "evidence_role": "sequential validation decoder control",
                }
            )
    kernel = dcwt_controls[dcwt_controls["generator"].eq("kernel_ridge")]
    alternatives = dcwt_controls[~dcwt_controls["generator"].eq("kernel_ridge")]
    if len(kernel) == 1 and not alternatives.empty:
        kernel_d = float(kernel.iloc[0]["D_heldout_pair_micro_f1_mean"])
        best_control = alternatives.sort_values(
            "D_heldout_pair_micro_f1_mean", ascending=False
        ).iloc[0]
        best_d = float(best_control["D_heldout_pair_micro_f1_mean"])
        if kernel_d <= best_d:
            rows.append(
                {
                    "finding": "Kernel-ridge description-to-weight transfer did not outperform its strongest simple generator control",
                    "method": "Description-to-weight transfer",
                    "estimate": kernel_d - best_d,
                    "evidence_role": (
                        "appendix generator diagnostic versus "
                        f"{best_control['generator']}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _l1_reference(path: Path) -> pd.DataFrame:
    payload = _read_json(path)
    if (
        payload.get("protocol_id") != "taxonomy_level1_closed_reference_v1"
        or payload.get("failure_count") != 0
        or payload.get("non_finite_value_count") != 0
        or payload.get("test_contract_count") != 0
    ):
        raise ValueError("Unsafe Level 1 development-reference artifact.")
    selected = payload["selected"]
    return pd.DataFrame(
        [
            {
                "method": "word+character TF-IDF OvR logistic regression",
                "validation_reviews": payload["validation_rows"],
                "pair_micro_f1": selected["pair_micro_f1"],
                "pair_macro_f1": selected["pair_macro_f1"],
                "aspect_micro_f1": selected["aspect_micro_f1"],
                "selected_c": selected["c"],
                "selected_threshold": selected["threshold"],
                "selection_and_evaluation_boundary": payload["selection_partition"],
                "interpretation": "closed-taxonomy development reference; not an unbiased generalisation-loss estimate",
            }
        ]
    )


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in outputs:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _figures(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    l4: pd.DataFrame,
    metrics: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    outputs: list[Path] = []
    order = list(METHOD_LABELS)
    positions = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    boxes = [
        (0.02, "Outer fold\nremove held-out\ntraining rows"),
        (0.23, "Stage 1\n12 supplied aspects\n12 presence scores"),
        (0.45, "Seen-only\naspect threshold\nselect 0..12 aspects"),
        (0.67, "Stage 2\n3 sentiment scores\nper selected aspect"),
        (0.87, "Decoder\ntop-1 + optional\nrunner-up sentiment"),
    ]
    for index, (x, label) in enumerate(boxes):
        ax.text(
            x,
            0.55,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#e8f1f8", "edgecolor": "#005eb8"},
        )
        if index < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[index + 1][0] - 0.075, 0.55),
                xytext=(x + 0.075, 0.55),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 1.5},
            )
    ax.text(
        0.34,
        0.13,
        "L2 matched intervention: held-out name (N) vs the same name + frozen minimal definition (D)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#333333",
    )
    ax.set_axis_off()
    ax.set_title("Supplied-taxonomy two-stage evaluation pipeline", pad=14)
    outputs.extend(_save_figure(fig, figure_dir / "two_stage_pipeline"))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), sharey=True)
    views = (
        (
            "aspect_balanced_heldout_pair_micro_f1",
            "aspect_balanced_review_bootstrap_95_ci_low",
            "aspect_balanced_review_bootstrap_95_ci_high",
            "Aspect-balanced held-out pair micro-F1",
        ),
        (
            "pooled_heldout_pair_micro_f1_sensitivity",
            "pooled_review_bootstrap_95_ci_low",
            "pooled_review_bootstrap_95_ci_high",
            "Pooled held-out pair micro-F1 (sensitivity)",
        ),
    )
    for ax, (point_col, low_col, high_col, title) in zip(axes, views):
        for offset, condition, colour in ((-0.10, "N", "#7f8c8d"), (0.10, "D", "#005eb8")):
            part = summary[summary["condition"].eq(condition)].set_index("method_id").loc[order]
            x = part[point_col].to_numpy()
            low = part[low_col].to_numpy()
            high = part[high_col].to_numpy()
            ax.errorbar(x, positions + offset, xerr=[x - low, high - x], fmt="o", capsize=3, label=condition, color=colour)
        ax.set_title(title)
        ax.set_xlabel("Pair micro-F1")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(positions, [METHOD_LABELS[value] for value in order])
    axes[1].legend(title="Held-out card")
    fig.suptitle("Level 2 performance by supplied held-out representation")
    outputs.extend(_save_figure(fig, figure_dir / "l2_model_condition_f1"))

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    part = effects.set_index("method_id").loc[order]
    y = part["D_minus_N"].to_numpy()
    low = part["paired_review_bootstrap_95_ci_low"].to_numpy()
    high = part["paired_review_bootstrap_95_ci_high"].to_numpy()
    ax.errorbar(y, positions, xerr=[y - low, high - y], fmt="o", capsize=3, color="#005eb8")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(positions, [METHOD_LABELS[value] for value in order])
    ax.set_xlabel("D minus N: aspect-balanced held-out pair micro-F1")
    ax.set_title("Level 2 minimal-description effects")
    ax.grid(axis="x", alpha=0.25)
    outputs.extend(_save_figure(fig, figure_dir / "l2_description_effect_forest"))

    diagnostics = metrics[metrics["condition"].eq("D")].set_index("method_id").loc[order]
    panels = (
        ("heldout_presence_average_precision_mean", "Stage 1 presence AP"),
        ("heldout_presence_f1_mean", "Thresholded presence F1"),
        ("oracle_gated_sentiment_set_micro_f1_mean", "Oracle-gated sentiment-set F1"),
        ("heldout_pair_micro_f1_mean", "End-to-end held-out pair F1"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 5.2), sharey=True)
    for ax, (column, title) in zip(axes, panels):
        ax.plot(diagnostics[column].to_numpy(), positions, "o", color="#005eb8")
        ax.set_xlim(left=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Fold mean")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(positions, [METHOD_LABELS[value] for value in order])
    fig.suptitle("Level 2-D two-stage diagnostic decomposition (point estimates)")
    outputs.extend(_save_figure(fig, figure_dir / "l2_stage_decomposition"))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), sharey=True)
    folds = list(dict.fromkeys(l4["parent_group_fold"].astype(str)))
    for ax, fold in zip(axes, folds):
        part = l4[l4["parent_group_fold"].eq(fold)].set_index("method_id").loc[
            [value for value in order if value in set(l4["method_id"])]
        ]
        methods = list(part.index)
        pos = np.arange(len(methods))
        y = part["heldout_pair_micro_f1"].to_numpy()
        low = part["review_bootstrap_95_ci_low"].to_numpy()
        high = part["review_bootstrap_95_ci_high"].to_numpy()
        ax.errorbar(y, pos, xerr=[y - low, high - y], fmt="o", capsize=3, color="#005eb8")
        ax.set_title(str(part["heldout_parent_group"].iloc[0]))
        ax.set_xlabel("Held-out pair micro-F1")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(pos, [METHOD_LABELS[value] for value in methods])
    fig.suptitle("Level 4 parent-group stress tests (separate, non-exchangeable folds)")
    outputs.extend(_save_figure(fig, figure_dir / "l4_parent_group_f1"))
    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_dir = args.table_dir.resolve()
    stage = _stage_table(
        args.formal_results.resolve(),
        args.local_root.resolve(),
        args.dcwt_root.resolve(),
    )
    diagnostics = _fold_diagnostics(stage, args.data_dir.resolve())
    model_condition = (
        stage.groupby(["method_id", "method", "condition"], as_index=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            heldout_pair_micro_f1_mean=("heldout_pair_micro_f1", "mean"),
            overall_pair_micro_f1_mean=("overall_pair_micro_f1", "mean"),
            seen_pair_micro_f1_mean=("seen_pair_micro_f1", "mean"),
            heldout_presence_f1_mean=("heldout_presence_f1", "mean"),
            heldout_presence_average_precision_mean=("heldout_presence_ap", "mean"),
            oracle_gated_sentiment_set_micro_f1_mean=(
                "oracle_gated_sentiment_set_micro_f1",
                "mean",
            ),
        )
        .sort_values(["method_id", "condition"], kind="stable")
    )
    matrix = _comparability_matrix()
    rich = _selected_r_table(args.rich_audit.resolve(), stage)
    controls = _controls_table(args.experimental_root.resolve())
    dcwt_controls = _dcwt_controls_table(args.dcwt_root.resolve())
    summary = pd.read_csv(table_dir / "l2_model_condition_review_bootstrap.csv")
    effects = pd.read_csv(table_dir / "l2_description_effects_review_bootstrap.csv")
    l4 = pd.read_csv(table_dir / "l4_parent_group_review_bootstrap.csv")
    negative = _negative_findings(effects, rich, controls, dcwt_controls)
    l1 = _l1_reference(args.l1_result.resolve())
    tables = {
        "l2_stage_decomposition_fold_results.csv": stage,
        "l2_complete_model_condition_metrics.csv": model_condition,
        "l2_fold_support_and_filtering.csv": diagnostics,
        "headline_contrast_comparability_matrix.csv": matrix,
        "selected_rich_description_confirmation.csv": rich,
        "architecture_and_decoder_controls.csv": controls,
        "dcwt_generator_controls.csv": dcwt_controls,
        "negative_findings.csv": negative,
        "l1_closed_taxonomy_development_reference.csv": l1,
    }
    hashes: dict[str, str] = {}
    for name, frame in tables.items():
        path = table_dir / name
        _atomic_csv(path, frame)
        hashes[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)
    figures = _figures(summary, effects, l4, model_condition, args.figure_dir.resolve())
    for path in figures:
        hashes[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)
    audit = {
        "schema_version": "taxonomy_scientific_freeze_reporting_v1",
        "status": "pass",
        "include_official_test": False,
        "official_test_opened": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "seven_method_stage_rows": len(stage),
        "model_condition_rows": len(model_condition),
        "fold_diagnostic_rows": len(diagnostics),
        "comparability_rows": len(matrix),
        "selected_rich_rows": len(rich),
        "control_rows": len(controls),
        "dcwt_control_rows": len(dcwt_controls),
        "negative_finding_rows": len(negative),
        "l1_reference_rows": len(l1),
        "artifact_sha256": hashes,
    }
    _write_json(args.audit_output.resolve(), audit)
    print(json.dumps(audit, indent=2), flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-results",
        type=Path,
        default=Path("docs/thesis_figure_data/taxonomy_scientific_freeze_v1/formal_fold_results.csv"),
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("outputs/experimental/taxonomy_scientific_freeze_v1/local_replay"),
    )
    parser.add_argument(
        "--dcwt-root",
        type=Path,
        default=Path("outputs/experimental/taxonomy_scientific_freeze_v1/dcwt_replay"),
    )
    parser.add_argument(
        "--experimental-root", type=Path, default=Path("outputs/experimental")
    )
    parser.add_argument(
        "--rich-audit",
        type=Path,
        default=Path(
            "docs/experiments/"
            "taxonomy_rich_description_validation_confirmation_v1_audit.json"
        ),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("docs/thesis_figure_data/taxonomy_scientific_freeze_v1"),
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("thesis/figures/taxonomy_scientific_freeze_v1")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir()
    )
    parser.add_argument(
        "--l1-result",
        type=Path,
        default=Path("outputs/experimental/taxonomy_level1_closed_reference_v1/result.json"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("docs/experiments/taxonomy_scientific_freeze_reporting_audit_20260822.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
