from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.evaluation.metrics import evaluate_pair_and_aspect


POLICIES = (
    "local_only",
    "qwen_only",
    "local_nonempty_else_qwen",
    "qwen_nonempty_else_local",
    "pair_agreement",
    "aspect_agreement_local_sentiment",
    "aspect_agreement_qwen_sentiment",
    "union_pairs",
)

TEST_COMPARISON_POLICIES = (
    "local_only",
    "qwen_only",
    "aspect_agreement_qwen_sentiment",
    "score_abs_replace_le_0.05",
)

SCORE_DISTANCE_BANDS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)
SENTIMENT_MARGIN_BANDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
PRIMARY_METRIC = "pair_micro_f1"


def aspect_slug(aspect: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", aspect).strip("_").lower()
    return slug or "aspect"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, Any]) -> str:
    row_uid = row.get("row_uid")
    if row_uid:
        return str(row_uid)
    return f"{row.get('original_split', '')}:{row.get('id', '')}"


def split_path(qwen_dir: Path, split: str, aspect: str) -> Path:
    return qwen_dir / "predictions" / aspect_slug(aspect) / f"predictions_{split}_indexed.jsonl"


def local_path(local_dir: Path, split: str, aspect: str) -> Path:
    return (
        local_dir
        / "cross_encoder"
        / "example_filtered"
        / aspect_slug(aspect)
        / f"best_{split}_predictions.jsonl"
    )


def as_set(labels: list[str]) -> set[str]:
    return set(labels or [])


def policy_threshold(policy: str) -> float:
    return float(policy.rsplit("_", maxsplit=1)[-1])


def asymmetric_score_policy_parts(policy: str) -> tuple[float, str, float] | None:
    match = re.fullmatch(
        r"score_asym_rescue_le_([0-9.]+)_(confirm|veto)_le_([0-9.]+)",
        policy,
    )
    if not match:
        return None
    return float(match.group(1)), match.group(2), float(match.group(3))


def local_score_state(local_row: dict[str, Any] | None, aspect: str | None) -> tuple[float, float, float, float]:
    if local_row is None or aspect is None:
        raise ValueError("Score-aware policies require local row features and a held-out aspect.")
    score_features = local_row.get("score_features") or {}
    scores = score_features.get("candidate_aspect_scores") or {}
    score = float(scores.get(aspect, score_features.get("top_score", 0.0)))
    threshold = float(score_features.get("threshold", 0.5))
    signed_distance = score - threshold
    return score, threshold, signed_distance, abs(signed_distance)


def local_sentiment_margin(local_row: dict[str, Any] | None, aspect: str | None) -> float:
    if local_row is None or aspect is None:
        raise ValueError("Sentiment-margin policies require local row features and a held-out aspect.")
    row_features = local_row.get("sentiment_features") or {}
    aspect_features = row_features.get(aspect) or {}
    return float(aspect_features.get("sentiment_margin", 1.0))


def choose_score_policy(
    local_labels: list[str],
    qwen_labels: list[str],
    policy: str,
    local_row: dict[str, Any] | None,
    aspect: str | None,
) -> set[str]:
    local = as_set(local_labels)
    qwen = as_set(qwen_labels)
    _, _, signed_distance, abs_distance = local_score_state(local_row, aspect)

    if policy.startswith("score_abs_replace_le_"):
        return qwen if abs_distance <= policy_threshold(policy) else local
    if policy.startswith("score_abs_qwen_nonempty_else_local_le_"):
        return qwen if abs_distance <= policy_threshold(policy) and qwen else local
    if policy.startswith("score_abs_agreement_qwen_sentiment_le_"):
        if abs_distance <= policy_threshold(policy) and local:
            return qwen if qwen else set()
        return local
    if policy.startswith("score_below_rescue_le_"):
        band = policy_threshold(policy)
        return qwen if not local and -band <= signed_distance < 0 else local
    if policy.startswith("score_above_confirm_le_"):
        band = policy_threshold(policy)
        if local and 0 <= signed_distance <= band:
            return qwen if qwen else set()
        return local
    asymmetric_parts = asymmetric_score_policy_parts(policy)
    if asymmetric_parts is not None:
        rescue_band, above_mode, above_band = asymmetric_parts
        if -rescue_band <= signed_distance < 0:
            return qwen if qwen else local
        if local and 0 <= signed_distance <= above_band:
            if above_mode == "confirm":
                return qwen if qwen else set()
            return local if qwen else set()
        return local
    raise ValueError(f"Unknown score-aware policy: {policy}")


def choose_sentiment_margin_policy(
    local_labels: list[str],
    qwen_labels: list[str],
    policy: str,
    local_row: dict[str, Any] | None,
    aspect: str | None,
) -> set[str]:
    local = as_set(local_labels)
    qwen = as_set(qwen_labels)
    margin = local_sentiment_margin(local_row, aspect)
    if policy.startswith("sentiment_margin_replace_le_"):
        return qwen if local and margin <= policy_threshold(policy) and qwen else local
    if policy.startswith("sentiment_margin_confirm_le_"):
        if local and margin <= policy_threshold(policy):
            return qwen if qwen else set()
        return local
    raise ValueError(f"Unknown sentiment-margin policy: {policy}")


def choose_predictions(
    local_labels: list[str],
    qwen_labels: list[str],
    policy: str,
    local_row: dict[str, Any] | None = None,
    qwen_row: dict[str, Any] | None = None,
    aspect: str | None = None,
) -> list[str]:
    local = as_set(local_labels)
    qwen = as_set(qwen_labels)
    if policy == "local_only":
        output = local
    elif policy == "qwen_only":
        output = qwen
    elif policy == "local_nonempty_else_qwen":
        output = local if local else qwen
    elif policy == "qwen_nonempty_else_local":
        output = qwen if qwen else local
    elif policy == "pair_agreement":
        output = local & qwen
    elif policy == "aspect_agreement_local_sentiment":
        output = local if local and qwen else set()
    elif policy == "aspect_agreement_qwen_sentiment":
        output = qwen if local and qwen else set()
    elif policy == "union_pairs":
        output = local | qwen
    elif policy.startswith("score_"):
        output = choose_score_policy(local_labels, qwen_labels, policy, local_row, aspect)
    elif policy.startswith("sentiment_margin_"):
        output = choose_sentiment_margin_policy(local_labels, qwen_labels, policy, local_row, aspect)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return sorted(output)


def qwen_call_count(
    local_labels: list[str],
    policy: str,
    local_row: dict[str, Any] | None = None,
    aspect: str | None = None,
) -> int:
    local_nonempty = bool(local_labels)
    if policy in {"local_only"}:
        return 0
    if policy in {"qwen_only", "qwen_nonempty_else_local", "union_pairs"}:
        return 1
    if policy == "local_nonempty_else_qwen":
        return 0 if local_nonempty else 1
    if policy in {"pair_agreement", "aspect_agreement_local_sentiment", "aspect_agreement_qwen_sentiment"}:
        return 1 if local_nonempty else 0
    if policy.startswith("score_abs_replace_le_") or policy.startswith("score_abs_qwen_nonempty_else_local_le_"):
        _, _, _, abs_distance = local_score_state(local_row, aspect)
        return 1 if abs_distance <= policy_threshold(policy) else 0
    if policy.startswith("score_abs_agreement_qwen_sentiment_le_"):
        _, _, _, abs_distance = local_score_state(local_row, aspect)
        return 1 if local_nonempty and abs_distance <= policy_threshold(policy) else 0
    if policy.startswith("score_below_rescue_le_"):
        _, _, signed_distance, _ = local_score_state(local_row, aspect)
        band = policy_threshold(policy)
        return 1 if not local_nonempty and -band <= signed_distance < 0 else 0
    if policy.startswith("score_above_confirm_le_"):
        _, _, signed_distance, _ = local_score_state(local_row, aspect)
        band = policy_threshold(policy)
        return 1 if local_nonempty and 0 <= signed_distance <= band else 0
    asymmetric_parts = asymmetric_score_policy_parts(policy)
    if asymmetric_parts is not None:
        rescue_band, _, above_band = asymmetric_parts
        _, _, signed_distance, _ = local_score_state(local_row, aspect)
        below_rescue = -rescue_band <= signed_distance < 0
        above_confirm_or_veto = local_nonempty and 0 <= signed_distance <= above_band
        return 1 if below_rescue or above_confirm_or_veto else 0
    if policy.startswith("sentiment_margin_replace_le_") or policy.startswith("sentiment_margin_confirm_le_"):
        margin = local_sentiment_margin(local_row, aspect)
        return 1 if local_nonempty and margin <= policy_threshold(policy) else 0
    raise ValueError(f"Unknown policy: {policy}")


def aligned_rows(local_rows: list[dict[str, Any]], qwen_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    qwen_by_key = {row_key(row): row for row in qwen_rows}
    aligned = []
    for local_row in local_rows:
        key = row_key(local_row)
        if key not in qwen_by_key:
            raise ValueError(f"Qwen predictions are missing row {key}")
        qwen_row = qwen_by_key[key]
        if sorted(local_row["gold_pair_labels"]) != sorted(qwen_row["gold_pair_labels"]):
            raise ValueError(f"Gold labels differ for row {key}")
        aligned.append((local_row, qwen_row))
    if len(aligned) != len(qwen_rows):
        raise ValueError(f"Row count mismatch: local={len(local_rows)} qwen={len(qwen_rows)}")
    return aligned


def evaluate_policy(
    local_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    aspect: str,
    split: str,
    policy: str,
) -> dict[str, Any]:
    aligned = aligned_rows(local_rows, qwen_rows)
    gold = [local_row["gold_pair_labels"] for local_row, _ in aligned]
    pred = [
        choose_predictions(
            local_row["pred_pair_labels"],
            qwen_row["pred_pair_labels"],
            policy,
            local_row=local_row,
            qwen_row=qwen_row,
            aspect=aspect,
        )
        for local_row, qwen_row in aligned
    ]
    metrics = evaluate_pair_and_aspect(gold, pred, candidate_pair_labels([aspect]))
    qwen_calls = sum(
        qwen_call_count(local_row["pred_pair_labels"], policy, local_row=local_row, aspect=aspect)
        for local_row, _ in aligned
    )
    row_count = len(aligned)
    metrics.update(
        {
            "heldout_aspect": aspect,
            "split": split,
            "policy": policy,
            "examples": row_count,
            "positive_gold_rows": sum(1 for labels in gold if labels),
            "qwen_call_rows": qwen_calls,
            "qwen_call_rate": qwen_calls / row_count if row_count else 0.0,
            "predicted_labels_per_example": sum(len(labels) for labels in pred) / row_count if row_count else 0.0,
            "gold_labels_per_example": sum(len(labels) for labels in gold) / row_count if row_count else 0.0,
        }
    )
    return metrics


def aggregate(rows: list[dict[str, Any]], split: str, policy: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split and row["policy"] == policy]
    output: dict[str, Any] = {"split": split, "policy": policy, "aspects": len(selected)}
    numeric_fields = [
        "pair_samples_f1",
        "pair_micro_f1",
        "pair_micro_precision",
        "pair_micro_recall",
        "pair_false_positive_rows_per_100",
        "pair_false_negative_rows_per_100",
        "aspect_micro_f1",
        "sentiment_accuracy_when_gold_aspect_predicted",
        "qwen_call_rate",
        "predicted_labels_per_example",
    ]
    for field in numeric_fields:
        values = [float(row[field]) for row in selected if field in row]
        if values:
            output[f"{field}_mean"] = mean(values)
            output[f"{field}_std"] = pstdev(values)
            output[f"{field}_min"] = min(values)
            output[f"{field}_max"] = max(values)
    return output


def best_policy(rows: list[dict[str, Any]], split: str, aspect: str) -> str:
    candidates = [row for row in rows if row["split"] == split and row["heldout_aspect"] == aspect]
    return max(
        candidates,
        key=lambda row: (
            float(row[PRIMARY_METRIC]),
            float(row["pair_micro_precision"]),
            -float(row["pair_false_positive_rows_per_100"]),
            -float(row["qwen_call_rate"]),
        ),
    )["policy"]


def validation_selected_rows(rows: list[dict[str, Any]], aspects: list[str], split: str) -> list[dict[str, Any]]:
    selected = []
    for aspect in aspects:
        policy = best_policy(rows, "validation", aspect)
        match = next(
            row
            for row in rows
            if row["split"] == split and row["heldout_aspect"] == aspect and row["policy"] == policy
        )
        selected.append({**match, "selection": "per_aspect_validation_selected", "selected_policy": policy})
    return selected


def global_validation_policy(rows: list[dict[str, Any]], policies: list[str]) -> str:
    aggregate_rows = [aggregate(rows, "validation", policy) for policy in policies]
    return max(
        aggregate_rows,
        key=lambda row: (
            float(row[f"{PRIMARY_METRIC}_mean"]),
            float(row["pair_micro_precision_mean"]),
            -float(row["pair_false_positive_rows_per_100_mean"]),
            -float(row["qwen_call_rate_mean"]),
        ),
    )["policy"]


def feature_status(local_loao_dir: Path, aspects: list[str]) -> dict[str, bool]:
    has_scores = True
    has_sentiment = True
    for aspect in aspects:
        rows = read_jsonl(local_path(local_loao_dir, "validation", aspect))
        if not rows:
            return {"score_features": False, "sentiment_features": False}
        has_scores = has_scores and all("score_features" in row for row in rows[:10])
        has_sentiment = has_sentiment and all("sentiment_features" in row for row in rows[:10])
    return {"score_features": has_scores, "sentiment_features": has_sentiment}


def policy_grid(status: dict[str, bool]) -> list[str]:
    policies = list(POLICIES)
    if status["score_features"]:
        for band in SCORE_DISTANCE_BANDS:
            suffix = f"{band:.2f}"
            policies.extend(
                [
                    f"score_abs_replace_le_{suffix}",
                    f"score_abs_qwen_nonempty_else_local_le_{suffix}",
                    f"score_abs_agreement_qwen_sentiment_le_{suffix}",
                    f"score_below_rescue_le_{suffix}",
                    f"score_above_confirm_le_{suffix}",
                ]
            )
        for rescue_band in SCORE_DISTANCE_BANDS:
            rescue_suffix = f"{rescue_band:.2f}"
            for above_band in SCORE_DISTANCE_BANDS:
                above_suffix = f"{above_band:.2f}"
                policies.extend(
                    [
                        f"score_asym_rescue_le_{rescue_suffix}_confirm_le_{above_suffix}",
                        f"score_asym_rescue_le_{rescue_suffix}_veto_le_{above_suffix}",
                    ]
                )
    if status["sentiment_features"]:
        for band in SENTIMENT_MARGIN_BANDS:
            suffix = f"{band:.2f}"
            policies.extend(
                [
                    f"sentiment_margin_replace_le_{suffix}",
                    f"sentiment_margin_confirm_le_{suffix}",
                ]
            )
    return policies


def policy_family(policy: str) -> str:
    if policy in {"local_only", "qwen_only"}:
        return policy
    if policy in {
        "local_nonempty_else_qwen",
        "qwen_nonempty_else_local",
        "pair_agreement",
        "aspect_agreement_local_sentiment",
        "aspect_agreement_qwen_sentiment",
        "union_pairs",
    }:
        return "prediction_combination"
    if policy.startswith("score_asym_rescue"):
        parts = asymmetric_score_policy_parts(policy)
        if parts is not None:
            return f"asymmetric_rescue_{parts[1]}"
        return "asymmetric_score"
    if policy.startswith("score_abs_"):
        return "symmetric_score_distance"
    if policy.startswith("score_below_"):
        return "below_threshold_rescue_only"
    if policy.startswith("score_above_"):
        return "above_threshold_confirm_only"
    if policy.startswith("sentiment_margin_"):
        return "sentiment_margin"
    return "other"


def is_pareto_efficient(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    quality_field: str = f"{PRIMARY_METRIC}_mean",
    cost_field: str = "qwen_call_rate_mean",
) -> bool:
    candidate_quality = float(candidate[quality_field])
    candidate_cost = float(candidate[cost_field])
    for row in rows:
        if row is candidate:
            continue
        quality = float(row[quality_field])
        cost = float(row[cost_field])
        if (
            quality >= candidate_quality
            and cost <= candidate_cost
            and (quality > candidate_quality or cost < candidate_cost)
        ):
            return False
    return True


def build_pareto_rows(aggregate_rows: list[dict[str, Any]], global_policy: str) -> list[dict[str, Any]]:
    complete_rows = [row for row in aggregate_rows if int(row.get("aspects", 0)) > 0]
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in complete_rows:
        by_split.setdefault(str(row["split"]), []).append(row)

    output = []
    for row in complete_rows:
        split_rows = by_split[str(row["split"])]
        output.append(
            {
                "split": row["split"],
                "policy": row["policy"],
                "policy_family": policy_family(str(row["policy"])),
                "aspects": row["aspects"],
                "pair_micro_f1_mean": row.get("pair_micro_f1_mean"),
                "pair_samples_f1_mean": row.get("pair_samples_f1_mean"),
                "precision_mean": row.get("pair_micro_precision_mean"),
                "recall_mean": row.get("pair_micro_recall_mean"),
                "false_positive_rows_per_100_mean": row.get("pair_false_positive_rows_per_100_mean"),
                "false_negative_rows_per_100_mean": row.get("pair_false_negative_rows_per_100_mean"),
                "qwen_call_rate_mean": row.get("qwen_call_rate_mean"),
                "is_pareto_frontier": int(is_pareto_efficient(row, split_rows)),
                "is_global_validation_selected": int(row["policy"] == global_policy),
                "is_test_comparison": int(row["policy"] in TEST_COMPARISON_POLICIES),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["split"]),
            float(row["qwen_call_rate_mean"]),
            -float(row["pair_micro_f1_mean"]),
            str(row["policy"]),
        ),
    )


def analyse(args: argparse.Namespace) -> dict[str, Any]:
    aspects = [
        str(row["heldout_aspect"])
        for row in read_json(args.qwen_validation_dir / "summary.json")["results"]
        if not row.get("dry_run")
    ]
    status = feature_status(args.local_loao_dir, aspects)
    policies = policy_grid(status)
    per_aspect_rows = []

    for aspect in aspects:
        local_rows = read_jsonl(local_path(args.local_loao_dir, "validation", aspect))
        qwen_rows = read_jsonl(split_path(args.qwen_validation_dir, "validation", aspect))
        for policy in policies:
            per_aspect_rows.append(evaluate_policy(local_rows, qwen_rows, aspect, "validation", policy))

    global_policy = global_validation_policy(per_aspect_rows, policies)
    per_aspect_policy = {aspect: best_policy(per_aspect_rows, "validation", aspect) for aspect in aspects}
    if args.evaluate_all_test_policies:
        test_policies = list(policies)
    else:
        test_policy_set = {policy for policy in TEST_COMPARISON_POLICIES if policy in policies}
        test_policy_set.add(global_policy)
        test_policies = [policy for policy in policies if policy in test_policy_set]

    test_rows_by_aspect_policy: dict[tuple[str, str], dict[str, Any]] = {}
    for aspect in aspects:
        local_rows = read_jsonl(local_path(args.local_loao_dir, "test", aspect))
        qwen_rows = read_jsonl(split_path(args.qwen_test_dir, "test", aspect))
        aspect_test_policies = set(test_policies)
        aspect_test_policies.add(per_aspect_policy[aspect])
        for policy in [candidate for candidate in policies if candidate in aspect_test_policies]:
            row = evaluate_policy(local_rows, qwen_rows, aspect, "test", policy)
            per_aspect_rows.append(row)
            test_rows_by_aspect_policy[(aspect, policy)] = row

    aggregate_rows = [
        aggregate(per_aspect_rows, "validation", policy)
        for policy in policies
    ]
    aggregate_rows.extend(aggregate(per_aspect_rows, "test", policy) for policy in test_policies)
    global_selected = [
        row
        for row in per_aspect_rows
        if row["split"] == "test" and row["policy"] == global_policy
    ]
    per_aspect_selected = [
        {
            **test_rows_by_aspect_policy[(aspect, per_aspect_policy[aspect])],
            "selection": "per_aspect_validation_selected",
            "selected_policy": per_aspect_policy[aspect],
        }
        for aspect in aspects
    ]
    selected_rows = []
    for name, rows in [
        ("global_validation_selected", global_selected),
        ("per_aspect_validation_selected", per_aspect_selected),
    ]:
        selected_policy = global_policy if name == "global_validation_selected" else "mixed"
        selected_rows.append(
            {
                "selection": name,
                "split": "test",
                "selected_policy": selected_policy,
                "selected_policy_family": policy_family(selected_policy) if selected_policy != "mixed" else "mixed",
                "aspects": len(rows),
                "pair_micro_f1_mean": mean(float(row["pair_micro_f1"]) for row in rows),
                "pair_micro_precision_mean": mean(float(row["pair_micro_precision"]) for row in rows),
                "pair_micro_recall_mean": mean(float(row["pair_micro_recall"]) for row in rows),
                "pair_samples_f1_mean": mean(float(row["pair_samples_f1"]) for row in rows),
                "pair_false_positive_rows_per_100_mean": mean(
                    float(row["pair_false_positive_rows_per_100"]) for row in rows
                ),
                "pair_false_negative_rows_per_100_mean": mean(
                    float(row["pair_false_negative_rows_per_100"]) for row in rows
                ),
                "qwen_call_rate_mean": mean(float(row["qwen_call_rate"]) for row in rows),
            }
        )
    pareto_rows = build_pareto_rows(aggregate_rows, global_policy)

    output = {
        "task": "Local DistilBERT to Qwen LOAO offline cascade analysis",
        "local_loao_dir": str(args.local_loao_dir),
        "qwen_validation_dir": str(args.qwen_validation_dir),
        "qwen_test_dir": str(args.qwen_test_dir),
        "local_feature_status": status,
        "policies": policies,
        "primary_metric": PRIMARY_METRIC,
        "test_policy_scope": "all" if args.evaluate_all_test_policies else "selected_comparisons",
        "test_comparison_policies": test_policies,
        "per_aspect_validation_selected_policies": per_aspect_policy,
        "limitation": (
            "This is an offline prediction-combination diagnostic. When the selected local LOAO directory "
            "contains score_features or sentiment_features, the policy grid includes validation-selected "
            "score/margin routing. It still reuses existing Qwen predictions rather than making new model calls."
        ),
        "global_validation_selected_policy": global_policy,
        "aggregate": aggregate_rows,
        "pareto": pareto_rows,
        "selected": selected_rows,
    }
    write_csv(args.output_dir / "per_aspect_policy_results.csv", per_aspect_rows)
    write_csv(args.output_dir / "aggregate_policy_results.csv", aggregate_rows)
    write_csv(args.output_dir / "pareto_policy_results.csv", pareto_rows)
    write_csv(args.output_dir / "selected_policy_results.csv", selected_rows)
    if args.public_pareto_csv is not None:
        write_csv(args.public_pareto_csv, pareto_rows)
    if args.public_selected_csv is not None:
        write_csv(args.public_selected_csv, selected_rows)
    write_json(output, args.output_dir / "summary.json")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse offline local-to-Qwen LOAO cascade policies.")
    parser.add_argument(
        "--local-loao-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "baselines"
        / "loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630",
    )
    parser.add_argument(
        "--qwen-validation-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "llm" / "qwen_loao_heldout_aspect_all_rows_validation_20260701",
    )
    parser.add_argument(
        "--qwen-test-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "llm" / "qwen_loao_heldout_aspect_all_rows_test_20260701",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "local_qwen_loao_cascade_20260702",
    )
    parser.add_argument(
        "--public-pareto-csv",
        type=Path,
        default=None,
        help="Optional tracked aggregate CSV for plot-ready F1/call-rate policy data.",
    )
    parser.add_argument(
        "--public-selected-csv",
        type=Path,
        default=None,
        help="Optional tracked aggregate CSV for validation-selected test results.",
    )
    parser.add_argument(
        "--evaluate-all-test-policies",
        action="store_true",
        help="Evaluate every policy on test. By default, test evaluation is limited to comparisons and selected rules.",
    )
    args = parser.parse_args()
    summary = analyse(args)
    print(json.dumps(summary["selected"], indent=2), flush=True)
    print(f"Saved local-to-Qwen LOAO cascade analysis to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
