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


def choose_predictions(local_labels: list[str], qwen_labels: list[str], policy: str) -> list[str]:
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
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return sorted(output)


def qwen_call_count(local_labels: list[str], policy: str) -> int:
    local_nonempty = bool(local_labels)
    if policy in {"local_only"}:
        return 0
    if policy in {"qwen_only", "qwen_nonempty_else_local", "union_pairs"}:
        return 1
    if policy == "local_nonempty_else_qwen":
        return 0 if local_nonempty else 1
    if policy in {"pair_agreement", "aspect_agreement_local_sentiment", "aspect_agreement_qwen_sentiment"}:
        return 1 if local_nonempty else 0
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
        choose_predictions(local_row["pred_pair_labels"], qwen_row["pred_pair_labels"], policy)
        for local_row, qwen_row in aligned
    ]
    metrics = evaluate_pair_and_aspect(gold, pred, candidate_pair_labels([aspect]))
    qwen_calls = sum(qwen_call_count(local_row["pred_pair_labels"], policy) for local_row, _ in aligned)
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


def global_validation_policy(rows: list[dict[str, Any]]) -> str:
    aggregate_rows = [aggregate(rows, "validation", policy) for policy in POLICIES]
    return max(
        aggregate_rows,
        key=lambda row: (
            float(row[f"{PRIMARY_METRIC}_mean"]),
            float(row["pair_micro_precision_mean"]),
            -float(row["pair_false_positive_rows_per_100_mean"]),
            -float(row["qwen_call_rate_mean"]),
        ),
    )["policy"]


def analyse(args: argparse.Namespace) -> dict[str, Any]:
    aspects = [
        str(row["heldout_aspect"])
        for row in read_json(args.qwen_validation_dir / "summary.json")["results"]
        if not row.get("dry_run")
    ]
    per_aspect_rows = []
    for aspect in aspects:
        for split, qwen_dir in [("validation", args.qwen_validation_dir), ("test", args.qwen_test_dir)]:
            local_rows = read_jsonl(local_path(args.local_loao_dir, split, aspect))
            qwen_rows = read_jsonl(split_path(qwen_dir, split, aspect))
            for policy in POLICIES:
                per_aspect_rows.append(evaluate_policy(local_rows, qwen_rows, aspect, split, policy))

    aggregate_rows = [
        aggregate(per_aspect_rows, split, policy)
        for split in ["validation", "test"]
        for policy in POLICIES
    ]
    global_policy = global_validation_policy(per_aspect_rows)
    global_selected = [
        row
        for row in per_aspect_rows
        if row["split"] == "test" and row["policy"] == global_policy
    ]
    per_aspect_selected = validation_selected_rows(per_aspect_rows, aspects, "test")
    selected_rows = []
    for name, rows in [
        ("global_validation_selected", global_selected),
        ("per_aspect_validation_selected", per_aspect_selected),
    ]:
        selected_rows.append(
            {
                "selection": name,
                "split": "test",
                "selected_policy": global_policy if name == "global_validation_selected" else "mixed",
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

    output = {
        "task": "Local DistilBERT to Qwen LOAO offline cascade analysis",
        "local_loao_dir": str(args.local_loao_dir),
        "qwen_validation_dir": str(args.qwen_validation_dir),
        "qwen_test_dir": str(args.qwen_test_dir),
        "policies": list(POLICIES),
        "primary_metric": PRIMARY_METRIC,
        "limitation": (
            "This is an offline prediction-combination diagnostic. Existing LOAO local predictions do not "
            "include row-level score/margin features, so it does not yet implement a true confidence-based "
            "uncertainty gate."
        ),
        "global_validation_selected_policy": global_policy,
        "aggregate": aggregate_rows,
        "selected": selected_rows,
    }
    write_csv(args.output_dir / "per_aspect_policy_results.csv", per_aspect_rows)
    write_csv(args.output_dir / "aggregate_policy_results.csv", aggregate_rows)
    write_csv(args.output_dir / "selected_policy_results.csv", selected_rows)
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
    args = parser.parse_args()
    summary = analyse(args)
    print(json.dumps(summary["selected"], indent=2), flush=True)
    print(f"Saved local-to-Qwen LOAO cascade analysis to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
