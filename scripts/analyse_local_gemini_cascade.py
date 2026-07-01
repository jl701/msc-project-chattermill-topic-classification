from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS
from msc_project.evaluation.cascade import label_reliability, row_key
from msc_project.evaluation.metrics import evaluate_pair_and_aspect, pair_to_aspect, pair_to_components
from scripts.run_local_gemini_cascade import (
    Policy,
    add_features,
    aspect_classes,
    build_feature_rows,
    selected_keys_for_policy,
    sentiment_classes,
)


DEFAULT_LOCAL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered"
    / "example_filtered"
)
DEFAULT_PRO_DIR = PROJECT_ROOT / "outputs" / "llm" / "gemini_candidate_label_20260701_031040_pro_fixed_full"
DEFAULT_CASCADE_DIR = PROJECT_ROOT / "outputs" / "analysis" / "local_gemini_cascade_pro_grid1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "local_gemini_cascade_pro_deep_dive"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_classes() -> list[str]:
    return candidate_pair_labels(DEFAULT_HELDOUT_ASPECTS)


def sample_f1(gold_labels: list[str] | set[str], pred_labels: list[str] | set[str]) -> float:
    gold = set(gold_labels)
    pred = set(pred_labels)
    denominator = len(gold) + len(pred)
    return 2 * len(gold & pred) / denominator if denominator else 0.0


def label_sets(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    return set(row["gold_pair_labels"]), set(row["pred_pair_labels"])


def row_counts(gold: set[str], pred: set[str]) -> dict[str, int]:
    return {
        "tp": len(gold & pred),
        "fp": len(pred - gold),
        "fn": len(gold - pred),
    }


def error_tags(row: dict[str, Any]) -> list[str]:
    gold, pred = label_sets(row)
    gold_aspects = {pair_to_aspect(label) for label in gold}
    pred_aspects = {pair_to_aspect(label) for label in pred}
    tags: list[str] = []

    if gold == pred:
        tags.append("exact")
    if gold and not pred:
        tags.append("missed_all")
    if gold_aspects - pred_aspects:
        tags.append("aspect_miss")
    if pred_aspects - gold_aspects:
        tags.append("aspect_overpredict")
    if any(pair_to_aspect(label) in pred_aspects and label not in pred for label in gold):
        tags.append("sentiment_error_when_aspect_predicted")
    if not tags:
        tags.append("partial_pair_mismatch")
    return tags


def prediction_cardinality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_counts = [len(row["pred_pair_labels"]) for row in rows]
    gold_counts = [len(row["gold_pair_labels"]) for row in rows]
    return {
        "rows": len(rows),
        "gold_label_count": int(sum(gold_counts)),
        "predicted_label_count": int(sum(pred_counts)),
        "mean_gold_labels_per_row": mean(gold_counts) if gold_counts else 0.0,
        "mean_predicted_labels_per_row": mean(pred_counts) if pred_counts else 0.0,
        "empty_prediction_rows": int(sum(count == 0 for count in pred_counts)),
        "predicted_more_than_gold_rows": int(sum(pred > gold for pred, gold in zip(pred_counts, gold_counts))),
        "predicted_fewer_than_gold_rows": int(sum(pred < gold for pred, gold in zip(pred_counts, gold_counts))),
        "same_cardinality_rows": int(sum(pred == gold for pred, gold in zip(pred_counts, gold_counts))),
    }


def summarise_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tags = Counter(tag for row in rows for tag in error_tags(row))
    return {
        "metrics": evaluate_pair_and_aspect(
            [row["gold_pair_labels"] for row in rows],
            [row["pred_pair_labels"] for row in rows],
            pair_classes(),
        ),
        "cardinality": prediction_cardinality(rows),
        "error_categories": dict(sorted(tags.items())),
    }


def align_rows(*row_sets: list[dict[str, Any]]) -> list[tuple[dict[str, Any], ...]]:
    maps = [{row_key(row): row for row in rows} for rows in row_sets]
    keys = list(maps[0])
    aligned = []
    for key in keys:
        if any(key not in mapping for mapping in maps):
            raise ValueError(f"Missing row {key} in one or more prediction files.")
        rows = tuple(mapping[key] for mapping in maps)
        gold = {tuple(sorted(row["gold_pair_labels"])) for row in rows}
        if len(gold) != 1:
            raise ValueError(f"Gold labels differ for row {key}.")
        aligned.append(rows)
    return aligned


def model_win_counts(aligned_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    deltas = {
        "cascade_minus_pro": [],
        "cascade_minus_local": [],
        "pro_minus_local": [],
    }
    exact_patterns: Counter[str] = Counter()
    for local, pro, cascade in aligned_rows:
        gold = set(local["gold_pair_labels"])
        local_f1 = sample_f1(gold, local["pred_pair_labels"])
        pro_f1 = sample_f1(gold, pro["pred_pair_labels"])
        cascade_f1 = sample_f1(gold, cascade["pred_pair_labels"])
        deltas["cascade_minus_pro"].append(cascade_f1 - pro_f1)
        deltas["cascade_minus_local"].append(cascade_f1 - local_f1)
        deltas["pro_minus_local"].append(pro_f1 - local_f1)

        for left, right, name in [
            (cascade_f1, pro_f1, "cascade_vs_pro"),
            (cascade_f1, local_f1, "cascade_vs_local"),
            (pro_f1, local_f1, "pro_vs_local"),
        ]:
            if left > right:
                counts[f"{name}_better"] += 1
            elif left < right:
                counts[f"{name}_worse"] += 1
            else:
                counts[f"{name}_equal"] += 1

        exact_patterns[
            "local={};pro={};cascade={}".format(
                int(set(local["pred_pair_labels"]) == gold),
                int(set(pro["pred_pair_labels"]) == gold),
                int(set(cascade["pred_pair_labels"]) == gold),
            )
        ] += 1

    return {
        "row_counts": dict(sorted(counts.items())),
        "mean_row_sample_f1_delta": {
            key: mean(values) if values else 0.0
            for key, values in deltas.items()
        },
        "exact_match_patterns": dict(sorted(exact_patterns.items())),
    }


def pro_empty_analysis(aligned_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    selected = [rows for rows in aligned_rows if not rows[1]["pred_pair_labels"]]
    if not selected:
        return {"rows": 0}
    local_f1 = []
    pro_f1 = []
    cascade_f1 = []
    cascade_nonempty = 0
    for local, pro, cascade in selected:
        gold = set(local["gold_pair_labels"])
        local_f1.append(sample_f1(gold, local["pred_pair_labels"]))
        pro_f1.append(sample_f1(gold, pro["pred_pair_labels"]))
        cascade_f1.append(sample_f1(gold, cascade["pred_pair_labels"]))
        cascade_nonempty += int(bool(cascade["pred_pair_labels"]))
    return {
        "rows": len(selected),
        "cascade_nonempty_rows": cascade_nonempty,
        "mean_local_sample_f1": mean(local_f1),
        "mean_pro_sample_f1": mean(pro_f1),
        "mean_cascade_sample_f1": mean(cascade_f1),
        "mean_cascade_minus_pro": mean(cascade_f1) - mean(pro_f1),
    }


def improvement_source_counts(
    aligned_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    escalated_keys: set[str],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "cascade_better_than_pro": [],
        "cascade_worse_than_pro": [],
        "cascade_better_than_local": [],
    }
    max_examples = 8

    for local, pro, cascade in aligned_rows:
        key = row_key(local)
        gold = set(local["gold_pair_labels"])
        local_pred = set(local["pred_pair_labels"])
        pro_pred = set(pro["pred_pair_labels"])
        cascade_pred = set(cascade["pred_pair_labels"])
        local_f1 = sample_f1(gold, local_pred)
        pro_f1 = sample_f1(gold, pro_pred)
        cascade_f1 = sample_f1(gold, cascade_pred)
        local_counts = row_counts(gold, local_pred)
        pro_counts = row_counts(gold, pro_pred)
        cascade_counts = row_counts(gold, cascade_pred)

        if key in escalated_keys:
            counts["escalated_rows"] += 1
        else:
            counts["not_escalated_rows"] += 1

        if cascade_f1 > pro_f1:
            counts["cascade_better_than_pro_rows"] += 1
            if not pro_pred:
                counts["better_than_pro_from_pro_empty"] += 1
            if cascade_counts["fp"] < pro_counts["fp"]:
                counts["better_than_pro_reduced_fp"] += 1
            if cascade_counts["fn"] < pro_counts["fn"]:
                counts["better_than_pro_reduced_fn"] += 1
            if cascade_pred == local_pred and local_f1 > pro_f1:
                counts["better_than_pro_by_preserving_or_falling_back_to_local"] += 1
            add_example(examples["cascade_better_than_pro"], local, pro, cascade, local_f1, pro_f1, cascade_f1, max_examples)
        elif cascade_f1 < pro_f1:
            counts["cascade_worse_than_pro_rows"] += 1
            add_example(examples["cascade_worse_than_pro"], local, pro, cascade, local_f1, pro_f1, cascade_f1, max_examples)

        if cascade_f1 > local_f1:
            counts["cascade_better_than_local_rows"] += 1
            if cascade_counts["fp"] < local_counts["fp"]:
                counts["better_than_local_reduced_fp"] += 1
            if cascade_counts["fn"] < local_counts["fn"]:
                counts["better_than_local_reduced_fn"] += 1
            add_example(examples["cascade_better_than_local"], local, pro, cascade, local_f1, pro_f1, cascade_f1, max_examples)

    return {"counts": dict(sorted(counts.items())), "examples": examples}


def add_example(
    target: list[dict[str, Any]],
    local: dict[str, Any],
    pro: dict[str, Any],
    cascade: dict[str, Any],
    local_f1: float,
    pro_f1: float,
    cascade_f1: float,
    max_examples: int,
) -> None:
    if len(target) >= max_examples:
        return
    target.append(
        {
            "row_uid": row_key(local),
            "gold_pair_labels": sorted(local["gold_pair_labels"]),
            "local_pred_pair_labels": sorted(local["pred_pair_labels"]),
            "pro_pred_pair_labels": sorted(pro["pred_pair_labels"]),
            "cascade_pred_pair_labels": sorted(cascade["pred_pair_labels"]),
            "local_sample_f1": local_f1,
            "pro_sample_f1": pro_f1,
            "cascade_sample_f1": cascade_f1,
        }
    )


def label_count_table(
    aligned_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    label_axis: str,
) -> list[dict[str, Any]]:
    if label_axis == "pair":
        labels = pair_classes()
        transform = lambda values: set(values)
    elif label_axis == "aspect":
        labels = list(DEFAULT_HELDOUT_ASPECTS)
        transform = lambda values: {pair_to_aspect(label) for label in values}
    else:
        labels = ["negative", "neutral", "positive"]
        transform = lambda values: {pair_to_components(label)[1] for label in values}

    rows = []
    for label in labels:
        row: dict[str, Any] = {"label": label}
        for index, name in [(0, "local"), (1, "pro"), (2, "cascade")]:
            tp = fp = fn = 0
            for triple in aligned_rows:
                gold_values = transform(triple[0]["gold_pair_labels"])
                pred_values = transform(triple[index]["pred_pair_labels"])
                tp += int(label in gold_values and label in pred_values)
                fp += int(label not in gold_values and label in pred_values)
                fn += int(label in gold_values and label not in pred_values)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            row.update(
                {
                    f"{name}_tp": tp,
                    f"{name}_fp": fp,
                    f"{name}_fn": fn,
                    f"{name}_precision": precision,
                    f"{name}_recall": recall,
                    f"{name}_f1": f1,
                }
            )
        row["cascade_minus_pro_f1"] = row["cascade_f1"] - row["pro_f1"]
        row["cascade_minus_local_f1"] = row["cascade_f1"] - row["local_f1"]
        row["cascade_minus_pro_fp"] = row["cascade_fp"] - row["pro_fp"]
        row["cascade_minus_pro_fn"] = row["cascade_fn"] - row["pro_fn"]
        rows.append(row)
    return sorted(rows, key=lambda item: (item["cascade_minus_pro_f1"], item["cascade_f1"]), reverse=True)


def derive_escalated_keys(local_validation: list[dict[str, Any]], local_test: list[dict[str, Any]], summary: dict[str, Any]) -> set[str]:
    pair_stats = label_reliability(local_validation, pair_classes(), "pair")
    aspect_stats = label_reliability(local_validation, aspect_classes(), "aspect")
    sentiment_stats = label_reliability(local_validation, sentiment_classes(), "sentiment")
    feature_rows = build_feature_rows(local_test, pair_stats, aspect_stats, sentiment_stats)
    local_test_copy = [dict(row) for row in local_test]
    add_features(local_test_copy, feature_rows)
    policy = Policy(**summary["selected_by_validation"]["policy"])
    return set(selected_keys_for_policy(local_test_copy, policy))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse why a local-to-Gemini cascade differs from pure Gemini.")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--gemini-dir", type=Path, default=DEFAULT_PRO_DIR)
    parser.add_argument("--cascade-dir", type=Path, default=DEFAULT_CASCADE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    local_validation = read_jsonl(args.local_dir / "best_validation_predictions.jsonl")
    local_test = read_jsonl(args.local_dir / "best_test_predictions.jsonl")
    pro_test = read_jsonl(args.gemini_dir / "predictions_test_indexed.jsonl")
    cascade_test = read_jsonl(args.cascade_dir / "selected_test_predictions.jsonl")
    cascade_summary = json.loads((args.cascade_dir / "summary.json").read_text(encoding="utf-8"))

    aligned = align_rows(local_test, pro_test, cascade_test)
    escalated_keys = derive_escalated_keys(local_validation, local_test, cascade_summary)
    if len(escalated_keys) != cascade_summary["selected_by_validation"]["test_metrics"]["gemini_call_count"]:
        raise ValueError("Derived escalation count does not match the cascade summary.")

    pair_rows = label_count_table(aligned, "pair")
    aspect_rows = label_count_table(aligned, "aspect")
    sentiment_rows = label_count_table(aligned, "sentiment")
    report = {
        "task": "Local-to-Gemini Pro cascade deep-dive",
        "source_paths": {
            "local_dir": str(args.local_dir),
            "gemini_dir": str(args.gemini_dir),
            "cascade_dir": str(args.cascade_dir),
        },
        "selected_policy": cascade_summary["selected_by_validation"]["policy"],
        "models": {
            "local": summarise_model(local_test),
            "pro": summarise_model(pro_test),
            "cascade": summarise_model(cascade_test),
        },
        "row_level": {
            **model_win_counts(aligned),
            "pro_empty_recovery": pro_empty_analysis(aligned),
            "improvement_sources": improvement_source_counts(aligned, escalated_keys),
        },
        "label_level": {
            "pair": pair_rows,
            "aspect": aspect_rows,
            "sentiment": sentiment_rows,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, args.output_dir / "summary.json")
    write_csv(pair_rows, args.output_dir / "pair_label_comparison.csv")
    write_csv(aspect_rows, args.output_dir / "aspect_label_comparison.csv")
    write_csv(sentiment_rows, args.output_dir / "sentiment_label_comparison.csv")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "cascade_pair_samples_f1": report["models"]["cascade"]["metrics"]["pair_samples_f1"],
                "pro_pair_samples_f1": report["models"]["pro"]["metrics"]["pair_samples_f1"],
                "cascade_better_than_pro_rows": report["row_level"]["row_counts"]["cascade_vs_pro_better"],
                "cascade_worse_than_pro_rows": report["row_level"]["row_counts"]["cascade_vs_pro_worse"],
                "pro_empty_recovered_rows": report["row_level"]["pro_empty_recovery"]["cascade_nonempty_rows"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved local-to-Gemini cascade deep-dive to {args.output_dir}")


if __name__ == "__main__":
    main()
