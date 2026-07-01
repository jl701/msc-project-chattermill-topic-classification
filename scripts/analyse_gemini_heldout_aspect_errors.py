from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS
from msc_project.evaluation.metrics import PAIR_SEPARATOR, evaluate_pair_and_aspect, pair_to_aspect


DEFAULT_GEMINI = PROJECT_ROOT / "outputs" / "llm" / "gemini_candidate_label_20260701_0145_fixed_full"
DEFAULT_LOCAL = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered"
    / "example_filtered"
    / "best_test_predictions.jsonl"
)
DEFAULT_QWEN = (
    PROJECT_ROOT
    / "outputs"
    / "qwen_heldout_aspect_smoke"
    / "test_indexed_full"
    / "predictions_indexed.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sentiment(label: str) -> str:
    return label.rsplit(PAIR_SEPARATOR, maxsplit=1)[1]


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def row_key(row: dict[str, Any]) -> str:
    row_uid = row.get("row_uid")
    if row_uid:
        return str(row_uid)
    original_split = row.get("original_split", "test")
    return f"{original_split}:{row.get('id', '')}"


def row_sample_f1(gold_labels: set[str], pred_labels: set[str]) -> float:
    denominator = len(gold_labels) + len(pred_labels)
    return 2 * len(gold_labels & pred_labels) / denominator if denominator else 0.0


def prediction_cardinality(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    gold_counts = [len(row["gold_pair_labels"]) for row in rows]
    pred_counts = [len(row["pred_pair_labels"]) for row in rows]
    return {
        "rows": len(rows),
        "mean_gold_labels_per_row": mean(gold_counts) if gold_counts else 0.0,
        "mean_pred_labels_per_row": mean(pred_counts) if pred_counts else 0.0,
        "gold_label_count": int(sum(gold_counts)),
        "predicted_label_count": int(sum(pred_counts)),
        "empty_prediction_rows": int(sum(1 for count in pred_counts if count == 0)),
        "same_cardinality_rows": int(sum(1 for gold, pred in zip(gold_counts, pred_counts) if gold == pred)),
        "predicted_more_than_gold_rows": int(sum(1 for gold, pred in zip(gold_counts, pred_counts) if pred > gold)),
        "predicted_fewer_than_gold_rows": int(sum(1 for gold, pred in zip(gold_counts, pred_counts) if pred < gold)),
    }


def count_label_axis(rows: list[dict[str, Any]], labels: list[str], axis: str) -> dict[str, dict[str, float | int]]:
    if axis not in {"pair", "aspect", "sentiment"}:
        raise ValueError(f"Unknown label axis: {axis}")

    output: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            gold_pairs = set(row["gold_pair_labels"])
            pred_pairs = set(row["pred_pair_labels"])
            if axis == "pair":
                gold_values = gold_pairs
                pred_values = pred_pairs
            elif axis == "aspect":
                gold_values = {pair_to_aspect(pair) for pair in gold_pairs}
                pred_values = {pair_to_aspect(pair) for pair in pred_pairs}
            else:
                gold_values = {pair for pair in gold_pairs if sentiment(pair) == label}
                pred_values = {pair for pair in pred_pairs if sentiment(pair) == label}
                label = str(label)

            tp += int(label in gold_values and label in pred_values) if axis != "sentiment" else len(gold_values & pred_values)
            fp += int(label not in gold_values and label in pred_values) if axis != "sentiment" else len(pred_values - gold_values)
            fn += int(label in gold_values and label not in pred_values) if axis != "sentiment" else len(gold_values - pred_values)

        stats = precision_recall_f1(tp, fp, fn)
        stats["support"] = tp + fn
        stats["predicted"] = tp + fp
        output[str(label)] = stats
    return output


def llm_diagnostics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    diagnostics: dict[str, float | int] = {}
    if not rows:
        return diagnostics

    for key in ["valid_json", "schema_valid"]:
        if key in rows[0]:
            diagnostics[f"{key}_rate"] = sum(1 for row in rows if row.get(key)) / len(rows)

    seconds = [float(row["seconds"]) for row in rows if row.get("seconds") is not None]
    if seconds:
        diagnostics["mean_latency_seconds"] = mean(seconds)
        diagnostics["median_latency_seconds"] = median(seconds)

    for key in ["input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"]:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if values:
            diagnostics[key] = int(sum(int(value) for value in values))
    return diagnostics


def error_categories(rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        gold_pairs = set(row["gold_pair_labels"])
        pred_pairs = set(row["pred_pair_labels"])
        gold_aspects = {pair_to_aspect(label) for label in gold_pairs}
        pred_aspects = {pair_to_aspect(label) for label in pred_pairs}
        tags: list[str] = []

        if gold_pairs == pred_pairs:
            tags.append("exact")
        if gold_pairs and not pred_pairs:
            tags.append("missed_all")
        if gold_aspects - pred_aspects:
            tags.append("aspect_miss")
        if pred_aspects - gold_aspects:
            tags.append("aspect_overpredict")
        if any(pair_to_aspect(label) in pred_aspects and label not in pred_pairs for label in gold_pairs):
            tags.append("sentiment_error_when_aspect_predicted")
        if not tags:
            tags.append("partial_pair_mismatch")

        for tag in tags:
            counts[tag] += 1
            if len(examples[tag]) < max_examples:
                examples[tag].append(
                    {
                        "row_uid": row_key(row),
                        "gold_pair_labels": sorted(gold_pairs),
                        "pred_pair_labels": sorted(pred_pairs),
                    }
                )

    return {"counts": dict(sorted(counts.items())), "examples": dict(examples)}


def summarise_model(rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    pair_classes = candidate_pair_labels(DEFAULT_HELDOUT_ASPECTS)
    metrics = evaluate_pair_and_aspect(
        [row["gold_pair_labels"] for row in rows],
        [row["pred_pair_labels"] for row in rows],
        pair_classes,
    )
    sentiment_labels = ["negative", "neutral", "positive"]
    return {
        "metrics": metrics,
        "cardinality": prediction_cardinality(rows),
        "aspect_metrics": count_label_axis(rows, list(DEFAULT_HELDOUT_ASPECTS), "aspect"),
        "sentiment_metrics": count_label_axis(rows, sentiment_labels, "sentiment"),
        "pair_metrics": count_label_axis(rows, pair_classes, "pair"),
        "error_categories": error_categories(rows, max_examples),
        "llm_diagnostics": llm_diagnostics(rows),
    }


def compare_models(
    primary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    primary_name: str,
    comparison_name: str,
    max_examples: int,
) -> dict[str, Any]:
    comparison_by_key = {row_key(row): row for row in comparison_rows}
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    deltas: list[float] = []

    for primary in primary_rows:
        key = row_key(primary)
        comparison = comparison_by_key.get(key)
        if comparison is None:
            continue

        primary_gold = set(primary["gold_pair_labels"])
        comparison_gold = set(comparison["gold_pair_labels"])
        if primary_gold != comparison_gold:
            raise ValueError(f"Gold labels differ for {key}")

        primary_pred = set(primary["pred_pair_labels"])
        comparison_pred = set(comparison["pred_pair_labels"])
        primary_exact = primary_gold == primary_pred
        comparison_exact = comparison_gold == comparison_pred

        if primary_exact and comparison_exact:
            counts["both_exact"] += 1
        elif primary_exact:
            counts[f"{primary_name}_only_exact"] += 1
        elif comparison_exact:
            counts[f"{comparison_name}_only_exact"] += 1
        else:
            counts["neither_exact"] += 1

        delta = row_sample_f1(primary_gold, primary_pred) - row_sample_f1(comparison_gold, comparison_pred)
        deltas.append(delta)
        if len(examples) < max_examples and abs(delta) >= 0.5:
            examples.append(
                {
                    "row_uid": key,
                    "gold_pair_labels": sorted(primary_gold),
                    primary_name: sorted(primary_pred),
                    comparison_name: sorted(comparison_pred),
                    "row_sample_f1_delta": delta,
                }
            )

    counts["shared_rows"] = len(deltas)
    return {
        "counts": dict(sorted(counts.items())),
        f"mean_row_sample_f1_delta_{primary_name}_minus_{comparison_name}": mean(deltas) if deltas else 0.0,
        "examples": examples,
    }


def sampled_subset(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit >= len(rows):
        return list(rows)
    selected_positions = set(sorted(random.Random(seed).sample(list(range(len(rows))), limit)))
    return [row for row in rows if int(row["row_index"]) in selected_positions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Gemini fixed held-out-aspect prediction errors.")
    parser.add_argument("--gemini-dir", type=Path, default=DEFAULT_GEMINI)
    parser.add_argument("--local-predictions", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--qwen-predictions", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--pro-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "analysis" / "gemini_error_analysis")
    parser.add_argument("--pro-subset-limit", type=int, default=50)
    parser.add_argument("--pro-subset-seed", type=int, default=13)
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()

    model_rows: dict[str, list[dict[str, Any]]] = {}
    paths = {
        "gemini_flash_test": args.gemini_dir / "predictions_test_indexed.jsonl",
        "gemini_flash_validation": args.gemini_dir / "predictions_validation_indexed.jsonl",
        "local_distilbert_test": args.local_predictions,
        "qwen_test": args.qwen_predictions,
    }
    if args.pro_predictions is not None:
        paths["gemini_pro_test_subset"] = args.pro_predictions

    missing_paths = {}
    for name, path in paths.items():
        if path.exists():
            model_rows[name] = read_jsonl(path)
        else:
            missing_paths[name] = str(path)

    if "gemini_flash_test" not in model_rows:
        raise SystemExit(f"Missing required Gemini test predictions: {paths['gemini_flash_test']}")

    report: dict[str, Any] = {
        "task": "Gemini fixed held-out-aspect error analysis",
        "source_paths": {name: str(path) for name, path in paths.items()},
        "missing_paths": missing_paths,
        "models": {
            name: summarise_model(rows, args.max_examples)
            for name, rows in model_rows.items()
        },
        "comparisons": {},
    }

    for comparison_name in ["local_distilbert_test", "qwen_test"]:
        if comparison_name in model_rows:
            report["comparisons"][f"gemini_flash_test_vs_{comparison_name}"] = compare_models(
                model_rows["gemini_flash_test"],
                model_rows[comparison_name],
                "gemini_flash",
                comparison_name.replace("_test", ""),
                args.max_examples,
            )

    if "gemini_pro_test_subset" in model_rows:
        pro_keys = {row_key(row) for row in model_rows["gemini_pro_test_subset"]}
        flash_matching_pro = [row for row in model_rows["gemini_flash_test"] if row_key(row) in pro_keys]
        report["comparisons"]["gemini_pro_test_subset_vs_flash_same_rows"] = compare_models(
            model_rows["gemini_pro_test_subset"],
            flash_matching_pro,
            "gemini_pro",
            "gemini_flash",
            args.max_examples,
        )

    flash_subset = sampled_subset(
        model_rows["gemini_flash_test"],
        limit=args.pro_subset_limit,
        seed=args.pro_subset_seed,
    )
    report["pro_subset_plan"] = {
        "split": "test",
        "limit": args.pro_subset_limit,
        "sample": True,
        "seed": args.pro_subset_seed,
        "flash_same_subset": summarise_model(flash_subset, args.max_examples),
        "selected_row_uids": [row_key(row) for row in flash_subset],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved Gemini error analysis to {output_path}")


if __name__ == "__main__":
    main()
