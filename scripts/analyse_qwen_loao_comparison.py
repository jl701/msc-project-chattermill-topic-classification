from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


COMPARISON_METRICS = [
    "pair_samples_f1",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
    "pair_false_positive_rows_per_100",
    "pair_false_negative_rows_per_100",
    "aspect_micro_f1",
    "aspect_samples_f1",
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "presence_prevalence",
    "sentiment_accuracy_when_gold_aspect_predicted",
    "valid_json_rate",
    "schema_valid_rate",
    "seconds_per_example",
]

PER_ASPECT_METRICS = [
    "pair_samples_f1",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
    "pair_false_positive_rows_per_100",
    "pair_false_negative_rows_per_100",
    "aspect_micro_f1",
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "presence_prevalence",
    "sentiment_accuracy_when_gold_aspect_predicted",
]

DOCUMENTED_LEXICAL_LOAO_ROW = {
    "system": "Documented lexical TF-IDF + global sentiment LOAO micro-selected lower bound",
    "split": "test",
    "aspects": 12,
    "pair_samples_f1_mean": 0.0903,
    "pair_micro_f1_mean": 0.3780,
    "pair_micro_precision_mean": 0.3778,
    "pair_micro_recall_mean": 0.4895,
    "pair_false_positive_rows_per_100_mean": 17.4753,
    "source_note": (
        "Recorded in docs/generalisation_baselines.md and docs/loao_heldout_aspect.md; "
        "the raw local output directory is not required for this Qwen-vs-DistilBERT comparison."
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def summary_results(run_dir: Path) -> list[dict[str, Any]]:
    return [row for row in read_json(run_dir / "summary.json")["results"] if not row.get("dry_run")]


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") == split]


def distilbert_rows(run_dir: Path, split: str) -> list[dict[str, Any]]:
    return split_rows(read_csv_rows(run_dir / "all_results.csv"), split)


def aggregate_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, float] | None:
    values = [value for value in (number(row.get(metric)) for row in rows) if value is not None]
    if not values:
        return None
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def aggregate_row(system: str, split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "system": system,
        "split": split,
        "aspects": len({str(item.get("heldout_aspect")) for item in rows}),
    }
    for metric in COMPARISON_METRICS:
        stats = aggregate_metric(rows, metric)
        if not stats:
            continue
        for key, value in stats.items():
            row[f"{metric}_{key}"] = value
    total_seconds = sum(number(item.get("seconds")) or 0.0 for item in rows)
    total_examples = sum(int(item.get("examples") or 0) for item in rows)
    if total_seconds:
        row["total_seconds"] = total_seconds
    if total_examples:
        row["total_examples"] = total_examples
    if total_seconds and total_examples:
        row["weighted_seconds_per_example"] = total_seconds / total_examples
    return row


def compare_per_aspect(
    qwen_rows: list[dict[str, Any]],
    distilbert_rows_: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qwen_by_aspect = {str(row["heldout_aspect"]): row for row in qwen_rows}
    distilbert_by_aspect = {str(row["heldout_aspect"]): row for row in distilbert_rows_}
    missing = sorted(set(qwen_by_aspect) ^ set(distilbert_by_aspect))
    if missing:
        raise ValueError(f"Qwen and DistilBERT aspects do not align: {missing}")

    rows = []
    for aspect in sorted(qwen_by_aspect):
        qwen = qwen_by_aspect[aspect]
        distilbert = distilbert_by_aspect[aspect]
        row: dict[str, Any] = {"heldout_aspect": aspect}
        for prefix, source in [("qwen", qwen), ("distilbert", distilbert)]:
            for metric in PER_ASPECT_METRICS:
                row[f"{prefix}_{metric}"] = number(source.get(metric))
        for metric in PER_ASPECT_METRICS:
            qwen_value = row[f"qwen_{metric}"]
            distilbert_value = row[f"distilbert_{metric}"]
            if qwen_value is not None and distilbert_value is not None:
                row[f"delta_{metric}"] = qwen_value - distilbert_value
        delta = row.get("delta_pair_micro_f1")
        row["winner_pair_micro"] = "Qwen" if delta and delta > 0 else "DistilBERT" if delta and delta < 0 else "Tie"
        rows.append(row)
    return rows


def positive_gap_rows(qwen_rows: list[dict[str, Any]], positive_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qwen_by_aspect = {str(row["heldout_aspect"]): row for row in qwen_rows}
    positive_by_aspect = {str(row["heldout_aspect"]): row for row in positive_rows}
    missing = sorted(set(qwen_by_aspect) ^ set(positive_by_aspect))
    if missing:
        raise ValueError(f"Qwen all-row and positive diagnostic aspects do not align: {missing}")

    rows = []
    for aspect in sorted(qwen_by_aspect):
        all_row = qwen_by_aspect[aspect]
        positive = positive_by_aspect[aspect]
        row: dict[str, Any] = {"heldout_aspect": aspect}
        for metric in [
            "pair_samples_f1",
            "pair_micro_f1",
            "pair_micro_precision",
            "pair_micro_recall",
            "pair_macro_f1",
            "sentiment_accuracy_when_gold_aspect_predicted",
        ]:
            all_value = number(all_row.get(metric))
            positive_value = number(positive.get(metric))
            row[f"all_row_{metric}"] = all_value
            row[f"positive_gold_{metric}"] = positive_value
            if all_value is not None and positive_value is not None:
                row[f"gap_{metric}"] = positive_value - all_value
        row["all_row_fp_rows_per_100"] = number(all_row.get("pair_false_positive_rows_per_100"))
        row["all_row_fn_rows_per_100"] = number(all_row.get("pair_false_negative_rows_per_100"))
        rows.append(row)
    return rows


def compact_aspect_rows(rows: list[dict[str, Any]], metric: str, descending: bool, limit: int = 4) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: number(row.get(metric)) or 0.0, reverse=descending)[:limit]
    keep = [
        "heldout_aspect",
        "qwen_pair_micro_f1",
        "distilbert_pair_micro_f1",
        "delta_pair_micro_f1",
        "qwen_pair_micro_precision",
        "distilbert_pair_micro_precision",
        "qwen_pair_micro_recall",
        "distilbert_pair_micro_recall",
        "qwen_pair_false_positive_rows_per_100",
        "distilbert_pair_false_positive_rows_per_100",
    ]
    return [{key: row.get(key) for key in keep} for row in selected]


def analyse(
    qwen_validation_dir: Path,
    qwen_test_dir: Path,
    qwen_positive_dir: Path,
    distilbert_loao_dir: Path,
    distilbert_lr2e5_dir: Path | None,
    include_documented_lexical: bool,
) -> dict[str, Any]:
    qwen_validation = summary_results(qwen_validation_dir)
    qwen_test = summary_results(qwen_test_dir)
    positive_results = read_json(qwen_positive_dir / "summary.json")["results"]
    positive_validation = split_rows(positive_results, "validation")
    positive_test = split_rows(positive_results, "test")
    distilbert_validation = distilbert_rows(distilbert_loao_dir, "validation")
    distilbert_test = distilbert_rows(distilbert_loao_dir, "test")

    system_rows = [
        aggregate_row("Qwen3-4B zero-shot LOAO all-row", "validation", qwen_validation),
        aggregate_row("Qwen3-4B zero-shot LOAO all-row", "test", qwen_test),
        aggregate_row("Qwen3-4B zero-shot positive-gold diagnostic", "validation", positive_validation),
        aggregate_row("Qwen3-4B zero-shot positive-gold diagnostic", "test", positive_test),
        aggregate_row(
            "DistilBERT cross-encoder + DistilBERT sentiment LOAO LR3e-5",
            "validation",
            distilbert_validation,
        ),
        aggregate_row("DistilBERT cross-encoder + DistilBERT sentiment LOAO LR3e-5", "test", distilbert_test),
    ]
    if distilbert_lr2e5_dir:
        system_rows.append(
            aggregate_row(
                "DistilBERT cross-encoder + DistilBERT sentiment LOAO LR2e-5",
                "test",
                distilbert_rows(distilbert_lr2e5_dir, "test"),
            )
        )
    if include_documented_lexical:
        system_rows.append(dict(DOCUMENTED_LEXICAL_LOAO_ROW))

    per_aspect_rows = compare_per_aspect(qwen_test, distilbert_test)
    qwen_gap_rows = positive_gap_rows(qwen_test, positive_test)

    qwen_micro = aggregate_metric(qwen_test, "pair_micro_f1")["mean"]
    distilbert_micro = aggregate_metric(distilbert_test, "pair_micro_f1")["mean"]
    qwen_total_seconds = sum(number(row.get("seconds")) or 0.0 for row in qwen_validation + qwen_test)
    qwen_total_examples = sum(int(row.get("examples") or 0) for row in qwen_validation + qwen_test)
    summary = {
        "qwen_test_mean_pair_micro_f1": qwen_micro,
        "distilbert_test_mean_pair_micro_f1": distilbert_micro,
        "qwen_minus_distilbert_pair_micro_mean": qwen_micro - distilbert_micro,
        "qwen_test_mean_pair_samples_f1": aggregate_metric(qwen_test, "pair_samples_f1")["mean"],
        "distilbert_test_mean_pair_samples_f1": aggregate_metric(distilbert_test, "pair_samples_f1")["mean"],
        "qwen_test_mean_precision": aggregate_metric(qwen_test, "pair_micro_precision")["mean"],
        "distilbert_test_mean_precision": aggregate_metric(distilbert_test, "pair_micro_precision")["mean"],
        "qwen_test_mean_recall": aggregate_metric(qwen_test, "pair_micro_recall")["mean"],
        "distilbert_test_mean_recall": aggregate_metric(distilbert_test, "pair_micro_recall")["mean"],
        "qwen_test_mean_fp_rows_per_100": aggregate_metric(qwen_test, "pair_false_positive_rows_per_100")["mean"],
        "distilbert_test_mean_fp_rows_per_100": aggregate_metric(
            distilbert_test,
            "pair_false_positive_rows_per_100",
        )["mean"],
        "qwen_positive_test_mean_pair_micro_f1": aggregate_metric(positive_test, "pair_micro_f1")["mean"],
        "qwen_positive_test_mean_pair_precision": aggregate_metric(positive_test, "pair_micro_precision")["mean"],
        "qwen_positive_test_mean_pair_recall": aggregate_metric(positive_test, "pair_micro_recall")["mean"],
        "qwen_test_valid_json_mean": aggregate_metric(qwen_test, "valid_json_rate")["mean"],
        "qwen_test_schema_valid_mean": aggregate_metric(qwen_test, "schema_valid_rate")["mean"],
        "qwen_aspects_beating_distilbert_pair_micro": sum(
            1 for row in per_aspect_rows if row["winner_pair_micro"] == "Qwen"
        ),
        "distilbert_aspects_beating_qwen_pair_micro": sum(
            1 for row in per_aspect_rows if row["winner_pair_micro"] == "DistilBERT"
        ),
        "largest_qwen_pair_micro_gains": compact_aspect_rows(
            per_aspect_rows,
            "delta_pair_micro_f1",
            descending=True,
        ),
        "largest_qwen_pair_micro_losses": compact_aspect_rows(
            per_aspect_rows,
            "delta_pair_micro_f1",
            descending=False,
        ),
        "qwen_generation_total_seconds_validation_test": qwen_total_seconds,
        "qwen_generation_total_examples_validation_test": qwen_total_examples,
        "qwen_weighted_seconds_per_example_validation_test": (
            qwen_total_seconds / qwen_total_examples if qwen_total_examples else None
        ),
    }
    return {
        "system_rows": system_rows,
        "per_aspect_rows": per_aspect_rows,
        "qwen_gap_rows": qwen_gap_rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qwen zero-shot LOAO with local LOAO baselines.")
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
        "--qwen-positive-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "qwen_loao_positive_diagnostic_20260701",
    )
    parser.add_argument(
        "--distilbert-loao-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "baselines"
        / "loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630",
    )
    parser.add_argument(
        "--distilbert-lr2e5-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "baselines"
        / "loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "qwen_loao_full_interpretation_20260701",
    )
    parser.add_argument("--no-documented-lexical", action="store_true")
    args = parser.parse_args()

    lr2_dir = args.distilbert_lr2e5_dir if args.distilbert_lr2e5_dir.exists() else None
    result = analyse(
        qwen_validation_dir=args.qwen_validation_dir,
        qwen_test_dir=args.qwen_test_dir,
        qwen_positive_dir=args.qwen_positive_dir,
        distilbert_loao_dir=args.distilbert_loao_dir,
        distilbert_lr2e5_dir=lr2_dir,
        include_documented_lexical=not args.no_documented_lexical,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "loao_system_comparison.csv", result["system_rows"])
    write_csv_rows(args.output_dir / "qwen_vs_distilbert_per_aspect_test.csv", result["per_aspect_rows"])
    write_csv_rows(args.output_dir / "qwen_all_row_vs_positive_gold_test.csv", result["qwen_gap_rows"])
    (args.output_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved Qwen LOAO comparison outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
