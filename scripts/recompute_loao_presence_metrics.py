from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.evaluation.metrics import evaluate_pair_and_aspect


AGGREGATE_METRICS = [
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "presence_prevalence",
    "presence_false_positive_rows_per_100",
    "presence_false_negative_rows_per_100",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_samples_f1",
    "aspect_micro_f1",
    "legacy_aspect_micro_f1",
]

FROZEN_PAIR_METRICS = [
    "pair_samples_f1",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
]


def aspect_slug(aspect: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", aspect).strip("_").lower()
    return slug or "aspect"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "gold_pair_labels" not in payload or "pred_pair_labels" not in payload:
                raise ValueError(f"Missing gold/pred labels in {path} line {line_number}.")
            rows.append(payload)
    return rows


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def corrected_row(
    system: str,
    split: str,
    aspect: str,
    prediction_path: Path,
    source_metrics: dict[str, Any],
) -> dict[str, Any]:
    prediction_rows = read_jsonl(prediction_path)
    row_uids = [str(row.get("row_uid", "")) for row in prediction_rows]
    populated_uids = [value for value in row_uids if value]
    if len(populated_uids) != len(set(populated_uids)):
        raise ValueError(f"Duplicate row_uid values in {prediction_path}.")

    gold = [list(row["gold_pair_labels"]) for row in prediction_rows]
    predicted = [list(row["pred_pair_labels"]) for row in prediction_rows]
    metrics = evaluate_pair_and_aspect(gold, predicted, candidate_pair_labels([aspect]))
    stability_fields: dict[str, float] = {}
    for metric in FROZEN_PAIR_METRICS:
        source_value = float(source_metrics[metric])
        delta = float(metrics[metric] - source_value)
        if abs(delta) > 1e-12:
            raise ValueError(
                f"{metric} changed for {system} {split} {aspect}: "
                f"source={source_value}, recomputed={metrics[metric]}"
            )
        stability_fields[f"source_{metric}"] = source_value
        stability_fields[f"{metric}_delta"] = delta
    if abs(float(metrics["presence_f1"]) - float(metrics["aspect_micro_f1"])) > 1e-12:
        raise ValueError(f"Singleton presence/aspect F1 mismatch for {system} {split} {aspect}.")

    return {
        "system": system,
        "split": split,
        "heldout_aspect": aspect,
        "examples": int(len(prediction_rows)),
        "prediction_file": str(prediction_path.relative_to(PROJECT_ROOT)),
        **stability_fields,
        "legacy_aspect_micro_f1": float(source_metrics["aspect_micro_f1"]),
        **metrics,
    }


def local_rows(run_dir: Path, system: str) -> list[dict[str, Any]]:
    source = pd.read_csv(run_dir / "all_results.csv")
    rows: list[dict[str, Any]] = []
    for record in source.to_dict(orient="records"):
        split = str(record["split"])
        aspect = str(record["heldout_aspect"])
        prediction_path = (
            run_dir
            / str(record["baseline"])
            / str(record["strategy"])
            / aspect_slug(aspect)
            / f"best_{split}_predictions.jsonl"
        )
        rows.append(corrected_row(system, split, aspect, prediction_path, record))
    return rows


def qwen_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = read_json(run_dir / "summary.json")
        for record in summary["results"]:
            if record.get("dry_run"):
                continue
            rows.append(
                corrected_row(
                    "Qwen",
                    str(record["split"]),
                    str(record["heldout_aspect"]),
                    resolve_project_path(record["predictions_file"]),
                    record,
                )
            )
    return rows


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    aggregate_rows: list[dict[str, Any]] = []
    for (system, split), group in rows.groupby(["system", "split"], sort=True):
        if group["heldout_aspect"].nunique() != 12:
            raise ValueError(f"Expected 12 aspects for {system} {split}, found {group['heldout_aspect'].nunique()}.")
        output: dict[str, Any] = {
            "system": system,
            "split": split,
            "aspects": int(group["heldout_aspect"].nunique()),
            "examples_per_fold": int(group["examples"].iloc[0]),
        }
        for metric in FROZEN_PAIR_METRICS:
            output[f"max_abs_{metric}_delta"] = float(group[f"{metric}_delta"].abs().max())
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_std"] = float(values.std(ddof=0))
            output[f"{metric}_min"] = float(values.min())
            output[f"{metric}_max"] = float(values.max())
        aggregate_rows.append(output)
    return pd.DataFrame(aggregate_rows).sort_values(["system", "split"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute corrected LOAO presence metrics from frozen predictions.")
    parser.add_argument(
        "--tfidf-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baselines" / "loao_lexical_global_score_export_20260713",
    )
    parser.add_argument(
        "--distilbert-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "baselines"
        / "loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702",
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
        default=PROJECT_ROOT / "outputs" / "analysis" / "loao_presence_metric_correction_20260717",
    )
    args = parser.parse_args()

    rows = [
        *local_rows(args.tfidf_dir, "TF-IDF"),
        *local_rows(args.distilbert_dir, "DistilBERT"),
        *qwen_rows([args.qwen_validation_dir, args.qwen_test_dir]),
    ]
    per_aspect = pd.DataFrame(rows).sort_values(["system", "split", "heldout_aspect"]).reset_index(drop=True)
    aggregates = aggregate(per_aspect)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_aspect.to_csv(args.output_dir / "per_aspect_corrected.csv", index=False)
    aggregates.to_csv(args.output_dir / "aggregate_corrected.csv", index=False)
    summary = {
        "task": "Correct singleton-candidate LOAO presence F1 from frozen prediction files",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_definition": (
            "Row-level positive-class presence precision, recall, and F1 use TP, FP, and FN only; "
            "true negatives do not enter F1."
        ),
        "model_calls_or_training": False,
        "sources": {
            "tfidf": str(args.tfidf_dir.relative_to(PROJECT_ROOT)),
            "distilbert": str(args.distilbert_dir.relative_to(PROJECT_ROOT)),
            "qwen_validation": str(args.qwen_validation_dir.relative_to(PROJECT_ROOT)),
            "qwen_test": str(args.qwen_test_dir.relative_to(PROJECT_ROOT)),
        },
        "outputs_contain_raw_review_text": False,
        "aggregate": aggregates.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    print(f"Saved corrected LOAO metrics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
