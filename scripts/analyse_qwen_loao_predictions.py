from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.evaluation.metrics import evaluate_pair_and_aspect


POSITIVE_DIAGNOSTIC_METRICS = [
    "pair_samples_f1",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
    "sentiment_accuracy_when_gold_aspect_predicted",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def positive_gold_result(result: dict[str, Any]) -> dict[str, Any]:
    aspect = str(result["heldout_aspect"])
    prediction_rows = read_jsonl(Path(result["predictions_file"]))
    positive_rows = [row for row in prediction_rows if row["gold_pair_labels"]]
    metrics = evaluate_pair_and_aspect(
        [row["gold_pair_labels"] for row in positive_rows],
        [row["pred_pair_labels"] for row in positive_rows],
        candidate_pair_labels([aspect]),
    )
    return {
        "split": result["split"],
        "heldout_aspect": aspect,
        "positive_rows": int(len(positive_rows)),
        **metrics,
    }


def aggregate_spread(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    spread = []
    for split, group in frame.groupby("split"):
        row: dict[str, Any] = {
            "split": split,
            "aspects": int(group["heldout_aspect"].nunique()),
        }
        for metric in POSITIVE_DIAGNOSTIC_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        spread.append(row)
    return sorted(spread, key=lambda row: str(row["split"]))


def analyse(summary_paths: list[Path]) -> dict[str, Any]:
    results = []
    source_outputs = []
    for summary_path in summary_paths:
        summary = read_json(summary_path)
        source_outputs.append(str(summary_path.parent))
        for result in summary["results"]:
            if result.get("dry_run"):
                continue
            results.append(positive_gold_result(result))
    return {
        "source_outputs": source_outputs,
        "results": results,
        "spread": aggregate_spread(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Qwen LOAO prediction files without new model calls.")
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "analysis" / "qwen_loao_positive_diagnostic")
    args = parser.parse_args()

    summary = analyse([args.validation_dir / "summary.json", args.test_dir / "summary.json"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["results"]).to_csv(args.output_dir / "positive_gold_results.csv", index=False)
    pd.DataFrame(summary["spread"]).to_csv(args.output_dir / "positive_gold_spread.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["spread"], indent=2), flush=True)
    print(f"Saved Qwen LOAO analysis outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
