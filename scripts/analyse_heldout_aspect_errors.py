from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.evaluation.error_analysis import count_labels, row_error_record, summarise_records


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def serialise_lists(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in [
        "gold_pair_labels",
        "pred_pair_labels",
        "pair_tp",
        "pair_fp",
        "pair_fn",
        "aspect_tp",
        "aspect_fp",
        "aspect_fn",
        "sentiment_error_aspects",
        "error_categories",
    ]:
        frame[column] = frame[column].apply(lambda value: json.dumps(value, ensure_ascii=False))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse held-out-aspect prediction errors.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-errors", type=int, default=50)
    args = parser.parse_args()

    prediction_rows = read_jsonl(args.predictions_jsonl)
    records = [row_error_record(row) for row in prediction_rows]
    summary = summarise_records(records)
    per_pair = count_labels(records, "pair")
    per_aspect = count_labels(records, "aspect")
    per_row = pd.DataFrame(records)
    error_rows = per_row[per_row["error_categories"].apply(lambda categories: categories != ["exact"])].copy()
    error_rows = error_rows.sort_values(["pair_sample_f1", "aspect_sample_f1"]).head(args.top_errors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_pair.to_csv(args.output_dir / "per_pair.csv", index=False)
    per_aspect.to_csv(args.output_dir / "per_aspect.csv", index=False)
    serialise_lists(per_row).to_csv(args.output_dir / "per_row.csv", index=False)
    serialise_lists(error_rows).to_csv(args.output_dir / "top_error_examples.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved held-out-aspect error analysis to {args.output_dir}")


if __name__ == "__main__":
    main()

