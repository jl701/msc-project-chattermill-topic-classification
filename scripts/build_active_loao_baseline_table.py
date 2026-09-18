from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_METHODS = [
    "bow_count_1_2_train_vocab",
    "tfidf_char_3_5_train_vocab",
    "minilm_l6_v2",
    "e5_base_v2",
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_candidate_pair",
    "qlora_qwen_candidate_pair",
]
SIMILARITY_METHODS = EXPECTED_METHODS[:4]
NUMERIC_COLUMNS = [
    "pair_micro_f1_mean",
    "pair_micro_precision_mean",
    "pair_micro_recall_mean",
    "presence_f1_mean",
    "false_positive_rows_per_100_mean",
    "false_negative_rows_per_100_mean",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_registry(
    registry: dict[str, Any],
    similarity_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    protocol = registry.get("outer_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Missing outer_protocol.")
    if (
        protocol.get("folds") != 12
        or protocol.get("split") != "test"
        or protocol.get("candidate_scope") != "singleton_heldout_aspect"
        or protocol.get("sentiment_requirement") != "candidate_conditioned"
    ):
        raise ValueError("The registry is not the locked 12-fold candidate-conditioned test protocol.")

    rows = registry.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Registry rows must be a list.")
    method_ids = [str(row.get("method_id")) for row in rows]
    if method_ids != EXPECTED_METHODS:
        raise ValueError(f"Active method order changed: {method_ids!r}.")

    guards = registry.get("retired_guards", {})
    forbidden_ids = {str(value) for value in guards.get("forbidden_method_ids", [])}
    forbidden_terms = [
        str(value).casefold() for value in guards.get("forbidden_sentiment_terms", [])
    ]
    forbidden_values = [
        float(value) for value in guards.get("forbidden_pair_micro_f1_values", [])
    ]
    for row in rows:
        method_id = str(row.get("method_id"))
        sentiment = str(row.get("sentiment_formulation"))
        if method_id in forbidden_ids:
            raise ValueError(f"Retired method entered active registry: {method_id}.")
        if any(term in sentiment.casefold() for term in forbidden_terms):
            raise ValueError(f"Retired sentiment formulation entered {method_id}: {sentiment}.")
        for column in NUMERIC_COLUMNS:
            value = float(row.get(column))
            if not math.isfinite(value):
                raise ValueError(f"{method_id}.{column} is not finite.")
        primary = float(row["pair_micro_f1_mean"])
        if any(math.isclose(primary, value, abs_tol=5e-7) for value in forbidden_values):
            raise ValueError(f"Retired result value entered active registry: {primary}.")

    similarity_by_method = {
        str(row["method"]): row for row in similarity_rows
    }
    if list(similarity_by_method) != SIMILARITY_METHODS:
        raise ValueError("The active similarity export does not contain exactly four locked methods.")
    registry_by_method = {str(row["method_id"]): row for row in rows}
    for method_id in SIMILARITY_METHODS:
        source = similarity_by_method[method_id]
        active = registry_by_method[method_id]
        if source["sentiment_component"] != "distilbert_aspect_conditioned":
            raise ValueError(f"{method_id} no longer uses the frozen aspect-conditioned head.")
        if int(source["folds"]) != 12:
            raise ValueError(f"{method_id} no longer contains twelve folds.")
        for column in (
            "pair_micro_f1_mean",
            "presence_f1_mean",
        ):
            if not math.isclose(
                float(source[column]), float(active[column]), rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"{method_id}.{column} disagrees with the frozen source export.")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "method_id",
        "display_name",
        "comparison_block",
        "outer_protocol",
        "split",
        "folds",
        "sentiment_formulation",
        "threshold_selection",
        *NUMERIC_COLUMNS,
        "thesis_role",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "outer_protocol": "target_calibrated_all_row_loao",
                    "split": "test",
                    "folds": 12,
                }
            )


def markdown(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            "| {display_name} | {sentiment_formulation} | {pair_micro_f1_mean:.6f} | "
            "{pair_micro_precision_mean:.6f} | {pair_micro_recall_mean:.6f} | "
            "{presence_f1_mean:.6f} | {false_positive_rows_per_100_mean:.3f} | "
            "{false_negative_rows_per_100_mean:.3f} | {thesis_role} |".format(**row)
        )
    return """# Authoritative LOAO Baseline Table

Last generated: 23 July 2026

This is the sole active baseline table for the dissertation. It is generated from
`configs/experiments/loao_active_baseline_registry_v1.json` and is limited to the
target-calibrated, twelve-fold, all-row LOAO test benchmark. Every retained method
predicts sentiment conditional on the supplied candidate aspect.

The primary metric is the unweighted mean of the twelve fold-level pair micro-F1
values. Precision, recall, presence F1, and false-positive/false-negative rows per
100 are fold means and are diagnostic rather than pooled-corpus scores.

| Method | Sentiment formulation | Pair F1 | Precision | Recall | Presence F1 | FP rows/100 | FN rows/100 | Thesis role |
|---|---|---:|---:|---:|---:|---:|---:|---|
{rows}

## Permitted comparisons

- Count, strict TF-IDF, MiniLM, and E5 form the strict representation block: they
  share the outer data, candidate scope, frozen DistilBERT aspect-conditioned
  sentiment head, threshold-selection rule, and evaluation.
- Frozen candidate-pair Qwen and candidate-pair QLoRA form the matched adaptation
  block.
- DistilBERT and cross-family values share the outer benchmark, but their internal
  model training and calibration are not identical.

## Exclusions

Global-document sentiment, shallow sentiment, legacy TF-IDF, historical JSON-prompt
Qwen, and the legacy TF-IDF-to-Qwen router are not active rows. A rebuilt router is
a deployment analysis, not a baseline, and must be reported separately.

The machine-readable companion is
`docs/thesis_figure_data/loao_active_baselines_v1.csv`.
""".format(rows="\n".join(table_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the locked active LOAO baseline table.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/experiments/loao_active_baseline_registry_v1.json"),
    )
    parser.add_argument(
        "--similarity-source",
        type=Path,
        default=Path(
            "docs/thesis_figure_data/"
            "loao_similarity_aspect_conditioned_sentiment_v1_active.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("docs/thesis_figure_data/loao_active_baselines_v1.csv"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("docs/thesis_result_tables.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = validate_registry(read_json(args.registry), read_csv(args.similarity_source))
    write_csv(args.output_csv, rows)
    args.output_markdown.write_text(markdown(rows), encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
