"""Create post-test thesis diagnostics from the frozen prediction sets.

The script is descriptive only. It does not alter predictions, thresholds,
models, the registered endpoint or the confirmatory decisions. It limits its
analysis to the three pre-registered Level-2 D confirmatory systems and writes
review-text-free aggregate tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    ROOT
    / "outputs"
    / "experimental"
    / "taxonomy_two_stage_final_test_v1"
    / "single_reveal_analysis"
    / "heldout_prediction_sets.jsonl"
)
DEFAULT_FOLD_RESULTS = (
    ROOT
    / "docs"
    / "thesis_figure_data"
    / "taxonomy_final_test_v1"
    / "official_l2_confirmatory_fold_results.csv"
)
DEFAULT_SUPPORT = (
    ROOT
    / "docs"
    / "thesis_figure_data"
    / "taxonomy_scientific_freeze_v1"
    / "l2_fold_support_and_filtering.csv"
)
DEFAULT_ASPECTS = (
    ROOT / "configs" / "experiments" / "fabsa_aspect_descriptions_minimal_v2.json"
)
DEFAULT_OUTPUT = (
    ROOT / "docs" / "thesis_figure_data" / "taxonomy_final_test_v1"
)

CONFIRMATORY_METHODS = (
    "frozen_qwen_few_shot_stage1__qlora_stage2",
    "frozen_qwen_few_shot",
    "qwen_candidate_pair_qlora",
)

METHOD_NAMES = {
    "frozen_qwen_few_shot_stage1__qlora_stage2": "Fixed stage-wise composition",
    "frozen_qwen_few_shot": "Frozen Qwen few-shot",
    "qwen_candidate_pair_qlora": "QLoRA",
}


def parse_pair(pair: str) -> tuple[str, str]:
    """Return the canonical aspect and sentiment from a rendered pair."""

    aspect, separator, sentiment = pair.rpartition(" | ")
    if not separator or not aspect or not sentiment:
        raise ValueError(f"Malformed aspect--sentiment pair: {pair!r}")
    return aspect, sentiment


def sentiments_for_aspect(pairs: Iterable[str], aspect: str) -> set[str]:
    """Extract the sentiment set attached to one candidate aspect."""

    result: set[str] = set()
    for pair in pairs:
        candidate, sentiment = parse_pair(pair)
        if candidate == aspect:
            result.add(sentiment)
    return result


def row_category(gold: set[str], predicted: set[str]) -> str:
    """Assign one mutually exclusive held-out-aspect outcome category."""

    if not gold and not predicted:
        return "true_absence"
    if gold and not predicted:
        return "missed_aspect"
    if not gold and predicted:
        return "spurious_aspect"
    if gold == predicted:
        return "correct_sentiment_set"
    return "wrong_sentiment_set"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f"No rows prepared for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse_predictions(
    prediction_path: Path, aspect_order: list[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Aggregate error categories and fold-level counts from frozen rows."""

    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    pair_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    seen_rows = 0
    with prediction_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record["level"] != "L2"
                or record["condition"] != "D"
                or record["method_id"] not in CONFIRMATORY_METHODS
            ):
                continue
            fold_id = str(record["fold_id"])
            fold_number = int(fold_id.rsplit("a", 1)[1])
            aspect = aspect_order[fold_number - 1]
            method = str(record["method_id"])
            key = (method, fold_id)
            gold = sentiments_for_aspect(record["gold_pairs"], aspect)
            predicted = sentiments_for_aspect(record["pred_pairs"], aspect)
            category = row_category(gold, predicted)
            counters[key][category] += 1
            counters[key]["rows"] += 1
            counters[key]["gold_presence_rows"] += int(bool(gold))
            counters[key]["predicted_presence_rows"] += int(bool(predicted))
            counters[key]["gold_multi_sentiment_rows"] += int(len(gold) == 2)
            counters[key]["predicted_multi_sentiment_rows"] += int(
                len(predicted) == 2
            )
            counters[key]["multi_sentiment_exact_rows"] += int(
                len(gold) == 2 and gold == predicted
            )
            counters[key]["multi_sentiment_mismatch_rows"] += int(
                len(gold) == 2 and gold != predicted
            )
            pair_counts[key]["pair_tp"] += len(gold & predicted)
            pair_counts[key]["pair_fp"] += len(predicted - gold)
            pair_counts[key]["pair_fn"] += len(gold - predicted)
            seen_rows += 1

    expected_rows = len(CONFIRMATORY_METHODS) * 12 * 1587
    if seen_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} confirmatory L2-D rows, found {seen_rows}"
        )

    fold_rows: list[dict[str, object]] = []
    for method in CONFIRMATORY_METHODS:
        for fold_index, aspect in enumerate(aspect_order, start=1):
            fold_id = f"l2-a{fold_index:02d}"
            key = (method, fold_id)
            count = counters[key]
            pairs = pair_counts[key]
            denominator = 2 * pairs["pair_tp"] + pairs["pair_fp"] + pairs["pair_fn"]
            f1 = 2 * pairs["pair_tp"] / denominator if denominator else 0.0
            fold_rows.append(
                {
                    "method_id": method,
                    "method": METHOD_NAMES[method],
                    "fold_id": fold_id,
                    "heldout_aspect": aspect,
                    "review_rows": count["rows"],
                    "gold_presence_rows": count["gold_presence_rows"],
                    "predicted_presence_rows": count["predicted_presence_rows"],
                    "true_absence_rows": count["true_absence"],
                    "missed_aspect_rows": count["missed_aspect"],
                    "spurious_aspect_rows": count["spurious_aspect"],
                    "correct_sentiment_set_rows": count["correct_sentiment_set"],
                    "wrong_sentiment_set_rows": count["wrong_sentiment_set"],
                    "gold_multi_sentiment_rows": count["gold_multi_sentiment_rows"],
                    "predicted_multi_sentiment_rows": count[
                        "predicted_multi_sentiment_rows"
                    ],
                    "multi_sentiment_exact_rows": count[
                        "multi_sentiment_exact_rows"
                    ],
                    "multi_sentiment_mismatch_rows": count[
                        "multi_sentiment_mismatch_rows"
                    ],
                    "heldout_pair_tp": pairs["pair_tp"],
                    "heldout_pair_fp": pairs["pair_fp"],
                    "heldout_pair_fn": pairs["pair_fn"],
                    "heldout_pair_f1_recomputed": f1,
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for method in CONFIRMATORY_METHODS:
        method_rows = [row for row in fold_rows if row["method_id"] == method]
        sums = {
            key: sum(int(row[key]) for row in method_rows)
            for key in (
                "review_rows",
                "gold_presence_rows",
                "predicted_presence_rows",
                "true_absence_rows",
                "missed_aspect_rows",
                "spurious_aspect_rows",
                "correct_sentiment_set_rows",
                "wrong_sentiment_set_rows",
                "gold_multi_sentiment_rows",
                "predicted_multi_sentiment_rows",
                "multi_sentiment_exact_rows",
                "multi_sentiment_mismatch_rows",
                "heldout_pair_tp",
                "heldout_pair_fp",
                "heldout_pair_fn",
            )
        }
        positive_rows = sums["gold_presence_rows"]
        aggregate_rows.append(
            {
                "method_id": method,
                "method": METHOD_NAMES[method],
                **sums,
                "missed_aspect_rate_among_gold_positive": (
                    sums["missed_aspect_rows"] / positive_rows
                    if positive_rows
                    else 0.0
                ),
                "wrong_sentiment_rate_among_detected_gold_positive": (
                    sums["wrong_sentiment_set_rows"]
                    / (
                        sums["correct_sentiment_set_rows"]
                        + sums["wrong_sentiment_set_rows"]
                    )
                    if sums["correct_sentiment_set_rows"]
                    + sums["wrong_sentiment_set_rows"]
                    else 0.0
                ),
            }
        )
    return aggregate_rows, fold_rows


def build_leave_one_fold_influence(
    fold_result_path: Path, aspect_order: list[str]
) -> list[dict[str, object]]:
    rows = _read_csv(fold_result_path)
    results: list[dict[str, object]] = []
    comparisons = (
        ("H1", "Frozen Qwen few-shot", "fixed_minus_few_shot"),
        ("H2", "QLoRA", "fixed_minus_qlora"),
    )
    for hypothesis, comparator, column in comparisons:
        effects = [float(row[column]) for row in rows]
        full_mean = sum(effects) / len(effects)
        for row, effect in zip(rows, effects, strict=True):
            fold_number = int(row["fold_id"].rsplit("a", 1)[1])
            index = rows.index(row)
            remaining = effects[:index] + effects[index + 1 :]
            leave_one_mean = sum(remaining) / len(remaining)
            results.append(
                {
                    "hypothesis": hypothesis,
                    "comparator": comparator,
                    "omitted_fold_id": row["fold_id"],
                    "omitted_aspect": aspect_order[fold_number - 1],
                    "omitted_fold_effect": effect,
                    "full_12_fold_mean_effect": full_mean,
                    "leave_one_fold_mean_effect": leave_one_mean,
                    "sign_reversal": (full_mean > 0 and leave_one_mean < 0)
                    or (full_mean < 0 and leave_one_mean > 0),
                }
            )
    return results


def build_aspect_context(
    support_path: Path,
    fold_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    support = {row["fold_id"]: row for row in _read_csv(support_path)}
    by_fold = {
        str(row["fold_id"]): row
        for row in fold_rows
        if row["method_id"]
        == "frozen_qwen_few_shot_stage1__qlora_stage2"
    }
    results: list[dict[str, object]] = []
    for fold_id in sorted(by_fold):
        source = support[fold_id]
        official = by_fold[fold_id]
        results.append(
            {
                "fold_id": fold_id,
                "heldout_aspect": source["heldout_aspect"],
                "original_train_reviews": int(source["original_train_reviews"]),
                "filtered_train_reviews": int(source["filtered_train_reviews"]),
                "removed_train_reviews": int(source["removed_train_reviews"]),
                "validation_reviews": int(source["validation_reviews"]),
                "validation_heldout_positive_review_support": int(
                    source["heldout_positive_review_support"]
                ),
                "official_test_reviews": int(official["review_rows"]),
                "official_test_heldout_positive_review_support": int(
                    official["gold_presence_rows"]
                ),
                "official_test_heldout_gold_pair_support": int(
                    official["heldout_pair_tp"] + official["heldout_pair_fn"]
                ),
                "fixed_composition_heldout_pair_f1": official[
                    "heldout_pair_f1_recomputed"
                ],
            }
        )
    return results


def validate_recomputed_folds(
    fold_rows: list[dict[str, object]], fold_result_path: Path
) -> None:
    official = {row["fold_id"]: row for row in _read_csv(fold_result_path)}
    columns = {
        "frozen_qwen_few_shot_stage1__qlora_stage2": "fixed_composition_f1",
        "frozen_qwen_few_shot": "frozen_qwen_few_shot_f1",
        "qwen_candidate_pair_qlora": "qlora_f1",
    }
    for row in fold_rows:
        expected = float(official[str(row["fold_id"])][columns[str(row["method_id"])]])
        actual = float(row["heldout_pair_f1_recomputed"])
        if abs(actual - expected) > 1e-12:
            raise ValueError(
                f"Fold F1 mismatch for {row['method_id']} {row['fold_id']}: "
                f"{actual} != {expected}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--fold-results", type=Path, default=DEFAULT_FOLD_RESULTS)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--aspects", type=Path, default=DEFAULT_ASPECTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    aspect_resource = json.loads(args.aspects.read_text(encoding="utf-8"))
    aspect_order = list(aspect_resource["canonical_order"])
    if len(aspect_order) != 12:
        raise ValueError("The frozen taxonomy must contain exactly 12 aspects")

    aggregate, folds = analyse_predictions(args.predictions, aspect_order)
    validate_recomputed_folds(folds, args.fold_results)
    influence = build_leave_one_fold_influence(args.fold_results, aspect_order)
    context = build_aspect_context(args.support, folds)

    outputs = {
        "official_l2_confirmatory_error_decomposition.csv": aggregate,
        "official_l2_confirmatory_fold_diagnostics.csv": folds,
        "official_l2_confirmatory_leave_one_fold_influence.csv": influence,
        "official_l2_aspect_context.csv": context,
    }
    for filename, rows in outputs.items():
        _write_csv(args.output_dir / filename, rows, args.overwrite)
    print(json.dumps({name: len(rows) for name, rows in outputs.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
