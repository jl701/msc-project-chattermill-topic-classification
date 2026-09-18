from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from msc_project.evaluation.metrics import PAIR_SEPARATOR, pair_to_aspect


@dataclass(frozen=True)
class LabelCounts:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


def split_pair_label(label: str) -> tuple[str, str]:
    aspect, sentiment = label.rsplit(PAIR_SEPARATOR, maxsplit=1)
    return aspect, sentiment


def sample_f1(gold: set[str], predicted: set[str]) -> float:
    denominator = len(gold) + len(predicted)
    if denominator == 0:
        return 0.0
    return 2 * len(gold & predicted) / denominator


def aspect_sentiments(labels: Iterable[str]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for label in labels:
        aspect, sentiment = split_pair_label(label)
        values.setdefault(aspect, set()).add(sentiment)
    return values


def row_error_record(row: dict[str, object]) -> dict[str, object]:
    gold_pairs = set(row["gold_pair_labels"])
    pred_pairs = set(row["pred_pair_labels"])
    gold_aspects = {pair_to_aspect(label) for label in gold_pairs}
    pred_aspects = {pair_to_aspect(label) for label in pred_pairs}
    gold_by_aspect = aspect_sentiments(gold_pairs)
    pred_by_aspect = aspect_sentiments(pred_pairs)

    pair_tp = sorted(gold_pairs & pred_pairs)
    pair_fp = sorted(pred_pairs - gold_pairs)
    pair_fn = sorted(gold_pairs - pred_pairs)
    aspect_tp = sorted(gold_aspects & pred_aspects)
    aspect_fp = sorted(pred_aspects - gold_aspects)
    aspect_fn = sorted(gold_aspects - pred_aspects)
    sentiment_errors = []

    for aspect in sorted(gold_aspects & pred_aspects):
        gold_sentiments = gold_by_aspect.get(aspect, set())
        pred_sentiments = pred_by_aspect.get(aspect, set())
        if gold_sentiments and pred_sentiments and not (gold_sentiments & pred_sentiments):
            sentiment_errors.append(aspect)

    categories = []
    if not pair_fp and not pair_fn:
        categories.append("exact")
    if not pred_pairs:
        categories.append("missed_all")
    if aspect_fn:
        categories.append("aspect_miss")
    if aspect_fp:
        categories.append("aspect_overpredict")
    if sentiment_errors:
        categories.append("sentiment_error")
    if pair_fp and pair_fn and not (aspect_fp or aspect_fn or sentiment_errors):
        categories.append("pair_mismatch")

    return {
        "id": str(row.get("id", "")),
        "row_uid": str(row.get("row_uid", "")),
        "original_split": str(row.get("original_split", "")),
        "org_index": row.get("org_index", ""),
        "text": row.get("text", ""),
        "gold_pair_labels": sorted(gold_pairs),
        "pred_pair_labels": sorted(pred_pairs),
        "pair_tp": pair_tp,
        "pair_fp": pair_fp,
        "pair_fn": pair_fn,
        "aspect_tp": aspect_tp,
        "aspect_fp": aspect_fp,
        "aspect_fn": aspect_fn,
        "sentiment_error_aspects": sentiment_errors,
        "pair_sample_f1": sample_f1(gold_pairs, pred_pairs),
        "aspect_sample_f1": sample_f1(gold_aspects, pred_aspects),
        "error_categories": categories,
    }


def count_labels(records: list[dict[str, object]], label_type: str) -> pd.DataFrame:
    if label_type not in {"pair", "aspect"}:
        raise ValueError(f"Unknown label type: {label_type}")

    tp_counter: Counter[str] = Counter()
    fp_counter: Counter[str] = Counter()
    fn_counter: Counter[str] = Counter()
    for record in records:
        tp_counter.update(record[f"{label_type}_tp"])
        fp_counter.update(record[f"{label_type}_fp"])
        fn_counter.update(record[f"{label_type}_fn"])

    labels = sorted(set(tp_counter) | set(fp_counter) | set(fn_counter))
    rows = []
    for label in labels:
        counts = LabelCounts(tp_counter[label], fp_counter[label], fn_counter[label])
        rows.append(
            {
                "label": label,
                "tp": counts.true_positive,
                "fp": counts.false_positive,
                "fn": counts.false_negative,
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "support": counts.true_positive + counts.false_negative,
            }
        )
    return pd.DataFrame(rows).sort_values(["support", "label"], ascending=[False, True])


def summarise_records(records: list[dict[str, object]]) -> dict[str, object]:
    category_counts: Counter[str] = Counter()
    for record in records:
        category_counts.update(record["error_categories"])

    return {
        "rows": len(records),
        "mean_pair_sample_f1": sum(record["pair_sample_f1"] for record in records) / max(1, len(records)),
        "mean_aspect_sample_f1": sum(record["aspect_sample_f1"] for record in records) / max(1, len(records)),
        "exact_rows": int(category_counts.get("exact", 0)),
        "missed_all_rows": int(category_counts.get("missed_all", 0)),
        "aspect_miss_rows": int(category_counts.get("aspect_miss", 0)),
        "aspect_overpredict_rows": int(category_counts.get("aspect_overpredict", 0)),
        "sentiment_error_rows": int(category_counts.get("sentiment_error", 0)),
        "category_counts": dict(sorted(category_counts.items())),
    }

