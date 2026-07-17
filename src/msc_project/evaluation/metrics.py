from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer


PAIR_SEPARATOR = " | "


def pair_to_aspect(label: str) -> str:
    return label.split(PAIR_SEPARATOR, maxsplit=1)[0]


def pair_to_components(label: str) -> tuple[str, str]:
    aspect, sentiment = label.split(PAIR_SEPARATOR, maxsplit=1)
    return aspect, sentiment


def pairs_to_aspects(rows: Iterable[Iterable[str]]) -> list[list[str]]:
    return [sorted({pair_to_aspect(label) for label in labels}) for labels in rows]


def binarize_labels(rows: list[list[str]], classes: list[str]) -> tuple[MultiLabelBinarizer, object]:
    binarizer = MultiLabelBinarizer(classes=classes)
    binarizer.fit([classes])
    return binarizer, binarizer.transform(rows)


def multilabel_scores(y_true: object, y_pred: object) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    true_counts = y_true_array.sum(axis=1)
    pred_counts = y_pred_array.sum(axis=1)
    true_positive_counts = (y_true_array & y_pred_array).sum(axis=1)
    false_positive_counts = ((1 - y_true_array) & y_pred_array).sum(axis=1)
    false_negative_counts = (y_true_array & (1 - y_pred_array)).sum(axis=1)
    label_tp = int(true_positive_counts.sum())
    label_fp = int(false_positive_counts.sum())
    label_fn = int(false_negative_counts.sum())
    class_tp = (y_true_array & y_pred_array).sum(axis=0)
    class_fp = ((1 - y_true_array) & y_pred_array).sum(axis=0)
    class_fn = (y_true_array & (1 - y_pred_array)).sum(axis=0)
    rows = int(y_true_array.shape[0])
    denominators = true_counts + pred_counts
    samples_f1 = np.divide(
        2 * true_positive_counts,
        denominators,
        out=np.zeros_like(denominators, dtype=float),
        where=denominators != 0,
    )
    precision = label_tp / (label_tp + label_fp) if label_tp + label_fp else 0.0
    recall = label_tp / (label_tp + label_fn) if label_tp + label_fn else 0.0
    micro_denominator = 2 * label_tp + label_fp + label_fn
    micro_f1 = 2 * label_tp / micro_denominator if micro_denominator else 0.0
    class_denominators = 2 * class_tp + class_fp + class_fn
    class_f1 = np.divide(
        2 * class_tp,
        class_denominators,
        out=np.zeros_like(class_denominators, dtype=float),
        where=class_denominators != 0,
    )
    macro_f1 = float(class_f1.mean()) if class_f1.size else 0.0
    empty_gold = true_counts == 0
    empty_pred = pred_counts == 0
    return {
        "micro_f1": float(micro_f1),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "macro_f1": macro_f1,
        "samples_f1": float(samples_f1.mean()),
        "label_tp": label_tp,
        "label_fp": label_fp,
        "label_fn": label_fn,
        "predicted_label_count": int(pred_counts.sum()),
        "gold_label_count": int(true_counts.sum()),
        "empty_gold_rows": int(empty_gold.sum()),
        "empty_prediction_rows": int(empty_pred.sum()),
        "empty_gold_and_prediction_rows": int((empty_gold & empty_pred).sum()),
        "false_positive_rows": int((empty_gold & ~empty_pred).sum()),
        "false_negative_rows": int((~empty_gold & empty_pred).sum()),
        "false_positive_rows_per_100": float((empty_gold & ~empty_pred).sum() * 100 / rows) if rows else 0.0,
        "false_positive_labels_per_100": float(label_fp * 100 / rows) if rows else 0.0,
        "false_negative_rows_per_100": float((~empty_gold & empty_pred).sum() * 100 / rows) if rows else 0.0,
        "exact_match_rate": float((y_true_array == y_pred_array).all(axis=1).mean()) if rows else 0.0,
    }


def binary_presence_scores(
    true_pairs: list[list[str]],
    pred_pairs: list[list[str]],
) -> dict[str, float]:
    """Score row-level candidate presence as a positive-class binary task.

    A non-empty pair set means that the supplied candidate is present. True
    negatives are reported but deliberately excluded from precision, recall,
    and F1 so a singleton-candidate LOAO fold cannot turn micro F1 into
    absence-dominated accuracy.
    """

    if len(true_pairs) != len(pred_pairs):
        raise ValueError("true and predicted pair rows must have the same length.")

    true_present = np.asarray([bool(row) for row in true_pairs], dtype=bool)
    pred_present = np.asarray([bool(row) for row in pred_pairs], dtype=bool)
    tp = int(np.sum(true_present & pred_present))
    fp = int(np.sum(~true_present & pred_present))
    fn = int(np.sum(true_present & ~pred_present))
    tn = int(np.sum(~true_present & ~pred_present))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    rows = len(true_pairs)
    return {
        "presence_precision": float(precision),
        "presence_recall": float(recall),
        "presence_f1": float(f1),
        "presence_tp_rows": tp,
        "presence_fp_rows": fp,
        "presence_fn_rows": fn,
        "presence_tn_rows": tn,
        "presence_prevalence": float(true_present.mean()) if rows else 0.0,
        "presence_false_positive_rows_per_100": float(fp * 100 / rows) if rows else 0.0,
        "presence_false_negative_rows_per_100": float(fn * 100 / rows) if rows else 0.0,
    }


def evaluate_label_sets(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
    classes: list[str],
) -> dict[str, float]:
    _, y_true = binarize_labels(true_labels, classes)
    _, y_pred = binarize_labels(pred_labels, classes)
    return multilabel_scores(y_true, y_pred)


def evaluate_pair_and_aspect(
    true_pairs: list[list[str]],
    pred_pairs: list[list[str]],
    pair_classes: list[str],
) -> dict[str, float]:
    pair_scores = evaluate_label_sets(true_pairs, pred_pairs, pair_classes)

    true_aspects = pairs_to_aspects(true_pairs)
    pred_aspects = pairs_to_aspects(pred_pairs)
    aspect_classes = sorted({pair_to_aspect(label) for label in pair_classes})
    aspect_scores = evaluate_label_sets(true_aspects, pred_aspects, aspect_classes)

    sentiment_total = 0
    sentiment_correct = 0
    for true_row, pred_row in zip(true_pairs, pred_pairs):
        predicted_sentiments_by_aspect: dict[str, set[str]] = {}
        for label in pred_row:
            aspect, sentiment = pair_to_components(label)
            predicted_sentiments_by_aspect.setdefault(aspect, set()).add(sentiment)

        for label in true_row:
            aspect, sentiment = pair_to_components(label)
            if aspect in predicted_sentiments_by_aspect:
                sentiment_total += 1
                if sentiment in predicted_sentiments_by_aspect[aspect]:
                    sentiment_correct += 1

    sentiment_accuracy = sentiment_correct / sentiment_total if sentiment_total else 0.0

    output: dict[str, float] = {}
    for prefix, scores in [("pair", pair_scores), ("aspect", aspect_scores)]:
        for key, value in scores.items():
            output[f"{prefix}_{key}"] = value
    output["sentiment_accuracy_when_gold_aspect_predicted"] = float(sentiment_accuracy)
    output["sentiment_correct_when_gold_aspect_predicted"] = int(sentiment_correct)
    output["sentiment_evaluated_gold_aspects"] = int(sentiment_total)
    output.update(binary_presence_scores(true_pairs, pred_pairs))
    return output


def per_label_report(
    y_true: object,
    y_pred: object,
    classes: list[str],
    output_path: Path | None = None,
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).transpose()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path)
    return frame
