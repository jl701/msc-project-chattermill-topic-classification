from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


PAIR_SEPARATOR = " | "


def pair_to_aspect(label: str) -> str:
    return label.split(PAIR_SEPARATOR, maxsplit=1)[0]


def pairs_to_aspects(rows: Iterable[Iterable[str]]) -> list[list[str]]:
    return [sorted({pair_to_aspect(label) for label in labels}) for labels in rows]


def binarize_labels(rows: list[list[str]], classes: list[str]) -> tuple[MultiLabelBinarizer, object]:
    binarizer = MultiLabelBinarizer(classes=classes)
    binarizer.fit([classes])
    return binarizer, binarizer.transform(rows)


def multilabel_scores(y_true: object, y_pred: object) -> dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
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

    return {
        "pair_micro_f1": pair_scores["micro_f1"],
        "pair_macro_f1": pair_scores["macro_f1"],
        "pair_samples_f1": pair_scores["samples_f1"],
        "aspect_micro_f1": aspect_scores["micro_f1"],
        "aspect_macro_f1": aspect_scores["macro_f1"],
        "aspect_samples_f1": aspect_scores["samples_f1"],
    }


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

