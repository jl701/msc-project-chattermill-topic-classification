from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass
class TfidfLogRegResult:
    labels: list[str]
    y_true: object
    y_pred: object
    pred_labels: list[list[str]]
    x_train: spmatrix
    x_eval: spmatrix
    vectorizer: TfidfVectorizer
    binarizer: MultiLabelBinarizer
    model: OneVsRestClassifier


def run_tfidf_logreg(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    labels: list[str],
    max_features: int = 30000,
) -> TfidfLogRegResult:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
    )
    x_train = vectorizer.fit_transform(train_df["text"])
    x_eval = vectorizer.transform(eval_df["text"])

    binarizer = MultiLabelBinarizer(classes=labels)
    y_train = binarizer.fit_transform(train_df["pair_labels"])
    y_eval = binarizer.transform(eval_df["pair_labels"])

    model = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
        )
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_eval)
    pred_labels = [list(row) for row in binarizer.inverse_transform(y_pred)]

    return TfidfLogRegResult(
        labels=labels,
        y_true=y_eval,
        y_pred=y_pred,
        pred_labels=pred_labels,
        x_train=x_train,
        x_eval=x_eval,
        vectorizer=vectorizer,
        binarizer=binarizer,
        model=model,
    )

