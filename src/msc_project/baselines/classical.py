from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer


@dataclass(frozen=True)
class ClassicalConfig:
    name: str
    feature_type: str
    max_features: int = 30000
    min_df: int = 2
    ngram_max: int = 2
    c: float = 1.0
    class_weight: str | None = "balanced"
    svd_components: int = 200
    model_type: str = "logreg"


@dataclass
class ClassicalResult:
    config: ClassicalConfig
    labels: list[str]
    y_true: object
    y_score: np.ndarray
    score_type: str
    binarizer: MultiLabelBinarizer
    model: Pipeline


def make_feature_step(config: ClassicalConfig):
    if config.feature_type == "bow":
        return CountVectorizer(
            lowercase=True,
            ngram_range=(1, config.ngram_max),
            min_df=config.min_df,
            max_features=config.max_features,
            binary=False,
        )

    if config.feature_type == "binary_bow":
        return CountVectorizer(
            lowercase=True,
            ngram_range=(1, config.ngram_max),
            min_df=config.min_df,
            max_features=config.max_features,
            binary=True,
        )

    if config.feature_type == "tfidf":
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, config.ngram_max),
            min_df=config.min_df,
            max_features=config.max_features,
            sublinear_tf=True,
        )

    if config.feature_type == "char_tfidf":
        return TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=config.min_df,
            max_features=config.max_features,
            sublinear_tf=True,
        )

    if config.feature_type == "word_char_tfidf":
        each_max = max(1000, config.max_features // 2)
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, config.ngram_max),
                        min_df=config.min_df,
                        max_features=each_max,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=config.min_df,
                        max_features=each_max,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

    if config.feature_type == "lsa":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, config.ngram_max),
                        min_df=config.min_df,
                        max_features=config.max_features,
                        sublinear_tf=True,
                    ),
                ),
                ("svd", TruncatedSVD(n_components=config.svd_components, random_state=13)),
                ("norm", Normalizer(copy=False)),
            ]
        )

    raise ValueError(f"Unknown feature type: {config.feature_type}")


def make_classifier(config: ClassicalConfig) -> OneVsRestClassifier:
    if config.model_type == "linear_svm":
        return OneVsRestClassifier(
            LinearSVC(
                C=config.c,
                class_weight=config.class_weight,
                dual="auto",
                max_iter=5000,
                random_state=13,
            )
        )

    if config.model_type != "logreg":
        raise ValueError(f"Unknown model type: {config.model_type}")

    return OneVsRestClassifier(
        LogisticRegression(
            C=config.c,
            class_weight=config.class_weight,
            max_iter=1000,
            solver="liblinear",
        )
    )


def train_classical_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    labels: list[str],
    config: ClassicalConfig,
    label_column: str = "pair_labels",
) -> ClassicalResult:
    binarizer = MultiLabelBinarizer(classes=labels)
    y_train = binarizer.fit_transform(train_df[label_column])
    y_eval = binarizer.transform(eval_df[label_column])

    model = Pipeline(
        [
            ("features", make_feature_step(config)),
            ("classifier", make_classifier(config)),
        ]
    )
    model.fit(train_df["text"], y_train)
    if config.model_type == "linear_svm":
        y_score = model.decision_function(eval_df["text"])
        score_type = "decision"
    else:
        y_score = model.predict_proba(eval_df["text"])
        score_type = "probability"

    return ClassicalResult(
        config=config,
        labels=labels,
        y_true=y_eval,
        y_score=y_score,
        score_type=score_type,
        binarizer=binarizer,
        model=model,
    )


def threshold_predictions(
    y_score: np.ndarray,
    threshold: float,
    ensure_one: bool = True,
) -> np.ndarray:
    y_pred = (y_score >= threshold).astype(int)
    if ensure_one:
        empty_rows = np.where(y_pred.sum(axis=1) == 0)[0]
        if len(empty_rows) > 0:
            best_labels = np.argmax(y_score[empty_rows], axis=1)
            y_pred[empty_rows, best_labels] = 1
    return y_pred


def predicted_label_lists(
    y_pred: np.ndarray,
    binarizer: MultiLabelBinarizer,
) -> list[list[str]]:
    return [list(row) for row in binarizer.inverse_transform(y_pred)]


def default_configs() -> list[ClassicalConfig]:
    configs: list[ClassicalConfig] = []
    for class_weight in ["balanced", None]:
        suffix = "balanced" if class_weight == "balanced" else "plain"
        for c in [0.5, 1.0, 2.0, 4.0]:
            configs.extend(
                [
                    ClassicalConfig(f"bow_1_2_{suffix}_c{c:g}", "bow", c=c, class_weight=class_weight),
                    ClassicalConfig(f"binary_bow_1_2_{suffix}_c{c:g}", "binary_bow", c=c, class_weight=class_weight),
                    ClassicalConfig(f"tfidf_1_2_{suffix}_c{c:g}", "tfidf", c=c, class_weight=class_weight),
                    ClassicalConfig(
                        f"word_char_tfidf_{suffix}_c{c:g}",
                        "word_char_tfidf",
                        max_features=60000,
                        c=c,
                        class_weight=class_weight,
                    ),
                ]
            )

    configs.extend(
        [
            ClassicalConfig("tfidf_1_1_balanced_c2", "tfidf", ngram_max=1, c=2.0),
            ClassicalConfig("tfidf_1_3_balanced_c2", "tfidf", ngram_max=3, max_features=60000, c=2.0),
            ClassicalConfig("char_tfidf_balanced_c2", "char_tfidf", max_features=60000, c=2.0),
            ClassicalConfig("lsa_200_balanced_c2", "lsa", max_features=60000, svd_components=200, c=2.0),
            ClassicalConfig("lsa_400_balanced_c2", "lsa", max_features=60000, svd_components=400, c=2.0),
        ]
    )

    for class_weight in ["balanced", None]:
        suffix = "balanced" if class_weight == "balanced" else "plain"
        for c in [0.25, 0.5, 1.0, 2.0]:
            configs.extend(
                [
                    ClassicalConfig(
                        f"svm_binary_bow_1_2_{suffix}_c{c:g}",
                        "binary_bow",
                        c=c,
                        class_weight=class_weight,
                        model_type="linear_svm",
                    ),
                    ClassicalConfig(
                        f"svm_tfidf_1_2_{suffix}_c{c:g}",
                        "tfidf",
                        c=c,
                        class_weight=class_weight,
                        model_type="linear_svm",
                    ),
                    ClassicalConfig(
                        f"svm_word_char_tfidf_{suffix}_c{c:g}",
                        "word_char_tfidf",
                        max_features=60000,
                        c=c,
                        class_weight=class_weight,
                        model_type="linear_svm",
                    ),
                ]
            )

    for c in [0.1, 0.2, 0.3, 0.5]:
        configs.extend(
            [
                ClassicalConfig(
                    f"svm_word_char_tfidf_100k_plain_c{c:g}",
                    "word_char_tfidf",
                    max_features=100000,
                    c=c,
                    class_weight=None,
                    model_type="linear_svm",
                ),
                ClassicalConfig(
                    f"svm_word_char_tfidf_1_3_100k_plain_c{c:g}",
                    "word_char_tfidf",
                    max_features=100000,
                    ngram_max=3,
                    c=c,
                    class_weight=None,
                    model_type="linear_svm",
                ),
            ]
        )
    return configs
