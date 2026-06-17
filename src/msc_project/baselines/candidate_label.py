from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from msc_project.baselines.classical import ClassicalConfig, make_feature_step
from msc_project.data.fabsa import format_pair_label


SENTIMENTS = ("negative", "neutral", "positive")


def aspect_query_text(aspect: str) -> str:
    leaf = aspect.split(":")[-1].strip()
    return f"{aspect.replace(':', ' ')} {leaf}"


@dataclass
class CandidateLexicalBaseline:
    candidate_aspects: list[str]
    vectorizer: TfidfVectorizer | None = None
    label_matrix: object | None = None
    sentiment_model: Pipeline | None = None

    def fit(self, train_df: pd.DataFrame) -> "CandidateLexicalBaseline":
        label_texts = [aspect_query_text(aspect) for aspect in self.candidate_aspects]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        self.vectorizer.fit(train_df["text"].tolist() + label_texts)
        self.label_matrix = normalize(self.vectorizer.transform(label_texts))
        self.sentiment_model = train_sentiment_model(train_df)
        return self

    def aspect_scores(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None or self.label_matrix is None:
            raise RuntimeError("CandidateLexicalBaseline must be fitted before prediction.")

        text_matrix = normalize(self.vectorizer.transform(texts))
        return np.asarray((text_matrix @ self.label_matrix.T).toarray())

    def predict(self, eval_df: pd.DataFrame, threshold: float, ensure_one: bool = True) -> list[list[str]]:
        if self.sentiment_model is None:
            raise RuntimeError("CandidateLexicalBaseline must be fitted before prediction.")

        scores = self.aspect_scores(eval_df["text"].tolist())
        sentiments = self.sentiment_model.predict(eval_df["text"].tolist())
        predictions: list[list[str]] = []

        for row_scores, sentiment in zip(scores, sentiments):
            selected = np.flatnonzero(row_scores >= threshold)
            if ensure_one and len(selected) == 0:
                selected = np.array([int(np.argmax(row_scores))])
            predictions.append(
                [
                    format_pair_label(self.candidate_aspects[index], str(sentiment))
                    for index in selected
                ]
            )
        return predictions


def train_sentiment_model(train_df: pd.DataFrame) -> Pipeline:
    texts: list[str] = []
    sentiments: list[str] = []
    for text, labels in zip(train_df["text"], train_df["supervision_labels"]):
        for _, sentiment in labels:
            texts.append(text)
            sentiments.append(sentiment)

    model = Pipeline(
        [
            (
                "features",
                make_feature_step(
                    ClassicalConfig(
                        name="sentiment_word_char_tfidf",
                        feature_type="word_char_tfidf",
                        max_features=60000,
                    )
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(texts, sentiments)
    return model


def candidate_pair_labels(candidate_aspects: list[str]) -> list[str]:
    return [
        format_pair_label(aspect, sentiment)
        for aspect in candidate_aspects
        for sentiment in SENTIMENTS
    ]
