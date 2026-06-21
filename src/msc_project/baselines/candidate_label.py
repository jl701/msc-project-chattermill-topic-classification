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
SENTIMENT_MODES = ("global", "aspect_conditioned")


def aspect_query_text(aspect: str) -> str:
    leaf = aspect.split(":")[-1].strip()
    return f"{aspect.replace(':', ' ')} {leaf}"


def aspect_conditioned_sentiment_text(text: str, aspect: str) -> str:
    leaf = aspect.split(":")[-1].strip()
    return f"Review: {text}\nAspect: {aspect}\nAspect keywords: {leaf}"


@dataclass
class AspectConditionedSentimentModel:
    estimator: Pipeline | None = None
    constant_sentiment: str | None = None

    def predict_pairs(self, texts: list[str], aspects: list[str]) -> list[str]:
        if len(texts) != len(aspects):
            raise ValueError("texts and aspects must have the same length.")
        if self.constant_sentiment is not None:
            return [self.constant_sentiment for _ in texts]
        if self.estimator is None:
            raise RuntimeError("Aspect-conditioned sentiment model has not been fitted.")

        inputs = [
            aspect_conditioned_sentiment_text(text, aspect)
            for text, aspect in zip(texts, aspects)
        ]
        return [str(value) for value in self.estimator.predict(inputs).tolist()]


@dataclass
class CandidateLexicalBaseline:
    candidate_aspects: list[str]
    sentiment_mode: str = "aspect_conditioned"
    vectorizer: TfidfVectorizer | None = None
    label_matrix: object | None = None
    sentiment_model: object | None = None

    def fit(self, train_df: pd.DataFrame) -> "CandidateLexicalBaseline":
        validate_sentiment_mode(self.sentiment_mode)
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
        self.sentiment_model = train_candidate_sentiment_model(train_df, self.sentiment_mode)
        return self

    def aspect_scores(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None or self.label_matrix is None:
            raise RuntimeError("CandidateLexicalBaseline must be fitted before prediction.")

        text_matrix = normalize(self.vectorizer.transform(texts))
        return np.asarray((text_matrix @ self.label_matrix.T).toarray())

    def predict(self, eval_df: pd.DataFrame, threshold: float, ensure_one: bool = True) -> list[list[str]]:
        if self.sentiment_model is None:
            raise RuntimeError("CandidateLexicalBaseline must be fitted before prediction.")

        texts = eval_df["text"].tolist()
        scores = self.aspect_scores(texts)
        predictions: list[list[str]] = []
        aspect_predictions: list[list[str]] = []

        for row_scores in scores:
            selected = np.flatnonzero(row_scores >= threshold)
            if ensure_one and len(selected) == 0:
                selected = np.array([int(np.argmax(row_scores))])
            aspect_predictions.append([self.candidate_aspects[index] for index in selected])

        sentiment_lookup = build_sentiment_lookup(
            self.sentiment_model,
            texts,
            self.candidate_aspects,
            self.sentiment_mode,
        )
        return pair_predictions_from_aspects(aspect_predictions, sentiment_lookup)


def validate_sentiment_mode(sentiment_mode: str) -> None:
    if sentiment_mode not in SENTIMENT_MODES:
        raise ValueError(f"Unknown sentiment mode: {sentiment_mode}")


def train_candidate_sentiment_model(train_df: pd.DataFrame, sentiment_mode: str) -> object:
    validate_sentiment_mode(sentiment_mode)
    if sentiment_mode == "global":
        return train_sentiment_model(train_df)
    return train_aspect_conditioned_sentiment_model(train_df)


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


def train_aspect_conditioned_sentiment_model(train_df: pd.DataFrame) -> AspectConditionedSentimentModel:
    texts: list[str] = []
    sentiments: list[str] = []
    for text, labels in zip(train_df["text"], train_df["supervision_labels"]):
        for aspect, sentiment in labels:
            texts.append(aspect_conditioned_sentiment_text(str(text), str(aspect)))
            sentiments.append(str(sentiment))

    if not sentiments:
        raise ValueError("Cannot train sentiment model without supervision labels.")

    unique_sentiments = sorted(set(sentiments))
    if len(unique_sentiments) == 1:
        return AspectConditionedSentimentModel(constant_sentiment=unique_sentiments[0])

    model = Pipeline(
        [
            (
                "features",
                make_feature_step(
                    ClassicalConfig(
                        name="aspect_conditioned_sentiment_word_char_tfidf",
                        feature_type="word_char_tfidf",
                        max_features=80000,
                        min_df=1,
                        ngram_max=2,
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
    return AspectConditionedSentimentModel(estimator=model)


def build_sentiment_lookup(
    sentiment_model: object,
    texts: list[str],
    candidate_aspects: list[str],
    sentiment_mode: str,
) -> list[dict[str, str]]:
    validate_sentiment_mode(sentiment_mode)
    if sentiment_mode == "global":
        row_sentiments = [str(value) for value in sentiment_model.predict(texts).tolist()]
        return [
            {aspect: sentiment for aspect in candidate_aspects}
            for sentiment in row_sentiments
        ]

    pair_texts: list[str] = []
    pair_aspects: list[str] = []
    pair_positions: list[tuple[int, str]] = []
    for row_index, text in enumerate(texts):
        for aspect in candidate_aspects:
            pair_texts.append(str(text))
            pair_aspects.append(aspect)
            pair_positions.append((row_index, aspect))

    predicted = sentiment_model.predict_pairs(pair_texts, pair_aspects)
    lookup = [dict() for _ in texts]
    for (row_index, aspect), sentiment in zip(pair_positions, predicted):
        lookup[row_index][aspect] = str(sentiment)
    return lookup


def pair_predictions_from_aspects(
    aspect_predictions: list[list[str]],
    sentiment_lookup: list[dict[str, str]],
) -> list[list[str]]:
    if len(aspect_predictions) != len(sentiment_lookup):
        raise ValueError("aspect predictions and sentiment lookup must have the same number of rows.")

    predictions: list[list[str]] = []
    for aspects, row_sentiments in zip(aspect_predictions, sentiment_lookup):
        predictions.append(
            [
                format_pair_label(aspect, row_sentiments[aspect])
                for aspect in aspects
                if aspect in row_sentiments
            ]
        )
    return predictions


def candidate_pair_labels(candidate_aspects: list[str]) -> list[str]:
    return [
        format_pair_label(aspect, sentiment)
        for aspect in candidate_aspects
        for sentiment in SENTIMENTS
    ]
