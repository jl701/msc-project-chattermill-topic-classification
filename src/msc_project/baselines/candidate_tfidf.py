"""CPU-only candidate-pair TF-IDF scorer and shared manifest validation.

Extracted without changing features, fitting order or default parameters. The
former unified_pair_scorers module re-exports these names for compatibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

PAIR_MANIFEST_COLUMNS = (
    "text",
    "candidate_text",
    "candidate_aspect",
    "candidate_sentiment",
    "row_uid",
    "negative_type",
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_GENERIC_CANDIDATE_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "aspect",
        "about",
        "for",
        "in",
        "is",
        "it",
        "keywords",
        "mentions",
        "of",
        "on",
        "or",
        "review",
        "sentiment",
        "the",
        "this",
        "to",
        "topic",
        "with",
    }
)
_SENTIMENT_CUES = {
    "positive": frozenset(
        {
            "amazing",
            "easy",
            "excellent",
            "fast",
            "friendly",
            "good",
            "great",
            "happy",
            "helpful",
            "love",
            "positive",
            "recommend",
            "satisfied",
        }
    ),
    "negative": frozenset(
        {
            "awful",
            "bad",
            "difficult",
            "disappointed",
            "issue",
            "negative",
            "poor",
            "problem",
            "rude",
            "slow",
            "terrible",
            "unhappy",
        }
    ),
    "neutral": frozenset({"average", "fine", "neutral", "neither", "okay", "ok"}),
}


def validate_pair_manifest(frame: pd.DataFrame, *, require_target: bool) -> None:
    """Validate the shared candidate-pair contract without mutating the input."""

    required = set(PAIR_MANIFEST_COLUMNS)
    if require_target:
        required.add("target")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate-pair manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Candidate-pair manifest must not be empty.")
    for column in ("text", "candidate_text", "candidate_aspect", "candidate_sentiment", "row_uid"):
        if frame[column].isna().any():
            raise ValueError(f"Candidate-pair manifest column {column!r} contains missing values.")
    if require_target:
        numeric_targets = pd.to_numeric(frame["target"], errors="raise")
        targets = set(numeric_targets.tolist())
        if not targets.issubset({0, 1}):
            raise ValueError("Candidate-pair target values must be binary (0 or 1).")


def _text_values(frame: pd.DataFrame, column: str) -> list[str]:
    return [str(value) for value in frame[column].tolist()]


def _tokens(value: object) -> set[str]:
    return set(_TOKEN_PATTERN.findall(str(value).lower()))


def _token_sequence(value: object) -> list[str]:
    return _TOKEN_PATTERN.findall(str(value).lower())


def _content_tokens(value: object) -> set[str]:
    return _tokens(value) - _GENERIC_CANDIDATE_TOKENS


def _coverage(review_tokens: set[str], cue_tokens: set[str]) -> float:
    if not cue_tokens:
        return 0.0
    return len(review_tokens & cue_tokens) / len(cue_tokens)


def _cue_density(review_tokens: list[str], cue_tokens: set[str]) -> float:
    if not review_tokens:
        return 0.0
    return sum(token in cue_tokens for token in review_tokens) / len(review_tokens)


def _rowwise_cosine(left, right) -> np.ndarray:
    left = normalize(left, norm="l2", copy=False)
    right = normalize(right, norm="l2", copy=False)
    return np.asarray(left.multiply(right).sum(axis=1)).reshape(-1)


@dataclass(frozen=True)
class UnifiedTfidfPairConfig:
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    min_df: int = 1
    max_word_features: int | None = 80_000
    max_char_features: int | None = 120_000
    classifier_c: float = 1.0
    feature_ablation: str = "all_six"
    max_iter: int = 1_000
    seed: int = 13


class UnifiedTfidfPairScorer:
    """Label-transferable candidate scorer based only on pairwise interactions.

    Both vectorisers and the logistic-regression calibration layer are fitted on
    the supplied training manifest.  No candidate identity feature is exposed
    to the classifier, so an unseen held-out aspect can be scored from its text.
    """

    feature_names = (
        "word_review_candidate_cosine",
        "character_review_candidate_cosine",
        "aspect_cue_token_coverage",
        "candidate_token_coverage",
        "sentiment_cue_token_coverage",
        "review_sentiment_cue_density",
    )
    _feature_ablation_indices = {
        "all_six": (0, 1, 2, 3, 4, 5),
        "char_cosine_and_cues": (1, 2, 3, 4, 5),
        "word_cosine_and_cues": (0, 2, 3, 4, 5),
    }

    def __init__(self, config: UnifiedTfidfPairConfig | None = None) -> None:
        self.config = config or UnifiedTfidfPairConfig()
        if self.config.feature_ablation not in self._feature_ablation_indices:
            raise ValueError(f"Unknown TF-IDF feature ablation: {self.config.feature_ablation!r}")
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.classifier: Pipeline | None = None

    def fit(self, train_manifest: pd.DataFrame) -> "UnifiedTfidfPairScorer":
        validate_pair_manifest(train_manifest, require_target=True)
        targets = pd.to_numeric(train_manifest["target"], errors="raise").astype(int).to_numpy()
        if len(np.unique(targets)) != 2:
            raise ValueError("TF-IDF pair scorer training requires both target classes.")

        candidate_corpus = self._candidate_corpus(train_manifest)
        fit_corpus = _text_values(train_manifest, "text") + candidate_corpus
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=self.config.word_ngram_range,
            min_df=self.config.min_df,
            max_features=self.config.max_word_features,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=self.config.char_ngram_range,
            min_df=self.config.min_df,
            max_features=self.config.max_char_features,
            sublinear_tf=True,
        )
        self.word_vectorizer.fit(fit_corpus)
        self.char_vectorizer.fit(fit_corpus)

        features = self.transform_features(train_manifest)
        self.classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.config.classifier_c,
                        class_weight="balanced",
                        max_iter=self.config.max_iter,
                        solver="lbfgs",
                        random_state=self.config.seed,
                    ),
                ),
            ]
        )
        self.classifier.fit(features, targets)
        return self

    @staticmethod
    def _candidate_corpus(manifest: pd.DataFrame) -> list[str]:
        return [
            f"{candidate_text} Aspect: {aspect}. Sentiment: {sentiment}."
            for candidate_text, aspect, sentiment in zip(
                _text_values(manifest, "candidate_text"),
                _text_values(manifest, "candidate_aspect"),
                _text_values(manifest, "candidate_sentiment"),
                strict=True,
            )
        ]

    def transform_features(self, manifest: pd.DataFrame) -> np.ndarray:
        validate_pair_manifest(manifest, require_target=False)
        if self.word_vectorizer is None or self.char_vectorizer is None:
            raise RuntimeError(
                "UnifiedTfidfPairScorer must be fitted before feature transformation."
            )

        reviews = _text_values(manifest, "text")
        candidates = self._candidate_corpus(manifest)
        word_cosine = _rowwise_cosine(
            self.word_vectorizer.transform(reviews),
            self.word_vectorizer.transform(candidates),
        )
        char_cosine = _rowwise_cosine(
            self.char_vectorizer.transform(reviews),
            self.char_vectorizer.transform(candidates),
        )

        candidate_coverages: list[float] = []
        aspect_coverages: list[float] = []
        sentiment_coverages: list[float] = []
        review_sentiment_densities: list[float] = []
        for review, candidate, aspect, sentiment in zip(
            reviews,
            _text_values(manifest, "candidate_text"),
            _text_values(manifest, "candidate_aspect"),
            _text_values(manifest, "candidate_sentiment"),
            strict=True,
        ):
            review_token_sequence = _token_sequence(review)
            review_tokens = set(review_token_sequence)
            aspect_coverages.append(_coverage(review_tokens, _content_tokens(aspect)))
            candidate_coverages.append(_coverage(review_tokens, _content_tokens(candidate)))
            sentiment_key = sentiment.strip().lower()
            sentiment_cues = set(_SENTIMENT_CUES.get(sentiment_key, frozenset())) | _content_tokens(
                sentiment_key
            )
            sentiment_coverages.append(_coverage(review_tokens, sentiment_cues))
            review_sentiment_densities.append(_cue_density(review_token_sequence, sentiment_cues))

        candidate_coverage = np.asarray(candidate_coverages, dtype=float)
        aspect_coverage = np.asarray(aspect_coverages, dtype=float)
        sentiment_coverage = np.asarray(sentiment_coverages, dtype=float)
        review_sentiment_density = np.asarray(review_sentiment_densities, dtype=float)
        full = np.column_stack(
            [
                word_cosine,
                char_cosine,
                aspect_coverage,
                candidate_coverage,
                sentiment_coverage,
                review_sentiment_density,
            ]
        )
        return full[:, self._feature_ablation_indices[self.config.feature_ablation]]

    def predict_proba(self, manifest: pd.DataFrame) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("UnifiedTfidfPairScorer must be fitted before scoring.")
        return np.asarray(
            self.classifier.predict_proba(self.transform_features(manifest)), dtype=float
        )

    def score_manifest(self, manifest: pd.DataFrame) -> np.ndarray:
        """Return one positive-class probability per manifest row."""

        return self.predict_proba(manifest)[:, 1]
