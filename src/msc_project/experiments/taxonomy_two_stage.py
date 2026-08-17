"""Validation-only helpers for hierarchical aspect-then-sentiment decisions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

from msc_project.data.fabsa import format_pair_label
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.experiments.taxonomy_protocol import (
    PAIR_KEY,
    _strict_threshold_sweep,
    strict_threshold_candidates,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS


@dataclass(frozen=True)
class DecoderThresholdSelection:
    threshold: float
    sweep: pd.DataFrame


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_GENERIC_ASPECT_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "aspect",
        "definition",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def _tokens(value: object) -> set[str]:
    return set(_TOKEN_PATTERN.findall(str(value).casefold()))


def _coverage(review_tokens: set[str], cue_tokens: set[str]) -> float:
    return len(review_tokens & cue_tokens) / len(cue_tokens) if cue_tokens else 0.0


def _rowwise_cosine(left: object, right: object) -> np.ndarray:
    left_value = normalize(left, norm="l2", copy=False)
    right_value = normalize(right, norm="l2", copy=False)
    return np.asarray(left_value.multiply(right_value).sum(axis=1)).reshape(-1)


class TfidfAspectPresenceScorer:
    """Cheap aspect-only scorer with no candidate-identity coefficient."""

    def __init__(
        self,
        *,
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        max_word_features: int = 80_000,
        max_char_features: int = 120_000,
        classifier_c: float = 1.0,
        max_iter: int = 1_000,
        seed: int = 13,
    ) -> None:
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=word_ngram_range,
            min_df=1,
            max_features=max_word_features,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            min_df=1,
            max_features=max_char_features,
            sublinear_tf=True,
        )
        self.classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=classifier_c,
                        class_weight="balanced",
                        max_iter=max_iter,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
        self._fitted = False

    @staticmethod
    def _validate(frame: pd.DataFrame, *, target: bool) -> None:
        required = {"text", "candidate_text", "candidate_aspect", "row_uid"}
        if target:
            required.add("target")
        missing = sorted(required - set(frame.columns))
        if missing or frame.empty:
            raise ValueError(f"Aspect manifest is invalid; missing={missing}.")

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        reviews = frame["text"].astype(str).tolist()
        candidates = frame["candidate_text"].astype(str).tolist()
        word_cosine = _rowwise_cosine(
            self.word_vectorizer.transform(reviews),
            self.word_vectorizer.transform(candidates),
        )
        char_cosine = _rowwise_cosine(
            self.char_vectorizer.transform(reviews),
            self.char_vectorizer.transform(candidates),
        )
        aspect_coverage = []
        candidate_coverage = []
        for review, candidate, aspect in zip(
            reviews, candidates, frame["candidate_aspect"].astype(str)
        ):
            review_tokens = _tokens(review)
            aspect_coverage.append(
                _coverage(
                    review_tokens,
                    _tokens(aspect) - _GENERIC_ASPECT_TOKENS,
                )
            )
            candidate_coverage.append(
                _coverage(
                    review_tokens,
                    _tokens(candidate) - _GENERIC_ASPECT_TOKENS,
                )
            )
        return np.column_stack(
            [
                word_cosine,
                char_cosine,
                np.asarray(aspect_coverage, dtype=float),
                np.asarray(candidate_coverage, dtype=float),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> "TfidfAspectPresenceScorer":
        self._validate(frame, target=True)
        targets = frame["target"].astype(int).to_numpy()
        if set(np.unique(targets)) != {0, 1}:
            raise ValueError("Aspect scorer training requires both target classes.")
        corpus = frame["text"].astype(str).tolist() + frame[
            "candidate_text"
        ].astype(str).tolist()
        self.word_vectorizer.fit(corpus)
        self.char_vectorizer.fit(corpus)
        self.classifier.fit(self._features(frame), targets)
        self._fitted = True
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        self._validate(frame, target=False)
        if not self._fitted:
            raise RuntimeError("Aspect scorer must be fitted before scoring.")
        values = np.asarray(
            self.classifier.predict_proba(self._features(frame))[:, 1],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("TF-IDF aspect scores must be finite.")
        return values


def _validate_pair_grid(frame: pd.DataFrame) -> None:
    required = {*PAIR_KEY, "pair_label", "target", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pair grid is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Pair grid must not be empty.")
    if frame.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Pair grid contains duplicate pair identities.")
    scores = frame["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Pair scores must be finite.")


def crossfit_partition(row_uid: str, folds: int = 5) -> int:
    if folds < 2:
        raise ValueError("Cross-fitting requires at least two folds.")
    digest = hashlib.sha256(str(row_uid).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % folds


def hierarchical_candidates(scored_grid: pd.DataFrame) -> pd.DataFrame:
    """Collapse three sentiment scores into one deterministic aspect candidate."""

    _validate_pair_grid(scored_grid)
    sentiments = set(scored_grid["candidate_sentiment"].astype(str))
    if sentiments != set(CANDIDATE_SENTIMENTS):
        raise ValueError("Hierarchical decoding requires all three sentiments.")
    counts = scored_grid.groupby(
        ["row_uid", "candidate_aspect"], sort=False
    )["candidate_sentiment"].nunique()
    if not counts.eq(len(CANDIDATE_SENTIMENTS)).all():
        raise ValueError("Every review-aspect group must contain three sentiments.")

    sentiment_order = {
        sentiment: index for index, sentiment in enumerate(CANDIDATE_SENTIMENTS)
    }
    ordered = scored_grid.copy()
    ordered["_sentiment_order"] = ordered["candidate_sentiment"].map(
        sentiment_order
    )
    ordered = ordered.sort_values(
        ["row_uid", "candidate_aspect", "score", "_sentiment_order"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    selected = ordered.drop_duplicates(
        ["row_uid", "candidate_aspect"], keep="first"
    ).copy()
    selected["is_correct_pair"] = selected["target"].astype(int).eq(1)
    return selected[
        [
            "row_uid",
            "candidate_aspect",
            "candidate_sentiment",
            "pair_label",
            "score",
            "is_correct_pair",
        ]
    ].reset_index(drop=True)


def _hierarchical_threshold_sweep(scored_grid: pd.DataFrame) -> pd.DataFrame:
    candidates = hierarchical_candidates(scored_grid)
    gold_counts = (
        scored_grid.assign(_gold=scored_grid["target"].astype(int).eq(1))
        .groupby(scored_grid["row_uid"].astype(str), sort=False)["_gold"]
        .sum()
    )
    row_uids = candidates["row_uid"].astype(str).to_numpy()
    unique_rows, row_indices = np.unique(row_uids, return_inverse=True)
    gold_by_row = np.asarray(
        [int(gold_counts.get(uid, 0)) for uid in unique_rows], dtype=np.int64
    )
    predicted_by_row = np.zeros(len(unique_rows), dtype=np.int64)
    true_positive_by_row = np.zeros(len(unique_rows), dtype=np.int64)
    scores = candidates["score"].to_numpy(dtype=float)
    correct = candidates["is_correct_pair"].to_numpy(dtype=bool)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    sorted_row_indices = row_indices[order]
    thresholds = sorted(strict_threshold_candidates(scores), reverse=True)

    total_gold = int(gold_by_row.sum())
    predicted_count = 0
    true_positive_count = 0
    samples_f1_sum = 0.0
    false_positive_rows = 0
    pointer = 0
    records: list[dict[str, float]] = []
    for threshold in thresholds:
        while pointer < len(sorted_scores) and sorted_scores[pointer] >= threshold:
            row_index = int(sorted_row_indices[pointer])
            old_predicted = int(predicted_by_row[row_index])
            old_true_positive = int(true_positive_by_row[row_index])
            gold_count = int(gold_by_row[row_index])
            old_denominator = gold_count + old_predicted
            old_samples_f1 = (
                2.0 * old_true_positive / old_denominator
                if old_denominator
                else 0.0
            )
            predicted_by_row[row_index] += 1
            predicted_count += 1
            if bool(sorted_correct[pointer]):
                true_positive_by_row[row_index] += 1
                true_positive_count += 1
            if old_predicted == 0 and gold_count == 0:
                false_positive_rows += 1
            new_predicted = int(predicted_by_row[row_index])
            new_true_positive = int(true_positive_by_row[row_index])
            new_samples_f1 = (
                2.0 * new_true_positive / (gold_count + new_predicted)
            )
            samples_f1_sum += new_samples_f1 - old_samples_f1
            pointer += 1

        false_positive_count = predicted_count - true_positive_count
        false_negative_count = total_gold - true_positive_count
        denominator = 2 * true_positive_count + false_positive_count + false_negative_count
        records.append(
            {
                "threshold": float(threshold),
                "pair_micro_f1": (
                    2.0 * true_positive_count / denominator if denominator else 0.0
                ),
                "pair_samples_f1": float(samples_f1_sum / len(unique_rows)),
                "pair_micro_precision": (
                    true_positive_count / predicted_count if predicted_count else 0.0
                ),
                "presence_false_positive_rows_per_100": float(
                    false_positive_rows * 100 / len(unique_rows)
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("threshold").reset_index(drop=True)


def _rank_threshold_sweep(sweep: pd.DataFrame) -> DecoderThresholdSelection:
    if sweep.empty:
        raise ValueError("Threshold sweep is empty.")
    rows = sweep.to_dict(orient="records")
    ranked = sorted(
        rows,
        key=lambda row: (
            row["pair_micro_f1"],
            row["pair_samples_f1"],
            row["pair_micro_precision"],
            -row["presence_false_positive_rows_per_100"],
            row["threshold"],
        ),
        reverse=True,
    )
    return DecoderThresholdSelection(float(ranked[0]["threshold"]), sweep)


def select_pair_threshold(scored_grid: pd.DataFrame) -> DecoderThresholdSelection:
    _validate_pair_grid(scored_grid)
    return _rank_threshold_sweep(_strict_threshold_sweep(scored_grid))


def select_hierarchical_threshold(
    scored_grid: pd.DataFrame,
) -> DecoderThresholdSelection:
    return _rank_threshold_sweep(_hierarchical_threshold_sweep(scored_grid))


def pair_prediction_mask(scored_grid: pd.DataFrame, threshold: float) -> pd.Series:
    _validate_pair_grid(scored_grid)
    return pd.Series(
        scored_grid["score"].to_numpy(dtype=float) >= threshold,
        index=scored_grid.index,
        dtype=bool,
    )


def hierarchical_prediction_mask(
    scored_grid: pd.DataFrame, threshold: float
) -> pd.Series:
    _validate_pair_grid(scored_grid)
    selected = hierarchical_candidates(scored_grid)
    selected = selected[selected["score"].astype(float) >= threshold]
    identities = set(
        zip(
            selected["row_uid"].astype(str),
            selected["candidate_aspect"].astype(str),
            selected["candidate_sentiment"].astype(str),
        )
    )
    return pd.Series(
        [
            (str(uid), str(aspect), str(sentiment)) in identities
            for uid, aspect, sentiment in zip(
                scored_grid["row_uid"],
                scored_grid["candidate_aspect"],
                scored_grid["candidate_sentiment"],
            )
        ],
        index=scored_grid.index,
        dtype=bool,
    )


def two_stage_candidates(scored_grid: pd.DataFrame) -> pd.DataFrame:
    """Select one sentiment per review-aspect while retaining an aspect score."""

    required = {*PAIR_KEY, "pair_label", "target", "aspect_score", "sentiment_score"}
    missing = sorted(required - set(scored_grid.columns))
    if missing or scored_grid.empty:
        raise ValueError(f"Two-stage grid is invalid; missing={missing}.")
    if scored_grid.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Two-stage grid contains duplicate pair identities.")
    values = scored_grid[["aspect_score", "sentiment_score"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Two-stage scores must be finite.")
    counts = scored_grid.groupby(
        ["row_uid", "candidate_aspect"], sort=False
    )["candidate_sentiment"].nunique()
    if not counts.eq(len(CANDIDATE_SENTIMENTS)).all():
        raise ValueError("Every two-stage review-aspect group needs three sentiments.")

    sentiment_order = {
        sentiment: index for index, sentiment in enumerate(CANDIDATE_SENTIMENTS)
    }
    ordered = scored_grid.copy()
    ordered["_sentiment_order"] = ordered["candidate_sentiment"].map(
        sentiment_order
    )
    ordered = ordered.sort_values(
        [
            "row_uid",
            "candidate_aspect",
            "sentiment_score",
            "_sentiment_order",
        ],
        ascending=[True, True, False, True],
        kind="stable",
    )
    selected = ordered.drop_duplicates(
        ["row_uid", "candidate_aspect"], keep="first"
    ).copy()
    selected["is_correct_pair"] = selected["target"].astype(int).eq(1)
    return selected[
        [
            "row_uid",
            "candidate_aspect",
            "candidate_sentiment",
            "pair_label",
            "aspect_score",
            "sentiment_score",
            "is_correct_pair",
        ]
    ].reset_index(drop=True)


def _two_stage_threshold_sweep(scored_grid: pd.DataFrame) -> pd.DataFrame:
    """Reuse the exact hierarchical sweep with independent aspect scores."""

    candidates = two_stage_candidates(scored_grid).rename(
        columns={"aspect_score": "score"}
    )
    proxy = scored_grid.copy()
    proxy["score"] = proxy["aspect_score"].astype(float)

    # The optimized sweep only needs the chosen pair and each row's total gold count.
    gold_counts = (
        proxy.assign(_gold=proxy["target"].astype(int).eq(1))
        .groupby(proxy["row_uid"].astype(str), sort=False)["_gold"]
        .sum()
    )
    row_uids = candidates["row_uid"].astype(str).to_numpy()
    unique_rows, row_indices = np.unique(row_uids, return_inverse=True)
    gold_by_row = np.asarray(
        [int(gold_counts.get(uid, 0)) for uid in unique_rows], dtype=np.int64
    )
    predicted_by_row = np.zeros(len(unique_rows), dtype=np.int64)
    true_positive_by_row = np.zeros(len(unique_rows), dtype=np.int64)
    scores = candidates["score"].to_numpy(dtype=float)
    correct = candidates["is_correct_pair"].to_numpy(dtype=bool)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    sorted_row_indices = row_indices[order]
    thresholds = sorted(strict_threshold_candidates(scores), reverse=True)

    total_gold = int(gold_by_row.sum())
    predicted_count = 0
    true_positive_count = 0
    samples_f1_sum = 0.0
    false_positive_rows = 0
    pointer = 0
    records: list[dict[str, float]] = []
    for threshold in thresholds:
        while pointer < len(sorted_scores) and sorted_scores[pointer] >= threshold:
            row_index = int(sorted_row_indices[pointer])
            old_predicted = int(predicted_by_row[row_index])
            old_true_positive = int(true_positive_by_row[row_index])
            gold_count = int(gold_by_row[row_index])
            old_denominator = gold_count + old_predicted
            old_samples_f1 = (
                2.0 * old_true_positive / old_denominator
                if old_denominator
                else 0.0
            )
            predicted_by_row[row_index] += 1
            predicted_count += 1
            if bool(sorted_correct[pointer]):
                true_positive_by_row[row_index] += 1
                true_positive_count += 1
            if old_predicted == 0 and gold_count == 0:
                false_positive_rows += 1
            new_predicted = int(predicted_by_row[row_index])
            new_true_positive = int(true_positive_by_row[row_index])
            samples_f1_sum += (
                2.0 * new_true_positive / (gold_count + new_predicted)
            ) - old_samples_f1
            pointer += 1

        false_positive_count = predicted_count - true_positive_count
        false_negative_count = total_gold - true_positive_count
        denominator = (
            2 * true_positive_count + false_positive_count + false_negative_count
        )
        records.append(
            {
                "threshold": float(threshold),
                "pair_micro_f1": (
                    2.0 * true_positive_count / denominator if denominator else 0.0
                ),
                "pair_samples_f1": float(samples_f1_sum / len(unique_rows)),
                "pair_micro_precision": (
                    true_positive_count / predicted_count if predicted_count else 0.0
                ),
                "presence_false_positive_rows_per_100": float(
                    false_positive_rows * 100 / len(unique_rows)
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("threshold").reset_index(
        drop=True
    )


def select_two_stage_threshold(
    scored_grid: pd.DataFrame,
) -> DecoderThresholdSelection:
    return _rank_threshold_sweep(_two_stage_threshold_sweep(scored_grid))


def two_stage_prediction_mask(
    scored_grid: pd.DataFrame, threshold: float
) -> pd.Series:
    selected = two_stage_candidates(scored_grid)
    selected = selected[selected["aspect_score"].astype(float) >= threshold]
    identities = set(
        zip(
            selected["row_uid"].astype(str),
            selected["candidate_aspect"].astype(str),
            selected["candidate_sentiment"].astype(str),
        )
    )
    return pd.Series(
        [
            (str(uid), str(aspect), str(sentiment)) in identities
            for uid, aspect, sentiment in zip(
                scored_grid["row_uid"],
                scored_grid["candidate_aspect"],
                scored_grid["candidate_sentiment"],
            )
        ],
        index=scored_grid.index,
        dtype=bool,
    )


def conditional_sentiment_metrics(scored_grid: pd.DataFrame) -> dict[str, float]:
    """Audit stage 2 independently on gold aspect instances."""

    selected = two_stage_candidates(scored_grid)
    gold_counts = (
        scored_grid[scored_grid["target"].astype(int).eq(1)]
        .groupby(["row_uid", "candidate_aspect"], sort=False)
        .size()
    )
    selected["gold_sentiment_count"] = [
        int(gold_counts.get((uid, aspect), 0))
        for uid, aspect in zip(
            selected["row_uid"].astype(str),
            selected["candidate_aspect"].astype(str),
        )
    ]
    gold = selected[selected["gold_sentiment_count"].gt(0)]
    single = gold[gold["gold_sentiment_count"].eq(1)]
    return {
        "conditional_sentiment_accuracy_all_gold_aspects": (
            float(gold["is_correct_pair"].mean()) if len(gold) else 0.0
        ),
        "conditional_sentiment_accuracy_single_gold_sentiment": (
            float(single["is_correct_pair"].mean()) if len(single) else 0.0
        ),
        "gold_aspect_instances": int(len(gold)),
        "single_gold_sentiment_instances": int(len(single)),
        "multi_gold_sentiment_instances": int(len(gold) - len(single)),
    }


def evaluate_prediction_mask(
    scored_grid: pd.DataFrame,
    prediction_mask: Sequence[bool] | pd.Series,
    *,
    aspects: Iterable[str],
) -> dict[str, float]:
    _validate_pair_grid(scored_grid)
    allowed_aspects = tuple(sorted(str(value) for value in aspects))
    frame = scored_grid[
        scored_grid["candidate_aspect"].astype(str).isin(allowed_aspects)
    ].copy()
    mask = pd.Series(prediction_mask, index=scored_grid.index, dtype=bool).loc[
        frame.index
    ]
    frame["_gold"] = np.where(
        frame["target"].astype(int).eq(1), frame["pair_label"], None
    )
    frame["_pred"] = np.where(mask, frame["pair_label"], None)
    grouped = frame.groupby(frame["row_uid"].astype(str), sort=False)
    row_uids = sorted(frame["row_uid"].astype(str).unique())
    gold_by_uid = {
        str(uid): sorted(value for value in group["_gold"] if value is not None)
        for uid, group in grouped
    }
    pred_by_uid = {
        str(uid): sorted(value for value in group["_pred"] if value is not None)
        for uid, group in grouped
    }
    classes = [
        format_pair_label(aspect, sentiment)
        for aspect in allowed_aspects
        for sentiment in CANDIDATE_SENTIMENTS
    ]
    return evaluate_pair_and_aspect(
        [gold_by_uid[uid] for uid in row_uids],
        [pred_by_uid[uid] for uid in row_uids],
        classes,
    )


def crossfit_decoder_comparison(
    scored_grid: pd.DataFrame,
    *,
    aspects: Iterable[str],
    folds: int = 5,
) -> dict[str, object]:
    _validate_pair_grid(scored_grid)
    frame = scored_grid.copy()
    frame["_crossfit_fold"] = frame["row_uid"].astype(str).map(
        lambda value: crossfit_partition(value, folds)
    )
    pair_masks: list[pd.Series] = []
    hierarchical_masks: list[pd.Series] = []
    thresholds: list[dict[str, float | int]] = []
    for fold_index in range(folds):
        train = frame[frame["_crossfit_fold"] != fold_index].copy()
        evaluation = frame[frame["_crossfit_fold"] == fold_index].copy()
        if train.empty or evaluation.empty:
            raise ValueError("A deterministic cross-fit partition is empty.")
        pair_selection = select_pair_threshold(train)
        hierarchy_selection = select_hierarchical_threshold(train)
        pair_masks.append(
            pair_prediction_mask(evaluation, pair_selection.threshold)
        )
        hierarchical_masks.append(
            hierarchical_prediction_mask(evaluation, hierarchy_selection.threshold)
        )
        thresholds.append(
            {
                "crossfit_fold": fold_index,
                "pair_threshold": pair_selection.threshold,
                "hierarchical_threshold": hierarchy_selection.threshold,
                "selection_rows": int(train["row_uid"].nunique()),
                "evaluation_rows": int(evaluation["row_uid"].nunique()),
            }
        )
    pair_mask = pd.concat(pair_masks).sort_index().reindex(frame.index)
    hierarchical_mask = pd.concat(hierarchical_masks).sort_index().reindex(
        frame.index
    )
    if pair_mask.isna().any() or hierarchical_mask.isna().any():
        raise AssertionError("Cross-fitted predictions do not cover every score row.")
    return {
        "pair_decoder": evaluate_prediction_mask(
            frame, pair_mask, aspects=aspects
        ),
        "hierarchical_decoder": evaluate_prediction_mask(
            frame, hierarchical_mask, aspects=aspects
        ),
        "thresholds": thresholds,
    }
