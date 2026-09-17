"""Leakage-safe utilities for the bounded rich-description interface study."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize


INTERFACES = (
    "D",
    "R1_concat",
    "R2_positive_concat",
    "R3_prototype_max",
    "R4_prototype_top2",
    "R5_contrastive_top2_l0.10",
    "R5_contrastive_top2_l0.25",
    "R5_contrastive_top2_l0.50",
)
ELIGIBLE_RICH_INTERFACES = INTERFACES[1:]
R5_LAMBDAS = (0.10, 0.25, 0.50)
SIMPLICITY_ORDER = (
    "R2_positive_concat",
    "R4_prototype_top2",
    "R3_prototype_max",
    "R5_contrastive_top2_l0.10",
    "R5_contrastive_top2_l0.25",
    "R5_contrastive_top2_l0.50",
    "R1_concat",
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def development_partition(row_uid: str, *, seed: int = 20260820, folds: int = 3) -> int:
    if folds < 2:
        raise ValueError("Development partitioning requires at least two folds.")
    digest = hashlib.sha256(f"{seed}\x1f{row_uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


@dataclass(frozen=True)
class RichAspectFields:
    aspect: str
    base: str
    aliases: tuple[str, ...]
    inclusion: str
    exclusion: str
    positive_concat: str
    full_concat: str

    @property
    def positive_prototypes(self) -> tuple[str, ...]:
        return (self.base, *self.aliases, self.inclusion)


def rich_aspect_fields(
    aspect: str,
    resource: Mapping[str, object],
) -> RichAspectFields:
    minimal = resource.get("minimal_aspects")
    rich = resource.get("rich_aspects")
    if not isinstance(minimal, Mapping) or not isinstance(rich, Mapping):
        raise ValueError("The description bundle lacks minimal or rich aspect mappings.")
    if aspect not in minimal or aspect not in rich:
        raise ValueError(f"Unknown canonical aspect: {aspect!r}.")
    raw = rich[aspect]
    if not isinstance(raw, Mapping):
        raise ValueError(f"Rich aspect card is invalid: {aspect!r}.")
    aliases_raw = raw.get("aliases")
    if not isinstance(aliases_raw, list) or not aliases_raw:
        raise ValueError(f"Rich aliases are invalid: {aspect!r}.")
    definition = str(minimal[aspect]).strip()
    aliases = tuple(f"Aspect expression: {str(value).strip()}." for value in aliases_raw)
    inclusion_value = str(raw.get("inclusion_boundary", "")).strip()
    exclusion_value = str(raw.get("contrastive_boundary", "")).strip()
    if not inclusion_value or not exclusion_value:
        raise ValueError(f"Rich boundaries are incomplete: {aspect!r}.")
    base = f"Aspect: {aspect}. Definition: {definition}"
    inclusion = f"Include when: {inclusion_value}"
    exclusion = f"Exclude when: {exclusion_value}"
    alias_list = "; ".join(str(value).strip() for value in aliases_raw)
    positive_concat = (
        f"{base} Aliases: {alias_list}. Inclusion boundary: {inclusion_value}"
    )
    full_concat = (
        f"{positive_concat} Contrastive boundary: {exclusion_value}"
    )
    return RichAspectFields(
        aspect=aspect,
        base=base,
        aliases=aliases,
        inclusion=inclusion,
        exclusion=exclusion,
        positive_concat=positive_concat,
        full_concat=full_concat,
    )


def all_field_texts(
    aspects: Sequence[str],
    resource: Mapping[str, object],
) -> tuple[str, ...]:
    texts: list[str] = []
    for aspect in aspects:
        fields = rich_aspect_fields(str(aspect), resource)
        texts.extend(
            [
                fields.base,
                *fields.aliases,
                fields.inclusion,
                fields.exclusion,
                fields.positive_concat,
                fields.full_concat,
            ]
        )
    return tuple(dict.fromkeys(texts))


def _rowwise_cosine(left: object, right: object) -> np.ndarray:
    left_value = normalize(left, norm="l2", copy=False)
    right_value = normalize(right, norm="l2", copy=False)
    return np.asarray(left_value.multiply(right_value).sum(axis=1)).reshape(-1)


class TfidfFieldSimilarity:
    """Word/character field similarity fitted without pseudo-unseen descriptors."""

    method_id = "tfidf_field_similarity"

    def __init__(self) -> None:
        self.word = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=80_000,
            sublinear_tf=True,
        )
        self.char = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=120_000,
            sublinear_tf=True,
        )
        self._fitted = False
        self.fit_candidate_text_sha256: str | None = None

    def fit(
        self,
        review_texts: Sequence[str],
        seen_field_texts: Sequence[str],
    ) -> "TfidfFieldSimilarity":
        review_values = [str(value) for value in review_texts]
        candidate_values = [str(value) for value in seen_field_texts]
        if not review_values or not candidate_values:
            raise ValueError("TF-IDF field fitting requires reviews and seen descriptors.")
        corpus = [*review_values, *candidate_values]
        self.word.fit(corpus)
        self.char.fit(corpus)
        self.fit_candidate_text_sha256 = canonical_json_sha256(candidate_values)
        self._fitted = True
        return self

    def score(self, review_texts: Sequence[str], candidate_text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF-IDF field scorer is not fitted.")
        reviews = [str(value) for value in review_texts]
        if not reviews:
            raise ValueError("TF-IDF field scoring requires review text.")
        candidates = [str(candidate_text)] * len(reviews)
        word = _rowwise_cosine(
            self.word.transform(reviews), self.word.transform(candidates)
        )
        char = _rowwise_cosine(
            self.char.transform(reviews), self.char.transform(candidates)
        )
        scores = (word + char) / 2.0
        if not np.isfinite(scores).all():
            raise ValueError("TF-IDF field scores must be finite.")
        return np.asarray(scores, dtype=float)


def score_interfaces(
    review_texts: Sequence[str],
    fields: RichAspectFields,
    score_text: Callable[[Sequence[str], str], np.ndarray],
) -> dict[str, np.ndarray]:
    base = score_text(review_texts, fields.base)
    full = score_text(review_texts, fields.full_concat)
    positive_concat = score_text(review_texts, fields.positive_concat)
    positive_parts = np.column_stack(
        [score_text(review_texts, value) for value in fields.positive_prototypes]
    )
    if positive_parts.shape[1] < 2:
        raise AssertionError("Rich prototype aggregation requires at least two fields.")
    maximum = positive_parts.max(axis=1)
    top_two = np.sort(positive_parts, axis=1)[:, -2:].mean(axis=1)
    exclusion = score_text(review_texts, fields.exclusion)
    output = {
        "D": base,
        "R1_concat": full,
        "R2_positive_concat": positive_concat,
        "R3_prototype_max": maximum,
        "R4_prototype_top2": top_two,
    }
    for value in R5_LAMBDAS:
        output[f"R5_contrastive_top2_l{value:.2f}"] = top_two - value * np.maximum(
            0.0, exclusion - top_two
        )
    if tuple(output) != INTERFACES:
        raise AssertionError("Rich interface output order changed.")
    if any(len(scores) != len(review_texts) for scores in output.values()):
        raise AssertionError("Rich interface score lengths changed.")
    if not all(np.isfinite(scores).all() for scores in output.values()):
        raise ValueError("Rich interface scores must be finite.")
    return output


def select_f1_threshold(targets: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(targets, dtype=int)
    values = np.asarray(scores, dtype=float)
    if truth.ndim != 1 or values.shape != truth.shape or not len(truth):
        raise ValueError("Threshold targets and scores must be aligned vectors.")
    if set(np.unique(truth)) != {0, 1} or not np.isfinite(values).all():
        raise ValueError("Threshold selection requires finite scores and both classes.")
    order = np.argsort(-values, kind="stable")
    ordered_scores = values[order]
    ordered_truth = truth[order]
    cumulative_tp = np.cumsum(ordered_truth)
    cumulative_fp = np.cumsum(1 - ordered_truth)
    boundary = np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    tp = cumulative_tp[boundary]
    fp = cumulative_fp[boundary]
    fn = int(truth.sum()) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    thresholds = ordered_scores[boundary]
    best_f1 = float(f1.max())
    best_threshold = float(thresholds[np.flatnonzero(f1 == best_f1)].max())
    prediction = values >= best_threshold
    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "predicted_positive": int(prediction.sum()),
        "positive": int(truth.sum()),
    }


def presence_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    truth = np.asarray(targets, dtype=int).astype(bool)
    values = np.asarray(scores, dtype=float)
    if truth.ndim != 1 or values.shape != truth.shape or not len(truth):
        raise ValueError("Presence targets and scores must be aligned vectors.")
    if not np.isfinite(values).all() or len(np.unique(truth)) != 2:
        raise ValueError("Presence metrics require finite scores and both classes.")
    prediction = values >= float(threshold)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    return {
        "average_precision": float(average_precision_score(truth, values)),
        "f1": float(2 * tp / denominator if denominator else 0.0),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_rows": int(truth.sum()),
        "false_positive_rows_per_100": float(fp * 100 / len(truth)),
        "positive_score_mean": float(values[truth].mean()),
        "negative_score_mean": float(values[~truth].mean()),
        "score_separation": float(values[truth].mean() - values[~truth].mean()),
    }


def select_global_rich_interface(
    records: Sequence[Mapping[str, object]],
    *,
    near_tie: float = 0.005,
) -> dict[str, object]:
    if not records:
        raise ValueError("Rich selection requires development records.")
    grouped: dict[str, list[float]] = {value: [] for value in INTERFACES}
    keys: set[tuple[str, str, int]] = set()
    for record in records:
        method = str(record["method_id"])
        aspect = str(record["pseudo_unseen_aspect"])
        partition = int(record["row_partition"])
        interface = str(record["interface"])
        if interface not in grouped:
            raise ValueError(f"Unexpected rich interface: {interface!r}.")
        key = (method, aspect, partition)
        keys.add(key)
        grouped[interface].append(float(record["average_precision"]))
    expected = len(keys)
    if any(len(values) != expected for values in grouped.values()):
        raise ValueError("Rich selection records do not form a complete interface grid.")
    baseline = np.asarray(grouped["D"], dtype=float)
    means = {
        interface: float(np.mean(np.asarray(values, dtype=float) - baseline))
        for interface, values in grouped.items()
        if interface != "D"
    }
    best_delta = max(means.values())
    near = {
        interface
        for interface, value in means.items()
        if best_delta - value <= near_tie + 1e-12
    }
    selected = next(value for value in SIMPLICITY_ORDER if value in near)
    return {
        "selected_interface": selected,
        "selected_mean_ap_delta_vs_D": means[selected],
        "best_mean_ap_delta_vs_D": best_delta,
        "near_tie_candidates": [value for value in SIMPLICITY_ORDER if value in near],
        "mean_ap_delta_vs_D": means,
        "record_keys": expected,
        "near_tie_tolerance": float(near_tie),
    }


def audit_rich_resource(resource: Mapping[str, object]) -> dict[str, object]:
    order = resource.get("canonical_order")
    if not isinstance(order, list):
        raise ValueError("Description resource lacks canonical order.")
    fields = {aspect: rich_aspect_fields(str(aspect), resource) for aspect in order}
    canonical_children = {
        str(aspect): str(aspect).split(":", 1)[1].strip().casefold()
        for aspect in order
    }
    rows: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []
    for aspect in order:
        value = fields[str(aspect)]
        positive_tokens = set(_TOKEN_PATTERN.findall(value.positive_concat.casefold()))
        exclusion_tokens = set(_TOKEN_PATTERN.findall(value.exclusion.casefold()))
        mentions = sorted(
            other
            for other, child in canonical_children.items()
            if other != aspect and child in value.exclusion.casefold()
        )
        rows.append(
            {
                "aspect": aspect,
                "base_tokens": len(_TOKEN_PATTERN.findall(value.base.casefold())),
                "alias_count": len(value.aliases),
                "positive_concat_tokens": len(
                    _TOKEN_PATTERN.findall(value.positive_concat.casefold())
                ),
                "exclusion_tokens": len(_TOKEN_PATTERN.findall(value.exclusion.casefold())),
                "positive_exclusion_token_overlap": len(positive_tokens & exclusion_tokens),
                "other_aspects_named_in_exclusion": mentions,
            }
        )
    for index, left in enumerate(order):
        left_tokens = set(
            _TOKEN_PATTERN.findall(fields[str(left)].positive_concat.casefold())
        )
        for right in order[index + 1 :]:
            right_tokens = set(
                _TOKEN_PATTERN.findall(fields[str(right)].positive_concat.casefold())
            )
            union = left_tokens | right_tokens
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "positive_token_jaccard": float(
                        len(left_tokens & right_tokens) / len(union) if union else 0.0
                    ),
                }
            )
    return {
        "aspects": rows,
        "pairwise_positive_overlap": pairwise,
        "aspects_with_cross_label_exclusion_mentions": int(
            sum(bool(row["other_aspects_named_in_exclusion"]) for row in rows)
        ),
        "mean_positive_tokens": float(
            np.mean([float(row["positive_concat_tokens"]) for row in rows])
        ),
        "mean_exclusion_tokens": float(
            np.mean([float(row["exclusion_tokens"]) for row in rows])
        ),
        "resource_audit_sha256": canonical_json_sha256(rows),
    }
