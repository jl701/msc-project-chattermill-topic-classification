"""Data and lightweight runtimes for the preregistered true two-stage study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from msc_project.baselines.unified_pair_scorers import (
    UnifiedTfidfPairConfig,
    UnifiedTfidfPairScorer,
)
from msc_project.data.fabsa import format_pair_label
from msc_project.experiments.taxonomy_two_stage import TfidfAspectPresenceScorer
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS


REPRESENTATION_VARIANTS = (
    "name_and_description",
    "name_only",
    "description_only",
)


def _minimal_aspects(resource: Mapping[str, object]) -> Mapping[str, object]:
    value = resource.get("minimal_aspects", resource.get("aspects"))
    if not isinstance(value, Mapping):
        raise ValueError("Description resource has no minimal aspect mapping.")
    return value


def render_aspect_candidate(
    aspect: str,
    variant: str,
    resource: Mapping[str, object],
) -> str:
    if variant not in REPRESENTATION_VARIANTS:
        raise ValueError(f"Unknown two-stage representation: {variant!r}.")
    aspects = _minimal_aspects(resource)
    if aspect not in aspects:
        raise ValueError(f"Unknown canonical aspect: {aspect!r}.")
    description = str(aspects[aspect]).strip()
    if variant == "name_and_description":
        return f"Aspect: {aspect}. Definition: {description}"
    if variant == "name_only":
        return f"Aspect: {aspect}."
    return f"Aspect definition: {description}"


def render_sentiment_candidate(
    aspect: str,
    sentiment: str,
    variant: str,
    resource: Mapping[str, object],
) -> str:
    if sentiment not in CANDIDATE_SENTIMENTS:
        raise ValueError(f"Unknown sentiment: {sentiment!r}.")
    return (
        f"{render_aspect_candidate(aspect, variant, resource)} "
        f"Candidate sentiment: {sentiment}."
    )


def representation_map(
    aspects: Sequence[str],
    heldout_aspects: Sequence[str],
    heldout_variant: str,
) -> dict[str, str]:
    if heldout_variant not in REPRESENTATION_VARIANTS:
        raise ValueError(f"Unknown held-out representation: {heldout_variant!r}.")
    heldout = set(str(value) for value in heldout_aspects)
    return {
        str(aspect): (
            heldout_variant if str(aspect) in heldout else "name_and_description"
        )
        for aspect in aspects
    }


def _row_labels(row: pd.Series) -> tuple[tuple[str, str], ...]:
    for column in ("supervision_labels", "labels"):
        if column in row.index:
            value = row[column]
            return tuple((str(aspect), str(sentiment)) for aspect, sentiment in value)
    raise ValueError("Rows require supervision_labels or labels.")


def build_aspect_grid(
    rows: pd.DataFrame,
    aspects: Sequence[str],
    variants: Mapping[str, str],
    resource: Mapping[str, object],
) -> pd.DataFrame:
    required = {"row_uid", "text"}
    missing = sorted(required - set(rows.columns))
    if missing or rows.empty:
        raise ValueError(f"Aspect-grid rows are invalid; missing={missing}.")
    if rows["row_uid"].astype(str).duplicated().any():
        raise ValueError("Aspect-grid row_uid values must be unique.")
    if set(aspects) != set(variants):
        raise ValueError("Aspect-grid variants must cover exactly the candidates.")
    records: list[dict[str, object]] = []
    for _, row in rows.assign(_uid=rows["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    ).iterrows():
        present = {aspect for aspect, _ in _row_labels(row)}
        for aspect in aspects:
            records.append(
                {
                    "row_uid": str(row["row_uid"]),
                    "text": "" if pd.isna(row["text"]) else str(row["text"]),
                    "candidate_aspect": str(aspect),
                    "candidate_text": render_aspect_candidate(
                        str(aspect), variants[str(aspect)], resource
                    ),
                    "representation_variant": variants[str(aspect)],
                    "target": int(str(aspect) in present),
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.duplicated(["row_uid", "candidate_aspect"]).any():
        raise AssertionError("Aspect grid identities are not unique.")
    return result


def build_sentiment_grid(
    rows: pd.DataFrame,
    aspects: Sequence[str],
    variants: Mapping[str, str],
    resource: Mapping[str, object],
    *,
    gold_aspects_only: bool = False,
) -> pd.DataFrame:
    required = {"row_uid", "text"}
    missing = sorted(required - set(rows.columns))
    if missing or rows.empty:
        raise ValueError(f"Sentiment-grid rows are invalid; missing={missing}.")
    if set(aspects) != set(variants):
        raise ValueError("Sentiment-grid variants must cover exactly the candidates.")
    allowed = set(str(value) for value in aspects)
    records: list[dict[str, object]] = []
    for _, row in rows.assign(_uid=rows["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    ).iterrows():
        labels = set(_row_labels(row))
        row_aspects = sorted(
            {aspect for aspect, _ in labels if aspect in allowed}
        ) if gold_aspects_only else list(aspects)
        for aspect in row_aspects:
            for sentiment in CANDIDATE_SENTIMENTS:
                target = int((str(aspect), str(sentiment)) in labels)
                records.append(
                    {
                        "row_uid": str(row["row_uid"]),
                        "text": "" if pd.isna(row["text"]) else str(row["text"]),
                        "candidate_aspect": str(aspect),
                        "candidate_sentiment": str(sentiment),
                        "candidate_text": render_sentiment_candidate(
                            str(aspect),
                            str(sentiment),
                            variants[str(aspect)],
                            resource,
                        ),
                        "representation_variant": variants[str(aspect)],
                        "target": target,
                        "pair_label": format_pair_label(str(aspect), str(sentiment)),
                        "negative_type": (
                            "gold_positive" if target else "conditional_wrong_sentiment"
                        ),
                    }
                )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("Sentiment grid contains no rows.")
    if result.duplicated(
        ["row_uid", "candidate_aspect", "candidate_sentiment"]
    ).any():
        raise AssertionError("Sentiment grid identities are not unique.")
    return result


class TfidfTrueTwoStageRuntime:
    """Separate TF-IDF aspect-presence and conditional-sentiment models."""

    method_id = "strict_train_only_tfidf"

    def __init__(self, *, seed: int = 13) -> None:
        self.aspect_scorer = TfidfAspectPresenceScorer(seed=seed)
        self.sentiment_scorer = UnifiedTfidfPairScorer(
            UnifiedTfidfPairConfig(
                classifier_c=1.0,
                feature_ablation="all_six",
                max_iter=1_000,
                seed=seed,
            )
        )
        self._fitted = False

    def fit(
        self,
        aspect_training_grid: pd.DataFrame,
        sentiment_training_grid: pd.DataFrame,
    ) -> "TfidfTrueTwoStageRuntime":
        self.aspect_scorer.fit(aspect_training_grid)
        self.sentiment_scorer.fit(sentiment_training_grid)
        self._fitted = True
        return self

    def score_aspects(self, grid: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Two-stage TF-IDF runtime is not fitted.")
        return self.aspect_scorer.score(grid)

    def score_sentiments(self, grid: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Two-stage TF-IDF runtime is not fitted.")
        values = self.sentiment_scorer.score_manifest(grid)
        if not np.isfinite(values).all():
            raise ValueError("TF-IDF sentiment scores must be finite.")
        return np.asarray(values, dtype=float)


def join_two_stage_scores(
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    aspect_scores: Sequence[float],
    sentiment_scores: Sequence[float],
) -> pd.DataFrame:
    aspects = aspect_grid.copy()
    sentiments = sentiment_grid.copy()
    aspects["aspect_score"] = np.asarray(aspect_scores, dtype=float)
    sentiments["sentiment_score"] = np.asarray(sentiment_scores, dtype=float)
    if len(aspects) != len(aspect_scores) or len(sentiments) != len(sentiment_scores):
        raise ValueError("Score lengths do not match the two-stage grids.")
    if not np.isfinite(aspects["aspect_score"]).all() or not np.isfinite(
        sentiments["sentiment_score"]
    ).all():
        raise ValueError("Two-stage scores must be finite.")
    merged = sentiments.merge(
        aspects[["row_uid", "candidate_aspect", "aspect_score"]],
        on=["row_uid", "candidate_aspect"],
        how="left",
        validate="many_to_one",
    )
    if merged["aspect_score"].isna().any():
        raise AssertionError("An aspect score is missing from the sentiment grid.")
    # Compatibility column for the shared metric validator.  In a true
    # two-stage decoder the thresholded scalar is the aspect-presence score.
    merged["score"] = merged["aspect_score"].astype(float)
    return merged
