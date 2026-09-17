"""Data and lightweight runtimes for the preregistered true two-stage study."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from msc_project.data.fabsa import format_pair_label
from msc_project.experiments.taxonomy_two_stage import TfidfAspectPresenceScorer
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS
from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration


REPRESENTATION_VARIANTS = (
    "name_and_description",
    "name_only",
    "description_only",
    "rich",
    "rich_positive",
)


def _minimal_aspects(resource: Mapping[str, object]) -> Mapping[str, object]:
    value = resource.get("minimal_aspects", resource.get("aspects"))
    if not isinstance(value, Mapping):
        raise ValueError("Description resource has no minimal aspect mapping.")
    return value


def _rich_aspects(resource: Mapping[str, object]) -> Mapping[str, object]:
    value = resource.get("rich_aspects")
    if not isinstance(value, Mapping):
        raise ValueError("Description resource has no rich aspect mapping.")
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
    if variant in {"rich", "rich_positive"}:
        raw_card = _rich_aspects(resource).get(aspect)
        if not isinstance(raw_card, Mapping):
            raise ValueError(f"Missing rich aspect card: {aspect!r}.")
        aliases = raw_card.get("aliases")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"Rich aspect aliases are invalid: {aspect!r}.")
        positive = (
            f"Aspect: {aspect}. Definition: {raw_card['definition']} "
            f"Aliases: {'; '.join(str(value) for value in aliases)}. "
            f"Inclusion boundary: {raw_card['inclusion_boundary']}"
        )
        if variant == "rich_positive":
            return positive
        return f"{positive} Contrastive boundary: {raw_card['contrastive_boundary']}"
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


def _stable_demo_key(
    *,
    seed: int,
    stage: str,
    answer: str,
    row_uid: str,
    aspect: str,
) -> str:
    value = f"{seed}|{stage}|{answer}|{row_uid}|{aspect}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _select_distinct_aspects(
    candidates: Sequence[TwoStageDemonstration],
    *,
    count: int,
    seed: int,
    stage: str,
    answer: str,
) -> list[TwoStageDemonstration]:
    ordered = sorted(
        candidates,
        key=lambda value: (
            _stable_demo_key(
                seed=seed,
                stage=stage,
                answer=answer,
                row_uid=value.row_uid,
                aspect=value.candidate_aspect,
            ),
            value.row_uid,
            value.candidate_aspect,
        ),
    )
    selected: list[TwoStageDemonstration] = []
    used_aspects: set[str] = set()
    for value in ordered:
        if value.candidate_aspect in used_aspects:
            continue
        selected.append(value)
        used_aspects.add(value.candidate_aspect)
        if len(selected) == count:
            return selected
    for value in ordered:
        if value in selected:
            continue
        selected.append(value)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Insufficient eligible {stage} demonstrations for answer {answer!r}."
    )


def select_qwen_two_stage_demonstrations(
    train_rows: pd.DataFrame,
    seen_aspects: Sequence[str],
    resource: Mapping[str, object],
    *,
    seed: int = 13,
) -> dict[str, tuple[TwoStageDemonstration, ...]]:
    """Select fixed cross-aspect examples using train-only stable hashes."""

    aspects = tuple(str(value) for value in seen_aspects)
    if not aspects or len(aspects) != len(set(aspects)):
        raise ValueError("Seen aspects must be non-empty and unique.")
    if train_rows.empty or {"row_uid", "text"} - set(train_rows.columns):
        raise ValueError("Training rows lack the fields required for demonstrations.")
    variants = {aspect: "name_and_description" for aspect in aspects}
    aspect_grid = build_aspect_grid(train_rows, aspects, variants, resource)
    stage_1: dict[str, list[TwoStageDemonstration]] = {"Y": [], "N": []}
    for row in aspect_grid.itertuples(index=False):
        answer = "Y" if int(row.target) == 1 else "N"
        stage_1[answer].append(
            TwoStageDemonstration(
                row_uid=str(row.row_uid),
                candidate_aspect=str(row.candidate_aspect),
                review_text=str(row.text),
                aspect_candidate=str(row.candidate_text),
                answer=answer,
            )
        )
    selected_aspect = tuple(
        _select_distinct_aspects(
            stage_1[answer],
            count=2,
            seed=seed,
            stage="aspect",
            answer=answer,
        )[index]
        for answer in ("Y", "N")
        for index in range(2)
    )

    answer_for_sentiment = {"negative": "A", "neutral": "B", "positive": "C"}
    stage_2: dict[str, list[TwoStageDemonstration]] = {
        answer: [] for answer in answer_for_sentiment.values()
    }
    for row in train_rows.itertuples(index=False):
        labels = tuple(
            (str(aspect), str(sentiment))
            for aspect, sentiment in _row_labels(pd.Series(row._asdict()))
            if str(aspect) in variants
        )
        sentiments_by_aspect: dict[str, list[str]] = {}
        for aspect, sentiment in labels:
            sentiments_by_aspect.setdefault(aspect, []).append(sentiment)
        for aspect, sentiments in sentiments_by_aspect.items():
            unique_sentiments = tuple(dict.fromkeys(sentiments))
            if len(unique_sentiments) != 1:
                continue
            sentiment = unique_sentiments[0]
            if sentiment not in answer_for_sentiment:
                continue
            answer = answer_for_sentiment[sentiment]
            stage_2[answer].append(
                TwoStageDemonstration(
                    row_uid=str(row.row_uid),
                    candidate_aspect=aspect,
                    review_text="" if pd.isna(row.text) else str(row.text),
                    aspect_candidate=render_aspect_candidate(
                        aspect, "name_and_description", resource
                    ),
                    answer=answer,
                )
            )
    selected_sentiment = tuple(
        _select_distinct_aspects(
            stage_2[answer],
            count=1,
            seed=seed,
            stage="sentiment",
            answer=answer,
        )[0]
        for answer in ("A", "B", "C")
    )
    return {"aspect": selected_aspect, "sentiment": selected_sentiment}


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
        # Grid construction and prompt rendering do not need neural dependencies.
        from msc_project.baselines.candidate_tfidf import (
            UnifiedTfidfPairConfig,
            UnifiedTfidfPairScorer,
        )

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
