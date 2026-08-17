"""Deterministic train-only manifests for genuine two-stage taxonomy models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from msc_project.experiments.taxonomy_two_stage_runtime import (
    build_aspect_grid,
    render_aspect_candidate,
)


SENTIMENT_TO_ANSWER = {"negative": "A", "neutral": "B", "positive": "C"}
TASKS = ("aspect_presence", "sentiment")


def _stable_key(
    row: Mapping[str, object],
    *,
    seed: int,
    task: str,
    label: str,
) -> str:
    identity = "|".join(
        (
            str(seed),
            task,
            label,
            str(row.get("row_uid", "")),
            str(row.get("candidate_aspect", "")),
            str(row.get("candidate_sentiment", "")),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _balanced_without_replacement(
    frame: pd.DataFrame,
    *,
    label_column: str,
    label_order: Sequence[str],
    budget: int,
    seed: int,
    task: str,
) -> pd.DataFrame:
    if budget < 1:
        raise ValueError("Training budget must be positive.")
    labels = tuple(str(value) for value in label_order)
    if set(frame[label_column].astype(str)) - set(labels):
        raise ValueError(f"Unexpected labels in {task} manifest.")
    ordered: dict[str, list[dict[str, object]]] = {}
    for label in labels:
        records = frame[frame[label_column].astype(str).eq(label)].to_dict(
            orient="records"
        )
        ordered[label] = sorted(
            records,
            key=lambda row: (
                _stable_key(row, seed=seed, task=task, label=label),
                str(row["row_uid"]),
                str(row["candidate_aspect"]),
                str(row.get("candidate_sentiment", "")),
            ),
        )
    if sum(len(values) for values in ordered.values()) < budget:
        raise ValueError(f"Insufficient unique {task} examples for budget {budget}.")

    quotas = {
        label: budget // len(labels) + int(index < budget % len(labels))
        for index, label in enumerate(labels)
    }
    counts = {label: min(quotas[label], len(ordered[label])) for label in labels}
    remaining = budget - sum(counts.values())
    while remaining:
        progressed = False
        for label in labels:
            if counts[label] < len(ordered[label]):
                counts[label] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise AssertionError("Balanced allocation could not fill its budget.")
    selected = [row for label in labels for row in ordered[label][: counts[label]]]
    selected.sort(
        key=lambda row: (
            _stable_key(
                row,
                seed=seed,
                task=task,
                label=str(row[label_column]),
            ),
            str(row["row_uid"]),
            str(row["candidate_aspect"]),
            str(row.get("candidate_sentiment", "")),
        )
    )
    return pd.DataFrame.from_records(selected).reset_index(drop=True)


def _labels(row: pd.Series) -> tuple[tuple[str, str], ...]:
    for column in ("supervision_labels", "labels"):
        if column in row.index:
            return tuple(
                (str(aspect), str(sentiment)) for aspect, sentiment in row[column]
            )
    raise ValueError("Training rows require supervision_labels or labels.")


def build_two_stage_training_manifests(
    train_rows: pd.DataFrame,
    seen_aspects: Sequence[str],
    resource: Mapping[str, object],
    *,
    total_budget: int = 4096,
    aspect_budget: int = 2048,
    seed: int = 13,
) -> dict[str, pd.DataFrame]:
    """Create deterministic stage-1 and stage-2 manifests without replacement."""

    if total_budget < 2 or aspect_budget < 1 or aspect_budget >= total_budget:
        raise ValueError("Two-stage budgets are invalid.")
    if "original_split" in train_rows and set(
        train_rows["original_split"].astype(str)
    ) != {"train"}:
        raise ValueError("Two-stage training manifests may contain only train rows.")
    aspects = tuple(str(value) for value in seen_aspects)
    if not aspects or len(aspects) != len(set(aspects)):
        raise ValueError("Seen aspects must be non-empty and unique.")
    variants = {aspect: "name_and_description" for aspect in aspects}

    aspect = build_aspect_grid(train_rows, aspects, variants, resource).rename(
        columns={"target": "binary_target"}
    )
    aspect["task"] = "aspect_presence"
    aspect["answer"] = aspect["binary_target"].map({0: "N", 1: "Y"})
    aspect["candidate_sentiment"] = ""
    aspect = _balanced_without_replacement(
        aspect,
        label_column="answer",
        label_order=("Y", "N"),
        budget=aspect_budget,
        seed=seed,
        task="aspect_presence",
    )

    records: list[dict[str, object]] = []
    allowed = set(aspects)
    for _, row in train_rows.assign(_uid=train_rows["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    ).iterrows():
        for candidate_aspect, sentiment in _labels(row):
            if candidate_aspect not in allowed:
                continue
            if sentiment not in SENTIMENT_TO_ANSWER:
                raise ValueError(f"Unexpected sentiment label: {sentiment!r}.")
            records.append(
                {
                    "row_uid": str(row["row_uid"]),
                    "text": "" if pd.isna(row["text"]) else str(row["text"]),
                    "candidate_aspect": candidate_aspect,
                    "candidate_sentiment": sentiment,
                    "candidate_text": render_aspect_candidate(
                        candidate_aspect, "name_and_description", resource
                    ),
                    "representation_variant": "name_and_description",
                    "task": "sentiment",
                    "answer": SENTIMENT_TO_ANSWER[sentiment],
                }
            )
    sentiment = pd.DataFrame.from_records(records)
    identity = ["row_uid", "candidate_aspect", "candidate_sentiment"]
    if sentiment.empty or sentiment.duplicated(identity).any():
        raise ValueError("Sentiment training identities are empty or duplicated.")
    sentiment = _balanced_without_replacement(
        sentiment,
        label_column="answer",
        label_order=("A", "B", "C"),
        budget=total_budget - aspect_budget,
        seed=seed,
        task="sentiment",
    )
    if len(aspect) + len(sentiment) != total_budget:
        raise AssertionError("Two-stage training manifests violate the total budget.")
    for manifest in (aspect, sentiment):
        if set(manifest["candidate_aspect"].astype(str)) - allowed:
            raise AssertionError("A held-out aspect entered a two-stage training manifest.")
    return {"aspect_presence": aspect, "sentiment": sentiment}


def training_manifest_sha256(frame: pd.DataFrame) -> str:
    required = {
        "task",
        "row_uid",
        "text",
        "candidate_aspect",
        "candidate_sentiment",
        "candidate_text",
        "answer",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Training manifest is invalid; missing={missing}.")
    records = frame[sorted(required)].to_dict(orient="records")
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_manifest_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "task": str(frame["task"].iloc[0]),
        "answer_counts": {
            str(key): int(value)
            for key, value in frame["answer"].value_counts().sort_index().items()
        },
        "unique_rows": int(frame["row_uid"].astype(str).nunique()),
        "unique_aspects": int(frame["candidate_aspect"].astype(str).nunique()),
        "manifest_sha256": training_manifest_sha256(frame),
    }
