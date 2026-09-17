"""Frozen, train-only retrieval helpers for dynamic Qwen demonstrations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration


def pair_embeddings(
    review_embeddings: np.ndarray, candidate_embeddings: np.ndarray
) -> np.ndarray:
    """Return the preregistered equal-weight review-plus-card representation."""

    reviews = np.asarray(review_embeddings, dtype=np.float32)
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    if reviews.shape != candidates.shape or reviews.ndim != 2 or not len(reviews):
        raise ValueError("Review and candidate embeddings must be aligned matrices.")
    if not np.isfinite(reviews).all() or not np.isfinite(candidates).all():
        raise ValueError("Pair components must be finite.")
    values = reviews + candidates
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0.0).any():
        raise ValueError("A retrieval pair has zero vector norm.")
    values = values / norms
    if not np.isfinite(values).all():
        raise ValueError("Pair normalisation produced a non-finite value.")
    return values.astype(np.float32, copy=False)


def select_diverse_examples(
    similarities: np.ndarray,
    pool: pd.DataFrame,
    *,
    answer: str,
    count: int = 2,
) -> tuple[TwoStageDemonstration, ...]:
    """Select deterministic high-similarity examples with row/aspect diversity."""

    required = {"row_uid", "candidate_aspect", "text", "candidate_text", "target"}
    missing = sorted(required - set(pool.columns))
    scores = np.asarray(similarities, dtype=float)
    if missing or pool.empty or scores.shape != (len(pool),):
        raise ValueError(f"Retrieval pool is invalid; missing={missing}.")
    expected_target = 1 if answer == "Y" else 0 if answer == "N" else None
    if expected_target is None or set(pool["target"].astype(int)) != {expected_target}:
        raise ValueError("Retrieval answer does not match the pool class.")
    if not np.isfinite(scores).all():
        raise ValueError("Retrieval similarities must be finite.")
    identities = pool[["row_uid", "candidate_aspect"]].astype(str)
    order = np.lexsort(
        (
            identities["candidate_aspect"].to_numpy(),
            identities["row_uid"].to_numpy(),
            -scores,
        )
    )
    selected: list[TwoStageDemonstration] = []
    used_rows: set[str] = set()
    used_aspects: set[str] = set()
    for require_new_aspect in (True, False):
        for index in order:
            row = pool.iloc[int(index)]
            row_uid = str(row["row_uid"])
            aspect = str(row["candidate_aspect"])
            if row_uid in used_rows or (require_new_aspect and aspect in used_aspects):
                continue
            selected.append(
                TwoStageDemonstration(
                    row_uid=row_uid,
                    candidate_aspect=aspect,
                    review_text=str(row["text"]),
                    aspect_candidate=str(row["candidate_text"]),
                    answer=answer,
                )
            )
            used_rows.add(row_uid)
            used_aspects.add(aspect)
            if len(selected) == count:
                return tuple(selected)
    raise ValueError(f"Insufficient diverse {answer} retrieval demonstrations.")


def assemble_demonstrations(
    positive: Sequence[TwoStageDemonstration],
    negative: Sequence[TwoStageDemonstration],
) -> tuple[TwoStageDemonstration, ...]:
    values = tuple(positive) + tuple(negative)
    if tuple(value.answer for value in values) != ("Y", "Y", "N", "N"):
        raise ValueError("Dynamic aspect demonstrations must be ordered Y,Y,N,N.")
    return values
