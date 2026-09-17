from __future__ import annotations

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_retrieval_few_shot import (
    assemble_demonstrations,
    pair_embeddings,
    select_diverse_examples,
)


def _pool(target: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_uid": ["train:1", "train:1", "train:2", "train:3"],
            "candidate_aspect": ["A", "B", "B", "C"],
            "text": ["one", "one", "two", "three"],
            "candidate_text": ["card A", "card B", "card B", "card C"],
            "target": [target] * 4,
        }
    )


def test_pair_embeddings_are_finite_unit_vectors() -> None:
    reviews = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cards = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    values = pair_embeddings(reviews, cards)
    assert values.shape == (2, 2)
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)


def test_retrieval_enforces_row_and_aspect_diversity() -> None:
    values = select_diverse_examples(
        np.asarray([0.99, 0.98, 0.97, 0.96]),
        _pool(1),
        answer="Y",
        count=2,
    )
    assert [(value.row_uid, value.candidate_aspect) for value in values] == [
        ("train:1", "A"),
        ("train:2", "B"),
    ]
    assembled = assemble_demonstrations(values, tuple(
        value.__class__(
            row_uid=value.row_uid,
            candidate_aspect=value.candidate_aspect,
            review_text=value.review_text,
            aspect_candidate=value.aspect_candidate,
            answer="N",
        )
        for value in values
    ))
    assert tuple(value.answer for value in assembled) == ("Y", "Y", "N", "N")


def test_retrieval_allows_same_aspect_only_after_row_diversity() -> None:
    pool = pd.DataFrame(
        {
            "row_uid": ["train:1", "train:2", "train:3"],
            "candidate_aspect": ["A", "A", "A"],
            "text": ["one", "two", "three"],
            "candidate_text": ["card A"] * 3,
            "target": [1, 1, 1],
        }
    )
    values = select_diverse_examples(
        np.asarray([0.9, 0.8, 0.7]), pool, answer="Y", count=2
    )
    assert [value.row_uid for value in values] == ["train:1", "train:2"]
    assert [value.candidate_aspect for value in values] == ["A", "A"]
