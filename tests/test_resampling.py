"""Synthetic regression checks against the pre-extraction implementation."""

import json
from pathlib import Path

import numpy as np
import pytest

from msc_project.evaluation.resampling import f1_from_counts, review_cluster_bootstrap


def test_matches_pre_extraction_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/resampling_baseline.json").read_text())
    balanced, pooled = review_cluster_bootstrap(fixture["counts"], draws=9)
    np.testing.assert_allclose(balanced, fixture["balanced"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(pooled, fixture["pooled"], rtol=0, atol=1e-15)


@pytest.mark.parametrize("draws", [1, 199, 200, 201, 401])
def test_pairing_seed_and_batch_boundary(draws):
    counts = np.random.default_rng(5).integers(0, 5, (1, 3, 8, 3))
    identical = np.repeat(counts, 2, axis=0)
    first = review_cluster_bootstrap(identical, draws=draws)
    second = review_cluster_bootstrap(identical, draws=draws)
    for a, b in zip(first, second, strict=True):
        assert a.shape == (draws, 2)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, 0], a[:, 1])


def test_empty_denominator_and_known_counts():
    np.testing.assert_array_equal(f1_from_counts([[0, 0, 0], [3, 1, 1]]), [0, 0.75])
    balanced, pooled = review_cluster_bootstrap(np.zeros((2, 3, 4, 3)), draws=7)
    assert not balanced.any()
    assert not pooled.any()


@pytest.mark.parametrize("counts", [0, [], [[1, 2]], [[1, -1, 0]], [[1, float("nan"), 0]]])
def test_f1_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        f1_from_counts(counts)


@pytest.mark.parametrize(
    "counts",
    [
        [],
        np.zeros((0, 2, 3, 3)),
        np.zeros((2, 3, 3)),
        np.full((1, 1, 1, 3), -1),
        np.full((1, 1, 1, 3), float("inf")),
    ],
)
def test_bootstrap_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        review_cluster_bootstrap(counts, draws=2)


@pytest.mark.parametrize("draws", [0, -1, True, 2.5])
def test_bootstrap_rejects_invalid_draws(draws):
    with pytest.raises(ValueError):
        review_cluster_bootstrap(np.ones((1, 1, 2, 3)), draws=draws)
