from __future__ import annotations

import numpy as np

from msc_project.experiments.taxonomy_description_weight_transfer import (
    pseudo_unseen_scores,
    select_presence_threshold,
    sigmoid,
    synthesise_weight,
)


def test_kernel_ridge_recovers_known_linear_descriptor_weight_structure() -> None:
    rng = np.random.default_rng(13)
    descriptors = rng.normal(size=(11, 5))
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    mapping = rng.normal(size=(5, 4))
    offset = rng.normal(size=4)
    weights = descriptors @ mapping + offset
    target = rng.normal(size=5)
    target /= np.linalg.norm(target)
    observed = target @ mapping + offset
    predicted = synthesise_weight(
        descriptors,
        weights,
        target,
        family="kernel_ridge",
        alpha=1e-8,
    )
    np.testing.assert_allclose(predicted, observed, atol=2e-5, rtol=2e-5)


def test_transfer_baselines_are_finite_and_deterministic() -> None:
    rng = np.random.default_rng(7)
    descriptors = rng.normal(size=(6, 3))
    weights = rng.normal(size=(6, 5))
    target = rng.normal(size=3)
    settings = (
        ("mean_seen_weight", {}),
        ("nearest_description_weight", {}),
        ("cosine_barycentric_weight", {"temperature": 0.2}),
        ("kernel_ridge", {"alpha": 0.1}),
    )
    for family, kwargs in settings:
        first = synthesise_weight(descriptors, weights, target, family=family, **kwargs)
        second = synthesise_weight(descriptors, weights, target, family=family, **kwargs)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)


def test_pseudo_unseen_selection_does_not_accept_an_outer_target() -> None:
    rng = np.random.default_rng(11)
    descriptors = rng.normal(size=(4, 3))
    weights = rng.normal(size=(4, 3))
    features = rng.normal(size=(9, 2))
    targets = rng.integers(0, 2, size=(9, 4))
    pooled_targets, pooled_scores, cosine = pseudo_unseen_scores(
        descriptors,
        weights,
        features,
        targets,
        family="mean_seen_weight",
    )
    assert pooled_targets.shape == pooled_scores.shape == (36,)
    assert np.isfinite(pooled_scores).all()
    assert np.isfinite(cosine)
    selection = select_presence_threshold(pooled_targets, pooled_scores)
    assert 0.0 <= selection.f1 <= 1.0


def test_stable_sigmoid_handles_large_logits() -> None:
    values = sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(values, np.asarray([0.0, 0.5, 1.0]), atol=1e-12)
