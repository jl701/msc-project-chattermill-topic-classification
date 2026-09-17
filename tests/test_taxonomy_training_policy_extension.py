"""Synthetic tests only. No official partition is loaded."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_taxonomy_training_policy_extension as extension


def test_fitted_feature_replay_matches_legacy_exactly():
    train = pd.DataFrame({"text": [f"helpful staff fast app service {i % 7}" for i in range(30)]
                         + [f"bad slow software crash support {i % 9}" for i in range(30)]})
    validation = train.iloc[:7].copy()
    actual, fitted = extension.fit_spaces(train, validation, (2, 4))
    expected = extension.legacy._fit_review_space(train, validation, (2, 4))
    for k in actual:
        for j in (0, 1):
            np.testing.assert_array_equal(actual[k][j], expected[k][j])
        sparse = fitted["vectoriser"].transform(validation.text)
        transform = fitted["transforms"][k]
        np.testing.assert_array_equal(transform["scaler"].transform(transform["projector"].transform(sparse)), actual[k][1])


def test_single_class_seen_head_refused():
    with pytest.raises(ValueError, match="both classes"):
        extension.legacy._fit_seen_weights(np.ones((5, 3)), np.zeros((5, 1)), classifier_c=1)


def test_original_selection_ties_are_deterministic():
    record = {"pseudo_unseen_f1": .5, "pseudo_unseen_average_precision": .6,
              "svd_components": 8, "classifier_c": 1.0, "generator_parameter": 1.0,
              "family": "kernel_ridge", "configuration_id": "test"}
    a = dict(record, generator_parameter=10.)
    assert extension.legacy._rank_configuration(a) > extension.legacy._rank_configuration(record)


def test_receipt_rejects_altered_artifact(tmp_path):
    extension.write_json(tmp_path / "evidence.json", {"value": 1})
    extension.receipt(tmp_path, {"study_id": "fixture"})
    extension.verify_receipt(tmp_path)
    extension.write_json(tmp_path / "evidence.json", {"value": 2})
    with pytest.raises((ValueError, RuntimeError)):
        extension.verify_receipt(tmp_path)


def test_synchronised_bootstrap_preserves_structural_invariance():
    from analyse_taxonomy_training_policy_sensitivity import bootstrap
    matrix = np.random.default_rng(13).integers(0, 3, (3, 10, 3))
    values, pooled = bootstrap(np.stack([matrix, matrix]), draws=30, seed=13)
    np.testing.assert_array_equal(values[:, 0], values[:, 1])
    np.testing.assert_array_equal(pooled[:, 0], pooled[:, 1])


def test_failed_parent_never_released(tmp_path):
    from campaign_taxonomy_training_policy_extension import check_dependency
    path = tmp_path / "state.json"
    extension.write_json(path, {"status": "complete", "failure_count": 1, "test_contract_count": 0})
    with pytest.raises(RuntimeError, match="failed"):
        check_dependency(path)
