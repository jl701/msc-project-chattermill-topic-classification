import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyse_taxonomy_training_policy_sensitivity as analysis
import run_taxonomy_policy_qwen_sensitivity as qwen
import finish_taxonomy_training_policy_sensitivity as finish


def test_mean_fold_and_pooled_are_distinct():
    counts = np.array([[1, 0, 0], [0, 9, 1]])
    assert analysis.f1(counts).mean() == .5
    assert analysis.f1(counts.sum(axis=0)) == pytest.approx(1 / 6)


def test_bootstrap_synchronises_reviews_across_identical_systems():
    base = np.array([[[1, 0, 0], [0, 1, 1]], [[2, 0, 0], [0, 0, 1]]])
    counts = np.stack([base, base])
    draws, pooled = analysis.bootstrap(counts, draws=401, seed=13)
    assert draws.shape == (401, 2)
    assert np.array_equal(draws[:, 0], draws[:, 1])
    assert np.array_equal(pooled[:, 0], pooled[:, 1])
    assert np.array_equal(draws, analysis.bootstrap(counts, draws=401, seed=13)[0])


def test_zero_count_f1_convention():
    assert analysis.f1(np.zeros(3)) == 0


def test_invalid_probabilities_stop():
    for values in ([[float("nan"), .5]], [[1.2, -.2]], [[.4, .4]], [[1, 0, 0]]):
        with pytest.raises(ValueError):
            qwen.validate_probabilities("aspect", values)
    assert qwen.validate_probabilities("sentiment", [[.2, .3, .5]]).shape == (1, 3)


def test_campaign_release_requires_exact_completion():
    state = dict(failure_count=0, test_contract_count=0, planned=24, completed=24, status="complete", child_pid=None)
    assert finish.campaign_complete(state, 24)
    for change in ({"completed": 23}, {"planned": 25}, {"failure_count": 1},
                   {"test_contract_count": 1}, {"child_pid": 100}, {"status": "stopped_failure"}):
        with pytest.raises(RuntimeError):
            finish.campaign_complete({**state, **change}, 24)
    assert not finish.campaign_complete({**state, "status": "running", "completed": 2, "child_pid": 100}, 24)


def test_gpu_guard_rejects_repeated_slowdown_or_overheat():
    sample = {"temperature.gpu": "70", "clocks_event_reasons.hw_slowdown": "Not Active"}
    assert finish.guard_gpu(sample, 0, 86) == 0
    active = {**sample, "clocks_event_reasons.hw_slowdown": "Active"}
    assert finish.guard_gpu(active, 0, 86) == 1
    with pytest.raises(RuntimeError):
        finish.guard_gpu(active, 1, 86)
    with pytest.raises(RuntimeError):
        finish.guard_gpu({**sample, "temperature.gpu": "86"}, 0, 86)
