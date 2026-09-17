from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import analyse_taxonomy_two_stage_smooth_fusion as study


def test_preregistered_config_is_validation_only() -> None:
    path = (
        PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_no_retraining_extensions_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    study.validate_config(config)
    config["include_official_test"] = True
    try:
        study.validate_config(config)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsafe official-test access was not rejected.")


def test_source_evidence_centres_the_frozen_threshold() -> None:
    values = np.asarray([0.2, 0.4, 0.8], dtype=float)
    evidence = study.source_evidence(values, 0.4, epsilon=1e-6)
    assert evidence[0] < 0.0
    assert evidence[1] == 0.0
    assert evidence[2] > 0.0


def test_fusion_endpoints_reproduce_each_source_evidence() -> None:
    fewshot = np.asarray([-2.0, 0.0, 2.0])
    qlora = np.asarray([1.0, -1.0, 0.5])
    expected_fewshot = 1.0 / (1.0 + np.exp(-fewshot))
    expected_qlora = 1.0 / (1.0 + np.exp(-qlora))
    assert np.allclose(study.fused_probability(fewshot, qlora, 1.0), expected_fewshot)
    assert np.allclose(study.fused_probability(fewshot, qlora, 0.0), expected_qlora)


def test_policy_selection_respects_registered_tie_break() -> None:
    fewshot = np.asarray([2.0, 2.0, -2.0, -2.0])
    qlora = fewshot.copy()
    sentiment_mask = np.asarray([[True, False, False]] * 4, dtype=bool)
    target = np.asarray(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )
    selected = study.select_policy(
        fewshot,
        qlora,
        sentiment_mask,
        target,
        np.ones(4, dtype=bool),
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
    )
    assert selected["pair_micro_f1"] == 1.0
    assert selected["alpha"] == 1.0
    assert selected["threshold"] == 0.5
