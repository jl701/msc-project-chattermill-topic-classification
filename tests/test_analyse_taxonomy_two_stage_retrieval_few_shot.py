from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for value in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import analyse_taxonomy_two_stage_retrieval_few_shot as study


def test_config_is_strictly_validation_only() -> None:
    path = (
        PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_no_retraining_extensions_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    study.validate_config(config)
    config["official_test_permitted"] = True
    try:
        study.validate_config(config)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsafe official-test access was accepted.")


def test_threshold_selection_prefers_frozen_boundary_when_metrics_tie() -> None:
    probability = np.asarray([0.9, 0.8, 0.2, 0.1])
    sentiment = np.asarray([[True, False, False]] * 4, dtype=bool)
    target = np.asarray(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )
    selected = study.select_threshold(
        probability,
        sentiment,
        target,
        np.ones(4, dtype=bool),
        [0.4, 0.5, 0.6],
    )
    assert selected["pair_micro_f1"] == 1.0
    assert selected["threshold"] == 0.5
