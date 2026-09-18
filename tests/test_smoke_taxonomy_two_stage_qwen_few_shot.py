from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke_taxonomy_two_stage_qwen_few_shot.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_few_shot_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_probabilities_accepts_normalised_values() -> None:
    MODULE._validate_probabilities(np.asarray([[0.3, 0.7]]), width=2)


def test_validate_probabilities_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="3-way"):
        MODULE._validate_probabilities(np.asarray([[0.2, 0.2, 0.2]]), width=3)
