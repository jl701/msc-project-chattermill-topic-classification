from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke_taxonomy_two_stage_distilbert.py"
)
SPEC = importlib.util.spec_from_file_location("distilbert_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_assert_probabilities_accepts_normalised_rows() -> None:
    MODULE._assert_probabilities(
        np.asarray([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]]), width=3
    )


def test_assert_probabilities_rejects_non_finite_or_unnormalised_rows() -> None:
    with pytest.raises(ValueError, match="Invalid 3-class"):
        MODULE._assert_probabilities(np.asarray([[0.2, 0.3, np.nan]]), width=3)
    with pytest.raises(ValueError, match="Invalid 3-class"):
        MODULE._assert_probabilities(np.asarray([[0.2, 0.3, 0.4]]), width=3)
