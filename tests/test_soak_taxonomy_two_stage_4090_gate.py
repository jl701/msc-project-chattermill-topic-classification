from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "soak_taxonomy_two_stage_4090_gate.py"
SPEC = importlib.util.spec_from_file_location("taxonomy_4090_soak", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preregistered_soak_configuration_is_accepted() -> None:
    MODULE.validate_soak_configuration(
        scoring_seconds=600,
        query_count=32,
        scoring_batch_size=8,
        qlora_training_budget=4096,
    )


@pytest.mark.parametrize(
    ("seconds", "queries", "batch", "budget"),
    [
        (59, 32, 8, 4096),
        (600, 30, 8, 4096),
        (600, 32, 6, 4096),
        (600, 32, 8, 2048),
    ],
)
def test_soak_configuration_rejects_protocol_drift(
    seconds: int,
    queries: int,
    batch: int,
    budget: int,
) -> None:
    with pytest.raises(ValueError):
        MODULE.validate_soak_configuration(
            scoring_seconds=seconds,
            query_count=queries,
            scoring_batch_size=batch,
            qlora_training_budget=budget,
        )


def test_probability_summary_requires_finite_normalised_rows() -> None:
    summary = MODULE.probability_summary(
        np.asarray([[0.25, 0.75], [0.8, 0.2]], dtype=np.float64),
        width=2,
    )
    assert summary["minimum"] == pytest.approx(0.2)
    assert summary["maximum"] == pytest.approx(0.8)
    assert summary["standard_deviation"] > 0

    with pytest.raises(ValueError):
        MODULE.probability_summary(
            np.asarray([[float("nan"), float("nan")]], dtype=np.float64),
            width=2,
        )
