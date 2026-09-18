from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_taxonomy_two_stage_training_manifests.py"
)
SPEC = importlib.util.spec_from_file_location("training_manifest_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sha256_values_is_order_independent() -> None:
    assert MODULE._sha256_values(["train:2", "train:1"]) == MODULE._sha256_values(
        ["train:1", "train:2"]
    )


def test_audit_rejects_non_train_input() -> None:
    frame = pd.DataFrame(
        {"original_split": ["train", "validation"], "row_uid": ["a", "b"]}
    )
    try:
        MODULE.audit_training_scopes(frame, {})
    except ValueError as error:
        assert "train split only" in str(error)
    else:
        raise AssertionError("Expected mixed split input to be rejected.")
