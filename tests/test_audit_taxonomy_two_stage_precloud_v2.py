from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_taxonomy_two_stage_precloud_v2.py"
)
SPEC = importlib.util.spec_from_file_location("precloud_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_non_finite_paths_reports_nested_values() -> None:
    assert MODULE._non_finite_paths({"a": [1.0, {"b": float("inf")}]}) == [
        "a[1].b"
    ]
    assert MODULE._non_finite_paths({"a": 1.0}) == []
