from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _script_module():
    path = PROJECT_ROOT / "scripts/analyse_taxonomy_post_supervisor_formal_v2.py"
    spec = importlib.util.spec_from_file_location("formal_v2_analysis_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_formal_result_graph_is_three_times_twenty_seven() -> None:
    module = _script_module()
    assert len(module.EXPECTED_RESULTS) == 81
    for method in module.METHODS:
        identities = [key for key in module.EXPECTED_RESULTS if key[0] == method]
        assert len(identities) == 27


def test_exact_sign_flip_and_bootstrap_are_deterministic() -> None:
    module = _script_module()
    values = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    first = module.percentile_bootstrap(values)
    second = module.percentile_bootstrap(values)
    assert first == second
    assert 0.0 <= module.exact_sign_flip_p(values) <= 1.0


def test_nonfinite_guard_rejects_nested_nan() -> None:
    module = _script_module()
    try:
        module.reject_nonfinite({"safe": [1.0, {"bad": math.nan}]}, "root")
    except ValueError as error:
        assert "root.safe[1].bad" in str(error)
    else:
        raise AssertionError("Nested NaN should fail closed")
