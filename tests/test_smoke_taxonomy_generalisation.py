from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SCRIPT = PROJECT_ROOT / "scripts" / "smoke_taxonomy_generalisation.py"
SPEC = importlib.util.spec_from_file_location("smoke_taxonomy_generalisation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_synthetic_smoke_fixture_has_no_official_rows() -> None:
    frame = MODULE.synthetic_frame()
    assert set(frame["data_source"]) == {"synthetic"}
    assert set(frame["original_split"]) == {"train", "validation"}
    assert not frame["row_uid"].str.startswith("test:").any()


def test_smoke_parameter_reductions_are_explicit() -> None:
    distil = MODULE.smoke_parameters(
        "distilbert_review_candidate_cross_encoder"
    )
    qlora = MODULE.smoke_parameters("qwen_candidate_pair_qlora")
    assert distil["epochs"] == 1
    assert distil["max_length"] == 64
    assert qlora["epochs"] == 1
    assert qlora["max_length"] == 128
