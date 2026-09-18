from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "run_taxonomy_matched_one_vs_two_stage_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_taxonomy_matched_one_vs_two_stage_validation", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_one_stage_training_manifest_is_rerendered_to_matched_representation() -> None:
    manifest = pd.DataFrame(
        {
            "row_uid": ["train:1"],
            "text": ["alpha was good"],
            "candidate_aspect": ["Group: Alpha"],
            "candidate_sentiment": ["positive"],
            "candidate_text": ["old minimal text"],
            "representation_variant": ["minimal"],
            "target": [1],
        }
    )
    resource = {"minimal_aspects": {"Group: Alpha": "Alpha definition."}}
    result = MODULE._rerender_one_stage_training_manifest(manifest, resource)
    assert result.loc[0, "representation_variant"] == "name_and_description"
    assert result.loc[0, "candidate_text"] == (
        "Aspect: Group: Alpha. Definition: Alpha definition. "
        "Candidate sentiment: positive."
    )


def test_matched_grid_check_rejects_candidate_text_mismatch() -> None:
    frame = pd.DataFrame(
        {
            "row_uid": ["validation:1"],
            "candidate_aspect": ["Group: Alpha"],
            "candidate_sentiment": ["positive"],
            "candidate_text": ["matched"],
            "target": [1],
        }
    )
    MODULE._assert_matched_evaluation_grids(frame, frame.copy())
    changed = frame.copy()
    changed.loc[0, "candidate_text"] = "different"
    with pytest.raises(AssertionError, match="not matched"):
        MODULE._assert_matched_evaluation_grids(frame, changed)


def test_sentiment_cardinality_records_dataset_faithful_cap() -> None:
    frame = pd.DataFrame(
        {
            "row_uid": ["validation:1"] * 3,
            "candidate_aspect": ["Group: Alpha"] * 3,
        }
    )
    cardinality = MODULE._sentiment_cardinality(
        frame, pd.Series([True, True, False])
    )
    assert cardinality == {"minimum": 2, "maximum": 2}
