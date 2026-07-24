from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SCRIPT = PROJECT_ROOT / "scripts" / "summarise_taxonomy_tuning.py"
SPEC = importlib.util.spec_from_file_location("summarise_taxonomy_tuning", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_loader_accepts_only_validation_selection_summaries(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "event": "threshold_selected",
                "tuning_observation": {
                    "method_id": "m",
                    "candidate_id": "c",
                },
            }
        ),
        encoding="utf-8",
    )
    frame = MODULE.load_observations([path])
    assert frame.iloc[0]["candidate_id"] == "c"
    invalid = tmp_path / "test.json"
    invalid.write_text(
        json.dumps({"event": "test_analysis_complete"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Not a tuning"):
        MODULE.load_observations([invalid])


def test_selection_writer_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    MODULE.write_selection(path, {"selected": 1})
    assert json.loads(path.read_text(encoding="utf-8"))["selected"] == 1
    with pytest.raises(FileExistsError):
        MODULE.write_selection(path, {"selected": 2})


def test_fixed_selection_can_be_written_without_validation_summaries(
    tmp_path: Path,
) -> None:
    selection = MODULE.fixed_registered_selection("frozen_qwen_candidate_pair")
    output = tmp_path / "fixed.json"
    MODULE.write_selection(output, selection)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["selection_scope"] == "registered_fixed_recipe"
