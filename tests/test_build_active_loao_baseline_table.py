from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_active_loao_baseline_table.py"
SPEC = importlib.util.spec_from_file_location("build_active_loao_baseline_table", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_inputs() -> tuple[dict, list[dict[str, str]]]:
    registry = json.loads(
        (
            ROOT / "configs" / "experiments" / "loao_active_baseline_registry_v1.json"
        ).read_text(encoding="utf-8")
    )
    with (
        ROOT
        / "docs"
        / "thesis_figure_data"
        / "loao_similarity_aspect_conditioned_sentiment_v1_active.csv"
    ).open("r", encoding="utf-8", newline="") as handle:
        similarity = list(csv.DictReader(handle))
    return registry, similarity


def test_locked_registry_validates() -> None:
    registry, similarity = load_inputs()
    rows = MODULE.validate_registry(registry, similarity)
    assert [row["method_id"] for row in rows] == MODULE.EXPECTED_METHODS
    assert rows[-1]["pair_micro_f1_mean"] == pytest.approx(0.4831582925751108)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sentiment_formulation", "Global document-level sentiment", "Retired sentiment"),
        ("method_id", "legacy_tfidf", "Active method order changed"),
        ("pair_micro_f1_mean", 0.4549, "Retired result value"),
    ],
)
def test_retired_evidence_is_rejected(field: str, value: object, message: str) -> None:
    registry, similarity = load_inputs()
    changed = copy.deepcopy(registry)
    changed["rows"][1][field] = value
    if field == "method_id":
        changed["retired_guards"]["forbidden_method_ids"].append("legacy_tfidf")
    with pytest.raises(ValueError, match=message):
        MODULE.validate_registry(changed, similarity)


def test_similarity_source_must_match_frozen_registry() -> None:
    registry, similarity = load_inputs()
    similarity[1]["pair_micro_f1_mean"] = "0.999"
    with pytest.raises(ValueError, match="disagrees with the frozen source export"):
        MODULE.validate_registry(registry, similarity)
