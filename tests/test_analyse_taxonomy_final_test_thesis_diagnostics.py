from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyse_taxonomy_final_test_thesis_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("thesis_diagnostics", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_pair_preserves_colon_in_aspect() -> None:
    assert MODULE.parse_pair("Staff support: Email | negative") == (
        "Staff support: Email",
        "negative",
    )


def test_sentiments_for_aspect_filters_other_candidates() -> None:
    pairs = [
        "Staff support: Email | negative",
        "Staff support: Email | neutral",
        "Staff support: Phone | positive",
    ]
    assert MODULE.sentiments_for_aspect(pairs, "Staff support: Email") == {
        "negative",
        "neutral",
    }


def test_row_categories_are_mutually_exclusive() -> None:
    assert MODULE.row_category(set(), set()) == "true_absence"
    assert MODULE.row_category({"positive"}, set()) == "missed_aspect"
    assert MODULE.row_category(set(), {"neutral"}) == "spurious_aspect"
    assert (
        MODULE.row_category({"negative"}, {"negative"})
        == "correct_sentiment_set"
    )
    assert (
        MODULE.row_category({"negative"}, {"positive"})
        == "wrong_sentiment_set"
    )
