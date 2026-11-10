import numpy as np
import pytest
import json
from argparse import Namespace

from profile_rag_scores import BANDS, describe, distribution, length_band, run


def test_length_boundaries_and_nonfinite_values():
    assert [length_band(n) for n in (0, 100, 101, 300, 301, 1000, 1001)] == [BANDS[i] for i in (0, 0, 1, 1, 2, 2, 3)]
    assert distribution([])["median"] is None
    for bad in ([np.inf], [np.nan], [-1]):
        with pytest.raises(ValueError):
            distribution(bad)


def test_failed_review_stays_in_denominator_and_prompt_counts_are_distinct():
    trim = {"original_prompt_tokens": 1200, "final_prompt_tokens": 1024,
            "demo_characters_removed": [10, 0, 2, 0], "query_characters_removed": 0}
    records = {"a": {"status": "ok", "elapsed_ms": 10, "response": {"diagnostics": {
        "aspect_input_tokens": 1024, "batch_aspect": {"retrieval": [{"truncation": trim}]}}}}}
    prediction = {"a": {("x", "positive")}, "missing": set()}
    gold = {"a": {("x", "positive")}, "missing": {("x", "negative")}}
    result = describe(["a", "missing"], records, prediction, ["x"], "x", gold)
    assert result["registered_reviews"] == 2
    assert result["failed_or_missing_reviews"] == 1
    assert result["exact_set_accuracy_all_attempts"] == .5
    assert result["quality"]["scopes"]["overall"]["pair"]["fn"] == 1
    assert result["any_demo_trim_prompts"] == 1
    assert result["any_query_trim_prompts"] == 0
    assert result["demo_characters_removed_per_prompt"]["median"] == 12
    blind = describe(["a", "missing"], records, prediction, ["x"], "x")
    assert "quality" not in blind and "exact_set_accuracy_all_attempts" not in blind
    assert blind["predicted_presence_reviews_by_aspect"] == {"x": 1}


def test_incomplete_run_blocks_before_trying_to_open_gold(tmp_path):
    (tmp_path / "identity.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "state.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed inference"):
        run(Namespace(scores=tmp_path, historical_labels=tmp_path / "does_not_exist"))
