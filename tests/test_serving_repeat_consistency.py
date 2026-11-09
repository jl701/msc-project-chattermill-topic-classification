"""A repeated review is a consistency probe, never an extra quality sample."""
import runpy
from pathlib import Path

consistency_counts = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / "scripts/check_serving_repeat_consistency.py"
))["consistency_counts"]


def row(index, source, sentiments=None):
    return {
        "attempt_index": index, "request_id": f"attempt-{index}", "sample_id": source,
        "success": sentiments is not None,
        "response": {"predictions": [{"aspect": "topic", "sentiments": sentiments or []}]},
    }


def test_prediction_sets_ignore_sentiment_order_and_duplicates():
    result = consistency_counts([row(0, "a", ["positive", "negative"]),
                                 row(1, "a", ["negative", "positive", "positive"])])
    assert result["reviews_with_repeated_successful_deliveries"] == 1
    assert result["repeated_deliveries_differing_from_first"] == 0


def test_failed_first_attempt_is_not_a_reference_prediction():
    result = consistency_counts([row(0, "a"), row(1, "a", ["positive"]),
                                 row(2, "a", ["positive"]), row(3, "b")])
    assert result["attempts"] == 4 and result["failed_attempts"] == 2
    assert result["unique_attempted_reviews"] == 2
    assert result["unique_successful_reviews"] == 1
    assert result["repeated_successful_deliveries_after_first"] == 1
    assert result["reviews_with_differing_prediction_sets"] == 0


def test_completion_order_cannot_change_the_first_attempt_reference():
    result = consistency_counts([row(2, "a", ["negative"]), row(0, "a", ["positive"]),
                                 row(1, "a", ["positive"]), row(3, "a", ["negative"])])
    assert result["repeated_successful_deliveries_after_first"] == 3
    assert result["reviews_with_differing_prediction_sets"] == 1
    assert result["repeated_deliveries_differing_from_first"] == 2
