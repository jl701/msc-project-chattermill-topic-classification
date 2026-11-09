"""CPU-only checks for paired review bootstrap and coverage boundaries."""

import json

import pytest

from serving_quality_intervals import analyse, canonical_sha256


def fixtures():
    labels = [
        {"request_id": "a", "pairs": [{"aspect": "held", "sentiment": "negative"}]},
        {"request_id": "b", "pairs": [{"aspect": "seen", "sentiment": "positive"}]},
        {"request_id": "c", "pairs": [{"aspect": "held", "sentiment": "neutral"}]},
    ]
    artifact = {"fold_id": "l2-a01", "taxonomy": {
        "canonical_order": ["held", "seen"], "heldout_aspects": ["held"]},
        "replay": {"splits": {"evaluation": {"count": 3,
                   "request_id_sha256": canonical_sha256(["a", "b", "c"])}}}}
    return labels, artifact


def row(sample, aspect="held", sentiment="negative", *, success=True, started=0,
        variant="hybrid", fold="l2-a01"):
    return {"sample_id": sample, "request_id": f"attempt-{sample}-{started}",
            "started_s": started, "success": success,
            "response": {"variant": variant, "fold_id": fold, "predictions": [
                {"aspect": aspect, "sentiments": [sentiment]}] if success else []}}


def test_identical_systems_have_exact_zero_paired_difference_and_deduplicate():
    labels, artifact = fixtures()
    rows = [row("a"), row("b", "seen", "positive", started=1),
            row("c", sentiment="neutral", started=2), row("a", started=3)]
    output = analyse({"left": rows, "right": rows}, labels, artifact, replicates=2000)
    assert output["coverage"]["complete_evaluation_coverage"] is True
    assert output["systems"]["left"]["successful_repeats_excluded"] == 1
    for view in output["paired_differences"][0]["views"].values():
        for scope in view["scopes"].values():
            for metric in ("presence", "pair"):
                assert scope[metric]["difference"] == 0
                assert scope[metric]["percentile_95_interval"] == [0, 0]
    # Rare heldout support can be absent in resamples even with perfect predictions.
    held = output["systems"]["left"]["views"]["successful_only"]["scopes"]["heldout"]
    assert held["pair"]["bootstrap_zero_denominator_replicates"] > 0


def test_failed_delivery_and_common_success_comparison_have_distinct_denominators():
    labels, artifact = fixtures()
    left = [row("a"), row("b", success=False, started=1), row("c", sentiment="neutral", started=2)]
    right = [row("a"), row("b", "seen", "positive", started=1), row("c", sentiment="neutral", started=2)]
    output = analyse({"left": left, "right": right}, labels, artifact, replicates=100)
    views = output["systems"]["left"]["views"]
    assert views["failure_as_empty"]["unique_reviews"] == 3
    assert views["successful_only"]["unique_reviews"] == 2
    assert views["failure_as_empty"]["scopes"]["overall"]["pair"]["micro_f1"] == pytest.approx(.8)
    paired = output["paired_differences"][0]["views"]
    assert paired["successful_only_common_subset"]["unique_reviews"] == 2
    assert paired["failure_as_empty"]["scopes"]["overall"]["pair"]["difference"] == pytest.approx(-.2)


def test_partial_coverage_is_reported_and_fixed_seed_is_reproducible():
    labels, artifact = fixtures()
    rows = [row("a"), row("b", "seen", "negative", started=1)]
    first = analyse({"left": rows}, labels, artifact, replicates=100)
    second = analyse({"left": rows}, labels, artifact, replicates=100)
    assert first == second
    assert first["coverage"]["unattempted_evaluation_reviews"] == 1
    assert first["coverage"]["complete_evaluation_coverage"] is False
    serialized = json.dumps(first)
    assert "sample_id" not in serialized and "attempt-a" not in serialized


def test_fold_and_sample_set_mismatches_fail_closed():
    labels, artifact = fixtures()
    with pytest.raises(ValueError, match="fold"):
        analyse({"wrong": [row("a", fold="l2-a02")]}, labels, artifact, replicates=10)
    with pytest.raises(ValueError, match="sets differ"):
        analyse({"left": [row("a")], "right": [row("b")]}, labels, artifact, replicates=10)
    with pytest.raises(ValueError, match="identities/order"):
        analyse({"left": [row("a")]}, list(reversed(labels)), artifact, replicates=10)


def test_presence_and_pair_remain_distinct_and_first_success_is_not_best_success():
    labels, artifact = fixtures()
    rows = [row("a", sentiment="positive"), row("a", sentiment="negative", started=1)]
    output = analyse({"system": rows}, labels, artifact, replicates=100)
    scope = output["systems"]["system"]["views"]["successful_only"]["scopes"]["heldout"]
    assert scope["presence"]["micro_f1"] == 1
    assert scope["pair"]["micro_f1"] == 0
    assert output["systems"]["system"]["repeat_prediction_disagreements"] == 1
