import copy
import json

import numpy as np
import pytest

from msc_project.serving.rag_evaluation import (
    bootstrap_weights, count_tensor, decode_records, gold_by_id, paired_contrast,
    records_by_id, score_maps, select_threshold, summarize_counts, threshold_candidates,
)
from msc_project.serving.review_routing import (
    assess_assignment, assign_review, call_manifest, clopper_pearson_lower,
    comparator_call_accounting, comparator_call_ids, freeze_boundary, matched_random_assignment,
    risk_features, select_policy,
)

ASPECTS = ["seen", "heldout"]


def record(key, seen=.9, heldout=.1, *, sentiment=None, success=True):
    sentiment = sentiment or {"negative": .8, "neutral": .1, "positive": .1}
    return {"request_id": key, "status": "ok" if success else "error", "elapsed_ms": 2,
            "response": {"request_id": key,
                "aspect_scores": [{"aspect": "seen", "score": seen},
                                  {"aspect": "heldout", "score": heldout}],
                "full_grid_scores": [{"aspect": a, "sentiment_scores": sentiment}
                                     for a in ASPECTS], "diagnostics": {}}}


def test_threshold_candidates_resolve_near_one_and_keep_old():
    values = [.9999991, .9999997, .9999999]
    candidates = threshold_candidates(values, .99999922514)
    assert len(candidates) <= 102
    assert .99999922514 in candidates
    assert any(.9999991 < value < .9999997 for value in candidates)
    with pytest.raises(ValueError):
        threshold_candidates([float("nan")], .5)


def test_heldout_scores_and_labels_cannot_change_selection():
    rows = records_by_id([record("a", .95), record("b", .7), record("c", .2)])
    gold = {"a": {("seen", "negative")}, "b": set(), "c": set()}
    first = select_threshold(rows, gold, ASPECTS, "heldout", .5, .5)
    changed = copy.deepcopy(rows)
    for row in changed.values():
        row["response"]["aspect_scores"][1]["score"] = .99
    changed_gold = {key: pairs | {("heldout", "positive")} for key, pairs in gold.items()}
    second = select_threshold(changed, changed_gold, ASPECTS, "heldout", .5, .5)
    assert first == second
    assert first["seen_pair_f1"] == 1
    assert first["aspect_threshold"] > .7


def test_rethreshold_uses_full_grid_and_runner_up():
    rec = record("a", .4, .9, sentiment={"negative": .5, "neutral": .4, "positive": .1})
    assert decode_records({"a": rec}, ["a"], ASPECTS, .3, .35)["a"] == {
        ("seen", "negative"), ("seen", "neutral"),
        ("heldout", "negative"), ("heldout", "neutral")}
    rec["response"]["full_grid_scores"].pop()
    with pytest.raises(ValueError, match="complete"):
        score_maps(rec, ASPECTS)


def test_population_includes_heldout_negatives_and_failures():
    gold = {"a": {("heldout", "negative")}, "b": set(), "c": {("seen", "negative")}}
    rows = records_by_id([record("a", .1, .9), record("b", .1, .9), record("c", success=False)])
    predictions = decode_records(rows, sorted(gold), ASPECTS, .5, .5)
    tensor = count_tensor(sorted(gold), predictions, gold, ASPECTS, "heldout")
    summary = summarize_counts(tensor)
    assert summary["scopes"]["heldout"]["presence"]["fp"] == 1
    assert summary["scopes"]["seen"]["presence"]["fn"] == 1
    assert summary["review_count"] == 3


def test_paired_bootstrap_identical_and_zero_denominators():
    gold = {"a": {("seen", "negative")}, "b": set()}
    tensor = count_tensor(sorted(gold), gold, gold, ASPECTS, "heldout")
    weights = bootstrap_weights(2, 50, 13)
    contrast = paired_contrast(tensor, tensor, weights)
    assert contrast["scopes"]["overall"]["pair_f1_difference_points_interval"] == [0, 0]
    assert contrast["scopes"]["heldout"]["pair_f1_relative_gain_percent"] is None
    assert contrast["scopes"]["overall"]["presence_fp_relative_reduction_percent"] is None
    json.dumps(contrast, allow_nan=False)


def test_identity_and_label_validation():
    with pytest.raises(ValueError, match="duplicate"):
        records_by_id([record("a"), record("a")])
    with pytest.raises(ValueError, match="disagree"):
        gold_by_id([{"request_id": "a", "pairs": [], "aspects": ["seen"]}], ASPECTS)
    mismatch = record("a")
    mismatch["response"]["request_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        records_by_id([mismatch])


def test_router_is_label_blind_and_audits_are_manual():
    rows = records_by_id([record(str(i)) for i in range(200)])
    # A human annotation flag deliberately injected into records must be ignored.
    for row in rows.values():
        row["gold_out_of_scope"] = True
    features = risk_features(rows, sorted(rows), ASPECTS, .5, .5)
    assignment = assign_review(features, {"min_margin": 0.0})
    assert len(assignment["random_audit"]) == 10
    assert len(assignment["automatic"]) == 190
    predictions = decode_records(rows, sorted(rows), ASPECTS, .5, .5)
    summary = assess_assignment(assignment, predictions, predictions, rows, ASPECTS, "heldout")
    assert summary["quality_gate_passed"]
    assert summary["qualified_review_item_reduction_percent"] == 95
    assert summary["automatic_count_after_audit"] == 190


def test_confident_missing_theme_still_fails_quality_gate():
    rows = records_by_id([record(str(i)) for i in range(200)])
    features = risk_features(rows, sorted(rows), ASPECTS, .5, .5)
    assignment = assign_review(features, {"min_margin": 0.0})
    predictions = decode_records(rows, sorted(rows), ASPECTS, .5, .5)
    gold = {key: pairs | {("heldout", "negative")} for key, pairs in predictions.items()}
    summary = assess_assignment(assignment, predictions, gold, rows, ASPECTS, "heldout")
    assert summary["automatic_exact_set_correct_count"] == 0
    assert not summary["quality_gate_passed"]
    assert summary["qualified_review_item_reduction_percent"] == 0


def test_failures_empty_and_severe_truncation_force_review():
    rows = records_by_id([record("failure", success=False), record("empty", .1, .1), record("long")])
    rows["long"]["response"]["diagnostics"]["severe_truncation"] = True
    features = risk_features(rows, sorted(rows), ASPECTS, .5, .5)
    assignment = assign_review(features, {"min_margin": 0})
    assert assignment["manual"] == set(rows)
    assert not assignment["automatic"]


def test_exact_bound_and_calibration_vs_final_minimum():
    assert clopper_pearson_lower(0, 0) == 0
    assert clopper_pearson_lower(100, 100) == pytest.approx(.05 ** .01)
    rows = records_by_id([record(str(i)) for i in range(100)])
    predictions = decode_records(rows, sorted(rows), ASPECTS, .5, .5)
    features = risk_features(rows, sorted(rows), ASPECTS, .5, .5)
    policy = select_policy(features, predictions, predictions, rows, ASPECTS, "heldout")
    assert "fallback" not in policy
    assignment = assign_review(features, policy)
    final = assess_assignment(assignment, predictions, predictions, rows, ASPECTS, "heldout")
    assert not final["quality_gate_passed"]  # final population has <100 untouched items


def test_sentiment_truncation_routes_only_aspects_selected_after_threshold():
    row = record("a", .9, .1)
    row["response"]["sentiment_truncation_by_aspect"] = {
        "seen": {"left_truncated": False}, "heldout": {"left_truncated": True}}
    assert not risk_features({"a": row}, ["a"], ASPECTS, .5, .5)["a"]["mandatory"]
    feat = risk_features({"a": row}, ["a"], ASPECTS, .05, .5)["a"]
    assert feat["reasons"] == ["selected_sentiment_left_truncated"]


def test_review_all_fallback_and_random_budget_are_reproducible():
    rows = records_by_id([record(str(i)) for i in range(100)])
    predictions = decode_records(rows, sorted(rows), ASPECTS, .5, .5)
    gold = {key: set() for key in rows}
    features = risk_features(rows, sorted(rows), ASPECTS, .5, .5)
    policy = select_policy(features, predictions, gold, rows, ASPECTS, "heldout")
    assert policy["fallback"] == "review_all_no_calibration_policy_passed"
    assert not assign_review(features, policy)["automatic"]
    random = matched_random_assignment(features, 25)
    assert len(random["manual"]) == 25
    assert random == matched_random_assignment(features, 25)


def test_disagreement_missing_comparator_requires_review():
    rows = {"a": record("a"), "b": record("b")}
    features = risk_features(rows, sorted(rows), ASPECTS, .5, .5, {"a": set()})
    policy = {"min_margin": 0, "use_disagreement": True, "boundary": freeze_boundary(features, 0)}
    assignment = assign_review(features, policy)
    assert len(assignment["comparator_calls"]) == 1
    assert assignment["targeted_review"] == assignment["comparator_calls"]


def fake_features(n=100):
    return {str(i): {"margin": float(i), "mandatory": False,
                     "comparator_available": True, "disagreement": False} for i in range(n)}


def test_boundary_is_frozen_per_threshold_and_ignores_nonboundary_disagreement():
    features = fake_features()
    policy = {"min_margin": 10, "use_disagreement": True,
              "boundary": freeze_boundary(features, 10)}
    calls = comparator_call_ids(features, policy)
    assert calls == {str(i) for i in range(10, 28)}
    assert policy["boundary"]["calibration_eligible_reviews"] == 90
    before = assign_review(features, policy)
    for key in set(features) - calls:
        features[key]["disagreement"] = True
        features[key]["comparator_available"] = False
    assert assign_review(features, policy) == before
    features["12"]["comparator_available"] = False
    after = assign_review(features, policy)
    assert after["comparator_unavailable"] == {"12"}
    assert "12" in after["targeted_review"]
    # Evaluation shifts can change call coverage. The frozen boundary is NOT
    # refit to keep an artificial20% evaluation quota.
    shifted = {key: dict(value, margin=value["margin"] + 100) for key, value in features.items()}
    assert not comparator_call_ids(shifted, policy)


def test_boundary_ties_use_stable_hash_and_do_not_call_mandatory_reviews():
    features = {key: dict(value, margin=1.) for key, value in fake_features().items()}
    features["0"]["mandatory"] = True
    band = freeze_boundary(features, 0, seed=17)
    policy = {"min_margin": 0, "use_disagreement": True, "boundary": band}
    calls = comparator_call_ids(features, policy)
    assert len(calls) == 20 and "0" not in calls
    assert freeze_boundary(dict(reversed(list(features.items()))), 0, seed=17) == band
    assert comparator_call_ids(features, {**policy, "min_margin": 2}) == set()


def test_boundary_cost_counts_missing_failures_and_keeps_runtime_scopes_distinct():
    features = fake_features(20)
    features["1"]["comparator_available"] = False
    policy = {"min_margin": 0, "use_disagreement": True, "boundary": freeze_boundary(features, 0)}
    assignment = assign_review(features, policy)
    success = record("0")
    success["response"]["diagnostics"]["sentiment_prompt_count"] = 2
    success["latency_ms"] = 100
    failure = {"success": False, "latency_ms": 200}
    cost = comparator_call_accounting(assignment, {"0": success, "1": failure}, 12)
    assert cost["additional_comparator_review_requests"] == 4
    assert cost["additional_requested_presence_scores"] == 48
    assert cost["stored_comparator_failures"] == 1 and cost["missing_comparator_records"] == 2
    assert cost["observed_conditional_sentiment_prompts"] == 2
    assert cost["runtime_proxies"]["historical_HTTP_latency_ms"]["sum_ms"] == 300
    assert cost["runtime_proxies"]["offline_scorer_elapsed_ms"]["sum_ms"] == 2
    assert not cost["combined_service_latency_measured"]
    manifest = call_manifest(features, assignment)
    assert sum(row["comparator_call_required"] for row in manifest) == 4
    assert all(row["disagreement_used"] is None for row in manifest if not row["comparator_call_required"])


def test_disagreement_review_all_fallback_makes_zero_extra_calls():
    rows = records_by_id([record(str(i)) for i in range(100)])
    predictions = decode_records(rows, sorted(rows), ASPECTS, .5, .5)
    features = risk_features(rows, sorted(rows), ASPECTS, .5, .5, predictions)
    policy = select_policy(features, predictions, {key: set() for key in rows}, rows,
                           ASPECTS, "heldout", use_disagreement=True)
    assert policy["fallback"] == "review_all_no_calibration_policy_passed"
    assignment = assign_review(features, policy)
    assert not assignment["comparator_calls"]
    assert comparator_call_accounting(assignment, rows, 12)["additional_comparator_review_requests"] == 0


def test_disagreement_search_can_qualify_when_confidence_only_falls_back_to_review_all():
    rows = records_by_id([record(str(i)) for i in range(100)])
    predictions = decode_records(rows, sorted(rows), ASPECTS, .5, .5)
    original_features = risk_features(rows, sorted(rows), ASPECTS, .5, .5)
    band = freeze_boundary(original_features, 0)
    boundary = comparator_call_ids(original_features, {"min_margin": 0,
        "use_disagreement": True, "boundary": band})
    error_ids = set(sorted(boundary)[:10])
    gold = {key: set() if key in error_ids else pairs for key, pairs in predictions.items()}
    features = risk_features(rows, sorted(rows), ASPECTS, .5, .5, gold)
    confidence = select_policy(features, predictions, gold, rows, ASPECTS, "heldout")
    assert confidence["fallback"] == "review_all_no_calibration_policy_passed"
    disagreement = select_policy(features, predictions, gold, rows, ASPECTS, "heldout", use_disagreement=True)
    assert "fallback" not in disagreement
    assert disagreement["calibration_summary"]["calibration_eligible"]
    assert not disagreement["calibration_summary"]["quality_gate_passed"]  # final minimum remains100
    assert disagreement["calibration_summary"]["automatic_count_after_audit"] == 80
    assert disagreement["calibration_summary"]["additional_comparator_calls"] == 20
