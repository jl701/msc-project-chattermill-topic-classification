"""Synthetic selection-versus-qualification checks; never real review outcomes."""

from copy import deepcopy

import pytest

from msc_project.serving.review_routing import (
    assess_assignment, assign_review, choose_primary_policy, load_routing_config, select_policy,
    validate_routing_config,
)
from test_rag_evaluation import ASPECTS, record


def fixture(n):
    keys = [str(i) for i in range(n)]
    features = {key: {"margin": 1., "mandatory": False, "comparator_available": True,
                      "disagreement": False} for key in keys}
    records = {key: record(key) for key in keys}
    predictions = {key: {("seen", "negative")} for key in keys}
    return features, records, predictions, deepcopy(predictions)


def test_smaller_perfect_calibration_group_can_be_selected_without_claiming_final_qualification():
    features, records, predictions, gold = fixture(50)
    policy = select_policy(features, predictions, gold, records, ASPECTS, "heldout")
    result = policy["calibration_summary"]
    assert result["automatic_count_after_audit"] == 40
    assert result["calibration_eligible"]
    assert result["automatic_one_sided_95_clopper_pearson_lower"] < .95
    assert result["qualified_review_item_reduction_percent"] == 0
    assert policy["prior_cp_calibration_sensitivity"]["fallback"] == "review_all_no_calibration_policy_passed"
    assert "not_an_evaluation_candidate" in policy["prior_cp_calibration_sensitivity"]["scope"]


@pytest.mark.parametrize("n,eligible", [(39, False), (40, True)])
def test_calibration_minimum_counts_automatic_reviews_after_audit(n, eligible):
    features, records, predictions, gold = fixture(n)
    policy = select_policy(features, predictions, gold, records, ASPECTS, "heldout")
    assert ("fallback" not in policy) is eligible
    if eligible:
        assert policy["calibration_summary"]["automatic_count_after_audit"] == 30
    else:
        assert not assign_review(features, policy)["automatic"]


@pytest.mark.parametrize("errors,eligible", [(2, True), (3, False)])
def test_empirical_exact_set_threshold_is_applied_independently_of_development_interval(errors, eligible):
    features, records, predictions, gold = fixture(50)
    automatic = sorted(assign_review(features, {"min_margin": 0, "use_disagreement": False})["automatic"])
    for key in automatic[:errors]:
        gold[key] = set()
    policy = select_policy(features, predictions, gold, records, ASPECTS, "heldout")
    assert ("fallback" not in policy) is eligible
    if eligible:
        result = policy["calibration_summary"]
        assert result["automatic_exact_set_accuracy"] == .95
        assert result["automatic_one_sided_95_clopper_pearson_lower"] < .95


def test_calibration_oracle_f1_is_required_even_when_exact_set_accuracy_passes():
    features, records, predictions, gold = fixture(50)
    assignment = assign_review(features, {"min_margin": 0, "use_disagreement": False})
    for key in assignment["automatic"]:
        predictions[key] = gold[key] = set()
    for key in sorted(assignment["automatic"])[:2]:
        predictions[key] = {(aspect, sentiment) for aspect in ASPECTS
                            for sentiment in ("negative", "neutral", "positive")}
    policy = select_policy(features, predictions, gold, records, ASPECTS, "heldout")
    assert policy["fallback"] == "review_all_no_calibration_policy_passed"
    assert all(row["automatic_exact_set_accuracy"] == .95 for row in policy["calibration_curve"])
    assert all(row["oracle_corrected_pair_f1"] < .95 for row in policy["calibration_curve"])


@pytest.mark.parametrize("n,errors,qualified", [(99, 0, False), (100, 0, True), (100, 5, False)])
def test_final_quality_gate_is_unchanged(n, errors, qualified):
    _, records, predictions, gold = fixture(n)
    for key in list(gold)[:errors]:
        gold[key] = set()
    assignment = {"automatic": set(gold), "manual": set(), "targeted_review": set(), "random_audit": set()}
    result = assess_assignment(assignment, predictions, gold, records, ASPECTS, "heldout")
    assert result["quality_gate_passed"] is qualified
    assert result["minimum_auto_required"] == 100
    if not qualified:
        assert result["qualified_review_item_reduction_percent"] == 0


def test_config_cannot_silently_relax_final_requirement():
    config = load_routing_config()
    config["evaluation"]["minimum_one_sided_95_clopper_pearson_lower"] = .9
    with pytest.raises(ValueError, match="Unsupported routing protocol"):
        validate_routing_config(config)


def test_routing_cannot_silently_change_registered_audit_seed():
    features, records, predictions, gold = fixture(50)
    with pytest.raises(ValueError, match="Routing seed differs"):
        select_policy(features, predictions, gold, records, ASPECTS, "heldout", seed=17)


def candidate(method, disagreement=False, *, count=40, calls=0, margin=1, fallback=False):
    policy = {"use_disagreement": disagreement, "min_margin": margin}
    if fallback:
        policy["fallback"] = "review_all_no_calibration_policy_passed"
    else:
        policy["calibration_summary"] = {"calibration_eligible": True,
            "automatic_count_after_audit": count, "additional_comparator_calls": calls}
    return {"method": method, "policy": policy}


@pytest.mark.parametrize("confidence,disagreement,expected", [
    ({}, {}, "confidence"),
    ({"count": 40}, {"count": 41, "calls": 15}, "confidence_disagreement"),
    ({"calls": 0}, {"calls": 10, "margin": 2}, "confidence"),
    ({"margin": 1}, {"margin": 2}, "confidence_disagreement"),
    ({"fallback": True, "margin": 1e9}, {}, "confidence_disagreement"),
    ({}, {"fallback": True, "margin": 1e9}, "confidence"),
])
def test_primary_choice_uses_only_registered_calibration_order(confidence, disagreement, expected):
    policies = {"e5__confidence": candidate("e5", **confidence),
                "e5__confidence_disagreement": candidate("e5", True, **disagreement)}
    chosen = choose_primary_policy(policies, "e5")
    assert chosen["primary_policy_name"] == f"e5__{expected}"
    assert chosen["selected_on"] == "calibration_only_before_evaluation_labels"
    assert not chosen["evaluation_reselection_allowed"]
    assert len(chosen["secondary_policy_names"]) == 1


def test_primary_all_fallback_uses_confidence_even_when_disagreement_margin_is_higher():
    policies = {"e5__confidence": candidate("e5", fallback=True, margin=10),
                "e5__confidence_disagreement": candidate("e5", True, fallback=True, margin=20)}
    chosen = choose_primary_policy(policies, "e5")
    assert chosen["primary_policy_name"] == "e5__confidence"
    assert chosen["reason"] == "all_selected_method_candidates_fallback_choose_confidence_review_all"
    assert chosen["candidate_summaries"]["e5__confidence"]["automatic_count_after_audit"] == 0


def test_nonselected_classifier_cannot_be_primary_even_with_better_calibration():
    policies = {"fixed__confidence": candidate("fixed", fallback=True),
                "fixed__confidence_disagreement": candidate("fixed", True, fallback=True),
                "e5__confidence": candidate("e5", count=90)}
    chosen = choose_primary_policy(policies, "fixed")
    assert chosen["primary_policy_name"] == "fixed__confidence"
    assert set(chosen["candidate_summaries"]) == {"fixed__confidence", "fixed__confidence_disagreement"}
    assert "e5__confidence" in chosen["secondary_policy_names"]


def test_primary_requires_confidence_and_consistent_classifier_identity():
    with pytest.raises(ValueError, match="requires its confidence"):
        choose_primary_policy({"e5__confidence": candidate("e5")}, "fixed")
    with pytest.raises(ValueError, match="identity differs"):
        choose_primary_policy({"fixed__confidence": candidate("e5")}, "fixed")
