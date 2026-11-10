"""Label-blind risk routing and labelled *offline* manual-workload simulation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta

from msc_project.serving.rag_evaluation import (
    count_tensor, decode_records, f1_from_counts, score_maps, successful,
)

DEFAULT_ROUTING_CONFIG = Path(__file__).resolve().parents[3] / "configs/experiments/review_routing_20261111.json"


def load_routing_config(path: Path = DEFAULT_ROUTING_CONFIG) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_routing_config(value)
    return value


def validate_routing_config(value: Mapping) -> None:
    """Reject unsupported rule changes rather than silently ignoring config fields."""
    expected = {
        "primary_policy": {"method_scope": "frozen_selected_classifier_method_only",
            "allowed_variants": ["confidence", "confidence_disagreement"],
            "objective": ["calibration_eligible_first", "maximum_automatic_count_after_audit",
                          "minimum_additional_comparator_calls", "higher_minimum_margin",
                          "prefer_confidence_on_exact_tie"],
            "all_fallback": "selected_method_confidence_review_all",
            "other_policies": "secondary_comparisons_not_evaluation_time_alternatives",
            "freeze_before_evaluation_labels": True},
        "calibration": {"minimum_auto_after_audit": 30,
            "minimum_empirical_exact_set_accuracy": .95,
            "minimum_oracle_corrected_pair_f1": .95,
            "clopper_pearson_use": "descriptive_only_not_selection_gate",
            "objective": ["maximum_automatic_count_after_audit", "minimum_additional_comparator_calls",
                          "higher_minimum_margin"]},
        "evaluation": {"minimum_auto_after_audit": 100,
            "minimum_one_sided_95_clopper_pearson_lower": .95,
            "minimum_oracle_corrected_pair_f1": .95, "failed_gate_qualified_reduction_percent": 0},
        "sensitivity": {"minimum_auto_after_audit": 1,
            "minimum_one_sided_95_clopper_pearson_lower": .95,
            "minimum_oracle_corrected_pair_f1": .95,
            "use": "development_descriptive_only_never_select_using_evaluation"},
        "unchanged_rules": {"audit_fraction": .05, "audit_minimum": 10,
            "seed": 20261111, "boundary_fraction": .20,
            "margin_grid": "0_and_development_observable_margin_0_to_100_percentiles_deduplicated",
            "boundary_rule": "per_threshold_nonmandatory_survivors_lowest_margin_then_stable_ID_hash_frozen_on_development",
            "fallback": "review_all_if_no_calibration_candidate_is_eligible"}}
    if value.get("schema_version") != "review_routing_selection_v1" or any(
            value.get(section, {}).get(key) != expected_value
            for section, entries in expected.items() for key, expected_value in entries.items()):
        raise ValueError("Unsupported routing protocol; preserve registered calibration and final gates")


def clopper_pearson_lower(correct: int, n: int, confidence: float = .95) -> float:
    if not 0 <= correct <= n or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    return float(beta.ppf(1 - confidence, correct, n - correct + 1)) if correct else 0.0


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-9, 1 - 1e-9))
    return math.log(value / (1 - value))


def risk_features(records: Mapping[str, Mapping], keys: Sequence[str], aspects: Sequence[str],
                  aspect_threshold: float, runner_threshold: float,
                  comparator_predictions: Mapping[str, set] | None = None, *,
                  presence_score_space: str = "probability") -> dict[str, dict]:
    """Uses input/output diagnostics, NEVER labels or annotation flags."""
    predictions = decode_records(records, keys, aspects, aspect_threshold, runner_threshold,
                                 presence_score_space=presence_score_space)
    result = {}
    for key in keys:
        record = records.get(key)
        reasons = []
        margin = float("-inf")
        if not successful(record):
            reasons.append("inference_failure_or_missing")
        else:
            presence, sentiment = score_maps(record, aspects, presence_score_space=presence_score_space)
            margins = [abs(p - aspect_threshold) if presence_score_space == "logit_margin" else
                       abs(_logit(p) - _logit(aspect_threshold)) for p in presence.values()]
            selected = {a for a, _ in predictions[key]}
            if not selected:
                reasons.append("empty_aspect_output")
            for aspect in selected:
                values = sorted(sentiment[aspect], reverse=True)
                margins.extend([abs(_logit(values[1]) - _logit(runner_threshold)),
                                max(0.0, _logit(values[0]) - _logit(values[1]))])
            margin = min(margins)
            diagnostics = record["response"].get("diagnostics", {})
            # Explicit runtime flags only; absent flags are not inferred from gold.
            if diagnostics.get("severe_truncation") is True:
                reasons.append("severe_truncation")
            sentiment_truncation = record["response"].get("sentiment_truncation_by_aspect", {})
            if any(sentiment_truncation.get(aspect, {}).get("left_truncated") is True
                   for aspect in selected):
                reasons.append("selected_sentiment_left_truncated")
        comparator_available = comparator_predictions is not None and key in comparator_predictions
        result[key] = {"margin": margin, "mandatory": bool(reasons), "reasons": reasons,
                       "presence_score_space": presence_score_space,
                       "disagreement": comparator_available and predictions[key] !=
                       comparator_predictions[key], "comparator_available": comparator_available}
    return result


def boundary_key(key: str, feature: Mapping, seed: int) -> tuple[float, str]:
    return (float(feature["margin"]), hashlib.sha256(
        f"review-routing-boundary:{seed}:{key}".encode("utf-8")).hexdigest())


def freeze_boundary(features: Mapping[str, Mapping], min_margin: float,
                    *, seed: int = 20261111) -> dict:
    """A label-blind lowest20% band, frozen separately for each manual threshold."""
    eligible = sorted(boundary_key(key, feature, seed) for key, feature in features.items()
                      if not feature["mandatory"] and math.isfinite(feature["margin"]) and
                      feature["margin"] >= min_margin)
    count = math.ceil(.20 * len(eligible))
    cutoff = eligible[count - 1] if count else (None, None)
    return {"fraction": .20, "calibration_eligible_reviews": len(eligible),
            "calibration_requested_reviews": count, "upper_margin": cutoff[0],
            "upper_tiebreak_hash": cutoff[1], "tie_break_seed": seed,
            "definition": "lowest20percent_of_nonmandatory_confidence_survivors_frozen_on_development"}


def comparator_call_ids(features: Mapping[str, Mapping], policy: Mapping) -> set[str]:
    if not policy.get("use_disagreement", False):
        return set()
    band = policy["boundary"]
    if band["upper_margin"] is None:
        return set()
    cutoff = (float(band["upper_margin"]), band["upper_tiebreak_hash"])
    return {key for key, feature in features.items() if not feature["mandatory"] and
            math.isfinite(feature["margin"]) and feature["margin"] >= policy["min_margin"] and
            boundary_key(key, feature, band["tie_break_seed"]) <= cutoff}


def assign_review(features: Mapping[str, Mapping], policy: Mapping,
                  *, seed: int = 20261111) -> dict[str, set]:
    if "presence_score_space" in policy and any(
            value.get("presence_score_space", "probability") != policy["presence_score_space"]
            for value in features.values()):
        raise ValueError("Review policy and risk features have different presence score spaces")
    keys = sorted(features)
    comparator_calls = comparator_call_ids(features, policy)
    unavailable = {key for key in comparator_calls if not features[key]["comparator_available"]}
    disagreement = {key for key in comparator_calls if features[key]["comparator_available"] and
                    features[key]["disagreement"]}
    manual = {key for key, value in features.items() if value["mandatory"] or
              value["margin"] < float(policy["min_margin"])} | unavailable | disagreement
    candidates = sorted(set(keys) - manual)
    count = min(len(candidates), max(10, math.ceil(.05 * len(candidates)))) if candidates else 0
    audit = set(np.random.default_rng(seed).choice(candidates, size=count, replace=False).tolist())
    return {"targeted_review": manual, "random_audit": audit,
            "manual": manual | audit, "automatic": set(candidates) - audit,
            "comparator_calls": comparator_calls, "comparator_unavailable": unavailable,
            "comparator_disagreement": disagreement}


def comparator_call_accounting(assignment: Mapping[str, set], comparator_records: Mapping,
                              candidate_count: int) -> dict:
    """Logical routed calls and separately identified stored-runtime cost proxies."""
    calls = assignment.get("comparator_calls", set())
    n = len(assignment["manual"] | assignment["automatic"])
    known = [comparator_records[key] for key in sorted(calls) if key in comparator_records]
    valid = [record for record in known if successful(record)]
    timings = {}
    for field, name in (("latency_ms", "historical_HTTP_latency_ms"),
                        ("elapsed_ms", "offline_scorer_elapsed_ms")):
        values = [float(record[field]) for record in known if record.get(field) is not None]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Comparator cost proxy contains invalid elapsed time")
        timings[name] = {"covered_calls": len(values), "uncovered_calls": len(calls) - len(values),
                         "sum_ms": sum(values), "mean_ms": float(np.mean(values)) if values else None}
    sentiment_counts = [record["response"].get("diagnostics", {}).get("sentiment_prompt_count")
                        for record in valid]
    observed_sentiment = [value for value in sentiment_counts if value is not None]
    if any(not isinstance(value, int) or value < 0 for value in observed_sentiment):
        raise ValueError("Invalid observed comparator sentiment prompt count")
    return {"primary_review_requests": n, "additional_comparator_review_requests": len(calls),
        "additional_review_call_rate_percent": 100 * len(calls) / n if n else 0.0,
        "logical_total_review_requests": n + len(calls),
        "additional_requested_presence_scores": len(calls) * candidate_count,
        "stored_comparator_successes": len(valid),
        "stored_comparator_failures": len(known) - len(valid),
        "missing_comparator_records": len(calls) - len(known),
        "observed_conditional_sentiment_prompts": sum(observed_sentiment),
        "sentiment_count_coverage_calls": len(observed_sentiment),
        "runtime_proxies": timings,
        "request_count_scope": "logical_boundary_calls_in_offline_routing_simulation",
        "runtime_scope": "stored_reference_cost_proxies_not_measured_combined_service_latency",
        "combined_service_latency_measured": False}


def call_manifest(features: Mapping[str, Mapping], assignment: Mapping[str, set]) -> list[dict]:
    """Private request-ID manifest; no review text or gold labels."""
    calls = assignment.get("comparator_calls", set())
    return [{"request_id": key, "comparator_call_required": key in calls,
             "comparator_available": features[key]["comparator_available"] if key in calls else None,
             "disagreement_used": features[key]["disagreement"] if key in calls and
                                  features[key]["comparator_available"] else None,
             "targeted_review": key in assignment["targeted_review"],
             "random_audit": key in assignment["random_audit"],
             "automatic": key in assignment["automatic"]} for key in sorted(features)]


def assess_assignment(assignment: Mapping[str, set], predictions: Mapping[str, set],
                      gold: Mapping[str, set], records: Mapping[str, Mapping],
                      aspects: Sequence[str], heldout: str, *, minimum_auto: int = 100) -> dict:
    keys = sorted(gold)
    automatic, manual = assignment["automatic"], assignment["manual"]
    if automatic & manual or automatic | manual != set(keys):
        raise ValueError("Routing must partition the full labelled population")
    correct = sum(successful(records.get(key)) and predictions[key] == gold[key]
                  for key in automatic)
    n = len(automatic)
    lower = clopper_pearson_lower(correct, n)
    # Oracle manual correction is a retrospective simulation, not measured human accuracy.
    corrected = {key: gold[key] if key in manual else predictions[key] for key in keys}
    counts = count_tensor(keys, corrected, gold, aspects, heldout).sum(axis=0)
    pair_f1 = float(f1_from_counts(counts[0, 1]))
    eligible = n >= minimum_auto and lower >= .95 and pair_f1 >= .95
    reduction = 100 * n / len(keys)
    return {"review_count": len(keys), "targeted_review_count": len(assignment["targeted_review"]),
            "random_audit_count": len(assignment["random_audit"]), "manual_count": len(manual),
            "automatic_count_after_audit": n, "automatic_exact_set_correct_count": int(correct),
            "automatic_exact_set_accuracy": correct / n if n else None,
            "automatic_one_sided_95_clopper_pearson_lower": lower,
            "oracle_corrected_pair_f1": pair_f1,
            "observed_review_item_reduction_percent": reduction,
            "quality_gate_passed": eligible,
            "qualified_review_item_reduction_percent": reduction if eligible else 0.0,
            "minimum_auto_required": minimum_auto,
            "additional_comparator_calls": len(assignment.get("comparator_calls", set())),
            "additional_comparator_call_rate_percent": 100 * len(assignment.get("comparator_calls", set())) / len(keys),
            "comparator_unavailable_calls": len(assignment.get("comparator_unavailable", set())),
            "comparator_disagreement_calls": len(assignment.get("comparator_disagreement", set())),
            "manual_correction": "offline_gold_label_substitution_not_actual_human_time",
            "interval_population": "final_untouched_automatic_after_random_audit"}


def select_policy(features: Mapping[str, Mapping], predictions: Mapping[str, set],
                  gold: Mapping[str, set], records: Mapping[str, Mapping], aspects: Sequence[str],
                  heldout: str, *, use_disagreement: bool = False, seed: int = 20261111,
                  routing_config: Mapping | None = None) -> dict:
    config = load_routing_config() if routing_config is None else routing_config
    validate_routing_config(config)
    if seed != config["unchanged_rules"]["seed"]:
        raise ValueError("Routing seed differs from registered audit and boundary seed")
    calibration = config["calibration"]
    values = [f["margin"] for f in features.values() if math.isfinite(f["margin"])]
    candidates = sorted({0.0, *map(float, np.quantile(values, np.arange(0, 101) / 100))}) \
        if values else [0.0]
    trials, legacy_trials, curve = [], [], []
    for threshold in candidates:
        policy = {"min_margin": threshold, "use_disagreement": use_disagreement}
        if use_disagreement:
            policy["boundary"] = freeze_boundary(features, threshold, seed=seed)
        assignment = assign_review(features, policy, seed=seed)
        # These searched development summaries are descriptive. Final qualification
        # remains a separate unchanged test on frozen-policy evaluation outcomes.
        result = assess_assignment(assignment, predictions, gold, records, aspects, heldout,
                                   minimum_auto=config["evaluation"]["minimum_auto_after_audit"])
        result["interval_population"] = "calibration_automatic_after_audit_same_data_selection_descriptive"
        result["calibration_eligible"] = (
            result["automatic_count_after_audit"] >= calibration["minimum_auto_after_audit"] and
            result["automatic_exact_set_accuracy"] >= calibration["minimum_empirical_exact_set_accuracy"] and
            result["oracle_corrected_pair_f1"] >= calibration["minimum_oracle_corrected_pair_f1"])
        result["prior_cp_calibration_eligible"] = (
            result["automatic_count_after_audit"] >= config["sensitivity"]["minimum_auto_after_audit"] and
            result["automatic_one_sided_95_clopper_pearson_lower"] >=
            config["sensitivity"]["minimum_one_sided_95_clopper_pearson_lower"] and
            result["oracle_corrected_pair_f1"] >= config["sensitivity"]["minimum_oracle_corrected_pair_f1"])
        curve.append({"min_margin": threshold, **result})
        trial = (result["automatic_count_after_audit"],
                 -result["additional_comparator_calls"], threshold, policy, result)
        if result["calibration_eligible"]:
            trials.append(trial)
        if result["prior_cp_calibration_eligible"]:
            legacy_trials.append(trial)
    if trials:
        _, _, _, policy, summary = max(trials, key=lambda item: (item[0], item[1], item[2]))
        policy = dict(policy, selected_on="calibration_only", audit_seed=seed,
                      calibration_candidate_count=len(candidates), calibration_summary=summary,
                      calibration_curve=curve)
    else:
        # Finite sentinel enables strict JSON; larger than any clipped logit margin.
        policy = {"min_margin": 1e9, "use_disagreement": use_disagreement,
                  "selected_on": "calibration_only", "audit_seed": seed,
                  "calibration_candidate_count": len(candidates),
                  "calibration_curve": curve,
                  "fallback": "review_all_no_calibration_policy_passed"}
        if use_disagreement:
            policy["boundary"] = freeze_boundary(features, policy["min_margin"], seed=seed)
    legacy = {"scope": "development_descriptive_only_not_an_evaluation_candidate",
              "eligible_candidate_count": len(legacy_trials),
              "fallback": "review_all_no_calibration_policy_passed"}
    if legacy_trials:
        _, _, _, legacy_policy, legacy_summary = max(legacy_trials, key=lambda item: (item[0], item[1], item[2]))
        legacy = {"scope": legacy["scope"], "eligible_candidate_count": len(legacy_trials),
                  "policy": legacy_policy, "calibration_summary": legacy_summary}
    policy["calibration_criterion"] = dict(calibration)
    policy["prior_cp_calibration_sensitivity"] = legacy
    return policy


def choose_primary_policy(policies: Mapping[str, Mapping], selected_method: str,
                          *, routing_config: Mapping | None = None) -> dict:
    """Freeze one primary from calibration summaries, without evaluation inputs."""
    config = load_routing_config() if routing_config is None else routing_config
    validate_routing_config(config)
    confidence_name = f"{selected_method}__confidence"
    if confidence_name not in policies:
        raise ValueError("Selected classifier requires its confidence routing policy")
    candidates = {}
    for variant in config["primary_policy"]["allowed_variants"]:
        name = f"{selected_method}__{variant}"
        if name not in policies:
            continue
        entry = policies[name]
        policy = entry["policy"]
        if entry["method"] != selected_method or policy["use_disagreement"] != (variant == "confidence_disagreement"):
            raise ValueError("Primary routing policy identity differs from the selected classifier")
        summary = policy.get("calibration_summary", {})
        fallback = bool(policy.get("fallback"))
        eligible = not fallback and bool(summary.get("calibration_eligible"))
        if not eligible and not fallback:
            raise ValueError("Routing policy is neither calibration eligible nor an all-review fallback")
        candidates[name] = {"calibration_eligible": eligible,
            "automatic_count_after_audit": summary["automatic_count_after_audit"] if eligible else 0,
            "additional_comparator_calls": summary["additional_comparator_calls"] if eligible else 0,
            "min_margin": policy["min_margin"], "confidence_tie_preference": name == confidence_name,
            "fallback": fallback}
    if not any(row["calibration_eligible"] for row in candidates.values()):
        selected = confidence_name
        reason = "all_selected_method_candidates_fallback_choose_confidence_review_all"
    else:
        selected = max(candidates, key=lambda name: (
            candidates[name]["calibration_eligible"], candidates[name]["automatic_count_after_audit"],
            -candidates[name]["additional_comparator_calls"], candidates[name]["min_margin"],
            candidates[name]["confidence_tie_preference"]))
        reason = "registered_calibration_order_with_confidence_final_tie_break"
    return {"primary_policy_name": selected, "selected_classifier_method": selected_method,
            "selected_on": "calibration_only_before_evaluation_labels",
            "reason": reason, "objective": config["primary_policy"]["objective"],
            "candidate_summaries": candidates,
            "secondary_policy_names": sorted(set(policies) - {selected}),
            "evaluation_reselection_allowed": False}


def matched_random_assignment(features: Mapping[str, Mapping], manual_budget: int,
                              *, seed: int = 20261111) -> dict[str, set]:
    mandatory = {key for key, value in features.items() if value["mandatory"]}
    if not len(mandatory) <= manual_budget <= len(features):
        raise ValueError("Random budget cannot omit mandatory review or exceed population")
    available = sorted(set(features) - mandatory)
    extra = set(np.random.default_rng(seed).choice(available, manual_budget - len(mandatory),
                                                  replace=False).tolist())
    manual = mandatory | extra
    return {"targeted_review": mandatory, "random_audit": extra, "manual": manual,
            "automatic": set(features) - manual}
