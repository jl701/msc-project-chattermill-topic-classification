"""Label-blind risk routing and labelled *offline* manual-workload simulation."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta

from msc_project.serving.rag_evaluation import (
    count_tensor, decode_records, f1_from_counts, score_maps, successful,
)


def clopper_pearson_lower(correct: int, n: int, confidence: float = .95) -> float:
    if not 0 <= correct <= n or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    return float(beta.ppf(1 - confidence, correct, n - correct + 1)) if correct else 0.0


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-9, 1 - 1e-9))
    return math.log(value / (1 - value))


def risk_features(records: Mapping[str, Mapping], keys: Sequence[str], aspects: Sequence[str],
                  aspect_threshold: float, runner_threshold: float,
                  comparator_predictions: Mapping[str, set] | None = None) -> dict[str, dict]:
    """Uses input/output diagnostics, NEVER labels or annotation flags."""
    predictions = decode_records(records, keys, aspects, aspect_threshold, runner_threshold)
    result = {}
    for key in keys:
        record = records.get(key)
        reasons = []
        margin = float("-inf")
        if not successful(record):
            reasons.append("inference_failure_or_missing")
        else:
            presence, sentiment = score_maps(record, aspects)
            margins = [abs(_logit(p) - _logit(aspect_threshold)) for p in presence.values()]
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
                  heldout: str, *, use_disagreement: bool = False, seed: int = 20261111) -> dict:
    values = [f["margin"] for f in features.values() if math.isfinite(f["margin"])]
    candidates = sorted({0.0, *map(float, np.quantile(values, np.arange(0, 101) / 100))}) \
        if values else [0.0]
    trials, curve = [], []
    for threshold in candidates:
        policy = {"min_margin": threshold, "use_disagreement": use_disagreement}
        if use_disagreement:
            policy["boundary"] = freeze_boundary(features, threshold, seed=seed)
        assignment = assign_review(features, policy, seed=seed)
        # The final evaluation requires 100 untouched cases. Calibration has only
        # 100 historical reviews, so its separate criterion uses the same exact
        # bound, without imposing the final sample-count requirement here.
        result = assess_assignment(assignment, predictions, gold, records, aspects, heldout,
                                   minimum_auto=1)
        curve.append({"min_margin": threshold, **result})
        if result["quality_gate_passed"]:
            trials.append((result["automatic_count_after_audit"],
                           -result["additional_comparator_calls"], threshold, policy, result))
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
    return policy


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
