"""Calibrate on development only, then report historical/new-source RAG evidence.

Aggregate metrics, frozen thresholds and a private boundary-call ID manifest
are written. Review text, predictions and gold labels are never copied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.serving.rag_evaluation import (
    active_presence_threshold, bootstrap_weights, count_tensor, decode_records, gold_by_id, paired_contrast,
    read_jsonl, records_by_id, select_threshold, successful, summarize_counts, validate_record_spaces,
)
from msc_project.serving.review_routing import (
    assess_assignment, assign_review, call_manifest, comparator_call_accounting,
    matched_random_assignment, risk_features, select_policy,
)
from msc_project.serving.annotation_workflow import load_reviewed_split


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Different evidence already exists at {path}; use a new output directory")
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_paths(items: list[str], root: Path | None) -> dict[str, Path]:
    if root and (root / "identity.json").exists():
        # Only registered classifier outputs are methods. Telemetry and other
        # auxiliary JSONL journals can share the run directory.
        identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
        names = identity["methods"]
        if not names or len(names) != len(set(names)) or any(
                not isinstance(name, str) or not name or
                not all(char.isascii() and (char.isalnum() or char == "_") for char in name)
                for name in names):
            raise ValueError("Invalid registered method names")
        result = {name: root / f"{name}.jsonl" for name in names}
    else:
        result = {path.stem: path for path in sorted(root.glob("*.jsonl"))
                  if not path.name.startswith("_")} if root else {}
    for item in items:
        name, path = item.split("=", 1)
        if not name or name in result:
            raise ValueError("Duplicate or empty method name")
        result[name] = Path(path)
    return result


def comparator_records(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    result = {}
    rows = sorted(read_jsonl(path), key=lambda row: (row.get("started_s") is None,
                                                   row.get("started_s") or 0))
    for record in rows:
        # Historical HTTP benchmarks identify the original review by sample_id;
        # request_id can instead identify an operation or a retry.
        key = str(record.get("sample_id") or record.get("request_id") or
                  record.get("response", {}).get("request_id") or "")
        if not key:
            raise ValueError("Comparator record lacks review identity")
        if key not in result:
            # Preserve the first attempt even if it failed. Selecting a later
            # successful retry would hide failures and underestimate routed cost.
            result[key] = record
    return result


def predictions_from_comparator(records: dict | None) -> dict[str, set] | None:
    if records is None:
        return None
    return {key: {(item["aspect"], sentiment)
                  for item in record["response"].get("predictions", [])
                  for sentiment in item["sentiments"]}
            for key, record in records.items() if successful(record)}


def comparator_predictions(path: Path | None) -> dict[str, set] | None:
    return predictions_from_comparator(comparator_records(path))


def population_check(records: dict, gold: dict) -> None:
    if set(records) - set(gold):
        raise ValueError("Scores contain IDs outside the registered population")


def reviewed_population(args: argparse.Namespace, artifacts: dict, split: str,
                        labels_path: Path) -> dict:
    """Validate one split only; calibration must not open locked gold first."""
    if args.annotation_data is None or args.reviewed_release is None:
        raise ValueError("new_source_locked requires --annotation-data and --reviewed-release")
    verified = load_reviewed_split(args.annotation_data, args.reviewed_release, split)
    taxonomy = {row["aspect"]: row["definition"] for row in verified["taxonomy"]}
    candidates = artifacts["taxonomy"]["candidates"]
    if (len(taxonomy) != len(verified["taxonomy"]) or len(candidates) != len(taxonomy)
            or {row["aspect"]: row.get("text") for row in candidates} != taxonomy):
        raise ValueError("Reviewed taxonomy differs from the frozen model candidate definitions")
    expected_hash = verified["provenance"]["files"][f"labels_{split}.jsonl"]
    if sha256(labels_path) != expected_hash:
        raise ValueError(f"Provided {split} gold file differs from the reviewed release")
    return verified


def operational_summary(records: dict, keys: list[str], predictions: dict, gold: dict) -> dict:
    success = sum(successful(records.get(key)) for key in keys)
    elapsed = [float(records[key]["elapsed_ms"]) for key in keys if key in records and
               records[key].get("elapsed_ms") is not None]
    if elapsed and (not np.isfinite(elapsed).all() or min(elapsed) < 0):
        raise ValueError("Invalid scorer elapsed time")
    return {"successful_reviews": success, "failed_or_missing_reviews": len(keys) - success,
            "registered_reviews": len(keys),
            "explicit_failed_reviews": sum(key in records and not successful(records[key]) for key in keys),
            "missing_record_reviews": sum(key not in records for key in keys),
            "exact_set_accuracy_all_attempts": sum(successful(records.get(key)) and
                predictions[key] == gold[key] for key in keys) / len(keys),
            "scorer_elapsed_ms_median": float(np.median(elapsed)) if elapsed else None,
            "timing_scope": "offline_score_campaign_not_HTTP_service_latency"}


def run(args: argparse.Namespace) -> dict:
    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8"))
    aspects = [c["aspect"] for c in artifacts["taxonomy"]["candidates"]]
    heldouts = artifacts["taxonomy"]["heldout_aspects"]
    if len(heldouts) != 1:
        raise ValueError("This registered campaign requires one heldout aspect")
    heldout = heldouts[0]
    old = artifacts["variants"]["hybrid"]
    score_space = args.presence_score_space
    development = method_paths(args.development, args.development_root)
    evaluation = method_paths(args.evaluation, args.evaluation_root)
    if args.baseline not in development:
        raise ValueError("Fixed baseline is required for development selection")
    if not (args.selection_only or args.classification_only) and set(evaluation) != set(development):
        raise ValueError("Development and evaluation methods must match exactly")
    reviewed_calibration = reviewed_population(args, artifacts, "calibration", args.development_labels) \
        if args.scope == "new_source_locked" else None
    gold_dev = gold_by_id(reviewed_calibration["labels"] if reviewed_calibration else
                          read_jsonl(args.development_labels), aspects)
    keys_dev = sorted(gold_dev)
    records_dev = {name: records_by_id(read_jsonl(path)) for name, path in development.items()}
    selections, dev_predictions, dev_summaries = {}, {}, {}
    legacy_selections, legacy_dev_summaries = {}, {}
    for name, records in records_dev.items():
        population_check(records, gold_dev)
        validate_record_spaces(records, score_space)
        selections[name] = select_threshold(records, gold_dev, aspects, heldout,
            old["aspect_threshold"], old["runner_up_threshold"], presence_score_space=score_space)
        threshold = active_presence_threshold(selections[name])
        dev_predictions[name] = decode_records(records, keys_dev, aspects, threshold,
            old["runner_up_threshold"], presence_score_space=score_space)
        # Development summaries are restricted to seen outcomes; heldout outcomes
        # must not guide the next improvement round.
        seen_gold = {key: {p for p in pairs if p[0] != heldout} for key, pairs in gold_dev.items()}
        counts = count_tensor(keys_dev, dev_predictions[name], seen_gold, aspects, heldout)
        dev_summaries[name] = {"review_count": len(keys_dev),
            "seen": summarize_counts(counts)["scopes"]["seen"],
            "failed_or_missing_reviews": sum(not successful(records.get(k)) for k in keys_dev)}
        if score_space == "logit_margin":
            legacy_selections[name] = select_threshold(records, gold_dev, aspects, heldout,
                                                       old["aspect_threshold"], old["runner_up_threshold"])
            legacy_prediction = decode_records(records, keys_dev, aspects,
                legacy_selections[name]["aspect_threshold"], old["runner_up_threshold"])
            legacy_dev_summaries[name] = {"review_count": len(keys_dev), "seen": summarize_counts(
                count_tensor(keys_dev, legacy_prediction, seen_gold, aspects, heldout))["scopes"]["seen"]}
    candidates = args.candidate or [name for name in development if name != args.baseline and
                                    not name.lower().startswith("random")]
    if set(candidates) - set(development):
        raise ValueError("Unknown candidate method")
    best_dynamic = max(candidates, key=lambda name: (
        selections[name]["seen_pair_f1"], selections[name]["seen_presence_f1"],
        -selections[name]["seen_aspect_calls"], -candidates.index(name))) if candidates else None
    gain = (selections[best_dynamic]["seen_pair_f1"] -
            selections[args.baseline]["seen_pair_f1"]) if best_dynamic else None
    selected = best_dynamic if gain is not None and gain >= .005 else args.baseline
    selection = {"schema_version": "rag_selection_v1", "scope": args.scope,
        "presence_score_space": score_space,
        "baseline": args.baseline, "selected_method": selected,
        "best_dynamic_method": best_dynamic, "best_dynamic_seen_gain_points":
        100 * gain if gain is not None else None,
        "promotion_minimum_seen_pair_f1_points": .5,
        "selection_outcomes": "seen_aspects_only",
        "thresholds": selections, "development": dev_summaries,
        "input_sha256": {"artifacts": sha256(args.artifacts),
                         "development_labels": sha256(args.development_labels),
                         "development_scores": {n: sha256(p) for n, p in development.items()}}}
    if score_space == "logit_margin":
        selection["legacy_probability_recalibration"] = {
            "scope": "same_forward_preserved_float32_probability_seen_only_calibration",
            "thresholds": legacy_selections, "development": legacy_dev_summaries}
    if reviewed_calibration:
        selection["reviewed_calibration_provenance"] = reviewed_calibration["provenance"]
    # Persist method/threshold decisions BEFORE opening evaluation labels.
    write_json(args.output / "selection.json", selection)
    if args.classification_only:
        return selection
    comparator_dev_records = comparator_records(args.comparator_development)
    comparator_dev = predictions_from_comparator(comparator_dev_records)
    if comparator_dev_records is not None and not (set(comparator_dev_records) & set(keys_dev)):
        raise ValueError("Comparator development IDs do not match calibration population")
    frozen_routing, development_calls = {}, {}
    for name in dict.fromkeys([args.baseline, selected, best_dynamic]):
        if name is None:
            continue
        threshold = active_presence_threshold(selections[name])
        feat_dev = risk_features(records_dev[name], keys_dev, aspects, threshold,
            old["runner_up_threshold"], comparator_dev, presence_score_space=score_space)
        for disagreement in ([False, True] if comparator_dev is not None else [False]):
            policy_name = f"{name}__{'confidence_disagreement' if disagreement else 'confidence'}"
            policy = select_policy(feat_dev, dev_predictions[name], gold_dev, records_dev[name],
                                   aspects, heldout, use_disagreement=disagreement, seed=args.seed)
            policy["presence_score_space"] = score_space
            assignment = assign_review(feat_dev, policy, seed=args.seed)
            frozen_routing[policy_name] = {"method": name, "policy": policy,
                "calibration_comparator_cost": comparator_call_accounting(
                    assignment, comparator_dev_records or {}, len(aspects))}
            if disagreement:
                development_calls[policy_name] = call_manifest(feat_dev, assignment)
    write_json(args.output / "review_routing_selection.json", {
        "classification_selection_sha256": sha256(args.output / "selection.json"),
        "calibration_labels": "complete_labels_after_strict_unseen_method_freeze",
        "comparator_sha256": sha256(args.comparator_development)
        if args.comparator_development else None,
        "policies": frozen_routing})
    if development_calls:
        write_json(args.output / "review_routing_call_manifest_development.json", {
            "scope": "private_offline_boundary_call_manifest_not_executed_API_requests",
            "policies": development_calls})
    if args.selection_only:
        return selection
    if args.evaluation_labels is None:
        raise ValueError("Evaluation labels are required")
    # Locked label/history files are first validated after both selections have
    # been frozen above. Missing score records remain failures, never exclusions.
    reviewed_evaluation = reviewed_population(args, artifacts, "locked", args.evaluation_labels) \
        if args.scope == "new_source_locked" else None
    gold_eval = gold_by_id(reviewed_evaluation["labels"] if reviewed_evaluation else
                           read_jsonl(args.evaluation_labels), aspects)
    if set(gold_eval) & set(gold_dev):
        raise ValueError("Development and evaluation review IDs overlap")
    keys_eval = sorted(gold_eval)
    records_eval = {n: records_by_id(read_jsonl(p)) for n, p in evaluation.items()}
    weights = bootstrap_weights(len(keys_eval), args.bootstrap_draws, args.seed)
    tensors, eval_predictions, summaries, old_threshold_summaries = {}, {}, {}, {}
    legacy_tensors, legacy_eval_summaries = {}, {}
    for name, records in records_eval.items():
        population_check(records, gold_eval)
        validate_record_spaces(records, score_space)
        predictions = decode_records(records, keys_eval, aspects,
            active_presence_threshold(selections[name]), old["runner_up_threshold"],
            presence_score_space=score_space)
        eval_predictions[name] = predictions
        tensors[name] = count_tensor(keys_eval, predictions, gold_eval, aspects, heldout)
        summaries[name] = summarize_counts(tensors[name], weights)
        summaries[name]["operational"] = operational_summary(records, keys_eval, predictions,
                                                              gold_eval)
        raw = decode_records(records, keys_eval, aspects, old["aspect_threshold"],
                             old["runner_up_threshold"])
        old_threshold_summaries[name] = summarize_counts(
            count_tensor(keys_eval, raw, gold_eval, aspects, heldout))
        if score_space == "logit_margin":
            legacy_prediction = decode_records(records, keys_eval, aspects,
                legacy_selections[name]["aspect_threshold"], old["runner_up_threshold"])
            legacy_tensors[name] = count_tensor(keys_eval, legacy_prediction, gold_eval, aspects, heldout)
            legacy_eval_summaries[name] = summarize_counts(legacy_tensors[name], weights)
    contrasts = {f"{name}_vs_{args.baseline}": paired_contrast(tensor, tensors[args.baseline], weights)
                 for name, tensor in tensors.items() if name != args.baseline}
    for specification in args.compare:
        new, baseline = specification.split(":", 1)
        contrasts[f"{new}_vs_{baseline}"] = paired_contrast(tensors[new], tensors[baseline], weights)
    summary = {"schema_version": "rag_evaluation_v1", "scope": args.scope,
        "presence_score_space": score_space,
        "primary_comparison": "all_methods_vs_matched_fixed_in_the_same_presence_score_space",
        "interpretation": "historically_exposed_validation_replay_not_new_blind_test"
        if args.scope == "historical_replay" else "reviewed_frozen_new_source_locked_population",
        "selected_method": selected, "heldout_aspect": heldout,
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed,
                      "unit": "complete_review_paired", "interval": "percentile_95"},
        "methods": summaries, "old_threshold_diagnostics": old_threshold_summaries,
        "contrasts": contrasts, "selection_sha256": sha256(args.output / "selection.json"),
        "input_sha256": {"evaluation_labels": sha256(args.evaluation_labels),
                         "evaluation_scores": {n: sha256(p) for n, p in evaluation.items()}}}
    if score_space == "logit_margin":
        summary["legacy_probability_recalibration"] = {
            "scope": "same_forward_preserved_float32_probability_with_separate_frozen_seen_thresholds",
            "methods": legacy_eval_summaries,
            "contrasts_vs_fixed": {f"{name}_vs_{args.baseline}": paired_contrast(tensor,
                legacy_tensors[args.baseline], weights) for name, tensor in legacy_tensors.items() if name != args.baseline},
            "postprocessing_margin_vs_probability": {name: paired_contrast(tensors[name], legacy_tensors[name], weights)
                                                    for name in tensors}}
    if reviewed_evaluation:
        summary["reviewed_evaluation_provenance"] = reviewed_evaluation["provenance"]
        summary["prediction_coverage_policy"] = "full_registered_gold_population_with_explicit_and_missing_predictions_counted_as_failures"
    write_json(args.output / "quality.json", summary)
    # Routing is a separate business study. Its calibration can inspect complete
    # labels AFTER the strict-unseen classification policy has been frozen.
    comparator_eval_records = comparator_records(args.comparator_evaluation)
    comparator_eval = predictions_from_comparator(comparator_eval_records)
    if (comparator_dev is None) != (comparator_eval is None):
        raise ValueError("Both comparator populations must be supplied together")
    if comparator_eval_records is not None and not (set(comparator_eval_records) & set(keys_eval)):
        raise ValueError("Comparator evaluation IDs do not match evaluation population")
    routing_results, evaluation_calls = {}, {}
    for policy_name, entry in frozen_routing.items():
        name, policy = entry["method"], entry["policy"]
        threshold = active_presence_threshold(selections[name])
        feat_eval = risk_features(records_eval[name], keys_eval, aspects, threshold,
            old["runner_up_threshold"], comparator_eval, presence_score_space=score_space)
        assignment = assign_review(feat_eval, policy, seed=args.seed)
        result = assess_assignment(assignment, eval_predictions[name], gold_eval,
                                    records_eval[name], aspects, heldout)
        result["comparator_cost"] = comparator_call_accounting(
            assignment, comparator_eval_records or {}, len(aspects))
        if policy["use_disagreement"]:
            evaluation_calls[policy_name] = call_manifest(feat_eval, assignment)
        random_assignment = matched_random_assignment(feat_eval, len(assignment["manual"]),
                                                      seed=args.seed)
        result["matched_budget_random"] = assess_assignment(random_assignment,
            eval_predictions[name], gold_eval, records_eval[name], aspects, heldout)
        routing_results[policy_name] = result
    write_json(args.output / "review_routing.json", {
        "schema_version": "rag_review_routing_v2_boundary", "scope": args.scope,
        "classification_selection_frozen_before_routing": True,
        "manual_workload_unit": "review_items_not_time", "gold_substitution_is_simulation": True,
        "policy_comparison": "report_all_calibration_frozen_policies_no_evaluation_winner_selection",
        "policies": frozen_routing, "evaluation": routing_results,
        "frozen_routing_selection_sha256": sha256(args.output / "review_routing_selection.json"),
        "comparator_available": comparator_dev is not None,
        "full_review_reference": {"review_items": len(keys_eval), "automatic_items": 0,
                                  "review_item_reduction_percent": 0.0}})
    if evaluation_calls:
        write_json(args.output / "review_routing_call_manifest_evaluation.json", {
            "scope": "private_offline_boundary_call_manifest_not_executed_API_requests",
            "policies": evaluation_calls})
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifacts", type=Path, required=True)
    result.add_argument("--development", action="append", default=[], metavar="METHOD=JSONL")
    result.add_argument("--evaluation", action="append", default=[], metavar="METHOD=JSONL")
    result.add_argument("--development-root", type=Path)
    result.add_argument("--evaluation-root", type=Path)
    result.add_argument("--development-labels", type=Path, required=True)
    result.add_argument("--evaluation-labels", type=Path)
    result.add_argument("--annotation-data", type=Path,
                        help="Required for new_source_locked: original frozen annotation snapshot")
    result.add_argument("--reviewed-release", type=Path,
                        help="Required for new_source_locked: immutable reviewed release with split/history hashes")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--baseline", default="fixed")
    result.add_argument("--presence-score-space", choices=["probability", "logit_margin"], default="probability")
    result.add_argument("--candidate", action="append", default=[])
    result.add_argument("--compare", action="append", default=[], metavar="NEW:BASELINE")
    result.add_argument("--selection-only", action="store_true")
    result.add_argument("--classification-only", action="store_true",
                        help="Write seen-only selection and stop before all review routing")
    result.add_argument("--scope", choices=["historical_replay", "new_source_locked"],
                        default="historical_replay")
    result.add_argument("--bootstrap-draws", type=int, default=2000)
    result.add_argument("--seed", type=int, default=20261111)
    result.add_argument("--comparator-development", type=Path)
    result.add_argument("--comparator-evaluation", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({"scope": args.scope, "selected_method": result["selected_method"],
                      "output": str(args.output)}, ensure_ascii=False))
