"""Calibrate on development only, then report historical/new-source RAG evidence.

Only aggregate metrics and frozen thresholds are written. Prediction records,
gold labels and their identifiers remain in the private campaign directories.
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
    bootstrap_weights, count_tensor, decode_records, gold_by_id, paired_contrast,
    read_jsonl, records_by_id, select_threshold, successful, summarize_counts,
)
from msc_project.serving.review_routing import (
    assess_assignment, assign_review, matched_random_assignment, risk_features, select_policy,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Different evidence already exists at {path}; use a new output directory")
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_paths(items: list[str], root: Path | None) -> dict[str, Path]:
    result = {path.stem: path for path in sorted(root.glob("*.jsonl"))
              if not path.name.startswith("_")} if root else {}
    for item in items:
        name, path = item.split("=", 1)
        if not name or name in result:
            raise ValueError("Duplicate or empty method name")
        result[name] = Path(path)
    return result


def comparator_predictions(path: Path | None) -> dict[str, set] | None:
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
        if successful(record) and key not in result:
            result[key] = {(item["aspect"], sentiment)
                           for item in record["response"].get("predictions", [])
                           for sentiment in item["sentiments"]}
    return result


def population_check(records: dict, gold: dict) -> None:
    if set(records) - set(gold):
        raise ValueError("Scores contain IDs outside the registered population")


def operational_summary(records: dict, keys: list[str], predictions: dict, gold: dict) -> dict:
    success = sum(successful(records.get(key)) for key in keys)
    elapsed = [float(records[key]["elapsed_ms"]) for key in keys if key in records and
               records[key].get("elapsed_ms") is not None]
    if elapsed and (not np.isfinite(elapsed).all() or min(elapsed) < 0):
        raise ValueError("Invalid scorer elapsed time")
    return {"successful_reviews": success, "failed_or_missing_reviews": len(keys) - success,
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
    development = method_paths(args.development, args.development_root)
    evaluation = method_paths(args.evaluation, args.evaluation_root)
    if args.baseline not in development:
        raise ValueError("Fixed baseline is required for development selection")
    if not (args.selection_only or args.classification_only) and set(evaluation) != set(development):
        raise ValueError("Development and evaluation methods must match exactly")
    gold_dev = gold_by_id(read_jsonl(args.development_labels), aspects)
    keys_dev = sorted(gold_dev)
    records_dev = {name: records_by_id(read_jsonl(path)) for name, path in development.items()}
    selections, dev_predictions, dev_summaries = {}, {}, {}
    for name, records in records_dev.items():
        population_check(records, gold_dev)
        selections[name] = select_threshold(records, gold_dev, aspects, heldout,
                                             old["aspect_threshold"], old["runner_up_threshold"])
        threshold = selections[name]["aspect_threshold"]
        dev_predictions[name] = decode_records(records, keys_dev, aspects, threshold,
                                                old["runner_up_threshold"])
        # Development summaries are restricted to seen outcomes; heldout outcomes
        # must not guide the next improvement round.
        seen_gold = {key: {p for p in pairs if p[0] != heldout} for key, pairs in gold_dev.items()}
        counts = count_tensor(keys_dev, dev_predictions[name], seen_gold, aspects, heldout)
        dev_summaries[name] = {"review_count": len(keys_dev),
            "seen": summarize_counts(counts)["scopes"]["seen"],
            "failed_or_missing_reviews": sum(not successful(records.get(k)) for k in keys_dev)}
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
        "baseline": args.baseline, "selected_method": selected,
        "best_dynamic_method": best_dynamic, "best_dynamic_seen_gain_points":
        100 * gain if gain is not None else None,
        "promotion_minimum_seen_pair_f1_points": .5,
        "selection_outcomes": "seen_aspects_only",
        "thresholds": selections, "development": dev_summaries,
        "input_sha256": {"artifacts": sha256(args.artifacts),
                         "development_labels": sha256(args.development_labels),
                         "development_scores": {n: sha256(p) for n, p in development.items()}}}
    # Persist method/threshold decisions BEFORE opening evaluation labels.
    write_json(args.output / "selection.json", selection)
    if args.classification_only:
        return selection
    comparator_dev = comparator_predictions(args.comparator_development)
    if comparator_dev is not None and not (set(comparator_dev) & set(keys_dev)):
        raise ValueError("Comparator development IDs do not match calibration population")
    frozen_routing = {}
    for name in dict.fromkeys([args.baseline, selected, best_dynamic]):
        if name is None:
            continue
        threshold = selections[name]["aspect_threshold"]
        feat_dev = risk_features(records_dev[name], keys_dev, aspects, threshold,
                                old["runner_up_threshold"], comparator_dev)
        for disagreement in ([False, True] if comparator_dev is not None else [False]):
            policy_name = f"{name}__{'confidence_disagreement' if disagreement else 'confidence'}"
            policy = select_policy(feat_dev, dev_predictions[name], gold_dev, records_dev[name],
                                   aspects, heldout, use_disagreement=disagreement, seed=args.seed)
            frozen_routing[policy_name] = {"method": name, "policy": policy}
    write_json(args.output / "review_routing_selection.json", {
        "classification_selection_sha256": sha256(args.output / "selection.json"),
        "calibration_labels": "complete_labels_after_strict_unseen_method_freeze",
        "comparator_sha256": sha256(args.comparator_development)
        if args.comparator_development else None,
        "policies": frozen_routing})
    if args.selection_only:
        return selection
    if args.evaluation_labels is None:
        raise ValueError("Evaluation labels are required")
    gold_eval = gold_by_id(read_jsonl(args.evaluation_labels), aspects)
    if set(gold_eval) & set(gold_dev):
        raise ValueError("Development and evaluation review IDs overlap")
    keys_eval = sorted(gold_eval)
    records_eval = {n: records_by_id(read_jsonl(p)) for n, p in evaluation.items()}
    weights = bootstrap_weights(len(keys_eval), args.bootstrap_draws, args.seed)
    tensors, eval_predictions, summaries, old_threshold_summaries = {}, {}, {}, {}
    for name, records in records_eval.items():
        population_check(records, gold_eval)
        predictions = decode_records(records, keys_eval, aspects,
            selections[name]["aspect_threshold"], old["runner_up_threshold"])
        eval_predictions[name] = predictions
        tensors[name] = count_tensor(keys_eval, predictions, gold_eval, aspects, heldout)
        summaries[name] = summarize_counts(tensors[name], weights)
        summaries[name]["operational"] = operational_summary(records, keys_eval, predictions,
                                                              gold_eval)
        raw = decode_records(records, keys_eval, aspects, old["aspect_threshold"],
                             old["runner_up_threshold"])
        old_threshold_summaries[name] = summarize_counts(
            count_tensor(keys_eval, raw, gold_eval, aspects, heldout))
    contrasts = {f"{name}_vs_{args.baseline}": paired_contrast(tensor, tensors[args.baseline], weights)
                 for name, tensor in tensors.items() if name != args.baseline}
    for specification in args.compare:
        new, baseline = specification.split(":", 1)
        contrasts[f"{new}_vs_{baseline}"] = paired_contrast(tensors[new], tensors[baseline], weights)
    summary = {"schema_version": "rag_evaluation_v1", "scope": args.scope,
        "interpretation": "historically_exposed_validation_replay_not_new_blind_test"
        if args.scope == "historical_replay" else "independently_labelled_locked_population",
        "selected_method": selected, "heldout_aspect": heldout,
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed,
                      "unit": "complete_review_paired", "interval": "percentile_95"},
        "methods": summaries, "old_threshold_diagnostics": old_threshold_summaries,
        "contrasts": contrasts, "selection_sha256": sha256(args.output / "selection.json"),
        "input_sha256": {"evaluation_labels": sha256(args.evaluation_labels),
                         "evaluation_scores": {n: sha256(p) for n, p in evaluation.items()}}}
    write_json(args.output / "quality.json", summary)
    # Routing is a separate business study. Its calibration can inspect complete
    # labels AFTER the strict-unseen classification policy has been frozen.
    comparator_eval = comparator_predictions(args.comparator_evaluation)
    if (comparator_dev is None) != (comparator_eval is None):
        raise ValueError("Both comparator populations must be supplied together")
    if comparator_eval is not None and not (set(comparator_eval) & set(keys_eval)):
        raise ValueError("Comparator evaluation IDs do not match evaluation population")
    routing_results = {}
    for policy_name, entry in frozen_routing.items():
        name, policy = entry["method"], entry["policy"]
        threshold = selections[name]["aspect_threshold"]
        feat_eval = risk_features(records_eval[name], keys_eval, aspects, threshold,
                                 old["runner_up_threshold"], comparator_eval)
        assignment = assign_review(feat_eval, policy, seed=args.seed)
        result = assess_assignment(assignment, eval_predictions[name], gold_eval,
                                    records_eval[name], aspects, heldout)
        random_assignment = matched_random_assignment(feat_eval, len(assignment["manual"]),
                                                      seed=args.seed)
        result["matched_budget_random"] = assess_assignment(random_assignment,
            eval_predictions[name], gold_eval, records_eval[name], aspects, heldout)
        routing_results[policy_name] = result
    write_json(args.output / "review_routing.json", {
        "schema_version": "rag_review_routing_v1", "scope": args.scope,
        "classification_selection_frozen_before_routing": True,
        "manual_workload_unit": "review_items_not_time", "gold_substitution_is_simulation": True,
        "policies": frozen_routing, "evaluation": routing_results,
        "frozen_routing_selection_sha256": sha256(args.output / "review_routing_selection.json"),
        "comparator_available": comparator_dev is not None,
        "full_review_reference": {"review_items": len(keys_eval), "automatic_items": 0,
                                  "review_item_reduction_percent": 0.0}})
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
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--baseline", default="fixed")
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
