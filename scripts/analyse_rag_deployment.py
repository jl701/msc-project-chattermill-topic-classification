"""Post-inference deployment quality audit; never selects models or thresholds.

Complete-review bootstrap compares actual API outputs with frozen offline scores.
Raw HTTP requests and score files stay unchanged. No HTTP/offline speedup is computed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHODS = ["fixed", "random_13", "random_37", "random_73", "bm25", "e5", "rrf", "e5_negative_seen"]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.rag_evaluation import (
    METRICS, SCOPES, SENTIMENTS, active_presence_threshold, bootstrap_weights,
    count_tensor, decode_records, f1_from_counts, gold_by_id, presence_space,
    read_jsonl, records_by_id, sampled_totals, score_maps, successful, summarize_counts,
    validate_record_spaces,
)
from msc_project.serving.presence_policy import validated_presence_policy


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Different analysis evidence exists; use a fresh output directory")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)


def frozen_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Frozen deployment path escapes its evidence directory")
    return path


def require_hash(path: Path, expected: str, description: str) -> None:
    if not expected or sha(path) != expected:
        raise ValueError(f"{description} hash changed or is missing")


def coverage(rows: list[dict], expected: list[str]) -> tuple[dict, dict[str, dict]]:
    groups, attempt_indices = defaultdict(list), []
    for row in rows:
        if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
            raise ValueError("HTTP attempt lacks its original sample identity")
        index = row.get("attempt_index")
        if type(index) is not int or index < 0:
            raise ValueError("HTTP attempt lacks a valid attempt index")
        groups[row["sample_id"]].append(row)
        attempt_indices.append(index)
    if len(set(attempt_indices)) != len(attempt_indices):
        raise ValueError("Repeated attempt indices cannot establish a unique earliest attempt")
    expected_set = set(expected)
    known = set(groups) & expected_set
    duplicate_ids = {key for key in groups if len(groups[key]) > 1}
    first = {key: min(groups[key], key=lambda row: row["attempt_index"]) for key in known}
    first_success = sum(row.get("success") is True for row in first.values())
    result = {"expected_reviews": len(expected), "attempts": len(rows),
        "successful_attempts": sum(row.get("success") is True for row in rows),
        "failed_attempts": sum(row.get("success") is not True for row in rows),
        "unique_expected_reviews_attempted": len(known), "missing_reviews": len(expected_set - known),
        "duplicate_review_ids": len(duplicate_ids),
        "duplicate_extra_attempts": sum(len(groups[key]) - 1 for key in duplicate_ids),
        "unexpected_review_ids": len(set(groups) - expected_set),
        "unexpected_attempts": sum(len(groups[key]) for key in set(groups) - expected_set),
        "first_attempt_successful_reviews": first_success,
        "first_attempt_failed_reviews": len(first) - first_success,
        "failed_or_missing_reviews_in_quality_denominator": len(expected) - first_success,
        "first_attempt_error_counts": dict(Counter(str(row.get("error") or "unspecified")
            for row in first.values() if row.get("success") is not True)),
        "clean_once_per_review_coverage": set(groups) == expected_set and len(rows) == len(expected)
            and not duplicate_ids,
        "attempt_resolution": "minimum_attempt_index_including_failures_never_prefer_successful_retry"}
    return result, first


def verified_inputs(root: Path, offline_path: Path, method: str) -> dict:
    """All provenance and coverage verification occurs before any gold-file access."""
    if not (root / "deployment_state.json").is_file():
        raise ValueError("Deployment inference must be complete before analysis; no completion state")
    state = read_json(root / "deployment_state.json")
    if state.get("status") != "complete" or state.get("method") != method:
        raise ValueError("Deployment inference must be complete for the requested method")
    completion_path = root / "deployment_completion.json"
    require_hash(completion_path, state.get("completion_sha256"), "Deployment completion receipt")
    completion = read_json(completion_path)
    if (completion.get("schema_version") != "rag_deployment_quality_complete_v1" or
            completion.get("status") != "complete" or completion.get("method") != method):
        raise ValueError("Invalid deployment completion receipt")
    require_hash(root / "registration.json", completion.get("registration_sha256"), "Registration")
    require_hash(root / "plan.json", completion.get("service_plan_sha256"), "Frozen service plan")
    require_hash(root / "offline_dependency_receipt.json", completion.get("offline_dependency_receipt_sha256"), "Offline dependency receipt")
    registration, plan = read_json(root / "registration.json"), read_json(root / "plan.json")
    if registration.get("schema_version") != "rag_deployment_quality_registration_v1" or registration.get("method") != method:
        raise ValueError("Wrong deployment registration or method")
    if registration["service_plan"]["sha256"] != sha(root / "plan.json"):
        raise ValueError("Registration differs from the exact frozen service plan")
    if registration["method_config"] != plan["methods"][method]:
        raise ValueError("Registered deployed method differs from frozen plan")
    if registration["index_files"] != plan["input_hashes"]["index_files"]:
        raise ValueError("Registered index differs from frozen service plan")
    if any(registration["source_hashes"].get(k) != v for k, v in plan["source_hashes"].items()):
        raise ValueError("Registered service source hashes differ from frozen plan")
    replay_meta, clone_meta = registration["evaluation_replay"], registration["clone"]
    replay_path = frozen_path(root, replay_meta["frozen_copy"])
    require_hash(replay_path, replay_meta["sha256"], "Evaluation replay")
    replay = read_jsonl(replay_path)
    if any(set(row) != {"request_id", "text"} or not isinstance(row["text"], str) for row in replay):
        raise ValueError("Deployment replay must contain input text and ID only")
    keys = [row["request_id"] for row in replay]
    if len(keys) != len(set(keys)) or keys != replay_meta["request_ids"] or len(keys) != replay_meta["reviews"]:
        raise ValueError("Registered evaluation replay population differs")
    if (not keys or any(not isinstance(key, str) or not key for key in keys) or
            registration["index_exclusion"]["manifest_sha256"] != registration["index_files"]["manifest.json"] or
            registration["index_exclusion"]["replay_sha256"] != replay_meta["sha256"] or
            not registration["index_exclusion"]["matching_exclusion_sources"]):
        raise ValueError("Replay population or frozen index exclusion evidence is invalid")
    clone_path = frozen_path(root, clone_meta["frozen_copy"])
    require_hash(clone_path, clone_meta["sha256"], "Frozen service clone")
    if clone_meta["sha256"] != plan["methods"][method]["artifacts_sha256"]:
        raise ValueError("Deployment clone differs from service plan")
    clone = read_json(clone_path)
    if (clone["service_calibration"]["method"] != method or
            clone["service_calibration"]["selection_sha256"] != plan["input_hashes"]["selection"]):
        raise ValueError("Wrong classifier method in service clone")
    if (clone["service_execution"]["sentiment_execution"] != plan["server"]["sentiment_execution"] or
            clone["service_execution"]["sentiment_group_contract"] != plan["sentiment_group_contract"]):
        raise ValueError("Clone sentiment execution differs from frozen service plan")
    dependency = read_json(root / "offline_dependency_receipt.json")
    campaign = registration["offline_campaign"]
    config_path = frozen_path(root, campaign["config_frozen_copy"])
    require_hash(config_path, campaign["config_sha256"], "Offline campaign registration")
    configuration = read_json(config_path)
    methods = campaign["methods"]
    if (methods != EXPECTED_METHODS or
            methods != [row["id"] for row in configuration["methods"]] or
            dependency["config_sha256"] != campaign["config_sha256"] or
            dependency["methods"] != methods or
            presence_space(campaign) != presence_space(plan) or
            presence_space(campaign) != presence_space(clone["variants"]["hybrid"]) or
            presence_space(campaign) != presence_space(configuration["inference"])):
        raise ValueError("Offline dependency differs from registered complete campaign")
    if (dependency["method"] != method or dependency["review_count"] != len(keys) or
            dependency["replay_sha256"] != replay_meta["sha256"] or
            dependency["artifacts_sha256"] != plan["input_hashes"]["artifacts"] or
            Path(dependency["score_path"]).resolve() != offline_path.resolve()):
        raise ValueError("Offline comparator does not match registered deployment inputs")
    offline_root = Path(dependency["root"])
    if offline_root.resolve() != Path(registration["offline_root"]).resolve():
        raise ValueError("Offline dependency root differs from registration")
    for name in ("identity", "summary", "state"):
        require_hash(offline_root / f"{name}.json", dependency[f"{name}_sha256"], f"Offline {name}")
    if read_json(offline_root / "state.json").get("status") != "complete":
        raise ValueError("Offline matrix is not complete")
    identity, offline_summary = read_json(offline_root / "identity.json"), read_json(offline_root / "summary.json")
    if (identity["replay"] != replay_meta["sha256"] or identity["artifacts"] != dependency["artifacts_sha256"] or
            set(identity["request_ids"]) != set(keys) or len(identity["request_ids"]) != len(keys) or
            identity["index_files"] != registration["index_files"] or method not in identity["methods"] or
            offline_summary["methods"][method]["file_sha256"] != dependency["score_sha256"]):
        raise ValueError("Offline matrix provenance differs from frozen dependency receipt")
    if (identity["config"] != campaign["config_sha256"] or identity["methods"] != methods or
            set(offline_summary["methods"]) != set(methods) or
            set(dependency["method_score_sha256"]) != set(methods)):
        raise ValueError("Offline dependency lacks the registered complete method matrix")
    aspects = [row["aspect"] for row in clone["taxonomy"]["candidates"]]
    for name in methods:
        method_path = offline_root / f"{name}.jsonl"
        expected_sha = dependency["method_score_sha256"][name]
        require_hash(method_path, expected_sha, "Offline matrix method score")
        records = records_by_id(read_jsonl(method_path))
        if (set(records) != set(keys) or offline_summary["methods"][name]["file_sha256"] != expected_sha or
                offline_summary["methods"][name]["successful"] != len(keys)):
            raise ValueError("Offline method population or successful count differs from complete receipt")
        validate_record_spaces(records, campaign["presence_score_space"])
        for row in records.values():
            if (row.get("status") != "ok" or row.get("response", {}).get("variant") != "hybrid" or
                    row["response"].get("request_id") != row["request_id"]):
                raise ValueError("Offline method contains failed or malformed records")
            score_maps(row, aspects, presence_score_space=campaign["presence_score_space"])
        expected_validation = {"reviews": len(keys), "unique_review_ids": len(keys), "successful": len(keys),
            "candidate_count": len(aspects), "presence_score_space": campaign["presence_score_space"], "complete_outputs": True}
        if dependency["per_method_validation"][name] != expected_validation:
            raise ValueError("Offline validation receipt disagrees with complete score outputs")
    require_hash(offline_path, dependency["score_sha256"], "Offline score file")
    run_dir = frozen_path(root, completion["run_directory"])
    if run_dir.name != registration["run"]["name"]:
        raise ValueError("Completed run name differs from registration")
    for name in ("requests.jsonl", "summary.json", "complete.json", "registered_run.json", "config.json"):
        require_hash(run_dir / name, completion["files_sha256"].get(name), f"Deployment {name}")
    summary, completed = read_json(run_dir / "summary.json"), read_json(run_dir / "complete.json")
    if completed["summary_sha256"] != sha(run_dir / "summary.json") or summary.get("interrupted"):
        raise ValueError("Run incomplete or completion summary changed")
    run, client = read_json(run_dir / "registered_run.json"), read_json(run_dir / "config.json")
    if (run != registration["run"] or run.get("mode") != "closed" or run.get("concurrency") != 1 or
            run.get("requests") != len(keys) or run.get("shuffle") is not False or
            Path(run["replay"]).resolve() != replay_path):
        raise ValueError("Completed run differs from the registered one-attempt C1 replay")
    if any(client.get(field) != run[field] for field in ("mode", "concurrency", "requests", "shuffle", "seed", "replay")):
        raise ValueError("Actual HTTP client configuration differs from registered replay")
    rows = read_jsonl(run_dir / "requests.jsonl")
    observed, first = coverage(rows, keys)
    if summary["attempts"] != observed["attempts"] or summary["successful"] != observed["successful_attempts"]:
        raise ValueError("Run summary does not match preserved request journal")
    coverage_keys = {"unique_expected_reviews": "unique_expected_reviews_attempted",
                     "extra_duplicate_attempts": "duplicate_extra_attempts", "valid": "clean_once_per_review_coverage"}
    if any(value != observed[coverage_keys.get(key, key)] for key, value in completion["coverage"].items()):
        raise ValueError("Completion coverage differs from preserved request journal")
    return {"registration": registration, "plan": plan, "clone": clone, "keys": keys,
            "coverage": observed, "first": first, "offline_path": offline_path,
            "completion_sha256": sha(completion_path), "requests_sha256": sha(run_dir / "requests.jsonl")}


def api_predictions(record: dict, clone: dict, plan: dict, aspects: list[str]) -> tuple[set, dict]:
    response = record.get("response") or {}
    if not (record.get("success") is True and type(record.get("status_code")) is int and
            200 <= record["status_code"] < 300 and response.get("request_id") == record["request_id"]):
        raise ValueError("Successful HTTP attempt has invalid status or response identity")
    diag = response.get("diagnostics") or {}
    if response.get("variant") != "hybrid" or presence_space(response) != presence_space(clone["variants"]["hybrid"]):
        raise ValueError("API variant or presence score space differs from frozen deployment")
    fallback = diag.get("retrieval_fallback") is True
    method = clone["service_calibration"]["method"]
    thresholds = clone["retrieval_fallback_policy"] if fallback and method != "fixed" else clone["variants"]["hybrid"]
    if (validated_presence_policy(diag["effective_thresholds"]) != validated_presence_policy(thresholds) or
            diag["effective_thresholds"]["runner_up_threshold"] != thresholds["runner_up_threshold"]):
        raise ValueError("API effective thresholds differ from frozen inference/fallback policy")
    if (diag.get("sentiment_execution") != plan["server"]["sentiment_execution"] or
            diag.get("sentiment_group_contract") != plan["sentiment_group_contract"]):
        raise ValueError("API execution identity differs from frozen service plan")
    expected_prefix = (f"{clone['model']['revision'][:8]}-{clone['fold_id']}-"
                       f"{plan['methods'][method]['artifacts_sha256'][:12]}-hybrid-"
                       f"prefix{plan['methods'][method]['prefix_cache']}")
    version = response.get("model_version", "")
    if (not version.startswith(expected_prefix + "-rag-") or
            diag.get("batch_aspect", {}).get("prefix_cache_configured") is not (plan["methods"][method]["prefix_cache"] == "on")):
        raise ValueError("API model version or cache configuration differs from frozen clone")
    expected_retrieval = {"method": "fixed", "selection_policy": "global"} if fallback else clone["service_calibration"]["retrieval"]
    if (diag.get("effective_retrieval_method") != expected_retrieval["method"] or
            diag.get("effective_retrieval_selection_policy") != expected_retrieval["selection_policy"]):
        raise ValueError("API retrieval identity differs from frozen deployment")
    fallback_source = plan["methods"]["fixed"]["artifacts_sha256"] if method == "fixed" else thresholds.get("source_artifacts_sha256")
    if fallback and diag.get("fallback_policy_source_sha256") != fallback_source:
        raise ValueError("API fallback does not name its frozen source")
    pairs, seen = set(), set()
    for item in response["predictions"]:
        aspect, labels = item["aspect"], item["sentiments"]
        if aspect not in aspects or aspect in seen or not labels or len(set(labels)) != len(labels) or set(labels) - set(SENTIMENTS):
            raise ValueError("Malformed API predicted label set")
        seen.add(aspect)
        pairs.update((aspect, label) for label in labels)
    grid = {}
    for item in response["aspect_scores"]:
        aspect, probability = item["aspect"], float(item["score"])
        margin = float(item["presence_logit_margin"])
        if aspect in grid or not math.isfinite(probability) or not 0 <= probability <= 1 or not math.isfinite(margin):
            raise ValueError("Invalid complete API presence score grid")
        grid[aspect] = (probability, margin)
    if set(grid) != set(aspects):
        raise ValueError("API response must contain every candidate's probability and margin")
    return pairs, grid


def paired_difference(api: np.ndarray, offline: np.ndarray, weights: np.ndarray) -> dict:
    a, b = api.sum(axis=0), offline.sum(axis=0)
    sa, sb = sampled_totals(api, weights), sampled_totals(offline, weights)
    result = {"direction": "API_minus_frozen_offline", "scopes": {}}
    for si, scope in enumerate(SCOPES):
        result["scopes"][scope] = {}
        for mi, metric in enumerate(METRICS):
            f1_difference = 100 * (f1_from_counts(a[si, mi]) - f1_from_counts(b[si, mi]))
            sampled_difference = 100 * (f1_from_counts(sa[:, si, mi]) - f1_from_counts(sb[:, si, mi]))
            entry = {"f1_difference_points": float(f1_difference),
                     "f1_difference_points_percentile_95_interval": np.quantile(sampled_difference, [.025, .975]).tolist()}
            for ci, count in enumerate(("tp", "fp", "fn")):
                entry[count + "_difference"] = int(a[si, mi, ci] - b[si, mi, ci])
                entry[count + "_difference_percentile_95_interval"] = np.quantile(sa[:, si, mi, ci] - sb[:, si, mi, ci], [.025, .975]).tolist()
            result["scopes"][scope][metric] = entry
    return result


def analyse(args: argparse.Namespace) -> dict:
    inputs = verified_inputs(args.deployment_root.resolve(), args.offline_scores, args.method)
    observed = inputs["coverage"]
    write_json(args.output / "coverage.json", observed)
    if not observed["clean_once_per_review_coverage"] and not args.allow_incomplete_audit:
        raise ValueError("Deployment coverage is incomplete or duplicated; coverage diagnostic saved; gold not opened")
    keys, clone, plan = inputs["keys"], inputs["clone"], inputs["plan"]
    aspects = [row["aspect"] for row in clone["taxonomy"]["candidates"]]
    heldouts = clone["taxonomy"]["heldout_aspects"]
    if len(heldouts) != 1:
        raise ValueError("Exactly one registered heldout aspect is required")
    thresholds = clone["variants"]["hybrid"]
    offline = records_by_id(read_jsonl(args.offline_scores))
    if set(offline) != set(keys):
        raise ValueError("Offline score population does not match complete deployment population")
    validate_record_spaces(offline, presence_space(thresholds))
    offline_predictions = decode_records(offline, keys, aspects, active_presence_threshold(thresholds),
        thresholds["runner_up_threshold"], presence_score_space=presence_space(thresholds))
    actual, differences = {}, []
    max_probability = max_margin = 0.0
    pairs_compared = comparable_reviews = 0
    max_by_aspect = {aspect: {"probability": 0.0, "logit_margin": 0.0} for aspect in aspects}
    changes, success_changes = Counter(), Counter()
    for key in keys:
        row = inputs["first"].get(key)
        is_success = row is not None and row.get("success") is True
        if is_success:
            actual[key], grid = api_predictions(row, clone, plan, aspects)
            if successful(offline[key]):
                old_grid = {item["aspect"]: item for item in offline[key]["response"]["aspect_scores"]}
                comparable_reviews += 1
                for aspect in aspects:
                    pd = abs(grid[aspect][0] - float(old_grid[aspect]["score"]))
                    md = abs(grid[aspect][1] - float(old_grid[aspect]["presence_logit_margin"]))
                    if not math.isfinite(md):
                        raise ValueError("Offline margin is non-finite")
                    max_probability, max_margin = max(max_probability, pd), max(max_margin, md)
                    max_by_aspect[aspect]["probability"] = max(max_by_aspect[aspect]["probability"], pd)
                    max_by_aspect[aspect]["logit_margin"] = max(max_by_aspect[aspect]["logit_margin"], md)
                    pairs_compared += 1
        else:
            actual[key] = set()
        old, new = offline_predictions[key], actual[key]
        category = "unchanged" if old == new else "presence_changed" if {p[0] for p in old} != {p[0] for p in new} else "sentiment_only_changed"
        changes[category] += 1
        if is_success:
            success_changes[category] += 1
        if category != "unchanged":
            differences.append({"request_id": key, "api_successful": is_success, "change_type": category,
                "API_only": sorted(new - old), "offline_only": sorted(old - new)})
    # Gold is first accessed here, after completion, provenance, coverage and score-schema checks.
    gold = gold_by_id(read_jsonl(args.labels), aspects)
    if set(gold) != set(keys):
        raise ValueError("Gold must cover exactly the registered evaluation population")
    weights = bootstrap_weights(len(keys), draws=2000, seed=20261111)
    api_counts = count_tensor(keys, actual, gold, aspects, heldouts[0])
    offline_counts = count_tensor(keys, offline_predictions, gold, aspects, heldouts[0])
    result = {"schema_version": "rag_deployment_quality_analysis_v1", "method": args.method,
        "scope": "historically_exposed_deployment_replay" if observed["clean_once_per_review_coverage"] else "coverage_audit_not_complete_deployment",
        "population_scope": "historically_exposed_validation_replay",
        "population_note": "These reviews were historically exposed; current selection isolation does not make them a new blind test",
        "no_classifier_or_threshold_selection": True, "coverage": observed,
        "frozen_thresholds": thresholds, "bootstrap": {"draws": 2000, "seed": 20261111,
            "unit": "complete_review_paired", "interval": "percentile_95"},
        "API": summarize_counts(api_counts, weights), "frozen_offline": summarize_counts(offline_counts, weights),
        "API_minus_offline": paired_difference(api_counts, offline_counts, weights),
        "review_label_set_changes_all_expected_reviews": dict(changes),
        "review_label_set_changes_API_successes_only": dict(success_changes),
        "presence_score_deltas": {"successful_paired_reviews": comparable_reviews, "candidate_pairs": pairs_compared,
            "probability_max_abs_delta": max_probability if pairs_compared else None,
            "logit_margin_max_abs_delta": max_margin if pairs_compared else None, "max_abs_delta_by_aspect": max_by_aspect},
        "actual_API_exact_set_accuracy_all_expected_reviews": sum(
            inputs["first"].get(key, {}).get("success") is True and actual[key] == gold[key] for key in keys) / len(keys),
        "offline_failed_reviews": sum(not successful(offline[key]) for key in keys),
        "timing_scope": "HTTP delivery timing is not compared with offline component timing; no speedup ratio",
        "input_sha256": {"deployment_completion": inputs["completion_sha256"], "requests": inputs["requests_sha256"],
            "offline_scores": sha(args.offline_scores), "gold": sha(args.labels), "analysis_source": sha(Path(__file__)),
            "metric_source": sha(ROOT / "src/msc_project/serving/rag_evaluation.py"),
            "presence_policy_source": sha(ROOT / "src/msc_project/serving/presence_policy.py")}}
    write_json(args.output / "quality.json", result)
    write_json(args.output / "private_review_differences.json", {"contains_gold": False, "reviews": differences})
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--deployment-root", type=Path, required=True)
    value.add_argument("--offline-scores", type=Path, required=True)
    value.add_argument("--labels", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--method", default="fixed")
    value.add_argument("--allow-incomplete-audit", action="store_true")
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    result = analyse(args)
    print(json.dumps({"scope": result["scope"], "coverage": result["coverage"], "output": str(args.output)}))
