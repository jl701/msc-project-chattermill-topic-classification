"""Descriptive length/truncation profiles; never calibrates or selects a model.

Historical gold is optional and loaded only after complete inference is verified.
New-source use is label-blind; its predicted distributions are not quality metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyse_rag_campaign import method_paths, sha256, write_json
from msc_project.serving.rag_evaluation import (
    active_presence_threshold, count_tensor, decode_records, gold_by_id, read_jsonl,
    records_by_id, score_maps, successful, summarize_counts, validate_record_spaces,
)
from msc_project.serving.presence_policy import presence_space

BANDS = ("0_to_100", "101_to_300", "301_to_1000", "over_1000")


def length_band(characters):
    return BANDS[0 if characters <= 100 else 1 if characters <= 300 else 2 if characters <= 1000 else 3]


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "median": None, "p95": None, "maximum": None}
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Profile values must be finite and nonnegative")
    return {"count": len(values), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "maximum": float(values.max())}


def describe(keys, records, predictions, aspects, heldout, gold=None):
    valid = [key for key in keys if successful(records.get(key))]
    rows = [records[key] for key in valid]
    diagnostics = [row["response"].get("diagnostics", {}) for row in rows]
    prompts = [item for diag in diagnostics for item in diag.get("batch_aspect", {}).get("retrieval", [])]
    trims = [item["truncation"] for item in prompts if isinstance(item.get("truncation"), dict)]
    sentiment_audits = [item for row in rows for item in row["response"].get("sentiment_truncation_by_aspect", {}).values()]
    selected_sentiment_audits = [item for key in valid for aspect, item in records[key]["response"].get("sentiment_truncation_by_aspect", {}).items()
        if aspect in {a for a, _ in predictions[key]}]
    result = {
        "registered_reviews": len(keys), "successful_reviews": len(valid),
        "failed_or_missing_reviews": len(keys) - len(valid),
        "diagnostics_reviews": sum(bool(diag.get("batch_aspect")) for diag in diagnostics),
        "empty_output_successful_reviews": sum(not predictions[key] for key in valid),
        "retrieval_fallback_reviews": sum(bool(diag.get("retrieval_fallback")) for diag in diagnostics),
        "severe_query_trim_reviews": sum(bool(diag.get("severe_truncation")) for diag in diagnostics),
        "retrieval_prompt_count": len(prompts), "truncation_audited_prompts": len(trims),
        "any_demo_trim_prompts": sum(any(t.get("demo_characters_removed", [])) for t in trims),
        "any_query_trim_prompts": sum(t.get("query_characters_removed", 0) > 0 for t in trims),
        "severe_query_trim_prompts": sum(bool(t.get("severe_query_truncation")) for t in trims),
        "original_prompt_tokens": distribution([t["original_prompt_tokens"] for t in trims]),
        "final_prompt_tokens": distribution([t["final_prompt_tokens"] for t in trims]),
        "demo_characters_removed_per_prompt": distribution([sum(t.get("demo_characters_removed", [])) for t in trims]),
        "query_characters_removed_per_prompt": distribution([t.get("query_characters_removed", 0) for t in trims]),
        "successful_presence_stage_elapsed_ms": distribution([row["elapsed_ms"] for row in rows if "elapsed_ms" in row]),
        "presence_input_tokens_per_review": distribution([d["aspect_input_tokens"] for d in diagnostics if "aspect_input_tokens" in d]),
        "predicted_presence_reviews_by_aspect": {aspect: sum(any(a == aspect for a, _ in predictions[key]) for key in keys) for aspect in aspects},
        "full_grid_sentiment_audited_prompts": len(sentiment_audits),
        "full_grid_sentiment_left_truncated_prompts": sum(bool(t.get("left_truncated")) for t in sentiment_audits),
        "selected_sentiment_audited_prompts": len(selected_sentiment_audits),
        "selected_sentiment_left_truncated_prompts": sum(bool(t.get("left_truncated")) for t in selected_sentiment_audits),
        "selected_sentiment_left_truncated_reviews": sum(any(t.get("left_truncated") for a, t in records[key]["response"].get("sentiment_truncation_by_aspect", {}).items()
            if a in {aspect for aspect, _ in predictions[key]}) for key in valid),
        "timing_scope": "offline_presence_stage_including_retrieval_not_HTTP_or_full_two_stage_latency",
    }
    if gold is not None and keys:
        result["quality"] = summarize_counts(count_tensor(keys, predictions, gold, aspects, heldout))
        result["exact_set_accuracy_all_attempts"] = sum(key in valid and predictions[key] == gold[key] for key in keys) / len(keys)
    return result


def run(args):
    identity = json.loads((args.scores / "identity.json").read_text(encoding="utf-8"))
    state = json.loads((args.scores / "state.json").read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise ValueError("Profile requires completed inference before any gold read")
    if identity["replay"] != sha256(args.replay) or identity["artifacts"] != sha256(args.artifacts):
        raise ValueError("Replay or artifacts differ from the inference identity")
    replay_rows = read_jsonl(args.replay)
    if any(set(row) != {"request_id", "text"} or not isinstance(row["text"], str) for row in replay_rows):
        raise ValueError("Replay must contain only request_id and text")
    replay = records_by_id(replay_rows)
    keys = sorted(replay)
    if set(identity["request_ids"]) != set(keys) or len(identity["request_ids"]) != len(keys):
        raise ValueError("Inference identity does not cover the exact replay")
    if args.scope == "new_source_labelblind" and args.historical_labels:
        raise ValueError("New-source profile cannot read labels")
    if args.scope == "historical_replay" and not args.historical_labels:
        raise ValueError("Historical quality profile requires historical labels")
    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection["input_sha256"]["artifacts"] != identity["artifacts"]:
        raise ValueError("Selection artifacts differ from inference artifacts")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_methods = [method["id"] for method in config["methods"]]
    summary = json.loads((args.scores / "summary.json").read_text(encoding="utf-8"))
    if (identity["config"] != sha256(args.config) or identity["methods"] != expected_methods or
            state.get("methods") != expected_methods or set(summary["methods"]) != set(expected_methods) or
            summary.get("reviews") != len(keys) or state.get("reviews") != len(keys)):
        raise ValueError("Completed matrix does not match the registered config, methods or population")
    aspects = [row["aspect"] for row in artifacts["taxonomy"]["candidates"]]
    heldout, = artifacts["taxonomy"]["heldout_aspects"]
    paths = method_paths([], args.scores)
    records = {name: records_by_id(read_jsonl(path)) for name, path in paths.items()}
    if set(paths) - set(selection["thresholds"]):
        raise ValueError("Missing frozen threshold; profile never selects one")
    predictions = {}
    for name, values in records.items():
        if (set(values) != set(keys) or summary["methods"][name]["file_sha256"] != sha256(paths[name]) or
                summary["methods"][name].get("successful") != sum(successful(row) for row in values.values())):
            raise ValueError("Completed score file hash, coverage or successful count differs")
        validate_record_spaces(values, selection["presence_score_space"])
        for row in values.values():
            if successful(row):
                score_maps(row, aspects, presence_score_space=selection["presence_score_space"])
                if any(abs(sum(item["sentiment_scores"].values()) - 1) > 1e-4 for item in row["response"]["full_grid_scores"]):
                    raise ValueError("Sentiment probabilities must sum to one")
        threshold = selection["thresholds"][name]
        if presence_space(threshold) != selection["presence_score_space"]:
            raise ValueError("Frozen method threshold score space differs from the selection")
        predictions[name] = decode_records(values, keys, aspects, active_presence_threshold(threshold),
            threshold["runner_up_threshold"], presence_score_space=selection["presence_score_space"])
    if args.historical_labels:
        if args.historical_manifest is None:
            raise ValueError("Historical gold requires the original replay manifest")
        historical = json.loads(args.historical_manifest.read_text(encoding="utf-8"))
        registered = historical["splits"]["evaluation"]
        if (historical.get("source_split") != "validation" or historical.get("blind_evaluation") is not False or
                registered["count"] != len(keys) or registered["files"]["replay"]["sha256"] != identity["replay"] or
                registered["files"]["labels"]["sha256"] != sha256(args.historical_labels)):
            raise ValueError("Historical files do not match the original historical manifest")
    gold = gold_by_id(read_jsonl(args.historical_labels), aspects) if args.historical_labels else None
    if gold is not None and set(gold) != set(keys):
        raise ValueError("Historical gold must cover the exact registered replay")
    output = {
        "schema": "rag_descriptive_profile_v1", "scope": args.scope,
        "length_bands_characters": {"0_to_100": [0, 100], "101_to_300": [101, 300], "301_to_1000": [301, 1000], "over_1000": [1001, None]},
        "interpretation": "descriptive_subgroups_without_selection_or_significance_claims; labelblind_predictions_are_not_accuracy",
        "threshold_origin": "frozen_historical_seen_development_selection; no_new_source_calibration",
        "input_sha256": {"identity": sha256(args.scores / "identity.json"), "state": sha256(args.scores / "state.json"),
            "summary": sha256(args.scores / "summary.json"), "config": sha256(args.config),
            "historical_manifest": sha256(args.historical_manifest) if args.historical_labels else None,
            "replay": sha256(args.replay), "artifacts": sha256(args.artifacts), "selection": sha256(args.selection),
            "labels": sha256(args.historical_labels) if args.historical_labels else None,
            "scores": {name: sha256(path) for name, path in paths.items()}},
        "source_sha256": {"script": sha256(Path(__file__)),
            "evaluation_helpers": sha256(Path(__file__).resolve().parents[1] / "src/msc_project/serving/rag_evaluation.py"),
            "campaign_helpers": sha256(Path(__file__).with_name("analyse_rag_campaign.py"))}, "methods": {},
    }
    for name, values in records.items():
        prediction = predictions[name]
        entry = {"all": describe(keys, values, prediction, aspects, heldout, gold),
            "length_groups": {band: describe([key for key in keys if length_band(len(replay[key]["text"])) == band],
                values, prediction, aspects, heldout, gold) for band in BANDS}}
        if gold is not None:
            entry["quality_by_aspect"] = {aspect: summarize_counts(count_tensor(keys, prediction, gold, [aspect], aspect))["scopes"]["overall"] for aspect in aspects}
        output["methods"][name] = entry
    write_json(args.output, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifacts", "replay", "scores", "selection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scope", choices=["historical_replay", "new_source_labelblind"], required=True)
    parser.add_argument("--historical-labels", type=Path)
    parser.add_argument("--historical-manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/experiments/serving_rag_negative_20261111.json")
    result = run(parser.parse_args())
    print(json.dumps({"scope": result["scope"], "methods": list(result["methods"])}))
