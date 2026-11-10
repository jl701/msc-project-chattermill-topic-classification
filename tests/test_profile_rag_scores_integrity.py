"""Synthetic provenance and gold-access boundaries for descriptive profiles."""
from argparse import Namespace
import json
import math

import pytest

import profile_rag_scores as profile
from test_rag_evaluation import record


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def lines(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding="utf-8")


def setup_profile(tmp_path):
    args = Namespace(scores=tmp_path / "scores", replay=tmp_path / "replay.jsonl",
        artifacts=tmp_path / "artifacts.json", selection=tmp_path / "selection.json",
        config=tmp_path / "config.json", historical_labels=tmp_path / "synthetic_gold.jsonl",
        historical_manifest=tmp_path / "historical_manifest.json", output=tmp_path / "profile.json",
        scope="historical_replay")
    args.scores.mkdir()
    keys = [f"synthetic{i}" for i in range(4)]
    methods = ["fixed", "e5"]
    lines(args.replay, [{"request_id": key, "text": "x" * size}
                       for key, size in zip(keys, (100, 101, 301, 1001))])
    dump(args.artifacts, {"taxonomy": {"candidates": [{"aspect": "seen"}, {"aspect": "heldout"}],
                                      "heldout_aspects": ["heldout"]}})
    dump(args.config, {"methods": [{"id": name} for name in methods],
                      "inference": {"presence_score_space": "logit_margin"}})
    dump(args.selection, {"presence_score_space": "logit_margin",
        "input_sha256": {"artifacts": profile.sha256(args.artifacts)},
        "thresholds": {name: {"presence_score_space": "logit_margin", "aspect_threshold": .5,
                              "presence_margin_threshold": 1., "runner_up_threshold": .4}
                       for name in methods}})
    rows = []
    for i, key in enumerate(keys):
        row = record(key, seen=.1 if i == 2 else .9, heldout=.9 if i == 1 else .1, success=i != 3)
        row["elapsed_ms"] = 10 * (i + 1)
        response = row["response"]
        response.update(variant="hybrid", presence_score_space="logit_margin")
        for item in response["aspect_scores"]:
            item["presence_logit_margin"] = math.log(item["score"] / (1 - item["score"]))
        response["sentiment_truncation_by_aspect"] = {
            "seen": {"left_truncated": i in (0, 2), "original_prompt_tokens": 400, "retained_prompt_tokens": 384},
            "heldout": {"left_truncated": i == 1, "original_prompt_tokens": 400, "retained_prompt_tokens": 384}}
        rows.append(row)
    for name in methods:
        lines(args.scores / f"{name}.jsonl", rows)
    dump(args.scores / "identity.json", {"replay": profile.sha256(args.replay),
        "artifacts": profile.sha256(args.artifacts), "config": profile.sha256(args.config),
        "request_ids": keys, "methods": methods})
    dump(args.scores / "state.json", {"status": "complete", "methods": methods, "reviews": 4})
    dump(args.scores / "summary.json", {"reviews": 4, "methods": {
        name: {"successful": 3, "file_sha256": profile.sha256(args.scores / f"{name}.jsonl")}
        for name in methods}})
    lines(args.historical_labels, [{"request_id": key, "pairs":
        ([{"aspect": "seen", "sentiment": "negative"}] if i != 2 else []) +
        ([{"aspect": "heldout", "sentiment": "negative"}] if i == 1 else [])}
        for i, key in enumerate(keys)])
    dump(args.historical_manifest, {"source_split": "validation", "blind_evaluation": False,
        "splits": {"evaluation": {"count": 4, "files": {
            "replay": {"sha256": profile.sha256(args.replay)},
            "labels": {"sha256": profile.sha256(args.historical_labels)}}}}})
    return args


def guard_gold(monkeypatch, args):
    original = profile.read_jsonl
    gold_path = args.historical_labels
    def guarded(path):
        if path == gold_path:
            pytest.fail("Gold parsed before the invalid evidence was rejected")
        return original(path)
    monkeypatch.setattr(profile, "read_jsonl", guarded)


def refresh_score_hash(args, method="e5"):
    path = args.scores / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["methods"][method]["file_sha256"] = profile.sha256(args.scores / f"{method}.jsonl")
    dump(path, summary)


def test_historical_profile_keeps_failure_denominator_and_selected_stage2_scope(tmp_path):
    args = setup_profile(tmp_path)
    result = profile.run(args)
    overall = result["methods"]["fixed"]["all"]
    assert overall["registered_reviews"] == 4 and overall["failed_or_missing_reviews"] == 1
    assert overall["quality"]["scopes"]["overall"]["pair"]["micro_f1"] == pytest.approx(6 / 7)
    assert overall["exact_set_accuracy_all_attempts"] == .75
    assert overall["successful_presence_stage_elapsed_ms"] == {"count": 3, "median": 20., "p95": 29., "maximum": 30.}
    assert overall["full_grid_sentiment_audited_prompts"] == 6
    assert overall["full_grid_sentiment_left_truncated_prompts"] == 3
    assert overall["selected_sentiment_audited_prompts"] == 3
    assert overall["selected_sentiment_left_truncated_prompts"] == 2
    assert overall["selected_sentiment_left_truncated_reviews"] == 2
    groups = result["methods"]["fixed"]["length_groups"]
    assert [groups[band]["registered_reviews"] for band in profile.BANDS] == [1, 1, 1, 1]
    assert groups["301_to_1000"]["selected_sentiment_left_truncated_reviews"] == 0
    assert groups["over_1000"]["quality"]["scopes"]["seen"]["pair"]["fn"] == 1


def test_labelblind_has_no_gold_quality_even_with_failed_records(tmp_path, monkeypatch):
    args = setup_profile(tmp_path)
    guard_gold(monkeypatch, args)
    args.scope = "new_source_labelblind"
    args.historical_labels = args.historical_manifest = None
    result = profile.run(args)
    for method in result["methods"].values():
        assert "quality_by_aspect" not in method
        assert "quality" not in method["all"]
        assert "exact_set_accuracy_all_attempts" not in method["all"]
        assert method["all"]["failed_or_missing_reviews"] == 1


@pytest.mark.parametrize("fault", ["config", "methods", "population", "successful", "file_hash", "selection_artifact"])
def test_matrix_provenance_failures_are_rejected_before_gold(tmp_path, monkeypatch, fault):
    args = setup_profile(tmp_path)
    guard_gold(monkeypatch, args)
    if fault == "config":
        path = args.scores / "identity.json"
        value = json.loads(path.read_text())
        value["config"] = "other-config"
    elif fault == "methods":
        path = args.scores / "state.json"
        value = json.loads(path.read_text())
        value["methods"] = ["fixed"]
    elif fault == "population":
        path = args.scores / "summary.json"
        value = json.loads(path.read_text())
        value["reviews"] = 3
    elif fault == "successful":
        path = args.scores / "summary.json"
        value = json.loads(path.read_text())
        value["methods"]["e5"]["successful"] = 4
    elif fault == "selection_artifact":
        path = args.selection
        value = json.loads(path.read_text())
        value["input_sha256"]["artifacts"] = "other-artifact"
    else:
        path = args.scores / "e5.jsonl"
        path.write_text(path.read_text() + '\n')
        value = None
    if value is not None:
        dump(path, value)
    with pytest.raises(ValueError):
        profile.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("fault", ["missing_grid", "invalid_margin", "bad_normalization", "missing_review", "threshold_space"])
def test_bad_success_scores_and_thresholds_block_gold_even_with_current_hash(tmp_path, monkeypatch, fault):
    args = setup_profile(tmp_path)
    guard_gold(monkeypatch, args)
    if fault == "threshold_space":
        selection = json.loads(args.selection.read_text())
        selection["thresholds"]["e5"]["presence_score_space"] = "probability"
        dump(args.selection, selection)
    else:
        path = args.scores / "e5.jsonl"
        rows = profile.read_jsonl(path)
        if fault == "missing_grid":
            rows[0]["response"]["full_grid_scores"].pop()
        elif fault == "invalid_margin":
            rows[0]["response"]["aspect_scores"][0]["presence_logit_margin"] = float("nan")
        elif fault == "bad_normalization":
            rows[0]["response"]["full_grid_scores"][0]["sentiment_scores"]["negative"] = .7
        else:
            rows.pop()
        lines(path, rows)
        refresh_score_hash(args)
    with pytest.raises(ValueError):
        profile.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("fault", ["count", "replay_sha", "labels_sha", "source_split", "blind", "missing_manifest"])
def test_historical_manifest_failures_block_gold(tmp_path, monkeypatch, fault):
    args = setup_profile(tmp_path)
    guard_gold(monkeypatch, args)
    manifest = json.loads(args.historical_manifest.read_text())
    if fault == "count":
        manifest["splits"]["evaluation"]["count"] = 3
    elif fault in ("replay_sha", "labels_sha"):
        name = fault.removesuffix("_sha")
        manifest["splits"]["evaluation"]["files"][name]["sha256"] = "different-source"
    elif fault == "source_split":
        manifest["source_split"] = "test"
    elif fault == "blind":
        manifest["blind_evaluation"] = True
    dump(args.historical_manifest, manifest)
    if fault == "missing_manifest":
        args.historical_manifest = None
    with pytest.raises(ValueError):
        profile.run(args)


@pytest.mark.parametrize("fault", ["label_argument", "labels_in_replay"])
def test_labelblind_rejects_label_channels(tmp_path, monkeypatch, fault):
    args = setup_profile(tmp_path)
    guard_gold(monkeypatch, args)
    args.scope = "new_source_labelblind"
    if fault == "labels_in_replay":
        rows = profile.read_jsonl(args.replay)
        rows[0]["pairs"] = []
        lines(args.replay, rows)
        identity_path = args.scores / "identity.json"
        identity = json.loads(identity_path.read_text())
        identity["replay"] = profile.sha256(args.replay)
        dump(identity_path, identity)
        args.historical_labels = args.historical_manifest = None
    with pytest.raises(ValueError):
        profile.run(args)
