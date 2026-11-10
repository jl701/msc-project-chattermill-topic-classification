"""Validate the fault harness without loading weights or using a GPU."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from check_rag_faults_gpu import (
    check_policy, compare_sentiment_anchors, make_small_versioned_index, select_inputs,
    sha_file, validate_fixed_source, fault_retrieval_settings, fault_sentiment_execution,
    compare_canonical_conditional,
)
from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration
from msc_project.serving.retrieval import DemonstrationRetriever, digest
from msc_project.serving.sentiment_execution import build_sentiment_group_contract


def test_length_selection_rejects_labels_and_duplicate_identities(tmp_path):
    replay = tmp_path / "input.jsonl"
    rows = [{"request_id": str(i), "text": "word " * (i+1)} for i in range(7)]
    replay.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    assert [r["request_id"] for r in select_inputs(replay)] == ["0", "3", "6"]
    rows[0]["labels"] = []
    replay.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="input-only"):
        select_inputs(replay)


def test_policy_threshold_and_frozen_resource_must_match_hashed_fixed_source(tmp_path):
    fixed = {"service_calibration": {"method": "fixed"}, "fold_id": "f",
             "model": {"revision": "frozen"}, "few_shot": {"max_length": 1024},
             "variants": {"hybrid": {"aspect_threshold": .4, "runner_up_threshold": .3}}}
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
    dynamic = deepcopy(fixed)
    dynamic["service_calibration"]["method"] = "bm25"
    dynamic["variants"]["hybrid"]["aspect_threshold"] = .6
    dynamic["retrieval_fallback_policy"] = {
        "method": "fixed", "variant": "hybrid", "aspect_threshold": .4, "runner_up_threshold": .3,
        "source_artifacts_path": str(fixed_path), "source_artifacts_sha256": sha_file(fixed_path)}
    assert validate_fixed_source(dynamic, fixed, fixed_path)["aspect_threshold"] == .4
    for field, value in [("aspect_threshold", .6), ("source_artifacts_sha256", "bad")]:
        altered = deepcopy(dynamic)
        altered["retrieval_fallback_policy"][field] = value
        with pytest.raises(ValueError):
            validate_fixed_source(altered, fixed, fixed_path)
    altered = deepcopy(dynamic)
    altered["few_shot"]["max_length"] = 512
    with pytest.raises(ValueError, match="frozen inference resource"):
        validate_fixed_source(altered, fixed, fixed_path)


def test_subset_index_changes_identity_preserves_fixed_and_embedding_alignment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pool = [{"row_uid": f"row-{i}", "candidate_aspect": "A", "review_text": f"Legal train review {i}",
             "aspect_candidate": "A card", "answer": "Y" if i < 12 else "N"} for i in range(24)]
    fixed = [TwoStageDemonstration(**pool[i]) for i in (10,11,22,23)]
    embeddings = np.arange(48, dtype=np.float32).reshape(24,2)
    np.save(source / "embeddings.npy", embeddings)
    original = SimpleNamespace(fixed=fixed, pool=pool, index_dir=source,
        fingerprint=digest(pool), manifest={"schema_version": "test", "pool_sha256": digest(pool),
                                          "heldout_aspect": "H"})
    versioned_path = tmp_path / "versioned"
    manifest = make_small_versioned_index(original, versioned_path)
    rows = [json.loads(line) for line in (versioned_path / "pool.jsonl").read_text().splitlines()]
    loaded_embeddings = np.load(versioned_path / "embeddings.npy")
    assert len(rows) == 20
    assert manifest["pool_sha256"] != original.fingerprint
    assert all(asdict(d) in rows for d in fixed)
    for i,row in enumerate(rows):
        np.testing.assert_array_equal(loaded_embeddings[i], embeddings[int(row["row_uid"].split("-")[1])])
    # The concrete retriever accepts this version, while a changed hash is rejected.
    actual = DemonstrationRetriever(versioned_path, method="bm25", fixed_demonstrations=fixed)
    assert actual.fingerprint == manifest["pool_sha256"]
    manifest["pool_sha256"] = "bad"
    (versioned_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        DemonstrationRetriever(versioned_path, method="bm25", fixed_demonstrations=fixed)


def test_fault_response_gate_rejects_mixed_policy_and_wrong_thresholds():
    thresholds = {"aspect_threshold": .4, "runner_up_threshold": .3}
    diagnostics = {"retrieval_fallback": True, "effective_retrieval_method": "fixed",
        "effective_thresholds": thresholds, "fallback_policy_source_sha256": "source",
        "batch_aspect": {"retrieval": [{"fallback": True, "error_type": "OSError"}]}}
    response = {"diagnostics": diagnostics}
    check_policy([response], fallback=True, method="fixed", thresholds=thresholds,
                 source_sha="source", error_type="OSError")
    diagnostics["batch_aspect"]["retrieval"].append({"fallback": False})
    with pytest.raises(AssertionError, match="Mixed"):
        check_policy([response], fallback=True, method="fixed", thresholds=thresholds, source_sha="source")
    diagnostics["effective_thresholds"] = {"aspect_threshold": .6, "runner_up_threshold": .3}
    with pytest.raises(AssertionError, match="thresholds"):
        check_policy([response], fallback=True, method="fixed", thresholds=thresholds, source_sha="source")


def test_fault_method_id_locks_new_policy_and_rejects_mismatched_clone_or_override():
    settings = {"method": "e5", "seed": 13, "selection_policy": "same_aspect_negative_seen"}
    artifacts = {"service_calibration": {"method": "e5_negative_seen", "retrieval": settings}}
    assert fault_retrieval_settings("e5_negative_seen", artifacts) == settings
    assert fault_retrieval_settings("e5_negative_seen", artifacts,
        selection_policy="same_aspect_negative_seen", seed=13) == settings
    with pytest.raises(ValueError, match="selection policy conflicts"):
        fault_retrieval_settings("e5_negative_seen", artifacts, selection_policy="global")
    with pytest.raises(ValueError, match="seed conflicts"):
        fault_retrieval_settings("e5_negative_seen", artifacts, seed=37)
    with pytest.raises(ValueError, match="method ID differs"):
        fault_retrieval_settings("e5", artifacts)
    bad = deepcopy(artifacts)
    bad["service_calibration"]["retrieval"]["selection_policy"] = "global"
    with pytest.raises(ValueError, match="settings differ"):
        fault_retrieval_settings("e5_negative_seen", bad)
    legacy = {"service_calibration": {"method": "bm25"}}
    assert fault_retrieval_settings("bm25", legacy)["selection_policy"] == "global"


def test_fault_execution_is_identical_for_dynamic_and_fixed_clones():
    candidates = [{"aspect": "A"}]
    contract = build_sentiment_group_contract(candidates, execution="canonical_full_grid_groups",
        pair_batch_size=4, max_batch_tokens=4096, logits_mode="last_token")
    policy = {"taxonomy": {"candidates": candidates}, "service_execution": {
        "sentiment_execution": "canonical_full_grid_groups", "sentiment_group_contract": contract}}
    assert fault_sentiment_execution(policy, policy) == "canonical_full_grid_groups"
    assert fault_sentiment_execution({}, {}) == "selected_only"
    with pytest.raises(ValueError, match="differ in sentiment"):
        fault_sentiment_execution(policy, {})
    with pytest.raises(ValueError, match="override differs"):
        fault_sentiment_execution(policy, policy, "selected_only")
    altered = deepcopy(policy)
    altered["service_execution"]["sentiment_group_contract"]["pair_batch_size"] = 8
    with pytest.raises(ValueError, match="group contract"):
        fault_sentiment_execution(altered, policy)


def test_canonical_fault_gate_requires_exact_vectors_even_if_labels_are_unchanged():
    sentiment = {"negative": .8, "neutral": .1, "positive": .1}
    contract = build_sentiment_group_contract([{"aspect": "A"}, {"aspect": "B"}],
        execution="canonical_full_grid_groups", pair_batch_size=4, max_batch_tokens=4096, logits_mode="last_token")
    response = {"request_id": "a", "aspect_scores": [{"aspect": "A", "score": 1., "presence_logit_margin": 30.}],
        "predictions": [{"aspect": "A", "sentiments": ["negative"], "sentiment_scores": sentiment}],
        "diagnostics": {"sentiment_execution": "canonical_full_grid_groups", "sentiment_group_contract": contract}}
    full = {"responses": [{**deepcopy(response), "full_grid_scores": [
        {"aspect": "A", "sentiment_scores": deepcopy(sentiment)},
        {"aspect": "B", "sentiment_scores": deepcopy(sentiment)}]}]}
    conditional = {"responses": [deepcopy(response)]}
    assert compare_canonical_conditional(full, conditional)["passed"]
    conditional["responses"][0]["predictions"][0]["sentiment_scores"]["negative"] += .00001
    result = compare_canonical_conditional(full, conditional)
    assert result["label_sets_equal"]
    assert not result["selected_sentiment_vectors_exact"]
    assert not result["passed"]


def test_fault_gate_requires_new_policy_only_for_seen_candidates():
    thresholds = {"aspect_threshold": .4, "runner_up_threshold": .3}
    diagnostics = {"retrieval_fallback": False, "effective_retrieval_method": "e5",
        "effective_retrieval_selection_policy": "same_aspect_negative_seen",
        "effective_thresholds": thresholds, "batch_aspect": {"retrieval": [
            {"candidate_aspect": "A", "effective_selection_policy": "same_aspect_negative_seen"},
            {"candidate_aspect": "H", "effective_selection_policy": "global"}]}}
    check_policy([{"diagnostics": diagnostics}], fallback=False, method="e5", thresholds=thresholds,
                 selection_policy="same_aspect_negative_seen", heldout_aspect="H")
    diagnostics["batch_aspect"]["retrieval"][1]["effective_selection_policy"] = "same_aspect_negative_seen"
    with pytest.raises(AssertionError, match="per-candidate"):
        check_policy([{"diagnostics": diagnostics}], fallback=False, method="e5", thresholds=thresholds,
                     selection_policy="same_aspect_negative_seen", heldout_aspect="H")


def test_versioned_fault_subset_keeps_negative_support_for_each_seen_aspect(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pool = [{"row_uid": f"row-{aspect}-{answer}-{i}", "candidate_aspect": aspect,
             "review_text": f"Legal {aspect} {answer} training review {i}",
             "aspect_candidate": f"{aspect} card", "answer": answer}
            for aspect in ("A", "B", "C") for answer in ("Y", "N") for i in range(12)]
    fixed = [TwoStageDemonstration(**pool[i]) for i in (10,11,22,23)]
    original = SimpleNamespace(fixed=fixed, pool=pool, index_dir=source,
        fingerprint=digest(pool), manifest={"schema_version": "test", "pool_sha256": digest(pool),
                                          "heldout_aspect": "H"})
    target = tmp_path / "subset"
    make_small_versioned_index(original, target)
    rows = [json.loads(line) for line in (target / "pool.jsonl").read_text().splitlines()]
    for aspect in ("A", "B", "C"):
        assert sum(row["candidate_aspect"] == aspect and row["answer"] == "N" for row in rows) >= 8
    assert len(rows) < len(pool)


def test_sentiment_anchor_gate_compares_all_vectors_inputs_and_decoded_labels():
    row = {"request_id":"one", "full_grid_scores":[{"aspect":"A",
        "sentiment_scores":{"negative":.6, "neutral":.3, "positive":.1}}],
        "sentiment_truncation_by_aspect":{"A":{"left_truncated":False}}}
    sentiment_trace = {"mode":"sentiment", "adapter":True, "encoded_inputs_sha256":"same", "prompt_count":1}
    reference = {"responses":[row], "encoded_inputs":[{"mode":"aspect"}, sentiment_trace]}
    candidate = {"responses":[deepcopy(row)], "encoded_inputs":[deepcopy(sentiment_trace)]}
    gate = compare_sentiment_anchors(reference, candidate, [.25])
    assert gate["encoded_sentiment_inputs_equal"] and gate["sentiment_truncation_equal"]
    assert gate["max_abs_score_difference"] == 0 and gate["label_changed_aspects"] == 0
    candidate["responses"][0]["full_grid_scores"][0]["sentiment_scores"]["neutral"] = .2
    gate = compare_sentiment_anchors(reference, candidate, [.25])
    assert gate["label_changed_aspects"] == 1 and gate["max_abs_score_difference"] > 0
    candidate["encoded_inputs"].append({"mode":"aspect"})
    with pytest.raises(AssertionError, match="Stage1"):
        compare_sentiment_anchors(reference, candidate, [.25])
