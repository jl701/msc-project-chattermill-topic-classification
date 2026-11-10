"""Validate the fault harness without loading weights or using a GPU."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from check_rag_faults_gpu import (
    check_policy, compare_sentiment_anchors, make_small_versioned_index, select_inputs,
    sha_file, validate_fixed_source,
)
from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration
from msc_project.serving.retrieval import DemonstrationRetriever, digest


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
