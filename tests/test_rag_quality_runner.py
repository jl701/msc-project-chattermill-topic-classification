from copy import deepcopy
import json

import pytest

import run_rag_quality as runner


def registration():
    config = {"fold":"f", "condition":"D", "model_id":"q", "model_revision":"r", "variant":"hybrid",
              "heldout_aspect":"h", "inference":{"fewshot_max_length":1024,"qlora_max_length":384,"quality_prefix_cache":False},
              "retrieval":{"bm25_k1":1.2,"bm25_b":.75,"rrf_c":60,"rrf_weights":[1,1],
                           "shortlist_per_class":256,"model_max_length":256,"e5_device":"cpu","e5_revision":"e"}}
    artifacts = {"fold_id":"f","condition":"D","model":{"model_id":"q","revision":"r"},
                 "few_shot":{"max_length":1024},"qlora":{"max_length":384}}
    manifest = {"e5_revision":"e","heldout_aspect":"h","exclusion_source_hashes":{"replay":"s"}}
    return config, artifacts, manifest


def test_configuration_and_dedup_coverage_gate():
    c,a,m = registration()
    runner.validate_registration(c,a,m,"s")
    with pytest.raises(ValueError,match="duplicate-exclusion"):
        runner.validate_registration(c,a,m,"new")
    a["few_shot"]["max_length"] = 512
    with pytest.raises(ValueError,match="fewshot_max_length"):
        runner.validate_registration(c,a,m,"s")


def test_anchor_duplicate_mismatch_and_partial_grid_rejected():
    row = {"request_id":"a","status":"ok","response":{"request_id":"a","full_grid_scores":[{"aspect":"x"}]}}
    assert runner.checked_anchors([row],{"a"},{"x"})["a"] == row
    with pytest.raises(ValueError,match="duplicate"):
        runner.checked_anchors([row,row],{"a"},{"x"})
    wrong = deepcopy(row)
    wrong["response"]["request_id"] = "b"
    with pytest.raises(ValueError,match="mismatched"):
        runner.checked_anchors([wrong],{"a"},{"x"})
    with pytest.raises(ValueError,match="Incomplete"):
        runner.checked_anchors([row],{"a"},{"x","y"})


def test_failure_state_written_only_for_accepted_run(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"_ACTIVE_OUTPUT",None)
    runner.record_failure(RuntimeError("synthetic"))
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(runner,"_ACTIVE_OUTPUT",tmp_path)
    runner.record_failure(RuntimeError("synthetic"))
    assert json.loads((tmp_path/"state.json").read_text())["status"] == "failed"
