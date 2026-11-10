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


def test_margin_runtime_artifact_preserves_old_threshold_and_separate_reference():
    original = {"variants":{"hybrid":{"aspect_threshold":.75,"runner_up_threshold":.2}}}
    result = runner.runtime_artifacts(original,{"inference":{"presence_score_space":"logit_margin"}})
    assert result["variants"]["hybrid"]["aspect_threshold"] == .75
    assert result["variants"]["hybrid"]["presence_margin_threshold"] > 1.
    assert "presence_score_space" not in original["variants"]["hybrid"]


def test_anchor_reuse_checks_source_identity_stage2_settings_and_pinned_hash(tmp_path):
    c = {"inference":{"pair_batch_size":4,"max_batch_tokens":4096,"logits_mode":"last_token"}}
    identity = {"artifacts":"a","replay":"r","request_ids":["one"],"inference_code":{
        "src/msc_project/llm/qwen_two_stage_classifier.py":runner.digest(runner.ROOT/"src/msc_project/llm/qwen_two_stage_classifier.py")}}
    runtime = {"engine":{"variant":"hybrid","candidate_count":1,"prefix_cache_enabled":False,
        "artifacts_sha256":"a",**c["inference"]}}
    for name,value in [("identity",identity),("runtime",runtime),("state",{"status":"complete"})]:
        (tmp_path/f"{name}.json").write_text(json.dumps(value),encoding="utf-8")
    row = {"request_id":"one","status":"ok","response":{"request_id":"one",
        "full_grid_scores":[{"aspect":"A","sentiment_scores":{"negative":.2,"neutral":.3,"positive":.5}}],
        "sentiment_truncation_by_aspect":{"A":{"left_truncated":False}},
        "diagnostics":{"batch_sentiment":{"adapter_enabled":True,"prompt_count":1}}}}
    anchor = tmp_path/"_sentiment_anchor.jsonl"
    anchor.write_text(json.dumps(row)+"\n",encoding="utf-8")
    kw = dict(artifacts_sha="a",replay_sha="r",expected={"one"},aspects=["A"],config=c,required_sha=runner.digest(anchor))
    values,receipt = runner.verified_anchor_reuse(anchor,**kw)
    assert values["one"] == row and receipt["reused_reviews"] == 1
    with pytest.raises(ValueError,match="SHA"):
        runner.verified_anchor_reuse(anchor,**{**kw,"required_sha":"bad"})
    runtime["engine"]["pair_batch_size"] = 8
    (tmp_path/"runtime.json").write_text(json.dumps(runtime),encoding="utf-8")
    with pytest.raises(ValueError,match="Stage2 setting"):
        runner.verified_anchor_reuse(anchor,**kw)
