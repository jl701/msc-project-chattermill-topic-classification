from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from msc_project.llm.qwen_two_stage_classifier import score_two_stage_prompts
from msc_project.serving.inference import (
    QwenServingEngine,
    decode_sentiments,
    plan_microbatches,
)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(message["content"] for message in messages) + " assistant"

    def __call__(self, text, **kwargs):
        fixed = {"Y": [7], "N": [8], "A": [10], "B": [11], "C": [12]}
        return {"input_ids": fixed.get(text, [20 + sum(map(ord, word)) % 90 for word in text.split()])}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.adapter_enabled = True
        self.trace = []

    @contextmanager
    def disable_adapter(self):
        old = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = old

    def forward(self, input_ids, attention_mask, use_cache=None, logits_to_keep=None):
        self.trace.append(self.adapter_enabled)
        prefix = (input_ids * attention_mask).cumsum(dim=-1).float()
        logits = torch.zeros((*input_ids.shape, 120), device=input_ids.device)
        logits[:, :, 7] = prefix.remainder(5) - 2
        logits[:, :, 8] = 0
        logits[:, :, 10] = prefix.remainder(3)
        logits[:, :, 11] = 1.5
        logits[:, :, 12] = prefix.remainder(7) / 2 + float(self.adapter_enabled)
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep, :]
        return SimpleNamespace(logits=logits)


@pytest.fixture
def artifacts():
    demonstrations = {}
    for mode, answers in [("aspect", "YYNN"), ("sentiment", "ABC")]:
        demonstrations[mode] = [
            {"row_uid": f"{mode}-{index}", "candidate_aspect": "Training topic",
             "review_text": "Example training review.", "aspect_candidate": "Training topic",
             "answer": answer}
            for index, answer in enumerate(answers)
        ]
    return {
        "fold_id": "l2-test",
        "model": {"model_id": "test", "revision": "test"},
        "taxonomy": {"candidates": [{"aspect": "one", "text": "First topic"},
                                    {"aspect": "two", "text": "Second topic longer"}]},
        "few_shot": {"max_length": 1024, "demonstrations": demonstrations},
        "qlora": {"max_length": 384},
        "variants": {variant: {"aspect_threshold": 0.5, "runner_up_threshold": 0.25}
                     for variant in ["few_shot", "qlora", "hybrid"]},
    }


def predictions_only(responses):
    return [{"predictions": response["predictions"], "aspect_scores": response["aspect_scores"]}
            for response in responses]


def configured_fallback(artifacts, tmp_path, *, fixed_thresholds=None):
    fixed = deepcopy(artifacts)
    fixed.pop("retrieval_fallback_policy", None)
    fixed["service_calibration"] = {"method":"fixed"}
    if fixed_thresholds:
        fixed["variants"]["hybrid"].update(fixed_thresholds)
    path = tmp_path / "fixed_artifacts.json"
    path.write_text(json.dumps(fixed), encoding="utf-8")
    dynamic = deepcopy(artifacts)
    dynamic["service_calibration"] = {"method":"bm25"}
    dynamic["retrieval_fallback_policy"] = {"method":"fixed", "variant":"hybrid",
        **fixed["variants"]["hybrid"], "source_artifacts_path":str(path),
        "source_artifacts_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    return dynamic, fixed


def test_microbatch_budget_uses_padded_tokens_and_preserves_order():
    assert plan_microbatches([3, 7, 2, 8], pair_batch_size=4, max_batch_tokens=16) == [[0, 1], [2, 3]]
    with pytest.raises(ValueError, match="budget"):
        plan_microbatches([17], pair_batch_size=4, max_batch_tokens=16)


def test_decoder_keeps_stable_ties_inclusive_boundary_and_never_three():
    assert decode_sentiments([0.4, 0.4, 0.2], 0.4) == ["negative", "neutral"]
    assert decode_sentiments([0.4, 0.4, 0.2], 0.40001) == ["negative"]
    assert len(decode_sentiments([1/3, 1/3, 1/3], 0.0)) == 2


@pytest.mark.parametrize("variant", ["few_shot", "qlora", "hybrid"])
def test_conditional_execution_matches_full_grid_and_model_state_restores(artifacts, variant):
    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant=variant, pair_batch_size=2)
    requests = [{"request_id": "a", "text": "A useful review"},
                {"request_id": "b", "text": "A longer review with a different length"}]
    lazy = engine.predict_batch(requests)
    full = engine.predict_batch(requests, full_grid=True)
    assert predictions_only(lazy) == predictions_only(full)
    for response in lazy:
        assert response["diagnostics"]["sentiment_prompt_count"] == len(response["predictions"])
    assert all(response["diagnostics"]["sentiment_prompt_count"] == 2 for response in full)
    assert engine.model.adapter_enabled
    if variant == "few_shot":
        assert not any(engine.model.trace)
    elif variant == "qlora":
        assert all(engine.model.trace)


@pytest.mark.parametrize("variant", ["few_shot", "qlora", "hybrid"])
def test_reduced_lm_head_matches_original_full_logits_with_unequal_lengths(artifacts, variant):
    requests = [{"request_id": "a", "text": "Short."},
                {"request_id": "b", "text": "This review has several more words."}]
    full = QwenServingEngine(artifacts, Tokenizer(), Model(), variant=variant, logits_mode="full")
    reduced = QwenServingEngine(artifacts, Tokenizer(), Model(), variant=variant, logits_mode="last_token")
    assert predictions_only(full.predict_batch(requests)) == predictions_only(reduced.predict_batch(requests))


def test_full_path_matches_historical_scorer_and_individual_requests(artifacts):
    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant="qlora", pair_batch_size=3)
    requests = [{"request_id": "a", "text": "First review"}, {"request_id": "b", "text": "Longer second review example"}]
    batch = engine.predict_batch(requests)
    separate = [engine.predict_batch([request])[0] for request in requests]
    assert predictions_only(batch) == predictions_only(separate)
    historical = score_two_stage_prompts(
        engine.model, engine.tokenizer,
        [request["text"] for request in requests for _ in engine.candidates],
        [candidate["text"] for _ in requests for candidate in engine.candidates],
        mode="aspect", max_length=384, batch_size=3,
    )[:, 0]
    actual = [item["score"] for response in batch for item in response["aspect_scores"]]
    np.testing.assert_array_equal(historical, actual)


def test_no_aspects_selected_skips_all_sentiment_forwards(artifacts):
    config = deepcopy(artifacts)
    config["variants"]["hybrid"]["aspect_threshold"] = 2.0
    engine = QwenServingEngine(config, Tokenizer(), Model(), variant="hybrid")
    response = engine.predict_batch([{"request_id": "a", "text": "No topic"}])[0]
    assert response["predictions"] == []
    assert response["diagnostics"]["batch_sentiment"]["prompt_count"] == 0
    assert response["diagnostics"]["batch_sentiment"]["microbatch_count"] == 0
    assert not any(engine.model.trace)


def test_presence_only_and_full_grid_allow_threshold_replay(artifacts):
    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant="hybrid")
    requests = [{"request_id":"x", "text":"Example current review"}]
    presence = engine.score_presence(requests)[0]
    response = engine.predict_batch(requests, full_grid=True)[0]
    assert presence["aspect_scores"] == response["aspect_scores"]
    assert {r["aspect"] for r in response["full_grid_scores"]} == {"one","two"}
    assert all(set(r["sentiment_scores"]) == {"negative","neutral","positive"}
               for r in response["full_grid_scores"])
    with pytest.raises(ValueError, match="labels"):
        engine.score_presence([{"request_id":"x", "text":"x", "labels":[]}])


def test_dynamic_retrieval_bypasses_fixed_prefix_and_fallback_is_explicit(artifacts, monkeypatch, tmp_path):
    artifacts, _ = configured_fallback(artifacts, tmp_path)
    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant="hybrid", enable_prefix_cache=True)
    class Retriever:
        method, seed, fingerprint, prepare_ms = "random", 13, "test-fingerprint", 0.0
        fixed = engine.demonstrations["aspect"]
        fail = False
        def metadata(self):
            return {"method":self.method, "pool_sha256":self.fingerprint}
        def prepare(self,texts,cards):
            if self.fail:
                raise OSError("Synthetic unavailable index")
        def select(self,text,aspect,card):
            return self.fixed, {"fallback":False,"retrieval_ms":0.0}
    retriever = Retriever()
    engine.set_retriever(retriever)
    def forbidden_cache(*args, **kwargs):
        raise AssertionError("Dynamic aspect prompts must not use frozen-demo prefix")
    monkeypatch.setattr(engine.prefix_cache,"score_if_compatible",forbidden_cache)
    request = [{"request_id":"x","text":"Current review"}]
    out = engine.predict_batch(request)[0]
    assert not any(r["fallback"] for r in out["diagnostics"]["batch_aspect"]["retrieval"])
    assert "rag-random" in out["model_version"]
    retriever.selection_policy = "same_aspect_negative_seen"
    policy_version = engine.model_version()
    assert "same_aspect_negative_seen" in policy_version
    assert engine._effective_policy({},"hybrid")["effective_retrieval_selection_policy"] == "same_aspect_negative_seen"
    retriever.fail = True
    with pytest.raises(OSError):
        engine.score_presence(request)
    engine.set_retriever(retriever, fallback_on_error=True)
    out = engine.score_presence(request)[0]
    assert all(r["fallback"] for r in out["diagnostics"]["batch_aspect"]["retrieval"])
    assert out["diagnostics"]["effective_retrieval_selection_policy"] == "global"
    assert all(r["effective_selection_policy"] == "global" for r in out["diagnostics"]["batch_aspect"]["retrieval"])


def test_late_selection_failure_switches_all_aspects_and_both_thresholds(artifacts, tmp_path):
    from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration
    from msc_project.serving.inference import UnsafeFixedFallbackError

    config = deepcopy(artifacts)
    config["variants"]["hybrid"].update(aspect_threshold=1.0, runner_up_threshold=1.0)
    dynamic_config, fixed_config = configured_fallback(config, tmp_path,
        fixed_thresholds={"aspect_threshold":0.0, "runner_up_threshold":0.2})
    engine = QwenServingEngine(dynamic_config, Tokenizer(), Model(), variant="hybrid")
    baseline = QwenServingEngine(fixed_config, Tokenizer(), Model(), variant="hybrid")
    class Retriever:
        method, seed, fingerprint = "bm25", 13, "pool"
        fixed = engine.demonstrations["aspect"]
        calls = 0
        def metadata(self):
            return {"method":self.method, "pool_sha256":self.fingerprint}
        def prepare(self, texts, cards):
            self.calls = 0
        def select(self, text, aspect, card):
            self.calls += 1
            if self.method == "fixed":
                return self.fixed, {"fallback":False, "retrieval_ms":0.}
            if self.calls == 2:
                raise OSError("Failure after one successful selection")
            changed = [TwoStageDemonstration(d.row_uid,d.candidate_aspect,"different retrieved words",d.aspect_candidate,d.answer)
                       for d in self.fixed]
            return changed, {"fallback":False,"retrieval_ms":0.}
    dynamic = Retriever()
    fixed = Retriever()
    fixed.method = "fixed"
    engine.set_retriever(dynamic, fallback_on_error=True)
    baseline.set_retriever(fixed)
    requests = [{"request_id":"x", "text":"Current review"}]
    expected, actual = baseline.predict_batch(requests), engine.predict_batch(requests)
    assert predictions_only(expected) == predictions_only(actual)
    report = actual[0]["diagnostics"]
    assert report["retrieval_fallback"] and report["effective_retrieval_method"] == "fixed"
    assert report["effective_thresholds"] == {"presence_score_space":"probability", **fixed_config["variants"]["hybrid"]}
    assert report["fallback_policy_source_sha256"] == dynamic_config["retrieval_fallback_policy"]["source_artifacts_sha256"]
    assert all(r["fallback"] and r["error_type"] == "OSError" for r in report["batch_aspect"]["retrieval"])
    assert len(actual[0]["predictions"]) == 2  # Dynamic threshold1 would select none.
    engine.model.trace.clear()
    with pytest.raises(UnsafeFixedFallbackError):
        engine.predict_batch([{"request_id":"one", "text":"Current review"},
                              {"request_id":"two", "text":engine.demonstrations["aspect"][0].review_text.upper()+"!"}])
    assert not engine.model.trace
    with pytest.raises(ValueError, match="variant"):
        engine.set_variant("few_shot")


@pytest.mark.parametrize("variant", ["few_shot", "qlora", "hybrid"])
def test_full_sentiment_only_matches_legacy_inputs_grid_and_truncation(artifacts, variant, monkeypatch):
    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant=variant)
    requests = [{"request_id":"short", "text":"Short review"},
                {"request_id":"long", "text":"A longer review with some more words "*80}]
    original = engine._score_encoded
    trace = []
    def traced(items, ids, **kwargs):
        trace.append((kwargs["mode"], deepcopy(items), list(ids), kwargs["use_adapter"]))
        return original(items, ids, **kwargs)
    monkeypatch.setattr(engine, "_score_encoded", traced)
    legacy = engine.predict_batch(requests, full_grid=True)
    legacy_sentiment_trace = [r for r in trace if r[0] == "sentiment"]
    trace.clear()
    current = engine.score_full_sentiment(requests)
    assert trace == legacy_sentiment_trace
    assert [r["full_grid_scores"] for r in legacy] == [r["full_grid_scores"] for r in current]
    assert [r["sentiment_truncation_by_aspect"] for r in legacy] == [r["sentiment_truncation_by_aspect"] for r in current]
    assert engine.model.adapter_enabled
    with pytest.raises(ValueError, match="labels"):
        engine.score_full_sentiment([{"request_id":"x", "text":"query", "labels":[]}])


def test_fallback_source_tampering_rejected_before_model_loader(artifacts, tmp_path, monkeypatch):
    from msc_project.llm import qwen_pair_classifier
    config, _ = configured_fallback(artifacts, tmp_path)
    config["retrieval_fallback_policy"]["source_artifacts_sha256"] = "0"*64
    path = tmp_path/"tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid fallback must fail before model loading")
    monkeypatch.setattr(qwen_pair_classifier,"load_saved_qwen_pair_adapter",forbidden)
    with pytest.raises(ValueError, match="SHA"):
        QwenServingEngine.from_artifacts(path)
    config, _ = configured_fallback(artifacts, tmp_path)
    config["retrieval_fallback_policy"]["aspect_threshold"] = .9
    with pytest.raises(ValueError, match="threshold"):
        QwenServingEngine(config, Tokenizer(), object())


@pytest.mark.parametrize("logits_mode", ["full","last_token"])
def test_native_margin_is_unsaturated_and_float64_nextafter_rejects(artifacts,logits_mode):
    class SaturatedModel(Model):
        def forward(self, input_ids, attention_mask, use_cache=None, logits_to_keep=None):
            value = super().forward(input_ids,attention_mask,use_cache,logits_to_keep)
            value.logits[:,:,7] = 30.0
            value.logits[:,:,8] = 1.0
            return value
    config = deepcopy(artifacts)
    config["variants"]["hybrid"].update(presence_score_space="logit_margin",
        presence_margin_threshold=float(np.nextafter(np.float64(29.),np.float64(np.inf))))
    engine = QwenServingEngine(config,Tokenizer(),SaturatedModel(),variant="hybrid",logits_mode=logits_mode)
    response = engine.predict_batch([{"request_id":"x","text":"A query"}])[0]
    assert all(r["score"] == 1.0 and r["presence_logit_margin"] == 29.0 for r in response["aspect_scores"])
    assert response["presence_score_space"] == "logit_margin" and not response["predictions"]
    engine.artifacts["variants"]["hybrid"]["presence_margin_threshold"] = 29.0
    assert len(engine.predict_batch([{"request_id":"x","text":"A query"}])[0]["predictions"]) == 2


def test_margin_fallback_policy_requires_same_space_and_verified_margin(artifacts,tmp_path):
    config = deepcopy(artifacts)
    config["variants"]["hybrid"].update(presence_score_space="logit_margin",presence_margin_threshold=20.0)
    dynamic,fixed = configured_fallback(config,tmp_path,fixed_thresholds={"presence_margin_threshold":30.0})
    engine = QwenServingEngine(dynamic,Tokenizer(),Model())
    assert engine.fallback_policy["presence_margin_threshold"] == 30.0
    altered = deepcopy(dynamic)
    altered["retrieval_fallback_policy"]["presence_margin_threshold"] = 20.0
    with pytest.raises(ValueError,match="margin threshold"):
        QwenServingEngine(altered,Tokenizer(),Model())
    altered = deepcopy(dynamic)
    altered["variants"]["hybrid"]["presence_score_space"] = "probability"
    with pytest.raises(ValueError,match="spaces"):
        QwenServingEngine(altered,Tokenizer(),Model())


def test_adapter_left_truncation_is_audited_without_changing_tokens(artifacts):
    config = deepcopy(artifacts)
    config["qlora"]["max_length"] = 30
    engine = QwenServingEngine(config, Tokenizer(), Model(), variant="hybrid")
    response = engine.predict_batch([{"request_id":"x", "text":"long text "*100}], full_grid=True)[0]
    audit = response["sentiment_truncation_by_aspect"]
    assert set(audit) == {"one","two"}
    assert all(v["left_truncated"] and v["retained_prompt_tokens"] == 30 for v in audit.values())
    assert all(v["original_prompt_tokens"] > 30 for v in audit.values())


def test_request_contract_rejects_labels_empty_text_duplicate_ids(artifacts):
    engine = QwenServingEngine(artifacts, Tokenizer(), Model())
    with pytest.raises(ValueError, match="labels"):
        engine.predict_batch([{"request_id": "a", "text": "X", "labels": []}])
    with pytest.raises(ValueError, match="nonempty review"):
        engine.predict_batch([{"request_id": "a", "text": " "}])
    with pytest.raises(ValueError, match="unique"):
        engine.predict_batch([{"request_id": "a", "text": "X"}] * 2)


def test_parallel_calls_have_same_results_as_serial_and_do_not_change_adapter(artifacts):
    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant="hybrid")
    requests = [[{"request_id": str(i), "text": "Review " + "words " * i}] for i in range(1, 5)]
    serial = [engine.predict_batch(request) for request in requests]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(engine.predict_batch, requests))
    assert [predictions_only(values) for values in serial] == [predictions_only(values) for values in concurrent]
    assert engine.model.adapter_enabled


def test_gpu_gate_historical_decoder_and_score_difference_reporting(artifacts):
    from check_serving_gpu import compare_outputs, historical_reference

    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant="hybrid")
    requests = [{"request_id": "a", "text": "A useful review"}]
    reference = historical_reference(engine, requests)
    response = engine.predict_batch(requests)
    comparison = compare_outputs(reference, response, artifacts["variants"]["hybrid"])
    assert comparison["label_gate_pass"]
    assert comparison["aspect_max_abs_difference"] == 0
    changed = deepcopy(response)
    changed[0]["aspect_scores"][0]["score"] += 1e-8
    comparison = compare_outputs(reference, changed, artifacts["variants"]["hybrid"])
    assert comparison["label_gate_pass"]
    assert comparison["aspect_nonidentical_score_count"] == 1
    assert comparison["aspect_max_abs_difference"] > 0


def test_gpu_gate_fails_any_changed_label(artifacts):
    from check_serving_gpu import compare_outputs

    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant="qlora")
    response = engine.predict_batch([{"request_id": "a", "text": "A useful review"}])
    changed = deepcopy(response)
    changed[0]["predictions"].append({"aspect": "new", "sentiments": ["negative"],
                                     "sentiment_scores": {"negative": 1., "neutral": 0., "positive": 0.}})
    comparison = compare_outputs(response, changed, artifacts["variants"]["qlora"])
    assert not comparison["label_gate_pass"]
    assert comparison["label_changed_requests"] == 1


def test_profile_aggregate_counts_reviews_and_preserves_failures():
    from profile_serving_engine import aggregate_configuration

    row = {"phase": "measure", "status": "ok", "request_count": 2,
           "wall_seconds": 4.0, "input_tokens": 10, "padded_tokens": 16,
           "encode_ms": 1., "compute_ms": 3999., "peak_allocated_bytes": 20,
           "peak_reserved_bytes": 25, "prediction_fingerprints": {"a": "x", "b": "y"}}
    result = aggregate_configuration([row], expected_reviews=2, memory_limit=100,
                                     baseline={"a": "x", "b": "y"})
    assert result["reviews_per_second"] == .5
    assert result["batch_wall_p50_ms"] == 4000
    assert result["amortized_cost_p50_ms_per_review"] == 2000
    assert result["padding_overhead_ratio"] == 1.6
    assert result["eligible_for_automatic_selection"]
    failure = {"phase": "measure", "status": "oom", "request_count": 2,
               "error": {"message": "out of memory"}}
    result = aggregate_configuration([row, failure], expected_reviews=4,
                                     memory_limit=100, baseline={})
    assert not result["complete"]
    assert result["completed_reviews"] == 2
    assert result["failure_count"] == 1
    assert not result["eligible_for_automatic_selection"]


def test_profile_changed_labels_cannot_win_automatic_selection():
    from profile_serving_engine import aggregate_configuration

    row = {"phase": "measure", "status": "ok", "request_count": 1,
           "wall_seconds": 0.1, "input_tokens": 10, "padded_tokens": 10,
           "encode_ms": 1., "compute_ms": 99., "peak_allocated_bytes": 20,
           "prediction_fingerprints": {"a": "changed"}}
    result = aggregate_configuration([row], expected_reviews=1, memory_limit=100,
                                     baseline={"a": "original"})
    assert result["complete"]
    assert result["label_changed_ids_vs_first_successful_pair_config"] == ["a"]
    assert not result["eligible_for_automatic_selection"]


def test_prefix_integration_falls_back_with_unsupported_tokenizer_and_versions(artifacts):
    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant="hybrid")
    request = [{"request_id": "a", "text": "A useful review"}]
    baseline = engine.predict_batch(request)
    engine.enable_prefix_cache = True
    candidate = engine.predict_batch(request)
    assert predictions_only(baseline) == predictions_only(candidate)
    assert baseline[0]["model_version"].endswith("hybrid-prefixoff")
    assert candidate[0]["model_version"].endswith("hybrid-prefixon")
    assert candidate[0]["diagnostics"]["batch_aspect"]["prefix_cache_delta"]["miss_prompts"] == 2
    assert candidate[0]["diagnostics"]["batch_sentiment"]["prefix_cache_enabled"] is False
