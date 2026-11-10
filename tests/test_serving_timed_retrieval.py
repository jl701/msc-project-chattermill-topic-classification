"""CPU-only retrieval deadlines, late-result isolation and explicit fallback."""
import hashlib
import threading
import time

import pytest

from msc_project.llm.qwen_two_stage_classifier import TwoStageDemonstration
from msc_project.serving.timed_retrieval import (
    RetrievalBusyError, RetrievalIdentityError, RetrievalTimeoutError,
    TimedDemonstrationRetriever,
)


class Retriever:
    method, seed, fingerprint = "bm25", 13, "test-pool"

    def __init__(self, *, release=None, failure=None, tamper=None):
        self.release, self.failure, self.tamper = release, failure, tamper
        self.prepare_calls = []
        self.select_calls = []
        self.started = threading.Event()
        self.fixed = [TwoStageDemonstration(f"fixed-{i}", "A", "Fixed train example", "A card", answer)
                      for i, answer in enumerate("YYNN")]

    def metadata(self):
        return {"method": self.method, "seed": self.seed,
                "pool_sha256": self.fingerprint, "index_version": "test-v1"}

    def prepare(self, texts, cards):
        self.prepare_calls.append((tuple(texts), tuple(cards)))
        self.started.set()
        if self.release is not None:
            assert self.release.wait(5), "Test failed to release its retrieval worker"
        if self.failure is not None:
            raise self.failure

    def select(self, text, aspect, card):
        self.select_calls.append((text, aspect, card))
        prefix = hashlib.sha256(text.encode()).hexdigest()[:12]
        demos = [TwoStageDemonstration(f"{prefix}-{i}", "A", "Other train example", "A card", answer)
                 for i, answer in enumerate("YYNN")]
        info = {**self.metadata(), "candidate_aspect": aspect,
                "selected_ids": [d.row_uid + "|" + d.candidate_aspect for d in demos],
                "selected_answers": list("YYNN"), "retrieval_ms": 1.0, "fallback": False}
        if self.tamper:
            info.update(self.tamper)
        return demos, info


def test_exact_query_card_and_index_identity_no_cross_request_result_cache():
    inner = Retriever()
    wrapper = TimedDemonstrationRetriever(inner, candidate_aspects={"A card": "A", "B card": "B"})
    try:
        wrapper.prepare(["first", "first"], ["A card", "B card", "A card"])
        demos, info = wrapper.select("first", "A", "A card")
        assert len(inner.prepare_calls) == 1
        assert len(inner.select_calls) == 2
        assert info["index_version"] == "test-v1"
        assert info["query_text_sha256"] == hashlib.sha256(b"first").hexdigest()
        assert info["retrieval_work_ms"] == 1.0
        assert info["retrieval_timing_scope"].startswith("work_in_prepare")
        for bad in [("second", "A", "A card"), ("first", "B", "A card")]:
            with pytest.raises(RetrievalIdentityError, match="not in"):
                wrapper.select(*bad)
        # Caller mutation does not corrupt the stored result.
        info["selected_ids"].clear()
        demos.clear()
        assert len(wrapper.select("first", "A", "A card")[0]) == 4
        previous_identity = info["retrieval_batch_sha256"]
        wrapper.prepare(["first"], ["A card"])
        assert len(inner.prepare_calls) == 2
        assert wrapper.select("first", "A", "A card")[1]["retrieval_batch_sha256"] != previous_identity
    finally:
        wrapper.close(wait=True)


@pytest.mark.parametrize("method", ["random", "bm25"])
def test_real_retriever_selects_identical_examples_through_wrapper(tmp_path, method):
    from test_serving_retrieval import make_index
    from msc_project.serving.retrieval import DemonstrationRetriever

    make_index(tmp_path)
    inner = DemonstrationRetriever(tmp_path, method=method, seed=13)
    inner.prepare(["A new helpful review"], ["A meaning"])
    expected, expected_info = inner.select("A new helpful review", "A", "A meaning")
    wrapper = TimedDemonstrationRetriever(inner, candidate_aspects={"A meaning": "A"})
    try:
        wrapper.prepare(["A new helpful review"], ["A meaning"])
        actual, info = wrapper.select("A new helpful review", "A", "A meaning")
        assert actual == expected
        assert info["demonstration_sha256"] == expected_info["demonstration_sha256"]
    finally:
        wrapper.close(wait=True)


def test_timeout_and_busy_never_queue_work_or_publish_late_results():
    release = threading.Event()
    inner = Retriever(release=release)
    wrapper = TimedDemonstrationRetriever(inner, candidate_aspects={"A card": "A"}, timeout_s=.02)
    try:
        started = time.perf_counter()
        with pytest.raises(RetrievalTimeoutError):
            wrapper.prepare(["first"], ["A card"])
        assert time.perf_counter() - started < 1.0
        assert inner.started.is_set()
        with pytest.raises(RetrievalIdentityError, match="No completed"):
            wrapper.select("first", "A", "A card")
        for i in range(3):
            with pytest.raises(RetrievalBusyError):
                wrapper.prepare([f"blocked-{i}"], ["A card"])
        assert len(inner.prepare_calls) == 1
        release.set()
        wrapper._future.result(timeout=2)  # Old completion still cannot publish state.
        with pytest.raises(RetrievalIdentityError):
            wrapper.select("first", "A", "A card")
        wrapper.prepare(["fresh"], ["A card"])
        _, info = wrapper.select("fresh", "A", "A card")
        assert info["query_text_sha256"] == hashlib.sha256(b"fresh").hexdigest()
        with pytest.raises(RetrievalIdentityError):
            wrapper.select("first", "A", "A card")
        assert len(inner.prepare_calls) == 2
        meta = wrapper.metadata()
        assert meta["timed_retrieval_counters"] == {
            "submitted": 2, "completed": 1, "timed_out": 1,
            "busy_rejected": 3, "failed": 0, "late_discarded": 1}
        assert meta["timeout_cancels_computation"] is False
        assert meta["queued_retrieval_batches"] == 0
    finally:
        release.set()
        wrapper.close(wait=True)


@pytest.mark.parametrize("failure", [OSError("synthetic retrieval failure"), TimeoutError("inner timeout")])
def test_inner_exceptions_are_preserved_and_recovery_works(failure):
    inner = Retriever(failure=failure)
    wrapper = TimedDemonstrationRetriever(inner, candidate_aspects={"A card": "A"})
    try:
        with pytest.raises(type(failure)) as caught:
            wrapper.prepare(["query"], ["A card"])
        assert caught.value is failure
        assert wrapper.metadata()["timed_retrieval_counters"]["failed"] == 1
        assert wrapper.metadata()["timed_retrieval_counters"]["timed_out"] == 0
        inner.failure = None
        wrapper.prepare(["recovered"], ["A card"])
        assert len(wrapper.select("recovered", "A", "A card")[0]) == 4
    finally:
        wrapper.close(wait=True)


@pytest.mark.parametrize("tamper", [{"pool_sha256": "other-index"}, {"index_version": "other-version"},
                                  {"candidate_aspect": "wrong"}, {"selected_ids": []},
                                  {"selected_answers": list("YNYY")}])
def test_wrong_index_candidate_or_selection_metadata_is_rejected(tamper):
    wrapper = TimedDemonstrationRetriever(Retriever(tamper=tamper), candidate_aspects={"A card": "A"})
    try:
        with pytest.raises(RetrievalIdentityError):
            wrapper.prepare(["query"], ["A card"])
        with pytest.raises(RetrievalIdentityError):
            wrapper.select("query", "A", "A card")
    finally:
        wrapper.close(wait=True)


def test_invalid_new_batch_clears_old_results_and_close_rejects_work():
    wrapper = TimedDemonstrationRetriever(Retriever(), candidate_aspects={"A card": "A"})
    wrapper.prepare(["first"], ["A card"])
    with pytest.raises(RetrievalIdentityError):
        wrapper.prepare(["second"], ["unknown"])
    with pytest.raises(RetrievalIdentityError):
        wrapper.select("first", "A", "A card")
    wrapper.close(wait=True)
    with pytest.raises(RuntimeError, match="closed"):
        wrapper.prepare(["third"], ["A card"])


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected_before_creating_worker(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        TimedDemonstrationRetriever(Retriever(), candidate_aspects={"A card": "A"}, timeout_s=timeout)


def test_fixed_mode_and_ambiguous_candidate_mapping_are_rejected():
    inner = Retriever()
    with pytest.raises(ValueError, match="uniquely"):
        TimedDemonstrationRetriever(inner, candidate_aspects={"A card": "A", "other card": "A"})
    inner.method = "fixed"
    with pytest.raises(ValueError, match="dynamic"):
        TimedDemonstrationRetriever(inner, candidate_aspects={"A card": "A"})


def test_engine_explicit_fixed_fallback_on_timeout_and_busy_then_recovers(tmp_path):
    # Reuse the existing tiny CPU tokenizer/model without loading pretrained weights.
    from test_serving_inference import artifacts as artifact_fixture, configured_fallback, Model, Tokenizer
    from msc_project.serving.inference import QwenServingEngine

    artifacts = artifact_fixture.__wrapped__()
    artifacts, _ = configured_fallback(artifacts, tmp_path)
    engine = QwenServingEngine(artifacts, Tokenizer(), Model(), variant="hybrid")
    release = threading.Event()
    inner = Retriever(release=release)
    inner.fixed = engine.demonstrations["aspect"]
    wrapper = TimedDemonstrationRetriever(
        inner, candidate_aspects={c["text"]: c["aspect"] for c in engine.candidates}, timeout_s=.02)
    engine.set_retriever(wrapper, fallback_on_error=True)
    try:
        first = engine.predict_batch([{"request_id": "timeout", "text": "First query"}])[0]
        assert first["request_id"] == "timeout"
        assert all(d["fallback"] and d["error_type"] == "RetrievalTimeoutError"
                   for d in first["diagnostics"]["batch_aspect"]["retrieval"])
        second = engine.predict_batch([{"request_id": "busy", "text": "Second query"}])[0]
        assert all(d["fallback"] and d["error_type"] == "RetrievalBusyError"
                   for d in second["diagnostics"]["batch_aspect"]["retrieval"])
        assert len(inner.prepare_calls) == 1
        release.set()
        wrapper._future.result(timeout=2)
        third = engine.predict_batch([{"request_id": "fresh", "text": "Third query"}])[0]
        assert third["request_id"] == "fresh"
        assert not any(d["fallback"] for d in third["diagnostics"]["batch_aspect"]["retrieval"])
        assert engine.model.adapter_enabled
    finally:
        release.set()
        wrapper.close(wait=True)
