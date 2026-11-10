"""Bound request waiting for CPU retrieval without cancelling its computation.

The caller serialises inference requests. This wrapper owns at most one worker
future and never queues another batch behind timed-out work. Late results remain
local to that future and are discarded; they cannot become a later batch's
demonstrations. Construct the underlying retriever (including encoder loading)
before constructing this wrapper. Offline quality runs use the raw retriever.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import threading
import time
from typing import Any, Mapping, Sequence


class RetrievalTimeoutError(TimeoutError):
    """This batch exhausted its retrieval waiting budget."""


class RetrievalBusyError(RuntimeError):
    """Earlier timed-out work is still running; no new batch was queued."""


class RetrievalIdentityError(ValueError):
    """A retrieval result does not match its query, candidate or index."""


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _BatchResult:
    identity: str
    results: dict[tuple[str, str, str], tuple[tuple, dict[str, Any]]]
    work_ms: float


class TimedDemonstrationRetriever:
    """Retrieve a whole batch in one bounded wait, then expose exact-key results.

    A timeout is propagated to the engine, which explicitly chooses either
    validated fixed demonstrations or a failed request. It is not a hard CPU
    compute timeout: Python cannot safely stop an already running model thread.
    """

    def __init__(self, inner, *, candidate_aspects: Mapping[str, str],
                 timeout_s: float = 10.0) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Retrieval timeout must be finite and positive")
        if inner.method == "fixed":
            raise ValueError("The timed wrapper is only for dynamic retrieval")
        self.candidate_aspects = dict(candidate_aspects)
        if (not self.candidate_aspects or
                any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                    for k, v in self.candidate_aspects.items()) or
                len(set(self.candidate_aspects.values())) != len(self.candidate_aspects)):
            raise ValueError("Candidate cards must map uniquely to nonempty aspect names")
        self._inner = inner
        self._base_metadata = deepcopy(inner.metadata())
        for field in ("method", "seed", "pool_sha256", "index_version"):
            if field not in self._base_metadata:
                raise ValueError(f"Retriever metadata requires {field}")
        self.method = inner.method
        self.seed = inner.seed
        self.fingerprint = inner.fingerprint
        if (self._base_metadata["method"] != self.method or
                self._base_metadata["seed"] != self.seed or
                self._base_metadata["pool_sha256"] != self.fingerprint):
            raise RetrievalIdentityError("Retriever attributes disagree with index metadata")
        self.fixed = tuple(inner.fixed)
        self.timeout_s = float(timeout_s)
        self.prepare_ms = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-retrieval")
        self._state_lock = threading.Lock()
        self._prepare_lock = threading.Lock()
        self._future: Future | None = None
        self._active: _BatchResult | None = None
        self._generation = 0
        self._closed = False
        self._last_batch: dict[str, Any] = {"state": "not_started"}
        self._counters = dict(submitted=0, completed=0, timed_out=0,
                              busy_rejected=0, failed=0, late_discarded=0)

    def metadata(self) -> dict[str, Any]:
        with self._state_lock:
            return {**deepcopy(self._base_metadata),
                    "waiting_timeout_s": self.timeout_s,
                    "timeout_scope": "prepare_and_all_batch_selections",
                    "timeout_cancels_computation": False,
                    "maximum_inflight_retrieval_batches": 1,
                    "queued_retrieval_batches": 0,
                    "encoder_initialisation_scope": "service_startup_outside_request_timeout",
                    "result_scope": "exact_current_batch_query_and_candidate_only",
                    "timed_retrieval_counters": dict(self._counters),
                    "timed_retrieval_last_batch": dict(self._last_batch)}

    def _run_batch(self, identity: str, texts: tuple[str, ...],
                   cards: tuple[str, ...]) -> _BatchResult:
        started = time.perf_counter()
        # Only this one worker may access the inner retriever's mutable buffers.
        self._inner.prepare(texts, cards)
        results = {}
        for text in texts:
            for card in cards:
                aspect = self.candidate_aspects[card]
                demonstrations, original = self._inner.select(text, aspect, card)
                info = deepcopy(original)
                for field in ("method", "seed", "pool_sha256", "index_version"):
                    if info.get(field) != self._base_metadata[field]:
                        raise RetrievalIdentityError(f"Selection {field} differs from loaded index")
                ids = [d.row_uid + "|" + d.candidate_aspect for d in demonstrations]
                answers = [d.answer for d in demonstrations]
                if (info.get("candidate_aspect") != aspect or
                        info.get("selected_ids") != ids or
                        info.get("selected_answers") != answers or
                        answers != list("YYNN")):
                    raise RetrievalIdentityError("Selection identity or 2Y2N structure differs")
                # Hashes reveal no raw review text and identify precisely what was
                # requested. The worker returns values; it never publishes state.
                info.update(retrieval_batch_sha256=identity,
                            query_text_sha256=_sha(text), candidate_text_sha256=_sha(card),
                            retrieval_work_ms=info.get("retrieval_ms", 0.0),
                            retrieval_ms=0.0,
                            retrieval_timing_scope="work_in_prepare; select_is_exact_result_lookup")
                results[(text, aspect, card)] = (tuple(demonstrations), info)
        return _BatchResult(identity, results, (time.perf_counter() - started) * 1000)

    def prepare(self, texts: Sequence[str], candidate_texts: Sequence[str]) -> None:
        if not self._prepare_lock.acquire(blocking=False):
            raise RetrievalBusyError("A retrieval prepare call is already active")
        started = time.perf_counter()
        try:
            with self._state_lock:
                self._active = None
                self.prepare_ms = 0.0
            unique_texts = tuple(dict.fromkeys(texts))
            unique_cards = tuple(dict.fromkeys(candidate_texts))
            if (not unique_texts or not unique_cards or
                    any(not isinstance(t, str) or not t for t in unique_texts) or
                    any(c not in self.candidate_aspects for c in unique_cards)):
                raise RetrievalIdentityError("Batch has empty queries or unknown candidate cards")
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Timed retriever is closed")
                self._generation += 1
                identity = _sha(json.dumps({"generation": self._generation,
                                           "queries": [_sha(t) for t in unique_texts],
                                           "cards": [_sha(c) for c in unique_cards]},
                                          sort_keys=True))
                self._last_batch = {"state": "preparing", "sha256": identity}
                if self._future is not None:
                    if not self._future.done():
                        self._counters["busy_rejected"] += 1
                        self._last_batch["state"] = "prior_work_still_running"
                        raise RetrievalBusyError("Earlier retrieval is still running; no batch queued")
                    # Never reuse a result that arrived after its caller timed out.
                    self._future = None
                    self._counters["late_discarded"] += 1
                self._future = self._executor.submit(self._run_batch, identity,
                                                     unique_texts, unique_cards)
                future = self._future
                self._counters["submitted"] += 1
            try:
                outcome = future.result(timeout=max(0.0, self.timeout_s - (time.perf_counter() - started)))
            except FutureTimeout as exc:
                # An inner TimeoutError is an execution failure, not evidence that
                # Future.result exhausted this wrapper's waiting budget.
                if future.done() and future.exception() is exc:
                    with self._state_lock:
                        self._future = None
                        self._counters["failed"] += 1
                        self._last_batch["state"] = "failed"
                    raise
                with self._state_lock:
                    self._counters["timed_out"] += 1
                    self._last_batch["state"] = "waiting_deadline_exceeded"
                raise RetrievalTimeoutError("Retrieval waiting deadline exceeded") from exc
            except Exception:
                with self._state_lock:
                    self._future = None
                    self._counters["failed"] += 1
                    self._last_batch["state"] = "failed"
                raise
            if outcome.identity != identity:
                raise RetrievalIdentityError("Worker returned a different batch identity")
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Timed retriever closed while the batch was running")
                self._active = outcome
                self._future = None
                self._counters["completed"] += 1
                self._last_batch.update(state="ready", work_ms=outcome.work_ms)
        finally:
            with self._state_lock:
                self.prepare_ms = (time.perf_counter() - started) * 1000
            self._prepare_lock.release()

    def select(self, text: str, candidate_aspect: str, candidate_text: str):
        started = time.perf_counter()
        with self._state_lock:
            batch = self._active
            if batch is None:
                raise RetrievalIdentityError("No completed retrieval batch is active")
            result = batch.results.get((text, candidate_aspect, candidate_text))
            if result is None:
                raise RetrievalIdentityError("Query and candidate were not in the active batch")
            demonstrations, original = result
            info = deepcopy(original)
            if (info["retrieval_batch_sha256"] != batch.identity or
                    info["query_text_sha256"] != _sha(text) or
                    info["candidate_text_sha256"] != _sha(candidate_text)):
                raise RetrievalIdentityError("Stored result identity differs from this query")
            info["retrieval_ms"] = (time.perf_counter() - started) * 1000
            return list(demonstrations), info

    def close(self, *, wait: bool = False) -> None:
        """Stop accepting work. Running CPU work remains until it finishes."""
        with self._state_lock:
            self._closed = True
            self._active = None
        self._executor.shutdown(wait=wait, cancel_futures=True)
