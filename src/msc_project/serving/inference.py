"""Real, serialized Qwen inference for the frozen two-stage serving experiment.

No prediction cache or generation is used. The original prompt encoders,
right-padding collator and restricted verbalizer probabilities are retained.
One lock covers both stages so disabling the shared LoRA cannot affect another
request. Stage timers describe a whole batch, not independently measured GPU
latencies of its constituent requests.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from msc_project.llm.qwen_pair_classifier import (
    CandidatePairBatchCollator,
    _model_device,
    last_nonpad_indices,
    validate_verbalizer_token_ids,
)
from msc_project.llm.qwen_two_stage_classifier import (
    TwoStageDemonstration,
    _encode_segment_aware_few_shot_prompt,
    encode_two_stage_prompt,
    render_two_stage_prompt,
    _tokenize_without_special_tokens,
    restricted_probabilities_from_logits,
    validate_sentiment_verbalizer_token_ids,
)

VARIANTS = ("few_shot", "qlora", "hybrid")
SENTIMENTS = ("negative", "neutral", "positive")


def plan_microbatches(
    lengths: Sequence[int], *, pair_batch_size: int, max_batch_tokens: int
) -> list[list[int]]:
    """Preserve order and bound padded tokens, not just summed token counts."""
    if pair_batch_size < 1 or max_batch_tokens < 1:
        raise ValueError("Batch limits must be positive.")
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index, length in enumerate(lengths):
        if length < 1 or length > max_batch_tokens:
            raise ValueError("A prompt is empty or exceeds the padded-token budget.")
        next_longest = max(longest, length)
        if current and (
            len(current) >= pair_batch_size
            or next_longest * (len(current) + 1) > max_batch_tokens
        ):
            batches.append(current)
            current = []
            longest = 0
        current.append(index)
        longest = max(longest, length)
    if current:
        batches.append(current)
    return batches


def decode_sentiments(scores: Sequence[float], runner_up_threshold: float) -> list[str]:
    """The frozen top-one plus thresholded runner-up rule, including stable ties."""
    values = np.asarray(scores, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected three finite sentiment scores.")
    order = np.argsort(-values, kind="stable")
    indices = [int(order[0])]
    if values[order[1]] >= runner_up_threshold:
        indices.append(int(order[1]))
    return [SENTIMENTS[index] for index in indices]


class QwenServingEngine:
    """Single resident quantized base and one immutable selected LoRA adapter."""

    def __init__(
        self,
        artifacts: Mapping[str, Any],
        tokenizer: Any,
        model: Any,
        *,
        variant: str = "hybrid",
        pair_batch_size: int = 4,
        max_batch_tokens: int = 4096,
        logits_mode: str = "full",
        startup_seconds: float = 0.0,
        enable_prefix_cache: bool = False,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        if logits_mode not in {"full", "last_token"}:
            raise ValueError("logits_mode must be full or last_token.")
        if pair_batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("Batch limits must be positive.")
        self.artifacts = dict(artifacts)
        self.tokenizer = tokenizer
        self.model = model
        self.variant = variant
        self.pair_batch_size = pair_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.logits_mode = logits_mode
        self.startup_seconds = startup_seconds
        self.enable_prefix_cache = bool(enable_prefix_cache)
        self.artifacts_fingerprint = hashlib.sha256(
            json.dumps(artifacts, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.candidates = list(artifacts["taxonomy"]["candidates"])
        names = [str(value["aspect"]) for value in self.candidates]
        if not names or len(names) != len(set(names)):
            raise ValueError("Taxonomy candidates must be nonempty and unique.")
        self.demonstrations = {
            mode: [TwoStageDemonstration(**value) for value in values]
            for mode, values in artifacts["few_shot"]["demonstrations"].items()
        }
        self.aspect_ids = validate_verbalizer_token_ids(tokenizer)
        self.sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)
        self.collator = CandidatePairBatchCollator(tokenizer, padding_side="right")
        self._lock = threading.Lock()
        self.retriever = None
        self.retrieval_fallback_on_error = False
        from msc_project.serving.prefix_cache import FrozenFewShotPrefixCache

        self.prefix_cache = FrozenFewShotPrefixCache(
            model, tokenizer, self.demonstrations,
            fingerprint=self.artifacts_fingerprint,
            max_length=int(artifacts["few_shot"]["max_length"]),
        )
        self.model.eval()
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        for variant in VARIANTS:
            thresholds = self.artifacts["variants"][variant]
            for name in ("aspect_threshold", "runner_up_threshold"):
                if not math.isfinite(float(thresholds[name])):
                    raise ValueError(f"Non-finite threshold: {variant}/{name}")

    @classmethod
    def from_artifacts(
        cls,
        path: str | Path,
        *,
        variant: str = "hybrid",
        pair_batch_size: int = 4,
        max_batch_tokens: int = 4096,
        logits_mode: str = "full",
        device: str = "cuda",
        local_files_only: bool = True,
        enable_prefix_cache: bool = False,
    ) -> "QwenServingEngine":
        import torch

        from msc_project.llm.qwen_pair_classifier import load_saved_qwen_pair_adapter

        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The recorded quantized serving experiment requires CUDA.")
        path = Path(path)
        artifacts = json.loads(path.read_text(encoding="utf-8"))
        started = time.perf_counter()
        # Retain NF4, double quantization and float16 compute from the locked loader.
        tokenizer, model, _ = load_saved_qwen_pair_adapter(
            artifacts["model"]["local_snapshot_path"],
            artifacts["adapter"]["path"],
            local_files_only=local_files_only,
        )
        placements = getattr(model, "hf_device_map", {})
        if any(str(value) in {"cpu", "disk"} for value in placements.values()):
            raise RuntimeError("CPU/disk offload is outside the single-GPU protocol.")
        torch.cuda.synchronize()
        result = cls(
            artifacts, tokenizer, model,
            variant=variant,
            pair_batch_size=pair_batch_size,
            max_batch_tokens=max_batch_tokens,
            logits_mode=logits_mode,
            startup_seconds=time.perf_counter() - started,
            enable_prefix_cache=enable_prefix_cache,
        )
        result.artifacts_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version(),
            "variant": self.variant,
            "fold_id": self.artifacts["fold_id"],
            "model_id": self.artifacts["model"]["model_id"],
            "model_revision": self.artifacts["model"]["revision"],
            "candidate_count": len(self.candidates),
            "pair_batch_size": self.pair_batch_size,
            "max_batch_tokens": self.max_batch_tokens,
            "logits_mode": self.logits_mode,
            "quantization": "NF4 double quantization, float16 compute",
            "prediction_cache": False,
            "kv_cache": self.enable_prefix_cache,
            "prefix_cache_enabled": self.enable_prefix_cache,
            "prefix_cache": self.prefix_cache.stats(),
            "startup_seconds": self.startup_seconds,
            "retrieval": self.retriever.metadata() if self.retriever else None,
            "artifacts_sha256": getattr(self, "artifacts_sha256", None),
        }

    def model_version(self, variant: str | None = None) -> str:
        digest = getattr(self, "artifacts_sha256", self.artifacts_fingerprint)
        retrieval_suffix = (f"-rag-{self.retriever.method}-{self.retriever.seed}-"
                            f"{self.retriever.fingerprint[:12]}-demofirst1") if self.retriever else ""
        return (f"{self.artifacts['model']['revision'][:8]}-{self.artifacts['fold_id']}-"
                f"{digest[:12]}-{variant or self.variant}-prefix"
                f"{'on' if self.enable_prefix_cache else 'off'}{retrieval_suffix}")

    def set_variant(self, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        with self._lock:
            self.variant = variant

    def set_retriever(self, retriever, *, fallback_on_error: bool = False) -> None:
        """Change the immutable example index/method only between full requests."""
        if retriever is not None and not all(callable(getattr(retriever, name, None))
                                             for name in ("select", "prepare", "metadata")):
            raise ValueError("Retriever must provide select, prepare and metadata")
        with self._lock:
            self.retriever = retriever
            self.retrieval_fallback_on_error = bool(fallback_on_error)

    def _synchronize(self) -> None:
        import torch

        device = _model_device(self.model)
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)

    def _score_encoded(self, items: list[dict[str, Any]], token_ids: Sequence[int],
                       *, mode: str | None = None, use_adapter: bool = True) -> np.ndarray:
        import torch

        dynamic_stage = (mode == "aspect" and self.retriever is not None
                         and self.retriever.method != "fixed")
        if self.enable_prefix_cache and not use_adapter and not dynamic_stage:
            cached = self.prefix_cache.score_if_compatible(
                items, token_ids, mode=mode, fingerprint=self.artifacts_fingerprint,
                adapter_enabled=use_adapter,
            )
            if cached is not None:
                return cached
        batch = self.collator(items)
        device = _model_device(self.model)
        if device is not None:
            batch = {name: value.to(device) for name, value in batch.items()}
        arguments = dict(batch, use_cache=False)
        final_positions = last_nonpad_indices(batch["attention_mask"])
        kept_positions = None
        if self.logits_mode == "last_token":
            kept_positions = torch.unique(final_positions, sorted=True)
            arguments["logits_to_keep"] = kept_positions
        with torch.inference_mode():
            output = self.model(**arguments)
            logits = output["logits"] if isinstance(output, dict) else output.logits
            if kept_positions is None:
                probabilities = restricted_probabilities_from_logits(
                    logits, batch["attention_mask"], token_ids
                )
            else:
                if logits.shape[1] != len(kept_positions):
                    raise RuntimeError("Model did not honour logits_to_keep; stop this optimization.")
                positions = torch.searchsorted(kept_positions, final_positions)
                final = logits[torch.arange(len(items), device=logits.device), positions]
                probabilities = torch.softmax(final[:, list(token_ids)].float(), dim=-1)
        values = probabilities.detach().cpu().numpy().astype(float, copy=False)
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite model probabilities.")
        return values

    def _score_stage(
        self,
        pairs: Sequence[tuple[int, int, str]],
        *,
        mode: str,
        use_adapter: bool,
    ) -> tuple[np.ndarray, dict[str, Any], list[int]]:
        token_ids = ((self.aspect_ids.yes, self.aspect_ids.no)
                     if mode == "aspect" else self.sentiment_ids.ordered())
        start = time.perf_counter()
        items = []
        prompt_truncation = []
        retrieval_diagnostics = []
        retrieval_error = None
        retrieval_prepare_ms = 0.0
        retrieval_active = mode == "aspect" and not use_adapter and self.retriever is not None
        if retrieval_active:
            prepare_started = time.perf_counter()
            try:
                self.retriever.prepare([p[2] for p in pairs],
                                       [str(self.candidates[p[1]]["text"]) for p in pairs])
            except Exception as error:
                if not self.retrieval_fallback_on_error or not self.retriever.fixed:
                    raise
                retrieval_error = type(error).__name__
            finally:
                retrieval_prepare_ms = (time.perf_counter() - prepare_started) * 1000
        for _, aspect_index, text in pairs:
            candidate = str(self.candidates[aspect_index]["text"])
            truncation_audit = {}
            if use_adapter:
                item = encode_two_stage_prompt(
                    self.tokenizer, text, candidate, mode=mode,
                    max_length=int(self.artifacts["qlora"]["max_length"]),
                )
                original_ids = _tokenize_without_special_tokens(
                    self.tokenizer, render_two_stage_prompt(self.tokenizer, text, candidate, mode=mode))
                truncation_audit = {"left_truncated": len(original_ids) > len(item["input_ids"]),
                                    "original_prompt_tokens": len(original_ids),
                                    "retained_prompt_tokens": len(item["input_ids"]),
                                    "policy": "retain_rightmost_prompt_tokens"}
            elif retrieval_active:
                from msc_project.serving.retrieval import encode_budgeted_prompt
                selection_started = time.perf_counter()
                try:
                    if retrieval_error:
                        raise RuntimeError(retrieval_error)
                    demonstrations, selected = self.retriever.select(
                        text, str(self.candidates[aspect_index]["aspect"]), candidate)
                except Exception as error:
                    if not self.retrieval_fallback_on_error or not self.retriever.fixed:
                        raise
                    demonstrations = self.retriever.fixed
                    selected = {**self.retriever.metadata(), "fallback": True,
                                "error_type": retrieval_error or type(error).__name__,
                                "candidate_aspect": self.candidates[aspect_index]["aspect"],
                                "selected_ids": [d.row_uid + "|" + d.candidate_aspect for d in demonstrations],
                                "retrieval_ms": (time.perf_counter() - selection_started) * 1000}
                item, truncation = encode_budgeted_prompt(
                    self.tokenizer, text, candidate, demonstrations,
                    max_length=int(self.artifacts["few_shot"]["max_length"]))
                retrieval_diagnostics.append({**selected, "truncation": truncation})
            else:
                item = _encode_segment_aware_few_shot_prompt(
                    self.tokenizer, text, candidate, mode=mode,
                    demonstrations=self.demonstrations[mode],
                    max_length=int(self.artifacts["few_shot"]["max_length"]),
                )
            items.append(item)
            prompt_truncation.append(truncation_audit)
        lengths = [len(item["input_ids"]) for item in items]
        batches = plan_microbatches(
            lengths, pair_batch_size=self.pair_batch_size,
            max_batch_tokens=self.max_batch_tokens,
        )
        encode_seconds = time.perf_counter() - start
        self._synchronize()
        compute_start = time.perf_counter()
        outputs = []
        cache_before = self.prefix_cache.stats()
        # This context is always inside predict_batch's full-request lock.
        context = nullcontext() if use_adapter else self.model.disable_adapter()
        with context:
            for indices in batches:
                outputs.append(self._score_encoded([items[i] for i in indices], token_ids,
                                                   mode=mode, use_adapter=use_adapter))
        self._synchronize()
        compute_seconds = time.perf_counter() - compute_start
        result = np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(token_ids)))
        metrics = {
            "prompt_count": len(items),
            "input_tokens": sum(lengths),
            "padded_tokens": sum(max(lengths[i] for i in batch) * len(batch) for batch in batches),
            "max_prompt_tokens": max(lengths, default=0),
            "microbatch_count": len(batches),
            "encode_ms": encode_seconds * 1000,
            "compute_ms": compute_seconds * 1000,
            "wall_ms": (time.perf_counter() - start) * 1000,
            "adapter_enabled": use_adapter,
            "retrieval_prepare_ms": retrieval_prepare_ms,
            "retrieval": retrieval_diagnostics,
            "prompt_truncation": prompt_truncation,
            "prefix_cache_configured": self.enable_prefix_cache,
            "prefix_cache_enabled": (self.enable_prefix_cache and not use_adapter
                                     and not (retrieval_active and self.retriever.method != "fixed")),
            "prefix_cache_delta": {
                name: self.prefix_cache.stats()[name] - cache_before[name]
                for name in ["hit_microbatches", "hit_prompts", "miss_microbatches",
                             "miss_prompts", "reused_prefix_tokens", "build_ms"]
            },
        }
        return result, metrics, lengths

    @staticmethod
    def _validate_requests(requests):
        identities = [str(request.get("request_id", "")) for request in requests]
        if any(not identity for identity in identities) or len(set(identities)) != len(identities):
            raise ValueError("request_id must be nonempty and unique within a batch.")
        for request in requests:
            if set(request) - {"request_id", "text"}:
                raise ValueError("Only request_id and text are accepted; labels cannot enter inference.")
            if not isinstance(request.get("text"), str) or not request["text"].strip():
                raise ValueError("A nonempty review text is required.")

    def score_presence(self, requests):
        """Offline presence-only scoring; not end-to-end API timing."""
        if not requests:
            return []
        self._validate_requests(requests)
        with self._lock:
            pairs = [(ri, ai, request["text"]) for ri, request in enumerate(requests)
                     for ai in range(len(self.candidates))]
            scores, metrics, lengths = self._score_stage(
                pairs, mode="aspect", use_adapter=self.variant == "qlora")
            responses = []
            for ri, request in enumerate(requests):
                owned = [i for i,p in enumerate(pairs) if p[0] == ri]
                responses.append({"request_id": str(request["request_id"]), "variant": self.variant,
                                  "fold_id": self.artifacts["fold_id"], "model_version": self.model_version(),
                                  "aspect_scores": [{"aspect": self.candidates[pairs[i][1]]["aspect"],
                                                     "score": float(scores[i, 0])} for i in owned],
                                  "diagnostics": {"scope": "presence_only_offline", "batch_aspect": metrics,
                                                  "aspect_input_tokens": sum(lengths[i] for i in owned),
                                                  "severe_truncation": any(
                                                      metrics["retrieval"][i]["truncation"]["severe_query_truncation"]
                                                      for i in owned) if metrics["retrieval"] else None}})
            return responses

    def predict_batch(
        self, requests: Sequence[Mapping[str, Any]], *, full_grid: bool = False
    ) -> list[dict[str, Any]]:
        """Analyze complete reviews under one model/threshold configuration.

        full_grid=True is the correctness reference: score all sentiments before
        applying the same detection gate. It never emits unselected aspects.
        """
        if not requests:
            return []
        identities = [str(request.get("request_id", "")) for request in requests]
        if any(not identity for identity in identities) or len(set(identities)) != len(identities):
            raise ValueError("request_id must be nonempty and unique within a batch.")
        for request in requests:
            if set(request) - {"request_id", "text"}:
                raise ValueError("Only request_id and text are accepted; labels cannot enter inference.")
            if not isinstance(request.get("text"), str) or not request["text"].strip():
                raise ValueError("A nonempty review text is required.")
        arrival = time.perf_counter()
        with self._lock:
            start = time.perf_counter()
            variant = self.variant
            thresholds = self.artifacts["variants"][variant]
            aspect_threshold = float(thresholds["aspect_threshold"])
            runner_threshold = float(thresholds["runner_up_threshold"])
            pairs = [(ri, ai, request["text"])
                     for ri, request in enumerate(requests)
                     for ai in range(len(self.candidates))]
            aspect, aspect_metrics, aspect_lengths = self._score_stage(
                pairs, mode="aspect", use_adapter=variant == "qlora",
            )
            selected = aspect[:, 0] >= aspect_threshold
            stage2_positions = [i for i, value in enumerate(selected) if value or full_grid]
            stage2_pairs = [pairs[i] for i in stage2_positions]
            sentiment, sentiment_metrics, sentiment_lengths = self._score_stage(
                stage2_pairs, mode="sentiment", use_adapter=variant != "few_shot",
            )
            sentiment_by_position = dict(zip(stage2_positions, sentiment))
            result = []
            for ri, request in enumerate(requests):
                owned = [i for i, pair in enumerate(pairs) if pair[0] == ri]
                stage2_owned = [i for i, pair in enumerate(stage2_pairs) if pair[0] == ri]
                predictions = []
                aspect_scores = []
                for i in owned:
                    ai = pairs[i][1]
                    candidate = self.candidates[ai]
                    aspect_scores.append({"aspect": candidate["aspect"], "score": float(aspect[i, 0])})
                    if selected[i]:
                        scores = sentiment_by_position[i]
                        predictions.append({
                            "aspect": candidate["aspect"],
                            "sentiments": decode_sentiments(scores, runner_threshold),
                            "aspect_score": float(aspect[i, 0]),
                            "sentiment_scores": dict(zip(SENTIMENTS, map(float, scores))),
                        })
                result.append({
                    "request_id": str(request["request_id"]),
                    "variant": variant,
                    "fold_id": self.artifacts["fold_id"],
                    "model_version": self.model_version(variant),
                    "predictions": predictions,
                    "aspect_scores": aspect_scores,
                    "sentiment_truncation_by_aspect": {
                        self.candidates[stage2_pairs[j][1]]["aspect"]: sentiment_metrics["prompt_truncation"][j]
                        for j in stage2_owned if sentiment_metrics["prompt_truncation"][j]},
                    "diagnostics": {
                        "severe_truncation": any(
                            aspect_metrics["retrieval"][i]["truncation"]["severe_query_truncation"]
                            for i in owned) if aspect_metrics["retrieval"] else None,
                        "candidate_count": len(self.candidates),
                        "selected_aspect_count": len(predictions),
                        "sentiment_prompt_count": len(stage2_owned),
                        "aspect_input_tokens": sum(aspect_lengths[i] for i in owned),
                        "sentiment_input_tokens": sum(sentiment_lengths[i] for i in stage2_owned),
                        "full_grid": bool(full_grid),
                        "batch_request_count": len(requests),
                        "batch_aspect": aspect_metrics,
                        "batch_sentiment": sentiment_metrics,
                        "timing_scope": "whole batch shared by all returned requests; not per-request GPU time",
                        "engine_lock_wait_ms": (start - arrival) * 1000,
                    },
                })
                if full_grid:
                    result[-1]["full_grid_scores"] = [
                        {"aspect": self.candidates[pairs[i][1]]["aspect"],
                         "sentiment_scores": dict(zip(SENTIMENTS, map(float, sentiment_by_position[i])))}
                        for i in owned]
            elapsed = (time.perf_counter() - start) * 1000
            for response in result:
                response["diagnostics"]["batch_engine_ms"] = elapsed
                response["diagnostics"]["amortized_engine_ms"] = elapsed / len(requests)
            return result
