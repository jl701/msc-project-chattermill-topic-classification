"""Bounded frozen-few-shot KV prefix reuse, independent of the serving baseline.

Only system and training-demonstration tokens before the final query JSON may
be retained. Every fully encoded query must match those tokens exactly. A fresh
DynamicCache is constructed for each microbatch; appended query states never
enter the stored prefix. Caller must hold the engine lock and disable LoRA.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from msc_project.llm.qwen_pair_classifier import CandidatePairBatchCollator, _model_device
from msc_project.llm.qwen_pair_classifier import last_nonpad_indices
from msc_project.llm.qwen_two_stage_classifier import (
    TwoStageDemonstration,
    _encode_segment_aware_few_shot_prompt,
    _payload,
    demonstrations_sha256,
    render_two_stage_few_shot_prompt,
)


def build_frozen_prefix_ids(
    tokenizer: Any,
    *,
    mode: str,
    demonstrations: Sequence[TwoStageDemonstration],
    max_length: int,
) -> tuple[int, ...]:
    """Extract whole tokens strictly before query data from the original encoder.

    Marker text is synthetic and is excluded together with the complete final
    user payload. Offsets prevent a BPE token crossing that boundary from being
    retained. Separate prefix tokenization is never assumed equivalent.
    """
    if mode not in {"aspect", "sentiment"}:
        raise ValueError("Prefix mode must be aspect or sentiment.")
    marker_review = "PREFIX_QUERY_REVIEW_SENTINEL_63927"
    marker_candidate = "PREFIX_QUERY_CANDIDATE_SENTINEL_63927"
    prompt = render_two_stage_few_shot_prompt(
        tokenizer, marker_review, marker_candidate, mode=mode,
        demonstrations=demonstrations,
    )
    query_payload = _payload(marker_review, marker_candidate)
    boundary = prompt.rfind(query_payload)
    if boundary < 0 or prompt.count(query_payload) != 1:
        raise ValueError("Cannot identify one final query payload in the chat template.")
    tokens = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(tokens["input_ids"])
    offsets = list(tokens.get("offset_mapping", []))
    if len(ids) != len(offsets):
        raise ValueError("Exact token offsets are required for safe prefix extraction.")
    original = _encode_segment_aware_few_shot_prompt(
        tokenizer, marker_review, marker_candidate, mode=mode,
        demonstrations=demonstrations, max_length=max_length,
    )["input_ids"]
    if original != ids:
        raise ValueError("Synthetic template required truncation; use uncached inference.")
    count = 0
    for start, end in offsets:
        # Zero offsets do not establish a safe character boundary.
        if not (0 <= int(start) < int(end) <= boundary):
            break
        count += 1
    if count < 1 or count >= len(ids):
        raise ValueError("No safe nonempty training-only prefix could be extracted.")
    return tuple(int(value) for value in original[:count])


@dataclass(frozen=True)
class _Prefix:
    ids: tuple[int, ...]
    key_values: tuple[tuple[Any, Any], ...]
    fingerprint: str

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for pair in self.key_values for t in pair)


class FrozenFewShotPrefixCache:
    """At most one immutable training-only cache per task mode.

    This component does not change model adapter state. It must only be called
    while the enclosing engine holds its lock and has disabled its adapter.
    ``fingerprint`` must identify the frozen model/tokenizer/quantization/fold.
    """

    def __init__(
        self, model: Any, tokenizer: Any,
        demonstrations: Mapping[str, Sequence[TwoStageDemonstration]],
        *, fingerprint: str, max_length: int = 1024,
    ) -> None:
        if not fingerprint:
            raise ValueError("A frozen model/tokenizer fingerprint is required.")
        self.model = model
        self.tokenizer = tokenizer
        self.demonstrations = {mode: tuple(values) for mode, values in demonstrations.items()}
        self.fingerprint = fingerprint
        self.max_length = max_length
        self._prefixes: dict[str, _Prefix] = {}
        self._disabled: dict[str, str] = {}
        self._stats = {"hit_microbatches": 0, "hit_prompts": 0,
                       "miss_microbatches": 0, "miss_prompts": 0,
                       "reused_prefix_tokens": 0, "build_ms": 0.0}
        self._miss_reasons: dict[str, int] = {}
        self.last_logit_margins: np.ndarray | None = None
        self.collator = CandidatePairBatchCollator(tokenizer, padding_side="right")

    def _miss(self, reason: str, count: int):
        self._stats["miss_microbatches"] += 1
        self._stats["miss_prompts"] += count
        self._miss_reasons[reason] = self._miss_reasons.get(reason, 0) + 1
        return None

    def stats(self) -> dict[str, Any]:
        return {**self._stats, "miss_reasons": dict(self._miss_reasons),
                "prefix_tokens": {mode: len(value.ids) for mode, value in self._prefixes.items()},
                "prefix_bytes": {mode: value.bytes for mode, value in self._prefixes.items()},
                "prefix_identity": {mode: value.fingerprint for mode, value in self._prefixes.items()},
                "disabled_modes": dict(self._disabled),
                "cached_review_text_or_predictions": False}

    def clear(self) -> None:
        self._prefixes.clear()

    def _prepare(self, mode: str) -> _Prefix | None:
        import torch

        if mode in self._prefixes:
            return self._prefixes[mode]
        if mode in self._disabled:
            return None
        if mode not in {"aspect", "sentiment"}:
            raise ValueError("Unknown task mode.")
        try:
            ids = build_frozen_prefix_ids(
                self.tokenizer, mode=mode, demonstrations=self.demonstrations[mode],
                max_length=self.max_length,
            )
        except (ValueError, KeyError, NotImplementedError) as error:
            self._disabled[mode] = str(error)
            return None
        device = _model_device(self.model)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=True, logits_to_keep=1)
            cache = output["past_key_values"] if isinstance(output, dict) else output.past_key_values
            if cache is None or cache.get_seq_length() != len(ids):
                raise RuntimeError("Model did not return the complete prefix KV cache.")
            pairs = tuple((layer.keys.detach().clone(), layer.values.detach().clone())
                          for layer in cache.layers)
            if not pairs or any(k.shape[-2] != len(ids) or v.shape[-2] != len(ids) for k, v in pairs):
                raise RuntimeError("Sliding or incomplete cache is unsupported by this experiment.")
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        self._stats["build_ms"] += (time.perf_counter() - started) * 1000
        identity = {"model_tokenizer": self.fingerprint, "mode": mode,
                    "max_length": self.max_length,
                    "demonstrations": demonstrations_sha256(self.demonstrations[mode]),
                    "prefix_ids": ids, "adapter_enabled": False}
        value = _Prefix(ids=ids, key_values=pairs,
                        fingerprint=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())
        self._prefixes[mode] = value
        return value

    def score_if_compatible(
        self, items: Sequence[dict[str, list[int]]], token_ids: Sequence[int],
        *, mode: str, fingerprint: str, adapter_enabled: bool,
    ) -> np.ndarray | None:
        """Return scores or None to request unchanged full-prompt fallback.

        Cache misses do not shorten/rewrite the prompt. On a hit the suffix
        receives absolute positions P..P+S-1 and a P+S attention mask. Last-token
        logit selection uses suffix-relative positions, unlike cache_position.
        """
        import torch
        from transformers import DynamicCache

        self.last_logit_margins = None
        if not items:
            return np.empty((0, len(token_ids)))
        if adapter_enabled:
            return self._miss("adapter_enabled", len(items))
        if fingerprint != self.fingerprint:
            return self._miss("model_tokenizer_fingerprint", len(items))
        prefix = self._prepare(mode)
        if prefix is None:
            return self._miss("unsupported_prefix", len(items))
        length = len(prefix.ids)
        if any(tuple(item["input_ids"][:length]) != prefix.ids
               or len(item["input_ids"]) <= length
               or item["attention_mask"] != [1] * len(item["input_ids"])
               for item in items):
            return self._miss("exact_token_prefix_mismatch", len(items))
        suffixes = [{"input_ids": item["input_ids"][length:],
                     "attention_mask": item["attention_mask"][length:]} for item in items]
        batch = self.collator(suffixes)
        device = _model_device(self.model)
        if device is not None:
            batch = {name: value.to(device) for name, value in batch.items()}
        suffix_attention = batch["attention_mask"]
        final_positions = last_nonpad_indices(suffix_attention)
        kept_positions = torch.unique(final_positions, sorted=True)
        complete_attention = torch.cat(
            [torch.ones((len(items), length), dtype=suffix_attention.dtype,
                        device=suffix_attention.device), suffix_attention], dim=1,
        )
        # New tensors and a new cache object for EVERY call. DynamicCache.update
        # is mutable, so stored prefix tensors must never be handed to it.
        with torch.inference_mode():
            copied_pairs = ((key.repeat(len(items), 1, 1, 1), value.repeat(len(items), 1, 1, 1))
                            for key, value in prefix.key_values)
            cache = DynamicCache(ddp_cache_data=copied_pairs, config=self.model.config)
            positions = torch.arange(length, length + batch["input_ids"].shape[1],
                                     dtype=torch.long, device=batch["input_ids"].device)
            output = self.model(
                input_ids=batch["input_ids"], attention_mask=complete_attention,
                past_key_values=cache, cache_position=positions,
                position_ids=positions.unsqueeze(0), use_cache=True,
                logits_to_keep=kept_positions,
            )
            logits = output["logits"] if isinstance(output, dict) else output.logits
            if logits.shape[1] != len(kept_positions):
                raise RuntimeError("Model ignored suffix logits_to_keep.")
            relative = torch.searchsorted(kept_positions, final_positions)
            final = logits[torch.arange(len(items), device=logits.device), relative]
            probabilities = torch.softmax(final[:, list(token_ids)].float(), dim=-1)
            if mode == "aspect":
                margins = final[:, token_ids[0]].to(torch.float64) - final[:, token_ids[1]].to(torch.float64)
                self.last_logit_margins = margins.detach().cpu().numpy()
                if not np.isfinite(self.last_logit_margins).all():
                    raise RuntimeError("Non-finite logit margins after prefix reuse")
        values = probabilities.detach().cpu().numpy().astype(float, copy=False)
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite scores after prefix reuse.")
        self._stats["hit_microbatches"] += 1
        self._stats["hit_prompts"] += len(items)
        self._stats["reused_prefix_tokens"] += length * len(items)
        return values
