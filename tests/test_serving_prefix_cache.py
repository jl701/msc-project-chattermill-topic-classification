from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import DynamicCache, Qwen3Config

from msc_project.llm.qwen_two_stage_classifier import (
    TwoStageDemonstration, _encode_segment_aware_few_shot_prompt,
    restricted_probabilities_from_logits,
)
from msc_project.serving.prefix_cache import FrozenFewShotPrefixCache, build_frozen_prefix_ids


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{row['role']}>\n{row['content']}\n<end>\n" for row in messages) + "<assistant>\n"

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        result = {"input_ids": [ord(char) + 1 for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class ToyCausalModel(torch.nn.Module):
    """CPU causal sums test prefix/suffix positions and mutable cache isolation."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = Qwen3Config(num_hidden_layers=1, num_key_value_heads=1,
                                 num_attention_heads=1, head_dim=1, hidden_size=4)
        self.calls = 0
        self.last_cache = None

    def forward(self, input_ids, attention_mask, past_key_values=None,
                cache_position=None, position_ids=None, use_cache=False,
                logits_to_keep=0):
        self.calls += 1
        old_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        assert attention_mask.shape[1] == old_length + input_ids.shape[1]
        if cache_position is not None:
            assert torch.equal(cache_position, torch.arange(old_length, old_length + input_ids.shape[1]))
            assert torch.equal(position_ids, cache_position.unsqueeze(0))
        values = (input_ids * attention_mask[:, old_length:]).float()[:, None, :, None]
        cache = past_key_values if past_key_values is not None else DynamicCache(config=self.config)
        keys, _ = cache.update(values, values, 0)
        prefix_sums = keys[:, 0, :, 0].cumsum(dim=-1)[:, old_length:]
        logits = torch.stack([prefix_sums.remainder(13) / 4,
                              prefix_sums.remainder(17) / 5], dim=-1)
        if isinstance(logits_to_keep, int):
            logits = logits[:, -logits_to_keep:, :]
        else:
            logits = logits[:, logits_to_keep, :]
        self.last_cache = cache
        return SimpleNamespace(logits=logits, past_key_values=cache if use_cache else None)


@pytest.fixture
def setup_cache():
    tokenizer = CharacterTokenizer()
    model = ToyCausalModel()
    demos = {mode: [TwoStageDemonstration(str(i), "Training", "Training text", "Training", answer)
                   for i, answer in enumerate(answers)]
             for mode, answers in [("aspect", "YYNN"), ("sentiment", "ABC")]}
    cache = FrozenFewShotPrefixCache(model, tokenizer, demos, fingerprint="frozen-model-tokenizer",
                                    max_length=8192)
    return tokenizer, model, demos, cache


def encoded(tokenizer, demos, mode, review):
    return _encode_segment_aware_few_shot_prompt(
        tokenizer, review, "Candidate aspect", mode=mode,
        demonstrations=demos[mode], max_length=8192,
    )


def test_training_prefix_ends_before_query_json_and_contains_no_query(setup_cache):
    tokenizer, _, demos, _ = setup_cache
    ids = build_frozen_prefix_ids(tokenizer, mode="aspect", demonstrations=demos["aspect"], max_length=8192)
    decoded = "".join(chr(value - 1) for value in ids)
    assert decoded.endswith("<user>\n")
    assert "Training text" in decoded
    assert "SENTINEL" not in decoded
    original = encoded(tokenizer, demos, "aspect", "PRIVATE_QUERY_NOT_ALLOWED_IN_CACHE")
    assert tuple(original["input_ids"][:len(ids)]) == ids
    assert "PRIVATE_QUERY" not in decoded


def test_prefix_scores_match_full_with_variable_length_and_cache_is_immutable(setup_cache):
    tokenizer, model, demos, cache = setup_cache
    items = [encoded(tokenizer, demos, "aspect", value) for value in ["short", "a longer private review"]]
    kwargs = dict(mode="aspect", fingerprint="frozen-model-tokenizer", adapter_enabled=False)
    result = cache.score_if_compatible(items, [0, 1], **kwargs)
    stored = cache._prefixes["aspect"]
    snapshot = tuple((k.clone(), v.clone()) for k, v in stored.key_values)
    batch = cache.collator(items)
    output = model(**batch, use_cache=False)
    expected = restricted_probabilities_from_logits(output.logits, batch["attention_mask"], [0, 1]).numpy()
    np.testing.assert_array_equal(result, expected)
    second = cache.score_if_compatible(list(reversed(items)), [0, 1], **kwargs)
    np.testing.assert_array_equal(second, expected[::-1])
    for (key, value), (old_key, old_value) in zip(stored.key_values, snapshot):
        assert torch.equal(key, old_key) and torch.equal(value, old_value)
        assert key.shape[-2] == len(stored.ids)
    with torch.inference_mode():
        model.last_cache.layers[0].keys.zero_()
    assert torch.equal(stored.key_values[0][0], snapshot[0][0])
    assert cache.stats()["hit_prompts"] == 4


def test_mismatch_falls_back_without_scoring_or_replacing_prefix(setup_cache):
    tokenizer, model, demos, cache = setup_cache
    item = encoded(tokenizer, demos, "aspect", "Review")
    kwargs = dict(mode="aspect", fingerprint="frozen-model-tokenizer", adapter_enabled=False)
    assert cache.score_if_compatible([item], [0, 1], **kwargs) is not None
    before = model.calls
    mutated = {key: list(value) for key, value in item.items()}
    mutated["input_ids"][5] += 1
    assert cache.score_if_compatible([mutated], [0, 1], **kwargs) is None
    assert model.calls == before
    assert cache.stats()["miss_reasons"]["exact_token_prefix_mismatch"] == 1
    assert cache.score_if_compatible([item], [0, 1], **{**kwargs, "adapter_enabled": True}) is None
    assert cache.score_if_compatible([item], [0, 1], **{**kwargs, "fingerprint": "different"}) is None
    assert model.calls == before


def test_modes_are_isolated_and_only_two_cache_entries_exist(setup_cache):
    tokenizer, _, demos, cache = setup_cache
    for mode in ["aspect", "sentiment"]:
        item = encoded(tokenizer, demos, mode, "Review")
        cache.score_if_compatible([item], [0, 1], mode=mode,
                                  fingerprint="frozen-model-tokenizer", adapter_enabled=False)
    assert set(cache._prefixes) == {"aspect", "sentiment"}
    assert cache._prefixes["aspect"].fingerprint != cache._prefixes["sentiment"].fingerprint
    aspect = encoded(tokenizer, demos, "aspect", "Review")
    assert cache.score_if_compatible([aspect], [0, 1], mode="sentiment",
                                     fingerprint="frozen-model-tokenizer", adapter_enabled=False) is None


def test_truncated_template_disables_cache_instead_of_changing_prompt(setup_cache):
    tokenizer, model, demos, _ = setup_cache
    cache = FrozenFewShotPrefixCache(model, tokenizer, demos, fingerprint="frozen", max_length=32)
    item = encoded(tokenizer, demos, "aspect", "Review")
    assert cache.score_if_compatible([item], [0, 1], mode="aspect", fingerprint="frozen", adapter_enabled=False) is None
    assert model.calls == 0
    assert "aspect" in cache.stats()["disabled_modes"]
