from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from msc_project.llm.qwen_two_stage_classifier import (
    SentimentVerbalizerTokenIds,
    TwoStageDemonstration,
    demonstrations_sha256,
    encode_two_stage_training_example,
    qwen_two_stage_contract_sha256,
    qwen_two_stage_few_shot_contract_sha256,
    render_two_stage_few_shot_prompt,
    render_two_stage_prompt,
    restricted_probabilities_from_logits,
    score_two_stage_prompts,
    unique_missing_positions,
    validate_sentiment_verbalizer_token_ids,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def __init__(self) -> None:
        self.last_messages = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        self.last_messages = messages
        value = " ".join(row["content"] for row in messages)
        return value + (" <assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        fixed = {"Y": [7], "N": [8], "A": [10], "B": [11], "C": [12]}
        return {"input_ids": fixed.get(text, list(range(20, 20 + len(text.split()))))}


class FixedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids, attention_mask):
        logits = torch.zeros((*input_ids.shape, 20), device=input_ids.device)
        logits[:, :, 7] = 3.0
        logits[:, :, 8] = 1.0
        logits[:, :, 10] = 1.0
        logits[:, :, 11] = 2.0
        logits[:, :, 12] = 4.0
        return SimpleNamespace(logits=logits)


def test_prompts_keep_review_and_aspect_as_quoted_data() -> None:
    tokenizer = FakeTokenizer()
    render_two_stage_prompt(
        tokenizer,
        'review "data"',
        "Aspect: Value.",
        mode="aspect",
    )
    payload = json.loads(tokenizer.last_messages[1]["content"])
    assert payload == {
        "review": 'review "data"',
        "aspect_candidate": "Aspect: Value.",
    }


def test_sentiment_verbalizers_and_restricted_softmax() -> None:
    ids = validate_sentiment_verbalizer_token_ids(FakeTokenizer())
    assert ids == SentimentVerbalizerTokenIds(10, 11, 12)
    logits = torch.zeros((1, 2, 20))
    logits[0, 1, 10:13] = torch.tensor([1.0, 2.0, 4.0])
    probabilities = restricted_probabilities_from_logits(
        logits, torch.tensor([[1, 1]]), ids.ordered()
    )
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    assert int(probabilities.argmax(dim=1)[0]) == 2


def test_scoring_returns_binary_and_three_way_probabilities() -> None:
    tokenizer = FakeTokenizer()
    model = FixedModel()
    aspect = score_two_stage_prompts(
        model,
        tokenizer,
        ["one", "two"],
        ["aspect a", "aspect b"],
        mode="aspect",
        max_length=32,
        batch_size=2,
    )
    sentiment = score_two_stage_prompts(
        model,
        tokenizer,
        ["one", "two"],
        ["aspect a", "aspect b"],
        mode="sentiment",
        max_length=32,
        batch_size=2,
    )
    assert aspect.shape == (2, 2)
    assert sentiment.shape == (2, 3)
    assert np.allclose(aspect.sum(axis=1), 1.0)
    assert np.allclose(sentiment.sum(axis=1), 1.0)


def test_missing_prompt_positions_are_unique_and_stable() -> None:
    assert unique_missing_positions(
        ["cached", "new-a", "new-a", "new-b", "cached"], {"cached"}
    ) == [1, 3]


def _demonstrations(mode: str) -> tuple[TwoStageDemonstration, ...]:
    answers = ("Y", "Y", "N", "N") if mode == "aspect" else ("A", "B", "C")
    return tuple(
        TwoStageDemonstration(
            row_uid=f"train:{index}",
            candidate_aspect=f"Aspect {index}",
            review_text=f"example review {index}",
            aspect_candidate=f"Aspect: Aspect {index}. Definition: definition {index}",
            answer=answer,
        )
        for index, answer in enumerate(answers)
    )


def test_few_shot_prompt_has_locked_role_and_answer_sequence() -> None:
    tokenizer = FakeTokenizer()
    demonstrations = _demonstrations("aspect")
    render_two_stage_few_shot_prompt(
        tokenizer,
        "query review",
        "Aspect: query.",
        mode="aspect",
        demonstrations=demonstrations,
    )
    assert [message["role"] for message in tokenizer.last_messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [
        message["content"]
        for message in tokenizer.last_messages
        if message["role"] == "assistant"
    ] == ["Y", "Y", "N", "N"]
    assert json.loads(tokenizer.last_messages[-1]["content"])["aspect_candidate"] == (
        "Aspect: query."
    )


def test_few_shot_contract_and_manifest_are_separate_from_zero_shot() -> None:
    demonstrations = _demonstrations("sentiment")
    assert qwen_two_stage_few_shot_contract_sha256(max_length=1024) != (
        qwen_two_stage_contract_sha256(max_length=1024)
    )
    assert demonstrations_sha256(demonstrations) == demonstrations_sha256(
        tuple(demonstrations)
    )
    assert demonstrations_sha256(demonstrations) != demonstrations_sha256(
        tuple(reversed(demonstrations))
    )


def test_few_shot_scoring_uses_segment_aware_budget() -> None:
    tokenizer = FakeTokenizer()
    model = FixedModel()
    probabilities = score_two_stage_prompts(
        model,
        tokenizer,
        ["query " * 50],
        ["Aspect: query."],
        mode="sentiment",
        max_length=100,
        batch_size=1,
        demonstrations=_demonstrations("sentiment"),
    )
    assert probabilities.shape == (1, 3)
    assert int(probabilities.argmax(axis=1)[0]) == 2


def test_two_stage_training_example_supervises_one_task_specific_token() -> None:
    tokenizer = FakeTokenizer()
    aspect = encode_two_stage_training_example(
        tokenizer,
        "review",
        "Aspect: alpha.",
        mode="aspect",
        answer="N",
        max_length=32,
    )
    sentiment = encode_two_stage_training_example(
        tokenizer,
        "review",
        "Aspect: alpha.",
        mode="sentiment",
        answer="B",
        max_length=32,
    )
    assert [value for value in aspect["labels"] if value != -100] == [8]
    assert aspect["input_ids"][-1] == 8
    assert [value for value in sentiment["labels"] if value != -100] == [11]
    assert sentiment["input_ids"][-1] == 11
