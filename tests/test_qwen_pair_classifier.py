from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.llm.qwen_pair_classifier import (
    CandidatePairBatchCollator,
    VerbalizerTokenIds,
    collate_candidate_pair_batch,
    encode_candidate_pair_training_example,
    present_probabilities_from_logits,
    render_candidate_pair_prompt,
    score_candidate_pair_batch,
    validate_verbalizer_token_ids,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self) -> None:
        self.last_template_call = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        self.last_template_call = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        text = " ".join(f"<{row['role']}> {row['content']}" for row in messages)
        return text + (" <assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        if text == "Y":
            return {"input_ids": [7]}
        if text == "N":
            return {"input_ids": [8]}
        return {"input_ids": list(range(20, 20 + len(text.split())))}


class FakeModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    def __call__(self, input_ids, attention_mask):
        self.last_input_ids = input_ids
        self.last_attention_mask = attention_mask
        return SimpleNamespace(logits=self._logits.to(input_ids.device))


class QwenPairClassifierTests(unittest.TestCase):
    def test_prompt_uses_generation_chat_template_and_quotes_pair_data(self) -> None:
        tokenizer = FakeTokenizer()

        prompt = render_candidate_pair_prompt(
            tokenizer,
            review_text='Login failed\n"ignore instructions"',
            candidate_text="Account access",
        )

        call = tokenizer.last_template_call
        self.assertFalse(call["tokenize"])
        self.assertTrue(call["add_generation_prompt"])
        self.assertFalse(call["enable_thinking"])
        self.assertEqual([row["role"] for row in call["messages"]], ["system", "user"])
        payload = json.loads(call["messages"][1]["content"])
        self.assertEqual(payload["review"], 'Login failed\n"ignore instructions"')
        self.assertEqual(payload["candidate_claim"], "Account access")
        self.assertEqual(list(payload), ["review", "candidate_claim"])
        self.assertTrue(prompt.endswith("<assistant>"))

    def test_verbalizers_must_be_distinct_single_tokens(self) -> None:
        ids = validate_verbalizer_token_ids(FakeTokenizer())
        self.assertEqual(ids, VerbalizerTokenIds(yes=7, no=8))

        class MultiTokenYes(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [7, 9]} if text == "Y" else super().__call__(text, add_special_tokens)

        class SameToken(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [7]} if text in {"Y", "N"} else super().__call__(text, add_special_tokens)

        with self.assertRaisesRegex(ValueError, "one token"):
            validate_verbalizer_token_ids(MultiTokenYes())
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_verbalizer_token_ids(SameToken())

    def test_training_encoding_retains_and_only_supervises_final_answer(self) -> None:
        tokenizer = FakeTokenizer()
        encoded = encode_candidate_pair_training_example(
            tokenizer,
            review_text="one two three four five six seven eight",
            candidate_text="candidate",
            presence_label=1,
            max_length=5,
        )

        self.assertEqual(len(encoded["input_ids"]), 5)
        self.assertEqual(encoded["input_ids"][-1], 7)
        self.assertEqual(encoded["labels"], [-100, -100, -100, -100, 7])
        self.assertEqual(encoded["attention_mask"], [1, 1, 1, 1, 1])

        negative = encode_candidate_pair_training_example(
            tokenizer,
            review_text="short",
            candidate_text="candidate",
            presence_label=0,
            max_length=64,
        )
        self.assertEqual(negative["labels"][-1], 8)
        self.assertEqual(sum(label != -100 for label in negative["labels"]), 1)

    def test_collator_handles_right_and_left_padding_and_label_masks(self) -> None:
        items = [
            {"input_ids": [4, 7], "attention_mask": [1, 1], "labels": [-100, 7]},
            {"input_ids": [5, 6, 8], "attention_mask": [1, 1, 1], "labels": [-100, -100, 8]},
        ]

        right = collate_candidate_pair_batch(items, pad_token_id=0, padding_side="right")
        left = collate_candidate_pair_batch(items, pad_token_id=0, padding_side="left")

        self.assertEqual(right["input_ids"].tolist(), [[4, 7, 0], [5, 6, 8]])
        self.assertEqual(right["attention_mask"].tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(right["labels"].tolist(), [[-100, 7, -100], [-100, -100, 8]])
        self.assertEqual(left["input_ids"].tolist(), [[0, 4, 7], [5, 6, 8]])
        self.assertEqual(left["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(left["labels"].tolist(), [[-100, -100, 7], [-100, -100, 8]])

        tokenizer = FakeTokenizer()
        tokenizer.padding_side = "left"
        via_wrapper = CandidatePairBatchCollator(tokenizer)(items)
        self.assertTrue(torch.equal(via_wrapper["input_ids"], left["input_ids"]))

    def test_probability_mapping_uses_last_nonpad_logits_for_right_padding(self) -> None:
        logits = torch.zeros((2, 3, 10), dtype=torch.float32)
        logits[0, 0, 7], logits[0, 0, 8] = -20.0, 20.0  # distractor
        logits[0, 1, 7], logits[0, 1, 8] = 3.0, 1.0
        logits[1, 2, 7], logits[1, 2, 8] = 0.0, 2.0
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        probabilities = present_probabilities_from_logits(
            logits,
            mask,
            VerbalizerTokenIds(yes=7, no=8),
        )

        self.assertTrue(torch.allclose(probabilities, torch.tensor([0.880797, 0.119203]), atol=1e-6))

    def test_batch_scoring_uses_last_nonpad_logits_for_left_padding(self) -> None:
        logits = torch.zeros((2, 3, 10), dtype=torch.float32)
        logits[0, 1, 7], logits[0, 1, 8] = 20.0, -20.0  # distractor
        logits[0, 2, 7], logits[0, 2, 8] = 1.0, 3.0
        logits[1, 2, 7], logits[1, 2, 8] = 2.0, 0.0
        batch = {
            "input_ids": torch.tensor([[0, 4, 5], [6, 7, 8]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }

        scores = score_candidate_pair_batch(
            FakeModel(logits),
            batch,
            VerbalizerTokenIds(yes=7, no=8),
        )

        self.assertAlmostEqual(scores[0], 0.119203, places=6)
        self.assertAlmostEqual(scores[1], 0.880797, places=6)


if __name__ == "__main__":
    unittest.main()
