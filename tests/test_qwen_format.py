from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.llm.qwen_format import build_candidate_messages, build_messages, format_gold_json, parse_model_output


class QwenFormatTest(unittest.TestCase):
    def test_format_gold_json(self) -> None:
        text = format_gold_json(["Online experience: App website | negative"])
        self.assertEqual(
            text,
            '[{"aspect": "Online experience: App website", "sentiment": "negative"}]',
        )

    def test_parse_model_output(self) -> None:
        parsed = parse_model_output(
            '[{"aspect": "Online experience: App website", "sentiment": "Negative"}]',
            ["Online experience: App website"],
        )
        self.assertTrue(parsed.valid_json)
        self.assertEqual(parsed.pair_labels, ["Online experience: App website | negative"])

    def test_parse_filters_unknown_aspects(self) -> None:
        parsed = parse_model_output(
            '[{"aspect": "App usability", "sentiment": "negative"}]',
            ["Online experience: App website"],
        )
        self.assertTrue(parsed.valid_json)
        self.assertEqual(parsed.pair_labels, [])

    def test_parse_model_output_accepts_aspect_id(self) -> None:
        parsed = parse_model_output(
            '[{"aspect_id": "A2", "sentiment": "positive"}]',
            ["Account management: Account access", "Value: Discounts promotions"],
        )
        self.assertTrue(parsed.valid_json)
        self.assertEqual(parsed.pair_labels, ["Value: Discounts promotions | positive"])

    def test_build_messages(self) -> None:
        messages = build_messages(
            "The app crashes.",
            ["Online experience: App website"],
            ["Online experience: App website | negative"],
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])

    def test_build_candidate_messages_conservative_descriptive(self) -> None:
        messages = build_candidate_messages(
            "The discount code worked.",
            ["Value: Discounts promotions"],
            prompt_variant="conservative_descriptive",
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("keywords: Discounts promotions", messages[1]["content"])
        self.assertIn("return an empty JSON array []", messages[1]["content"])

    def test_build_candidate_messages_indexed(self) -> None:
        messages = build_candidate_messages(
            "The discount code worked.",
            ["Value: Discounts promotions"],
            prompt_variant="indexed",
        )
        self.assertIn("A1. Value: Discounts promotions", messages[1]["content"])
        self.assertIn('"aspect_id"', messages[1]["content"])

    def test_build_candidate_messages_indexed_gold_uses_aspect_id(self) -> None:
        messages = build_candidate_messages(
            "The discount code worked.",
            ["Value: Discounts promotions"],
            prompt_variant="indexed",
            gold_pair_labels=["Value: Discounts promotions | positive"],
        )
        self.assertEqual(messages[2]["content"], '[{"aspect_id": "A1", "sentiment": "positive"}]')

    def test_build_candidate_messages_indexed_descriptive(self) -> None:
        messages = build_candidate_messages(
            "The discount code worked.",
            ["Value: Discounts promotions"],
            prompt_variant="indexed_descriptive",
        )
        self.assertIn("A1. Value: Discounts promotions (keywords: Discounts promotions)", messages[1]["content"])
        self.assertIn('"aspect_id"', messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
