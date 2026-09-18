from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.llm.candidate_label import (
    CostRates,
    aggregate_llm_diagnostics,
    build_candidate_messages,
    candidate_json_schema,
    estimate_cost,
    extract_usage,
    parse_candidate_output,
)


ASPECTS = [
    "Account management: Account access",
    "Company brand: Competitor",
    "Value: Discounts promotions",
]


class LlmCandidateLabelTest(unittest.TestCase):
    def test_build_indexed_object_prompt_for_json_mode(self) -> None:
        messages = build_candidate_messages(
            "The discount code worked.",
            ASPECTS,
            prompt_variant="indexed",
            output_container="object",
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("A3. Value: Discounts promotions", messages[1]["content"])
        self.assertIn('"labels"', messages[1]["content"])
        self.assertIn('"aspect_id"', messages[1]["content"])

    def test_build_indexed_prompt_with_generated_descriptions(self) -> None:
        description = "Mentions discounts, promotional offers, codes, rewards, cashback, or loyalty benefits."
        messages = build_candidate_messages(
            "The discount code worked.",
            ASPECTS,
            prompt_variant="indexed_generated_descriptions",
            output_container="object",
            aspect_descriptions={ASPECTS[2]: description},
        )

        prompt = messages[1]["content"]
        self.assertIn(f"A3. Value: Discounts promotions (description: {description})", prompt)
        self.assertNotIn("Value: Discounts promotions (keywords:", prompt)
        self.assertIn('"aspect_id"', prompt)

    def test_build_conservative_generated_description_prompt(self) -> None:
        messages = build_candidate_messages(
            "This app is much better than my old bank.",
            ASPECTS,
            prompt_variant="indexed_conservative_generated_descriptions",
            aspect_descriptions={ASPECTS[1]: "Mentions comparisons with competing banks, apps, or providers."},
        )

        prompt = messages[1]["content"]
        self.assertIn("A2. Company brand: Competitor (description:", prompt)
        self.assertIn("Most reviews match at most one candidate aspect.", prompt)

    def test_candidate_json_schema_uses_labels_wrapper(self) -> None:
        schema = candidate_json_schema()

        self.assertEqual(schema["type"], "json_schema")
        self.assertEqual(schema["json_schema"]["schema"]["required"], ["labels"])

    def test_parse_valid_indexed_array(self) -> None:
        parsed = parse_candidate_output(
            '[{"aspect_id": "A1", "sentiment": "Negative"}]',
            ASPECTS,
            require_aspect_id=True,
        )

        self.assertTrue(parsed.valid_json)
        self.assertTrue(parsed.schema_valid)
        self.assertEqual(parsed.pair_labels, ["Account management: Account access | negative"])

    def test_parse_valid_object_wrapper(self) -> None:
        parsed = parse_candidate_output(
            '{"labels": [{"aspect_id": "A2", "sentiment": "positive"}]}',
            ASPECTS,
            require_aspect_id=True,
        )

        self.assertTrue(parsed.valid_json)
        self.assertTrue(parsed.schema_valid)
        self.assertEqual(parsed.pair_labels, ["Company brand: Competitor | positive"])

    def test_parse_invalid_json_scores_empty(self) -> None:
        parsed = parse_candidate_output("not json", ASPECTS, require_aspect_id=True)

        self.assertFalse(parsed.valid_json)
        self.assertFalse(parsed.schema_valid)
        self.assertEqual(parsed.pair_labels, [])
        self.assertIsNotNone(parsed.parse_error)

    def test_parse_drops_invalid_id_and_sentiment(self) -> None:
        parsed = parse_candidate_output(
            '[{"aspect_id": "A9", "sentiment": "positive"}, {"aspect_id": "A1", "sentiment": "mixed"}]',
            ASPECTS,
            require_aspect_id=True,
        )

        self.assertTrue(parsed.valid_json)
        self.assertFalse(parsed.schema_valid)
        self.assertEqual(parsed.pair_labels, [])
        self.assertEqual(parsed.diagnostics.invalid_candidate_count, 1)
        self.assertEqual(parsed.diagnostics.invalid_sentiment_count, 1)

    def test_parse_deduplicates_and_records_conflicts(self) -> None:
        parsed = parse_candidate_output(
            (
                "["
                '{"aspect_id": "A1", "sentiment": "positive"},'
                '{"aspect_id": "A1", "sentiment": "positive"},'
                '{"aspect_id": "A1", "sentiment": "negative"}'
                "]"
            ),
            ASPECTS,
            require_aspect_id=True,
        )

        self.assertFalse(parsed.schema_valid)
        self.assertEqual(
            parsed.pair_labels,
            [
                "Account management: Account access | positive",
                "Account management: Account access | negative",
            ],
        )
        self.assertEqual(parsed.diagnostics.duplicate_prediction_count, 1)
        self.assertEqual(parsed.diagnostics.conflicting_sentiment_count, 1)

    def test_parse_accepts_exact_aspect_name_but_marks_indexed_schema_invalid(self) -> None:
        parsed = parse_candidate_output(
            '[{"aspect": "Value: Discounts promotions", "sentiment": "positive"}]',
            ASPECTS,
            require_aspect_id=True,
        )

        self.assertTrue(parsed.valid_json)
        self.assertFalse(parsed.schema_valid)
        self.assertEqual(parsed.pair_labels, ["Value: Discounts promotions | positive"])
        self.assertEqual(parsed.diagnostics.aspect_name_reference_count, 1)

    def test_parse_recovers_prefixed_aspect_id_but_marks_schema_invalid(self) -> None:
        parsed = parse_candidate_output(
            '[{"aspect_id": "A1. Account management: Account access", "sentiment": "neutral"}]',
            ASPECTS,
            require_aspect_id=True,
        )

        self.assertTrue(parsed.valid_json)
        self.assertFalse(parsed.schema_valid)
        self.assertEqual(parsed.pair_labels, ["Account management: Account access | neutral"])
        self.assertEqual(parsed.diagnostics.prefixed_aspect_id_count, 1)

    def test_extract_usage_supports_openai_and_gemini_fields(self) -> None:
        openai_usage = extract_usage(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                }
            }
        )
        gemini_usage = extract_usage(
            {
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 4,
                    "thoughtsTokenCount": 3,
                    "totalTokenCount": 18,
                }
            }
        )

        self.assertEqual(openai_usage.reasoning_tokens, 2)
        self.assertEqual(gemini_usage.input_tokens, 11)
        self.assertEqual(gemini_usage.reasoning_tokens, 3)

    def test_estimate_cost_requires_token_and_rate_information(self) -> None:
        usage = extract_usage({"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})

        self.assertAlmostEqual(
            estimate_cost(usage, CostRates(input_per_1m=0.10, output_per_1m=0.40)),
            0.00003,
        )
        self.assertIsNone(estimate_cost(usage, CostRates(input_per_1m=0.10)))

    def test_aggregate_llm_diagnostics(self) -> None:
        rows = [
            {
                "valid_json": True,
                "schema_valid": True,
                "pred_pair_labels": ["A | positive"],
                "seconds": 1.0,
                "input_tokens": 10,
                "output_tokens": 5,
                "reasoning_tokens": 2,
                "total_tokens": 17,
            },
            {
                "valid_json": False,
                "schema_valid": False,
                "pred_pair_labels": [],
                "seconds": 3.0,
                "invalid_candidate_count": 1,
            },
        ]

        summary = aggregate_llm_diagnostics(rows)

        self.assertAlmostEqual(summary["valid_json_rate"], 0.5)
        self.assertAlmostEqual(summary["median_latency_seconds"], 2.0)
        self.assertEqual(summary["invalid_candidate_label_count"], 1)
        self.assertEqual(summary["input_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
