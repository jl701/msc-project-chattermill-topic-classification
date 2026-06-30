from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from statistics import mean, median
from typing import Any


SENTIMENTS = ("positive", "negative", "neutral")

SYSTEM_PROMPT = (
    "You are an aspect-based sentiment analysis tagger. "
    "Select only canonical FABSA aspects from the provided taxonomy. "
    "Return valid JSON only."
)

PROMPT_VARIANTS = (
    "standard",
    "conservative",
    "descriptive",
    "conservative_descriptive",
    "indexed",
    "indexed_conservative",
    "indexed_descriptive",
    "indexed_conservative_descriptive",
)


@dataclass(frozen=True)
class CandidateParseDiagnostics:
    valid_json: bool
    schema_valid: bool
    parse_error: str | None = None
    strict_json: bool = True
    extracted_json: bool = False
    top_level_type: str | None = None
    item_count: int = 0
    valid_item_count: int = 0
    non_object_item_count: int = 0
    invalid_candidate_count: int = 0
    invalid_sentiment_count: int = 0
    duplicate_prediction_count: int = 0
    conflicting_sentiment_count: int = 0
    aspect_name_reference_count: int = 0
    numeric_aspect_id_count: int = 0


@dataclass(frozen=True)
class ParsedCandidateOutput:
    pair_labels: list[str]
    diagnostics: CandidateParseDiagnostics

    @property
    def valid_json(self) -> bool:
        return self.diagnostics.valid_json

    @property
    def schema_valid(self) -> bool:
        return self.diagnostics.schema_valid

    @property
    def parse_error(self) -> str | None:
        return self.diagnostics.parse_error


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostRates:
    input_per_1m: float | None = None
    output_per_1m: float | None = None
    reasoning_per_1m: float | None = None


def aspect_id(index: int) -> str:
    return f"A{index + 1}"


def aspect_line(aspect: str, index: int | None = None, with_keywords: bool = False) -> str:
    prefix = f"{aspect_id(index)}. " if index is not None else "- "
    if not with_keywords:
        return f"{prefix}{aspect}"

    leaf = aspect.split(":")[-1].strip()
    return f"{prefix}{aspect} (keywords: {leaf})"


def format_pair_label(aspect: str, sentiment: str) -> str:
    return f"{aspect} | {sentiment}"


def format_gold_json(
    pair_labels: list[str],
    aspects: list[str],
    indexed: bool = False,
    output_container: str = "array",
) -> str:
    rows = []
    aspect_to_id = {aspect: aspect_id(index) for index, aspect in enumerate(aspects)}
    for label in pair_labels:
        aspect, sentiment = label.rsplit(" | ", maxsplit=1)
        if indexed:
            if aspect not in aspect_to_id:
                continue
            rows.append({"aspect_id": aspect_to_id[aspect], "sentiment": sentiment})
        else:
            rows.append({"aspect": aspect, "sentiment": sentiment})

    if output_container == "object":
        return json.dumps({"labels": rows}, ensure_ascii=False)
    return json.dumps(rows, ensure_ascii=False)


def build_candidate_user_prompt(
    text: str,
    aspects: list[str],
    prompt_variant: str = "indexed",
    output_container: str = "array",
) -> str:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")
    if output_container not in {"array", "object"}:
        raise ValueError(f"Unknown output container: {output_container}")

    with_keywords = prompt_variant in {
        "descriptive",
        "conservative_descriptive",
        "indexed_descriptive",
        "indexed_conservative_descriptive",
    }
    conservative = prompt_variant in {
        "conservative",
        "conservative_descriptive",
        "indexed_conservative",
        "indexed_conservative_descriptive",
    }
    indexed = prompt_variant.startswith("indexed")

    taxonomy = "\n".join(
        aspect_line(aspect, index=index if indexed else None, with_keywords=with_keywords)
        for index, aspect in enumerate(aspects)
    )
    if indexed:
        output_keys = '"aspect_id" and "sentiment"'
        output_constraint = (
            "Use only the exact aspect_id values from the candidate list, such as A1. "
            "Do not output aspect names.\n"
        )
    else:
        output_keys = '"aspect" and "sentiment"'
        output_constraint = "Use only the exact candidate aspect strings.\n"

    guidance = (
        "For each relevant candidate aspect, choose exactly one sentiment from: "
        "positive, negative, neutral.\n"
    )
    if conservative:
        guidance += (
            "A review can match zero, one, or multiple candidate aspects. "
            "Most reviews match at most one candidate aspect. "
            "Include an aspect only when the review explicitly discusses that candidate. "
            "If no candidate aspect is relevant, return an empty JSON array [].\n"
        )

    if output_container == "object":
        output_instruction = (
            f"Return a JSON object with exactly one key, \"labels\". "
            f"The value of \"labels\" must be a JSON array of objects with keys {output_keys}. "
        )
        empty_instruction = "If no candidate aspect is relevant, return {\"labels\": []}. "
    else:
        output_instruction = f"Return a JSON array of objects with keys {output_keys}. "
        empty_instruction = "If no candidate aspect is relevant, return an empty JSON array []. "

    return (
        "Review:\n"
        f"{text}\n\n"
        "Candidate aspects:\n"
        f"{taxonomy}\n\n"
        f"{guidance}"
        f"{output_instruction}"
        f"{empty_instruction}"
        f"{output_constraint}"
        "Use no explanations and no extra text."
    )


def build_candidate_messages(
    text: str,
    aspects: list[str],
    prompt_variant: str = "indexed",
    gold_pair_labels: list[str] | None = None,
    output_container: str = "array",
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_candidate_user_prompt(
                text=text,
                aspects=aspects,
                prompt_variant=prompt_variant,
                output_container=output_container,
            ),
        },
    ]
    if gold_pair_labels is not None:
        messages.append(
            {
                "role": "assistant",
                "content": format_gold_json(
                    pair_labels=gold_pair_labels,
                    aspects=aspects,
                    indexed=prompt_variant.startswith("indexed"),
                    output_container=output_container,
                ),
            }
        )
    return messages


def candidate_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "fabsa_candidate_labels",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "aspect_id": {"type": "string"},
                                "sentiment": {"type": "string", "enum": list(SENTIMENTS)},
                            },
                            "required": ["aspect_id", "sentiment"],
                        },
                    }
                },
                "required": ["labels"],
            },
        },
    }


def json_from_text(text: str) -> tuple[Any, bool, bool]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped), True, False
    except json.JSONDecodeError as exc:
        decode_error = exc

    array_match = re.search(r"\[[\s\S]*\]", stripped)
    object_match = re.search(r"\{[\s\S]*\}", stripped)
    candidates = [match for match in (array_match, object_match) if match is not None]
    if not candidates:
        raise decode_error
    candidates.sort(key=lambda match: match.start())
    return json.loads(candidates[0].group(0)), False, True


def unwrap_prediction_items(parsed: Any, allow_object_wrapper: bool) -> tuple[list[Any] | None, str]:
    if isinstance(parsed, list):
        return parsed, "list"
    if allow_object_wrapper and isinstance(parsed, dict):
        for key in ("labels", "predictions", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value, "dict"
    return None, type(parsed).__name__


def parse_candidate_output(
    text: str,
    allowed_aspects: list[str],
    require_aspect_id: bool = False,
    allow_object_wrapper: bool = True,
) -> ParsedCandidateOutput:
    allowed = set(allowed_aspects)
    allowed_ids = {aspect_id(index): aspect for index, aspect in enumerate(allowed_aspects)}
    numeric_ids = {str(index + 1): aspect for index, aspect in enumerate(allowed_aspects)}
    labels: list[str] = []
    sentiments_by_aspect: dict[str, set[str]] = {}

    try:
        parsed, strict_json, extracted_json = json_from_text(text)
    except Exception as exc:
        return ParsedCandidateOutput(
            pair_labels=[],
            diagnostics=CandidateParseDiagnostics(
                valid_json=False,
                schema_valid=False,
                parse_error=str(exc),
                strict_json=False,
                extracted_json=False,
            ),
        )

    items, top_level_type = unwrap_prediction_items(parsed, allow_object_wrapper)
    if items is None:
        return ParsedCandidateOutput(
            pair_labels=[],
            diagnostics=CandidateParseDiagnostics(
                valid_json=True,
                schema_valid=False,
                parse_error="Top-level JSON value is not a list or supported labels object.",
                strict_json=strict_json,
                extracted_json=extracted_json,
                top_level_type=top_level_type,
            ),
        )

    non_object_count = 0
    invalid_candidate_count = 0
    invalid_sentiment_count = 0
    duplicate_count = 0
    conflict_count = 0
    aspect_name_count = 0
    numeric_id_count = 0
    valid_item_count = 0

    for item in items:
        if not isinstance(item, dict):
            non_object_count += 1
            continue

        raw_aspect = str(item.get("aspect", "")).strip()
        raw_aspect_id = str(item.get("aspect_id", "")).strip()
        aspect = raw_aspect if raw_aspect in allowed else ""
        used_aspect_name = bool(aspect)
        used_numeric_id = False

        if not aspect and raw_aspect_id:
            upper_id = raw_aspect_id.upper()
            if upper_id in allowed_ids:
                aspect = allowed_ids[upper_id]
            elif raw_aspect_id in numeric_ids:
                aspect = numeric_ids[raw_aspect_id]
                used_numeric_id = True

        sentiment = str(item.get("sentiment", "")).strip().lower()

        if used_aspect_name:
            aspect_name_count += 1
        if used_numeric_id:
            numeric_id_count += 1

        candidate_valid = aspect in allowed
        sentiment_valid = sentiment in SENTIMENTS
        if not candidate_valid:
            invalid_candidate_count += 1
        if not sentiment_valid:
            invalid_sentiment_count += 1
        if not candidate_valid or not sentiment_valid:
            continue

        valid_item_count += 1
        previous_sentiments = sentiments_by_aspect.setdefault(aspect, set())
        if previous_sentiments and sentiment not in previous_sentiments:
            conflict_count += 1
        previous_sentiments.add(sentiment)

        label = format_pair_label(aspect, sentiment)
        if label in labels:
            duplicate_count += 1
            continue
        labels.append(label)

    schema_valid = (
        strict_json
        and non_object_count == 0
        and invalid_candidate_count == 0
        and invalid_sentiment_count == 0
        and duplicate_count == 0
        and conflict_count == 0
        and (not require_aspect_id or aspect_name_count == 0)
        and numeric_id_count == 0
    )

    return ParsedCandidateOutput(
        pair_labels=labels,
        diagnostics=CandidateParseDiagnostics(
            valid_json=True,
            schema_valid=schema_valid,
            strict_json=strict_json,
            extracted_json=extracted_json,
            top_level_type=top_level_type,
            item_count=len(items),
            valid_item_count=valid_item_count,
            non_object_item_count=non_object_count,
            invalid_candidate_count=invalid_candidate_count,
            invalid_sentiment_count=invalid_sentiment_count,
            duplicate_prediction_count=duplicate_count,
            conflicting_sentiment_count=conflict_count,
            aspect_name_reference_count=aspect_name_count,
            numeric_aspect_id_count=numeric_id_count,
        ),
    )


def extract_usage(response: dict[str, Any]) -> TokenUsage:
    usage = response.get("usage") or response.get("usageMetadata") or {}
    if not isinstance(usage, dict):
        usage = {}

    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(completion_details, dict):
        completion_details = {}

    prompt_tokens = _first_int(
        usage.get("prompt_tokens"),
        usage.get("promptTokenCount"),
        usage.get("input_tokens"),
        usage.get("inputTokenCount"),
    )
    output_tokens = _first_int(
        usage.get("completion_tokens"),
        usage.get("candidatesTokenCount"),
        usage.get("output_tokens"),
        usage.get("outputTokenCount"),
    )
    reasoning_tokens = _first_int(
        completion_details.get("reasoning_tokens"),
        usage.get("reasoning_tokens"),
        usage.get("thoughtsTokenCount"),
        usage.get("thinkingTokenCount"),
    )
    total_tokens = _first_int(
        usage.get("total_tokens"),
        usage.get("totalTokenCount"),
        usage.get("total_token_count"),
    )
    if total_tokens is None:
        known = [value for value in (prompt_tokens, output_tokens, reasoning_tokens) if value is not None]
        if known:
            total_tokens = int(sum(known))

    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        raw_usage=usage,
    )


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def estimate_cost(usage: TokenUsage, rates: CostRates) -> float | None:
    if rates.input_per_1m is None or rates.output_per_1m is None:
        return None
    if usage.input_tokens is None or usage.output_tokens is None:
        return None

    total = usage.input_tokens * rates.input_per_1m / 1_000_000
    total += usage.output_tokens * rates.output_per_1m / 1_000_000
    if rates.reasoning_per_1m is not None and usage.reasoning_tokens is not None:
        total += usage.reasoning_tokens * rates.reasoning_per_1m / 1_000_000
    return float(total)


def diagnostic_dict(parsed: ParsedCandidateOutput) -> dict[str, Any]:
    return asdict(parsed.diagnostics)


def usage_dict(usage: TokenUsage) -> dict[str, Any]:
    return asdict(usage)


def aggregate_llm_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    seconds = [float(row["seconds"]) for row in rows if row.get("seconds") is not None]
    costs = [float(row["estimated_cost_usd"]) for row in rows if row.get("estimated_cost_usd") is not None]

    def sum_int(key: str) -> int | None:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if not values:
            return None
        return int(sum(int(value) for value in values))

    return {
        "valid_json_rate": sum(1 for row in rows if row.get("valid_json")) / total,
        "schema_valid_rate": sum(1 for row in rows if row.get("schema_valid")) / total,
        "parse_failure_count": sum(1 for row in rows if not row.get("valid_json")),
        "schema_invalid_count": sum(1 for row in rows if not row.get("schema_valid")),
        "invalid_candidate_label_count": int(sum(row.get("invalid_candidate_count", 0) for row in rows)),
        "invalid_sentiment_count": int(sum(row.get("invalid_sentiment_count", 0) for row in rows)),
        "duplicate_prediction_count": int(sum(row.get("duplicate_prediction_count", 0) for row in rows)),
        "conflicting_sentiment_count": int(sum(row.get("conflicting_sentiment_count", 0) for row in rows)),
        "non_object_item_count": int(sum(row.get("non_object_item_count", 0) for row in rows)),
        "aspect_name_reference_count": int(sum(row.get("aspect_name_reference_count", 0) for row in rows)),
        "numeric_aspect_id_count": int(sum(row.get("numeric_aspect_id_count", 0) for row in rows)),
        "empty_prediction_count": sum(1 for row in rows if not row.get("pred_pair_labels")),
        "mean_latency_seconds": float(mean(seconds)) if seconds else None,
        "median_latency_seconds": float(median(seconds)) if seconds else None,
        "input_tokens": sum_int("input_tokens"),
        "output_tokens": sum_int("output_tokens"),
        "reasoning_tokens": sum_int("reasoning_tokens"),
        "total_tokens": sum_int("total_tokens"),
        "estimated_cost_usd": float(sum(costs)) if costs else None,
    }
