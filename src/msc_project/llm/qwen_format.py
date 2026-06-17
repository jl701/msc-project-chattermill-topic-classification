from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pandas as pd


SENTIMENTS = ("positive", "negative", "neutral")


SYSTEM_PROMPT = (
    "You are an aspect-based sentiment analysis tagger. "
    "Select only canonical FABSA aspects from the provided taxonomy. "
    "Return valid JSON only."
)


@dataclass(frozen=True)
class ParsedOutput:
    pair_labels: list[str]
    valid_json: bool
    error: str | None = None


def aspect_taxonomy(train_df: pd.DataFrame) -> list[str]:
    aspects: set[str] = set()
    for labels in train_df["labels"]:
        aspects.update(aspect for aspect, _ in labels)
    return sorted(aspects)


def format_pair_label(aspect: str, sentiment: str) -> str:
    return f"{aspect} | {sentiment}"


def format_gold_json(pair_labels: list[str]) -> str:
    rows = []
    for label in pair_labels:
        aspect, sentiment = label.rsplit(" | ", maxsplit=1)
        rows.append({"aspect": aspect, "sentiment": sentiment})
    return json.dumps(rows, ensure_ascii=False)


def format_candidate_gold_json(pair_labels: list[str], aspects: list[str], prompt_variant: str) -> str:
    if not prompt_variant.startswith("indexed"):
        return format_gold_json(pair_labels)

    aspect_to_id = {aspect: aspect_id(index) for index, aspect in enumerate(aspects)}
    rows = []
    for label in pair_labels:
        aspect, sentiment = label.rsplit(" | ", maxsplit=1)
        if aspect in aspect_to_id:
            rows.append({"aspect_id": aspect_to_id[aspect], "sentiment": sentiment})
    return json.dumps(rows, ensure_ascii=False)


def build_user_prompt(text: str, aspects: list[str]) -> str:
    taxonomy = "\n".join(f"- {aspect}" for aspect in aspects)
    return (
        "Review:\n"
        f"{text}\n\n"
        "Candidate aspects:\n"
        f"{taxonomy}\n\n"
        "For each relevant aspect, choose exactly one sentiment from: "
        "positive, negative, neutral.\n"
        "Return a JSON array of objects with keys \"aspect\" and \"sentiment\". "
        "Use no explanations and no extra text."
    )


def aspect_id(index: int) -> str:
    return f"A{index + 1}"


def aspect_line(aspect: str, index: int | None = None, with_keywords: bool = False) -> str:
    prefix = f"{aspect_id(index)}. " if index is not None else "- "
    if not with_keywords:
        return f"{prefix}{aspect}"

    leaf = aspect.split(":")[-1].strip()
    return f"{prefix}{aspect} (keywords: {leaf})"


def build_candidate_user_prompt(
    text: str,
    aspects: list[str],
    prompt_variant: str = "standard",
) -> str:
    if prompt_variant not in {
        "standard",
        "conservative",
        "descriptive",
        "conservative_descriptive",
        "indexed",
        "indexed_conservative",
        "indexed_descriptive",
        "indexed_conservative_descriptive",
    }:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")

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
    indexed = prompt_variant in {
        "indexed",
        "indexed_conservative",
        "indexed_descriptive",
        "indexed_conservative_descriptive",
    }
    taxonomy = "\n".join(
        aspect_line(aspect, index=index if indexed else None, with_keywords=with_keywords)
        for index, aspect in enumerate(aspects)
    )
    if indexed:
        output_keys = "\"aspect_id\" and \"sentiment\""
        output_constraint = (
            "Use only the exact aspect_id values from the candidate list, such as A1. "
            "Do not output aspect names.\n"
        )
    else:
        output_keys = "\"aspect\" and \"sentiment\""
        output_constraint = "Use only the exact candidate aspect strings. "

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

    return (
        "Review:\n"
        f"{text}\n\n"
        "Candidate aspects:\n"
        f"{taxonomy}\n\n"
        f"{guidance}"
        f"Return a JSON array of objects with keys {output_keys}. "
        f"{output_constraint}"
        "Use no explanations and no extra text."
    )


def build_messages(text: str, aspects: list[str], gold_pair_labels: list[str] | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text, aspects)},
    ]
    if gold_pair_labels is not None:
        messages.append({"role": "assistant", "content": format_gold_json(gold_pair_labels)})
    return messages


def build_candidate_messages(
    text: str,
    aspects: list[str],
    prompt_variant: str = "standard",
    gold_pair_labels: list[str] | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_candidate_user_prompt(text, aspects, prompt_variant)},
    ]
    if gold_pair_labels is not None:
        messages.append({"role": "assistant", "content": format_candidate_gold_json(gold_pair_labels, aspects, prompt_variant)})
    return messages


def json_from_text(text: str):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", stripped)
        if match:
            return json.loads(match.group(0))
        raise


def parse_model_output(text: str, allowed_aspects: list[str]) -> ParsedOutput:
    allowed = set(allowed_aspects)
    allowed_ids = {aspect_id(index): aspect for index, aspect in enumerate(allowed_aspects)}
    numeric_ids = {str(index + 1): aspect for index, aspect in enumerate(allowed_aspects)}
    labels: list[str] = []

    try:
        parsed = json_from_text(text)
    except Exception as exc:
        return ParsedOutput(pair_labels=[], valid_json=False, error=str(exc))

    if not isinstance(parsed, list):
        return ParsedOutput(pair_labels=[], valid_json=False, error="Top-level JSON value is not a list.")

    for item in parsed:
        if not isinstance(item, dict):
            continue

        aspect = str(item.get("aspect", "")).strip()
        raw_aspect_id = str(item.get("aspect_id", "")).strip()
        if aspect not in allowed and raw_aspect_id:
            aspect = allowed_ids.get(raw_aspect_id.upper(), numeric_ids.get(raw_aspect_id, aspect))
        sentiment = str(item.get("sentiment", "")).strip().lower()

        if aspect not in allowed or sentiment not in SENTIMENTS:
            continue

        label = format_pair_label(aspect, sentiment)
        if label not in labels:
            labels.append(label)

    return ParsedOutput(pair_labels=labels, valid_json=True)
