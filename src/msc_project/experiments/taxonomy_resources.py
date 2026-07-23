"""Frozen label-side resources for strict taxonomy-generalisation experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MINIMAL_DESCRIPTION_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "fabsa_aspect_descriptions_minimal_v2.json"
)
MINIMAL_DESCRIPTION_VERSION = "fabsa_aspect_descriptions_minimal_v2"
DESCRIPTION_STATUSES = ("pending_user_approval", "approved_and_frozen")
CANDIDATE_SENTIMENTS = ("negative", "neutral", "positive")
REPRESENTATIONS = ("name_only", "minimal")


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def description_hash_payload(resource: Mapping[str, object]) -> dict[str, object]:
    return {
        "version": resource.get("version"),
        "canonical_order": resource.get("canonical_order"),
        "aspects": resource.get("aspects"),
    }


def load_minimal_descriptions(
    path: Path | None = None,
    *,
    require_approved: bool = False,
) -> dict[str, object]:
    resource = _read_json_object(path or MINIMAL_DESCRIPTION_PATH)
    if resource.get("version") != MINIMAL_DESCRIPTION_VERSION:
        raise ValueError("Unexpected minimal-description version.")
    status = resource.get("status")
    if status not in DESCRIPTION_STATUSES:
        raise ValueError(f"Unexpected minimal-description status: {status!r}")
    if require_approved and status != "approved_and_frozen":
        raise ValueError("Formal runs require an approved_and_frozen description resource.")

    canonical_order = resource.get("canonical_order")
    aspects = resource.get("aspects")
    if not isinstance(canonical_order, list) or len(canonical_order) != 12:
        raise ValueError("Minimal descriptions require exactly twelve labels in canonical_order.")
    if not all(isinstance(aspect, str) and ":" in aspect for aspect in canonical_order):
        raise ValueError("Every canonical aspect must be a non-empty hierarchical name.")
    if len(set(canonical_order)) != len(canonical_order):
        raise ValueError("canonical_order contains duplicate aspects.")
    if not isinstance(aspects, dict) or list(aspects) != canonical_order:
        raise ValueError("Aspect definitions must exactly match canonical_order and insertion order.")
    if not all(isinstance(text, str) and text.strip() for text in aspects.values()):
        raise ValueError("Every minimal aspect definition must be a non-empty string.")

    allowed = resource.get("allowed_sources")
    forbidden = resource.get("forbidden_sources")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("Minimal descriptions must record allowed_sources.")
    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError("Minimal descriptions must record forbidden_sources.")

    expected_hash = resource.get("content_sha256")
    observed_hash = canonical_json_sha256(description_hash_payload(resource))
    if not isinstance(expected_hash, str) or expected_hash != observed_hash:
        raise ValueError(
            "Minimal-description content hash mismatch: "
            f"expected={expected_hash!r}, observed={observed_hash!r}."
        )
    return resource


def render_candidate_claim(
    aspect: str,
    sentiment: str,
    representation: str,
    resource: Mapping[str, object] | None = None,
) -> str:
    if sentiment not in CANDIDATE_SENTIMENTS:
        raise ValueError(f"Unknown candidate sentiment: {sentiment!r}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown candidate representation: {representation!r}")
    descriptions = dict(resource) if resource is not None else load_minimal_descriptions()
    aspects = descriptions.get("aspects")
    if not isinstance(aspects, Mapping) or aspect not in aspects:
        raise ValueError(f"Unknown canonical aspect: {aspect!r}")

    claim = f"Aspect: {aspect}."
    if representation == "minimal":
        claim += f" Definition: {aspects[aspect]}"
    return f"{claim} Candidate sentiment: {sentiment}."


def representation_sha256(
    representation: str,
    resource: Mapping[str, object] | None = None,
) -> str:
    descriptions = dict(resource) if resource is not None else load_minimal_descriptions()
    canonical_order = descriptions.get("canonical_order")
    if not isinstance(canonical_order, list):
        raise ValueError("Description resource lacks canonical_order.")
    rendered = [
        render_candidate_claim(aspect, sentiment, representation, descriptions)
        for aspect in canonical_order
        for sentiment in CANDIDATE_SENTIMENTS
    ]
    return canonical_json_sha256(
        {
            "description_content_sha256": descriptions.get("content_sha256"),
            "representation": representation,
            "claims": rendered,
        }
    )


def mixed_representation_sha256(
    aspects: list[str] | tuple[str, ...],
    sentiments: list[str] | tuple[str, ...],
    representations: Mapping[str, str],
    resource: Mapping[str, object] | None = None,
) -> str:
    """Hash an ordered candidate set whose aspects may use different variants."""

    descriptions = dict(resource) if resource is not None else load_minimal_descriptions()
    canonical_order = descriptions.get("canonical_order")
    if not isinstance(canonical_order, list):
        raise ValueError("Description resource lacks canonical_order.")
    aspect_values = tuple(str(value) for value in aspects)
    sentiment_values = tuple(str(value) for value in sentiments)
    if not aspect_values or len(aspect_values) != len(set(aspect_values)):
        raise ValueError("aspects must be non-empty and unique.")
    unknown_aspects = sorted(set(aspect_values) - set(canonical_order))
    if unknown_aspects:
        raise ValueError(f"Unknown aspects in representation manifest: {unknown_aspects}")
    if sentiment_values != CANDIDATE_SENTIMENTS:
        raise ValueError(
            f"sentiments must use the canonical order {CANDIDATE_SENTIMENTS}."
        )
    if set(representations) != set(aspect_values):
        raise ValueError("representations must cover exactly the supplied aspects.")
    invalid = sorted(
        {
            str(representations[aspect])
            for aspect in aspect_values
            if representations[aspect] not in REPRESENTATIONS
        }
    )
    if invalid:
        raise ValueError(f"Unknown candidate representations: {invalid}")
    claims = [
        {
            "aspect": aspect,
            "sentiment": sentiment,
            "representation": representations[aspect],
            "claim": render_candidate_claim(
                aspect,
                sentiment,
                representations[aspect],
                descriptions,
            ),
        }
        for aspect in aspect_values
        for sentiment in sentiment_values
    ]
    return canonical_json_sha256(
        {
            "description_content_sha256": descriptions.get("content_sha256"),
            "claims": claims,
        }
    )
