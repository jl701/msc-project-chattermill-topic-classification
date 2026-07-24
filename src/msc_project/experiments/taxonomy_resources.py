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
RICH_GUIDANCE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "fabsa_aspect_rich_taxonomy_guidance_v1.json"
)
MINIMAL_DESCRIPTION_VERSION = "fabsa_aspect_descriptions_minimal_v2"
RICH_GUIDANCE_VERSION = "fabsa_aspect_rich_taxonomy_guidance_v1"
DESCRIPTION_BUNDLE_VERSION = "fabsa_description_bundle_v1"
DESCRIPTION_STATUSES = ("pending_user_approval", "approved_and_frozen")
CANDIDATE_SENTIMENTS = ("negative", "neutral", "positive")
REPRESENTATIONS = ("name_only", "minimal", "rich")


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
    if resource.get("version") == DESCRIPTION_BUNDLE_VERSION:
        return description_bundle_hash_payload(resource)
    return {
        "version": resource.get("version"),
        "canonical_order": resource.get("canonical_order"),
        "aspects": resource.get("aspects"),
    }


def rich_guidance_hash_payload(resource: Mapping[str, object]) -> dict[str, object]:
    return {
        "version": resource.get("version"),
        "template": resource.get("template"),
        "canonical_order": resource.get("canonical_order"),
        "aspects": resource.get("aspects"),
    }


def description_bundle_hash_payload(
    resource: Mapping[str, object],
) -> dict[str, object]:
    return {
        "version": resource.get("version"),
        "canonical_order": resource.get("canonical_order"),
        "minimal_resource_version": resource.get("minimal_resource_version"),
        "minimal_resource_sha256": resource.get("minimal_resource_sha256"),
        "rich_resource_version": resource.get("rich_resource_version"),
        "rich_resource_sha256": resource.get("rich_resource_sha256"),
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


def load_rich_taxonomy_guidance(
    path: Path | None = None,
    *,
    require_approved: bool = False,
) -> dict[str, object]:
    resource = _read_json_object(path or RICH_GUIDANCE_PATH)
    if resource.get("version") != RICH_GUIDANCE_VERSION:
        raise ValueError("Unexpected rich-guidance version.")
    status = resource.get("status")
    if status not in DESCRIPTION_STATUSES:
        raise ValueError(f"Unexpected rich-guidance status: {status!r}")
    if require_approved and status != "approved_and_frozen":
        raise ValueError("Formal runs require approved_and_frozen rich guidance.")

    canonical_order = resource.get("canonical_order")
    aspects = resource.get("aspects")
    template = resource.get("template")
    if not isinstance(canonical_order, list) or len(canonical_order) != 12:
        raise ValueError("Rich guidance requires exactly twelve canonical labels.")
    if (
        len(set(canonical_order)) != len(canonical_order)
        or not all(isinstance(aspect, str) and ":" in aspect for aspect in canonical_order)
    ):
        raise ValueError("Rich-guidance canonical_order is invalid.")
    if not isinstance(aspects, dict) or list(aspects) != canonical_order:
        raise ValueError("Rich cards must exactly match canonical_order and insertion order.")
    if not isinstance(template, dict):
        raise ValueError("Rich guidance must record one common template.")
    expected_fields = [
        "canonical_name",
        "definition",
        "aliases",
        "inclusion_boundary",
        "contrastive_boundary",
        "candidate_sentiment",
    ]
    if template.get("field_order") != expected_fields:
        raise ValueError("Rich-guidance template field order is not canonical.")
    alias_min = template.get("alias_count_min")
    alias_max = template.get("alias_count_max")
    if not isinstance(alias_min, int) or not isinstance(alias_max, int):
        raise ValueError("Rich-guidance alias bounds must be integers.")
    if alias_min < 1 or alias_max < alias_min:
        raise ValueError("Rich-guidance alias bounds are invalid.")
    for aspect, raw_card in aspects.items():
        if not isinstance(raw_card, dict):
            raise ValueError(f"Rich card for {aspect!r} must be an object.")
        required_fields = {
            "definition",
            "aliases",
            "inclusion_boundary",
            "contrastive_boundary",
        }
        if set(raw_card) != required_fields:
            raise ValueError(
                f"Rich card for {aspect!r} must contain exactly {sorted(required_fields)}."
            )
        if not isinstance(raw_card["definition"], str) or not raw_card[
            "definition"
        ].strip():
            raise ValueError(f"Rich card for {aspect!r} lacks a definition.")
        aliases = raw_card["aliases"]
        if not isinstance(aliases, list) or not alias_min <= len(aliases) <= alias_max:
            raise ValueError(
                f"Rich card for {aspect!r} must contain {alias_min}-{alias_max} aliases."
            )
        if (
            not all(isinstance(alias, str) and alias.strip() for alias in aliases)
            or len({alias.strip().casefold() for alias in aliases}) != len(aliases)
        ):
            raise ValueError(f"Rich card for {aspect!r} has invalid aliases.")
        for boundary in ("inclusion_boundary", "contrastive_boundary"):
            if not isinstance(raw_card[boundary], str) or not raw_card[
                boundary
            ].strip():
                raise ValueError(f"Rich card for {aspect!r} lacks {boundary}.")

    allowed = resource.get("allowed_sources")
    forbidden = resource.get("forbidden_sources")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("Rich guidance must record allowed_sources.")
    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError("Rich guidance must record forbidden_sources.")

    expected_hash = resource.get("content_sha256")
    observed_hash = canonical_json_sha256(rich_guidance_hash_payload(resource))
    if not isinstance(expected_hash, str) or expected_hash != observed_hash:
        raise ValueError(
            "Rich-guidance content hash mismatch: "
            f"expected={expected_hash!r}, observed={observed_hash!r}."
        )
    return resource


def load_description_bundle(
    *,
    minimal_path: Path | None = None,
    rich_path: Path | None = None,
    require_approved: bool = False,
) -> dict[str, object]:
    """Load and bind the frozen minimal and rich label-side resources."""

    minimal = load_minimal_descriptions(
        minimal_path,
        require_approved=require_approved,
    )
    rich = load_rich_taxonomy_guidance(
        rich_path,
        require_approved=require_approved,
    )
    if minimal["canonical_order"] != rich["canonical_order"]:
        raise ValueError("Minimal and rich resources use different canonical orders.")
    minimal_aspects = minimal["aspects"]
    rich_aspects = rich["aspects"]
    if not isinstance(minimal_aspects, Mapping) or not isinstance(
        rich_aspects, Mapping
    ):
        raise ValueError("Description resources have invalid aspect mappings.")
    mismatched_definitions = [
        aspect
        for aspect in minimal["canonical_order"]
        if rich_aspects[aspect]["definition"] != minimal_aspects[aspect]  # type: ignore[index]
    ]
    if mismatched_definitions:
        raise ValueError(
            "Rich cards must preserve the exact minimal definitions: "
            f"{mismatched_definitions}"
        )
    status = (
        "approved_and_frozen"
        if minimal["status"] == rich["status"] == "approved_and_frozen"
        else "pending_user_approval"
    )
    bundle: dict[str, object] = {
        "version": DESCRIPTION_BUNDLE_VERSION,
        "status": status,
        "canonical_order": list(minimal["canonical_order"]),
        "minimal_aspects": dict(minimal_aspects),
        "rich_aspects": dict(rich_aspects),
        "minimal_resource_version": minimal["version"],
        "minimal_resource_sha256": minimal["content_sha256"],
        "rich_resource_version": rich["version"],
        "rich_resource_sha256": rich["content_sha256"],
        "allowed_sources": list(minimal["allowed_sources"]),
        "forbidden_sources": sorted(
            {
                *[str(value) for value in minimal["forbidden_sources"]],
                *[str(value) for value in rich["forbidden_sources"]],
            }
        ),
    }
    bundle["content_sha256"] = canonical_json_sha256(
        description_bundle_hash_payload(bundle)
    )
    return bundle


def _minimal_aspects(resource: Mapping[str, object]) -> Mapping[str, object]:
    aspects = resource.get("minimal_aspects", resource.get("aspects"))
    if not isinstance(aspects, Mapping):
        raise ValueError("Description resource lacks minimal aspect definitions.")
    return aspects


def _rich_aspects(resource: Mapping[str, object]) -> Mapping[str, object]:
    aspects = resource.get("rich_aspects")
    if not isinstance(aspects, Mapping):
        raise ValueError("Description resource lacks rich taxonomy guidance.")
    return aspects


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
    descriptions = (
        dict(resource) if resource is not None else load_description_bundle()
    )
    aspects = _minimal_aspects(descriptions)
    if aspect not in aspects:
        raise ValueError(f"Unknown canonical aspect: {aspect!r}")

    claim = f"Aspect: {aspect}."
    if representation == "minimal":
        claim += f" Definition: {aspects[aspect]}"
    elif representation == "rich":
        rich_aspects = _rich_aspects(descriptions)
        raw_card = rich_aspects.get(aspect)
        if not isinstance(raw_card, Mapping):
            raise ValueError(f"Missing rich card for canonical aspect: {aspect!r}")
        aliases = raw_card.get("aliases")
        if not isinstance(aliases, list):
            raise ValueError(f"Rich card aliases are invalid for {aspect!r}.")
        claim += (
            f" Definition: {raw_card['definition']}"
            f" Aliases: {'; '.join(str(value) for value in aliases)}."
            f" Inclusion boundary: {raw_card['inclusion_boundary']}"
            f" Contrastive boundary: {raw_card['contrastive_boundary']}"
        )
    return f"{claim} Candidate sentiment: {sentiment}."


def representation_sha256(
    representation: str,
    resource: Mapping[str, object] | None = None,
) -> str:
    descriptions = (
        dict(resource) if resource is not None else load_description_bundle()
    )
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

    descriptions = (
        dict(resource) if resource is not None else load_description_bundle()
    )
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
