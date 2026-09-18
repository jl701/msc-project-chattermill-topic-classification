from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_resources import (
    CANDIDATE_SENTIMENTS,
    canonical_json_sha256,
    description_bundle_hash_payload,
    description_hash_payload,
    load_description_bundle,
    load_minimal_descriptions,
    load_rich_taxonomy_guidance,
    mixed_representation_sha256,
    render_candidate_claim,
    rich_guidance_hash_payload,
    representation_sha256,
)


def write_resource(tmp_path: Path, resource: dict[str, object]) -> Path:
    path = tmp_path / "descriptions.json"
    path.write_text(json.dumps(resource, indent=2), encoding="utf-8")
    return path


def test_minimal_resource_has_exact_coverage_order_and_hash() -> None:
    resource = load_minimal_descriptions()
    assert len(resource["canonical_order"]) == 12
    assert list(resource["aspects"]) == resource["canonical_order"]
    assert resource["content_sha256"] == canonical_json_sha256(
        description_hash_payload(resource)
    )
    assert resource["status"] == "approved_and_frozen"


def test_formal_run_accepts_frozen_description_resource() -> None:
    assert load_minimal_descriptions(require_approved=True)["status"] == (
        "approved_and_frozen"
    )


def test_formal_run_rejects_pending_description_resource(tmp_path: Path) -> None:
    resource = copy.deepcopy(load_minimal_descriptions())
    resource["status"] = "pending_user_approval"
    path = write_resource(tmp_path, resource)
    with pytest.raises(ValueError, match="approved_and_frozen"):
        load_minimal_descriptions(path, require_approved=True)


def test_minimal_claim_differs_from_name_only_by_definition() -> None:
    resource = load_minimal_descriptions()
    aspect = resource["canonical_order"][0]
    name_only = render_candidate_claim(aspect, "negative", "name_only", resource)
    minimal = render_candidate_claim(aspect, "negative", "minimal", resource)
    assert name_only == f"Aspect: {aspect}. Candidate sentiment: negative."
    assert minimal.startswith(f"Aspect: {aspect}. Definition: ")
    assert minimal.endswith("Candidate sentiment: negative.")
    assert "Lexical cues:" not in minimal
    assert "Boundary:" not in minimal


def test_every_representation_covers_all_aspect_sentiment_claims() -> None:
    resource = load_description_bundle()
    for representation in ("name_only", "minimal", "rich"):
        claims = {
            render_candidate_claim(aspect, sentiment, representation, resource)
            for aspect in resource["canonical_order"]
            for sentiment in CANDIDATE_SENTIMENTS
        }
        assert len(claims) == 36


def test_rich_resource_has_uniform_cards_exact_definitions_and_hash() -> None:
    minimal = load_minimal_descriptions()
    rich = load_rich_taxonomy_guidance(require_approved=True)
    assert rich["canonical_order"] == minimal["canonical_order"]
    assert rich["content_sha256"] == canonical_json_sha256(
        rich_guidance_hash_payload(rich)
    )
    for aspect in rich["canonical_order"]:
        card = rich["aspects"][aspect]
        assert card["definition"] == minimal["aspects"][aspect]
        assert 3 <= len(card["aliases"]) <= 5
        assert len({alias.casefold() for alias in card["aliases"]}) == len(
            card["aliases"]
        )
        assert card["inclusion_boundary"].strip()
        assert card["contrastive_boundary"].strip()


def test_description_bundle_binds_both_frozen_resources() -> None:
    bundle = load_description_bundle(require_approved=True)
    assert bundle["status"] == "approved_and_frozen"
    assert bundle["content_sha256"] == canonical_json_sha256(
        description_bundle_hash_payload(bundle)
    )
    assert bundle["minimal_resource_sha256"] != bundle["rich_resource_sha256"]


def test_rich_claim_uses_one_fixed_template_and_keeps_sentiment_separate() -> None:
    bundle = load_description_bundle()
    aspect = bundle["canonical_order"][0]
    value = render_candidate_claim(aspect, "neutral", "rich", bundle)
    assert value.startswith(f"Aspect: {aspect}. Definition: ")
    assert " Aliases: " in value
    assert " Inclusion boundary: " in value
    assert " Contrastive boundary: " in value
    assert value.endswith("Candidate sentiment: neutral.")


def test_content_change_without_hash_update_fails_closed(tmp_path: Path) -> None:
    resource = copy.deepcopy(load_minimal_descriptions())
    aspect = resource["canonical_order"][0]
    resource["aspects"][aspect] += " Changed."
    path = write_resource(tmp_path, resource)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_minimal_descriptions(path)


def test_representation_hash_is_deterministic_and_condition_specific() -> None:
    first = representation_sha256("minimal")
    assert first == representation_sha256("minimal")
    assert first != representation_sha256("name_only")


def test_mixed_representation_hash_tracks_per_aspect_variants() -> None:
    resource = load_minimal_descriptions()
    aspects = tuple(resource["canonical_order"][:2])
    sentiments = tuple(CANDIDATE_SENTIMENTS)
    name_minimal = {aspects[0]: "name_only", aspects[1]: "minimal"}
    minimal_name = {aspects[0]: "minimal", aspects[1]: "name_only"}
    assert mixed_representation_sha256(
        aspects, sentiments, name_minimal, resource
    ) == mixed_representation_sha256(aspects, sentiments, name_minimal, resource)
    assert mixed_representation_sha256(
        aspects, sentiments, name_minimal, resource
    ) != mixed_representation_sha256(aspects, sentiments, minimal_name, resource)
