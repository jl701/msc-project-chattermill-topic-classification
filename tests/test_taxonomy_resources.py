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
    description_hash_payload,
    load_minimal_descriptions,
    mixed_representation_sha256,
    render_candidate_claim,
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
    assert resource["status"] == "pending_user_approval"


def test_formal_run_rejects_pending_description_resource() -> None:
    with pytest.raises(ValueError, match="approved_and_frozen"):
        load_minimal_descriptions(require_approved=True)


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
    resource = load_minimal_descriptions()
    for representation in ("name_only", "minimal"):
        claims = {
            render_candidate_claim(aspect, sentiment, representation, resource)
            for aspect in resource["canonical_order"]
            for sentiment in CANDIDATE_SENTIMENTS
        }
        assert len(claims) == 36


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
