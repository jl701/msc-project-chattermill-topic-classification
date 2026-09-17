from __future__ import annotations

import numpy as np

from msc_project.experiments.taxonomy_resources import load_description_bundle
from msc_project.experiments.taxonomy_rich_description import (
    INTERFACES,
    audit_rich_resource,
    development_partition,
    rich_aspect_fields,
    score_interfaces,
    select_f1_threshold,
    select_global_rich_interface,
)


def test_development_partition_is_deterministic_and_bounded() -> None:
    observed = [development_partition(f"train:{index}") for index in range(100)]
    assert observed == [development_partition(f"train:{index}") for index in range(100)]
    assert set(observed) == {0, 1, 2}


def test_positive_interface_keeps_exclusion_text_separate() -> None:
    resource = load_description_bundle(require_approved=True)
    aspect = str(resource["canonical_order"][8])
    fields = rich_aspect_fields(aspect, resource)
    assert "Contrastive boundary" not in fields.positive_concat
    assert "Exclude when" in fields.exclusion
    assert fields.exclusion not in fields.positive_concat


def test_interface_scoring_uses_registered_order_and_penalties() -> None:
    resource = load_description_bundle(require_approved=True)
    aspect = str(resource["canonical_order"][0])
    fields = rich_aspect_fields(aspect, resource)
    values = {
        fields.base: 0.4,
        fields.full_concat: 0.3,
        fields.positive_concat: 0.5,
        fields.inclusion: 0.6,
        fields.exclusion: 0.7,
        **{alias: 0.1 + index * 0.1 for index, alias in enumerate(fields.aliases)},
    }

    def score(_reviews: list[str], candidate: str) -> np.ndarray:
        return np.asarray([values[candidate], values[candidate]], dtype=float)

    result = score_interfaces(["a", "b"], fields, score)
    assert tuple(result) == INTERFACES
    assert np.allclose(result["R3_prototype_max"], 0.6)
    assert np.allclose(result["R4_prototype_top2"], 0.55)
    assert np.allclose(result["R5_contrastive_top2_l0.10"], 0.535)
    assert np.allclose(result["R5_contrastive_top2_l0.50"], 0.475)


def test_threshold_selection_uses_higher_threshold_on_f1_tie() -> None:
    result = select_f1_threshold(
        np.asarray([1, 0, 1, 0], dtype=int),
        np.asarray([0.9, 0.8, 0.7, 0.1], dtype=float),
    )
    assert result["threshold"] == 0.7
    assert result["f1"] > 0.79


def test_global_selection_is_one_interface_not_per_method() -> None:
    resource = load_description_bundle(require_approved=True)
    aspects = [str(value) for value in resource["canonical_order"][:2]]
    records = []
    for method in ("tfidf", "e5"):
        for aspect in aspects:
            for partition in range(3):
                for interface in INTERFACES:
                    delta = 0.0
                    if interface == "R2_positive_concat":
                        delta = 0.02
                    elif interface == "R4_prototype_top2":
                        delta = 0.023
                    records.append(
                        {
                            "method_id": method,
                            "pseudo_unseen_aspect": aspect,
                            "row_partition": partition,
                            "interface": interface,
                            "average_precision": 0.4 + delta,
                        }
                    )
    selected = select_global_rich_interface(records)
    assert selected["selected_interface"] == "R2_positive_concat"
    assert "R4_prototype_top2" in selected["near_tie_candidates"]


def test_resource_audit_is_aggregate_and_deterministic() -> None:
    resource = load_description_bundle(require_approved=True)
    first = audit_rich_resource(resource)
    second = audit_rich_resource(resource)
    assert first == second
    assert len(first["aspects"]) == 12
    assert len(first["pairwise_positive_overlap"]) == 66
    assert first["aspects_with_cross_label_exclusion_mentions"] > 0
