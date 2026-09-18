"""Shared parameter, contract, and artifact-path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from msc_project.experiments.taxonomy_checkpoints import TrainingContract
from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_methods import (
    load_method_registry,
    method_registry_sha256,
    resolve_method_spec,
)
from msc_project.experiments.taxonomy_protocol import (
    TaxonomyFold,
    scientific_protocol_sha256,
    training_scope_id,
)
from msc_project.experiments.taxonomy_tuning import registered_tuning_candidates
from msc_project.experiments.unified_candidate_pairs import manifest_hash


def resolve_parameters(
    method_id: str,
    *,
    candidate_id: str | None,
    selection_path: Path | None,
    expected_training_scope_id: str | None = None,
) -> tuple[dict[str, object], str]:
    if (candidate_id is None) == (selection_path is None):
        raise ValueError(
            "Supply exactly one of --candidate-id or --parameter-selection."
        )
    candidates = registered_tuning_candidates(method_id)
    if candidate_id is not None:
        matches = [
            value
            for value in candidates
            if value["candidate_id"] == candidate_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Unknown registered candidate_id: {candidate_id!r}")
        parameters = dict(matches[0]["parameters"])
        return parameters, str(matches[0]["parameters_sha256"])

    assert selection_path is not None
    value = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("method_id") != method_id:
        raise ValueError("Parameter-selection artifact has the wrong method.")
    if value.get("selection_scope") not in {
        "registered_seen_validation_grid",
        "registered_fixed_recipe",
    }:
        raise ValueError("Parameter-selection artifact is not a registered result.")
    registered_scope = (
        resolve_method_spec(method_id)
        .seen_only_tuning.get("selection_scope")
    )
    observed_scope = value.get("training_scope_id")
    if registered_scope == "nested_per_training_scope":
        if value.get("selection_scope") != "registered_seen_validation_grid":
            raise ValueError("A nested method cannot use a global fixed recipe.")
        if expected_training_scope_id is None:
            raise ValueError("Nested selection requires an expected training scope.")
        if observed_scope != expected_training_scope_id:
            raise ValueError(
                "Parameter-selection artifact has the wrong training scope."
            )
    elif (
        value.get("selection_scope") != "registered_fixed_recipe"
        or observed_scope != "global_fixed_recipe"
    ):
        raise ValueError(
            "A fixed method requires the registered global fixed recipe."
        )
    parameters = value.get("selected_parameters")
    digest = value.get("selected_parameters_sha256")
    if not isinstance(parameters, dict) or digest != canonical_sha256(parameters):
        raise ValueError("Parameter-selection artifact hash mismatch.")
    registered_hashes = {
        item["parameters_sha256"]
        for item in candidates
    }
    if digest not in registered_hashes:
        raise ValueError("Selected parameters are outside the registered grid.")
    return dict(parameters), str(digest)


def build_training_contract(
    method_id: str,
    fold: TaxonomyFold,
    training_manifest: pd.DataFrame,
    parameters_sha256: str,
    *,
    seed: int,
) -> TrainingContract:
    registry = load_method_registry()
    spec = resolve_method_spec(method_id, registry)
    return TrainingContract(
        protocol_id="taxonomy_generalisation_precloud_v1",
        scientific_protocol_sha256=scientific_protocol_sha256(),
        method_id=method_id,
        method_spec_sha256=spec.spec_sha256,
        method_registry_sha256=method_registry_sha256(registry),
        training_scope_id=training_scope_id(fold),
        seed=seed,
        training_manifest_sha256=manifest_hash(training_manifest),
        scientific_parameters_sha256=parameters_sha256,
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        training_pairs=int(len(training_manifest)),
    )


def checkpoint_dir_for(
    output_root: Path,
    contract: TrainingContract,
) -> Path:
    return (
        output_root
        / "checkpoints"
        / contract.method_id
        / contract.training_scope_id
        / contract.scientific_parameters_sha256[:16]
        / contract.contract_sha256[:16]
    )


def threshold_path_for(
    output_root: Path,
    method_id: str,
    fold: TaxonomyFold,
    parameters_sha256: str,
    seed: int,
) -> Path:
    return (
        output_root
        / "selection"
        / method_id
        / fold.fold_id
        / f"seed-{seed:04d}"
        / parameters_sha256[:16]
        / "threshold_transfer.json"
    )


def summary_path_for(
    output_root: Path,
    purpose: str,
    stage: str,
    method_id: str,
    fold: TaxonomyFold,
    parameters_sha256: str,
    seed: int,
) -> Path:
    if purpose not in {"tuning", "final"}:
        raise ValueError("Summary purpose must be tuning or final.")
    return (
        output_root
        / "summaries"
        / purpose
        / stage
        / method_id
        / fold.fold_id
        / f"seed-{seed:04d}"
        / parameters_sha256[:16]
        / "summary.json"
    )
