"""Composable fold preparation and sharded scoring for the taxonomy suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from msc_project.experiments.taxonomy_execution import (
    RunContract,
    build_score_artifact,
    canonical_sha256,
    dataframe_sha256,
    score_shard_resume_state,
    select_pair_shard,
    write_score_shard,
)
from msc_project.experiments.taxonomy_methods import (
    PairProbabilityRuntime,
    load_method_registry,
    method_registry_sha256,
    resolve_method_spec,
)
from msc_project.experiments.taxonomy_protocol import (
    TaxonomyFold,
    build_budgeted_training_manifest,
    build_candidate_table,
    build_taxonomy_eval_grid,
    build_taxonomy_fold_splits,
    pair_identity_hash,
    registered_folds,
    scientific_protocol_sha256,
)
from msc_project.experiments.taxonomy_resources import (
    mixed_representation_sha256,
)
from msc_project.experiments.unified_candidate_pairs import (
    CANDIDATE_SENTIMENTS,
    manifest_hash,
)


@dataclass(frozen=True)
class PreparedFoldEvaluation:
    fold: TaxonomyFold
    condition: str
    split_name: str
    training_split: pd.DataFrame
    evaluation_split: pd.DataFrame
    training_manifest: pd.DataFrame
    candidates: pd.DataFrame
    evaluation_grid: pd.DataFrame
    description_resource: Mapping[str, object]


def prepare_fold_training(
    source_frame: pd.DataFrame,
    fold: TaxonomyFold,
    description_resource: Mapping[str, object],
    *,
    total_budget: int = 4096,
    positive_budget: int = 2048,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build training evidence without requiring any evaluation split to load."""

    splits = build_taxonomy_fold_splits(
        source_frame,
        fold,
        evaluation_splits=(),
    )
    manifest = build_budgeted_training_manifest(
        splits["train"],
        fold,
        description_resource,
        total_budget=total_budget,
        positive_budget=positive_budget,
        seed=seed,
    )
    return splits["train"], manifest


def prepare_fold_evaluation(
    source_frame: pd.DataFrame,
    fold: TaxonomyFold,
    condition: str,
    split_name: str,
    description_resource: Mapping[str, object],
    *,
    total_budget: int = 4096,
    positive_budget: int = 2048,
    seed: int = 13,
) -> PreparedFoldEvaluation:
    if split_name not in {"validation", "test"}:
        raise ValueError("Prepared evaluation split must be validation or test.")
    splits = build_taxonomy_fold_splits(
        source_frame,
        fold,
        evaluation_splits=(split_name,),
    )
    training_manifest = build_budgeted_training_manifest(
        splits["train"],
        fold,
        description_resource,
        total_budget=total_budget,
        positive_budget=positive_budget,
        seed=seed,
    )
    candidates = build_candidate_table(
        fold,
        condition,
        description_resource,
    )
    grid = build_taxonomy_eval_grid(
        splits[split_name],
        candidates,
        fold_id=fold.fold_id,
        condition=condition,
    )
    return PreparedFoldEvaluation(
        fold=fold,
        condition=condition,
        split_name=split_name,
        training_split=splits["train"],
        evaluation_split=splits[split_name],
        training_manifest=training_manifest,
        candidates=candidates,
        evaluation_grid=grid,
        description_resource=description_resource,
    )


def prepare_strict_seen_calibration(
    source_frame: pd.DataFrame,
    fold: TaxonomyFold,
    description_resource: Mapping[str, object],
    *,
    total_budget: int = 4096,
    positive_budget: int = 2048,
    seed: int = 13,
) -> PreparedFoldEvaluation:
    """Build a validation grid containing seen candidates and labels only."""

    calibration_fold = fold
    if fold.level == "L1":
        index = tuple(registered_folds("L1")).index(fold)
        calibration_fold = registered_folds("L2")[index]
        if (
            calibration_fold.heldout_aspects != fold.heldout_aspects
            or calibration_fold.seen_aspects != fold.seen_aspects
        ):
            raise AssertionError("Matched Level 1/2 folds do not share training scope.")
    splits = build_taxonomy_fold_splits(
        source_frame,
        calibration_fold,
        evaluation_splits=("validation",),
    )
    training_manifest = build_budgeted_training_manifest(
        splits["train"],
        calibration_fold,
        description_resource,
        total_budget=total_budget,
        positive_budget=positive_budget,
        seed=seed,
    )
    description_condition = (
        "D" if "D" in calibration_fold.conditions else "DD"
    )
    candidates = build_candidate_table(
        calibration_fold,
        description_condition,
        description_resource,
    )
    candidates = candidates[
        candidates["candidate_aspect"].astype(str).isin(
            calibration_fold.seen_aspects
        )
    ].reset_index(drop=True)
    grid = build_taxonomy_eval_grid(
        splits["validation"],
        candidates,
        fold_id=calibration_fold.fold_id,
        condition="seen-calibration",
    )
    return PreparedFoldEvaluation(
        fold=calibration_fold,
        condition="seen-calibration",
        split_name="validation",
        training_split=splits["train"],
        evaluation_split=splits["validation"],
        training_manifest=training_manifest,
        candidates=candidates,
        evaluation_grid=grid,
        description_resource=description_resource,
    )


def prepare_l1_strict_calibration(
    source_frame: pd.DataFrame,
    fold: TaxonomyFold,
    description_resource: Mapping[str, object],
    *,
    total_budget: int = 4096,
    positive_budget: int = 2048,
    seed: int = 13,
) -> PreparedFoldEvaluation:
    """Backward-compatible wrapper for the general seen-only calibration."""

    if fold.level != "L1":
        raise ValueError("This wrapper is only defined for Level 1.")
    return prepare_strict_seen_calibration(
        source_frame,
        fold,
        description_resource,
        total_budget=total_budget,
        positive_budget=positive_budget,
        seed=seed,
    )


def build_run_contract(
    prepared: PreparedFoldEvaluation,
    method_id: str,
    scientific_parameters: Mapping[str, object],
    *,
    shard_count: int,
    formal: bool,
    protocol_id: str = "taxonomy_generalisation_precloud_v1",
    registry: Mapping[str, object] | None = None,
    seed: int = 13,
) -> RunContract:
    method_registry = dict(registry or load_method_registry())
    spec = resolve_method_spec(method_id, method_registry)
    variants = {
        str(row.candidate_aspect): str(row.representation_variant)
        for row in prepared.candidates.drop_duplicates("candidate_aspect").itertuples(
            index=False
        )
    }
    evaluation_data_hash = dataframe_sha256(
        prepared.evaluation_split,
        ["row_uid", "text", "supervision_pair_labels"],
        sort_columns=("row_uid",),
    )
    return RunContract(
        protocol_id=protocol_id,
        scientific_protocol_sha256=scientific_protocol_sha256(),
        method_id=method_id,
        method_spec_sha256=spec.spec_sha256,
        method_registry_sha256=method_registry_sha256(method_registry),
        description_resource_sha256=str(
            prepared.description_resource["content_sha256"]
        ),
        candidate_representation_sha256=mixed_representation_sha256(
            tuple(
                prepared.candidates["candidate_aspect"]
                .astype(str)
                .drop_duplicates()
            ),
            CANDIDATE_SENTIMENTS,
            variants,
            prepared.description_resource,
        ),
        level=prepared.fold.level,
        fold_id=prepared.fold.fold_id,
        condition=prepared.condition,
        split=prepared.split_name,
        seed=seed,
        training_manifest_sha256=manifest_hash(prepared.training_manifest),
        evaluation_data_sha256=evaluation_data_hash,
        evaluation_pair_identity_sha256=pair_identity_hash(
            prepared.evaluation_grid
        ),
        scientific_parameters_sha256=canonical_sha256(
            dict(scientific_parameters)
        ),
        shard_count=shard_count,
        formal=formal,
    )


def score_prepared_shard(
    prepared: PreparedFoldEvaluation,
    runtime: PairProbabilityRuntime,
    contract: RunContract,
    output_root: Path,
    *,
    shard_index: int,
    resume: bool,
) -> tuple[pd.DataFrame, str]:
    if runtime.method_id != contract.method_id:
        raise ValueError("Runtime method_id differs from the run contract.")
    if (
        contract.evaluation_pair_identity_sha256
        != pair_identity_hash(prepared.evaluation_grid)
    ):
        raise ValueError("Prepared grid differs from the run contract.")
    if resume:
        state, artifact = score_shard_resume_state(
            output_root,
            contract,
            prepared.evaluation_grid,
            shard_index=shard_index,
        )
        if state == "complete":
            assert artifact is not None
            return artifact, "resumed"
    shard = select_pair_shard(
        prepared.evaluation_grid,
        shard_index,
        contract.shard_count,
    )
    if shard.empty:
        raise ValueError("Requested score shard contains no review clusters.")
    scores = runtime.score(shard)
    artifact = build_score_artifact(
        shard,
        scores,
        contract,
        shard_index=shard_index,
    )
    write_score_shard(
        artifact,
        output_root,
        contract,
        shard_index=shard_index,
    )
    return artifact, "scored"


def score_prepared_conditions_shard(
    prepared_conditions: Sequence[PreparedFoldEvaluation],
    runtime: PairProbabilityRuntime,
    contracts: Mapping[str, RunContract],
    output_root: Path,
    *,
    shard_index: int,
    resume: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], int]:
    """Score each unique rendered claim once, then project to matched conditions."""

    assert_condition_pair_identity(prepared_conditions)
    by_condition = {value.condition: value for value in prepared_conditions}
    if set(by_condition) != set(contracts):
        raise ValueError("Prepared conditions and score contracts differ.")
    contract_values = list(contracts.values())
    if runtime.method_id != contract_values[0].method_id:
        raise ValueError("Runtime method differs from the condition contracts.")
    shared_fields = (
        "method_id",
        "level",
        "fold_id",
        "split",
        "seed",
        "training_manifest_sha256",
        "scientific_parameters_sha256",
        "shard_count",
        "formal",
    )
    for field in shared_fields:
        if len({getattr(contract, field) for contract in contract_values}) != 1:
            raise ValueError(f"Condition score contracts differ on {field}.")

    artifacts: dict[str, pd.DataFrame] = {}
    states: dict[str, str] = {}
    missing: list[str] = []
    for condition, prepared in by_condition.items():
        contract = contracts[condition]
        if (
            contract.evaluation_pair_identity_sha256
            != pair_identity_hash(prepared.evaluation_grid)
        ):
            raise ValueError(f"Prepared grid differs from {condition} contract.")
        if resume:
            state, artifact = score_shard_resume_state(
                output_root,
                contract,
                prepared.evaluation_grid,
                shard_index=shard_index,
            )
            if state == "complete":
                assert artifact is not None
                artifacts[condition] = artifact
                states[condition] = "resumed"
                continue
        missing.append(condition)
    if not missing:
        return artifacts, states, 0

    score_key = [
        "row_uid",
        "candidate_aspect",
        "candidate_sentiment",
        "candidate_text",
        "representation_variant",
    ]
    shards = {
        condition: select_pair_shard(
            by_condition[condition].evaluation_grid,
            shard_index,
            contracts[condition].shard_count,
        )
        for condition in missing
    }
    if any(frame.empty for frame in shards.values()):
        raise ValueError("Requested condition shard contains no review clusters.")
    union = pd.concat(
        [
            frame.assign(_source_condition=condition)
            for condition, frame in shards.items()
        ],
        ignore_index=True,
    )
    text_counts = union.groupby("row_uid", sort=False)["text"].nunique()
    if (text_counts > 1).any():
        raise ValueError("A row_uid maps to different review text across conditions.")
    union = union.drop_duplicates(score_key, keep="first").reset_index(drop=True)
    scores = runtime.score(union)
    score_lookup = {
        tuple(row[column] for column in score_key): float(score)
        for row, score in zip(union.to_dict(orient="records"), scores)
    }
    for condition, shard in shards.items():
        projected = [
            score_lookup[tuple(row[column] for column in score_key)]
            for row in shard.to_dict(orient="records")
        ]
        artifact = build_score_artifact(
            shard,
            projected,
            contracts[condition],
            shard_index=shard_index,
        )
        write_score_shard(
            artifact,
            output_root,
            contracts[condition],
            shard_index=shard_index,
        )
        artifacts[condition] = artifact
        states[condition] = "scored"
    return artifacts, states, int(len(union))


def unique_rendered_claim_count(
    prepared_conditions: Sequence[PreparedFoldEvaluation],
) -> int:
    """Count distinct model inputs across matched experimental conditions."""

    assert_condition_pair_identity(prepared_conditions)
    score_key = [
        "row_uid",
        "candidate_aspect",
        "candidate_sentiment",
        "candidate_text",
        "representation_variant",
    ]
    union = pd.concat(
        [value.evaluation_grid for value in prepared_conditions],
        ignore_index=True,
    )
    text_counts = union.groupby("row_uid", sort=False)["text"].nunique()
    if (text_counts > 1).any():
        raise ValueError("A row_uid maps to different review text across conditions.")
    return int(len(union.drop_duplicates(score_key, keep="first")))


def assert_condition_pair_identity(
    prepared_conditions: Sequence[PreparedFoldEvaluation],
) -> str:
    if not prepared_conditions:
        raise ValueError("At least one prepared condition is required.")
    fold_ids = {value.fold.fold_id for value in prepared_conditions}
    split_names = {value.split_name for value in prepared_conditions}
    hashes = {
        pair_identity_hash(value.evaluation_grid)
        for value in prepared_conditions
    }
    training_hashes = {
        manifest_hash(value.training_manifest)
        for value in prepared_conditions
    }
    if len(fold_ids) != 1 or len(split_names) != 1:
        raise ValueError("Condition comparison mixes folds or splits.")
    if len(hashes) != 1:
        raise ValueError("Condition comparison changes evaluation pair identities.")
    if len(training_hashes) != 1:
        raise ValueError("Condition comparison changes training pair identities.")
    return next(iter(hashes))
