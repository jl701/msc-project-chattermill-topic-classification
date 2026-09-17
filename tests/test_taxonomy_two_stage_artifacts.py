from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_execution import (
    RunContract,
    canonical_sha256,
    select_pair_shard,
)
from msc_project.experiments.taxonomy_protocol import pair_identity_hash
from msc_project.experiments.taxonomy_two_stage_artifacts import (
    build_two_stage_score_artifact,
    merge_two_stage_score_shards,
    two_stage_shard_paths,
    two_stage_shard_resume_state,
    validate_two_stage_score_shard,
    write_two_stage_score_shard,
)


def _grid() -> pd.DataFrame:
    records = []
    for row_index, uid in enumerate(("validation:1", "validation:2", "validation:3")):
        for sentiment_index, sentiment in enumerate(("negative", "neutral", "positive")):
            records.append(
                {
                    "fold_id": "l2-a01",
                    "condition": "D",
                    "row_index": row_index,
                    "row_uid": uid,
                    "candidate_aspect": "Quality",
                    "candidate_sentiment": sentiment,
                    "pair_label": f"Quality | {sentiment}",
                    "representation_variant": "name_and_description",
                    "is_seen": False,
                    "is_heldout": True,
                    "target": int(row_index == 0 and sentiment == "positive"),
                    "aspect_score": 0.8 - row_index * 0.1,
                    "sentiment_score": 0.1 + sentiment_index * 0.2,
                }
            )
    return pd.DataFrame.from_records(records)


def _contract(grid: pd.DataFrame, *, shard_count: int = 2) -> RunContract:
    return RunContract(
        protocol_id="taxonomy_two_stage_formal_v1",
        scientific_protocol_sha256="0" * 64,
        method_id="qwen_candidate_pair_qlora",
        method_spec_sha256="1" * 64,
        method_registry_sha256="2" * 64,
        description_resource_sha256="3" * 64,
        candidate_representation_sha256="4" * 64,
        level="L2",
        fold_id="l2-a01",
        condition="D",
        split="validation",
        seed=13,
        training_manifest_sha256="5" * 64,
        evaluation_data_sha256="6" * 64,
        evaluation_pair_identity_sha256=pair_identity_hash(grid),
        scientific_parameters_sha256=canonical_sha256({"lr": 5e-6}),
        shard_count=shard_count,
        formal=True,
    )


def test_two_stage_shards_round_trip_merge_and_resume(tmp_path: Path) -> None:
    grid = _grid()
    contract = _contract(grid, shard_count=2)
    artifacts = {}
    for index in range(contract.shard_count):
        shard = select_pair_shard(grid, index, contract.shard_count)
        if shard.empty:
            continue
        artifact = build_two_stage_score_artifact(shard, contract, shard_index=index)
        csv_path, manifest_path = write_two_stage_score_shard(
            artifact, tmp_path, contract, shard_index=index
        )
        artifacts[index] = validate_two_stage_score_shard(
            csv_path,
            manifest_path,
            contract,
            grid,
            shard_index=index,
        )
        state, resumed = two_stage_shard_resume_state(
            tmp_path, contract, grid, shard_index=index
        )
        assert state == "complete"
        assert resumed is not None
    merged = merge_two_stage_score_shards(artifacts, contract, grid)
    assert pair_identity_hash(merged) == pair_identity_hash(grid)


def test_two_stage_resume_rejects_partial_corrupt_and_changed_contract(tmp_path: Path) -> None:
    grid = _grid()
    contract = _contract(grid, shard_count=1)
    artifact = build_two_stage_score_artifact(grid, contract, shard_index=0)
    csv_path, _ = write_two_stage_score_shard(
        artifact, tmp_path, contract, shard_index=0
    )
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "corrupt", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt"):
        two_stage_shard_resume_state(tmp_path, contract, grid, shard_index=0)

    clean_root = tmp_path / "changed"
    artifact = build_two_stage_score_artifact(grid, contract, shard_index=0)
    write_two_stage_score_shard(artifact, clean_root, contract, shard_index=0)
    changed = replace(
        contract,
        scientific_parameters_sha256=canonical_sha256({"lr": 1e-5}),
    )
    old_csv, old_manifest = two_stage_shard_paths(clean_root, contract, 0)
    with pytest.raises(ValueError, match="different contract"):
        validate_two_stage_score_shard(
            old_csv, old_manifest, changed, grid, shard_index=0
        )


def test_two_stage_artifact_rejects_non_finite_score() -> None:
    grid = _grid()
    contract = _contract(grid, shard_count=1)
    grid.loc[0, "sentiment_score"] = float("nan")
    with pytest.raises(ValueError):
        build_two_stage_score_artifact(grid, contract, shard_index=0)
