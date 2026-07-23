from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_execution import (
    RunContract,
    build_score_artifact,
    canonical_sha256,
    merge_score_shards,
    row_shard_index,
    score_shard_paths,
    score_shard_resume_state,
    select_pair_shard,
    validate_score_shard,
    write_score_shard,
)
from msc_project.experiments.taxonomy_protocol import pair_identity_hash


def pair_grid() -> pd.DataFrame:
    rows = []
    for row_index, row_uid in enumerate(("validation:1", "validation:2", "validation:3")):
        for sentiment in ("negative", "neutral", "positive"):
            rows.append(
                {
                    "fold_id": "l2-a01",
                    "condition": "D",
                    "row_index": row_index,
                    "row_uid": row_uid,
                    "text": f"private review {row_uid}",
                    "candidate_aspect": "Aspect A",
                    "candidate_sentiment": sentiment,
                    "candidate_text": f"candidate {sentiment}",
                    "pair_label": f"Aspect A | {sentiment}",
                    "representation_variant": "minimal",
                    "is_seen": False,
                    "is_heldout": True,
                    "target": int(row_index == 0 and sentiment == "positive"),
                }
            )
    return pd.DataFrame(rows)


def contract(grid: pd.DataFrame, *, shard_count: int = 2) -> RunContract:
    digest = "a" * 64
    return RunContract(
        protocol_id="taxonomy_generalisation_precloud_v1",
        method_id="strict_train_only_tfidf",
        method_spec_sha256="b" * 64,
        method_registry_sha256="c" * 64,
        description_resource_sha256="d" * 64,
        candidate_representation_sha256="e" * 64,
        level="L2",
        fold_id="l2-a01",
        condition="D",
        split="validation",
        seed=13,
        training_manifest_sha256="f" * 64,
        evaluation_data_sha256=digest,
        evaluation_pair_identity_sha256=pair_identity_hash(grid),
        scientific_parameters_sha256=canonical_sha256({"c": 1.0}),
        shard_count=shard_count,
        formal=False,
    )


def test_sharding_is_deterministic_and_keeps_review_clusters_whole() -> None:
    grid = pair_grid()
    first = [
        select_pair_shard(grid, index, 3)
        for index in range(3)
    ]
    second = [
        select_pair_shard(grid.sample(frac=1, random_state=3), index, 3)
        for index in range(3)
    ]
    for row_uid in grid["row_uid"].unique():
        expected = row_shard_index(row_uid, 3)
        assert set(first[expected]["row_uid"]).issuperset({row_uid})
        assert len(first[expected][first[expected]["row_uid"] == row_uid]) == 3
    assert {
        row_uid
        for shard in first
        for row_uid in shard["row_uid"]
    } == set(grid["row_uid"])
    assert [
        set(frame["row_uid"]) for frame in first
    ] == [
        set(frame["row_uid"]) for frame in second
    ]


def test_run_contract_hash_changes_with_scientific_parameters() -> None:
    grid = pair_grid()
    first = contract(grid)
    second = replace(first, scientific_parameters_sha256=canonical_sha256({"c": 3.0}))
    assert first.contract_sha256 != second.contract_sha256


def test_score_artifact_excludes_review_and_candidate_text() -> None:
    grid = pair_grid()
    run = contract(grid, shard_count=1)
    artifact = build_score_artifact(
        grid,
        np.linspace(0.1, 0.9, len(grid)),
        run,
        shard_index=0,
    )
    assert "text" not in artifact
    assert "candidate_text" not in artifact
    assert set(artifact["contract_sha256"]) == {run.contract_sha256}


def test_write_validate_and_resume_round_trip(tmp_path: Path) -> None:
    grid = pair_grid()
    run = contract(grid, shard_count=1)
    artifact = build_score_artifact(
        grid,
        np.linspace(0.1, 0.9, len(grid)),
        run,
        shard_index=0,
    )
    csv_path, manifest_path = write_score_shard(
        artifact,
        tmp_path,
        run,
        shard_index=0,
    )
    observed = validate_score_shard(
        csv_path,
        manifest_path,
        run,
        grid,
        shard_index=0,
    )
    state, resumed = score_shard_resume_state(
        tmp_path,
        run,
        grid,
        shard_index=0,
    )
    assert state == "complete"
    assert resumed is not None
    pd.testing.assert_frame_equal(observed, resumed)
    with pytest.raises(FileExistsError):
        write_score_shard(artifact, tmp_path, run, shard_index=0)


def test_resume_fails_closed_on_partial_or_corrupt_state(tmp_path: Path) -> None:
    grid = pair_grid()
    run = contract(grid, shard_count=1)
    csv_path, _ = score_shard_paths(tmp_path, run, 0)
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Partial"):
        score_shard_resume_state(tmp_path, run, grid, shard_index=0)


def test_merge_requires_exact_nonempty_shard_set_and_pair_coverage() -> None:
    grid = pair_grid()
    run = contract(grid, shard_count=2)
    artifacts = {}
    for index in range(run.shard_count):
        shard = select_pair_shard(grid, index, run.shard_count)
        if shard.empty:
            continue
        artifacts[index] = build_score_artifact(
            shard,
            np.full(len(shard), 0.5),
            run,
            shard_index=index,
        )
    merged = merge_score_shards(artifacts, run, grid)
    assert pair_identity_hash(merged) == pair_identity_hash(grid)
    if len(artifacts) > 1:
        with pytest.raises(ValueError, match="incomplete"):
            merge_score_shards(
                {next(iter(artifacts)): next(iter(artifacts.values()))},
                run,
                grid,
            )
    corrupted = {key: value.copy() for key, value in artifacts.items()}
    first_key = next(iter(corrupted))
    corrupted[first_key].loc[corrupted[first_key].index[0], "target"] ^= 1
    with pytest.raises(ValueError, match="expected grid"):
        merge_score_shards(corrupted, run, grid)
