"""Content-addressed two-stage score shards with exact resume validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_execution import (
    RunContract,
    dataframe_sha256,
    select_pair_shard,
)
from msc_project.experiments.taxonomy_methods import validate_probability_scores
from msc_project.experiments.taxonomy_protocol import PAIR_KEY, pair_identity_hash
from msc_project.experiments.verified_artifact_sync import file_sha256


TWO_STAGE_SCORE_COLUMNS = (
    "fold_id",
    "condition",
    "row_index",
    "row_uid",
    "candidate_aspect",
    "candidate_sentiment",
    "pair_label",
    "representation_variant",
    "is_seen",
    "is_heldout",
    "target",
    "aspect_score",
    "sentiment_score",
    "method_id",
    "split",
    "contract_sha256",
    "shard_index",
    "shard_count",
)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale score-shard temporary file: {temporary}")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def two_stage_shard_paths(
    output_root: Path,
    contract: RunContract,
    shard_index: int,
) -> tuple[Path, Path]:
    contract.validate()
    if contract.split != "validation":
        raise ValueError("Formal two-stage artifacts are validation-only.")
    if not 0 <= shard_index < contract.shard_count:
        raise ValueError("shard_index is outside the run contract.")
    directory = (
        output_root
        / "scores"
        / contract.method_id
        / contract.level
        / contract.fold_id
        / contract.condition
        / contract.contract_sha256[:16]
    )
    stem = f"shard-{shard_index:05d}-of-{contract.shard_count:05d}"
    return directory / f"{stem}.csv", directory / f"{stem}.manifest.json"


def build_two_stage_score_artifact(
    scored_grid: pd.DataFrame,
    contract: RunContract,
    *,
    shard_index: int,
) -> pd.DataFrame:
    contract.validate()
    if contract.split != "validation":
        raise ValueError("Two-stage formal scoring may use validation only.")
    required = set(TWO_STAGE_SCORE_COLUMNS[:13])
    missing = sorted(required - set(scored_grid.columns))
    if missing or scored_grid.empty:
        raise ValueError(f"Two-stage scored grid is invalid; missing={missing}.")
    if scored_grid.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Two-stage scored grid has duplicate pair identities.")
    values = scored_grid[list(TWO_STAGE_SCORE_COLUMNS[:13])].copy()
    validate_probability_scores(values["aspect_score"], expected_rows=len(values))
    validate_probability_scores(values["sentiment_score"], expected_rows=len(values))
    expected = select_pair_shard(values, shard_index, contract.shard_count)
    expected_keys = set(map(tuple, expected[list(PAIR_KEY)].astype(str).to_numpy()))
    observed_keys = set(map(tuple, values[list(PAIR_KEY)].astype(str).to_numpy()))
    if expected_keys != observed_keys:
        raise ValueError("Two-stage scored rows do not exactly match the requested shard.")
    values["method_id"] = contract.method_id
    values["split"] = contract.split
    values["contract_sha256"] = contract.contract_sha256
    values["shard_index"] = int(shard_index)
    values["shard_count"] = int(contract.shard_count)
    return values[list(TWO_STAGE_SCORE_COLUMNS)].reset_index(drop=True)


def _manifest(
    artifact: pd.DataFrame,
    contract: RunContract,
    *,
    shard_index: int,
    csv_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "taxonomy_two_stage_score_shard_v1",
        "contract": contract.to_dict(),
        "shard_index": int(shard_index),
        "shard_count": int(contract.shard_count),
        "rows": int(len(artifact)),
        "review_clusters": int(artifact["row_uid"].astype(str).nunique()),
        "pair_identity_sha256": pair_identity_hash(artifact),
        "score_sha256": dataframe_sha256(
            artifact,
            [*PAIR_KEY, "aspect_score", "sentiment_score"],
            sort_columns=PAIR_KEY,
        ),
        "csv_sha256": csv_sha256,
        "finite_score_count": int(
            np.isfinite(
                artifact[["aspect_score", "sentiment_score"]].to_numpy(dtype=float)
            ).sum()
        ),
        "test_contract_count": 0,
    }


def write_two_stage_score_shard(
    artifact: pd.DataFrame,
    output_root: Path,
    contract: RunContract,
    *,
    shard_index: int,
) -> tuple[Path, Path]:
    csv_path, manifest_path = two_stage_shard_paths(output_root, contract, shard_index)
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite score-shard state: {csv_path.parent}")
    _atomic_write_text(csv_path, artifact.to_csv(index=False, lineterminator="\n"))
    manifest = _manifest(
        artifact,
        contract,
        shard_index=shard_index,
        csv_sha256=file_sha256(csv_path),
    )
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return csv_path, manifest_path


def validate_two_stage_score_shard(
    csv_path: Path,
    manifest_path: Path,
    contract: RunContract,
    expected_grid: pd.DataFrame,
    *,
    shard_index: int,
) -> pd.DataFrame:
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Both two-stage shard files are required.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "taxonomy_two_stage_score_shard_v1":
        raise ValueError("Unexpected two-stage score-shard schema.")
    if manifest.get("contract") != contract.to_dict():
        raise ValueError("Two-stage score shard belongs to a different contract.")
    if manifest.get("shard_index") != shard_index:
        raise ValueError("Two-stage score shard has the wrong index.")
    if manifest.get("test_contract_count") != 0:
        raise ValueError("A test contract entered a validation score shard.")
    if manifest.get("csv_sha256") != file_sha256(csv_path):
        raise ValueError("Two-stage score CSV hash mismatch.")
    artifact = pd.read_csv(csv_path, float_precision="round_trip")
    if tuple(artifact.columns) != TWO_STAGE_SCORE_COLUMNS:
        raise ValueError("Two-stage score columns differ from the frozen schema.")
    if int(manifest.get("rows", -1)) != len(artifact):
        raise ValueError("Two-stage score row count differs from its manifest.")
    expected = select_pair_shard(expected_grid, shard_index, contract.shard_count)
    expected_keys = expected[[*PAIR_KEY, "target"]].copy()
    observed_keys = artifact[[*PAIR_KEY, "target"]].copy()
    for frame in (expected_keys, observed_keys):
        frame["row_uid"] = frame["row_uid"].astype(str)
        frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
        frame.sort_values(list(PAIR_KEY), inplace=True, kind="stable")
        frame.reset_index(drop=True, inplace=True)
    if not expected_keys.equals(observed_keys):
        raise ValueError("Two-stage shard identities or targets differ from the expected grid.")
    for column in ("aspect_score", "sentiment_score"):
        validate_probability_scores(artifact[column], expected_rows=len(artifact))
    if set(artifact["contract_sha256"].astype(str)) != {contract.contract_sha256}:
        raise ValueError("Two-stage shard row contract hash mismatch.")
    if manifest.get("pair_identity_sha256") != pair_identity_hash(artifact):
        raise ValueError("Two-stage shard pair identity hash mismatch.")
    score_hash = dataframe_sha256(
        artifact,
        [*PAIR_KEY, "aspect_score", "sentiment_score"],
        sort_columns=PAIR_KEY,
    )
    if manifest.get("score_sha256") != score_hash:
        raise ValueError("Two-stage shard score hash mismatch.")
    return artifact


def two_stage_shard_resume_state(
    output_root: Path,
    contract: RunContract,
    expected_grid: pd.DataFrame,
    *,
    shard_index: int,
) -> tuple[str, pd.DataFrame | None]:
    csv_path, manifest_path = two_stage_shard_paths(output_root, contract, shard_index)
    if not csv_path.exists() and not manifest_path.exists():
        return "missing", None
    try:
        artifact = validate_two_stage_score_shard(
            csv_path,
            manifest_path,
            contract,
            expected_grid,
            shard_index=shard_index,
        )
    except Exception as error:
        raise RuntimeError(
            f"Partial, corrupt, or incompatible two-stage shard {shard_index}."
        ) from error
    return "complete", artifact


def merge_two_stage_score_shards(
    artifacts: Mapping[int, pd.DataFrame],
    contract: RunContract,
    expected_grid: pd.DataFrame,
) -> pd.DataFrame:
    expected_indices = {
        index
        for index in range(contract.shard_count)
        if not select_pair_shard(expected_grid, index, contract.shard_count).empty
    }
    if set(artifacts) != expected_indices:
        raise ValueError("Two-stage score-shard set is incomplete.")
    merged = pd.concat([artifacts[index] for index in sorted(artifacts)], ignore_index=True)
    if merged.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Merged two-stage score shards contain duplicate identities.")
    expected_keys = expected_grid[[*PAIR_KEY, "target"]].copy()
    observed_keys = merged[[*PAIR_KEY, "target"]].copy()
    for frame in (expected_keys, observed_keys):
        frame["row_uid"] = frame["row_uid"].astype(str)
        frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
        frame.sort_values(list(PAIR_KEY), inplace=True, kind="stable")
        frame.reset_index(drop=True, inplace=True)
    if not expected_keys.equals(observed_keys):
        raise ValueError("Merged two-stage shards do not cover the expected grid exactly.")
    return merged.sort_values(list(PAIR_KEY), kind="stable").reset_index(drop=True)
