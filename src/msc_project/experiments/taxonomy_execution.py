"""Fail-closed run contracts, sharding, score artifacts, and resumability."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_methods import validate_probability_scores
from msc_project.experiments.taxonomy_protocol import PAIR_KEY, pair_identity_hash


SCORE_ARTIFACT_COLUMNS = (
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
    "score",
    "method_id",
    "split",
    "contract_sha256",
    "shard_index",
    "shard_count",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataframe_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sort_columns: Sequence[str],
) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Hash frame is missing columns: {missing}")
    records = (
        frame[list(columns)]
        .sort_values(list(sort_columns), kind="stable")
        .to_dict(orient="records")
    )
    return canonical_sha256({"columns": list(columns), "records": records})


@dataclass(frozen=True)
class RunContract:
    protocol_id: str
    method_id: str
    method_spec_sha256: str
    method_registry_sha256: str
    description_resource_sha256: str
    candidate_representation_sha256: str
    level: str
    fold_id: str
    condition: str
    split: str
    seed: int
    training_manifest_sha256: str
    evaluation_data_sha256: str
    evaluation_pair_identity_sha256: str
    scientific_parameters_sha256: str
    shard_count: int
    formal: bool

    def validate(self) -> None:
        text_fields = (
            "protocol_id",
            "method_id",
            "method_spec_sha256",
            "method_registry_sha256",
            "description_resource_sha256",
            "candidate_representation_sha256",
            "level",
            "fold_id",
            "condition",
            "split",
            "training_manifest_sha256",
            "evaluation_data_sha256",
            "evaluation_pair_identity_sha256",
            "scientific_parameters_sha256",
        )
        for field in text_fields:
            if not str(getattr(self, field)).strip():
                raise ValueError(f"Run contract {field} must be non-empty.")
        for field in (
            "method_spec_sha256",
            "method_registry_sha256",
            "description_resource_sha256",
            "candidate_representation_sha256",
            "training_manifest_sha256",
            "evaluation_data_sha256",
            "evaluation_pair_identity_sha256",
            "scientific_parameters_sha256",
        ):
            value = str(getattr(self, field))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"Run contract {field} must be a lowercase SHA-256.")
        if self.split not in {"validation", "test"}:
            raise ValueError("Run contract split must be validation or test.")
        if self.seed < 0:
            raise ValueError("Run contract seed must be non-negative.")
        if self.shard_count < 1:
            raise ValueError("Run contract shard_count must be positive.")

    @property
    def contract_sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "contract_sha256": self.contract_sha256}


def row_shard_index(row_uid: str, shard_count: int) -> int:
    if not str(row_uid):
        raise ValueError("row_uid must be non-empty.")
    if shard_count < 1:
        raise ValueError("shard_count must be positive.")
    digest = hashlib.sha256(str(row_uid).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shard_count


def select_pair_shard(
    frame: pd.DataFrame,
    shard_index: int,
    shard_count: int,
) -> pd.DataFrame:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count).")
    if "row_uid" not in frame:
        raise ValueError("Pair frame is missing row_uid.")
    selected = frame[
        frame["row_uid"].astype(str).map(
            lambda value: row_shard_index(value, shard_count) == shard_index
        )
    ].copy()
    for row_uid, group in frame.groupby(frame["row_uid"].astype(str), sort=False):
        observed = set(
            selected.loc[selected["row_uid"].astype(str) == row_uid].index
        )
        if observed and len(observed) != len(group):
            raise AssertionError("A review cluster was split across score shards.")
    return selected.reset_index(drop=True)


def build_score_artifact(
    pair_grid: pd.DataFrame,
    scores: Sequence[float] | np.ndarray,
    contract: RunContract,
    *,
    shard_index: int,
) -> pd.DataFrame:
    contract.validate()
    if not 0 <= shard_index < contract.shard_count:
        raise ValueError("shard_index is outside the run contract.")
    required = set(SCORE_ARTIFACT_COLUMNS[:11])
    missing = sorted(required - set(pair_grid.columns))
    if missing:
        raise ValueError(f"Pair grid is missing score-artifact columns: {missing}")
    if pair_grid.empty:
        raise ValueError("Cannot build an empty score artifact.")
    if pair_grid.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Pair grid contains duplicate pair identities.")
    expected_shard = {
        row_shard_index(value, contract.shard_count)
        for value in pair_grid["row_uid"].astype(str)
    }
    if expected_shard != {shard_index}:
        raise ValueError("Pair grid rows do not all belong to the requested shard.")
    values = validate_probability_scores(scores, expected_rows=len(pair_grid))
    artifact = pair_grid[list(SCORE_ARTIFACT_COLUMNS[:11])].copy()
    artifact["score"] = values
    artifact["method_id"] = contract.method_id
    artifact["split"] = contract.split
    artifact["contract_sha256"] = contract.contract_sha256
    artifact["shard_index"] = int(shard_index)
    artifact["shard_count"] = int(contract.shard_count)
    return artifact[list(SCORE_ARTIFACT_COLUMNS)].reset_index(drop=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite stale temporary file: {temporary}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def score_shard_paths(
    output_root: Path,
    contract: RunContract,
    shard_index: int,
) -> tuple[Path, Path]:
    contract.validate()
    if not 0 <= shard_index < contract.shard_count:
        raise ValueError("shard_index is outside the run contract.")
    directory = (
        output_root
        / contract.protocol_id
        / contract.method_id
        / contract.level
        / contract.fold_id
        / contract.condition
        / contract.split
        / contract.contract_sha256[:16]
    )
    stem = f"shard-{shard_index:05d}-of-{contract.shard_count:05d}"
    return directory / f"{stem}.csv", directory / f"{stem}.manifest.json"


def _artifact_manifest(
    artifact: pd.DataFrame,
    contract: RunContract,
    shard_index: int,
    csv_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "taxonomy_score_shard_v1",
        "contract": contract.to_dict(),
        "shard_index": int(shard_index),
        "shard_count": int(contract.shard_count),
        "rows": int(len(artifact)),
        "review_clusters": int(artifact["row_uid"].astype(str).nunique()),
        "pair_identity_sha256": pair_identity_hash(artifact),
        "score_sha256": dataframe_sha256(
            artifact,
            [*PAIR_KEY, "score"],
            sort_columns=PAIR_KEY,
        ),
        "csv_sha256": csv_sha256,
    }


def write_score_shard(
    artifact: pd.DataFrame,
    output_root: Path,
    contract: RunContract,
    *,
    shard_index: int,
) -> tuple[Path, Path]:
    csv_path, manifest_path = score_shard_paths(
        output_root, contract, shard_index
    )
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing score-shard state: {csv_path.parent}"
        )
    csv_text = artifact.to_csv(index=False, lineterminator="\n")
    _atomic_write_text(csv_path, csv_text)
    try:
        manifest = _artifact_manifest(
            artifact,
            contract,
            shard_index,
            _file_sha256(csv_path),
        )
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
    except Exception:
        # Leave the unmatched CSV in place so resume fails closed instead of
        # silently treating a partial write as missing work.
        raise
    return csv_path, manifest_path


def validate_score_shard(
    csv_path: Path,
    manifest_path: Path,
    contract: RunContract,
    expected_pair_grid: pd.DataFrame,
    *,
    shard_index: int,
) -> pd.DataFrame:
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Both score CSV and shard manifest are required.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Score-shard manifest must contain an object.")
    if manifest.get("schema_version") != "taxonomy_score_shard_v1":
        raise ValueError("Unexpected score-shard schema version.")
    if manifest.get("contract") != contract.to_dict():
        raise ValueError("Score shard belongs to a different run contract.")
    if manifest.get("shard_index") != shard_index:
        raise ValueError("Score-shard manifest has the wrong shard index.")
    if manifest.get("csv_sha256") != _file_sha256(csv_path):
        raise ValueError("Score-shard CSV hash mismatch.")

    artifact = pd.read_csv(csv_path, float_precision="round_trip")
    if tuple(artifact.columns) != SCORE_ARTIFACT_COLUMNS:
        raise ValueError("Score-shard columns do not match the frozen schema.")
    if int(manifest.get("rows", -1)) != len(artifact):
        raise ValueError("Score-shard row count differs from its manifest.")
    expected = select_pair_shard(
        expected_pair_grid,
        shard_index,
        contract.shard_count,
    )
    if expected.empty:
        raise ValueError("The expected shard contains no rows and should not exist.")
    expected_keys = expected[[*PAIR_KEY, "target"]].copy()
    observed_keys = artifact[[*PAIR_KEY, "target"]].copy()
    for frame in (expected_keys, observed_keys):
        frame["row_uid"] = frame["row_uid"].astype(str)
        frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
        frame.sort_values(list(PAIR_KEY), inplace=True, kind="stable")
        frame.reset_index(drop=True, inplace=True)
    if not expected_keys.equals(observed_keys):
        raise ValueError("Score-shard pair identities or gold targets do not match.")
    validate_probability_scores(artifact["score"], expected_rows=len(artifact))
    if set(artifact["contract_sha256"].astype(str)) != {contract.contract_sha256}:
        raise ValueError("Score-shard rows have the wrong contract hash.")
    observed_pair_hash = pair_identity_hash(artifact)
    if manifest.get("pair_identity_sha256") != observed_pair_hash:
        raise ValueError("Score-shard pair identity hash mismatch.")
    observed_score_hash = dataframe_sha256(
        artifact,
        [*PAIR_KEY, "score"],
        sort_columns=PAIR_KEY,
    )
    if manifest.get("score_sha256") != observed_score_hash:
        raise ValueError("Score-shard score hash mismatch.")
    return artifact


def score_shard_resume_state(
    output_root: Path,
    contract: RunContract,
    expected_pair_grid: pd.DataFrame,
    *,
    shard_index: int,
) -> tuple[str, pd.DataFrame | None]:
    csv_path, manifest_path = score_shard_paths(
        output_root, contract, shard_index
    )
    if not csv_path.exists() and not manifest_path.exists():
        return "missing", None
    try:
        artifact = validate_score_shard(
            csv_path,
            manifest_path,
            contract,
            expected_pair_grid,
            shard_index=shard_index,
        )
    except Exception as error:
        raise RuntimeError(
            f"Partial, corrupt, or incompatible resume state for shard {shard_index}."
        ) from error
    return "complete", artifact


def merge_score_shards(
    artifacts: Mapping[int, pd.DataFrame],
    contract: RunContract,
    expected_pair_grid: pd.DataFrame,
) -> pd.DataFrame:
    expected_indices = {
        index
        for index in range(contract.shard_count)
        if not select_pair_shard(expected_pair_grid, index, contract.shard_count).empty
    }
    if set(artifacts) != expected_indices:
        raise ValueError(
            f"Score-shard set is incomplete: expected={sorted(expected_indices)}, "
            f"observed={sorted(artifacts)}."
        )
    frames = []
    for shard_index, artifact in sorted(artifacts.items()):
        if set(pd.to_numeric(artifact["shard_index"], errors="raise")) != {
            shard_index
        }:
            raise ValueError("A score artifact is assigned to the wrong shard.")
        if set(artifact["contract_sha256"].astype(str)) != {
            contract.contract_sha256
        }:
            raise ValueError("Cannot merge score artifacts from different contracts.")
        frames.append(artifact)
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Merged score shards contain duplicate pair identities.")
    expected_keys = expected_pair_grid[[*PAIR_KEY, "target"]].copy()
    observed_keys = merged[[*PAIR_KEY, "target"]].copy()
    for frame in (expected_keys, observed_keys):
        frame["row_uid"] = frame["row_uid"].astype(str)
        frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
        frame.sort_values(list(PAIR_KEY), inplace=True, kind="stable")
        frame.reset_index(drop=True, inplace=True)
    if not expected_keys.equals(observed_keys):
        raise ValueError("Merged score shards do not exactly cover the expected grid.")
    return merged.sort_values(list(PAIR_KEY), kind="stable").reset_index(drop=True)
