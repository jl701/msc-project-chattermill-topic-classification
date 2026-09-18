"""Lossless, split-isolated raw-score cache for the frozen Qwen control.

The cache is an execution optimisation, not a scientific result artifact.
Existing fold-specific score artifacts and run contracts remain authoritative.
Only a content hash and the corresponding raw P(Y) are persisted; review text,
gold labels, thresholds, predictions, and metrics are never written here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_methods import (
    PairProbabilityRuntime,
    load_method_registry,
    method_registry_sha256,
    resolve_method_spec,
    validate_probability_scores,
)
from msc_project.llm.qwen_pair_classifier import (
    qwen_pair_scoring_contract_sha256,
)

if TYPE_CHECKING:
    from msc_project.experiments.taxonomy_execution import RunContract


FROZEN_METHOD_ID = "frozen_qwen_candidate_pair"
CACHE_SCHEMA_VERSION = "frozen_qwen_raw_score_cache_v1"
CACHE_KEY_COLUMNS = (
    "text",
    "candidate_text",
    "candidate_aspect",
    "candidate_sentiment",
    "representation_variant",
)
SQLITE_QUERY_CHUNK = 900


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenRawScoreCacheContract:
    schema_version: str
    scientific_protocol_sha256: str
    method_id: str
    method_spec_sha256: str
    method_registry_sha256: str
    model_id: str
    model_revision: str
    scientific_parameters_sha256: str
    scoring_contract_sha256: str
    split: str

    def validate(self) -> None:
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError("Unexpected frozen-score cache schema version.")
        if self.method_id != FROZEN_METHOD_ID:
            raise ValueError("Raw-score caching is restricted to frozen Qwen.")
        if self.split not in {"validation", "test"}:
            raise ValueError("Frozen-score caches must be validation- or test-specific.")
        hash_fields = (
            self.scientific_protocol_sha256,
            self.method_spec_sha256,
            self.method_registry_sha256,
            self.scientific_parameters_sha256,
            self.scoring_contract_sha256,
        )
        if any(len(value) != 64 for value in hash_fields):
            raise ValueError("Frozen-score cache contract contains an invalid SHA-256.")
        if not self.model_id or not self.model_revision:
            raise ValueError("Frozen-score cache contract must pin model ID and revision.")

    @property
    def contract_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, str]:
        value = asdict(self)
        value["contract_sha256"] = self.contract_sha256
        return value


def build_frozen_raw_score_cache_contract(
    run_contract: "RunContract",
    scientific_parameters: Mapping[str, object],
    *,
    registry: Mapping[str, object] | None = None,
) -> FrozenRawScoreCacheContract:
    """Derive a fold-independent cache contract from a guarded run contract."""

    if run_contract.method_id != FROZEN_METHOD_ID:
        raise ValueError("Only frozen Qwen can build a frozen raw-score cache.")
    parameter_digest = _canonical_sha256(dict(scientific_parameters))
    if parameter_digest != run_contract.scientific_parameters_sha256:
        raise ValueError("Cache parameters differ from the run contract.")
    method_registry = dict(registry or load_method_registry())
    spec = resolve_method_spec(FROZEN_METHOD_ID, method_registry)
    registry_digest = method_registry_sha256(method_registry)
    if run_contract.method_spec_sha256 != spec.spec_sha256:
        raise ValueError("Run contract uses a stale frozen-Qwen method specification.")
    if run_contract.method_registry_sha256 != registry_digest:
        raise ValueError("Run contract uses a stale method registry.")
    if not spec.model_id or not spec.model_revision:
        raise ValueError("Frozen Qwen registry entry must pin model ID and revision.")
    max_length = int(scientific_parameters["max_length"])
    contract = FrozenRawScoreCacheContract(
        schema_version=CACHE_SCHEMA_VERSION,
        scientific_protocol_sha256=run_contract.scientific_protocol_sha256,
        method_id=FROZEN_METHOD_ID,
        method_spec_sha256=spec.spec_sha256,
        method_registry_sha256=registry_digest,
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        scientific_parameters_sha256=parameter_digest,
        scoring_contract_sha256=qwen_pair_scoring_contract_sha256(
            max_length=max_length
        ),
        split=run_contract.split,
    )
    contract.validate()
    return contract


def frozen_raw_score_cache_path(
    output_root: Path,
    contract: FrozenRawScoreCacheContract,
) -> Path:
    contract.validate()
    return (
        output_root
        / "_raw_score_cache"
        / FROZEN_METHOD_ID
        / contract.split
        / f"{contract.contract_sha256}.sqlite3"
    )


@dataclass(frozen=True)
class FrozenCacheScoreStats:
    requested_rows: int
    unique_inputs: int
    cached_unique_inputs: int
    model_scored_unique_inputs: int

    @property
    def cache_hit_fraction(self) -> float:
        if self.unique_inputs == 0:
            return 0.0
        return self.cached_unique_inputs / self.unique_inputs

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "cache_hit_fraction": self.cache_hit_fraction,
        }


def _input_sha256(row: Mapping[str, object]) -> str:
    payload = {
        column: "" if pd.isna(row[column]) else str(row[column])
        for column in CACHE_KEY_COLUMNS
    }
    if not payload["candidate_text"].strip():
        raise ValueError("Cache input candidate_text must be non-empty.")
    return _canonical_sha256(payload)


def frozen_cache_input_keys(pair_manifest: pd.DataFrame) -> list[str]:
    missing = sorted(set(CACHE_KEY_COLUMNS) - set(pair_manifest.columns))
    if missing:
        raise ValueError(f"Frozen-score cache manifest is missing columns: {missing}")
    if pair_manifest.empty:
        raise ValueError("Frozen-score cache cannot score an empty manifest.")
    return [
        _input_sha256(row)
        for row in pair_manifest[list(CACHE_KEY_COLUMNS)].to_dict(orient="records")
    ]


class FrozenRawScoreCache:
    """SQLite-backed mapping from exact model input hashes to raw P(Y)."""

    def __init__(
        self,
        path: Path,
        contract: FrozenRawScoreCacheContract,
    ) -> None:
        contract.validate()
        self.path = path
        self.contract = contract
        self._requested_rows = 0
        self._unique_inputs = 0
        self._cached_unique_inputs = 0
        self._model_scored_unique_inputs = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), timeout=60.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS scores "
            "(input_sha256 TEXT PRIMARY KEY, "
            "raw_present_probability REAL NOT NULL "
            "CHECK(raw_present_probability >= 0.0 "
            "AND raw_present_probability <= 1.0)) WITHOUT ROWID"
        )
        self._validate_or_initialise_metadata()
        integrity = self._connection.execute("PRAGMA quick_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            self.close()
            raise ValueError(f"Frozen-score cache failed SQLite integrity check: {integrity}")

    def _validate_or_initialise_metadata(self) -> None:
        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "contract_sha256": self.contract.contract_sha256,
            "contract_json": _canonical_json(self.contract.to_dict()),
        }
        observed = dict(
            self._connection.execute("SELECT key, value FROM metadata").fetchall()
        )
        if not observed:
            with self._connection:
                self._connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    expected.items(),
                )
            return
        if observed != expected:
            self.close()
            raise ValueError(
                "Frozen-score cache metadata is incompatible with the requested contract."
            )

    def get_many(self, keys: Sequence[str]) -> dict[str, float]:
        unique = list(dict.fromkeys(str(value) for value in keys))
        values: dict[str, float] = {}
        for start in range(0, len(unique), SQLITE_QUERY_CHUNK):
            chunk = unique[start : start + SQLITE_QUERY_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                "SELECT input_sha256, raw_present_probability FROM scores "
                f"WHERE input_sha256 IN ({placeholders})",
                chunk,
            ).fetchall()
            values.update((str(key), float(score)) for key, score in rows)
        return values

    def put_many(self, values: Mapping[str, float]) -> None:
        if not values:
            return
        keys = list(values)
        scores = validate_probability_scores(
            list(values.values()),
            expected_rows=len(values),
        )
        with self._connection:
            self._connection.executemany(
                "INSERT OR IGNORE INTO scores"
                "(input_sha256, raw_present_probability) VALUES (?, ?)",
                [(key, float(score)) for key, score in zip(keys, scores)],
            )
        persisted = self.get_many(keys)
        for key, expected in zip(keys, scores):
            if key not in persisted:
                raise RuntimeError("Frozen-score cache write did not persist a score.")
            if not np.isclose(
                persisted[key],
                float(expected),
                rtol=1e-7,
                atol=1e-8,
            ):
                raise ValueError(
                    "Concurrent frozen-score cache writers produced inconsistent scores."
                )

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM scores").fetchone()
        return int(row[0])

    def record(self, stats: FrozenCacheScoreStats) -> None:
        self._requested_rows += stats.requested_rows
        self._unique_inputs += stats.unique_inputs
        self._cached_unique_inputs += stats.cached_unique_inputs
        self._model_scored_unique_inputs += stats.model_scored_unique_inputs

    def session_summary(self) -> dict[str, object]:
        stats = FrozenCacheScoreStats(
            requested_rows=self._requested_rows,
            unique_inputs=self._unique_inputs,
            cached_unique_inputs=self._cached_unique_inputs,
            model_scored_unique_inputs=self._model_scored_unique_inputs,
        )
        return {
            "cache_path": str(self.path),
            "cache_contract_sha256": self.contract.contract_sha256,
            "split": self.contract.split,
            "persisted_unique_inputs": self.count(),
            **stats.to_dict(),
        }

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self) -> "FrozenRawScoreCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def score_frozen_qwen_with_cache(
    runtime: PairProbabilityRuntime,
    pair_manifest: pd.DataFrame,
    cache: FrozenRawScoreCache,
) -> tuple[np.ndarray, FrozenCacheScoreStats]:
    """Return raw scores in manifest order, inferring only uncached inputs."""

    if runtime.method_id != FROZEN_METHOD_ID:
        raise ValueError("Frozen-score cache received a non-frozen-Qwen runtime.")
    if cache.contract.method_id != runtime.method_id:
        raise ValueError("Frozen-score cache and runtime method differ.")
    if cache.contract.split == "validation":
        if "is_heldout" not in pair_manifest.columns:
            raise ValueError("Validation cache input must expose is_heldout.")
        if pair_manifest["is_heldout"].astype(bool).any():
            raise ValueError(
                "Strict validation cache input contains a held-out candidate."
            )

    keys = frozen_cache_input_keys(pair_manifest)
    unique_keys = list(dict.fromkeys(keys))
    cached = cache.get_many(unique_keys)
    first_index = {}
    for index, key in enumerate(keys):
        first_index.setdefault(key, index)
    missing_keys = [key for key in unique_keys if key not in cached]
    if missing_keys:
        missing_frame = pair_manifest.iloc[
            [first_index[key] for key in missing_keys]
        ].reset_index(drop=True)
        missing_scores = validate_probability_scores(
            runtime.score(missing_frame),
            expected_rows=len(missing_frame),
        )
        new_values = {
            key: float(score)
            for key, score in zip(missing_keys, missing_scores)
        }
        cache.put_many(new_values)
        cached.update(cache.get_many(missing_keys))
    if set(cached) != set(unique_keys):
        raise RuntimeError("Frozen-score cache failed to resolve every input.")
    ordered = validate_probability_scores(
        [cached[key] for key in keys],
        expected_rows=len(keys),
    )
    stats = FrozenCacheScoreStats(
        requested_rows=len(pair_manifest),
        unique_inputs=len(unique_keys),
        cached_unique_inputs=len(unique_keys) - len(missing_keys),
        model_scored_unique_inputs=len(missing_keys),
    )
    cache.record(stats)
    return ordered, stats
