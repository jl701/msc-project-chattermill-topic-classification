"""Immutable chunk cache for formal frozen-Qwen few-shot probabilities."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.verified_artifact_sync import (
    file_sha256,
    publish_artifact_unit,
)


CACHE_RECORD_SCHEMA = "taxonomy_frozen_qwen_few_shot_cache_record_v1"
CACHE_CHUNK_SCHEMA = "taxonomy_frozen_qwen_few_shot_cache_chunk_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FormalFewShotCache:
    """Cache exact few-shot query contracts in independently sealed chunks."""

    def __init__(
        self,
        root: Path,
        *,
        protocol_id: str,
        scope_id: str,
        scope_contract_sha256: str,
        demonstration_sha256s: Mapping[str, str],
        artifact_root: Path,
        chunk_size: int = 96,
    ) -> None:
        if not protocol_id or not scope_id:
            raise ValueError("Few-shot cache protocol and scope must be non-empty.")
        if not SHA256_PATTERN.fullmatch(scope_contract_sha256):
            raise ValueError("Few-shot scope contract must be a lowercase SHA-256.")
        if set(demonstration_sha256s) != {"aspect", "sentiment"} or any(
            not SHA256_PATTERN.fullmatch(str(value))
            for value in demonstration_sha256s.values()
        ):
            raise ValueError("Few-shot cache requires two demonstration SHA-256 values.")
        if chunk_size < 1:
            raise ValueError("Few-shot cache chunk size must be positive.")
        self.root = root
        self.protocol_id = protocol_id
        self.scope_id = scope_id
        self.scope_contract_sha256 = scope_contract_sha256
        self.demonstration_sha256s = {
            key: str(value) for key, value in demonstration_sha256s.items()
        }
        self.artifact_root = artifact_root
        self.chunk_size = int(chunk_size)
        self.values: dict[str, tuple[str, tuple[float, ...]]] = {}
        self.chunk_count = 0
        self.requested_rows = 0
        self.new_inference_count = 0
        self._load()

    def _key(self, review: str, candidate: str, mode: str) -> str:
        if mode not in {"aspect", "sentiment"}:
            raise ValueError(f"Unknown few-shot cache mode: {mode!r}")
        return canonical_sha256(
            {
                "scope_contract_sha256": self.scope_contract_sha256,
                "demonstrations_sha256": self.demonstration_sha256s[mode],
                "mode": mode,
                "review": str(review),
                "candidate": str(candidate),
            }
        )

    @staticmethod
    def _validate_values(mode: str, raw: Sequence[float]) -> tuple[float, ...]:
        width = 2 if mode == "aspect" else 3 if mode == "sentiment" else 0
        values = tuple(float(value) for value in raw)
        if (
            width == 0
            or len(values) != width
            or not np.isfinite(values).all()
            or min(values) < 0.0
            or max(values) > 1.0
            or not np.isclose(sum(values), 1.0, atol=1e-5)
        ):
            raise ValueError("Few-shot cache probabilities are invalid.")
        return values

    def _chunk_paths(self, index: int, payload_sha256: str) -> tuple[Path, Path]:
        stem = f"chunk-{index:06d}-{payload_sha256[:16]}"
        return self.root / f"{stem}.jsonl", self.root / f"{stem}.manifest.json"

    def _load(self) -> None:
        if not self.root.exists():
            return
        manifests = sorted(self.root.glob("chunk-*.manifest.json"))
        data_files = sorted(self.root.glob("chunk-*.jsonl"))
        declared_data: set[Path] = set()
        observed_indices: list[int] = []
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get(
                "schema_version"
            ) != CACHE_CHUNK_SCHEMA:
                raise ValueError("Unexpected few-shot cache chunk schema.")
            expected_manifest_sha = str(manifest.get("manifest_payload_sha256", ""))
            unhashed = dict(manifest)
            unhashed.pop("manifest_payload_sha256", None)
            if expected_manifest_sha != canonical_sha256(unhashed):
                raise ValueError("Few-shot cache chunk manifest hash mismatch.")
            if (
                manifest.get("protocol_id") != self.protocol_id
                or manifest.get("scope_id") != self.scope_id
                or manifest.get("scope_contract_sha256")
                != self.scope_contract_sha256
                or manifest.get("demonstration_sha256s")
                != self.demonstration_sha256s
                or manifest.get("test_contract_count") != 0
            ):
                raise ValueError("Few-shot cache chunk belongs to another contract.")
            index = int(manifest.get("chunk_index", -1))
            observed_indices.append(index)
            data_path = self.root / str(manifest.get("data_file", ""))
            declared_data.add(data_path)
            if not data_path.is_file() or file_sha256(data_path) != manifest.get(
                "data_sha256"
            ):
                raise ValueError("Few-shot cache data file hash mismatch.")
            records = [
                json.loads(line)
                for line in data_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(records) != int(manifest.get("record_count", -1)) or not records:
                raise ValueError("Few-shot cache chunk row count is invalid.")
            payload_sha = canonical_sha256(records)
            expected_stem = f"chunk-{index:06d}-{payload_sha[:16]}"
            if data_path.stem != expected_stem or manifest_path.name != (
                f"{expected_stem}.manifest.json"
            ):
                raise ValueError("Few-shot cache chunk name is not content-addressed.")
            for record in records:
                if record.get("schema_version") != CACHE_RECORD_SCHEMA:
                    raise ValueError("Unexpected few-shot cache record schema.")
                key = str(record.get("key", ""))
                mode = str(record.get("mode", ""))
                if not SHA256_PATTERN.fullmatch(key):
                    raise ValueError("Few-shot cache key is invalid.")
                values = self._validate_values(mode, record.get("values", []))
                current = (mode, values)
                previous = self.values.get(key)
                if previous is not None and previous != current:
                    raise ValueError("Conflicting few-shot cache key.")
                self.values[key] = current
            publish_artifact_unit(
                self.artifact_root,
                [
                    data_path.relative_to(self.artifact_root),
                    manifest_path.relative_to(self.artifact_root),
                ],
                unit_id=(
                    f"few-shot-cache-{self.scope_id}-{index:06d}-"
                    f"{payload_sha[:12]}"
                ),
                protocol_id=self.protocol_id,
                contract_sha256=self.scope_contract_sha256,
            )
        if declared_data != set(data_files):
            raise ValueError("Partial or orphaned few-shot cache chunk detected.")
        if observed_indices != list(range(len(manifests))):
            raise ValueError("Few-shot cache chunk sequence is incomplete.")
        self.chunk_count = len(manifests)

    def score(
        self,
        reviews: Sequence[str],
        candidates: Sequence[str],
        *,
        mode: str,
        scorer: Callable[[Sequence[str], Sequence[str]], np.ndarray],
    ) -> np.ndarray:
        if not reviews or len(reviews) != len(candidates):
            raise ValueError("Few-shot cache inputs must be non-empty and aligned.")
        keys = [
            self._key(str(review), str(candidate), mode)
            for review, candidate in zip(reviews, candidates)
        ]
        missing_positions: list[int] = []
        observed_missing: set[str] = set()
        for index, key in enumerate(keys):
            if key not in self.values and key not in observed_missing:
                observed_missing.add(key)
                missing_positions.append(index)
        for start in range(0, len(missing_positions), self.chunk_size):
            positions = missing_positions[start : start + self.chunk_size]
            probabilities = np.asarray(
                scorer(
                    [str(reviews[index]) for index in positions],
                    [str(candidates[index]) for index in positions],
                ),
                dtype=float,
            )
            width = 2 if mode == "aspect" else 3 if mode == "sentiment" else 0
            if probabilities.shape != (len(positions), width):
                raise ValueError("Few-shot scorer returned an invalid matrix shape.")
            records: list[dict[str, object]] = []
            for position, raw_values in zip(positions, probabilities):
                key = keys[position]
                values = self._validate_values(mode, raw_values)
                records.append(
                    {
                        "schema_version": CACHE_RECORD_SCHEMA,
                        "key": key,
                        "mode": mode,
                        "values": list(values),
                    }
                )
                self.values[key] = (mode, values)
            data_text = "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            )
            payload_sha = canonical_sha256(records)
            data_path, manifest_path = self._chunk_paths(self.chunk_count, payload_sha)
            if data_path.exists() or manifest_path.exists():
                raise FileExistsError("Few-shot cache chunk path already exists.")
            self.root.mkdir(parents=True, exist_ok=True)
            data_tmp = data_path.with_name(f".{data_path.name}.tmp-{os.getpid()}")
            manifest_tmp = manifest_path.with_name(
                f".{manifest_path.name}.tmp-{os.getpid()}"
            )
            data_tmp.write_text(data_text, encoding="utf-8", newline="\n")
            os.replace(data_tmp, data_path)
            manifest: dict[str, object] = {
                "schema_version": CACHE_CHUNK_SCHEMA,
                "protocol_id": self.protocol_id,
                "scope_id": self.scope_id,
                "scope_contract_sha256": self.scope_contract_sha256,
                "demonstration_sha256s": self.demonstration_sha256s,
                "chunk_index": self.chunk_count,
                "data_file": data_path.name,
                "data_sha256": file_sha256(data_path),
                "record_count": len(records),
                "test_contract_count": 0,
            }
            manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
            manifest_tmp.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(manifest_tmp, manifest_path)
            publish_artifact_unit(
                self.artifact_root,
                [
                    data_path.relative_to(self.artifact_root),
                    manifest_path.relative_to(self.artifact_root),
                ],
                unit_id=(
                    f"few-shot-cache-{self.scope_id}-{self.chunk_count:06d}-"
                    f"{payload_sha[:12]}"
                ),
                protocol_id=self.protocol_id,
                contract_sha256=self.scope_contract_sha256,
            )
            self.chunk_count += 1
            self.new_inference_count += len(records)
        output = np.asarray([self.values[key][1] for key in keys], dtype=float)
        width = 2 if mode == "aspect" else 3
        if output.shape != (len(keys), width) or not np.isfinite(output).all():
            raise ValueError("Few-shot cache produced invalid probabilities.")
        self.requested_rows += len(keys)
        return output

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": "taxonomy_frozen_qwen_few_shot_cache_summary_v1",
            "scope_id": self.scope_id,
            "scope_contract_sha256": self.scope_contract_sha256,
            "demonstration_sha256s": self.demonstration_sha256s,
            "chunk_count": self.chunk_count,
            "unique_cache_rows": len(self.values),
            "requested_rows": self.requested_rows,
            "new_inference_count": self.new_inference_count,
            "failure_count": 0,
            "test_contract_count": 0,
        }
