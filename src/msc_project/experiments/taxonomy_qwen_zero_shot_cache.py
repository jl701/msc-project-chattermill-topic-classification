"""Read-only exact prompt cache for frozen-Qwen true-two-stage re-decoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from msc_project.llm.qwen_two_stage_classifier import (
    qwen_two_stage_contract_sha256,
)


class ReadOnlyQwenTwoStageCache:
    """Resolve only byte-identical prompts; never infer or append on a miss."""

    def __init__(
        self,
        path: Path,
        *,
        expected_file_sha256: str,
        max_length: int = 384,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_file_sha256 != expected_file_sha256:
            raise ValueError(
                "Frozen-Qwen cache file hash mismatch: "
                f"expected={expected_file_sha256}, observed={observed_file_sha256}."
            )
        self.path = path
        self.file_sha256 = observed_file_sha256
        self.contract_sha256 = qwen_two_stage_contract_sha256(
            max_length=max_length
        )
        self.values: dict[str, tuple[str, tuple[float, ...]]] = {}
        self.requests = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("contract_sha256") != self.contract_sha256:
                raise ValueError(
                    f"Frozen-Qwen cache contract mismatch at line {line_number}."
                )
            mode = str(record.get("mode"))
            width = 2 if mode == "aspect" else 3 if mode == "sentiment" else 0
            values = tuple(float(value) for value in record.get("values", []))
            if width == 0 or len(values) != width or not np.isfinite(values).all():
                raise ValueError(
                    f"Frozen-Qwen cache value is invalid at line {line_number}."
                )
            key = str(record.get("key", ""))
            if not key:
                raise ValueError(f"Frozen-Qwen cache key is empty at line {line_number}.")
            previous = self.values.get(key)
            current = (mode, values)
            if previous is not None and previous != current:
                raise ValueError(f"Conflicting Frozen-Qwen cache key: {key}.")
            self.values[key] = current

    def _key(self, review: str, candidate: str, mode: str) -> str:
        payload = json.dumps(
            [self.contract_sha256, mode, review, candidate],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def score(self, grid: pd.DataFrame, *, mode: str) -> np.ndarray:
        if mode not in {"aspect", "sentiment"}:
            raise ValueError(f"Unknown Frozen-Qwen cache mode: {mode!r}.")
        required = {"text", "candidate_text"}
        missing_columns = sorted(required - set(grid.columns))
        if missing_columns or grid.empty:
            raise ValueError(
                f"Frozen-Qwen cache grid is invalid; missing={missing_columns}."
            )
        keys = [
            self._key(str(review), str(candidate), mode)
            for review, candidate in zip(grid["text"], grid["candidate_text"])
        ]
        missing_keys = sorted(set(keys) - set(self.values))
        wrong_mode = sorted(
            {key for key in keys if key in self.values and self.values[key][0] != mode}
        )
        if missing_keys or wrong_mode:
            raise KeyError(
                "Exact Frozen-Qwen cache miss; inference is disabled for this run: "
                f"missing_unique={len(missing_keys)}, wrong_mode={len(wrong_mode)}."
            )
        output = np.asarray([self.values[key][1] for key in keys], dtype=float)
        width = 2 if mode == "aspect" else 3
        if output.shape != (len(grid), width) or not np.isfinite(output).all():
            raise ValueError("Frozen-Qwen read-only cache produced invalid probabilities.")
        self.requests += len(grid)
        return output

    def summary(self) -> dict[str, object]:
        modes = {"aspect": 0, "sentiment": 0}
        for mode, _ in self.values.values():
            modes[mode] += 1
        return {
            "mode": "read_only_exact",
            "path": self.path.as_posix(),
            "file_sha256": self.file_sha256,
            "contract_sha256": self.contract_sha256,
            "unique_cache_rows": len(self.values),
            "modes": modes,
            "requested_rows": self.requests,
            "cache_miss_count": 0,
            "new_inference_count": 0,
        }
