from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msc_project.experiments.taxonomy_qwen_zero_shot_cache import (
    ReadOnlyQwenTwoStageCache,
)
from msc_project.llm.qwen_two_stage_classifier import (
    qwen_two_stage_contract_sha256,
)


def _key(contract: str, mode: str, review: str, candidate: str) -> str:
    payload = json.dumps(
        [contract, mode, review, candidate],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache(tmp_path: Path) -> tuple[Path, str]:
    contract = qwen_two_stage_contract_sha256(max_length=384)
    records = []
    for mode, values in (("aspect", [0.8, 0.2]), ("sentiment", [0.1, 0.3, 0.6])):
        records.append(
            {
                "contract_sha256": contract,
                "key": _key(contract, mode, "review", "candidate"),
                "mode": mode,
                "values": values,
            }
        )
    path = tmp_path / "scores.jsonl"
    path.write_text(
        "\n".join(json.dumps(value) for value in records) + "\n",
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_only_qwen_cache_requires_exact_file_and_prompt(tmp_path: Path) -> None:
    path, sha256 = _cache(tmp_path)
    cache = ReadOnlyQwenTwoStageCache(path, expected_file_sha256=sha256)
    grid = pd.DataFrame({"text": ["review"], "candidate_text": ["candidate"]})
    assert np.allclose(cache.score(grid, mode="aspect"), [[0.8, 0.2]])
    assert np.allclose(cache.score(grid, mode="sentiment"), [[0.1, 0.3, 0.6]])
    assert cache.summary()["new_inference_count"] == 0
    with pytest.raises(KeyError, match="cache miss"):
        cache.score(
            pd.DataFrame({"text": ["different"], "candidate_text": ["candidate"]}),
            mode="aspect",
        )


def test_read_only_qwen_cache_rejects_wrong_file_hash(tmp_path: Path) -> None:
    path, _ = _cache(tmp_path)
    with pytest.raises(ValueError, match="file hash mismatch"):
        ReadOnlyQwenTwoStageCache(path, expected_file_sha256="0" * 64)
