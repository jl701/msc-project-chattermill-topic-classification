from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_few_shot_cache import FormalFewShotCache
from msc_project.experiments.taxonomy_execution import canonical_sha256


def _cache(root: Path) -> FormalFewShotCache:
    return FormalFewShotCache(
        root / "cache",
        protocol_id="taxonomy_two_stage_formal_v1",
        scope_id="heldout-a01",
        scope_contract_sha256=canonical_sha256({"scope": "heldout-a01"}),
        demonstration_sha256s={
            "aspect": canonical_sha256({"demo": "aspect"}),
            "sentiment": canonical_sha256({"demo": "sentiment"}),
        },
        artifact_root=root,
        chunk_size=2,
    )


def test_few_shot_cache_deduplicates_and_resumes_exact_chunks(tmp_path: Path) -> None:
    calls: list[int] = []

    def scorer(reviews, candidates):
        calls.append(len(reviews))
        return np.asarray([[0.7, 0.3] for _ in reviews], dtype=float)

    cache = _cache(tmp_path)
    first = cache.score(
        ["same", "same", "different"],
        ["aspect", "aspect", "aspect"],
        mode="aspect",
        scorer=scorer,
    )
    assert first.shape == (3, 2)
    assert calls == [2]
    assert cache.summary()["unique_cache_rows"] == 2

    resumed = _cache(tmp_path)
    second = resumed.score(
        ["same", "different"],
        ["aspect", "aspect"],
        mode="aspect",
        scorer=lambda *_: (_ for _ in ()).throw(AssertionError("rescored")),
    )
    assert np.allclose(second, [[0.7, 0.3], [0.7, 0.3]])
    assert resumed.summary()["new_inference_count"] == 0


def test_few_shot_cache_rejects_an_orphaned_chunk(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.root.mkdir(parents=True)
    (cache.root / "chunk-000000-deadbeefdeadbeef.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="orphaned"):
        _cache(tmp_path)
