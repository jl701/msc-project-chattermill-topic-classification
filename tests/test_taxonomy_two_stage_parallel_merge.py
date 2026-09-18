from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_two_stage_parallel import load_parallel_plan
from msc_project.experiments.taxonomy_two_stage_parallel_merge import (
    _verify_sync_units,
    expected_result_assignments,
)
from msc_project.experiments.verified_artifact_sync import publish_artifact_unit


PLAN = (
    PROJECT_ROOT
    / "configs/experiments/taxonomy_two_stage_three_gpu_parallel_v1.json"
)


def test_parallel_merge_assignment_is_exact_for_all_three_methods() -> None:
    assignments = expected_result_assignments(load_parallel_plan(PLAN))

    assert len(assignments) == 297
    assert Counter(key[0] for key in assignments) == {
        "distilbert_review_candidate_cross_encoder": 99,
        "qwen_candidate_pair_qlora": 99,
        "frozen_qwen_few_shot": 99,
    }


def test_parallel_merge_verifies_published_bytes_and_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "validation.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"test_contract_count": 0}\n', encoding="utf-8")
    publish_artifact_unit(
        tmp_path,
        [artifact.relative_to(tmp_path)],
        unit_id="validation-result",
        protocol_id="taxonomy_two_stage_formal_v1",
        contract_sha256="a" * 64,
    )

    audit = _verify_sync_units(tmp_path)

    assert audit["unit_count"] == 1
    assert audit["published_path_count"] == 1
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte count mismatch|SHA-256 mismatch"):
        _verify_sync_units(tmp_path)
