from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_taxonomy_two_stage_frozen_few_shot_campaign.py"
SPEC = importlib.util.spec_from_file_location("frozen_few_shot_campaign", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_campaign_dry_run_is_complete_and_validation_only(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "worker-distil-frozen",
        data_dir=tmp_path / "fabsa_data",
        config=(
            PROJECT_ROOT
            / "configs/experiments/taxonomy_two_stage_cloud_execution_safety_v1.json"
        ),
        parallel_plan=MODULE.DEFAULT_PARALLEL_PLAN,
        start_after=None,
        max_scopes=None,
        hf_home=tmp_path / "hf-cache",
        local_files_only=True,
        dry_run=True,
    )

    result = MODULE.run(args)

    assert result["status"] == "pass"
    assert result["worker_id"] == "worker-distil-frozen"
    assert result["method_id"] == "frozen_qwen_few_shot"
    assert result["planned_count"] == 26
    assert result["first_scope"] == "heldout-a01"
    assert result["last_scope"] == "heldout-a12-a01"
    assert result["worker_manifest_sha256"]
    assert result["failure_count"] == 0
    assert result["test_contract_count"] == 0


def test_frozen_campaign_subset_preserves_registered_order(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "worker-distil-frozen",
        data_dir=tmp_path / "fabsa_data",
        config=(
            PROJECT_ROOT
            / "configs/experiments/taxonomy_two_stage_cloud_execution_safety_v1.json"
        ),
        parallel_plan=MODULE.DEFAULT_PARALLEL_PLAN,
        start_after="heldout-a01",
        max_scopes=2,
        hf_home=tmp_path / "hf-cache",
        local_files_only=True,
        dry_run=True,
    )

    result = MODULE.run(args)

    assert result["planned_count"] == 2
    assert result["first_scope"] == "heldout-a01-a02"
    assert result["last_scope"] == "heldout-a02"
