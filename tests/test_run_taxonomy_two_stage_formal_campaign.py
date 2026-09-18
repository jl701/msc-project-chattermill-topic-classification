from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_taxonomy_two_stage_formal_campaign.py"
SPEC = importlib.util.spec_from_file_location("formal_campaign", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_formal_job_uses_the_orchestrator_interpreter() -> None:
    job = MODULE.build_formal_jobs()[0]
    command = MODULE._materialize_job_command(job, local_files_only=True)

    assert command[0] == sys.executable
    assert command[1:-1] == list(job.command[1:])
    assert command[-1] == "--local-files-only"


def test_formal_job_rejects_an_unregistered_launcher() -> None:
    job = MODULE.build_formal_jobs()[0]
    invalid = replace(job, command=("python3", *job.command[1:]))

    with pytest.raises(ValueError, match="portable 'python' launcher"):
        MODULE._materialize_job_command(invalid, local_files_only=False)


def test_formal_job_uses_the_audited_persistent_model_cache(tmp_path: Path) -> None:
    hf_home = tmp_path / "hf-cache"
    (hf_home / "hub").mkdir(parents=True)

    environment = MODULE._materialize_job_environment(
        hf_home=hf_home,
        local_files_only=True,
    )

    assert environment["HF_HOME"] == str(hf_home.resolve())
    assert environment["HF_HUB_CACHE"] == str((hf_home / "hub").resolve())
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment is not os.environ


def test_formal_job_rejects_a_missing_persistent_model_cache(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cache is incomplete"):
        MODULE._materialize_job_environment(
            hf_home=tmp_path / "missing",
            local_files_only=True,
        )


def test_parallel_worker_dry_run_is_bound_to_the_registered_manifest(
    tmp_path: Path,
) -> None:
    hf_home = tmp_path / "hf-cache"
    (hf_home / "hub").mkdir(parents=True)
    args = argparse.Namespace(
        output_root=tmp_path / "worker-qlora-a",
        data_dir=tmp_path / "data",
        hf_home=hf_home,
        worker_id="worker-qlora-a",
        method=None,
        parallel_plan=MODULE.DEFAULT_PARALLEL_PLAN,
        start_after=None,
        max_jobs=None,
        local_files_only=True,
        dry_run=True,
    )

    result = MODULE.run(args)

    assert result["status"] == "pass"
    assert result["worker_id"] == "worker-qlora-a"
    assert result["planned_count"] == 65
    assert result["worker_manifest_sha256"]
    assert result["test_contract_count"] == 0
