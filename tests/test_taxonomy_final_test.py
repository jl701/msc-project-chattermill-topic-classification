from __future__ import annotations

import json
from pathlib import Path

import pytest

from msc_project.experiments.taxonomy_final_test import (
    FIXED_COMPOSITION_METHOD,
    PROTOCOL_ID,
    assert_pre_release_cannot_execute,
    build_final_test_jobs,
    job_plan,
    ordered_job_ids,
    preregistration_sha256,
    validate_job_graph,
    validate_preregistration,
    validate_release_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "taxonomy_final_test_v1.json"
RELEASE_TEMPLATE = (
    ROOT
    / "configs"
    / "experiments"
    / "taxonomy_final_test_release_TEMPLATE.json"
)


def load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_preregistration_and_exact_job_graph() -> None:
    config = load_config()
    audit = validate_preregistration(config)
    assert audit["status"] == "pass"
    assert audit["test_contract_count"] == 0

    jobs = build_final_test_jobs(config)
    assert validate_job_graph(jobs) == {
        "score": 93,
        "compose": 12,
        "seal": 1,
        "analyse": 1,
    }
    assert len(jobs) == 107
    assert len(ordered_job_ids(jobs)) == 107
    hybrid = [job for job in jobs if job.method_id == FIXED_COMPOSITION_METHOD]
    assert len(hybrid) == 12
    assert all(len(job.depends_on) == 2 for job in hybrid)


def test_plan_is_pre_release_and_contains_no_test_contract() -> None:
    plan = job_plan(load_config())
    assert plan["protocol_id"] == PROTOCOL_ID
    assert plan["include_official_test"] is False
    assert plan["test_contract_count"] == 0
    assert plan["job_counts"]["score"] == 93


def test_preregistration_rejects_roster_or_governance_drift() -> None:
    config = load_config()
    config["roster"]["confirmatory_L2_D"].append("global_router")
    with pytest.raises(ValueError, match="confirmatory"):
        validate_preregistration(config)

    config = load_config()
    config["governance"]["interim_outcome_inspection"] = True
    with pytest.raises(ValueError, match="prohibited"):
        validate_preregistration(config)


def test_pre_release_execute_fails_closed() -> None:
    config = load_config()
    with pytest.raises(PermissionError, match="no one-time release"):
        assert_pre_release_cannot_execute(config, None)
    with pytest.raises(PermissionError, match="not authorised"):
        assert_pre_release_cannot_execute(config, {"status": "draft"})
    template = json.loads(RELEASE_TEMPLATE.read_text(encoding="utf-8"))
    assert template["include_official_test"] is False
    with pytest.raises(PermissionError, match="not authorised"):
        assert_pre_release_cannot_execute(config, template)


def test_release_must_bind_config_commit_and_explicit_authorisation() -> None:
    config = load_config()
    commit = "1" * 40
    release = {
        "schema_version": "taxonomy_final_test_release_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "authorised_once",
        "include_official_test": True,
        "explicit_user_authorisation": True,
        "preregistration_sha256": preregistration_sha256(config),
        "execution_commit": commit,
        "authorised_at": "2026-08-24T12:00:00+01:00",
        "authorisation_id": "example-not-valid-for-real-execution",
        "official_test_file_sha256": "a" * 64,
        "official_test_row_uid_sha256": "b" * 64,
        "official_test_rows": 1587,
        "official_test_relative_path": "test.csv",
        "backup_root_id": "immutable-final-test-backup-v1",
        "batch_fallback_policy": "none_no_post_release_batch_change",
    }
    audit = validate_release_manifest(release, config, execution_commit=commit)
    assert audit["include_official_test"] is True
    assert audit["test_contract_count"] == 1

    release["explicit_user_authorisation"] = False
    with pytest.raises(PermissionError, match="Explicit user"):
        validate_release_manifest(release, config, execution_commit=commit)
