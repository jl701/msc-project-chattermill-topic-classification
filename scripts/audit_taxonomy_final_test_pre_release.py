"""Build the final pre-release scientific-freeze audit without test access."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.taxonomy_final_test import (
    assert_pre_release_cannot_execute,
    canonical_sha256,
    file_sha256,
    load_json_object,
    preregistration_sha256,
    validate_artifact_inventory,
    validate_preregistration,
)
from msc_project.experiments.taxonomy_final_test_workflow import (
    atomic_json,
)

CONFIG = PROJECT_ROOT / "configs/experiments/taxonomy_final_test_v1.json"
RELEASE_TEMPLATE = (
    PROJECT_ROOT / "configs/experiments/taxonomy_final_test_release_TEMPLATE.json"
)
INVENTORY = (
    PROJECT_ROOT / "docs/experiments/taxonomy_final_test_v1_artifact_inventory.json"
)
JOB_PLAN = PROJECT_ROOT / "docs/experiments/taxonomy_final_test_v1_job_plan.json"
PREFLIGHT = (
    PROJECT_ROOT / "docs/experiments/taxonomy_final_test_v1_preflight_audit.json"
)
MIRROR = (
    PROJECT_ROOT
    / "docs/experiments/taxonomy_final_test_v1_validation_mirror_audit.json"
)
JUNIT = PROJECT_ROOT / "docs/experiments/taxonomy_final_test_v1_pytest.xml"
DRY_ROOT = (
    PROJECT_ROOT
    / "outputs/experimental/taxonomy_final_test_v1_dry_run_20260824_05"
)
THESIS_PDF = PROJECT_ROOT / "thesis/main.pdf"
THESIS_LOG = PROJECT_ROOT / "thesis/main.log"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "docs/experiments/taxonomy_final_test_v1_pre_release_freeze_audit.json"
)

BINDINGS = (
    "configs/experiments/taxonomy_final_test_release_TEMPLATE.json",
    "configs/experiments/taxonomy_final_test_v1.json",
    "docs/dissertation_taxonomy_post_supervisor_plan_2026_08_20.md",
    "docs/experiment_reproducibility_register.md",
    "docs/experiment_log.md",
    "docs/taxonomy_current_authority_20260824.md",
    "docs/experiments/taxonomy_final_scientific_freeze_v2_20260824.md",
    "docs/experiments/taxonomy_final_test_v1_artifact_inventory.json",
    "docs/experiments/taxonomy_final_test_v1_execution_and_cost_plan_20260824.md",
    "docs/experiments/taxonomy_final_test_v1_phase6_readiness_20260824.md",
    "docs/experiments/taxonomy_final_test_v1_job_plan.json",
    "docs/experiments/taxonomy_final_test_v1_preflight_audit.json",
    "docs/experiments/taxonomy_final_test_v1_preregistration.md",
    "docs/experiments/taxonomy_final_test_v1_validation_mirror_audit.json",
    "scripts/audit_taxonomy_final_test_pre_release.py",
    "scripts/audit_taxonomy_final_test_validation_mirror.py",
    "scripts/build_taxonomy_final_test_artifact_inventory.py",
    "scripts/run_taxonomy_final_test.py",
    "src/msc_project/experiments/taxonomy_final_test.py",
    "src/msc_project/experiments/taxonomy_final_test_analysis.py",
    "src/msc_project/experiments/taxonomy_final_test_artifacts.py",
    "src/msc_project/experiments/taxonomy_final_test_scoring.py",
    "src/msc_project/experiments/taxonomy_final_test_workflow.py",
    "tests/test_taxonomy_final_test.py",
    "tests/test_taxonomy_final_test_analysis.py",
    "tests/test_taxonomy_final_test_artifacts.py",
    "tests/test_taxonomy_final_test_workflow.py",
    "tests/test_run_taxonomy_final_test.py",
    "thesis/chapters/01_introduction.tex",
    "thesis/chapters/04_methods.tex",
    "thesis/main.tex",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _payload_hash(value: dict[str, Any], key: str) -> str:
    payload = dict(value)
    observed = payload.pop(key, None)
    expected = canonical_sha256(payload)
    if observed != expected:
        raise ValueError(f"Payload hash mismatch for {key}.")
    return str(observed)


def _junit_audit(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    values = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if values != {"tests": 503, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"Whole-repository pytest evidence changed: {values!r}.")
    return {"status": "pass", **values, "sha256": file_sha256(path)}


def _thesis_audit() -> dict[str, object]:
    log = THESIS_LOG.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = (
        "Fatal error occurred",
        "! LaTeX Error",
        "Undefined control sequence",
        "Citation(s) may have changed",
        "There were undefined references",
    )
    fatal = [value for value in fatal_patterns if value in log]
    if fatal or not THESIS_PDF.is_file():
        raise ValueError(f"Thesis build audit failed: {fatal!r}.")
    matches = re.findall(r"Output written on main\.pdf \((\d+) pages?\)", log)
    if not matches:
        raise ValueError("Thesis log does not record a completed PDF.")
    return {
        "status": "pass",
        "pages": int(matches[-1]),
        "pdf_bytes": THESIS_PDF.stat().st_size,
        "pdf_sha256": file_sha256(THESIS_PDF),
        "fatal_error_count": 0,
    }


def _security_scan(paths: list[Path]) -> dict[str, object]:
    patterns = {
        "private_key": re.compile(r"BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY"),
        "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    }
    hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in patterns.items():
            if pattern.search(text):
                hits.append(f"{name}:{path.relative_to(PROJECT_ROOT).as_posix()}")
    if hits:
        raise ValueError(f"Potential secret in final changeset: {hits!r}.")
    return {
        "status": "pass",
        "files_scanned": len(paths),
        "potential_secret_count": 0,
    }


def run(
    output: Path,
    *,
    superseded_release: Path | None = None,
    superseded_note: Path | None = None,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    config = load_json_object(CONFIG)
    config_audit = validate_preregistration(config)
    prereg_hash = preregistration_sha256(config)
    template = load_json_object(RELEASE_TEMPLATE)
    try:
        assert_pre_release_cannot_execute(config, template)
    except PermissionError:
        release_locked = True
    else:
        release_locked = False
    if not release_locked:
        raise AssertionError("Inert release template unexpectedly authorises execution.")

    inventory = load_json_object(INVENTORY)
    inventory_audit = validate_artifact_inventory(inventory, config)
    plan = load_json_object(JOB_PLAN)
    preflight = load_json_object(PREFLIGHT)
    mirror = load_json_object(MIRROR)
    dry = load_json_object(DRY_ROOT / "dry_run_summary.json")
    seal = load_json_object(DRY_ROOT / "sealed_scores/score_seal.json")
    analysis = load_json_object(
        DRY_ROOT / "single_reveal_analysis/analysis_manifest.json"
    )
    if (
        plan.get("preregistration_sha256") != prereg_hash
        or plan.get("job_counts")
        != {"score": 93, "compose": 12, "seal": 1, "analyse": 1}
        or len(plan.get("jobs", [])) != 107
        or int(plan.get("test_contract_count", -1)) != 0
    ):
        raise ValueError("Frozen job plan does not match the preregistration.")
    if (
        preflight.get("status") != "pass"
        or preflight.get("official_test_accessed") is not False
        or int(preflight.get("test_contract_count", -1)) != 0
        or int(preflight["artifact_audit"].get("hash_mismatch_count", -1)) != 0
    ):
        raise ValueError("Artifact preflight has not passed test-blindly.")
    deltas = mirror.get("expected_value_checks", {})
    if (
        mirror.get("status") != "pass"
        or mirror.get("official_test_opened") is not False
        or int(mirror.get("test_contract_count", -1)) != 0
        or not deltas
        or any(abs(float(value)) > 1e-12 for value in deltas.values())
    ):
        raise ValueError("Validation-mirror audit is incomplete or changed.")
    if (
        dry.get("status") != "pass"
        or int(dry.get("score_bundle_count", -1)) != 105
        or int(dry.get("backup_receipt_count", -1)) != 210
        or dry.get("official_test_accessed") is not False
        or int(dry.get("test_contract_count", -1)) != 0
        or int(dry.get("failure_count", -1)) != 0
    ):
        raise ValueError("Synthetic end-to-end dry-run has not passed.")
    _payload_hash(seal, "seal_payload_sha256")
    _payload_hash(analysis, "manifest_payload_sha256")
    if (
        int(seal.get("score_bundle_count", -1)) != 105
        or int(seal.get("backup_receipt_count", -1)) != 210
        or int(seal.get("test_contract_count", -1)) != 0
        or int(analysis.get("score_bundle_count", -1)) != 105
        or int(analysis.get("test_contract_count", -1)) != 0
    ):
        raise ValueError("Dry-run seal/reveal contract differs from the pre-release path.")

    paths = [(PROJECT_ROOT / value).resolve() for value in BINDINGS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Freeze binding file is missing: {missing!r}.")
    security = _security_scan(paths)
    hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    }
    if (superseded_release is None) != (superseded_note is None):
        raise ValueError(
            "Superseded release and note must be supplied together or omitted."
        )
    authorisation_state: dict[str, object]
    if superseded_release is not None and superseded_note is not None:
        release = load_json_object(superseded_release)
        note = superseded_note.read_text(encoding="utf-8")
        if (
            release.get("status") != "authorised_once"
            or release.get("explicit_user_authorisation") is not True
            or "SUPERSEDED BEFORE SCORING" not in note
            or "Metrics revealed: 0" not in note
        ):
            raise ValueError("Superseded pre-scoring release evidence is incomplete.")
        authorisation_state = {
            "official_test_identity_previously_bound": True,
            "official_test_outcomes_inspected": False,
            "superseded_release_execution_commit": release["execution_commit"],
            "superseded_release_sha256": file_sha256(superseded_release),
            "superseded_release_note_sha256": file_sha256(superseded_note),
            "replacement_release_required_after_repair_commit": True,
            "explicit_user_authorisation_present": True,
            "test_contract_count": 1,
        }
        audit_status = "pass_ready_to_bind_replacement_release_after_commit"
    else:
        authorisation_state = {
            "official_test_identity_previously_bound": False,
            "official_test_outcomes_inspected": False,
            "replacement_release_required_after_repair_commit": False,
            "explicit_user_authorisation_present": False,
            "test_contract_count": 0,
        }
        audit_status = "pass_ready_to_request_one_time_authorisation"

    result: dict[str, object] = {
        "schema_version": "taxonomy_final_test_pre_release_freeze_audit_v1",
        "status": audit_status,
        "protocol_id": config["protocol_id"],
        "preparation_parent_commit": _git("rev-parse", "HEAD"),
        "future_release_must_bind_clean_synced_execution_commit": True,
        "preregistration_sha256": prereg_hash,
        "preregistration_audit": config_audit,
        "artifact_inventory_audit": inventory_audit,
        "artifact_inventory_entry_manifest_sha256": inventory[
            "entry_manifest_sha256"
        ],
        "job_counts": plan["job_counts"],
        "job_plan_sha256": file_sha256(JOB_PLAN),
        "artifact_preflight_sha256": file_sha256(PREFLIGHT),
        "validation_mirror_sha256": file_sha256(MIRROR),
        "synthetic_dry_run": {
            "summary_sha256": file_sha256(DRY_ROOT / "dry_run_summary.json"),
            "score_bundle_count": 105,
            "backup_receipt_count": 210,
            "score_seal_payload_sha256": seal["seal_payload_sha256"],
            "analysis_manifest_payload_sha256": analysis[
                "manifest_payload_sha256"
            ],
        },
        "whole_repository_tests": _junit_audit(JUNIT),
        "thesis_build": _thesis_audit(),
        "security_scan": security,
        "bound_files": hashes,
        "bound_file_manifest_sha256": canonical_sha256(hashes),
        "release_template_locked": True,
        "include_official_test": False,
        "failure_count": 0,
        **authorisation_state,
    }
    atomic_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--superseded-release", type=Path)
    parser.add_argument("--superseded-note", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        args.output.resolve(),
        superseded_release=(
            args.superseded_release.resolve() if args.superseded_release else None
        ),
        superseded_note=(
            args.superseded_note.resolve() if args.superseded_note else None
        ),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "bound_files": len(result["bound_files"]),
                "pytest_tests": result["whole_repository_tests"]["tests"],
                "official_test_identity_previously_bound": result[
                    "official_test_identity_previously_bound"
                ],
                "official_test_outcomes_inspected": result[
                    "official_test_outcomes_inspected"
                ],
                "test_contract_count": result["test_contract_count"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
