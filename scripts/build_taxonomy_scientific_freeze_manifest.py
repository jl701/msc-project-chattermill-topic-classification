"""Seal the validation-only taxonomy scientific-freeze evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    ).strip()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _audit_safe(path: Path) -> dict[str, Any]:
    value = _read(path)
    if value.get("status", "pass") not in {"pass", "pass_with_reuse_restriction"}:
        raise ValueError(f"Audit is not PASS: {path}")
    for key in (
        "failure_count",
        "test_contract_count",
        "official_test_artifact_count",
        "non_finite_value_count",
        "resume_conflict_count",
    ):
        if key in value and int(value[key]) != 0:
            raise ValueError(f"Audit has unsafe {key}: {path}")
    if value.get("include_official_test") is True:
        raise ValueError(f"Audit includes official test: {path}")
    if value.get("official_test_opened") is True:
        raise ValueError(f"Audit opened official test: {path}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    worktree_status = _git("status", "--porcelain")
    if worktree_status:
        raise ValueError(
            "Scientific-freeze manifest requires a clean committed worktree."
        )
    audit_paths = [
        Path("docs/experiments/taxonomy_scientific_freeze_formal_evidence_audit_20260822.json"),
        Path("docs/experiments/taxonomy_scientific_freeze_local_replay_audit_20260822.json"),
        Path("docs/experiments/taxonomy_scientific_freeze_inference_audit_20260822.json"),
        Path("docs/experiments/taxonomy_scientific_freeze_reporting_audit_20260822.json"),
        Path("docs/experiments/taxonomy_l3_posthoc_robustness_audit_20260822.json"),
        Path("docs/experiments/taxonomy_qwen_matched_interface_audit_20260822.json"),
        Path("docs/experiments/taxonomy_few_shot_contract_audit_20260822.json"),
        Path("docs/experiments/taxonomy_rich_description_validation_confirmation_v1_audit.json"),
        Path("docs/experiments/taxonomy_capped_two_sentiment_validation_v2_audit.json"),
        Path("docs/experiments/taxonomy_matched_one_vs_two_stage_validation_v1_audit.json"),
    ]
    audited = {}
    for relative in audit_paths:
        path = PROJECT_ROOT / relative
        _audit_safe(path)
        audited[str(relative)] = _sha256(path)

    registered_inputs = [
        Path("configs/experiments/taxonomy_two_stage_three_gpu_parallel_v2.json"),
        Path("configs/experiments/taxonomy_two_stage_precloud_v2.json"),
        Path("configs/experiments/taxonomy_description_to_classifier_weight_transfer_v1.json"),
        Path("configs/experiments/taxonomy_rich_description_selected_interface_v1.json"),
        Path("configs/experiments/taxonomy_capped_two_sentiment_validation_v2.json"),
        Path("configs/experiments/taxonomy_matched_one_vs_two_stage_validation_v1.json"),
        Path("configs/experiments/fabsa_aspect_descriptions_minimal_v2.json"),
        Path("requirements.txt"),
        Path("requirements-llm.txt"),
        Path("requirements-taxonomy-cloud.txt"),
        Path("outputs/experimental/taxonomy_level1_closed_reference_v1/result.json"),
        Path("src/msc_project/experiments/taxonomy_two_stage.py"),
        Path("src/msc_project/experiments/taxonomy_two_stage_training.py"),
        Path("src/msc_project/experiments/taxonomy_two_stage_distilbert.py"),
        Path("src/msc_project/experiments/taxonomy_scientific_freeze.py"),
        Path("src/msc_project/llm/qwen_two_stage_classifier.py"),
        Path("scripts/run_taxonomy_two_stage_trainable_validation.py"),
        Path("scripts/run_taxonomy_level2_local_validation.py"),
        Path("scripts/run_taxonomy_description_weight_transfer.py"),
        Path("scripts/build_taxonomy_scientific_freeze_formal_evidence.py"),
        Path("scripts/audit_taxonomy_scientific_freeze_local_replay.py"),
        Path("scripts/analyse_taxonomy_scientific_freeze_v1.py"),
        Path("scripts/build_taxonomy_scientific_freeze_reporting.py"),
        Path("scripts/audit_taxonomy_l3_posthoc_robustness.py"),
        Path("scripts/audit_taxonomy_qwen_matched_interface.py"),
        Path("scripts/audit_taxonomy_few_shot_contract.py"),
        Path("scripts/audit_taxonomy_capped_two_sentiment_validation.py"),
        Path("scripts/audit_taxonomy_matched_one_vs_two_stage_validation.py"),
        Path("scripts/build_taxonomy_scientific_freeze_manifest.py"),
        Path("scripts/build_taxonomy_scientific_freeze_results_markdown.py"),
        Path("tests/test_taxonomy_scientific_freeze.py"),
        Path("docs/experiments/taxonomy_capped_two_sentiment_validation_v2_results.md"),
        Path("docs/experiments/taxonomy_matched_one_vs_two_stage_validation_v1_results.md"),
        Path("docs/experiments/taxonomy_rich_description_validation_confirmation_v1_results.md"),
        Path("docs/thesis_figure_data/taxonomy_matched_one_vs_two_stage_validation_v1.csv"),
    ]
    inputs = {
        str(relative): _sha256(PROJECT_ROOT / relative)
        for relative in registered_inputs
    }

    output_paths = sorted(
        (PROJECT_ROOT / "docs/thesis_figure_data/taxonomy_scientific_freeze_v1").glob("*")
    ) + sorted(
        (PROJECT_ROOT / "thesis/figures/taxonomy_scientific_freeze_v1").glob("*")
    )
    outputs = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in output_paths
        if path.is_file()
    }
    if not outputs:
        raise ValueError("Scientific-freeze tables and figures are missing.")

    formal = _read(
        PROJECT_ROOT
        / "docs/experiments/taxonomy_scientific_freeze_formal_evidence_audit_20260822.json"
    )
    precloud = _read(
        PROJECT_ROOT / "configs/experiments/taxonomy_two_stage_precloud_v2.json"
    )
    qwen_interface = _read(
        PROJECT_ROOT
        / "docs/experiments/taxonomy_qwen_matched_interface_audit_20260822.json"
    )
    package_names = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "torch",
        "transformers",
    )
    package_versions = {}
    for package_name in package_names:
        try:
            package_versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package_name] = "not-installed-in-analysis-environment"
    manifest = {
        "schema_version": "taxonomy_scientific_freeze_manifest_v1",
        "status": "pass",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_boundary": {
            "protocol_id": "taxonomy_two_stage_formal_v2",
            "formal_execution_commit": "aa84212976a652d62cfca31ed8bf0516a216c485",
            "analysis_branch_head_before_freeze_commit": _git("rev-parse", "HEAD"),
            "source_control_clean_before_manifest": True,
            "allowed_splits": ["train", "validation"],
            "include_official_test": False,
            "official_test_opened": False,
            "test_contract_count": 0,
            "bootstrap": {
                "unit": "row_uid review cluster",
                "draws": 20000,
                "seed": 13,
                "synchronised_across": "methods, folds and N/D conditions",
            },
            "primary_estimand": "aspect-balanced mean of 12 held-out fold pair micro-F1 values",
            "sensitivity_estimand": "pooled held-out pair micro-F1",
            "neural_training_seeds": 1,
            "data_and_candidate_hashes": {
                "train_csv_sha256": precloud["exact_input_hashes"]["train_csv_sha256"],
                "validation_csv_sha256": precloud["exact_input_hashes"]["validation_csv_sha256"],
                "minimal_description_file_sha256": precloud["exact_input_hashes"]["minimal_description_file_sha256"],
            },
            "decoder": precloud["architecture"],
            "qwen_matched_interface": qwen_interface["matched_interface"],
        },
        "qlora_seen_invariant_repair": formal["qlora_seen_invariant_repairs"],
        "audit_sha256": audited,
        "registered_input_sha256": inputs,
        "reporting_output_sha256": outputs,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "analysis_package_versions": package_versions,
            "formal_compute": {
                "provider": "Runpod Secure Cloud",
                "gpu": "NVIDIA GeForce RTX 4090",
                "cuda": "12.8",
                "dependency_lock": "requirements-taxonomy-cloud.txt",
                "worker_count": 3,
            },
        },
        "regeneration_commands": [
            "python scripts/build_taxonomy_scientific_freeze_formal_evidence.py",
            "python scripts/audit_taxonomy_scientific_freeze_local_replay.py",
            "python scripts/analyse_taxonomy_scientific_freeze_v1.py --bootstrap-draws 20000 --bootstrap-seed 13",
            "python scripts/build_taxonomy_scientific_freeze_reporting.py",
            "python scripts/audit_taxonomy_qwen_matched_interface.py",
            "python scripts/audit_taxonomy_few_shot_contract.py",
            "python scripts/build_taxonomy_scientific_freeze_manifest.py",
            "python scripts/build_taxonomy_scientific_freeze_results_markdown.py",
        ],
        "explicit_exclusions": [
            "official test rows and labels",
            "multi-seed neural reruns",
            "new expensive Level 3 runs",
            "post-result rich-card variants other than selected R2_positive_concat confirmation",
            "raw score shards, checkpoints, private data, credentials, and SSH keys from Git",
        ],
        "claim_limits": [
            "D-N is matched within method/fold but includes train-inference format match for trainable methods",
            "cross-method differences are descriptive systems comparisons",
            "QLoRA versus frozen Qwen is not a causal LoRA-only effect",
            "L3 is post-hoc targeted appendix robustness with no inferential p-values",
            "L4 groups are non-exchangeable and reported separately",
            "review-bootstrap intervals are conditional on observed trained realisations and fixed taxonomy",
        ],
        "future_official_test_release_rule": {
            "status": "blocked_not_executed",
            "authorisation": "requires a later explicit user approval after this validation freeze",
            "execution": "one predeclared pass only",
            "post_test_tuning": "forbidden",
            "failure_handling": "preserve evidence and report; do not retune from test outcomes",
        },
    }
    _write(args.output.resolve(), manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "audits": len(audited),
                "inputs": len(inputs),
                "outputs": len(outputs),
                "test_contract_count": 0,
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/taxonomy_scientific_freeze_manifest_20260822.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
