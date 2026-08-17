"""Fail-closed exact-reuse auditing for the two-stage taxonomy mainline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from msc_project.llm.qwen_two_stage_classifier import qwen_two_stage_contract_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verify_artifact_manifest(root: Path, manifest: Mapping[str, object]) -> int:
    verified = 0
    for relative, expected in sorted(manifest.items()):
        relative_path = Path(str(relative))
        if any(part.casefold().startswith("test") for part in relative_path.parts):
            raise ValueError(f"Reuse manifest names a forbidden test artifact: {relative}")
        path = root / relative_path
        observed = file_sha256(path)
        if observed != str(expected).casefold():
            raise ValueError(
                f"Artifact hash mismatch for {relative}: expected={expected}, observed={observed}."
            )
        verified += 1
    return verified


def audit_precloud_exact_reuse(
    *,
    project_root: Path,
    data_dir: Path,
    config_path: Path,
) -> dict[str, object]:
    """Audit only train/validation inputs and completed validation artifacts."""

    config = _json_object(config_path)
    if config.get("protocol_id") != "taxonomy_two_stage_precloud_v2":
        raise ValueError("Unexpected pre-cloud protocol ID.")
    sealed = config.get("sealed_data_contract")
    if not isinstance(sealed, Mapping):
        raise ValueError("Pre-cloud protocol lacks a sealed data contract.")
    if sealed.get("include_official_test") is not False:
        raise ValueError("Official test must remain disabled.")
    expected = config.get("exact_input_hashes")
    if not isinstance(expected, Mapping):
        raise ValueError("Pre-cloud protocol lacks exact input hashes.")

    inputs = {
        "train.csv": data_dir / "train.csv",
        "validation.csv": data_dir / "validation.csv",
        "minimal_description": project_root
        / "configs/experiments/fabsa_aspect_descriptions_minimal_v2.json",
        "rich_guidance": project_root
        / "configs/experiments/fabsa_aspect_rich_taxonomy_guidance_v1.json",
    }
    expected_keys = {
        "train.csv": "train_csv_sha256",
        "validation.csv": "validation_csv_sha256",
        "minimal_description": "minimal_description_file_sha256",
        "rich_guidance": "rich_guidance_file_sha256",
    }
    input_hashes: dict[str, str] = {}
    for name, path in inputs.items():
        observed = file_sha256(path)
        wanted = str(expected[expected_keys[name]]).casefold()
        if observed != wanted:
            raise ValueError(
                f"Exact input hash mismatch for {name}: expected={wanted}, observed={observed}."
            )
        input_hashes[name] = observed

    two_stage_root = (
        project_root / "outputs/experimental/taxonomy_two_stage_validation_v1"
    )
    two_stage_audit = _json_object(two_stage_root / "artifact_audit.json")
    if int(two_stage_audit.get("failure_count", -1)) != 0:
        raise ValueError("Completed two-stage audit contains failures.")
    if int(two_stage_audit.get("official_test_artifact_count", -1)) != 0:
        raise ValueError("Completed two-stage audit contains an official-test artifact.")
    manifest = two_stage_audit.get("artifact_sha256")
    if not isinstance(manifest, Mapping):
        raise ValueError("Completed two-stage audit lacks artifact hashes.")
    two_stage_verified = _verify_artifact_manifest(two_stage_root, manifest)

    qwen = two_stage_audit.get("qwen_cache")
    if not isinstance(qwen, Mapping):
        raise ValueError("Completed two-stage audit lacks the Qwen cache ledger.")
    cache_path = (
        two_stage_root
        / "study_c/frozen_qwen_candidate_pair/raw_cache/prompt_scores.jsonl"
    )
    cache_hash = file_sha256(cache_path)
    required_cache_hash = str(
        expected["completed_l2_qwen_zero_shot_cache_sha256"]
    ).casefold()
    if cache_hash != required_cache_hash or cache_hash != str(
        qwen.get("file_sha256")
    ).casefold():
        raise ValueError("Completed Frozen-Qwen cache hash does not match both ledgers.")
    contract = qwen_two_stage_contract_sha256(max_length=384)
    required_contract = str(
        expected["completed_l2_qwen_zero_shot_prompt_contract_sha256"]
    ).casefold()
    if contract != required_contract or contract != str(
        qwen.get("contract_sha256")
    ).casefold():
        raise ValueError("Completed Frozen-Qwen prompt contract is incompatible.")
    required_rows = int(expected["completed_l2_qwen_zero_shot_cache_rows"])
    if int(qwen.get("rows", -1)) != required_rows:
        raise ValueError("Completed Frozen-Qwen cache row count is incompatible.")

    matched_root = (
        project_root
        / "outputs/experimental/taxonomy_matched_one_vs_two_stage_validation_v1"
    )
    matched_audit = _json_object(matched_root / "audit.json")
    for key in (
        "failure_count",
        "non_finite_value_count",
        "official_test_artifact_count",
        "resume_conflict_count",
    ):
        if int(matched_audit.get(key, -1)) != 0:
            raise ValueError(f"Matched L2 audit has non-zero {key}.")
    matched_manifest = matched_audit.get("artifact_sha256")
    if not isinstance(matched_manifest, Mapping):
        raise ValueError("Matched L2 audit lacks artifact hashes.")
    matched_verified = _verify_artifact_manifest(matched_root, matched_manifest)

    methods = matched_audit.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("Matched L2 audit lacks method completeness.")
    for method in ("strict_train_only_tfidf", "e5_base_v2"):
        payload = methods.get(method)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Matched L2 audit lacks {method}.")
        if (
            int(payload.get("completed_folds", -1)) != 12
            or int(payload.get("failed_folds", -1)) != 0
            or int(payload.get("test_contract_count", -1)) != 0
            or payload.get("matched_representation") is not True
            or payload.get("matched_validation_rows") is not True
            or int(payload.get("maximum_primary_sentiments_per_aspect", -1)) != 2
        ):
            raise ValueError(f"Matched L2 method contract failed for {method}.")

    return {
        "schema_version": "taxonomy_precloud_exact_reuse_audit_v1",
        "protocol_id": config["protocol_id"],
        "status": "pass",
        "official_test_files_opened": 0,
        "input_hashes": input_hashes,
        "reuse_exact": {
            "matched_l2_tfidf": True,
            "matched_l2_e5": True,
            "l2_frozen_qwen_zero_shot_cache": True,
        },
        "control_only": [
            "historical independent-pair results",
            "historical DistilBERT pair checkpoints",
            "historical QLoRA pair adapters",
        ],
        "verified_artifacts": {
            "two_stage": two_stage_verified,
            "matched_l2": matched_verified,
        },
        "qwen_cache": {
            "sha256": cache_hash,
            "prompt_contract_sha256": contract,
            "rows": required_rows,
        },
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "test_contract_count": 0,
    }
