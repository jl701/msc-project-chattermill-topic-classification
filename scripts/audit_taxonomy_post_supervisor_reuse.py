from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.experiments.taxonomy_post_supervisor import post_supervisor_l2_folds
from msc_project.experiments.taxonomy_protocol import registered_folds
from msc_project.experiments.taxonomy_qwen_zero_shot_cache import (
    ReadOnlyQwenTwoStageCache,
)
from msc_project.experiments.taxonomy_resources import load_description_bundle


POST_PLAN = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_post_supervisor_execution_plan_v1.json"
)
L2_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_level2_local_ndr_v1.json"
)
OLD_PRECLOUD = (
    PROJECT_ROOT / "outputs" / "experimental" / "taxonomy_two_stage_precloud_v2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _all_named_values(value: Any, name: str) -> Iterable[Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == name:
                yield item
            yield from _all_named_values(item, name)
    elif isinstance(value, list):
        for item in value:
            yield from _all_named_values(item, name)


def _fold_artifact_is_safe(path: Path) -> tuple[bool, list[str]]:
    reasons = []
    if not path.is_file():
        return False, ["missing artifact"]
    value = json.loads(path.read_text(encoding="utf-8"))
    test_counts = [int(item) for item in _all_named_values(value, "test_contract_count")]
    failure_counts = [int(item) for item in _all_named_values(value, "failure_count")]
    non_finite_counts = [
        int(item) for item in _all_named_values(value, "non_finite_value_count")
    ]
    if not test_counts or any(test_counts):
        reasons.append("test-contract evidence is absent or non-zero")
    if any(failure_counts):
        reasons.append("failure count is non-zero")
    if any(non_finite_counts):
        reasons.append("non-finite count is non-zero")
    return not reasons, reasons


def _artifact_set_status(paths: list[Path]) -> dict[str, object]:
    results = [_fold_artifact_is_safe(path) for path in paths]
    return {
        "expected": len(paths),
        "present": sum(path.is_file() for path in paths),
        "safe": sum(result[0] for result in results),
        "all_safe": all(result[0] for result in results),
        "reasons": sorted({reason for _, reasons in results for reason in reasons}),
        "sha256": {
            path.name: _sha256(path)
            for path in paths
            if path.is_file()
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    post_plan = json.loads(POST_PLAN.read_text(encoding="utf-8"))
    l2_config = json.loads(L2_CONFIG.read_text(encoding="utf-8"))
    if (
        post_plan["data_contract"].get("include_official_test") is not False
        or l2_config["data_contract"].get("include_official_test") is not False
    ):
        raise ValueError("Post-supervisor reuse audit requires sealed official test.")
    data_dir = args.data_dir.resolve()
    data_hashes = {
        split: _sha256(data_dir / f"{split}.csv")
        for split in ("train", "validation")
    }
    expected_hashes = {
        "train": str(post_plan["reuse_policy"] and l2_config["data_contract"]["train_csv_sha256"]),
        "validation": str(l2_config["data_contract"]["validation_csv_sha256"]),
    }
    if data_hashes != expected_hashes:
        raise ValueError("Current train/validation files do not match the frozen hashes.")
    frame = load_official_fabsa_splits(data_dir, ("train", "validation"))
    rows = {
        split: int(frame[frame["original_split"].eq(split)]["row_uid"].nunique())
        for split in ("train", "validation")
    }
    resource = load_description_bundle(require_approved=True)

    cache_config = l2_config["frozen_qwen_cache"]
    cache_path = PROJECT_ROOT / str(cache_config["path"])
    qwen_cache = ReadOnlyQwenTwoStageCache(
        cache_path,
        expected_file_sha256=str(cache_config["sha256"]),
    )
    cache_summary = qwen_cache.summary()
    if cache_summary["contract_sha256"] != cache_config["expected_contract_sha256"]:
        raise ValueError("Frozen-Qwen prompt contract hash changed.")

    l4_status = {}
    l4_folds = registered_folds("L4")
    for method in (
        "strict_train_only_tfidf",
        "e5_base_v2",
        "frozen_qwen_candidate_pair",
    ):
        paths = [
            OLD_PRECLOUD
            / "validation"
            / method
            / "l4"
            / "folds"
            / f"{fold.fold_id}.json"
            for fold in l4_folds
        ]
        l4_status[method] = _artifact_set_status(paths)

    optional_pairs = {
        "a01-a02": "l3-a01-a02",
        "a03-a04": "l3-a03-a04",
        "a05-a06": "l3-a05-a06",
        "a07-a08": "l3-a07-a08",
        "a09-a10": "l3-a09-a10",
        "a11-a12": "l3-a11-a12",
    }
    l3_status = {}
    for method in (
        "strict_train_only_tfidf",
        "e5_base_v2",
        "frozen_qwen_candidate_pair",
    ):
        paths = [
            OLD_PRECLOUD
            / "validation"
            / method
            / "l3"
            / "folds"
            / f"{fold_id}.json"
            for fold_id in optional_pairs.values()
        ]
        l3_status[method] = _artifact_set_status(paths)

    entries = [
        {
            "scope": "L1 closed reference",
            "method": "word_char_tfidf_ovr_logreg",
            "decision": "new_run_required",
            "reason": "new zero-unseen definition has no exact prior result contract",
        },
        {
            "scope": "L2 N/D/R",
            "method": "strict_train_only_tfidf",
            "decision": "new_scoring_bundle_required",
            "reason": "old D evidence lacks one unified N/D/R full-candidate score contract",
        },
        {
            "scope": "L2 N/D/R",
            "method": "e5_base_v2",
            "decision": "new_scoring_bundle_required",
            "reason": "old D evidence lacks one unified N/D/R full-candidate score contract",
        },
        {
            "scope": "L2 N/D/R",
            "method": "frozen_qwen_candidate_pair",
            "decision": "exact_raw_score_cache_redecode",
            "reason": "read-only cache file and prompt contract hashes are exact; aggregate artifacts will be regenerated",
            "cache_sha256": cache_summary["file_sha256"],
            "prompt_contract_sha256": cache_summary["contract_sha256"],
        },
        {
            "scope": "L2 N/D/R",
            "method": "description_to_classifier_weight_transfer_v1",
            "decision": "new_run_required",
            "reason": "pre-registered after the existing validation campaign",
        },
        {
            "scope": "L2 N/D/R",
            "method": "distilbert_review_candidate_true_two_stage",
            "decision": "cloud_run_required",
            "reason": "old independent-pair checkpoints are control-only",
        },
        {
            "scope": "L2 N/D/R",
            "method": "frozen_qwen_few_shot_true_two_stage",
            "decision": "cloud_run_required",
            "reason": "only bounded smoke evidence exists",
        },
        {
            "scope": "L2 N/D/R",
            "method": "qwen_qlora_true_two_stage",
            "decision": "cloud_run_required",
            "reason": "old independent-pair adapters are control-only",
        },
    ]
    for method, status in l4_status.items():
        entries.append(
            {
                "scope": "L4 D three parent groups",
                "method": method,
                "decision": (
                    "exact_result_reuse_approved"
                    if status["all_safe"]
                    else "rerun_required"
                ),
                "artifact_audit": status,
            }
        )
    for method in (
        "distilbert_review_candidate_true_two_stage",
        "frozen_qwen_few_shot_true_two_stage",
        "qwen_qlora_true_two_stage",
    ):
        entries.append(
            {
                "scope": "L4 D three parent groups",
                "method": method,
                "decision": "cloud_run_required",
                "reason": "no exact full validation artifact under the true two-stage contract",
            }
        )

    result: dict[str, object] = {
        "schema_version": "taxonomy_post_supervisor_exact_reuse_matrix_v1",
        "plan_id": str(post_plan["plan_id"]),
        "post_plan_sha256": _sha256(POST_PLAN),
        "level2_local_protocol_sha256": _sha256(L2_CONFIG),
        "allowed_splits": ["train", "validation"],
        "include_official_test": False,
        "test_contract_count": 0,
        "data_hashes": data_hashes,
        "rows": rows,
        "description_bundle": {
            "version": resource["version"],
            "content_sha256": resource["content_sha256"],
            "minimal_resource_sha256": resource["minimal_resource_sha256"],
            "rich_resource_sha256": resource["rich_resource_sha256"],
        },
        "level2_fold_count": len(post_supervisor_l2_folds()),
        "entries": entries,
        "optional_level3_exact_candidate_audit": l3_status,
        "cloud_infrastructure_reuse": {
            "decision": "implementation_reusable_but_final_commit_gates_must_rerun",
            "old_26_scope_launch_authorised": False,
        },
        "failure_count": 0,
    }
    _write_json(args.write.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--write",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_post_supervisor_local_v1"
        / "reuse_matrix.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
