"""Fail-closed audit for all pre-cloud post-supervisor local results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))
LOCAL_METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "frozen_qwen_candidate_pair",
)
EXPECTED_QWEN_CACHE_SHA256 = (
    "08d276ac8915a929c3849dc92a54130df53a9835a8f0b36a0040ee7dbfad9db3"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {location}.")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{location}[{index}]")


def _assert_zero_ledger(value: Mapping[str, Any], location: str) -> None:
    for field in (
        "failure_count",
        "non_finite_value_count",
        "resume_conflict_count",
        "test_contract_count",
    ):
        if field in value and int(value[field]) != 0:
            raise ValueError(f"Non-zero {field} in {location}.")


def audit(root: Path, level1_root: Path) -> dict[str, object]:
    reuse_path = root / "reuse_matrix.json"
    reuse = _read(reuse_path)
    if reuse.get("allowed_splits") != ["train", "validation"]:
        raise ValueError("Reuse audit permits an unexpected split.")
    _assert_zero_ledger(reuse, "reuse matrix")
    _assert_finite(reuse, "reuse matrix")

    level1_path = level1_root / "result.json"
    level1 = _read(level1_path)
    _assert_zero_ledger(level1, "Level 1")
    _assert_finite(level1, "Level 1")
    selected = level1.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Level 1 has no selected validation configuration.")

    dcwt_path = root / "dcwt" / "summary.json"
    dcwt = _read(dcwt_path)
    _assert_zero_ledger(dcwt, "DCWT")
    _assert_finite(dcwt, "DCWT")
    if dcwt.get("completed_folds") != 12 or tuple(dcwt.get("folds", [])) != EXPECTED_FOLDS:
        raise ValueError("DCWT fold union is incomplete or reordered.")

    local: dict[str, object] = {}
    for method_id in LOCAL_METHODS:
        method_root = root / "level2_local_ndr" / method_id
        summary_path = method_root / "summary.json"
        summary = _read(summary_path)
        _assert_zero_ledger(summary, method_id)
        _assert_finite(summary, method_id)
        if summary.get("completed_folds") != 12 or tuple(summary.get("folds", [])) != EXPECTED_FOLDS:
            raise ValueError(f"{method_id} fold union is incomplete or reordered.")
        for fold_id in EXPECTED_FOLDS:
            fold = _read(method_root / "folds" / f"{fold_id}.json")
            _assert_zero_ledger(fold, f"{method_id}/{fold_id}")
            _assert_finite(fold, f"{method_id}/{fold_id}")
            conditions = fold.get("conditions")
            if not isinstance(conditions, Mapping) or set(conditions) != {"N", "D", "R"}:
                raise ValueError(f"{method_id}/{fold_id} lacks the exact N/D/R set.")
            score_hashes = [
                str(condition["score_sha256"])
                for condition in conditions.values()
                if isinstance(condition, Mapping)
            ]
            if len(score_hashes) != 3 or any(len(value) != 64 for value in score_hashes):
                raise ValueError(f"{method_id}/{fold_id} has an invalid condition score hash.")
        if method_id == "frozen_qwen_candidate_pair":
            cache = summary.get("qwen_cache")
            if not isinstance(cache, Mapping):
                raise ValueError("Frozen-Qwen exact cache audit is absent.")
            if (
                cache.get("mode") != "read_only_exact"
                or cache.get("file_sha256") != EXPECTED_QWEN_CACHE_SHA256
                or cache.get("cache_miss_count") != 0
                or cache.get("new_inference_count") != 0
            ):
                raise ValueError("Frozen-Qwen exact cache was not reused safely.")
        local[method_id] = {
            "summary_sha256": _sha256(summary_path),
            "seconds": float(summary["seconds"]),
            "condition_macro_means": summary["aggregate"]["condition_macro_means"],
            "paired_fold_contrasts": summary["aggregate"]["paired_fold_contrasts"],
        }

    return {
        "schema_version": "taxonomy_post_supervisor_local_completion_audit_v1",
        "status": "pass",
        "allowed_splits": ["train", "validation"],
        "level1": {
            "result_sha256": _sha256(level1_path),
            "selected_config_id": selected["config_id"],
            "pair_micro_f1": selected["pair_micro_f1"],
            "aspect_micro_f1": selected["aspect_micro_f1"],
        },
        "dcwt": {
            "summary_sha256": _sha256(dcwt_path),
            "seconds": float(dcwt["seconds"]),
            "generator_count": len(dcwt["generator_aggregates"]),
            "kernel_ridge": dcwt["generator_aggregates"]["kernel_ridge"],
        },
        "local_level2": local,
        "completed_fold_count_per_method": 12,
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "test_contract_count": 0,
        "official_test_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_post_supervisor_local_v1",
    )
    parser.add_argument(
        "--level1-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_level1_closed_reference_v1",
    )
    args = parser.parse_args()
    value = audit(args.root, args.level1_root)
    path = args.root / "local_completion_audit.json"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
