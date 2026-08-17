"""Strict artifact audit for taxonomy two-stage validation v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "frozen_qwen_candidate_pair",
)
VARIANTS = {"name_and_description", "name_only", "description_only"}
PARTITIONS = {"overall", "seen", "heldout"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _assert_finite(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value at {location}.")


def audit(output_root: Path, *, require_complete: bool) -> dict[str, Any]:
    output_root = output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    forbidden_paths = [
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and "official-test" in str(path).casefold()
    ]
    if forbidden_paths:
        raise ValueError(f"Official-test artifacts are forbidden: {forbidden_paths}")

    b_path = output_root / "study_b" / "summary.json"
    b = _read_json(b_path)
    if b.get("official_test_accessed") is not False:
        raise ValueError("Study B does not declare official_test_accessed=false.")
    aggregate_b = b.get("aggregate")
    if not isinstance(aggregate_b, list) or len(aggregate_b) != 5:
        raise ValueError("Study B must contain five method summaries.")
    if any(int(row.get("folds", -1)) != 12 for row in aggregate_b):
        raise ValueError("Every Study B method must cover twelve folds.")
    _assert_finite(b, "study_b")

    method_audits: dict[str, Any] = {}
    artifact_paths = [b_path]
    for method in METHODS:
        method_root = output_root / "study_c" / method
        fold_root = method_root / "folds"
        fold_paths = sorted(fold_root.glob("l2-a*.json")) if fold_root.is_dir() else []
        if require_complete and len(fold_paths) != 12:
            raise ValueError(f"{method} has {len(fold_paths)} completed folds, expected 12.")
        fold_ids: set[str] = set()
        for path in fold_paths:
            fold = _read_json(path)
            _assert_finite(fold, f"{method}.{path.stem}")
            if fold.get("method_id") != method or int(fold.get("test_contract_count", -1)) != 0:
                raise ValueError(f"Invalid method/test contract in {path}.")
            fold_id = str(fold.get("fold_id"))
            if fold_id in fold_ids:
                raise ValueError(f"Duplicate fold ID for {method}: {fold_id}")
            fold_ids.add(fold_id)
            variants = fold.get("variants")
            if not isinstance(variants, dict) or set(variants) != VARIANTS:
                raise ValueError(f"Incomplete representation variants in {path}.")
            thresholds = {float(value["threshold"]) for value in variants.values()}
            if len(thresholds) != 1:
                raise ValueError(f"Held-out representation changed seen threshold in {path}.")
            for variant, value in variants.items():
                partitions = value.get("partitions")
                if not isinstance(partitions, dict) or set(partitions) != PARTITIONS:
                    raise ValueError(f"Incomplete partitions for {variant} in {path}.")
        summary_path = method_root / "summary.json"
        if require_complete:
            summary = _read_json(summary_path)
            _assert_finite(summary, f"{method}.summary")
            if (
                int(summary.get("completed_folds", -1)) != 12
                or int(summary.get("failed_folds", -1)) != 0
                or int(summary.get("test_contract_count", -1)) != 0
            ):
                raise ValueError(f"Invalid completion summary for {method}.")
            aggregate = summary.get("aggregate")
            if not isinstance(aggregate, list) or len(aggregate) != 9:
                raise ValueError(f"Invalid aggregate coverage for {method}.")
            artifact_paths.append(summary_path)
        artifact_paths.extend(fold_paths)
        method_audits[method] = {
            "completed_folds": len(fold_paths),
            "fold_ids": sorted(fold_ids),
            "complete": len(fold_paths) == 12,
        }

    cache_path = (
        output_root
        / "study_c"
        / "frozen_qwen_candidate_pair"
        / "raw_cache"
        / "prompt_scores.jsonl"
    )
    qwen_cache: dict[str, Any] = {"exists": cache_path.is_file(), "rows": 0}
    if cache_path.is_file():
        keys: set[str] = set()
        modes: dict[str, int] = {"aspect": 0, "sentiment": 0}
        contract_hashes: set[str] = set()
        for line_number, line in enumerate(
            cache_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record["key"])
            mode = str(record["mode"])
            values = [float(value) for value in record["values"]]
            if key in keys:
                raise ValueError(f"Duplicate Qwen cache key at line {line_number}.")
            if mode not in modes or len(values) != (2 if mode == "aspect" else 3):
                raise ValueError(f"Invalid Qwen cache row at line {line_number}.")
            if not all(math.isfinite(value) for value in values) or not math.isclose(
                sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5
            ):
                raise ValueError(f"Invalid Qwen probabilities at line {line_number}.")
            keys.add(key)
            modes[mode] += 1
            contract_hashes.add(str(record["contract_sha256"]))
        if len(contract_hashes) != 1:
            raise ValueError("Qwen cache contains multiple prompt contracts.")
        qwen_cache = {
            "exists": True,
            "rows": len(keys),
            "modes": modes,
            "contract_sha256": next(iter(contract_hashes)),
            "file_sha256": _sha256(cache_path),
        }

    hashes = {
        str(path.relative_to(output_root)): _sha256(path)
        for path in sorted(set(artifact_paths))
        if path.is_file()
    }
    return {
        "schema_version": "taxonomy_two_stage_artifact_audit_v1",
        "failure_count": 0,
        "official_test_artifact_count": 0,
        "require_complete": require_complete,
        "study_b": {"complete": True, "methods": 5, "folds_per_method": 12},
        "study_c": method_audits,
        "qwen_cache": qwen_cache,
        "artifact_sha256": hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_two_stage_validation_v1",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = audit(args.output_root, require_complete=not args.allow_incomplete)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
