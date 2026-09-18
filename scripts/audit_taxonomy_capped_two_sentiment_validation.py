"""Strict artifact audit for capped-two sentiment validation v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("strict_train_only_tfidf", "e5_base_v2")
DECODERS = {"argmax", "capped_two_threshold"}
FOLD_IDS = {f"l2-a{index:02d}" for index in range(1, 13)}
METRICS = ("pair_micro_f1", "pair_micro_precision", "pair_micro_recall")


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


def _assert_close(observed: float, expected: float, location: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Metric mismatch at {location}: observed={observed} expected={expected}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(output_root: Path, parent_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    parent_root = parent_root.resolve()
    forbidden = [
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file()
        and (
            "official-test" in path.name.casefold()
            or "official_test" in path.name.casefold()
        )
    ]
    if forbidden:
        raise ValueError(f"Official-test artifacts are forbidden: {forbidden}")

    artifact_paths: list[Path] = []
    method_audits: dict[str, Any] = {}
    for method in METHODS:
        method_root = output_root / method
        fold_paths = sorted((method_root / "folds").glob("l2-a*.json"))
        if {path.stem for path in fold_paths} != FOLD_IDS:
            raise ValueError(f"Incomplete fold coverage for {method}.")
        fold_results: list[dict[str, Any]] = []
        for fold_path in fold_paths:
            fold = _read_json(fold_path)
            _assert_finite(fold, f"{method}.{fold_path.stem}")
            if (
                fold.get("schema_version")
                != "taxonomy_capped_two_sentiment_fold_v2"
                or fold.get("protocol_id")
                != "taxonomy_capped_two_sentiment_validation_v2"
                or fold.get("method_id") != method
                or fold.get("fold_id") != fold_path.stem
                or fold.get("representation") != "name_and_description"
                or int(fold.get("test_contract_count", -1)) != 0
            ):
                raise ValueError(f"Invalid fold contract: {fold_path}.")
            decoders = fold.get("decoders")
            if not isinstance(decoders, dict) or set(decoders) != DECODERS:
                raise ValueError(f"Decoder coverage mismatch: {fold_path}.")
            control = decoders["argmax"]
            capped = decoders["capped_two_threshold"]
            if (
                control["selection"]["selection_partition"] != "seen_validation"
                or capped["selection"]["selection_partition"] != "seen_validation"
                or capped["selection"].get("aspect_threshold_frozen") is not True
            ):
                raise ValueError(f"Invalid selection contract: {fold_path}.")
            _assert_close(
                float(capped["selection"]["aspect_threshold"]),
                float(control["selection"]["aspect_threshold"]),
                f"{method}.{fold_path.stem}.aspect_threshold",
            )
            _assert_close(
                float(control["heldout"]["sentiments_per_selected_aspect"]),
                1.0,
                f"{method}.{fold_path.stem}.argmax.cardinality",
            )
            capped_cardinality = float(
                capped["heldout"]["sentiments_per_selected_aspect"]
            )
            if not 1.0 <= capped_cardinality <= 2.0:
                raise ValueError(f"Capped-two cardinality violation: {fold_path}.")

            parent_path = parent_root / "study_c" / method / "folds" / fold_path.name
            parent = _read_json(parent_path)
            parent_metrics = parent["variants"]["name_and_description"]["partitions"][
                "heldout"
            ]["metrics"]
            for metric in METRICS:
                _assert_close(
                    float(control["heldout"]["metrics"][metric]),
                    float(parent_metrics[metric]),
                    f"{method}.{fold_path.stem}.parent.{metric}",
                )
            fold_results.append(fold)
            artifact_paths.append(fold_path)

        summary_path = method_root / "summary.json"
        summary = _read_json(summary_path)
        _assert_finite(summary, f"{method}.summary")
        if (
            int(summary.get("completed_folds", -1)) != 12
            or int(summary.get("failed_folds", -1)) != 0
            or int(summary.get("test_contract_count", -1)) != 0
            or summary.get("aspect_threshold_frozen") is not True
            or int(summary.get("maximum_sentiments_per_aspect", -1)) != 2
        ):
            raise ValueError(f"Invalid completion summary: {summary_path}.")
        aggregate = summary.get("aggregate")
        if (
            not isinstance(aggregate, list)
            or {row.get("decoder") for row in aggregate} != DECODERS
            or any(int(row.get("folds", -1)) != 12 for row in aggregate)
        ):
            raise ValueError(f"Invalid summary coverage: {summary_path}.")
        for aggregate_row in aggregate:
            decoder = str(aggregate_row["decoder"])
            for metric in METRICS:
                expected = sum(
                    float(fold["decoders"][decoder]["heldout"]["metrics"][metric])
                    for fold in fold_results
                ) / 12
                _assert_close(
                    float(aggregate_row[f"{metric}_mean"]),
                    expected,
                    f"{method}.summary.{decoder}.{metric}",
                )

        csv_path = method_root / "per_fold.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        if len(csv_rows) != 24:
            raise ValueError(f"Expected 24 per-fold decoder rows: {csv_path}.")
        artifact_paths.extend([summary_path, csv_path])
        method_audits[method] = {
            "completed_folds": 12,
            "failed_folds": 0,
            "test_contract_count": 0,
            "parent_argmax_reproduced": True,
            "aspect_threshold_frozen": True,
            "maximum_sentiments_per_aspect": 2,
            "decoder_rows": 24,
        }

    hashes = {
        str(path.relative_to(output_root)): _sha256(path)
        for path in sorted(artifact_paths)
    }
    return {
        "schema_version": "taxonomy_capped_two_sentiment_artifact_audit_v2",
        "failure_count": 0,
        "official_test_artifact_count": 0,
        "non_finite_value_count": 0,
        "methods": method_audits,
        "artifact_count": len(hashes),
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
        / "taxonomy_capped_two_sentiment_validation_v2",
    )
    parser.add_argument(
        "--parent-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_two_stage_validation_v1",
    )
    parser.add_argument("--write", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = audit(args.output_root, args.parent_root)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
