"""Strict artifact audit for the matched one-stage versus two-stage study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "taxonomy_matched_one_vs_two_stage_validation_v1"
METHODS = ("strict_train_only_tfidf", "e5_base_v2")
DECODERS = {
    "one_stage_independent_pairs",
    "two_stage_argmax",
    "two_stage_capped_two",
}
FOLD_IDS = {f"l2-a{index:02d}" for index in range(1, 13)}
METRICS = (
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "aspect_micro_f1",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _assert_hash_tree(value: Any, location: str) -> None:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"Empty hash mapping at {location}.")
        for key, item in value.items():
            _assert_hash_tree(item, f"{location}.{key}")
        return
    if value == "frozen_no_task_specific_training":
        return
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 evidence at {location}: {value!r}.")


def audit(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
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

    artifacts: list[Path] = []
    method_audits: dict[str, Any] = {}
    protocol_hash: str | None = None
    validation_hash: str | None = None
    for method in METHODS:
        method_root = output_root / method
        fold_paths = sorted((method_root / "folds").glob("l2-a*.json"))
        if {path.stem for path in fold_paths} != FOLD_IDS:
            raise ValueError(f"Incomplete fold coverage for {method}.")
        folds: list[dict[str, Any]] = []
        for path in fold_paths:
            fold = _read_json(path)
            _assert_finite(fold, f"{method}.{path.stem}")
            if (
                fold.get("schema_version")
                != "taxonomy_matched_one_vs_two_stage_fold_v1"
                or fold.get("protocol_id") != PROTOCOL_ID
                or fold.get("method_id") != method
                or fold.get("fold_id") != path.stem
                or fold.get("representation") != "name_and_description"
                or fold.get("selection_partition") != "seen_validation_only"
                or fold.get("evaluation_partition")
                != "heldout_validation_only"
                or int(fold.get("test_contract_count", -1)) != 0
            ):
                raise ValueError(f"Invalid fold contract: {path}")
            current_protocol_hash = str(fold.get("protocol_sha256"))
            if not SHA256.fullmatch(current_protocol_hash):
                raise ValueError(f"Invalid protocol hash: {path}")
            if protocol_hash is None:
                protocol_hash = current_protocol_hash
            elif protocol_hash != current_protocol_hash:
                raise ValueError("Protocol hash differs across fold artifacts.")

            hashes = fold.get("hashes")
            if not isinstance(hashes, dict):
                raise ValueError(f"Missing fold hashes: {path}")
            _assert_hash_tree(hashes, f"{method}.{path.stem}.hashes")
            current_validation_hash = str(hashes["validation_rows_sha256"])
            if validation_hash is None:
                validation_hash = current_validation_hash
            elif validation_hash != current_validation_hash:
                raise ValueError(
                    "Matched folds or methods used different validation rows."
                )

            checks = fold.get("matched_checks")
            if not isinstance(checks, dict) or not checks or not all(
                value is True for value in checks.values()
            ):
                raise ValueError(f"A matched comparison check failed: {path}")
            decoders = fold.get("decoders")
            if not isinstance(decoders, dict) or set(decoders) != DECODERS:
                raise ValueError(f"Decoder coverage mismatch: {path}")
            for decoder, payload in decoders.items():
                if payload["selection"]["selection_partition"] != "seen_validation":
                    raise ValueError(f"Non-seen threshold selection: {path}")
                for metric in METRICS:
                    value = float(payload["heldout"]["metrics"][metric])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(f"Out-of-range metric {metric}: {path}")
            capped = decoders["two_stage_capped_two"]
            argmax = decoders["two_stage_argmax"]
            if capped["selection"].get("aspect_threshold_frozen") is not True:
                raise ValueError(f"Capped-two aspect threshold is not frozen: {path}")
            _assert_close(
                float(capped["selection"]["aspect_threshold"]),
                float(argmax["selection"]["aspect_threshold"]),
                f"{method}.{path.stem}.aspect_threshold",
            )
            capped_cardinality = capped["sentiment_cardinality"]
            argmax_cardinality = argmax["sentiment_cardinality"]
            if int(capped_cardinality["maximum"]) > 2:
                raise ValueError(f"Capped-two emitted a third sentiment: {path}")
            if int(argmax_cardinality["maximum"]) > 1:
                raise ValueError(f"Argmax emitted multiple sentiments: {path}")
            folds.append(fold)
            artifacts.append(path)

        summary_path = method_root / "summary.json"
        summary = _read_json(summary_path)
        _assert_finite(summary, f"{method}.summary")
        if (
            summary.get("protocol_id") != PROTOCOL_ID
            or summary.get("protocol_sha256") != protocol_hash
            or int(summary.get("completed_folds", -1)) != 12
            or int(summary.get("failed_folds", -1)) != 0
            or int(summary.get("test_contract_count", -1)) != 0
            or summary.get("capped_two_is_default_sentiment_decoder") is not True
        ):
            raise ValueError(f"Invalid completion summary: {summary_path}")
        aggregate = summary.get("aggregate")
        if (
            not isinstance(aggregate, list)
            or {row.get("decoder") for row in aggregate} != DECODERS
            or any(int(row.get("folds", -1)) != 12 for row in aggregate)
        ):
            raise ValueError(f"Invalid aggregate coverage: {summary_path}")
        for row in aggregate:
            decoder = str(row["decoder"])
            for metric in METRICS:
                expected = sum(
                    float(fold["decoders"][decoder]["heldout"]["metrics"][metric])
                    for fold in folds
                ) / 12
                _assert_close(
                    float(row[f"{metric}_mean"]),
                    expected,
                    f"{method}.summary.{decoder}.{metric}",
                )

        csv_path = method_root / "per_fold.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        if len(csv_rows) != 36:
            raise ValueError(f"Expected 36 per-fold decoder rows: {csv_path}")
        artifacts.extend([summary_path, csv_path])
        method_audits[method] = {
            "completed_folds": 12,
            "failed_folds": 0,
            "test_contract_count": 0,
            "decoder_rows": 36,
            "matched_validation_rows": True,
            "matched_representation": True,
            "maximum_primary_sentiments_per_aspect": 2,
        }

    hashes = {
        str(path.relative_to(output_root)): _sha256(path)
        for path in sorted(artifacts)
    }
    return {
        "schema_version": "taxonomy_matched_one_vs_two_stage_artifact_audit_v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_hash,
        "failure_count": 0,
        "official_test_artifact_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
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
        / PROTOCOL_ID,
    )
    parser.add_argument("--write", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = audit(args.output_root)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
