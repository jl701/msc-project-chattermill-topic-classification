from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_protocol import registered_folds  # noqa: E402


DEFAULT_ROOT = Path("outputs/experimental/taxonomy_two_stage_precloud_v2")
METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "frozen_qwen_candidate_pair",
)
LEVELS = ("L1", "L3", "L4")
EXPECTED_CONDITIONS = {
    "L1": ("N", "D"),
    "L3": ("NN", "DN", "ND", "DD", "RR"),
    "L4": ("D",),
}
AUGMENTED_QWEN_CACHE_SHA256 = (
    "08d276ac8915a929c3849dc92a54130df53a9835a8f0b36a0040ee7dbfad9db3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _non_finite_paths(value: Any, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(
                _non_finite_paths(child, f"{prefix}.{key}" if prefix else str(key))
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_non_finite_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        failures.append(prefix)
    return failures


def _audit_cache(path: Path) -> dict[str, object]:
    counts = {"aspect": 0, "sentiment": 0}
    keys: set[str] = set()
    contracts: set[str] = set()
    conflicts = 0
    values_by_key: dict[str, tuple[str, tuple[float, ...]]] = {}
    non_finite = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record["key"])
        mode = str(record["mode"])
        values = tuple(float(item) for item in record["values"])
        contracts.add(str(record["contract_sha256"]))
        if mode not in counts:
            raise ValueError(f"Unexpected cache mode: {mode!r}.")
        counts[mode] += 1
        if not np.isfinite(values).all():
            non_finite += 1
        current = (mode, values)
        if key in values_by_key and values_by_key[key] != current:
            conflicts += 1
        values_by_key[key] = current
        keys.add(key)
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "rows": sum(counts.values()),
        "unique_keys": len(keys),
        "modes": counts,
        "contracts": sorted(contracts),
        "conflicting_key_count": conflicts,
        "non_finite_row_count": non_finite,
    }


def audit(root: Path) -> dict[str, object]:
    failures: list[str] = []
    artifact_sha256: dict[str, str] = {}
    records: list[dict[str, object]] = []
    collapse_totals: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    expected_groups: set[tuple[str, str, str]] = set()
    test_contract_total = 0
    validation_root = root / "validation"
    for method in METHODS:
        for level in LEVELS:
            expected_folds = registered_folds(level)
            method_root = validation_root / method / level.lower()
            summary_path = method_root / "summary.json"
            if not summary_path.is_file():
                failures.append(f"missing_summary:{method}:{level}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            artifact_sha256[summary_path.relative_to(root).as_posix()] = _sha256(
                summary_path
            )
            if (
                int(summary.get("completed_folds", -1)) != len(expected_folds)
                or int(summary.get("expected_folds", -1)) != len(expected_folds)
                or int(summary.get("failed_folds", -1)) != 0
            ):
                failures.append(f"bad_summary_counts:{method}:{level}")
            test_contract_total += int(summary.get("test_contract_count", 0))
            for fold in expected_folds:
                path = method_root / "folds" / f"{fold.fold_id}.json"
                if not path.is_file():
                    failures.append(f"missing_fold:{method}:{level}:{fold.fold_id}")
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
                artifact_sha256[path.relative_to(root).as_posix()] = _sha256(path)
                if _non_finite_paths(value):
                    failures.append(f"non_finite:{method}:{level}:{fold.fold_id}")
                for key in (
                    "failure_count",
                    "non_finite_value_count",
                    "resume_conflict_count",
                    "test_contract_count",
                ):
                    if int(value.get(key, 0)) != 0:
                        failures.append(
                            f"nonzero_{key}:{method}:{level}:{fold.fold_id}"
                        )
                test_contract_total += int(value.get("test_contract_count", 0))
                if not math.isfinite(float(value.get("aspect_threshold", math.nan))):
                    failures.append(f"bad_aspect_threshold:{method}:{level}:{fold.fold_id}")
                if not math.isfinite(
                    float(value.get("second_sentiment_threshold", math.nan))
                ):
                    failures.append(f"bad_runner_threshold:{method}:{level}:{fold.fold_id}")
                conditions = value.get("conditions")
                if not isinstance(conditions, dict) or set(conditions) != set(
                    EXPECTED_CONDITIONS[level]
                ):
                    failures.append(f"bad_conditions:{method}:{level}:{fold.fold_id}")
                    continue
                top_seen_hash = str(value.get("seen_score_sha256", ""))
                for condition, condition_value in conditions.items():
                    if str(condition_value.get("seen_score_sha256", "")) != top_seen_hash:
                        failures.append(
                            f"seen_hash_mismatch:{method}:{level}:{fold.fold_id}:{condition}"
                        )
                    heldout = condition_value["partitions"]["heldout"]
                    primary = heldout["primary_capped_two"]
                    maximum = int(
                        heldout.get("maximum_sentiments_per_selected_aspect", 0)
                    )
                    if maximum > 2:
                        failures.append(
                            f"third_sentiment:{method}:{level}:{fold.fold_id}:{condition}"
                        )
                    group = (method, level, str(condition))
                    expected_groups.add(group)
                    predicted = int(primary["pair_predicted_label_count"])
                    collapse_totals[group] += predicted
                    records.append(
                        {
                            "method_id": method,
                            "level": level,
                            "fold_id": fold.fold_id,
                            "condition": str(condition),
                            "heldout_pair_micro_f1": float(primary["pair_micro_f1"]),
                            "heldout_aspect_micro_f1": float(primary["aspect_micro_f1"]),
                            "heldout_pair_predictions": predicted,
                            "heldout_pair_gold": int(primary["pair_gold_label_count"]),
                            "maximum_sentiments_per_selected_aspect": maximum,
                        }
                    )
    for group in expected_groups:
        if collapse_totals[group] == 0:
            failures.append("prediction_collapse:" + ":".join(group))

    required_audits = {
        "reuse_audit.json": "pass",
        "training_manifest_audit.json": "pass",
        "cloud_gate_manifest.json": "pass",
        "smoke/distilbert_true_two_stage.json": "pass",
        "smoke/frozen_qwen_few_shot.json": "pass",
        "smoke/qwen_true_two_stage_qlora.json": "pass",
    }
    prerequisite_status: dict[str, object] = {}
    for relative, expected_status in required_audits.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing_prerequisite:{relative}")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        artifact_sha256[relative] = _sha256(path)
        observed = str(value.get("status"))
        prerequisite_status[relative] = observed
        if observed != expected_status or _non_finite_paths(value):
            failures.append(f"bad_prerequisite:{relative}")
        test_contract_total += int(value.get("test_contract_count", 0))

    cache_path = (
        root
        / "qwen_zero_shot_rich_seed/study_c/frozen_qwen_candidate_pair/"
        "raw_cache/prompt_scores.jsonl"
    )
    if not cache_path.is_file():
        failures.append("missing_augmented_qwen_cache")
        cache = {}
    else:
        cache = _audit_cache(cache_path)
        if (
            cache["sha256"] != AUGMENTED_QWEN_CACHE_SHA256
            or cache["rows"] != 100032
            or cache["unique_keys"] != 100032
            or cache["modes"] != {"aspect": 50016, "sentiment": 50016}
            or cache["conflicting_key_count"] != 0
            or cache["non_finite_row_count"] != 0
        ):
            failures.append("bad_augmented_qwen_cache")
        artifact_sha256[
            cache_path.relative_to(root).as_posix()
        ] = str(cache["sha256"])
    if test_contract_total != 0:
        failures.append(f"nonzero_test_contract_total:{test_contract_total}")

    aggregates: list[dict[str, object]] = []
    grouped: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["method_id"]),
                str(record["level"]),
                str(record["condition"]),
            )
        ].append(float(record["heldout_pair_micro_f1"]))
    for (method, level, condition), values in sorted(grouped.items()):
        aggregates.append(
            {
                "method_id": method,
                "level": level,
                "condition": condition,
                "folds": len(values),
                "heldout_pair_micro_f1_mean": float(np.mean(values)),
                "heldout_pair_micro_f1_std": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "heldout_pair_prediction_count": collapse_totals[
                    (method, level, condition)
                ],
            }
        )
    return {
        "schema_version": "taxonomy_two_stage_precloud_v2_artifact_audit_v1",
        "status": "pass" if not failures else "fail",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "official_test_artifact_count": 0,
        "test_contract_count": test_contract_total,
        "failure_count": len(failures),
        "failures": failures,
        "expected_methods": list(METHODS),
        "expected_levels": list(LEVELS),
        "fold_artifact_count": len(records),
        "prerequisite_status": prerequisite_status,
        "augmented_qwen_cache": cache,
        "aggregate": aggregates,
        "records": records,
        "artifact_sha256": artifact_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "artifact_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                key: value[key]
                for key in (
                    "status",
                    "failure_count",
                    "test_contract_count",
                    "fold_artifact_count",
                )
            },
            indent=2,
        )
    )
    return 0 if value["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
