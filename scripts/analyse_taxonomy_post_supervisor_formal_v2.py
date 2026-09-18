"""Audit and summarise the sealed taxonomy two-stage formal-v2 campaign.

This script is intentionally independent of the training stack.  It consumes only
the locally replicated validation artifacts; it never resolves or opens FABSA
source splits and therefore cannot touch the official test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

PROTOCOL_ID = "taxonomy_two_stage_formal_v2"
METHODS = (
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_few_shot",
    "qwen_candidate_pair_qlora",
)
TRAINABLE_METHODS = (
    "distilbert_review_candidate_cross_encoder",
    "qwen_candidate_pair_qlora",
)
L2_FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))
L4_FOLDS = ("l4-g01", "l4-g02", "l4-g03")
EXPECTED_RESULTS = {
    (method, "L2", fold, condition)
    for method in METHODS
    for fold in L2_FOLDS
    for condition in ("N", "D")
} | {(method, "L4", fold, "D") for method in METHODS for fold in L4_FOLDS}
SCORE_HASH_COLUMNS = (
    "row_uid",
    "candidate_aspect",
    "candidate_sentiment",
    "aspect_score",
    "sentiment_score",
)
SUMMARY_METRICS = (
    "heldout_pair_micro_f1",
    "heldout_aspect_micro_f1",
    "heldout_presence_ap",
    "heldout_presence_f1",
    "heldout_sentiment_accuracy",
    "oracle_pair_micro_f1",
    "oracle_top_one_sentiment_accuracy",
    "overall_pair_micro_f1",
    "seen_pair_micro_f1",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def reject_nonfinite(value: object, location: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite value at {location}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            reject_nonfinite(child, f"{location}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            reject_nonfinite(child, f"{location}[{index}]")


def validate_sealed(value: Mapping[str, Any], field: str, path: Path) -> None:
    expected = str(value.get(field, ""))
    payload = dict(value)
    payload.pop(field, None)
    if expected != canonical_sha256(payload):
        raise ValueError(f"Sealed payload SHA-256 mismatch: {path}")


def safe_relative_path(raw: str) -> Path:
    posix = PurePosixPath(raw)
    if (
        posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"Unsafe receipt path: {raw}")
    return Path(*posix.parts)


def worker_roots(backup_root: Path) -> dict[str, Path]:
    return {
        "worker-qlora-a": backup_root / "worker-qlora-a",
        "worker-qlora-b": backup_root / "worker-qlora-b",
        "worker-distil-frozen": backup_root / "worker-distil-frozen",
    }


def audit_states(backup_root: Path) -> dict[str, Any]:
    expected = {
        "worker-qlora-a": (40, 40),
        "worker-qlora-b": (35, 35),
        "worker-distil-frozen": (75, 75),
    }
    records: dict[str, Any] = {}
    for worker_id, root in worker_roots(backup_root).items():
        path = root / "_campaign" / "workers" / worker_id / "state.json"
        state = read_json(path)
        reject_nonfinite(state, str(path))
        completed, planned = expected[worker_id]
        if (
            state.get("status") != "complete"
            or state.get("completed_count") != completed
            or state.get("planned_count") != planned
            or state.get("current_job") is not None
            or state.get("failure_count") != 0
            or state.get("test_contract_count") != 0
        ):
            raise ValueError(f"Campaign state is not safely complete: {path}")
        records[worker_id] = {
            "completed_count": completed,
            "planned_count": planned,
            "failure_count": 0,
            "test_contract_count": 0,
            "updated_at": state.get("updated_at"),
        }

    frozen_path = (
        worker_roots(backup_root)["worker-distil-frozen"]
        / "_campaign/workers/worker-distil-frozen/frozen_qwen_few_shot/state.json"
    )
    frozen = read_json(frozen_path)
    reject_nonfinite(frozen, str(frozen_path))
    if (
        frozen.get("status") != "complete"
        or frozen.get("completed_count") != 15
        or frozen.get("planned_count") != 15
        or frozen.get("current_scope") is not None
        or frozen.get("failure_count") != 0
        or frozen.get("test_contract_count") != 0
        or len(set(frozen.get("completed_scopes", []))) != 15
    ):
        raise ValueError("Frozen-Qwen campaign state is not safely complete")
    records["frozen_qwen_few_shot"] = {
        "completed_count": 15,
        "planned_count": 15,
        "failure_count": 0,
        "test_contract_count": 0,
        "updated_at": frozen.get("updated_at"),
    }
    return records


def audit_receipts(backup_root: Path) -> tuple[dict[str, Any], dict[Path, str]]:
    summary: dict[str, Any] = {}
    published: dict[Path, str] = {}
    for worker_id, root in worker_roots(backup_root).items():
        receipt_dir = root / "_sync" / "received"
        receipts = sorted(receipt_dir.glob("*.json"))
        if not receipts:
            raise ValueError(f"No local synchronization receipts: {receipt_dir}")
        record_count = 0
        duplicate_same_hash = 0
        for receipt_path in receipts:
            receipt = read_json(receipt_path)
            reject_nonfinite(receipt, str(receipt_path))
            if (
                receipt.get("schema_version") != "verified_artifact_unit_v1"
                or receipt.get("protocol_id") != PROTOCOL_ID
                or receipt.get("unit_id") != receipt_path.stem
            ):
                raise ValueError(
                    f"Invalid synchronization receipt identity: {receipt_path}"
                )
            records = receipt.get("files")
            if not isinstance(records, list) or not records:
                raise ValueError(f"Empty synchronization receipt: {receipt_path}")
            paths = [
                str(item.get("path", ""))
                for item in records
                if isinstance(item, Mapping)
            ]
            if (
                len(paths) != len(records)
                or paths != sorted(paths)
                or len(set(paths)) != len(paths)
            ):
                raise ValueError(f"Receipt file list is invalid: {receipt_path}")
            for item in records:
                relative = safe_relative_path(str(item["path"]))
                path = root / relative
                expected_size = int(item["bytes"])
                expected_hash = str(item["sha256"])
                if not path.is_file() or path.stat().st_size != expected_size:
                    raise ValueError(
                        f"Missing or size-mismatched replicated file: {path}"
                    )
                observed_hash = file_sha256(path)
                if observed_hash != expected_hash:
                    raise ValueError(f"Replicated file SHA-256 mismatch: {path}")
                resolved = path.resolve()
                previous = published.get(resolved)
                if previous is not None:
                    if previous != expected_hash:
                        raise ValueError(
                            f"Conflicting receipts for replicated file: {path}"
                        )
                    duplicate_same_hash += 1
                else:
                    published[resolved] = expected_hash
                record_count += 1
        summary[worker_id] = {
            "receipt_count": len(receipts),
            "file_record_count": record_count,
            "unique_verified_file_count": record_count - duplicate_same_hash,
            "duplicate_same_hash_record_count": duplicate_same_hash,
        }
    return summary, published


def require_published(path: Path, published: Mapping[Path, str]) -> None:
    if path.resolve() not in published:
        raise ValueError(f"Critical artifact lacks a local verified receipt: {path}")


def audit_checkpoints(
    backup_root: Path, published: Mapping[Path, str]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    internal_file_count = 0
    for worker_root in worker_roots(backup_root).values():
        for manifest_path in sorted(
            worker_root.glob("checkpoints/**/checkpoint.manifest.json")
        ):
            require_published(manifest_path, published)
            manifest = read_json(manifest_path)
            reject_nonfinite(manifest, str(manifest_path))
            contract = manifest.get("training_contract")
            files = manifest.get("files")
            evidence = manifest.get("evidence")
            if (
                not isinstance(contract, Mapping)
                or not isinstance(files, Mapping)
                or not isinstance(evidence, Mapping)
            ):
                raise TypeError(f"Malformed checkpoint manifest: {manifest_path}")
            method = str(contract.get("method_id", ""))
            if (
                manifest.get("schema_version") != "taxonomy_training_checkpoint_v1"
                or method not in TRAINABLE_METHODS
                or contract.get("protocol_id") != PROTOCOL_ID
                or contract.get("training_pairs") != 4096
                or evidence.get("test_contract_count") != 0
            ):
                raise ValueError(f"Checkpoint contract mismatch: {manifest_path}")
            contract_payload = dict(contract)
            expected_contract_hash = str(contract_payload.pop("contract_sha256", ""))
            if expected_contract_hash != canonical_sha256(contract_payload):
                raise ValueError(
                    f"Checkpoint training-contract SHA-256 mismatch: {manifest_path}"
                )
            for relative_raw, expected_hash in files.items():
                relative = safe_relative_path(str(relative_raw))
                path = manifest_path.parent / relative
                if not path.is_file() or file_sha256(path) != str(expected_hash):
                    raise ValueError(
                        f"Checkpoint internal file SHA-256 mismatch: {path}"
                    )
                require_published(path, published)
                internal_file_count += 1
            counts[method] += 1
    if counts != Counter({method: 45 for method in TRAINABLE_METHODS}):
        raise ValueError(f"Unexpected checkpoint counts: {dict(counts)}")
    return {
        "manifest_count": sum(counts.values()),
        "manifest_count_by_method": dict(sorted(counts.items())),
        "internal_file_count": internal_file_count,
    }


def audit_selections(
    backup_root: Path, published: Mapping[Path, str]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for worker_root in worker_roots(backup_root).values():
        for path in sorted(worker_root.glob("selections/*/*.json")):
            require_published(path, published)
            value = read_json(path)
            validate_sealed(value, "selection_payload_sha256", path)
            reject_nonfinite(value, str(path))
            method = str(value.get("method_id", ""))
            scope = str(value.get("training_scope_id", ""))
            if (
                method not in METHODS
                or value.get("protocol_id") != PROTOCOL_ID
                or value.get("failure_count") != 0
                or value.get("test_contract_count") != 0
            ):
                raise ValueError(f"Selection safety mismatch: {path}")
            thresholds = value.get("selected_thresholds", value.get("thresholds"))
            if not isinstance(thresholds, Mapping):
                raise TypeError(f"Selection thresholds are missing: {path}")
            for name in ("aspect", "runner_up_sentiment"):
                threshold = float(thresholds[name])
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"Invalid selected threshold at {path}: {name}")
            identity = (method, scope)
            if identity in by_identity:
                raise ValueError(f"Duplicate selection identity: {identity}")
            normalised = dict(value)
            normalised["selected_thresholds"] = dict(thresholds)
            by_identity[identity] = normalised
            counts[method] += 1
    if counts != Counter({method: 15 for method in METHODS}):
        raise ValueError(f"Unexpected selection counts: {dict(counts)}")
    return {
        "selection_count": sum(counts.values()),
        "selection_count_by_method": dict(sorted(counts.items())),
        "values": by_identity,
    }


def dataframe_sha256(frame: pd.DataFrame) -> str:
    records = (
        frame[list(SCORE_HASH_COLUMNS)]
        .sort_values(
            ["row_uid", "candidate_aspect", "candidate_sentiment"], kind="stable"
        )
        .to_dict(orient="records")
    )
    return canonical_sha256({"columns": list(SCORE_HASH_COLUMNS), "records": records})


def _partition_metrics(result: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    views = result["post_supervisor_views"]
    return views["L2_E"]["partitions"][name]


def _check_confusion(metrics: Mapping[str, Any], location: str) -> None:
    for prefix in ("aspect", "pair"):
        tp = int(metrics[f"{prefix}_label_tp"])
        fp = int(metrics[f"{prefix}_label_fp"])
        fn = int(metrics[f"{prefix}_label_fn"])
        if int(metrics[f"{prefix}_predicted_label_count"]) != tp + fp:
            raise ValueError(
                f"Predicted-label confusion mismatch at {location}:{prefix}"
            )
        if int(metrics[f"{prefix}_gold_label_count"]) != tp + fn:
            raise ValueError(f"Gold-label confusion mismatch at {location}:{prefix}")
    presence = {
        name: int(metrics[f"presence_{name}_rows"]) for name in ("tp", "fp", "fn", "tn")
    }
    if any(value < 0 for value in presence.values()):
        raise ValueError(f"Invalid presence confusion counts at {location}")


def _result_record(result: Mapping[str, Any], path: Path) -> dict[str, Any]:
    level = str(result["level"])
    record: dict[str, Any] = {
        "method_id": result["method_id"],
        "level": level,
        "fold_id": result["fold_id"],
        "condition": result["condition"],
        "heldout_aspects": "; ".join(result.get("heldout_aspects", [])),
        "training_scope_id": result["training_scope_id"],
        "selected_learning_rate": result.get("selected_learning_rate"),
        "aspect_threshold": result["thresholds"]["aspect"],
        "runner_up_sentiment_threshold": result["thresholds"]["runner_up_sentiment"],
        "score_rows": result["score_rows"],
        "score_sha256": result["score_sha256"],
        "result_payload_sha256": result["result_payload_sha256"],
        "source_path": str(path),
    }
    partitions = result.get("partitions", {})
    if level == "L2":
        heldout = _partition_metrics(result, "heldout")
        seen = _partition_metrics(result, "seen")
        overall = _partition_metrics(result, "overall")
        stage = result["post_supervisor_views"]["L2_S"]
        aspect_presence = stage["aspect_presence"]
        oracle = stage["oracle_aspect_gated_sentiment"]
        oracle_pair = oracle["capped_two_pair_metrics"]
        heldout_row_count = sum(
            int(heldout[f"presence_{name}_rows"]) for name in ("tp", "fp", "fn", "tn")
        )
        record.update(
            {
                "heldout_pair_micro_f1": heldout["pair_micro_f1"],
                "heldout_aspect_micro_f1": heldout["aspect_micro_f1"],
                "heldout_presence_f1": heldout["presence_f1"],
                "heldout_presence_ap": aspect_presence["average_precision"],
                "heldout_sentiment_accuracy": heldout[
                    "sentiment_accuracy_when_gold_aspect_predicted"
                ],
                "oracle_pair_micro_f1": oracle_pair["pair_micro_f1"],
                "oracle_top_one_sentiment_accuracy": oracle[
                    "top_one_conditional_accuracy"
                ]["conditional_sentiment_accuracy_all_gold_aspects"],
                "overall_pair_micro_f1": overall["pair_micro_f1"],
                "seen_pair_micro_f1": seen["pair_micro_f1"],
                "heldout_aspect_predicted_label_count": heldout[
                    "aspect_predicted_label_count"
                ],
                "heldout_pair_predicted_label_count": heldout[
                    "pair_predicted_label_count"
                ],
                "heldout_presence_positive_rows": aspect_presence["positive_rows"],
                "heldout_presence_predicted_positive_rows": aspect_presence[
                    "predicted_positive_rows"
                ],
                "heldout_row_count": heldout_row_count,
                "heldout_candidate_aspect_count": len(
                    result.get("heldout_aspects", [])
                ),
                "heldout_max_aspect_label_count": heldout_row_count
                * len(result.get("heldout_aspects", [])),
            }
        )
    else:
        heldout = partitions["heldout"]
        seen = partitions.get("seen")
        overall = partitions.get("overall", partitions.get("full"))
        if not isinstance(overall, Mapping):
            raise ValueError(f"Level 4 full partition is missing: {path}")
        heldout_row_count = sum(
            int(heldout[f"presence_{name}_rows"]) for name in ("tp", "fp", "fn", "tn")
        )
        record.update(
            {
                "heldout_pair_micro_f1": heldout["pair_micro_f1"],
                "heldout_aspect_micro_f1": heldout["aspect_micro_f1"],
                "heldout_presence_f1": heldout["presence_f1"],
                "heldout_presence_ap": np.nan,
                "heldout_sentiment_accuracy": heldout[
                    "sentiment_accuracy_when_gold_aspect_predicted"
                ],
                "oracle_pair_micro_f1": np.nan,
                "oracle_top_one_sentiment_accuracy": np.nan,
                "overall_pair_micro_f1": overall["pair_micro_f1"],
                "seen_pair_micro_f1": (
                    seen["pair_micro_f1"] if isinstance(seen, Mapping) else np.nan
                ),
                "heldout_aspect_predicted_label_count": heldout[
                    "aspect_predicted_label_count"
                ],
                "heldout_pair_predicted_label_count": heldout[
                    "pair_predicted_label_count"
                ],
                "heldout_presence_positive_rows": heldout["presence_tp_rows"]
                + heldout["presence_fn_rows"],
                "heldout_presence_predicted_positive_rows": heldout["presence_tp_rows"]
                + heldout["presence_fp_rows"],
                "heldout_row_count": heldout_row_count,
                "heldout_candidate_aspect_count": len(
                    result.get("heldout_aspects", [])
                ),
                "heldout_max_aspect_label_count": heldout_row_count
                * len(result.get("heldout_aspects", [])),
            }
        )
    return record


def audit_results_and_scores(
    backup_root: Path,
    published: Mapping[Path, str],
    selections: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    observed_identities: set[tuple[str, str, str, str]] = set()
    score_shard_count = 0
    score_row_count = 0
    min_unique_aspect_score = math.inf
    min_unique_sentiment_score = math.inf
    for worker_root in worker_roots(backup_root).values():
        for path in sorted(worker_root.glob("results/*/*/*/*.json")):
            result = read_json(path)
            validate_sealed(result, "result_payload_sha256", path)
            reject_nonfinite(result, str(path))
            identity = (
                str(result.get("method_id", "")),
                str(result.get("level", "")),
                str(result.get("fold_id", "")),
                str(result.get("condition", "")),
            )
            if identity not in EXPECTED_RESULTS or identity in observed_identities:
                raise ValueError(f"Unexpected or duplicate formal result: {identity}")
            if (
                result.get("protocol_id") != PROTOCOL_ID
                or result.get("evaluation_partition") != "validation_only"
                or result.get("selection_partition") != "seen_validation_only"
                or result.get("failure_count") != 0
                or result.get("test_contract_count") != 0
                or int(result.get("maximum_sentiments_per_aspect", 99)) > 2
                or int(result.get("score_shards", 0)) != 8
            ):
                raise ValueError(f"Formal result safety mismatch: {path}")
            require_published(path, published)
            method, level, fold, condition = identity
            scope = str(result["training_scope_id"])
            selection = selections[(method, scope)]
            if dict(result["thresholds"]) != dict(selection["selected_thresholds"]):
                raise ValueError(
                    f"Result thresholds differ from seen-validation selection: {path}"
                )
            if method in TRAINABLE_METHODS:
                if result.get("training_contract_sha256") != selection.get(
                    "selected_candidate_contract_sha256"
                ) or result.get("selected_learning_rate") != selection.get(
                    "selected_learning_rate"
                ):
                    raise ValueError(f"Result uses an unselected checkpoint: {path}")
            else:
                if result.get("scope_contract_sha256") != selection.get(
                    "scope_contract_sha256"
                ):
                    raise ValueError(f"Frozen result scope contract mismatch: {path}")

            score_dir = (
                worker_root
                / "scores"
                / method
                / level
                / fold
                / condition
                / str(result["run_contract_sha256"])[:16]
            )
            manifests = sorted(score_dir.glob("shard-*.manifest.json"))
            if len(manifests) != 8:
                raise ValueError(f"Incomplete score-shard manifest set: {path}")
            frames: list[pd.DataFrame] = []
            indices: set[int] = set()
            for manifest_path in manifests:
                require_published(manifest_path, published)
                manifest = read_json(manifest_path)
                reject_nonfinite(manifest, str(manifest_path))
                contract = manifest.get("contract")
                if (
                    not isinstance(contract, Mapping)
                    or contract.get("protocol_id") != PROTOCOL_ID
                    or contract.get("split") != "validation"
                    or contract.get("contract_sha256") != result["run_contract_sha256"]
                    or manifest.get("test_contract_count") != 0
                ):
                    raise ValueError(f"Score-shard contract mismatch: {manifest_path}")
                index = int(manifest["shard_index"])
                if index in indices:
                    raise ValueError(f"Duplicate score-shard index: {manifest_path}")
                indices.add(index)
                csv_path = manifest_path.with_name(
                    manifest_path.name.replace(".manifest.json", ".csv")
                )
                require_published(csv_path, published)
                if file_sha256(csv_path) != manifest.get("csv_sha256"):
                    raise ValueError(f"Score-shard CSV SHA-256 mismatch: {csv_path}")
                frame = pd.read_csv(csv_path, float_precision="round_trip")
                if len(frame) != int(manifest["rows"]):
                    raise ValueError(f"Score-shard row count mismatch: {csv_path}")
                frames.append(frame)
                score_shard_count += 1
            if indices != set(range(8)):
                raise ValueError(f"Non-contiguous score-shard indices: {path}")
            merged = pd.concat(frames, ignore_index=True)
            if len(merged) != int(result["score_rows"]):
                raise ValueError(f"Merged score row count mismatch: {path}")
            if merged.duplicated(
                ["row_uid", "candidate_aspect", "candidate_sentiment"]
            ).any():
                raise ValueError(f"Duplicate score identity: {path}")
            scores = merged[["aspect_score", "sentiment_score"]].to_numpy(dtype=float)
            if (
                not np.isfinite(scores).all()
                or (scores < 0.0).any()
                or (scores > 1.0).any()
            ):
                raise ValueError(f"Non-finite or out-of-range probability: {path}")
            aspect_unique = int(merged["aspect_score"].nunique())
            sentiment_unique = int(merged["sentiment_score"].nunique())
            if aspect_unique < 2 or sentiment_unique < 2:
                raise ValueError(f"Collapsed score distribution: {path}")
            min_unique_aspect_score = min(min_unique_aspect_score, aspect_unique)
            min_unique_sentiment_score = min(
                min_unique_sentiment_score, sentiment_unique
            )
            if result.get("score_sha256") != dataframe_sha256(merged):
                raise ValueError(f"Merged score SHA-256 mismatch: {path}")

            partitions = (
                result["post_supervisor_views"]["L2_E"]["partitions"]
                if level == "L2"
                else result["partitions"]
            )
            for name, metrics in partitions.items():
                _check_confusion(metrics, f"{path}:{name}")
            records.append(_result_record(result, path))
            score_row_count += len(merged)
            observed_identities.add(identity)

    missing = EXPECTED_RESULTS - observed_identities
    if missing or len(observed_identities) != 81:
        raise ValueError(f"Formal result set is incomplete; missing={sorted(missing)}")
    frame = pd.DataFrame.from_records(records).sort_values(
        ["method_id", "level", "fold_id", "condition"], kind="stable"
    )

    collapse_checks: dict[str, Any] = {}
    for (method, level, condition), group in frame.groupby(
        ["method_id", "level", "condition"], sort=True
    ):
        predicted = int(group["heldout_aspect_predicted_label_count"].sum())
        positives = int(group["heldout_presence_positive_rows"].sum())
        maximum = int(group["heldout_max_aspect_label_count"].sum())
        if predicted <= 0:
            raise ValueError(
                f"Globally empty held-out predictions: {method}/{level}/{condition}"
            )
        if predicted >= maximum:
            raise ValueError(
                f"Globally saturated held-out predictions: {method}/{level}/{condition}"
            )
        collapse_checks[f"{method}/{level}/{condition}"] = {
            "fold_count": len(group),
            "heldout_aspect_predicted_label_count": predicted,
            "heldout_positive_row_count": positives,
            "heldout_max_aspect_label_count": maximum,
            "status": "non_empty_and_non_saturated",
        }
    return frame, {
        "result_count": len(frame),
        "score_shard_count": score_shard_count,
        "score_row_count": score_row_count,
        "minimum_unique_aspect_score_count_per_result": int(min_unique_aspect_score),
        "minimum_unique_sentiment_score_count_per_result": int(
            min_unique_sentiment_score
        ),
        "prediction_collapse_checks": collapse_checks,
    }


def percentile_bootstrap(
    values: np.ndarray, seed: int = 1729, draws: int = 20000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        means.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def holm_adjust(frame: pd.DataFrame, group_column: str, p_column: str) -> pd.Series:
    adjusted = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby(group_column, sort=False):
        ordered = group[p_column].sort_values(kind="stable")
        running = 0.0
        m = len(ordered)
        for rank, (index, pvalue) in enumerate(ordered.items()):
            running = max(running, min(1.0, float(pvalue) * (m - rank)))
            adjusted.loc[index] = running
    return adjusted


def build_model_effects(folds: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("frozen_qwen_few_shot", "distilbert_review_candidate_cross_encoder"),
        ("qwen_candidate_pair_qlora", "distilbert_review_candidate_cross_encoder"),
        ("qwen_candidate_pair_qlora", "frozen_qwen_few_shot"),
    )
    rows: list[dict[str, Any]] = []
    for (level, condition), level_frame in folds.groupby(
        ["level", "condition"], sort=True
    ):
        for comparator, baseline in comparisons:
            left = level_frame[level_frame["method_id"] == comparator].set_index(
                "fold_id"
            )
            right = level_frame[level_frame["method_id"] == baseline].set_index(
                "fold_id"
            )
            if set(left.index) != set(right.index):
                raise ValueError(
                    f"Model comparison folds do not match: {comparator}/{baseline}/{level}/{condition}"
                )
            for metric in SUMMARY_METRICS:
                joined = pd.concat(
                    [
                        left[metric].rename("comparator"),
                        right[metric].rename("baseline"),
                    ],
                    axis=1,
                ).dropna()
                if joined.empty:
                    continue
                delta = joined["comparator"].to_numpy(dtype=float) - joined[
                    "baseline"
                ].to_numpy(dtype=float)
                low, high = percentile_bootstrap(delta)
                rows.append(
                    {
                        "protocol": PROTOCOL_ID,
                        "level": level,
                        "condition": condition,
                        "comparison": f"{comparator}_minus_{baseline}",
                        "comparator_method": comparator,
                        "baseline_method": baseline,
                        "metric": metric,
                        "fold_count": len(delta),
                        "baseline_mean": joined["baseline"].mean(),
                        "comparator_mean": joined["comparator"].mean(),
                        "mean_difference": float(delta.mean()),
                        "bootstrap_95_ci_low": low,
                        "bootstrap_95_ci_high": high,
                        "improved_folds": int((delta > 1e-15).sum()),
                        "tied_folds": int((np.abs(delta) <= 1e-15).sum()),
                        "worsened_folds": int((delta < -1e-15).sum()),
                        "exact_sign_flip_p": exact_sign_flip_p(delta),
                        "holm_group": f"{level}/{condition}/{metric}",
                    }
                )
    frame = pd.DataFrame(rows)
    frame["holm_p_across_model_comparisons"] = holm_adjust(
        frame, "holm_group", "exact_sign_flip_p"
    )
    return frame.drop(columns=["holm_group"])


def build_formal_tables(
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    for (method, level, condition), group in folds.groupby(
        ["method_id", "level", "condition"], sort=True
    ):
        row: dict[str, Any] = {
            "protocol": PROTOCOL_ID,
            "method_id": method,
            "level": level,
            "condition": condition,
            "fold_count": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
        summary_rows.append(row)
    model_summary = pd.DataFrame(summary_rows)

    paired_rows: list[dict[str, Any]] = []
    l2 = folds[folds["level"] == "L2"]
    for method, method_frame in l2.groupby("method_id", sort=True):
        indexed = method_frame.set_index(["fold_id", "condition"])
        for metric in SUMMARY_METRICS:
            n_values = np.asarray(
                [indexed.loc[(fold, "N"), metric] for fold in L2_FOLDS], dtype=float
            )
            d_values = np.asarray(
                [indexed.loc[(fold, "D"), metric] for fold in L2_FOLDS], dtype=float
            )
            delta = d_values - n_values
            low, high = percentile_bootstrap(delta)
            paired_rows.append(
                {
                    "protocol": PROTOCOL_ID,
                    "method_id": method,
                    "metric": metric,
                    "fold_count": len(delta),
                    "N_mean": float(n_values.mean()),
                    "D_mean": float(d_values.mean()),
                    "D_minus_N_mean": float(delta.mean()),
                    "D_minus_N_sd": float(delta.std(ddof=1)),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "improved_folds": int((delta > 1e-15).sum()),
                    "tied_folds": int((np.abs(delta) <= 1e-15).sum()),
                    "worsened_folds": int((delta < -1e-15).sum()),
                    "exact_sign_flip_p": exact_sign_flip_p(delta),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired["holm_p_across_methods"] = holm_adjust(paired, "metric", "exact_sign_flip_p")

    stage_rows: list[dict[str, Any]] = []
    for (method, condition), group in l2.groupby(["method_id", "condition"], sort=True):
        stage_rows.append(
            {
                "protocol": PROTOCOL_ID,
                "method_id": method,
                "condition": condition,
                "fold_count": len(group),
                "stage_1_presence_ap_mean": group["heldout_presence_ap"].mean(),
                "stage_1_presence_f1_mean": group["heldout_presence_f1"].mean(),
                "stage_2_oracle_pair_micro_f1_mean": group[
                    "oracle_pair_micro_f1"
                ].mean(),
                "stage_2_oracle_top_one_accuracy_mean": group[
                    "oracle_top_one_sentiment_accuracy"
                ].mean(),
                "end_to_end_heldout_pair_micro_f1_mean": group[
                    "heldout_pair_micro_f1"
                ].mean(),
                "end_to_end_overall_pair_micro_f1_mean": group[
                    "overall_pair_micro_f1"
                ].mean(),
            }
        )
    stage_summary = pd.DataFrame(stage_rows)
    l4_groups = folds[folds["level"] == "L4"].copy()
    model_effects = build_model_effects(folds)
    return model_summary, paired, stage_summary, l4_groups, model_effects


def build_experiment_level_table(
    formal_summary: pd.DataFrame, local_audit_path: Path
) -> pd.DataFrame:
    audit = read_json(local_audit_path)
    if (
        audit.get("status") != "pass"
        or audit.get("official_test_opened") is not False
        or audit.get("test_contract_count") != 0
        or audit.get("failure_count") != 0
    ):
        raise ValueError(
            "Local completion audit is not safe to merge into the comparison table"
        )
    rows: list[dict[str, Any]] = []
    level1 = audit["level1"]
    rows.append(
        {
            "evidence_source": "local_completion_audit_v1",
            "level": "L1",
            "model_id": "strict_train_only_tfidf_logistic_regression",
            "condition": "full_seen_taxonomy",
            "fold_count": 1,
            "heldout_pair_micro_f1_mean": np.nan,
            "overall_pair_micro_f1_mean": level1["pair_micro_f1"],
            "overall_aspect_micro_f1_mean": level1["aspect_micro_f1"],
            "heldout_presence_ap_mean": np.nan,
            "comparability_note": "Seen-taxonomy Level 1 sanity baseline; not directly comparable with held-out Level 2/4.",
        }
    )
    for method, summary in sorted(audit["local_level2"].items()):
        for condition, metrics in sorted(summary["condition_macro_means"].items()):
            rows.append(
                {
                    "evidence_source": "local_completion_audit_v1",
                    "level": "L2",
                    "model_id": method,
                    "condition": condition,
                    "fold_count": 12,
                    "heldout_pair_micro_f1_mean": metrics["heldout_pair_micro_f1"],
                    "overall_pair_micro_f1_mean": metrics["overall_pair_micro_f1"],
                    "overall_aspect_micro_f1_mean": np.nan,
                    "heldout_presence_ap_mean": metrics["heldout_presence_ap"],
                    "comparability_note": "Validation-only local Level 2 run under the audited post-supervisor protocol.",
                }
            )
    dcwt = audit["dcwt"]["kernel_ridge"]["condition_macro_means"]
    for condition, metrics in sorted(dcwt.items()):
        rows.append(
            {
                "evidence_source": "local_completion_audit_v1",
                "level": "L2",
                "model_id": "description_conditioned_weight_transfer_kernel_ridge",
                "condition": condition,
                "fold_count": 12,
                "heldout_pair_micro_f1_mean": metrics["heldout_pair_micro_f1"],
                "overall_pair_micro_f1_mean": metrics["overall_pair_micro_f1"],
                "overall_aspect_micro_f1_mean": np.nan,
                "heldout_presence_ap_mean": metrics["heldout_presence_ap"],
                "comparability_note": "Exploratory appendix method; validation-only and not a primary architecture comparison.",
            }
        )
    for _, record in formal_summary.iterrows():
        rows.append(
            {
                "evidence_source": PROTOCOL_ID,
                "level": record["level"],
                "model_id": record["method_id"],
                "condition": record["condition"],
                "fold_count": int(record["fold_count"]),
                "heldout_pair_micro_f1_mean": record["heldout_pair_micro_f1_mean"],
                "overall_pair_micro_f1_mean": record["overall_pair_micro_f1_mean"],
                "overall_aspect_micro_f1_mean": np.nan,
                "heldout_presence_ap_mean": record["heldout_presence_ap_mean"],
                "comparability_note": "Formal validation-only result; Level 2 N/D or Level 4 D as labelled.",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["level", "evidence_source", "model_id", "condition"], kind="stable"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path(
            r"C:\Msc_DSML\Msc_Project\cloud_backups\taxonomy_two_stage_formal_v2_r2"
        ),
    )
    parser.add_argument(
        "--local-audit",
        type=Path,
        default=Path(
            "outputs/experimental/taxonomy_post_supervisor_local_v1/local_completion_audit.json"
        ),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("docs/thesis_figure_data/taxonomy_post_supervisor_formal_v2"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "docs/experiments/taxonomy_post_supervisor_formal_v2_audit_20260821.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backup_root = args.backup_root.resolve()
    states = audit_states(backup_root)
    receipt_summary, published = audit_receipts(backup_root)
    checkpoints = audit_checkpoints(backup_root, published)
    selection_audit = audit_selections(backup_root, published)
    folds, result_audit = audit_results_and_scores(
        backup_root, published, selection_audit["values"]
    )
    model_summary, paired, stage_summary, l4_groups, model_effects = (
        build_formal_tables(folds)
    )
    experiment_level = build_experiment_level_table(model_summary, args.local_audit)

    tables = {
        "formal_fold_results.csv": folds,
        "formal_model_condition_summary.csv": model_summary,
        "formal_l2_paired_description_effects.csv": paired,
        "formal_l2_stage_view_summary.csv": stage_summary,
        "formal_l4_group_results.csv": l4_groups,
        "formal_paired_model_effects.csv": model_effects,
        "experiment_level_model_comparison.csv": experiment_level,
    }
    table_hashes: dict[str, str] = {}
    for name, frame in tables.items():
        path = args.table_dir / name
        atomic_csv(path, frame)
        table_hashes[name] = file_sha256(path)

    receipt_counts = {
        worker: value["receipt_count"] for worker, value in receipt_summary.items()
    }
    audit = {
        "schema_version": "taxonomy_post_supervisor_formal_v2_local_audit_v1",
        "status": "pass",
        "protocol_id": PROTOCOL_ID,
        "deployed_commit": "aa84212976a652d62cfca31ed8bf0516a216c485",
        "source_backup_root": str(backup_root),
        "allowed_splits": ["train", "validation"],
        "include_official_test": False,
        "official_test_opened": False,
        "failure_count": 0,
        "test_contract_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "campaign_states": states,
        "synchronization": {
            "receipt_count_by_worker": receipt_counts,
            "receipt_count_total": sum(receipt_counts.values()),
            "workers": receipt_summary,
        },
        "checkpoints": checkpoints,
        "selections": {
            key: value for key, value in selection_audit.items() if key != "values"
        },
        "results_and_scores": result_audit,
        "expected_result_count": 81,
        "result_count_per_method": 27,
        "maximum_sentiments_per_aspect": 2,
        "table_sha256": table_hashes,
    }
    audit["audit_payload_sha256"] = canonical_sha256(audit)
    atomic_json(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
