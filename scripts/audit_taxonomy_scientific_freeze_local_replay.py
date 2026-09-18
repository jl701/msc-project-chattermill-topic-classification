"""Verify that local row-evidence replays exactly reproduce frozen results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "frozen_qwen_candidate_pair",
)
FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe(value: dict[str, Any], path: Path) -> None:
    if (
        value.get("failure_count") != 0
        or value.get("non_finite_value_count") != 0
        or value.get("resume_conflict_count") != 0
        or value.get("test_contract_count") != 0
    ):
        raise ValueError(f"Unsafe replay artifact: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = []
    for method in METHODS:
        for fold in FOLDS:
            original_path = args.original_root / "level2_local_ndr" / method / "folds" / f"{fold}.json"
            replay_path = args.replay_root / "local_replay" / method / "folds" / f"{fold}.json"
            original = _read(original_path)
            replay = _read(replay_path)
            _safe(original, original_path)
            _safe(replay, replay_path)
            fields = (
                "protocol_id",
                "method_id",
                "fold_id",
                "training_scope_id",
                "heldout_aspect",
                "train_rows",
                "validation_rows",
                "selection_partition",
                "aspect_threshold",
                "second_sentiment_threshold",
                "seen_score_sha256",
                "conditions",
            )
            mismatches = [field for field in fields if original[field] != replay[field]]
            if mismatches:
                raise ValueError(
                    f"Local replay differs for {method}/{fold}: {mismatches}"
                )
            records.append(
                {
                    "method_id": method,
                    "fold_id": fold,
                    "status": "exact_result_reproduced",
                    "original_sha256": _sha256(original_path),
                    "replay_sha256": _sha256(replay_path),
                }
            )

    for fold in FOLDS:
        original_path = args.original_root / "dcwt" / "folds" / f"{fold}.json"
        replay_path = args.replay_root / "dcwt_replay" / "folds" / f"{fold}.json"
        original = _read(original_path)
        replay = _read(replay_path)
        _safe(original, original_path)
        _safe(replay, replay_path)
        fields = (
            "protocol_id",
            "protocol_sha256",
            "fold_id",
            "training_scope_id",
            "heldout_aspect",
            "train_rows",
            "validation_rows",
            "target_labels_used_for_selection",
            "target_classifier_weight_used_for_selection",
            "generators",
        )
        mismatches = [field for field in fields if original[field] != replay[field]]
        if mismatches:
            raise ValueError(f"DCWT replay differs for {fold}: {mismatches}")
        records.append(
            {
                "method_id": "description_conditioned_weight_transfer_kernel_ridge",
                "fold_id": fold,
                "status": "exact_result_reproduced",
                "original_sha256": _sha256(original_path),
                "replay_sha256": _sha256(replay_path),
            }
        )

    evidence_paths = sorted(args.evidence_root.glob("*/*/[ND].csv"))
    if len(evidence_paths) != 96:
        raise ValueError(f"Expected 96 local evidence files, found {len(evidence_paths)}")
    prediction_checks = {}
    for path in evidence_paths:
        frame = pd.read_csv(path)
        numeric = frame[["pair_tp", "pair_fp", "pair_fn", "pair_gold_count", "pair_predicted_count"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Non-finite local row evidence: {path}")
        predicted = int(frame["pair_predicted_count"].sum())
        gold = int(frame["pair_gold_count"].sum())
        maximum = 2 * len(frame)
        if gold <= 0 or predicted <= 0 or predicted >= maximum:
            raise ValueError(f"Degenerate local predictions: {path}")
        key = "/".join(path.relative_to(args.evidence_root).parts)
        prediction_checks[key] = {
            "validation_reviews": len(frame),
            "heldout_gold_pair_count": gold,
            "heldout_predicted_pair_count": predicted,
            "maximum_decoder_pair_count": maximum,
            "status": "non_empty_and_non_saturated",
        }
    audit = {
        "schema_version": "taxonomy_scientific_freeze_local_replay_audit_v1",
        "status": "pass",
        "allowed_splits": ["train", "validation"],
        "include_official_test": False,
        "official_test_opened": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "non_finite_value_count": 0,
        "exact_reproduced_fold_results": len(records),
        "local_row_evidence_file_count": len(evidence_paths),
        "local_row_evidence_sha256": {
            str(path.resolve()): _sha256(path) for path in evidence_paths
        },
        "prediction_collapse_checks": prediction_checks,
        "records": records,
    }
    _write_json(args.audit_output.resolve(), audit)
    print(json.dumps({key: value for key, value in audit.items() if key not in {"records", "local_row_evidence_sha256"}}, indent=2))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("outputs/experimental/taxonomy_post_supervisor_local_v1"),
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=Path("outputs/experimental/taxonomy_scientific_freeze_v1"),
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path(
            "outputs/experimental/taxonomy_scientific_freeze_v1/local_row_evidence"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "docs/experiments/taxonomy_scientific_freeze_local_replay_audit_20260822.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
