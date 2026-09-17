"""Build immutable, row-level formal evidence for the scientific freeze.

The official test partition is never requested.  The only score repair is the
preregistered QLoRA seen-candidate invariance correction for L2 folds a07/a12:
unchanged seen candidates in D inherit their same-fold N scores, while every
held-out-candidate score remains exactly as published.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyse_taxonomy_post_supervisor_formal_v2 as legacy  # noqa: E402
from msc_project.experiments.taxonomy_post_supervisor import (  # noqa: E402
    evaluate_l2_condition,
    post_supervisor_l2_folds,
)
from msc_project.experiments.taxonomy_scientific_freeze import (  # noqa: E402
    row_pair_confusions,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    capped_two_sentiment_prediction_mask,
)


REPAIR_METHOD = "qwen_candidate_pair_qlora"
REPAIR_FOLDS = {"l2-a07", "l2-a12"}
IDENTITY = ["row_uid", "candidate_aspect", "candidate_sentiment"]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _load_scores(worker_root: Path, result: Mapping[str, Any]) -> pd.DataFrame:
    score_dir = (
        worker_root
        / "scores"
        / str(result["method_id"])
        / str(result["level"])
        / str(result["fold_id"])
        / str(result["condition"])
        / str(result["run_contract_sha256"])[:16]
    )
    frames = [
        pd.read_csv(path, float_precision="round_trip")
        for path in sorted(score_dir.glob("shard-*.csv"))
    ]
    if len(frames) != 8:
        raise ValueError(f"Expected eight score shards under {score_dir}")
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != int(result["score_rows"]):
        raise ValueError(f"Score row count mismatch under {score_dir}")
    if merged.duplicated(IDENTITY).any():
        raise ValueError(f"Duplicate score identities under {score_dir}")
    if legacy.dataframe_sha256(merged) != result["score_sha256"]:
        raise ValueError(f"Published score hash mismatch under {score_dir}")
    return merged


def _result_entries(backup_root: Path) -> list[tuple[Path, Path, dict[str, Any]]]:
    entries = []
    for worker_root in legacy.worker_roots(backup_root).values():
        for path in sorted(worker_root.glob("results/*/*/*/*.json")):
            entries.append((worker_root, path, legacy.read_json(path)))
    return entries


def _repair_seen_scores(
    original_d: pd.DataFrame,
    source_n: pd.DataFrame,
    *,
    heldout_aspects: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = original_d.set_index(IDENTITY, drop=False).sort_index()
    n = source_n.set_index(IDENTITY, drop=False).sort_index()
    if not d.index.equals(n.index):
        raise ValueError("N and D score identities differ; repair is unsafe.")
    heldout = d["candidate_aspect"].astype(str).isin(heldout_aspects)
    seen = ~heldout
    before_aspect = d.loc[seen, "aspect_score"].to_numpy(dtype=float)
    before_sentiment = d.loc[seen, "sentiment_score"].to_numpy(dtype=float)
    source_aspect = n.loc[seen, "aspect_score"].to_numpy(dtype=float)
    source_sentiment = n.loc[seen, "sentiment_score"].to_numpy(dtype=float)
    repaired = d.copy()
    repaired.loc[seen, "aspect_score"] = source_aspect
    repaired.loc[seen, "sentiment_score"] = source_sentiment
    if not np.array_equal(
        repaired.loc[heldout, ["aspect_score", "sentiment_score"]].to_numpy(),
        d.loc[heldout, ["aspect_score", "sentiment_score"]].to_numpy(),
    ):
        raise AssertionError("Held-out scores changed during seen-only repair.")
    if not np.array_equal(
        repaired.loc[seen, ["aspect_score", "sentiment_score"]].to_numpy(),
        n.loc[seen, ["aspect_score", "sentiment_score"]].to_numpy(),
    ):
        raise AssertionError("Repaired seen scores do not exactly equal N.")
    repaired = repaired.reset_index(drop=True)
    audit = {
        "seen_candidate_rows": int(seen.sum()),
        "heldout_candidate_rows_preserved": int(heldout.sum()),
        "aspect_values_replaced": int(np.count_nonzero(before_aspect != source_aspect)),
        "sentiment_values_replaced": int(
            np.count_nonzero(before_sentiment != source_sentiment)
        ),
        "maximum_absolute_aspect_delta": float(
            np.max(np.abs(before_aspect - source_aspect))
        ),
        "maximum_absolute_sentiment_delta": float(
            np.max(np.abs(before_sentiment - source_sentiment))
        ),
        "original_d_score_sha256": legacy.dataframe_sha256(original_d),
        "source_n_score_sha256": legacy.dataframe_sha256(source_n),
        "repaired_d_score_sha256": legacy.dataframe_sha256(repaired),
        "seen_scores_exactly_equal_after_repair": True,
        "heldout_scores_bitwise_unchanged": True,
    }
    return repaired, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    backup_root = args.backup_root.resolve()
    states = legacy.audit_states(backup_root)
    receipt_summary, published = legacy.audit_receipts(backup_root)
    checkpoints = legacy.audit_checkpoints(backup_root, published)
    selections = legacy.audit_selections(backup_root, published)
    original_folds, result_audit = legacy.audit_results_and_scores(
        backup_root, published, selections["values"]
    )
    entries = _result_entries(backup_root)
    l2_fold_by_id = {fold.fold_id: fold for fold in post_supervisor_l2_folds()}

    n_sources: dict[str, pd.DataFrame] = {}
    for worker_root, _, result in entries:
        if (
            result["method_id"] == REPAIR_METHOD
            and result["level"] == "L2"
            and result["fold_id"] in REPAIR_FOLDS
            and result["condition"] == "N"
        ):
            n_sources[str(result["fold_id"])] = _load_scores(worker_root, result)
    if set(n_sources) != REPAIR_FOLDS:
        raise ValueError("The two registered QLoRA N repair sources are incomplete.")

    repaired_records: list[dict[str, Any]] = []
    repairs: dict[str, Any] = {}
    evidence_files: dict[str, str] = {}
    for worker_root, path, result in entries:
        scored = _load_scores(worker_root, result)
        method = str(result["method_id"])
        level = str(result["level"])
        fold_id = str(result["fold_id"])
        condition = str(result["condition"])
        if (
            method == REPAIR_METHOD
            and level == "L2"
            and fold_id in REPAIR_FOLDS
            and condition == "D"
        ):
            scored, repair = _repair_seen_scores(
                scored,
                n_sources[fold_id],
                heldout_aspects=set(result["heldout_aspects"]),
            )
            repair.update(
                {
                    "method_id": method,
                    "level": level,
                    "fold_id": fold_id,
                    "condition": condition,
                    "repair_rule": (
                        "registered-first-condition N supplies scores only for "
                        "unchanged seen candidates; D held-out scores remain published"
                    ),
                    "outcome_labels_used_to_define_repair": False,
                }
            )
            repairs[fold_id] = repair

        mask = capped_two_sentiment_prediction_mask(
            scored,
            aspect_threshold=float(result["thresholds"]["aspect"]),
            second_sentiment_threshold=float(
                result["thresholds"]["runner_up_sentiment"]
            ),
        )
        heldout_aspects = tuple(str(value) for value in result["heldout_aspects"])
        evidence = row_pair_confusions(
            scored,
            mask,
            aspects=heldout_aspects,
            method_id=method,
            fold_id=fold_id,
            condition=condition,
            level=level,
        )
        evidence_path = (
            args.evidence_output_root.resolve()
            / method
            / level
            / fold_id
            / f"{condition}.csv"
        )
        _write_csv(evidence_path, evidence)
        evidence_files[str(evidence_path)] = legacy.file_sha256(evidence_path)

        derived_result = copy.deepcopy(result)
        if level == "L2":
            fold = l2_fold_by_id[fold_id]
            views = evaluate_l2_condition(
                scored,
                fold,
                aspect_threshold=float(result["thresholds"]["aspect"]),
                second_sentiment_threshold=float(
                    result["thresholds"]["runner_up_sentiment"]
                ),
            )
            derived_result["post_supervisor_views"] = views
            derived_result["score_sha256"] = legacy.dataframe_sha256(scored)
        record = legacy._result_record(derived_result, path)
        record["published_score_sha256"] = result["score_sha256"]
        record["analysis_score_sha256"] = legacy.dataframe_sha256(scored)
        record["analysis_seen_invariant_repair"] = bool(
            fold_id in repairs and method == REPAIR_METHOD and condition == "D"
        )
        repaired_records.append(record)

    if set(repairs) != REPAIR_FOLDS:
        raise ValueError("The two registered QLoRA repairs were not both applied.")
    fold_frame = pd.DataFrame(repaired_records).sort_values(
        ["method_id", "level", "fold_id", "condition"], kind="stable"
    )
    if len(fold_frame) != 81 or fold_frame.isna().all(axis=1).any():
        raise ValueError("Derived formal fold table is incomplete.")
    _write_csv(args.fold_output.resolve(), fold_frame)

    audit = {
        "schema_version": "taxonomy_scientific_freeze_formal_evidence_v1",
        "status": "pass",
        "protocol_id": legacy.PROTOCOL_ID,
        "source_backup_root": str(backup_root),
        "allowed_splits": ["train", "validation"],
        "include_official_test": False,
        "official_test_opened": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "immutable_published_audit": result_audit,
        "campaign_states": states,
        "checkpoint_audit": checkpoints,
        "receipt_count": int(
            sum(value["receipt_count"] for value in receipt_summary.values())
        ),
        "qlora_seen_invariant_repairs": repairs,
        "row_evidence_file_count": len(evidence_files),
        "row_evidence_sha256": evidence_files,
        "derived_fold_table": str(args.fold_output.resolve()),
        "derived_fold_table_sha256": legacy.file_sha256(args.fold_output.resolve()),
        "original_formal_fold_rows": int(len(original_folds)),
        "derived_formal_fold_rows": int(len(fold_frame)),
    }
    _write_json(args.audit_output.resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


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
        "--evidence-output-root",
        type=Path,
        default=Path(
            "outputs/experimental/taxonomy_scientific_freeze_v1/formal_row_evidence"
        ),
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=Path(
            "docs/thesis_figure_data/taxonomy_scientific_freeze_v1/formal_fold_results.csv"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "docs/experiments/taxonomy_scientific_freeze_formal_evidence_audit_20260822.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
