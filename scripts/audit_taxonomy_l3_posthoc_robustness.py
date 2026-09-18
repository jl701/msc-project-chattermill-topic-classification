"""Audit and relabel the historical six-pair L3 evidence conservatively."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "frozen_qwen_candidate_pair",
)
SELECTED_FOLDS = tuple(
    f"l3-a{left:02d}-a{left+1:02d}" for left in range(1, 12, 2)
)
ALTERNATE_FOLDS = tuple(
    [f"l3-a{left:02d}-a{left+1:02d}" for left in range(2, 12, 2)]
    + ["l3-a12-a01"]
)
FOLDS = SELECTED_FOLDS + ALTERNATE_FOLDS
CONDITIONS = ("NN", "DN", "ND", "DD", "RR")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.source_root.resolve()
    protocol_path = (PROJECT_ROOT / args.protocol_config).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    artifact_audits = []
    metric_rows = []
    for method in METHODS:
        for fold in FOLDS:
            matching_set = (
                "selected_canonical_matching"
                if fold in SELECTED_FOLDS
                else "alternate_cyclic_matching"
            )
            path = root / method / "l3" / "folds" / f"{fold}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("protocol_id") != "taxonomy_two_stage_precloud_v2"
                or value.get("method_id") != method
                or value.get("fold_id") != fold
                or value.get("test_contract_count") != 0
                or value.get("failure_count") != 0
                or value.get("non_finite_value_count") != 0
                or value.get("resume_conflict_count") != 0
                or set(value.get("conditions", {})) != set(CONDITIONS)
            ):
                raise ValueError(f"Unsafe or incomplete L3 artifact: {path}")
            shared_seen_hashes = {
                str(condition_value["seen_score_sha256"])
                for condition_value in value["conditions"].values()
            }
            if shared_seen_hashes != {str(value["seen_score_sha256"])}:
                raise ValueError(f"Seen score invariance failed: {path}")
            fields = {
                "code_commit_and_protocol_config": {
                    "status": "recoverable_from_preregistered_protocol",
                    "base_commit": protocol["base_commit"],
                    "protocol_config_sha256": _sha256(protocol_path),
                },
                "model_and_tokenizer_revisions": {
                    "status": "not_bound_inside_fold_artifact",
                },
                "prompt_or_feature_contract": {
                    "status": "method_id_and_protocol_only; exact rendered prompts/features absent",
                },
                "training_scope_and_fold": {
                    "status": "present",
                    "training_scope_id": value["training_scope_id"],
                    "fold_id": fold,
                    "heldout_aspects": value["heldout_aspects"],
                },
                "candidate_and_row_identities": {
                    "status": "counts_present_but_exact_identity lists absent",
                    "validation_rows": value["validation_rows"],
                    "threshold_selection_candidate_aspects": value[
                        "threshold_selection_candidate_aspects"
                    ],
                },
                "representations": {
                    "status": "present_per_condition",
                    "conditions": {
                        condition: value["conditions"][condition]["representations"]
                        for condition in CONDITIONS
                    },
                },
                "decoder_and_thresholds": {
                    "status": "present",
                    "decoder": "primary_capped_two",
                    "maximum_sentiments_per_aspect": 2,
                    "aspect_threshold": value["aspect_threshold"],
                    "runner_up_sentiment_threshold": value[
                        "second_sentiment_threshold"
                    ],
                },
                "raw_score_hashes": {
                    "status": "seen-only hash present; heldout raw-score hashes absent",
                    "seen_score_sha256": value["seen_score_sha256"],
                },
                "test_access_ledger": {
                    "status": "present_and_zero",
                    "test_contract_count": 0,
                },
            }
            artifact_audits.append(
                {
                    "method_id": method,
                    "fold_id": fold,
                    "matching_set": matching_set,
                    "artifact": str(path),
                    "artifact_sha256": _sha256(path),
                    "field_audit": fields,
                    "exact_result_reuse_approved": False,
                    "appendix_metrics_only_eligible": True,
                    "classification": "post_hoc_targeted_robustness_metrics_only",
                }
            )
            for condition in CONDITIONS:
                heldout = value["conditions"][condition]["partitions"]["heldout"]
                primary = heldout["primary_capped_two"]
                metric_rows.append(
                    {
                        "method_id": method,
                        "fold_id": fold,
                        "matching_set": matching_set,
                        "condition": condition,
                        "heldout_pair_micro_f1": primary["pair_micro_f1"],
                        "heldout_presence_f1": primary["presence_f1"],
                        "heldout_pair_gold_label_count": primary[
                            "pair_gold_label_count"
                        ],
                        "validation_rows": heldout["evaluation_rows"],
                        "maximum_sentiments_per_selected_aspect": heldout[
                            "maximum_sentiments_per_selected_aspect"
                        ],
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    descriptive = (
        metrics.groupby(["method_id", "matching_set", "condition"], sort=True)
        .agg(
            targeted_pair_count=("fold_id", "count"),
            heldout_pair_micro_f1_mean=("heldout_pair_micro_f1", "mean"),
            heldout_pair_micro_f1_sd=("heldout_pair_micro_f1", "std"),
            heldout_presence_f1_mean=("heldout_presence_f1", "mean"),
            heldout_pair_gold_label_support=("heldout_pair_gold_label_count", "sum"),
        )
        .reset_index()
    )
    descriptive.insert(0, "evidence_role", "appendix_post_hoc_targeted_robustness")
    descriptive["inferential_p_value_reported"] = False
    descriptive["outcome_blind_pair_selection"] = False
    _atomic_csv(args.table_output.resolve(), descriptive)
    audit = {
        "schema_version": "taxonomy_l3_posthoc_robustness_audit_v1",
        "status": "pass_with_reuse_restriction",
        "allowed_splits": ["train", "validation"],
        "include_official_test": False,
        "official_test_opened": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "artifact_count": len(artifact_audits),
        "historical_pair_fold_count": len(FOLDS),
        "selected_pair_count": len(SELECTED_FOLDS),
        "alternate_pair_count": len(ALTERNATE_FOLDS),
        "method_count": len(METHODS),
        "outcome_blind_pair_selection": False,
        "reason_not_outcome_blind": (
            "the six historical pair artifacts existed before their present appendix role was fixed"
        ),
        "scientific_role": (
            "appendix-only post-hoc targeted robustness with selected-versus-alternate "
            "matching sensitivity; no primary claim and no inferential p-values"
        ),
        "exact_result_reuse_approved": False,
        "metrics_only_appendix_reuse_approved": True,
        "artifacts": artifact_audits,
        "table": str(args.table_output.resolve()),
        "table_sha256": _sha256(args.table_output.resolve()),
    }
    _write_json(args.audit_output.resolve(), audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "artifacts"}, indent=2))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/experimental/taxonomy_two_stage_precloud_v2/validation"),
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/experiments/taxonomy_two_stage_precloud_v2.json"),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path(
            "docs/thesis_figure_data/taxonomy_scientific_freeze_v1/l3_posthoc_targeted_robustness.csv"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "docs/experiments/taxonomy_l3_posthoc_robustness_audit_20260822.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
