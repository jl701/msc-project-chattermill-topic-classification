from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/build_taxonomy_two_stage_no_retraining_extension_summary.py"
)
SPEC = importlib.util.spec_from_file_location("extension_summary", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _aggregate(method: str, role: str, d_value: float) -> pd.DataFrame:
    records = []
    for condition, heldout in (("D", d_value), ("N", d_value - 0.01)):
        records.append(
            {
                "method_id": method,
                "evidence_role": role,
                "condition": condition,
                "heldout_pair_micro_f1_mean": heldout,
                "overall_pair_micro_f1_mean": 0.6,
                "heldout_pair_precision_mean": 0.5,
                "heldout_pair_recall_mean": 0.6,
                "heldout_presence_f1_mean": 0.55,
                "heldout_presence_ap_mean": 0.58,
                "oracle_gated_sentiment_set_micro_f1_mean": 0.88,
                "aspect_call_rate_mean": 0.17,
            }
        )
    return pd.DataFrame(records)


def _pairwise(method: str, role: str, gain: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "condition": "D",
                "left_bootstrap_method_id": f"{method}__{role}",
                "right_bootstrap_method_id": (
                    f"{MODULE.REFERENCE_ID}__fixed_reference_reproduction"
                ),
                "left_minus_right_aspect_balanced_heldout_pair_micro_f1": gain,
                "shared_review_bootstrap_95_ci_low": gain - 0.02,
                "shared_review_bootstrap_95_ci_high": gain + 0.02,
            }
        ]
    )


def test_build_tables_ranks_and_applies_registered_promotion_rule() -> None:
    reference = _aggregate(
        MODULE.REFERENCE_ID, "fixed_reference_reproduction", 0.52
    )
    smooth = pd.concat(
        [
            reference,
            _aggregate(MODULE.SMOOTH_ID, "review_cross_fitted_primary", 0.50),
        ],
        ignore_index=True,
    )
    retrieval = pd.concat(
        [
            reference,
            _aggregate(MODULE.RETRIEVAL_ID, "review_cross_fitted_primary", 0.55),
        ],
        ignore_index=True,
    )
    primary, transfer = MODULE.build_tables(
        smooth_aggregate=smooth,
        retrieval_aggregate=retrieval,
        smooth_pairwise=_pairwise(
            MODULE.SMOOTH_ID, "review_cross_fitted_primary", -0.02
        ),
        retrieval_pairwise=_pairwise(
            MODULE.RETRIEVAL_ID, "review_cross_fitted_primary", 0.03
        ),
    )
    assert primary["method_id"].tolist() == [
        MODULE.RETRIEVAL_ID,
        MODULE.REFERENCE_ID,
        MODULE.SMOOTH_ID,
    ]
    assert primary["rank_by_D_heldout_pair_f1"].tolist() == [1, 2, 3]
    decisions = primary.set_index("method_id")["decision"].to_dict()
    assert decisions[MODULE.RETRIEVAL_ID] == "promote"
    assert decisions[MODULE.SMOOTH_ID] == "do_not_promote"
    assert len(transfer) == 6
    assert int(transfer["test_contract_count"].sum()) == 0


def test_build_tables_rejects_reference_mismatch() -> None:
    smooth_reference = _aggregate(
        MODULE.REFERENCE_ID, "fixed_reference_reproduction", 0.52
    )
    retrieval_reference = _aggregate(
        MODULE.REFERENCE_ID, "fixed_reference_reproduction", 0.51
    )
    smooth = pd.concat(
        [
            smooth_reference,
            _aggregate(MODULE.SMOOTH_ID, "review_cross_fitted_primary", 0.50),
        ],
        ignore_index=True,
    )
    retrieval = pd.concat(
        [
            retrieval_reference,
            _aggregate(MODULE.RETRIEVAL_ID, "review_cross_fitted_primary", 0.55),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="Reference reproduction differs"):
        MODULE.build_tables(
            smooth_aggregate=smooth,
            retrieval_aggregate=retrieval,
            smooth_pairwise=_pairwise(
                MODULE.SMOOTH_ID, "review_cross_fitted_primary", -0.02
            ),
            retrieval_pairwise=_pairwise(
                MODULE.RETRIEVAL_ID, "review_cross_fitted_primary", 0.03
            ),
        )


def _audit(*, retrieval_violation: int = 0) -> dict[str, object]:
    return {
        "receipt_summary": {
            "worker": {
                "receipt_count": 2,
                "unique_verified_file_count": 3,
            }
        },
        "critical_verified_input_file_count": 4,
        "selection_count": 5,
        "crossfit_fit_count": 6,
        "retrieval_inputs": {
            "retrieval_manifest_count": 24,
            "retrieval_score_unit_count": 24,
            "demonstration_heldout_aspect_count": retrieval_violation,
        },
        "crossfit_train_evaluation_review_overlap_count": 0,
        "heldout_aspect_selection_instance_count": 0,
        "maximum_sentiments_per_selected_aspect": 2,
        "duplicate_row_evidence_identity_count": 0,
        "prediction_collapse_count": 0,
        "non_finite_value_count": 0,
        "official_test_opened": False,
        "test_contract_count": 0,
        "promotion_rule_passed": False,
        "audit_payload_sha256": "a" * 64,
    }


def test_build_audit_table_summarises_and_rejects_violations() -> None:
    frame = MODULE.build_audit_table(_audit(), _audit())
    assert len(frame) == 2
    assert frame["formal_receipt_count"].tolist() == [2, 2]
    assert int(frame["retrieval_contract_violation_count"].sum()) == 0
    with pytest.raises(ValueError, match="audit contains a violation"):
        MODULE.build_audit_table(_audit(), _audit(retrieval_violation=1))
