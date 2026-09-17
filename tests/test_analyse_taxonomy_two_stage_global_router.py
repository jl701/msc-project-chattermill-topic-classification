from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import analyse_taxonomy_two_stage_global_router as router_analysis


def _policy_rows() -> pd.DataFrame:
    rows = []
    for fold in router_analysis.L2_FOLDS:
        for condition in router_analysis.CONDITIONS:
            for policy, heldout, route_rate in (
                ("policy-a", 0.50 if condition == "D" else 0.10, 0.05),
                ("policy-b", 0.40 if condition == "D" else 0.99, 0.01),
            ):
                rows.append(
                    {
                        "policy_id": policy,
                        "base_method_id": "base",
                        "expert_method_id": "expert",
                        "rescue_band": 0.1,
                        "confirmation_band": 0.1,
                        "fold_id": fold,
                        "condition": condition,
                        "heldout_pair_micro_f1": heldout,
                        "overall_pair_micro_f1": heldout - 0.01,
                        "route_rate": route_rate,
                        "heldout_pair_tp": 1,
                        "heldout_pair_fp": 1,
                        "heldout_pair_fn": 1,
                        "overall_pair_tp": 1,
                        "overall_pair_fp": 1,
                        "overall_pair_fn": 1,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_crossfit_selects_on_d_only_and_transfers_policy_to_n() -> None:
    frame, summary = router_analysis._crossfit(_policy_rows())
    assert set(frame[frame["condition"].eq("D")]["policy_id"]) == {"policy-a"}
    assert set(frame[frame["condition"].eq("N")]["policy_id"]) == {"policy-a"}
    assert set(frame["selection_fold_count"]) == {11}
    assert summary["D"]["aspect_balanced_heldout_pair_micro_f1"] == 0.5
    assert summary["N"]["aspect_balanced_heldout_pair_micro_f1"] == pytest.approx(0.1)


def test_policy_tie_break_prefers_lower_route_rate_then_lexical_id() -> None:
    frame = _policy_rows()
    d_only = frame[frame["condition"].eq("D")].copy()
    d_only.loc[d_only["policy_id"].eq("policy-b"), "heldout_pair_micro_f1"] = 0.5
    d_only.loc[d_only["policy_id"].eq("policy-b"), "overall_pair_micro_f1"] = 0.49
    selected = router_analysis._select_policy(d_only)
    assert selected["policy_id"] == "policy-b"
