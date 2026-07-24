from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SCRIPT = PROJECT_ROOT / "scripts" / "analyse_taxonomy_primary_comparisons.py"
SPEC = importlib.util.spec_from_file_location(
    "analyse_taxonomy_primary_comparisons",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from msc_project.experiments.taxonomy_comparisons import ProtocolResult
from msc_project.experiments.taxonomy_protocol import registered_folds


def test_primary_endpoint_conditions_match_the_locked_curve() -> None:
    assert MODULE.ENDPOINT_CONDITIONS == {
        "L1": "D",
        "L2": "D",
        "L3": "DD",
        "L4": "D",
    }


def _seed_result(unit_id: str, positive_score: float) -> ProtocolResult:
    fold = registered_folds("L1")[0]
    aspect = fold.heldout_aspects[0]
    rows = []
    for row_uid in ("r1", "r2"):
        for sentiment in ("negative", "neutral", "positive"):
            rows.append(
                {
                    "row_uid": row_uid,
                    "candidate_aspect": aspect,
                    "candidate_sentiment": sentiment,
                    "pair_label": f"{aspect} | {sentiment}",
                    "target": int(row_uid == "r1" and sentiment == "positive"),
                    "score": positive_score if sentiment == "positive" else 0.0,
                    "representation_variant": "minimal",
                    "is_seen": False,
                    "is_heldout": True,
                }
            )
    return ProtocolResult(
        unit_id,
        fold,
        "D",
        pd.DataFrame.from_records(rows),
        0.5,
    )


def test_seed_sensitivity_keeps_review_and_seed_uncertainty_separate() -> None:
    values = {
        13: [_seed_result("fold-1", 0.9)],
        23: [_seed_result("fold-1", 0.8)],
        42: [_seed_result("fold-1", 0.1)],
    }
    result = MODULE.build_level1_qlora_seed_sensitivity(
        values,
        replicates=50,
        bootstrap_seed=13,
    )
    assert result["training_seeds"] == [13, 23, 42]
    assert len(result["per_fold_across_seed_summary"]) == 1
    assert len(result["paired_seed_sensitivity_vs_seed13"]) == 2
