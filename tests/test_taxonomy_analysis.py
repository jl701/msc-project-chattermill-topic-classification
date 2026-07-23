from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_analysis import (
    assert_shared_seen_threshold,
    level3_condition_diagnostics,
    per_aspect_metrics,
    per_sentiment_metrics,
)
from msc_project.experiments.taxonomy_protocol import registered_folds


def l3_scored_grid(condition: str = "DN") -> pd.DataFrame:
    fold = registered_folds("L3")[0]
    rows = []
    gold = {
        "both": {
            (fold.heldout_aspects[0], "positive"),
            (fold.heldout_aspects[1], "negative"),
        },
        "name_only": {(fold.heldout_aspects[1], "positive")},
        "neither": set(),
    }
    represented = {
        fold.heldout_aspects[0]: "minimal" if condition[0] == "D" else "name_only",
        fold.heldout_aspects[1]: "minimal" if condition[1] == "D" else "name_only",
    }
    for row_index, (row_uid, gold_pairs) in enumerate(gold.items()):
        for aspect in fold.heldout_aspects:
            for sentiment in ("negative", "neutral", "positive"):
                pair = (aspect, sentiment)
                score = 0.1
                if pair in gold_pairs:
                    score = 0.9
                # Deliberately select the described A aspect on the name-only-only
                # row to expose the asymmetric false-positive diagnostic.
                if (
                    condition == "DN"
                    and row_uid == "name_only"
                    and aspect == fold.heldout_aspects[0]
                    and sentiment == "positive"
                ):
                    score = 0.8
                rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "condition": condition,
                        "row_index": row_index,
                        "row_uid": row_uid,
                        "candidate_aspect": aspect,
                        "candidate_sentiment": sentiment,
                        "pair_label": f"{aspect} | {sentiment}",
                        "representation_variant": represented[aspect],
                        "is_seen": False,
                        "is_heldout": True,
                        "target": int(pair in gold_pairs),
                        "score": score,
                    }
                )
    return pd.DataFrame(rows)


def test_per_aspect_and_sentiment_tables_keep_denominators() -> None:
    frame = l3_scored_grid()
    aspects = per_aspect_metrics(frame, 0.5)
    sentiments = per_sentiment_metrics(frame, 0.5)
    assert len(aspects) == 2
    assert set(aspects["examples"]) == {3}
    assert tuple(sentiments["candidate_sentiment"]) == (
        "negative",
        "neutral",
        "positive",
    )
    assert set(sentiments["candidate_pairs"]) == {6}


def test_level3_asymmetric_diagnostics_cover_both_only_and_neither_cases() -> None:
    fold = registered_folds("L3")[0]
    result = level3_condition_diagnostics(l3_scored_grid(), fold, "DN", 0.5)
    assert result["both_gold_rows"] == 1
    assert result["both_present_recall"] == pytest.approx(1.0)
    assert result["only_name_only_gold_rows"] == 1
    assert result["name_only_recall_when_only_name_only_is_gold"] == pytest.approx(1.0)
    assert result[
        "described_false_positive_rate_when_only_name_only_is_gold"
    ] == pytest.approx(1.0)
    assert result["neither_gold_rows"] == 1
    assert result["neither_false_positive_rate"] == pytest.approx(0.0)
    assert len(result["heldout_per_aspect"]) == 2


def test_symmetric_conditions_do_not_invent_asymmetric_bias() -> None:
    fold = registered_folds("L3")[0]
    result = level3_condition_diagnostics(
        l3_scored_grid("DD"),
        fold,
        "DD",
        0.5,
    )
    assert result["described_minus_name_only_selection_rate"] is None
    assert result["only_name_only_gold_rows"] is None


def test_threshold_guard_requires_same_value_and_seen_set() -> None:
    selections = {
        "NN": SimpleNamespace(threshold=0.4, calibration_aspects=("a", "b")),
        "DN": SimpleNamespace(threshold=0.4, calibration_aspects=("a", "b")),
    }
    assert assert_shared_seen_threshold(selections) == pytest.approx(0.4)
    selections["DN"] = SimpleNamespace(
        threshold=0.5,
        calibration_aspects=("a", "b"),
    )
    with pytest.raises(ValueError, match="thresholds differ"):
        assert_shared_seen_threshold(selections)
