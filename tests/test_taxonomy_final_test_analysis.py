from __future__ import annotations

import pandas as pd
import pytest

from msc_project.experiments.taxonomy_final_test_analysis import (
    LABEL_COLUMNS,
    build_label_vault,
    synchronised_aspect_balanced_difference,
    validate_label_vault,
)

ASPECTS = tuple(f"Parent: aspect {index:02d}" for index in range(1, 13))


def test_label_vault_is_complete_and_contains_no_text() -> None:
    rows = pd.DataFrame(
        {
            "row_uid": ["test:1", "test:2"],
            "text": ["secret one", "secret two"],
            "labels": [[(ASPECTS[0], "positive")], []],
        }
    )
    vault = build_label_vault(rows, ASPECTS)
    assert tuple(vault.columns) == LABEL_COLUMNS
    assert "text" not in vault
    assert len(vault) == 72
    assert vault["target"].sum() == 1
    audit = validate_label_vault(
        vault,
        expected_row_uids=("test:1", "test:2"),
        expected_aspects=ASPECTS,
    )
    assert audit["status"] == "pass"

    broken = vault.drop(index=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="complete"):
        validate_label_vault(
            broken,
            expected_row_uids=("test:1", "test:2"),
            expected_aspects=ASPECTS,
        )


def _predictions(perfect: bool) -> pd.DataFrame:
    records = []
    for fold in ("l2-a01", "l2-a02"):
        for uid in ("test:1", "test:2", "test:3"):
            # Keep every resampled cluster informative.  This avoids a
            # degenerate all-negative bootstrap draw in the tiny fixture while
            # preserving the intended perfect-versus-zero paired contrast.
            gold = [f"{ASPECTS[0]} | positive"]
            predicted = list(gold) if perfect else []
            records.append(
                {
                    "fold_id": fold,
                    "row_uid": uid,
                    "gold_pairs": gold,
                    "pred_pairs": predicted,
                }
            )
    return pd.DataFrame.from_records(records)


def test_synchronised_bootstrap_uses_shared_reviews_and_aspect_balanced_f1() -> None:
    value = synchronised_aspect_balanced_difference(
        _predictions(True),
        _predictions(False),
        replicates=500,
        seed=13,
    )
    assert value["challenger_point"] == 1.0
    assert value["reference_point"] == 0.0
    assert value["point_difference"] == 1.0
    assert value["folds"] == 2
    assert value["review_clusters"] == 3
    assert value["superiority"] is True

    mismatched = _predictions(False)
    mismatched.loc[0, "row_uid"] = "test:wrong"
    with pytest.raises(ValueError, match="identical folds/reviews"):
        synchronised_aspect_balanced_difference(
            _predictions(True), mismatched, replicates=10, seed=13
        )
