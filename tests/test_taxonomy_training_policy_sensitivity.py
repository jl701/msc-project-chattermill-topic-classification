import json
from types import SimpleNamespace

import pandas as pd
import pytest

from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    training_rows, receipt, verify_receipt, select_seen, write_json,
)


def fixture():
    fold = SimpleNamespace(fold_id="l2-a01", heldout_aspects=("A",), seen_aspects=("B", "C"))
    rows = pd.DataFrame({"row_uid": ["train:1", "train:2", "train:3", "train:4"],
                         "text": ["ab", "a", "b", "c"], "original_split": ["train"] * 4,
                         "labels": [[("A", "positive"), ("B", "negative")], [("A", "negative")],
                                    [("B", "positive")], [("C", "neutral")]],
                         "labels_json": ["raw A"] * 4})
    return fold, rows


def test_mask_preserves_target_only_and_removes_all_annotation_channels():
    fold, rows = fixture()
    masked = training_rows(rows, fold, "label_masked_all_reviews")
    assert len(masked) == 4
    assert masked.loc[masked.row_uid.eq("train:2"), "supervision_labels"].iloc[0] == []
    assert "labels_json" not in masked
    assert all(a != "A" for labels in masked.labels for a, _ in labels)
    assert rows.labels.iloc[0][0][0] == "A"  # no in-place mutation


def test_filtered_removes_whole_mixed_and_target_only_rows():
    fold, rows = fixture()
    assert training_rows(rows, fold, "review_filtered").row_uid.tolist() == ["train:3", "train:4"]


def test_size_match_is_exact_deterministic_and_seed_whitelisted():
    fold, rows = fixture()
    for seed in (13, 29, 47):
        a = training_rows(rows, fold, "size_matched_label_masked", seed)
        b = training_rows(rows.iloc[::-1], fold, "size_matched_label_masked", seed)
        assert len(a) == 2
        assert set(a.row_uid) == set(b.row_uid)
    with pytest.raises(ValueError):
        training_rows(rows, fold, "size_matched_label_masked", 99)


def test_reject_validation_in_training():
    fold, rows = fixture()
    rows.loc[0, "original_split"] = "validation"
    with pytest.raises(ValueError):
        training_rows(rows, fold, "label_masked_all_reviews")


def test_selection_rejects_heldout_before_computing_metrics():
    fold, _ = fixture()
    with pytest.raises(ValueError):
        select_seen(pd.DataFrame({"candidate_aspect": ["A", "B", "C"]}), fold)


def test_receipt_detects_content_and_contract_conflict(tmp_path):
    (tmp_path / "result.txt").write_text("original")
    receipt(tmp_path, {"policy": "masked"})
    verify_receipt(tmp_path, {"policy": "masked"})
    with pytest.raises(ValueError):
        verify_receipt(tmp_path, {"policy": "filtered"})
    (tmp_path / "result.txt").write_text("tampered")
    with pytest.raises(ValueError):
        verify_receipt(tmp_path)


def test_negative_training_targets_never_include_heldout():
    from msc_project.experiments.taxonomy_two_stage_runtime import build_aspect_grid
    fold, rows = fixture()
    masked = training_rows(rows, fold, "label_masked_all_reviews")
    resource = {"aspects": {"B": {"name": "B", "description": "B definition"},
                            "C": {"name": "C", "description": "C definition"}}}
    # Render with name-only to test supervision independently of resource schemas.
    grid = build_aspect_grid(masked, fold.seen_aspects, {a: "name_only" for a in fold.seen_aspects}, resource)
    assert set(grid.candidate_aspect) == {"B", "C"}
    assert grid.loc[grid.row_uid.eq("train:2"), "target"].sum() == 0
    assert len(grid) == 8


def test_threshold_dataframe_serialisation_and_nonfinite_rejection(tmp_path):
    import numpy as np
    write_json(tmp_path / "selection.json", {"grid": pd.DataFrame({"threshold": [0.5], "f1": [0.3]}),
                                             "count": np.int64(7)})
    assert json.loads((tmp_path / "selection.json").read_text())["grid"][0]["f1"] == 0.3
    with pytest.raises(ValueError):
        write_json(tmp_path / "nonfinite.json", {"score": float("nan")})
