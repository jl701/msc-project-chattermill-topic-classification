from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msc_project.experiments.duplicate_text_sensitivity import (
    FoldSensitivity,
    analyse_model_fold,
    assert_experimental_output_dir,
    compare_complete_models,
    duplicate_mask_against_training,
    load_validated_score_artifact,
    normalise_review_text,
    resolve_pair_score_path,
    validate_run_root,
)


ASPECT = "Company brand: Competitor"
SENTIMENTS = ("negative", "neutral", "positive")


def _eval_frame(split: str) -> pd.DataFrame:
    labels = [[], [f"{ASPECT} | negative"], [f"{ASPECT} | positive"]]
    return pd.DataFrame(
        {
            "row_uid": [f"{split}:2", f"{split}:1", f"{split}:3"],
            "text": ["Fresh row", " SAME\tText ", "Another fresh row"],
            "supervision_pair_labels": [labels[0], labels[1], labels[2]],
        }
    )


def _write_scores(path: Path, frame: pd.DataFrame, scores: np.ndarray) -> None:
    ordered = frame.assign(_uid=frame["row_uid"].astype(str)).sort_values("_uid").reset_index(drop=True)
    rows = []
    for row_index, row in ordered.iterrows():
        for sentiment_index, sentiment in enumerate(SENTIMENTS):
            rows.append(
                {
                    "row_index": row_index,
                    "row_uid": row["row_uid"],
                    "text": row["text"],
                    "candidate_aspect": ASPECT,
                    "candidate_sentiment": sentiment,
                    "target": int(f"{ASPECT} | {sentiment}" in row["supervision_pair_labels"]),
                    "variant": "enhanced",
                    "score": scores[row_index, sentiment_index],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_normalisation_and_duplicate_mask_are_conservative() -> None:
    assert normalise_review_text(" Ａ\nB  ") == "a b"
    train = pd.DataFrame({"text": ["Same text", "", "different"]})
    evaluation = pd.DataFrame({"text": [" SAME\tText ", "", "same-text"]})
    mask, counts = duplicate_mask_against_training(train, evaluation)
    assert mask.tolist() == [True, False, False]
    assert counts["same text"] == 1


def test_score_loader_rejects_row_identity_mismatch(tmp_path: Path) -> None:
    frame = _eval_frame("validation")
    path = tmp_path / "validation_pair_scores.csv"
    _write_scores(path, frame, np.full((3, 3), 0.2))
    scores = pd.read_csv(path)
    scores.loc[0, "row_uid"] = "validation:wrong"
    scores.to_csv(path, index=False)
    with pytest.raises(ValueError, match="row_uid mismatch"):
        load_validated_score_artifact(path, frame, ASPECT)


def test_pilot_validation_fallback_is_explicit(tmp_path: Path) -> None:
    full_root = tmp_path / "full"
    pilot_root = tmp_path / "pilot"
    pilot_path = pilot_root / "company_brand_competitor" / "validation_pair_scores.csv"
    pilot_path.parent.mkdir(parents=True)
    pilot_path.touch()
    assert resolve_pair_score_path(
        full_root=full_root,
        pilot_root=pilot_root,
        heldout_aspect=ASPECT,
        split="validation",
    ) == pilot_path
    with pytest.raises(ValueError, match="Missing full-run test"):
        resolve_pair_score_path(
            full_root=full_root,
            pilot_root=pilot_root,
            heldout_aspect=ASPECT,
            split="test",
        )


def test_saved_scores_are_rethresholded_after_duplicate_removal(tmp_path: Path) -> None:
    full_root = tmp_path / "full"
    pilot_root = tmp_path / "pilot"
    validation = _eval_frame("validation")
    test = _eval_frame("test")
    validation_scores = np.asarray(
        [
            [0.40, 0.05, 0.05],
            [0.05, 0.05, 0.05],
            [0.05, 0.05, 0.80],
        ]
    )
    test_scores = np.asarray(
        [
            [0.85, 0.05, 0.05],
            [0.05, 0.05, 0.05],
            [0.05, 0.05, 0.85],
        ]
    )
    _write_scores(
        pilot_root / "company_brand_competitor" / "validation_pair_scores.csv",
        validation,
        validation_scores,
    )
    _write_scores(
        full_root / "company_brand_competitor" / "test_pair_scores.csv",
        test,
        test_scores,
    )
    result = analyse_model_fold(
        model_name="frozen",
        heldout_aspect=ASPECT,
        train_frame=pd.DataFrame({"text": ["same text"]}),
        validation_frame=validation,
        test_frame=test,
        full_root=full_root,
        pilot_root=pilot_root,
    )
    assert result.result["validation_duplicate_rows_removed"] == 1
    assert result.result["test_duplicate_rows_removed"] == 1
    assert result.result["validation_gold_present_rows_removed"] == 1
    assert result.result["validation_rows_retained"] == 2
    assert result.result["retained_selected_threshold"] > result.result["original_selected_threshold"]
    assert result.result["retained_test_reselected_pair_micro_f1"] == pytest.approx(1.0)
    assert set(result.duplicate_rows["split"]) == {"validation", "test"}


def test_output_directory_is_restricted_to_experimental_tree(tmp_path: Path) -> None:
    project = tmp_path / "project"
    allowed = project / "outputs" / "experimental" / "sensitivity"
    assert assert_experimental_output_dir(allowed, project) == allowed.resolve()
    with pytest.raises(ValueError, match="Output must be under"):
        assert_experimental_output_dir(project / "docs" / "sensitivity", project)


def test_run_root_provenance_rejects_wrong_mode(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_text(
        """{
          "protocol_id": "loao_unified_candidate_pair_experimental_v1",
          "mode": "qlora",
          "stage": "pilot",
          "variant": "enhanced",
          "folds": ["Company brand: Competitor"],
          "finished_at": "2026-07-18T00:00:00"
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mode must be 'frozen'"):
        validate_run_root(
            root,
            expected_mode="frozen",
            expected_stage="pilot",
            expected_folds=["Company brand: Competitor"],
            require_finished=True,
        )


def test_complete_comparison_is_descriptive_and_requires_twelve_folds() -> None:
    aspects = [f"Aspect {index}" for index in range(12)]

    def fold(aspect: str, model: str, f1: float) -> FoldSensitivity:
        result: dict[str, object] = {
            "heldout_aspect": aspect,
            "model": model,
            "retained_test_reselected_pair_micro_f1": f1,
        }
        for split in ("validation", "test"):
            result.update(
                {
                    f"{split}_score_identity_sha256": f"{aspect}-{split}",
                    f"{split}_rows_original": 10,
                    f"{split}_duplicate_rows_removed": 1,
                    f"{split}_rows_retained": 9,
                }
            )
        empty = pd.DataFrame()
        return FoldSensitivity(result, empty, empty, empty)

    frozen = [fold(aspect, "frozen", 0.3) for aspect in aspects]
    qlora = [fold(aspect, "qlora", 0.4) for aspect in aspects]
    comparison = compare_complete_models(frozen, qlora, aspects)
    statistics = comparison["paired_descriptive_statistics"]
    assert statistics["mean_delta"] == pytest.approx(0.1)
    assert statistics["wins"] == 12
    assert statistics["inferential_test_performed"] is False
    with pytest.raises(ValueError, match="12 unique"):
        compare_complete_models(frozen[:1], qlora[:1], aspects[:1])
