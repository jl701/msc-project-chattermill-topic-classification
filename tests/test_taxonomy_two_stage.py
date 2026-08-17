from __future__ import annotations

import numpy as np
import pandas as pd

from msc_project.data.fabsa import format_pair_label
from msc_project.experiments.taxonomy_two_stage import (
    capped_two_sentiment_prediction_mask,
    crossfit_decoder_comparison,
    hierarchical_candidates,
    hierarchical_prediction_mask,
    multi_sentiment_prediction_mask,
    select_multi_sentiment_thresholds,
    select_second_sentiment_threshold,
    select_top_k_aspect_threshold,
    select_two_stage_threshold,
    top_k_sentiment_prediction_mask,
    two_stage_candidates,
    two_stage_prediction_mask,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
    render_aspect_candidate,
    select_qwen_two_stage_demonstrations,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS


ASPECTS = ("Group: A", "Group: B")


def _grid(rows: int = 15) -> pd.DataFrame:
    records = []
    for row in range(rows):
        uid = f"validation:{row:03d}"
        gold_aspect = ASPECTS[row % 2]
        gold_sentiment = CANDIDATE_SENTIMENTS[row % 3]
        for aspect in ASPECTS:
            for sentiment in CANDIDATE_SENTIMENTS:
                target = int(aspect == gold_aspect and sentiment == gold_sentiment)
                score = 0.9 if target else 0.1
                records.append(
                    {
                        "row_uid": uid,
                        "candidate_aspect": aspect,
                        "candidate_sentiment": sentiment,
                        "pair_label": format_pair_label(aspect, sentiment),
                        "target": target,
                        "score": score,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_hierarchical_candidates_choose_one_sentiment_per_aspect() -> None:
    frame = _grid(2)
    candidates = hierarchical_candidates(frame)
    assert len(candidates) == 2 * len(ASPECTS)
    assert not candidates.duplicated(["row_uid", "candidate_aspect"]).any()


def test_hierarchical_prediction_mask_is_aspect_exclusive() -> None:
    frame = _grid(2)
    mask = hierarchical_prediction_mask(frame, 0.5)
    selected = frame[mask]
    assert not selected.duplicated(["row_uid", "candidate_aspect"]).any()
    assert len(selected) == 2


def test_crossfit_decoder_comparison_covers_every_row() -> None:
    result = crossfit_decoder_comparison(_grid(100), aspects=ASPECTS, folds=5)
    assert result["pair_decoder"]["pair_micro_f1"] == 1.0
    assert result["hierarchical_decoder"]["pair_micro_f1"] == 1.0
    assert len(result["thresholds"]) == 5


def test_true_two_stage_uses_independent_aspect_and_sentiment_scores() -> None:
    frame = _grid(4)
    frame["aspect_score"] = frame["score"]
    frame["sentiment_score"] = np.where(
        frame["candidate_sentiment"].eq("positive"), 0.9, 0.1
    )
    candidates = two_stage_candidates(frame)
    assert set(candidates["candidate_sentiment"]) == {"positive"}
    selection = select_two_stage_threshold(frame)
    mask = two_stage_prediction_mask(frame, selection.threshold)
    assert mask.dtype == bool
    assert int(mask.sum()) <= frame[
        ["row_uid", "candidate_aspect"]
    ].drop_duplicates().shape[0]


def test_multi_sentiment_decoder_uses_threshold_and_argmax_fallback() -> None:
    frame = _grid(2)
    frame["aspect_score"] = 0.9
    frame["sentiment_score"] = [0.8, 0.1, 0.7, 0.2, 0.3, 0.4] * 2
    multi = multi_sentiment_prediction_mask(
        frame,
        aspect_threshold=0.5,
        sentiment_threshold=0.6,
    )
    counts = frame.loc[multi].groupby(["row_uid", "candidate_aspect"]).size()
    assert int(counts.max()) == 2
    fallback = multi_sentiment_prediction_mask(
        frame,
        aspect_threshold=0.5,
        sentiment_threshold=2.0,
    )
    fallback_counts = frame.loc[fallback].groupby(
        ["row_uid", "candidate_aspect"]
    ).size()
    assert fallback_counts.eq(1).all()


def test_multi_sentiment_selection_can_recover_two_gold_sentiments() -> None:
    frame = _grid(20)
    frame["aspect_score"] = 0.9
    frame["target"] = 0
    frame["sentiment_score"] = 0.05
    for (uid, aspect), indices in frame.groupby(
        ["row_uid", "candidate_aspect"], sort=False
    ).groups.items():
        positions = list(indices)
        frame.loc[positions[0], ["target", "sentiment_score"]] = [1, 0.9]
        frame.loc[positions[2], ["target", "sentiment_score"]] = [1, 0.8]
    selected = select_multi_sentiment_thresholds(frame, sentiment_quantiles=9)
    control = select_two_stage_threshold(frame)
    assert selected.sentiment_threshold is not None
    assert selected.selection_metrics["pair_micro_f1"] >= float(
        control.sweep["pair_micro_f1"].max()
    )
    mask = multi_sentiment_prediction_mask(
        frame,
        aspect_threshold=selected.aspect_threshold,
        sentiment_threshold=selected.sentiment_threshold,
    )
    assert int(mask.sum()) == int(frame["target"].sum())


def test_fixed_top_two_decoder_emits_two_sentiments_per_selected_aspect() -> None:
    frame = _grid(5)
    frame["aspect_score"] = frame.groupby(
        ["row_uid", "candidate_aspect"]
    )["score"].transform("max")
    frame["sentiment_score"] = frame["score"]
    selected = select_top_k_aspect_threshold(frame, top_k=2)
    mask = top_k_sentiment_prediction_mask(
        frame,
        aspect_threshold=selected.aspect_threshold,
        top_k=2,
    )
    counts = frame.loc[mask].groupby(["row_uid", "candidate_aspect"]).size()
    assert counts.eq(2).all()


def test_capped_two_decoder_never_emits_a_third_sentiment() -> None:
    frame = _grid(2)
    frame["aspect_score"] = 0.9
    frame["sentiment_score"] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4] * 2
    mask = capped_two_sentiment_prediction_mask(
        frame,
        aspect_threshold=0.5,
        second_sentiment_threshold=0.45,
    )
    counts = frame.loc[mask].groupby(["row_uid", "candidate_aspect"]).size()
    assert counts.between(1, 2).all()
    assert int(counts.max()) == 2


def test_second_sentiment_selection_freezes_aspect_threshold_and_nests_argmax() -> None:
    frame = _grid(20)
    frame["aspect_score"] = 0.9
    frame["target"] = 0
    frame["sentiment_score"] = 0.05
    for indices in frame.groupby(
        ["row_uid", "candidate_aspect"], sort=False
    ).groups.values():
        positions = list(indices)
        frame.loc[positions[0], ["target", "sentiment_score"]] = [1, 0.9]
        frame.loc[positions[2], ["target", "sentiment_score"]] = [1, 0.8]
    aspect_threshold = 0.5
    selected = select_second_sentiment_threshold(
        frame,
        aspect_threshold=aspect_threshold,
    )
    assert selected.aspect_threshold == aspect_threshold
    mask = capped_two_sentiment_prediction_mask(
        frame,
        aspect_threshold=aspect_threshold,
        second_sentiment_threshold=selected.second_sentiment_threshold,
    )
    counts = frame.loc[mask].groupby(["row_uid", "candidate_aspect"]).size()
    assert counts.eq(2).all()
    assert int(mask.sum()) == int(frame["target"].sum())


def test_two_stage_grid_variants_and_score_join() -> None:
    aspects = ("Parent: Alpha", "Parent: Beta")
    resource = {
        "minimal_aspects": {
            "Parent: Alpha": "Alpha definition.",
            "Parent: Beta": "Beta definition.",
        }
    }
    rows = pd.DataFrame(
        {
            "row_uid": ["validation:1"],
            "text": ["Alpha was positive"],
            "labels": [[("Parent: Alpha", "positive")]],
        }
    )
    variants = {
        "Parent: Alpha": "name_and_description",
        "Parent: Beta": "description_only",
    }
    assert "Parent: Beta" not in render_aspect_candidate(
        "Parent: Beta", "description_only", resource
    )
    aspect_grid = build_aspect_grid(rows, aspects, variants, resource)
    sentiment_grid = build_sentiment_grid(rows, aspects, variants, resource)
    scored = join_two_stage_scores(
        aspect_grid,
        sentiment_grid,
        np.linspace(0.1, 0.9, len(aspect_grid)),
        np.linspace(0.1, 0.9, len(sentiment_grid)),
    )
    assert len(aspect_grid) == 2
    assert len(sentiment_grid) == 6
    assert scored["aspect_score"].notna().all()


def test_rich_aspect_card_is_stage_aware_and_has_no_sentiment_answer() -> None:
    resource = {
        "minimal_aspects": {"Parent: Alpha": "Alpha definition."},
        "rich_aspects": {
            "Parent: Alpha": {
                "definition": "Alpha definition.",
                "aliases": ["alpha one", "alpha two", "alpha three"],
                "inclusion_boundary": "Include alpha topics.",
                "contrastive_boundary": "Exclude beta topics.",
            }
        },
    }
    rendered = render_aspect_candidate("Parent: Alpha", "rich", resource)
    assert "Aliases:" in rendered
    assert "Inclusion boundary:" in rendered
    assert "Contrastive boundary:" in rendered
    assert "Candidate sentiment" not in rendered
    assert all(value not in rendered for value in CANDIDATE_SENTIMENTS)


def test_few_shot_demonstrations_are_train_only_balanced_and_reproducible() -> None:
    aspects = ("Parent: Alpha", "Parent: Beta", "Parent: Gamma")
    resource = {
        "minimal_aspects": {aspect: f"Definition for {aspect}." for aspect in aspects}
    }
    rows = pd.DataFrame(
        {
            "row_uid": [f"train:{index}" for index in range(9)],
            "text": [f"review text {index}" for index in range(9)],
            "supervision_labels": [
                [(aspects[index % 3], CANDIDATE_SENTIMENTS[index % 3])]
                for index in range(9)
            ],
        }
    )
    first = select_qwen_two_stage_demonstrations(rows, aspects, resource, seed=13)
    second = select_qwen_two_stage_demonstrations(
        rows.sample(frac=1.0, random_state=99), aspects, resource, seed=13
    )
    assert first == second
    assert [value.answer for value in first["aspect"]] == ["Y", "Y", "N", "N"]
    assert [value.answer for value in first["sentiment"]] == ["A", "B", "C"]
    assert all(value.row_uid.startswith("train:") for values in first.values() for value in values)
    assert all(value.candidate_aspect in aspects for values in first.values() for value in values)
