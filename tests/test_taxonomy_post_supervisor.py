from __future__ import annotations

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_post_supervisor import (
    evaluate_l2_condition,
    post_supervisor_l2_folds,
)


def _scored_grid() -> pd.DataFrame:
    fold = post_supervisor_l2_folds()[0]
    rows = []
    for row_uid, gold_aspect in (("v:1", fold.heldout_aspects[0]), ("v:2", None)):
        for aspect in fold.evaluation_aspects:
            aspect_score = 0.9 if aspect == gold_aspect else 0.1
            for sentiment, sentiment_score in zip(
                ("negative", "neutral", "positive"),
                (0.1, 0.2, 0.8),
            ):
                rows.append(
                    {
                        "row_uid": row_uid,
                        "candidate_aspect": aspect,
                        "candidate_sentiment": sentiment,
                        "candidate_text": f"{aspect} {sentiment}",
                        "representation_variant": "name_only",
                        "pair_label": f"{aspect} | {sentiment}",
                        "target": int(aspect == gold_aspect and sentiment == "positive"),
                        "aspect_score": aspect_score,
                        "sentiment_score": sentiment_score,
                        "score": aspect_score,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_post_supervisor_level2_has_twelve_ndr_folds() -> None:
    folds = post_supervisor_l2_folds()
    assert len(folds) == 12
    assert all(fold.conditions == ("N", "D", "R") for fold in folds)


def test_l2_views_share_one_score_contract_and_cap_sentiments() -> None:
    fold = post_supervisor_l2_folds()[0]
    result = evaluate_l2_condition(
        _scored_grid(),
        fold,
        aspect_threshold=0.5,
        second_sentiment_threshold=0.95,
    )
    assert len(result["score_sha256"]) == 64
    assert np.isclose(result["L2_S"]["aspect_presence"]["f1"], 1.0)
    assert result["L2_S"]["positive_review_rank"]["top_1_rate"] == 1.0
    assert result["L2_E"]["maximum_sentiments_per_selected_aspect"] == 1
    assert result["L2_E"]["partitions"]["heldout"]["pair_micro_f1"] == 1.0
