from __future__ import annotations

import numpy as np
import pandas as pd

from msc_project.data.fabsa import format_pair_label
from msc_project.experiments.taxonomy_two_stage import (
    crossfit_decoder_comparison,
    hierarchical_candidates,
    hierarchical_prediction_mask,
    select_two_stage_threshold,
    two_stage_candidates,
    two_stage_prediction_mask,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
    render_aspect_candidate,
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
