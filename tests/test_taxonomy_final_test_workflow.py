from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msc_project.experiments.taxonomy_final_test import (
    FIXED_COMPOSITION_METHOD,
    FinalTestJob,
)
from msc_project.experiments.taxonomy_final_test_workflow import (
    SCORE_COLUMNS,
    build_unlabelled_grids,
    canonical_l2_nd_scores,
    compose_fixed_stage_scores,
    decode_capped_two,
    read_score_bundle,
    validate_score_frame,
    write_score_bundle,
)
from msc_project.experiments.taxonomy_protocol import TaxonomyFold

ASPECTS = tuple(f"Parent: aspect {index:02d}" for index in range(1, 13))
SENTIMENTS = ("negative", "neutral", "positive")


def test_unlabelled_sentiment_grid_satisfies_pair_scorer_contract() -> None:
    aspects = ("Parent: aspect 01", "Parent: aspect 02")
    fold = TaxonomyFold(
        level="L2",
        fold_id="l2-a01",
        heldout_aspects=(aspects[0],),
        seen_aspects=(aspects[1],),
        evaluation_aspects=aspects,
        evaluation_label_scope="all_candidates",
        conditions=("D", "N"),
    )
    reviews = pd.DataFrame(
        {"row_uid": ["test:1", "test:2"], "text": ["first", "second"]}
    )
    resource = {
        "minimal_aspects": {
            aspects[0]: "The first aspect.",
            aspects[1]: "The second aspect.",
        }
    }

    _, sentiment_grid = build_unlabelled_grids(reviews, fold, "D", resource)

    assert set(sentiment_grid["negative_type"]) == {"eval_candidate"}
    assert "target" not in sentiment_grid
    assert len(sentiment_grid) == len(reviews) * len(aspects) * len(SENTIMENTS)


def job(method: str = "frozen_qwen_few_shot") -> FinalTestJob:
    return FinalTestJob(
        job_id=f"score-L2-l2-a01-{method}",
        phase="score",
        role="test",
        method_id=method,
        level="L2",
        fold_id="l2-a01",
        training_scope_id="heldout-a01",
        conditions=("D", "N"),
    )


def condition_frame(value: FinalTestJob, condition: str, *, heldout_only: bool = False) -> pd.DataFrame:
    records = []
    for row_index, uid in enumerate(("test:1", "test:2")):
        for aspect_index, aspect in enumerate(ASPECTS):
            heldout = aspect_index == 0
            if heldout_only and not heldout:
                continue
            for sentiment_index, sentiment in enumerate(SENTIMENTS):
                aspect_score = 0.2 + 0.03 * aspect_index + 0.05 * row_index
                sentiment_score = 0.1 + 0.2 * sentiment_index + 0.01 * aspect_index
                if heldout and condition == "N":
                    aspect_score += 0.07
                    sentiment_score += 0.02
                records.append(
                    {
                        "job_id": value.job_id,
                        "method_id": value.method_id,
                        "level": value.level,
                        "fold_id": value.fold_id,
                        "training_scope_id": value.training_scope_id,
                        "condition": condition,
                        "row_uid": uid,
                        "candidate_aspect": aspect,
                        "candidate_sentiment": sentiment,
                        "representation_variant": (
                            "name_only" if heldout and condition == "N" else "minimal"
                        ),
                        "is_seen": not heldout,
                        "is_heldout": heldout,
                        "aspect_score": aspect_score,
                        "sentiment_score": sentiment_score,
                    }
                )
    return pd.DataFrame.from_records(records, columns=SCORE_COLUMNS)


def complete_frame(value: FinalTestJob) -> pd.DataFrame:
    return canonical_l2_nd_scores(
        value,
        condition_frame(value, "D"),
        condition_frame(value, "N", heldout_only=True),
    )


def test_canonical_nd_and_score_contract() -> None:
    value = job()
    frame = complete_frame(value)
    audit = validate_score_frame(frame, value, expected_row_uids=("test:1", "test:2"))
    assert audit["rows"] == 144
    assert audit["non_finite_value_count"] == 0

    broken = frame.copy()
    first_seen = broken[(broken["condition"].eq("N")) & broken["is_seen"]].iloc[0]
    mask = (
        broken["condition"].eq("N")
        & broken["row_uid"].eq(first_seen["row_uid"])
        & broken["candidate_aspect"].eq(first_seen["candidate_aspect"])
    )
    broken.loc[mask, "aspect_score"] += 0.001
    with pytest.raises(ValueError, match="bitwise shared"):
        validate_score_frame(broken, value)


def test_score_contract_rejects_labels_nonfinite_duplicates_and_incomplete_grid() -> None:
    value = job()
    frame = complete_frame(value)
    labelled = frame.assign(target=0)
    with pytest.raises(ValueError, match="label-free"):
        validate_score_frame(labelled, value)

    nonfinite = frame.copy()
    nonfinite.loc[0, "sentiment_score"] = np.nan
    with pytest.raises(ValueError, match="invalid probability"):
        validate_score_frame(nonfinite, value)

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_score_frame(duplicate, value)

    with pytest.raises(ValueError, match="three sentiments"):
        validate_score_frame(frame.drop(index=frame.index[0]).reset_index(drop=True), value)


def test_decoder_never_emits_three_sentiments() -> None:
    frame = complete_frame(job())
    prediction = decode_capped_two(
        frame, aspect_threshold=0.0, runner_up_threshold=0.0
    )
    counts = (
        frame.assign(prediction=prediction)
        .groupby(["condition", "row_uid", "candidate_aspect"])["prediction"]
        .sum()
    )
    assert counts.max() == 2
    assert counts.min() == 2


def test_fixed_composition_takes_fewshot_stage1_and_qlora_stage2() -> None:
    fewshot_job = job("frozen_qwen_few_shot")
    qlora_job = job("qwen_candidate_pair_qlora")
    fewshot = complete_frame(fewshot_job)
    qlora = complete_frame(qlora_job)
    qlora["sentiment_score"] = 1.0 - qlora["sentiment_score"]
    hybrid_job = FinalTestJob(
        job_id=f"compose-L2-l2-a01-{FIXED_COMPOSITION_METHOD}",
        phase="compose",
        role="confirmatory_fixed_composition",
        method_id=FIXED_COMPOSITION_METHOD,
        level="L2",
        fold_id="l2-a01",
        training_scope_id="heldout-a01",
        conditions=("D", "N"),
        depends_on=(fewshot_job.job_id, qlora_job.job_id),
    )
    hybrid = compose_fixed_stage_scores(hybrid_job, fewshot, qlora)
    keys = ["condition", "row_uid", "candidate_aspect", "candidate_sentiment"]
    fewshot = fewshot.sort_values(keys, kind="stable").reset_index(drop=True)
    qlora = qlora.sort_values(keys, kind="stable").reset_index(drop=True)
    assert np.array_equal(hybrid["aspect_score"], fewshot["aspect_score"])
    assert np.array_equal(hybrid["sentiment_score"], qlora["sentiment_score"])
    validate_score_frame(hybrid, hybrid_job)


def test_score_bundle_round_trip_is_immutable_and_hash_checked(tmp_path: Path) -> None:
    value = job()
    frame = complete_frame(value)
    manifest = write_score_bundle(
        tmp_path,
        value,
        frame,
        preregistration_sha256="a" * 64,
        execution_commit="b" * 40,
        authorisation_id="auth-1",
        aspect_threshold=0.3,
        runner_up_threshold=0.5,
        source_artifact_sha256s={"selection": "c" * 64},
        expected_row_uids=("test:1", "test:2"),
    )
    assert manifest["third_sentiment_violation_count"] == 0
    observed, _ = read_score_bundle(
        tmp_path,
        value,
        preregistration_sha256="a" * 64,
        execution_commit="b" * 40,
        authorisation_id="auth-1",
        expected_row_uids=("test:1", "test:2"),
    )
    assert len(observed) == len(frame)
    with pytest.raises(FileExistsError, match="already exists"):
        write_score_bundle(
            tmp_path,
            value,
            frame,
            preregistration_sha256="a" * 64,
            execution_commit="b" * 40,
            authorisation_id="auth-1",
            aspect_threshold=0.3,
            runner_up_threshold=0.5,
            source_artifact_sha256s={"selection": "c" * 64},
        )
