"""Small, offline walkthrough using synthetic reviews and illustrative scores.

This exercises the actual training-policy transformation, candidate grid,
capped-two decoder and pair evaluator. It does not load a language model, train
a classifier or estimate the performance reported in the dissertation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from msc_project.data.fabsa import format_pair_label
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.experiments.taxonomy_training_policy_sensitivity import training_rows
from msc_project.experiments.taxonomy_two_stage import capped_two_sentiment_prediction_mask
from msc_project.experiments.taxonomy_two_stage_runtime import build_aspect_grid

ASPECTS = ("App/website", "Staff support")
SENTIMENTS = ("negative", "neutral", "positive")


def run_demo() -> dict[str, Any]:
    """Run the real policy/decoder code on a deliberately tiny worked example."""
    training = pd.DataFrame(
        [
            {
                "row_uid": "train:0",
                "text": "The app crashed, but the agent was helpful.",
                "original_split": "train",
                "labels": [(ASPECTS[0], "negative"), (ASPECTS[1], "positive")],
            },
            {
                "row_uid": "train:1",
                "text": "The agent was helpful.",
                "original_split": "train",
                "labels": [(ASPECTS[1], "positive")],
            },
            {
                "row_uid": "train:2",
                "text": "The agent was rude.",
                "original_split": "train",
                "labels": [(ASPECTS[1], "negative")],
            },
        ]
    )
    fold = SimpleNamespace(
        fold_id="synthetic-app", heldout_aspects=(ASPECTS[0],), seen_aspects=(ASPECTS[1],)
    )
    policies = {}
    for policy in ("review_filtered", "label_masked_all_reviews"):
        rows = training_rows(training, fold, policy)
        grid = build_aspect_grid(
            rows,
            fold.seen_aspects,
            {ASPECTS[1]: "name_only"},
            {"minimal_aspects": {ASPECTS[1]: "Interactions with support staff."}},
        )
        policies[policy] = {
            "retained_reviews": len(rows),
            "training_candidates": sorted(grid.candidate_aspect.unique()),
            "heldout_training_targets": int(grid.candidate_aspect.eq(ASPECTS[0]).sum()),
        }

    reviews = [
        (
            "The app crashed, although the agent was helpful.",
            [(ASPECTS[0], "negative"), (ASPECTS[1], "positive")],
        ),
        ("The agent was helpful.", [(ASPECTS[1], "positive")]),
    ]
    # Fixed example scores make both accepted and rejected candidates visible.
    scores = [
        (0.9, (0.8, 0.1, 0.1)),
        (0.85, (0.1, 0.1, 0.8)),
        (0.1, (0.2, 0.3, 0.5)),
        (0.9, (0.1, 0.1, 0.8)),
    ]
    records = []
    for review_index, (_, labels) in enumerate(reviews):
        for aspect_index, aspect in enumerate(ASPECTS):
            presence, sentiments = scores[2 * review_index + aspect_index]
            for sentiment, score in zip(SENTIMENTS, sentiments, strict=True):
                records.append(
                    {
                        "row_uid": f"synthetic:{review_index}",
                        "candidate_aspect": aspect,
                        "candidate_sentiment": sentiment,
                        "aspect_score": presence,
                        "sentiment_score": score,
                        "target": int((aspect, sentiment) in labels),
                        "pair_label": format_pair_label(aspect, sentiment),
                    }
                )
    grid = pd.DataFrame(records)
    predicted = capped_two_sentiment_prediction_mask(
        grid,
        aspect_threshold=0.5,
        second_sentiment_threshold=0.6,
    )
    true_sets, predicted_sets, examples = [], [], []
    for index, (review, labels) in enumerate(reviews):
        pairs = grid.loc[predicted & grid.row_uid.eq(f"synthetic:{index}"), "pair_label"].tolist()
        true_sets.append([format_pair_label(*label) for label in labels])
        predicted_sets.append(pairs)
        examples.append({"review": review, "predicted_pairs": pairs})
    metrics = evaluate_pair_and_aspect(
        true_sets,
        predicted_sets,
        [format_pair_label(a, s) for a in ASPECTS for s in SENTIMENTS],
    )
    return {
        "example_type": "synthetic_mechanics_only",
        "model_loaded": False,
        "policies": policies,
        "examples": examples,
        "illustrative_pair_micro_f1": metrics["pair_micro_f1"],
    }


def main(argv: list[str] | None = None) -> int:
    """Print the worked example and optionally save it without overwriting a file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON file for the synthetic example")
    args = parser.parse_args(argv)
    text = json.dumps(run_demo(), indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        except FileExistsError:
            parser.error(f"Output already exists: {args.output}. Choose a new path.")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
