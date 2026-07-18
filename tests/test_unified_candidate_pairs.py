from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.unified_candidate_pairs import (
    CANDIDATE_SENTIMENTS,
    build_eval_grid,
    build_full_manifest,
    budget_sample,
    format_candidate_statement,
    load_descriptions,
    load_experiment_config,
    manifest_hash,
)


APP = "Online experience: App website"
EMAIL = "Staff support: Email"
PHONE = "Staff support: Phone"
STAFF = "Staff support: Attitude of staff"
PRICE = "Value: Price value for money"
HELDOUT = "Company brand: Competitor"


def frame_with_labels(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class UnifiedCandidatePairTests(unittest.TestCase):
    def test_frozen_resources_validate_twelve_labels_and_protocol(self) -> None:
        config = load_experiment_config()
        descriptions = load_descriptions()
        self.assertEqual(config["protocol_id"], "loao_unified_candidate_pair_experimental_v1")
        self.assertEqual(len(descriptions["aspects"]), 12)
        self.assertEqual(tuple(descriptions["sentiments"]), CANDIDATE_SENTIMENTS)

    def test_candidate_statement_control_and_enhanced_share_canonical_claim(self) -> None:
        control = format_candidate_statement(EMAIL, "negative", "control")
        enhanced = format_candidate_statement(EMAIL, "negative", "enhanced")
        self.assertEqual(control, f"Aspect: {EMAIL}. Candidate sentiment: negative.")
        self.assertIn(f"Aspect: {EMAIL}.", enhanced)
        self.assertIn("Definition:", enhanced)
        self.assertIn("Lexical cues:", enhanced)
        self.assertIn("Boundary:", enhanced)
        self.assertIn("Sentiment definition:", enhanced)

    def test_control_adds_four_deterministic_non_gold_pairs_per_positive(self) -> None:
        frame = frame_with_labels(
            [{"row_uid": "train:1", "text": "The app was great.", "supervision_labels": [(APP, "positive")]}]
        )
        candidates = [APP, EMAIL, PHONE]
        first = build_full_manifest(frame, candidates, "control", seed=13)
        second = build_full_manifest(frame.sample(frac=1), candidates, "control", seed=13)
        self.assertEqual(first["target"].value_counts().to_dict(), {0: 4, 1: 1})
        self.assertTrue((first.loc[first["target"] == 0, "negative_type"] == "random_non_gold_pair").all())
        pd.testing.assert_frame_equal(first, second)

    def test_enhanced_is_multi_sentiment_safe_and_never_adds_heldout(self) -> None:
        frame = frame_with_labels(
            [
                {
                    "row_uid": "train:2",
                    "text": "Email support was mixed.",
                    "supervision_labels": [
                        (EMAIL, "positive"),
                        (EMAIL, "negative"),
                        (HELDOUT, "neutral"),
                    ],
                }
            ]
        )
        manifest = build_full_manifest(frame, [EMAIL, PHONE, STAFF, PRICE], "enhanced", seed=13)
        keys = set(zip(manifest["candidate_aspect"], manifest["candidate_sentiment"]))
        targets = dict(
            zip(
                zip(manifest["candidate_aspect"], manifest["candidate_sentiment"]),
                manifest["target"],
            )
        )
        self.assertEqual(targets[(EMAIL, "positive")], 1)
        self.assertEqual(targets[(EMAIL, "negative")], 1)
        self.assertEqual(targets[(EMAIL, "neutral")], 0)
        self.assertNotIn(HELDOUT, set(manifest["candidate_aspect"]))
        self.assertEqual(len(keys), len(manifest))
        hard = manifest[manifest["negative_type"] == "same_parent_hard_absent"]
        self.assertFalse(hard.empty)
        self.assertTrue(set(hard["candidate_aspect"]) <= {PHONE, STAFF})

    def test_eval_grid_is_all_rows_by_three_sentiments_and_supports_multi_sentiment(self) -> None:
        frame = frame_with_labels(
            [
                {
                    "row_uid": "validation:1",
                    "text": "Email was both good and bad.",
                    "supervision_labels": [(EMAIL, "positive"), (EMAIL, "negative")],
                },
                {"row_uid": "validation:2", "text": "No email.", "supervision_labels": []},
            ]
        )
        grid = build_eval_grid(frame, EMAIL, "enhanced")
        self.assertEqual(len(grid), 6)
        self.assertEqual(grid["target"].sum(), 2)
        self.assertEqual(set(grid["candidate_sentiment"]), set(CANDIDATE_SENTIMENTS))
        self.assertEqual(grid.groupby("row_uid").size().to_dict(), {"validation:1": 3, "validation:2": 3})

    def test_budget_sample_is_deterministic_and_aspect_balanced(self) -> None:
        rows = []
        for target in (0, 1):
            for aspect in (APP, EMAIL, PRICE):
                for index in range(5):
                    rows.append(
                        {
                            "row_uid": f"train:{target}:{aspect}:{index}",
                            "candidate_aspect": aspect,
                            "candidate_sentiment": CANDIDATE_SENTIMENTS[index % 3],
                            "target": target,
                            "text": str(index),
                        }
                    )
        manifest = pd.DataFrame(rows)
        first = budget_sample(manifest, total_budget=12, positive_budget=6, seed=13)
        second = budget_sample(manifest.sample(frac=1, random_state=4), 12, 6, seed=13)
        self.assertEqual(len(first), 12)
        self.assertEqual(first["target"].value_counts().to_dict(), {1: 6, 0: 6})
        counts = first.groupby(["target", "candidate_aspect"]).size()
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(
            set(zip(first["row_uid"], first["candidate_aspect"], first["candidate_sentiment"])),
            set(zip(second["row_uid"], second["candidate_aspect"], second["candidate_sentiment"])),
        )

    def test_manifest_hash_is_order_invariant_and_change_sensitive(self) -> None:
        manifest = pd.DataFrame(
            [
                {"row_uid": "1", "candidate_aspect": APP, "candidate_sentiment": "positive", "target": 1},
                {"row_uid": "2", "candidate_aspect": EMAIL, "candidate_sentiment": "negative", "target": 0},
            ]
        )
        self.assertEqual(manifest_hash(manifest), manifest_hash(manifest.iloc[::-1]))
        changed = manifest.copy()
        changed.loc[0, "target"] = 0
        self.assertNotEqual(manifest_hash(manifest), manifest_hash(changed))


if __name__ == "__main__":
    unittest.main()
