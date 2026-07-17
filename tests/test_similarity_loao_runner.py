from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_similarity_loao_baselines import (
    aggregate_results,
    load_stage_frame,
    positive_average_precision,
    run,
    select_threshold,
    validate_test_selection,
    write_predictions,
)
from msc_project.baselines.candidate_similarity import REGISTERED_CONFIGS


ASPECT = "Delivery: Speed"


def tiny_eval_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1", "2"],
            "row_uid": ["validation:1", "validation:2"],
            "original_split": ["validation", "validation"],
            "org_index": [1, 2],
            "supervision_labels": [[(ASPECT, "positive")], []],
            "supervision_pair_labels": [[f"{ASPECT} | positive"], []],
        }
    )


class SimilarityLoaoRunnerTests(unittest.TestCase):
    def test_validation_stage_loader_never_requests_test(self) -> None:
        calls: list[str] = []

        def fake_load_split(_data_dir: Path, split: str) -> pd.DataFrame:
            calls.append(split)
            return pd.DataFrame(
                {
                    "id": [f"{split}-1"],
                    "labels": [[]],
                    "pair_labels": [[]],
                    "aspect_labels": [[]],
                    "text": ["review"],
                }
            )

        with patch("run_similarity_loao_baselines.load_split", side_effect=fake_load_split):
            frame = load_stage_frame(Path("unused"), "validation")

        self.assertEqual(calls, ["train", "validation"])
        self.assertNotIn("test", frame["original_split"].tolist())

    def test_test_stage_loader_never_requests_validation(self) -> None:
        calls: list[str] = []

        def fake_load_split(_data_dir: Path, split: str) -> pd.DataFrame:
            calls.append(split)
            return pd.DataFrame(
                {
                    "id": [f"{split}-1"],
                    "labels": [[]],
                    "pair_labels": [[]],
                    "aspect_labels": [[]],
                    "text": ["review"],
                }
            )

        with patch("run_similarity_loao_baselines.load_split", side_effect=fake_load_split):
            load_stage_frame(Path("unused"), "test")

        self.assertEqual(calls, ["train", "test"])

    def test_threshold_selection_is_pair_micro_first_and_conservative_on_ties(self) -> None:
        frame = tiny_eval_frame()
        sentiment_lookup = [{ASPECT: "positive"}, {ASPECT: "positive"}]

        threshold, sweep, metrics, aspects, pairs = select_threshold(
            frame,
            ASPECT,
            np.asarray([0.8, 0.2]),
            sentiment_lookup,
        )

        self.assertEqual(threshold, 0.8)
        self.assertEqual(float(sweep.iloc[0]["pair_micro_f1"]), 1.0)
        self.assertEqual(float(metrics["presence_f1"]), 1.0)
        self.assertEqual(aspects, [[ASPECT], []])
        self.assertEqual(pairs, [[f"{ASPECT} | positive"], []])

    def test_presence_average_precision_uses_continuous_score(self) -> None:
        self.assertEqual(positive_average_precision([["x"], []], np.asarray([0.9, 0.1])), 1.0)
        with self.assertRaisesRegex(ValueError, "both present and absent"):
            positive_average_precision([[], []], np.asarray([0.1, 0.2]))

    def test_test_selection_requires_exact_methods_aspects_and_fingerprint(self) -> None:
        config = REGISTERED_CONFIGS["bow_count_1_2_train_vocab"]
        selection = {
            "stage": "validation",
            "protocol_id": "loao_open_topic_all_row_v1",
            "config_fingerprint": "abc",
            "methods": [config.name],
            "aspects": [ASPECT],
            "thresholds": {config.name: {ASPECT: 0.2}},
            "protocol_complete": True,
        }

        thresholds = validate_test_selection(selection, [config], [ASPECT], "abc")
        self.assertEqual(thresholds[config.name][ASPECT], 0.2)
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_test_selection(selection, [config], [ASPECT], "different")

    def test_partial_validation_manifest_is_rejected_before_test_is_loaded(self) -> None:
        configs = list(REGISTERED_CONFIGS.values())
        partial = {
            "stage": "validation",
            "protocol_id": "loao_open_topic_all_row_v1",
            "config_fingerprint": "unused",
            "methods": list(REGISTERED_CONFIGS),
            "aspects": [f"aspect-{index}" for index in range(12)],
            "thresholds": {},
            "protocol_complete": False,
        }
        args = Namespace(
            stage="test",
            data_dir=Path("unused"),
            output_dir=Path("unused-output"),
            public_output_dir=None,
            model=[],
            heldout_aspect=[],
            selection_manifest=Path("selection.json"),
            device="cpu",
            local_files_only=True,
        )

        with (
            patch("run_similarity_loao_baselines.read_selection_manifest", return_value=partial),
            patch("run_similarity_loao_baselines.load_stage_frame") as load_stage,
            self.assertRaisesRegex(ValueError, "protocol-complete"),
        ):
            run(args)

        load_stage.assert_not_called()

    def test_threshold_mapping_must_cover_exact_nested_keys(self) -> None:
        config = REGISTERED_CONFIGS["bow_count_1_2_train_vocab"]
        selection = {
            "stage": "validation",
            "protocol_id": "loao_open_topic_all_row_v1",
            "config_fingerprint": "abc",
            "methods": [config.name],
            "aspects": [ASPECT],
            "thresholds": {config.name: {"wrong-aspect": 0.2}},
            "protocol_complete": True,
        }

        with self.assertRaisesRegex(ValueError, "cover all aspects exactly"):
            validate_test_selection(selection, [config], [ASPECT], "abc")

    def test_prediction_export_contains_no_review_text(self) -> None:
        frame = tiny_eval_frame()
        config = REGISTERED_CONFIGS["bow_count_1_2_train_vocab"]
        sentiment_features = [
            {ASPECT: {"predicted_sentiment": "positive"}},
            {ASPECT: {"predicted_sentiment": "positive"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            write_predictions(
                frame,
                config,
                "validation",
                1,
                ASPECT,
                np.asarray([0.8, 0.2]),
                0.8,
                [[ASPECT], []],
                [[f"{ASPECT} | positive"], []],
                sentiment_features,
                path,
            )
            payload = path.read_text(encoding="utf-8")

        self.assertNotIn('"text"', payload)
        self.assertIn('"row_uid": "validation:1"', payload)

    def test_aggregate_reports_unweighted_median(self) -> None:
        base = {metric: 0.0 for metric in __import__("run_similarity_loao_baselines").AGGREGATE_METRICS}
        rows = []
        for index, value in enumerate([0.1, 0.2, 0.9], start=1):
            row = {
                "method": "m",
                "split": "test",
                "heldout_aspect": f"a{index}",
                **base,
            }
            row["pair_micro_f1"] = value
            rows.append(row)

        aggregate = aggregate_results(pd.DataFrame(rows)).iloc[0]

        self.assertAlmostEqual(float(aggregate["pair_micro_f1_mean"]), 0.4)
        self.assertAlmostEqual(float(aggregate["pair_micro_f1_median"]), 0.2)


if __name__ == "__main__":
    unittest.main()
