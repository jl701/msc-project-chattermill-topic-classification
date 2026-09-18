from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_qwen_heldout_aspect_sft_data import build_singleton_train_rows, write_strategy_data


def tiny_frame() -> pd.DataFrame:
    rows = [
        {"id": "tr1", "labels": [("Seen", "positive")], "original_split": "train", "text": "seen train"},
        {"id": "tr2", "labels": [("Other", "negative")], "original_split": "train", "text": "other train"},
        {"id": "tr3", "labels": [("Held", "negative")], "original_split": "train", "text": "held train"},
        {"id": "va1", "labels": [("Seen", "positive")], "original_split": "validation", "text": "seen validation"},
        {"id": "va2", "labels": [("Held", "positive")], "original_split": "validation", "text": "held validation"},
        {"id": "te1", "labels": [("Seen", "negative")], "original_split": "test", "text": "seen test"},
    ]
    frame = pd.DataFrame(rows)
    frame["row_uid"] = frame["original_split"] + ":" + frame["id"].astype(str)
    return frame


class PrepareQwenHeldoutAspectSftDataTests(unittest.TestCase):
    def test_all_row_eval_scope_preserves_empty_heldout_eval_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_dir=Path(tmp),
                prompt_variant="indexed",
                limit=None,
                eval_row_scope="all",
                train_candidate_mode="grouped",
                singleton_negative_ratio=1,
                seed=13,
            )

            write_strategy_data(tiny_frame(), "example_filtered", ["Held"], args)

            data_dir = Path(tmp) / "example_filtered"
            metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
            validation_rows = [
                json.loads(line)
                for line in (data_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            test_rows = [
                json.loads(line)
                for line in (data_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(metadata["eval_row_scope"], "all")
        self.assertEqual([row["id"] for row in validation_rows], ["va1", "va2"])
        self.assertEqual(validation_rows[0]["pair_labels"], [])
        self.assertEqual(validation_rows[1]["pair_labels"], ["Held | positive"])
        self.assertEqual([row["id"] for row in test_rows], ["te1"])
        self.assertEqual(test_rows[0]["pair_labels"], [])

    def test_singleton_train_mode_adds_empty_negative_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_dir=Path(tmp),
                prompt_variant="indexed_conservative",
                limit=None,
                eval_row_scope="all",
                train_candidate_mode="singleton",
                singleton_negative_ratio=1,
                seed=13,
            )

            write_strategy_data(tiny_frame(), "example_filtered", ["Held"], args)

            data_dir = Path(tmp) / "example_filtered"
            metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
            train_rows = [
                json.loads(line)
                for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(metadata["train_candidate_mode"], "singleton")
        self.assertEqual(metadata["singleton_negative_ratio"], 1)
        self.assertEqual(len(train_rows), 4)
        self.assertEqual(sum(1 for row in train_rows if row["pair_labels"]), 2)
        self.assertEqual(sum(1 for row in train_rows if not row["pair_labels"]), 2)
        self.assertTrue(all(len(row["candidate_aspects"]) == 1 for row in train_rows))

    def test_singleton_train_mode_accepts_fractional_negative_ratio(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "id": "tr1",
                    "row_uid": "train:tr1",
                    "original_split": "train",
                    "text": "two positive aspects",
                    "supervision_pair_labels": ["A | positive", "B | negative"],
                }
            ]
        )

        rows = build_singleton_train_rows(
            frame,
            ["A", "B", "C"],
            "indexed_conservative",
            None,
            0.5,
            13,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(1 for row in rows if row["pair_labels"]), 2)
        self.assertEqual(sum(1 for row in rows if not row["pair_labels"]), 1)
        self.assertTrue(all(len(row["candidate_aspects"]) == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
