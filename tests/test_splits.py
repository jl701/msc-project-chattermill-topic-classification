from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.splits import build_heldout_aspect_split, build_heldout_org_split, leakage_checks


def tiny_frame() -> pd.DataFrame:
    rows = [
        {"id": 1, "org_index": 1, "labels": [("A", "positive"), ("B", "negative")], "original_split": "train"},
        {"id": 2, "org_index": 2, "labels": [("A", "negative")], "original_split": "train"},
        {"id": 3, "org_index": 3, "labels": [("B", "positive")], "original_split": "validation"},
        {"id": 4, "org_index": 4, "labels": [("B", "negative"), ("C", "positive")], "original_split": "test"},
    ]
    frame = pd.DataFrame(rows)
    frame["row_uid"] = frame["original_split"] + ":" + frame["id"].astype(str)
    frame["pair_labels"] = frame["labels"].apply(lambda labels: [f"{aspect} | {sentiment}" for aspect, sentiment in labels])
    frame["aspect_labels"] = frame["labels"].apply(lambda labels: sorted({aspect for aspect, _ in labels}))
    frame["text"] = ""
    frame["data_source"] = "source"
    frame["industry"] = "industry"
    return frame


class SplitTest(unittest.TestCase):
    def test_heldout_org_has_no_org_overlap(self) -> None:
        splits = build_heldout_org_split(tiny_frame(), validation_orgs=[3], test_orgs=[4])
        self.assertEqual(set(splits["train"]["org_index"]), {1, 2})
        self.assertEqual(set(splits["validation"]["org_index"]), {3})
        self.assertEqual(set(splits["test"]["org_index"]), {4})

    def test_label_masked_removes_heldout_training_labels(self) -> None:
        splits = build_heldout_aspect_split(tiny_frame(), ["B"], strategy="label_masked")
        train_labels = splits["train"]["supervision_pair_labels"].tolist()
        self.assertEqual(train_labels, [["A | positive"], ["A | negative"]])
        validation_labels = splits["validation"]["supervision_pair_labels"].tolist()
        self.assertEqual(validation_labels, [["B | positive"]])

    def test_example_filtered_drops_heldout_training_examples(self) -> None:
        splits = build_heldout_aspect_split(tiny_frame(), ["B"], strategy="example_filtered")
        self.assertEqual(splits["train"]["id"].tolist(), [2])

    def test_heldout_aspect_has_no_train_eval_aspect_overlap(self) -> None:
        splits = build_heldout_aspect_split(tiny_frame(), ["B"], strategy="label_masked")
        checks = leakage_checks(splits)
        self.assertEqual(checks["supervision_aspect_overlap"]["train_validation"]["count"], 0)
        self.assertEqual(checks["supervision_aspect_overlap"]["train_test"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
