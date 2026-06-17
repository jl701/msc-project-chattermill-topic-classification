from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import aspect_labels, pair_labels, parse_label_codes, parse_labels_json


class FabsaLoaderTest(unittest.TestCase):
    def test_parse_labels_json(self) -> None:
        labels = parse_labels_json('[["Online experience: App website", "negative"]]')
        self.assertEqual(labels, [("Online experience: App website", "negative")])

    def test_parse_label_codes(self) -> None:
        codes = parse_label_codes("['online-experience.app-website.-1']")
        self.assertEqual(codes, ["online-experience.app-website.-1"])

    def test_label_views(self) -> None:
        labels = [
            ("Online experience: App website", "negative"),
            ("Online experience: App website", "positive"),
        ]
        self.assertEqual(
            pair_labels(labels),
            [
                "Online experience: App website | negative",
                "Online experience: App website | positive",
            ],
        )
        self.assertEqual(aspect_labels(labels), ["Online experience: App website"])


if __name__ == "__main__":
    unittest.main()

