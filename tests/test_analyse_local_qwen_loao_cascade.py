from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyse_local_qwen_loao_cascade import choose_predictions, qwen_call_count


class AnalyseLocalQwenLoaoCascadeTests(unittest.TestCase):
    def test_choose_predictions_keeps_local_when_local_nonempty_else_qwen(self) -> None:
        local = ["Company brand: Competitor | positive"]
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(choose_predictions(local, qwen, "local_nonempty_else_qwen"), local)

    def test_choose_predictions_uses_qwen_when_local_empty(self) -> None:
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(choose_predictions([], qwen, "local_nonempty_else_qwen"), qwen)

    def test_pair_agreement_requires_same_sentiment(self) -> None:
        local = ["Company brand: Competitor | positive"]
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(choose_predictions(local, qwen, "pair_agreement"), [])

    def test_aspect_agreement_can_keep_local_sentiment(self) -> None:
        local = ["Company brand: Competitor | positive"]
        qwen = ["Company brand: Competitor | negative"]

        self.assertEqual(choose_predictions(local, qwen, "aspect_agreement_local_sentiment"), local)

    def test_qwen_call_count_matches_gate_policy(self) -> None:
        self.assertEqual(qwen_call_count([], "local_only"), 0)
        self.assertEqual(qwen_call_count([], "qwen_only"), 1)
        self.assertEqual(qwen_call_count([], "local_nonempty_else_qwen"), 1)
        self.assertEqual(
            qwen_call_count(["Company brand: Competitor | positive"], "local_nonempty_else_qwen"),
            0,
        )
        self.assertEqual(qwen_call_count([], "aspect_agreement_local_sentiment"), 0)
        self.assertEqual(
            qwen_call_count(["Company brand: Competitor | positive"], "aspect_agreement_local_sentiment"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
