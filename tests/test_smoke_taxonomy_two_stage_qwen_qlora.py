from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke_taxonomy_two_stage_qwen_qlora.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_qlora_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_finite_history_requires_loss_and_update() -> None:
    assert MODULE._finite_history(
        [{"train_loss": 1.0, "global_step": 1}]
    )
    assert not MODULE._finite_history([])
    assert not MODULE._finite_history(
        [{"train_loss": float("nan"), "global_step": 1}]
    )
    assert not MODULE._finite_history(
        [{"train_loss": 1.0, "global_step": 0}]
    )
