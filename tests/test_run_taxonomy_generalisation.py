from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SCRIPT = PROJECT_ROOT / "scripts" / "run_taxonomy_generalisation.py"
SPEC = importlib.util.spec_from_file_location("run_taxonomy_generalisation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from msc_project.experiments.taxonomy_tuning import (
    fixed_registered_selection,
    registered_tuning_candidates,
)


def test_parameter_resolution_accepts_only_registered_candidates() -> None:
    parameters, digest = MODULE.resolve_parameters(
        "strict_train_only_tfidf",
        candidate_id="strict_train_only_tfidf-001",
        selection_path=None,
    )
    assert parameters["classifier_c"] == pytest.approx(0.1)
    assert len(digest) == 64
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.resolve_parameters(
            "strict_train_only_tfidf",
            candidate_id=None,
            selection_path=None,
        )
    with pytest.raises(ValueError, match="Unknown registered"):
        MODULE.resolve_parameters(
            "strict_train_only_tfidf",
            candidate_id="invented",
            selection_path=None,
        )


def test_tuning_scope_is_nested_and_rejects_any_test_phase() -> None:
    fold = MODULE.resolve_fold("L3", "l3-a01-a02")
    MODULE._validate_purpose(
        "tuning",
        "train",
        "strict_train_only_tfidf",
        fold,
        "strict_train_only_tfidf-001",
    )
    with pytest.raises(ValueError, match="never access"):
        MODULE._validate_purpose(
            "tuning",
            "score-test",
            "strict_train_only_tfidf",
            fold,
            "strict_train_only_tfidf-001",
        )
    with pytest.raises(ValueError, match="no registered nested"):
        MODULE._validate_purpose(
            "tuning",
            "train",
            "e5_base_v2",
            MODULE.resolve_fold("L2", "l2-a01"),
            "e5_base_v2-001",
        )


def test_nested_method_cannot_use_global_parameter_selection(
    tmp_path: Path,
) -> None:
    candidate = registered_tuning_candidates("strict_train_only_tfidf")[0]
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "method_id": "strict_train_only_tfidf",
                "selection_scope": "registered_seen_validation_grid",
                "training_scope_id": "global_fixed_recipe",
                "selected_parameters": candidate["parameters"],
                "selected_parameters_sha256": candidate["parameters_sha256"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wrong training scope"):
        MODULE.resolve_parameters(
            "strict_train_only_tfidf",
            candidate_id=None,
            selection_path=path,
            expected_training_scope_id="holdout-a01",
        )


def test_current_formal_gate_blocks_before_official_data_load(tmp_path: Path) -> None:
    args = MODULE.argparse.Namespace(
        method="strict_train_only_tfidf",
        level="L2",
        fold_id="l2-a02",
        condition=["D"],
        purpose="tuning",
        phase="train",
        candidate_id="strict_train_only_tfidf-001",
        parameter_selection=None,
        data_dir=tmp_path / "missing-data",
        output_root=tmp_path / "output",
        seed=13,
        shard_count=1,
        shard_index=0,
        resume=False,
        local_files_only=True,
    )
    with pytest.raises(ValueError, match="approved_and_frozen descriptions"):
        MODULE.run(args)


def test_multiseed_threshold_and_summary_paths_cannot_collide(
    tmp_path: Path,
) -> None:
    fold = MODULE.resolve_fold("L1", "l1-a01")
    threshold_13 = MODULE.threshold_path_for(
        tmp_path, "qwen_candidate_pair_qlora", fold, "a" * 64, 13
    )
    threshold_23 = MODULE.threshold_path_for(
        tmp_path, "qwen_candidate_pair_qlora", fold, "a" * 64, 23
    )
    summary_13 = MODULE.summary_path_for(
        tmp_path,
        "final",
        "test",
        "qwen_candidate_pair_qlora",
        fold,
        "a" * 64,
        13,
    )
    summary_23 = MODULE.summary_path_for(
        tmp_path,
        "final",
        "test",
        "qwen_candidate_pair_qlora",
        fold,
        "a" * 64,
        23,
    )
    assert threshold_13 != threshold_23
    assert summary_13 != summary_23


def test_tuning_and_final_validation_summaries_cannot_collide(
    tmp_path: Path,
) -> None:
    fold = MODULE.resolve_fold("L2", "l2-a02")
    tuning = MODULE.summary_path_for(
        tmp_path,
        "tuning",
        "validation",
        "strict_train_only_tfidf",
        fold,
        "a" * 64,
        13,
    )
    final = MODULE.summary_path_for(
        tmp_path,
        "final",
        "validation",
        "strict_train_only_tfidf",
        fold,
        "a" * 64,
        13,
    )
    assert tuning != final


def test_l1_validation_scores_seen_calibration_not_heldout_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fold = MODULE.resolve_fold("L1", "l1-a01")
    seen = fold.seen_aspects[0]
    heldout = fold.heldout_aspects[0]
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "original_split": "train",
                "row_uid": "train:1",
                "text": "synthetic seen training row",
                "labels": [(seen, "positive")],
            },
            {
                "id": 2,
                "original_split": "validation",
                "row_uid": "validation:2",
                "text": "synthetic heldout validation row",
                "labels": [(heldout, "positive")],
            },
            {
                "id": 3,
                "original_split": "validation",
                "row_uid": "validation:3",
                "text": "synthetic seen validation row",
                "labels": [(seen, "negative")],
            },
        ]
    )
    selection = fixed_registered_selection("e5_base_v2")
    selection["source_summaries"] = []
    selection_path = tmp_path / "e5.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    class FakeRuntime:
        method_id = "e5_base_v2"

        def score(self, pair_manifest):
            return np.full(len(pair_manifest), 0.25)

        def close(self):
            return None

    opened = []

    def load_splits(_data_dir, splits):
        opened.append(tuple(splits))
        return frame.copy()

    monkeypatch.setattr(MODULE, "assert_formal_run_gates", lambda *_: None)
    monkeypatch.setattr(MODULE, "load_official_fabsa_splits", load_splits)
    monkeypatch.setattr(
        MODULE,
        "load_runtime_checkpoint",
        lambda *args, **kwargs: FakeRuntime(),
    )
    args = MODULE.argparse.Namespace(
        method="e5_base_v2",
        level="L1",
        fold_id=fold.fold_id,
        condition=[],
        purpose="final",
        phase="score-validation",
        candidate_id=None,
        parameter_selection=selection_path,
        data_dir=tmp_path / "unused",
        output_root=tmp_path / "output",
        seed=13,
        shard_count=1,
        shard_index=0,
        all_shards=True,
        resume=True,
        local_files_only=True,
    )
    result = MODULE.run(args)
    assert opened == [("train", "validation")]
    assert {value["condition"] for value in result["results"]} == {
        "seen-calibration"
    }
    assert result["unique_pairs_scored"] == 2 * 11 * 3
    select_args = MODULE.argparse.Namespace(
        **{**vars(args), "phase": "select-threshold"}
    )
    selected = MODULE.run(select_args)
    artifact = selected["threshold_artifact"]
    assert set(artifact["calibration_aspects"]) == set(fold.seen_aspects)
    assert not set(artifact["calibration_aspects"]) & set(fold.heldout_aspects)
    assert len(set(artifact["validation_score_contracts"].values())) == 1


def test_test_threshold_gate_precedes_frozen_cache_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fold = MODULE.resolve_fold("L2", "l2-a01")
    seen = fold.seen_aspects[0]
    heldout = fold.heldout_aspects[0]
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "original_split": "train",
                "row_uid": "train:1",
                "text": "synthetic training row",
                "labels": [(seen, "positive")],
            },
            {
                "id": 2,
                "original_split": "test",
                "row_uid": "test:2",
                "text": "synthetic test row",
                "labels": [(heldout, "negative")],
            },
        ]
    )
    selection = fixed_registered_selection("frozen_qwen_candidate_pair")
    selection["source_summaries"] = []
    selection_path = tmp_path / "frozen.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    class FakeRuntime:
        method_id = "frozen_qwen_candidate_pair"

        def score(self, pair_manifest):
            return np.full(len(pair_manifest), 0.25)

        def close(self):
            return None

    events = []
    real_cache = MODULE.FrozenRawScoreCache

    def cache_factory(*args, **kwargs):
        events.append("cache-open")
        return real_cache(*args, **kwargs)

    def threshold_gate(*args, **kwargs):
        events.append("threshold-loaded")
        return object()

    def load_runtime(*args, **kwargs):
        assert kwargs["lazy_frozen_qwen"] is True
        return FakeRuntime()

    monkeypatch.setattr(MODULE, "assert_formal_run_gates", lambda *_: None)
    monkeypatch.setattr(
        MODULE,
        "load_official_fabsa_splits",
        lambda *_: frame.copy(),
    )
    monkeypatch.setattr(
        MODULE,
        "load_runtime_checkpoint",
        load_runtime,
    )
    monkeypatch.setattr(
        MODULE,
        "load_threshold_transfer_artifact",
        threshold_gate,
    )
    monkeypatch.setattr(MODULE, "FrozenRawScoreCache", cache_factory)
    args = MODULE.argparse.Namespace(
        method="frozen_qwen_candidate_pair",
        level="L2",
        fold_id=fold.fold_id,
        condition=[],
        purpose="final",
        phase="score-test",
        candidate_id=None,
        parameter_selection=selection_path,
        data_dir=tmp_path / "unused",
        output_root=tmp_path / "output",
        seed=13,
        shard_count=1,
        shard_index=0,
        all_shards=True,
        resume=True,
        local_files_only=True,
    )
    result = MODULE.run(args)
    assert events[:2] == ["threshold-loaded", "cache-open"]
    assert result["raw_score_cache"]["split"] == "test"
    assert result["unique_pairs_scored"] == 12 * 3
