from __future__ import annotations

import dataclasses
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.frozen_raw_score_cache import (
    FrozenRawScoreCache,
    build_frozen_raw_score_cache_contract,
    frozen_cache_input_keys,
    frozen_raw_score_cache_path,
    score_frozen_qwen_with_cache,
)
from msc_project.experiments.taxonomy_methods import resolve_method_spec
from msc_project.experiments.taxonomy_pipeline import (
    build_run_contract,
    prepare_strict_seen_calibration,
)
from msc_project.experiments.taxonomy_protocol import registered_folds
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions


class CountingFrozenRuntime:
    method_id = "frozen_qwen_candidate_pair"

    def __init__(self) -> None:
        self.scored_rows = 0

    def fit(self, _manifest):
        return self

    def score(self, pair_manifest):
        self.scored_rows += len(pair_manifest)
        return np.asarray(
            [
                (
                    sum(
                        str(row[column]).encode("utf-8")[0]
                        if str(row[column])
                        else 0
                        for column in ("text", "candidate_text")
                    )
                    % 97
                )
                / 100.0
                for row in pair_manifest.to_dict(orient="records")
            ],
            dtype=float,
        )

    def close(self):
        return None


def _synthetic_frame(fold) -> pd.DataFrame:
    seen = fold.seen_aspects[0]
    return pd.DataFrame(
        [
            {
                "id": 1,
                "original_split": "train",
                "row_uid": "train:1",
                "text": "seen training evidence",
                "labels": [(seen, "positive")],
            },
            {
                "id": 2,
                "original_split": "validation",
                "row_uid": "validation:2",
                "text": "same review",
                "labels": [(seen, "negative")],
            },
            {
                "id": 3,
                "original_split": "validation",
                "row_uid": "validation:3",
                "text": "same review",
                "labels": [],
            },
        ]
    )


def _prepared_and_contract():
    fold = registered_folds("L2")[0]
    prepared = prepare_strict_seen_calibration(
        _synthetic_frame(fold),
        fold,
        load_minimal_descriptions(require_approved=False),
        total_budget=16,
        positive_budget=8,
    )
    parameters = resolve_method_spec(
        "frozen_qwen_candidate_pair"
    ).starting_recipe
    run_contract = build_run_contract(
        prepared,
        "frozen_qwen_candidate_pair",
        parameters,
        shard_count=1,
        formal=False,
    )
    return prepared, parameters, run_contract


def test_cache_reuses_exact_inputs_across_rows_and_fold_contracts(
    tmp_path: Path,
) -> None:
    prepared, parameters, run_contract = _prepared_and_contract()
    contract = build_frozen_raw_score_cache_contract(
        run_contract,
        parameters,
    )
    changed_fold_run = dataclasses.replace(
        run_contract,
        fold_id="different-fold",
        training_manifest_sha256="1" * 64,
        evaluation_data_sha256="2" * 64,
        evaluation_pair_identity_sha256="3" * 64,
        candidate_representation_sha256="4" * 64,
    )
    changed_fold_contract = build_frozen_raw_score_cache_contract(
        changed_fold_run,
        parameters,
    )
    assert changed_fold_contract.contract_sha256 == contract.contract_sha256

    runtime = CountingFrozenRuntime()
    manifest = prepared.evaluation_grid
    unique_count = len(set(frozen_cache_input_keys(manifest)))
    path = frozen_raw_score_cache_path(tmp_path, contract)
    with FrozenRawScoreCache(path, contract) as cache:
        first, first_stats = score_frozen_qwen_with_cache(
            runtime,
            manifest,
            cache,
        )
        second, second_stats = score_frozen_qwen_with_cache(
            runtime,
            manifest.sample(frac=1.0, random_state=13),
            cache,
        )
        assert first_stats.model_scored_unique_inputs == unique_count
        assert second_stats.model_scored_unique_inputs == 0
        assert second_stats.cached_unique_inputs == unique_count
        assert cache.count() == unique_count
    assert runtime.scored_rows == unique_count
    expected_second = {
        key: score
        for key, score in zip(frozen_cache_input_keys(manifest), first)
    }
    observed_second = {
        key: score
        for key, score in zip(
            frozen_cache_input_keys(
                manifest.sample(frac=1.0, random_state=13)
            ),
            second,
        )
    }
    assert observed_second == expected_second


def test_cache_database_contains_no_raw_text_gold_or_predictions(
    tmp_path: Path,
) -> None:
    prepared, parameters, run_contract = _prepared_and_contract()
    contract = build_frozen_raw_score_cache_contract(run_contract, parameters)
    path = frozen_raw_score_cache_path(tmp_path, contract)
    with FrozenRawScoreCache(path, contract) as cache:
        score_frozen_qwen_with_cache(
            CountingFrozenRuntime(),
            prepared.evaluation_grid,
            cache,
        )
    connection = sqlite3.connect(str(path))
    try:
        score_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(scores)").fetchall()
        }
        assert score_columns == {
            "input_sha256",
            "raw_present_probability",
        }
        serialised = "\n".join(
            value
            for (value,) in connection.execute(
                "SELECT value FROM metadata"
            ).fetchall()
        )
        assert "same review" not in serialised
        assert "target" not in score_columns
        assert "prediction" not in score_columns
        assert "threshold" not in score_columns
    finally:
        connection.close()


def test_validation_and_test_caches_are_isolated(tmp_path: Path) -> None:
    _, parameters, validation_run = _prepared_and_contract()
    test_run = dataclasses.replace(validation_run, split="test")
    validation_contract = build_frozen_raw_score_cache_contract(
        validation_run,
        parameters,
    )
    test_contract = build_frozen_raw_score_cache_contract(
        test_run,
        parameters,
    )
    assert validation_contract.contract_sha256 != test_contract.contract_sha256
    assert frozen_raw_score_cache_path(
        tmp_path, validation_contract
    ) != frozen_raw_score_cache_path(tmp_path, test_contract)


def test_cache_fails_closed_on_contract_conflict_and_heldout_validation(
    tmp_path: Path,
) -> None:
    prepared, parameters, run_contract = _prepared_and_contract()
    contract = build_frozen_raw_score_cache_contract(run_contract, parameters)
    path = frozen_raw_score_cache_path(tmp_path, contract)
    with FrozenRawScoreCache(path, contract):
        pass
    incompatible = dataclasses.replace(
        contract,
        model_revision="different-pinned-revision",
    )
    with pytest.raises(ValueError, match="incompatible"):
        FrozenRawScoreCache(path, incompatible)

    heldout = prepared.evaluation_grid.copy()
    heldout.loc[heldout.index[0], "is_heldout"] = True
    with FrozenRawScoreCache(path, contract) as cache:
        with pytest.raises(ValueError, match="held-out"):
            score_frozen_qwen_with_cache(
                CountingFrozenRuntime(),
                heldout,
                cache,
            )
        assert cache.count() == 0


def test_cache_rejects_parameter_digest_mismatch() -> None:
    _, parameters, run_contract = _prepared_and_contract()
    changed = {**parameters, "max_length": 128}
    with pytest.raises(ValueError, match="parameters differ"):
        build_frozen_raw_score_cache_contract(run_contract, changed)
    with pytest.raises(ValueError, match="stale frozen-Qwen"):
        build_frozen_raw_score_cache_contract(
            dataclasses.replace(run_contract, method_spec_sha256="0" * 64),
            parameters,
        )
