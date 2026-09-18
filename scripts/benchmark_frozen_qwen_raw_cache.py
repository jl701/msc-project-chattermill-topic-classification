"""Bounded real-model equivalence check for the frozen-Qwen raw-score cache.

Only hard-coded synthetic review text is used. This script never opens the
official FABSA train, validation, or test resources.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
from msc_project.experiments.taxonomy_runtime_factory import (
    create_runtime_for_training,
)


def _synthetic_prepared():
    fold = registered_folds("L2")[0]
    seen = fold.seen_aspects[0]
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "original_split": "train",
                "row_uid": "synthetic-train:1",
                "text": "The staff answered quickly and were helpful.",
                "labels": [(seen, "positive")],
            },
            {
                "id": 2,
                "original_split": "train",
                "row_uid": "synthetic-train:2",
                "text": "The service was difficult to use.",
                "labels": [(seen, "negative")],
            },
            {
                "id": 3,
                "original_split": "validation",
                "row_uid": "synthetic-validation:1",
                "text": "The app was simple to use, but support replied slowly.",
                "labels": [],
            },
        ]
    )
    return prepare_strict_seen_calibration(
        frame,
        fold,
        load_minimal_descriptions(require_approved=False),
        total_budget=16,
        positive_budget=8,
    )


def _write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run(*, output: Path, pair_count: int) -> dict[str, object]:
    if pair_count < 1 or pair_count > 33:
        raise ValueError("pair_count must be between 1 and 33.")
    prepared = _synthetic_prepared()
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
    cache_contract = build_frozen_raw_score_cache_contract(
        run_contract,
        parameters,
    )
    cache_root = output.parent / f".{output.stem}_cache"
    cache_path = frozen_raw_score_cache_path(cache_root, cache_contract)
    if cache_path.exists():
        raise FileExistsError(
            "Benchmark requires a cold cache; choose a new output path."
        )
    manifest = prepared.evaluation_grid.iloc[:pair_count].reset_index(drop=True)
    started = time.perf_counter()
    runtime = create_runtime_for_training(
        "frozen_qwen_candidate_pair",
        parameters,
        seed=13,
        device=torch.device("cuda"),
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        direct = runtime.score(manifest)
        direct_seconds = time.perf_counter() - started
        with FrozenRawScoreCache(cache_path, cache_contract) as cache:
            started = time.perf_counter()
            cold, cold_stats = score_frozen_qwen_with_cache(
                runtime,
                manifest,
                cache,
            )
            cold_seconds = time.perf_counter() - started
            shuffled = manifest.sample(frac=1.0, random_state=13)
            started = time.perf_counter()
            warm, warm_stats = score_frozen_qwen_with_cache(
                runtime,
                shuffled,
                cache,
            )
            warm_seconds = time.perf_counter() - started
            cache_summary = cache.session_summary()
    finally:
        runtime.close()

    direct_by_key = dict(zip(frozen_cache_input_keys(manifest), direct))
    warm_in_direct_order = np.asarray(
        [
            dict(zip(frozen_cache_input_keys(shuffled), warm))[key]
            for key in frozen_cache_input_keys(manifest)
        ],
        dtype=float,
    )
    result = {
        "schema_version": "frozen_qwen_raw_cache_benchmark_v1",
        "official_data_read": False,
        "synthetic_review_count": 1,
        "pair_count": pair_count,
        "unique_input_count": len(direct_by_key),
        "method_id": "frozen_qwen_candidate_pair",
        "model_id": resolve_method_spec(
            "frozen_qwen_candidate_pair"
        ).model_id,
        "model_revision": resolve_method_spec(
            "frozen_qwen_candidate_pair"
        ).model_revision,
        "scientific_parameters": parameters,
        "cache_contract_sha256": cache_contract.contract_sha256,
        "equivalence": {
            "direct_vs_cold_bitwise_equal": bool(np.array_equal(direct, cold)),
            "direct_vs_cold_max_abs_difference": float(
                np.max(np.abs(direct - cold))
            ),
            "direct_vs_warm_bitwise_equal": bool(
                np.array_equal(direct, warm_in_direct_order)
            ),
            "direct_vs_warm_max_abs_difference": float(
                np.max(np.abs(direct - warm_in_direct_order))
            ),
        },
        "cache_stats": {
            "cold": cold_stats.to_dict(),
            "warm": warm_stats.to_dict(),
            "session": cache_summary,
        },
        "timing_seconds": {
            "model_load": load_seconds,
            "direct_uncached": direct_seconds,
            "cold_cache_fill": cold_seconds,
            "warm_cache_lookup": warm_seconds,
        },
        "acceptance_rule": {
            "raw_score_max_abs_difference": 1e-7,
            "warm_model_scored_unique_inputs": 0,
            "no_official_data": True,
        },
        "accepted": bool(
            np.max(np.abs(direct - cold)) <= 1e-7
            and np.max(np.abs(direct - warm_in_direct_order)) <= 1e-7
            and warm_stats.model_scored_unique_inputs == 0
        ),
    }
    _write_new(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=12)
    args = parser.parse_args()
    result = run(output=args.output, pair_count=args.pair_count)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["accepted"]:
        raise SystemExit("Frozen-Qwen raw-cache equivalence benchmark failed.")


if __name__ == "__main__":
    main()
