"""Run validation-only genuine two-stage Levels 1-4 before cloud execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.baselines.candidate_similarity import (  # noqa: E402
    FrozenTransformerSentenceEncoder,
    REGISTERED_CONFIGS,
)
from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    candidate_representation_variants,
    canonical_aspects,
    registered_folds,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_qwen_zero_shot_cache import (  # noqa: E402
    ReadOnlyQwenTwoStageCache,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    capped_two_sentiment_prediction_mask,
    conditional_sentiment_metrics,
    evaluate_prediction_mask,
    select_second_sentiment_threshold,
    select_two_stage_threshold,
    two_stage_candidates,
    two_stage_prediction_mask,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    TfidfTrueTwoStageRuntime,
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
)


METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "frozen_qwen_candidate_pair",
)
LEVELS = ("L1", "L2", "L3", "L4")
VARIANT_MAP = {
    "name_only": "name_only",
    "minimal": "name_and_description",
    "rich": "rich",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _score_hash(frame: pd.DataFrame) -> str:
    records = (
        frame[
            [
                "row_uid",
                "candidate_aspect",
                "candidate_sentiment",
                "representation_variant",
                "aspect_score",
                "sentiment_score",
            ]
        ]
        .sort_values(
            ["row_uid", "candidate_aspect", "candidate_sentiment"], kind="stable"
        )
        .to_dict(orient="records")
    )
    return hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class E5ValidationCache:
    def __init__(self, validation_rows: pd.DataFrame, *, local_files_only: bool) -> None:
        self.encoder = FrozenTransformerSentenceEncoder(
            REGISTERED_CONFIGS["e5_base_v2"],
            device="auto",
            local_files_only=local_files_only,
        )
        ordered = validation_rows.assign(
            _uid=validation_rows["row_uid"].astype(str)
        ).sort_values("_uid", kind="stable")
        self.review_uids = ordered["row_uid"].astype(str).tolist()
        self.review_index = {
            value: index for index, value in enumerate(self.review_uids)
        }
        self.review_embeddings = self.encoder.encode(
            ordered["text"].astype(str).tolist(), role="review"
        )
        self.candidate_embeddings: dict[str, np.ndarray] = {}

    def score(self, grid: pd.DataFrame) -> np.ndarray:
        candidates = grid["candidate_text"].astype(str).tolist()
        missing = sorted(set(candidates) - set(self.candidate_embeddings))
        if missing:
            matrix = self.encoder.encode(missing, role="candidate")
            self.candidate_embeddings.update(dict(zip(missing, matrix)))
        values = np.asarray(
            [
                float(
                    self.review_embeddings[self.review_index[str(uid)]]
                    @ self.candidate_embeddings[candidate]
                )
                for uid, candidate in zip(grid["row_uid"], candidates)
            ],
            dtype=float,
        )
        output = (np.clip(values, -1.0, 1.0) + 1.0) / 2.0
        if len(output) != len(grid) or not np.isfinite(output).all():
            raise ValueError("E5 pre-cloud scores are invalid.")
        return output

    def close(self) -> None:
        self.encoder.close()


def _score_grids(
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    *,
    runtime: TfidfTrueTwoStageRuntime | None,
    e5: E5ValidationCache | None,
    qwen: ReadOnlyQwenTwoStageCache | None = None,
) -> pd.DataFrame:
    if runtime is not None:
        aspect_scores = runtime.score_aspects(aspect_grid)
        sentiment_scores = runtime.score_sentiments(sentiment_grid)
    elif e5 is not None:
        aspect_scores = e5.score(aspect_grid)
        sentiment_scores = e5.score(sentiment_grid)
    elif qwen is not None:
        aspect_probabilities = qwen.score(aspect_grid, mode="aspect")
        sentiment_probabilities = qwen.score(aspect_grid, mode="sentiment")
        aspect_scores = aspect_probabilities[:, 0]
        expected_keys = [
            (str(uid), str(aspect), str(sentiment))
            for uid, aspect, sentiment in zip(
                sentiment_grid["row_uid"],
                sentiment_grid["candidate_aspect"],
                sentiment_grid["candidate_sentiment"],
            )
        ]
        observed_keys = [
            (str(uid), str(aspect), sentiment)
            for uid, aspect in zip(
                aspect_grid["row_uid"], aspect_grid["candidate_aspect"]
            )
            for sentiment in ("negative", "neutral", "positive")
        ]
        if observed_keys != expected_keys:
            raise AssertionError("Aspect and sentiment grid ordering is inconsistent.")
        sentiment_scores = sentiment_probabilities.reshape(-1)
    else:
        raise AssertionError("No cheap-method runtime is available.")
    return join_two_stage_scores(
        aspect_grid, sentiment_grid, aspect_scores, sentiment_scores
    )


def _variants_for_condition(fold: Any, condition: str) -> dict[str, str]:
    values = candidate_representation_variants(fold, condition)
    return {aspect: VARIANT_MAP[variant] for aspect, variant in values.items()}


def _pipeline_conditional_sentiment_accuracy(
    scored: pd.DataFrame,
    *,
    aspect_threshold: float,
) -> dict[str, float | int]:
    top = two_stage_candidates(scored)
    gold_counts = (
        scored[scored["target"].astype(int).eq(1)]
        .groupby(["row_uid", "candidate_aspect"], sort=False)
        .size()
    )
    top["is_gold_aspect"] = [
        int(gold_counts.get((str(uid), str(aspect)), 0)) > 0
        for uid, aspect in zip(top["row_uid"], top["candidate_aspect"])
    ]
    selected = top[top["aspect_score"].astype(float).ge(aspect_threshold)]
    selected_gold = selected[selected["is_gold_aspect"]]
    return {
        "selected_aspect_instances": int(len(selected)),
        "selected_gold_aspect_instances": int(len(selected_gold)),
        "pipeline_conditional_sentiment_accuracy": (
            float(selected_gold["is_correct_pair"].mean())
            if len(selected_gold)
            else 0.0
        ),
    }


def _evaluation_partitions(fold: Any) -> dict[str, tuple[str, ...]]:
    evaluated = set(fold.evaluation_aspects)
    return {
        "overall": tuple(fold.evaluation_aspects),
        "seen": tuple(
            aspect for aspect in fold.seen_aspects if aspect in evaluated
        ),
        "heldout": tuple(
            aspect for aspect in fold.heldout_aspects if aspect in evaluated
        ),
    }


def _combine_l3_cached_seen_scores(
    full_sentiment_grid: pd.DataFrame,
    cached_seen_scored: pd.DataFrame,
    heldout_scored: pd.DataFrame,
) -> pd.DataFrame:
    identity = ["row_uid", "candidate_aspect", "candidate_sentiment"]
    combined = pd.concat(
        [cached_seen_scored, heldout_scored], ignore_index=True
    )
    expected = {
        tuple(str(row[column]) for column in identity)
        for _, row in full_sentiment_grid.iterrows()
    }
    observed = {
        tuple(str(row[column]) for column in identity)
        for _, row in combined.iterrows()
    }
    if combined.duplicated(identity).any() or observed != expected:
        raise AssertionError(
            "L3 seen-score reuse did not reconstruct the exact evaluation grid."
        )
    return combined


def _missing_resume_conditions(
    value: dict[str, object],
    *,
    method: str,
    level: str,
    fold: Any,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    expected = {
        "protocol_id": "taxonomy_two_stage_precloud_v2",
        "method_id": method,
        "level": level,
        "fold_id": fold.fold_id,
        "training_scope_id": training_scope_id(fold),
    }
    mismatched = {
        key: (value.get(key), expected_value)
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    conditions = value.get("conditions")
    if mismatched or not isinstance(conditions, dict):
        raise ValueError(
            f"Resume protocol mismatch for {fold.fold_id}: {mismatched}."
        )
    return tuple(condition for condition in requested if condition not in conditions)


def _evaluate(
    scored: pd.DataFrame,
    *,
    aspects: tuple[str, ...],
    aspect_threshold: float,
    second_sentiment_threshold: float,
) -> dict[str, object]:
    subset = scored[scored["candidate_aspect"].isin(aspects)].copy()
    primary = capped_two_sentiment_prediction_mask(
        subset,
        aspect_threshold=aspect_threshold,
        second_sentiment_threshold=second_sentiment_threshold,
    )
    argmax = two_stage_prediction_mask(subset, aspect_threshold)
    counts = (
        subset.loc[primary]
        .groupby(["row_uid", "candidate_aspect"], sort=False)
        .size()
    )
    if len(counts) and int(counts.max()) > 2:
        raise AssertionError("Capped-two emitted a third sentiment.")
    return {
        "primary_capped_two": evaluate_prediction_mask(
            subset, primary, aspects=aspects
        ),
        "argmax_control": evaluate_prediction_mask(subset, argmax, aspects=aspects),
        "oracle_gated_sentiment": conditional_sentiment_metrics(subset),
        "pipeline_gated_sentiment": _pipeline_conditional_sentiment_accuracy(
            subset, aspect_threshold=aspect_threshold
        ),
        "evaluation_rows": int(subset["row_uid"].nunique()),
        "maximum_sentiments_per_selected_aspect": (
            int(counts.max()) if len(counts) else 0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    validation = frame[frame["original_split"].eq("validation")].copy()
    validation["supervision_labels"] = validation["labels"]
    resource = load_description_bundle(require_approved=True)
    aspects = canonical_aspects()
    folds = registered_folds(args.level)
    if args.fold_id:
        folds = [value for value in folds if value.fold_id == args.fold_id]
        if not folds:
            raise ValueError(f"Unknown {args.level} fold: {args.fold_id!r}.")
    root = args.output_root.resolve() / args.method / args.level.lower()
    e5 = (
        E5ValidationCache(validation, local_files_only=args.local_files_only)
        if args.method == "e5_base_v2"
        else None
    )
    qwen = (
        ReadOnlyQwenTwoStageCache(
            args.qwen_cache_path,
            expected_file_sha256=args.qwen_cache_sha256,
        )
        if args.method == "frozen_qwen_candidate_pair"
        else None
    )
    results: list[dict[str, object]] = []
    try:
        for fold in folds:
            path = root / "folds" / f"{fold.fold_id}.json"
            requested_conditions = (
                tuple(args.condition)
                if args.condition
                else tuple(fold.conditions)
            )
            existing_result: dict[str, object] | None = None
            conditions_to_run = requested_conditions
            if args.resume and path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                conditions_to_run = _missing_resume_conditions(
                    value,
                    method=args.method,
                    level=args.level,
                    fold=fold,
                    requested=requested_conditions,
                )
                if not conditions_to_run:
                    results.append(value)
                    print(f"RESUME {args.method} {fold.fold_id}", flush=True)
                    continue
                existing_result = value
                print(
                    f"AUGMENT {args.method} {fold.fold_id} "
                    f"conditions={','.join(conditions_to_run)}",
                    flush=True,
                )
            started = time.perf_counter()
            splits = build_taxonomy_fold_splits(
                frame, fold, evaluation_splits=("validation",)
            )
            train = splits["train"]
            evaluation = splits["validation"]
            if args.max_validation_rows is not None:
                uids = set(
                    validation.sort_values("row_uid", kind="stable")
                    .head(args.max_validation_rows)["row_uid"]
                    .astype(str)
                )
                validation_scope = validation[
                    validation["row_uid"].astype(str).isin(uids)
                ].copy()
                evaluation = evaluation[
                    evaluation["row_uid"].astype(str).isin(uids)
                ].copy()
            else:
                validation_scope = validation
            train_variants = {
                aspect: "name_and_description" for aspect in fold.seen_aspects
            }
            runtime = None
            if args.method == "strict_train_only_tfidf":
                runtime = TfidfTrueTwoStageRuntime(seed=13).fit(
                    build_aspect_grid(
                        train, fold.seen_aspects, train_variants, resource
                    ),
                    build_sentiment_grid(
                        train,
                        fold.seen_aspects,
                        train_variants,
                        resource,
                        gold_aspects_only=True,
                    ),
                )

            selection_aspects = build_aspect_grid(
                validation_scope,
                fold.seen_aspects,
                train_variants,
                resource,
            )
            selection_sentiments = build_sentiment_grid(
                validation_scope,
                fold.seen_aspects,
                train_variants,
                resource,
            )
            selection_scored = _score_grids(
                selection_aspects,
                selection_sentiments,
                runtime=runtime,
                e5=e5,
                qwen=qwen,
            )
            aspect_selection = select_two_stage_threshold(selection_scored)
            runner_selection = select_second_sentiment_threshold(
                selection_scored,
                aspect_threshold=aspect_selection.threshold,
            )

            condition_results: dict[str, object] = (
                dict(existing_result["conditions"])
                if existing_result is not None
                else {}
            )
            seen_hash: str | None = (
                str(existing_result["seen_score_sha256"])
                if existing_result is not None
                else None
            )
            cached_seen_scored: pd.DataFrame | None = None
            if existing_result is not None:
                if not np.isclose(
                    float(existing_result["aspect_threshold"]),
                    float(aspect_selection.threshold),
                    atol=0.0,
                    rtol=0.0,
                ) or not np.isclose(
                    float(existing_result["second_sentiment_threshold"]),
                    float(runner_selection.second_sentiment_threshold),
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise ValueError(
                        "Resume threshold conflict while augmenting conditions."
                    )
            for condition in conditions_to_run:
                if condition not in fold.conditions:
                    raise ValueError(
                        f"Condition {condition!r} is not registered for {fold.fold_id}."
                    )
                variants = _variants_for_condition(fold, condition)
                aspect_grid = build_aspect_grid(
                    evaluation, fold.evaluation_aspects, variants, resource
                )
                sentiment_grid = build_sentiment_grid(
                    evaluation, fold.evaluation_aspects, variants, resource
                )
                if fold.level == "L3" and cached_seen_scored is not None:
                    heldout = set(fold.heldout_aspects)
                    heldout_aspect_grid = aspect_grid[
                        aspect_grid["candidate_aspect"].isin(heldout)
                    ].copy()
                    heldout_sentiment_grid = sentiment_grid[
                        sentiment_grid["candidate_aspect"].isin(heldout)
                    ].copy()
                    heldout_scored = _score_grids(
                        heldout_aspect_grid,
                        heldout_sentiment_grid,
                        runtime=runtime,
                        e5=e5,
                        qwen=qwen,
                    )
                    scored = _combine_l3_cached_seen_scores(
                        sentiment_grid,
                        cached_seen_scored,
                        heldout_scored,
                    )
                else:
                    scored = _score_grids(
                        aspect_grid,
                        sentiment_grid,
                        runtime=runtime,
                        e5=e5,
                        qwen=qwen,
                    )
                seen_scored = scored[
                    scored["candidate_aspect"].isin(fold.seen_aspects)
                ].copy()
                if fold.level == "L3" and cached_seen_scored is None:
                    cached_seen_scored = seen_scored.copy()
                observed_seen_hash = _score_hash(seen_scored)
                if seen_hash is None:
                    seen_hash = observed_seen_hash
                elif observed_seen_hash != seen_hash:
                    raise AssertionError(
                        "Seen candidate scores changed across representation conditions."
                    )
                partitions = _evaluation_partitions(fold)
                condition_results[condition] = {
                    "representations": variants,
                    "seen_score_sha256": observed_seen_hash,
                    "partitions": {
                        name: _evaluate(
                            scored,
                            aspects=partition_aspects,
                            aspect_threshold=aspect_selection.threshold,
                            second_sentiment_threshold=(
                                runner_selection.second_sentiment_threshold
                            ),
                        )
                        for name, partition_aspects in partitions.items()
                        if partition_aspects
                    },
                }
            result: dict[str, object] = {
                "schema_version": "taxonomy_two_stage_precloud_fold_v1",
                "protocol_id": "taxonomy_two_stage_precloud_v2",
                "method_id": args.method,
                "level": args.level,
                "fold_id": fold.fold_id,
                "training_scope_id": training_scope_id(fold),
                "heldout_aspects": list(fold.heldout_aspects),
                "train_rows": int(len(train)),
                "validation_rows": int(evaluation["row_uid"].nunique()),
                "aspect_threshold": float(aspect_selection.threshold),
                "second_sentiment_threshold": float(
                    runner_selection.second_sentiment_threshold
                ),
                "threshold_selection_candidate_aspects": len(fold.seen_aspects),
                "threshold_selection_rows": int(
                    validation_scope["row_uid"].nunique()
                ),
                "selection_partition": "seen validation only",
                "conditions": condition_results,
                "seen_score_sha256": seen_hash,
                "failure_count": 0,
                "non_finite_value_count": 0,
                "resume_conflict_count": 0,
                "test_contract_count": 0,
                "seconds": float(
                    (float(existing_result.get("seconds", 0.0)) if existing_result else 0.0)
                    + time.perf_counter()
                    - started
                ),
            }
            _write_json(path, result)
            results.append(result)
            print(
                f"COMPLETE {args.method} {fold.fold_id} seconds={result['seconds']:.1f}",
                flush=True,
            )
    finally:
        if e5 is not None:
            e5.close()

    summary: dict[str, object] = {
        "schema_version": "taxonomy_two_stage_precloud_summary_v1",
        "protocol_id": "taxonomy_two_stage_precloud_v2",
        "method_id": args.method,
        "level": args.level,
        "completed_folds": len(results),
        "expected_folds": len(folds),
        "failed_folds": 0,
        "test_contract_count": 0,
        "fold_seconds": float(sum(float(value["seconds"]) for value in results)),
        "folds": [str(value["fold_id"]) for value in results],
    }
    if qwen is not None:
        summary["qwen_cache"] = qwen.summary()
    _write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--level", choices=LEVELS, required=True)
    parser.add_argument("--fold-id")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_two_stage_precloud_v2/validation",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument(
        "--qwen-cache-path",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_two_stage_validation_v1/study_c/"
        "frozen_qwen_candidate_pair/raw_cache/prompt_scores.jsonl",
    )
    parser.add_argument(
        "--qwen-cache-sha256",
        default="32d0b82e417803dec5a7cdc620f4c7e947b133adfda43992a8cc4491562506dc",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
