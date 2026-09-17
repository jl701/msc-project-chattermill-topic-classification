"""Run the post-supervisor Level 2 N/D/R validation-only local methods."""

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
from msc_project.data.fabsa import default_data_dir  # noqa: E402
from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_post_supervisor import (  # noqa: E402
    aggregate_l2_folds,
    evaluate_l2_condition,
    post_supervisor_l2_folds,
)
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    candidate_representation_variants,
    training_scope_id,
)
from msc_project.experiments.taxonomy_qwen_zero_shot_cache import (  # noqa: E402
    ReadOnlyQwenTwoStageCache,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    capped_two_sentiment_prediction_mask,
    select_second_sentiment_threshold,
    select_two_stage_threshold,
)
from msc_project.experiments.taxonomy_scientific_freeze import (  # noqa: E402
    row_pair_confusions,
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _score_hash(frame: pd.DataFrame) -> str:
    columns = (
        "row_uid",
        "candidate_aspect",
        "candidate_sentiment",
        "aspect_score",
        "sentiment_score",
    )
    ordered = frame[list(columns)].sort_values(
        ["row_uid", "candidate_aspect", "candidate_sentiment"], kind="stable"
    )
    payload = json.dumps(
        ordered.to_dict(orient="records"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        similarities = np.asarray(
            [
                float(
                    self.review_embeddings[self.review_index[str(uid)]]
                    @ self.candidate_embeddings[candidate]
                )
                for uid, candidate in zip(grid["row_uid"], candidates)
            ],
            dtype=float,
        )
        output = (np.clip(similarities, -1.0, 1.0) + 1.0) / 2.0
        if len(output) != len(grid) or not np.isfinite(output).all():
            raise ValueError("E5 Level 2 scores are invalid.")
        return output

    def close(self) -> None:
        self.encoder.close()


def _score_grids(
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    *,
    runtime: TfidfTrueTwoStageRuntime | None,
    e5: E5ValidationCache | None,
    qwen: ReadOnlyQwenTwoStageCache | None,
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
        expected = [
            (str(uid), str(aspect), str(sentiment))
            for uid, aspect, sentiment in zip(
                sentiment_grid["row_uid"],
                sentiment_grid["candidate_aspect"],
                sentiment_grid["candidate_sentiment"],
            )
        ]
        observed = [
            (str(uid), str(aspect), sentiment)
            for uid, aspect in zip(
                aspect_grid["row_uid"], aspect_grid["candidate_aspect"]
            )
            for sentiment in ("negative", "neutral", "positive")
        ]
        if expected != observed:
            raise AssertionError("Qwen aspect and sentiment grids are misaligned.")
        sentiment_scores = sentiment_probabilities.reshape(-1)
    else:
        raise AssertionError("No local Level 2 scorer was provided.")
    return join_two_stage_scores(
        aspect_grid,
        sentiment_grid,
        aspect_scores,
        sentiment_scores,
    )


def _variants(fold: Any, condition: str) -> dict[str, str]:
    return {
        aspect: VARIANT_MAP[variant]
        for aspect, variant in candidate_representation_variants(
            fold, condition
        ).items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started_all = time.perf_counter()
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    validation = frame[frame["original_split"].eq("validation")].copy()
    validation["supervision_labels"] = validation["labels"]
    resource = load_description_bundle(require_approved=True)
    folds = list(post_supervisor_l2_folds())
    if args.fold_id:
        folds = [fold for fold in folds if fold.fold_id == args.fold_id]
        if not folds:
            raise ValueError(f"Unknown Level 2 fold: {args.fold_id!r}.")

    output_root = args.output_root.resolve() / args.method
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
    fold_results: list[dict[str, object]] = []
    try:
        for fold in folds:
            result_path = output_root / "folds" / f"{fold.fold_id}.json"
            if args.resume and result_path.is_file():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    existing.get("protocol_id") != "taxonomy_level2_local_ndr_v1"
                    or existing.get("method_id") != args.method
                    or existing.get("training_scope_id") != training_scope_id(fold)
                    or existing.get("failure_count") != 0
                    or existing.get("test_contract_count") != 0
                ):
                    raise ValueError(f"Level 2 local resume conflict for {fold.fold_id}.")
                fold_results.append(existing)
                print(f"RESUME {args.method} {fold.fold_id}", flush=True)
                continue

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

            selection_aspect_grid = build_aspect_grid(
                validation_scope, fold.seen_aspects, train_variants, resource
            )
            selection_sentiment_grid = build_sentiment_grid(
                validation_scope, fold.seen_aspects, train_variants, resource
            )
            selection_scored = _score_grids(
                selection_aspect_grid,
                selection_sentiment_grid,
                runtime=runtime,
                e5=e5,
                qwen=qwen,
            )
            aspect_selection = select_two_stage_threshold(selection_scored)
            runner_selection = select_second_sentiment_threshold(
                selection_scored,
                aspect_threshold=aspect_selection.threshold,
            )

            conditions: dict[str, object] = {}
            seen_hash: str | None = None
            for condition in ("N", "D", "R"):
                variants = _variants(fold, condition)
                aspect_grid = build_aspect_grid(
                    evaluation, fold.evaluation_aspects, variants, resource
                )
                sentiment_grid = build_sentiment_grid(
                    evaluation, fold.evaluation_aspects, variants, resource
                )
                scored = _score_grids(
                    aspect_grid,
                    sentiment_grid,
                    runtime=runtime,
                    e5=e5,
                    qwen=qwen,
                )
                primary_mask = capped_two_sentiment_prediction_mask(
                    scored,
                    aspect_threshold=aspect_selection.threshold,
                    second_sentiment_threshold=(
                        runner_selection.second_sentiment_threshold
                    ),
                )
                if args.evidence_output_root is not None and condition in {"N", "D"}:
                    evidence = row_pair_confusions(
                        scored,
                        primary_mask,
                        aspects=fold.heldout_aspects,
                        method_id=args.method,
                        fold_id=fold.fold_id,
                        condition=condition,
                    )
                    _write_csv(
                        args.evidence_output_root.resolve()
                        / args.method
                        / fold.fold_id
                        / f"{condition}.csv",
                        evidence,
                    )
                observed_seen_hash = _score_hash(
                    scored[
                        scored["candidate_aspect"].isin(fold.seen_aspects)
                    ].copy()
                )
                if seen_hash is None:
                    seen_hash = observed_seen_hash
                elif seen_hash != observed_seen_hash:
                    raise AssertionError("Seen scores changed across N/D/R conditions.")
                conditions[condition] = evaluate_l2_condition(
                    scored,
                    fold,
                    aspect_threshold=aspect_selection.threshold,
                    second_sentiment_threshold=(
                        runner_selection.second_sentiment_threshold
                    ),
                )

            result: dict[str, object] = {
                "schema_version": "taxonomy_level2_local_ndr_fold_v1",
                "protocol_id": "taxonomy_level2_local_ndr_v1",
                "method_id": args.method,
                "fold_id": fold.fold_id,
                "training_scope_id": training_scope_id(fold),
                "heldout_aspect": fold.heldout_aspects[0],
                "train_rows": int(len(train)),
                "validation_rows": int(evaluation["row_uid"].nunique()),
                "selection_partition": "seen validation candidates only",
                "aspect_threshold": float(aspect_selection.threshold),
                "second_sentiment_threshold": float(
                    runner_selection.second_sentiment_threshold
                ),
                "seen_score_sha256": seen_hash,
                "conditions": conditions,
                "failure_count": 0,
                "non_finite_value_count": 0,
                "resume_conflict_count": 0,
                "test_contract_count": 0,
                "seconds": float(time.perf_counter() - started),
            }
            _write_json(result_path, result)
            fold_results.append(result)
            print(
                f"COMPLETE {args.method} {fold.fold_id} seconds={result['seconds']:.1f}",
                flush=True,
            )
    finally:
        if e5 is not None:
            e5.close()

    summary: dict[str, object] = {
        "schema_version": "taxonomy_level2_local_ndr_summary_v1",
        "protocol_id": "taxonomy_level2_local_ndr_v1",
        "method_id": args.method,
        "completed_folds": len(fold_results),
        "expected_folds": len(folds),
        "folds": [result["fold_id"] for result in fold_results],
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "test_contract_count": 0,
        "seconds": float(time.perf_counter() - started_all),
    }
    if len(fold_results) == 12:
        summary["aggregate"] = aggregate_l2_folds(fold_results)
    if qwen is not None:
        summary["qwen_cache"] = qwen.summary()
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--fold-id")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--evidence-output-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_post_supervisor_local_v1"
        / "level2_local_ndr",
    )
    parser.add_argument(
        "--qwen-cache-path",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_two_stage_precloud_v2"
        / "qwen_zero_shot_rich_seed"
        / "study_c"
        / "frozen_qwen_candidate_pair"
        / "raw_cache"
        / "prompt_scores.jsonl",
    )
    parser.add_argument(
        "--qwen-cache-sha256",
        default="08d276ac8915a929c3849dc92a54130df53a9835a8f0b36a0040ee7dbfad9db3",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
