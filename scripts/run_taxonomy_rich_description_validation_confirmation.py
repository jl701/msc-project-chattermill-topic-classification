"""Confirm the frozen selected rich interface once on official validation."""

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
    evaluate_l2_condition,
    post_supervisor_l2_folds,
)
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_rich_description import (  # noqa: E402
    canonical_json_sha256,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    select_second_sentiment_threshold,
    select_two_stage_threshold,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    TfidfTrueTwoStageRuntime,
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
)


METHODS = ("strict_train_only_tfidf", "e5_base_v2")
PROTOCOL_ID = "taxonomy_rich_description_validation_confirmation_v1"
SELECTED_INTERFACE_ID = "taxonomy_rich_description_selected_interface_v1"
CONDITIONS = ("D", "R")


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
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
    return canonical_json_sha256(ordered.to_dict(orient="records"))


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
        self.review_index = {
            str(uid): index for index, uid in enumerate(ordered["row_uid"])
        }
        self.review_embeddings = self.encoder.encode(
            ordered["text"].astype(str).tolist(), role="review"
        )
        self.candidate_embeddings: dict[str, np.ndarray] = {}

    def score(self, grid: pd.DataFrame) -> np.ndarray:
        candidates = grid["candidate_text"].astype(str).tolist()
        missing = sorted(set(candidates) - set(self.candidate_embeddings))
        if missing:
            embeddings = self.encoder.encode(missing, role="candidate")
            self.candidate_embeddings.update(dict(zip(missing, embeddings)))
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
            raise ValueError("E5 validation confirmation scores are invalid.")
        return output

    def close(self) -> None:
        self.encoder.close()


def _score_grids(
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    *,
    runtime: TfidfTrueTwoStageRuntime | None,
    e5: E5ValidationCache | None,
) -> pd.DataFrame:
    if runtime is not None:
        aspect_scores = runtime.score_aspects(aspect_grid)
        sentiment_scores = runtime.score_sentiments(sentiment_grid)
    elif e5 is not None:
        aspect_scores = e5.score(aspect_grid)
        sentiment_scores = e5.score(sentiment_grid)
    else:
        raise AssertionError("No confirmation scorer is available.")
    return join_two_stage_scores(
        aspect_grid, sentiment_grid, aspect_scores, sentiment_scores
    )


def _aggregate(folds: list[dict[str, object]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for fold in folds:
        conditions = fold["conditions"]
        for condition in CONDITIONS:
            result = conditions[condition]
            presence = result["L2_S"]["aspect_presence"]
            end_to_end = result["L2_E"]["partitions"]
            rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "heldout_aspect": fold["heldout_aspect"],
                    "condition": condition,
                    "presence_average_precision": presence["average_precision"],
                    "presence_f1": presence["f1"],
                    "presence_precision": presence["precision"],
                    "presence_recall": presence["recall"],
                    "false_positive_rows_per_100": presence[
                        "false_positive_rows_per_100"
                    ],
                    "heldout_pair_micro_f1": end_to_end["heldout"][
                        "pair_micro_f1"
                    ],
                    "overall_pair_micro_f1": end_to_end["overall"][
                        "pair_micro_f1"
                    ],
                }
            )
    frame = pd.DataFrame.from_records(rows)
    metrics = (
        "presence_average_precision",
        "presence_f1",
        "presence_precision",
        "presence_recall",
        "false_positive_rows_per_100",
        "heldout_pair_micro_f1",
        "overall_pair_micro_f1",
    )
    means = {
        condition: {
            metric: float(frame.loc[frame["condition"].eq(condition), metric].mean())
            for metric in metrics
        }
        for condition in CONDITIONS
    }
    pivot = frame.pivot(index="fold_id", columns="condition", values=list(metrics))
    contrasts = {
        metric: {
            "mean": float((pivot[(metric, "R")] - pivot[(metric, "D")]).mean()),
            "R_higher_folds": int(
                (pivot[(metric, "R")] > pivot[(metric, "D")]).sum()
            ),
            "equal_folds": int(
                (pivot[(metric, "R")] == pivot[(metric, "D")]).sum()
            ),
            "R_lower_folds": int(
                (pivot[(metric, "R")] < pivot[(metric, "D")]).sum()
            ),
        }
        for metric in metrics
    }
    return {
        "records": frame.to_dict(orient="records"),
        "condition_macro_means": means,
        "R_minus_D": contrasts,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started_all = time.perf_counter()
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    if set(frame["original_split"].astype(str)) != {"train", "validation"}:
        raise AssertionError("Confirmation loaded an unexpected official split.")
    validation = frame[frame["original_split"].eq("validation")].copy()
    validation["supervision_labels"] = validation["labels"]
    resource = load_description_bundle(require_approved=True)
    folds = list(post_supervisor_l2_folds())
    output_root = args.output_root.resolve() / args.method
    e5 = (
        E5ValidationCache(validation, local_files_only=args.local_files_only)
        if args.method == "e5_base_v2"
        else None
    )
    fold_results: list[dict[str, object]] = []
    try:
        for fold in folds:
            result_path = output_root / "folds" / f"{fold.fold_id}.json"
            if args.resume and result_path.is_file():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    existing.get("protocol_id") != PROTOCOL_ID
                    or existing.get("selected_interface_id") != SELECTED_INTERFACE_ID
                    or existing.get("method_id") != args.method
                    or existing.get("training_scope_id") != training_scope_id(fold)
                    or existing.get("failure_count") != 0
                    or existing.get("test_contract_count") != 0
                ):
                    raise ValueError(f"Confirmation resume conflict: {result_path}.")
                fold_results.append(existing)
                print(f"RESUME {args.method} {fold.fold_id}", flush=True)
                continue

            started = time.perf_counter()
            splits = build_taxonomy_fold_splits(
                frame, fold, evaluation_splits=("validation",)
            )
            train = splits["train"]
            evaluation = splits["validation"]
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
                validation, fold.seen_aspects, train_variants, resource
            )
            selection_sentiment_grid = build_sentiment_grid(
                validation, fold.seen_aspects, train_variants, resource
            )
            selection_scored = _score_grids(
                selection_aspect_grid,
                selection_sentiment_grid,
                runtime=runtime,
                e5=e5,
            )
            aspect_selection = select_two_stage_threshold(selection_scored)
            runner_selection = select_second_sentiment_threshold(
                selection_scored,
                aspect_threshold=aspect_selection.threshold,
            )

            conditions: dict[str, object] = {}
            seen_score_hash: str | None = None
            sentiment_score_hash: str | None = None
            for condition in CONDITIONS:
                aspect_variants = {
                    aspect: "name_and_description" for aspect in fold.evaluation_aspects
                }
                if condition == "R":
                    aspect_variants[fold.heldout_aspects[0]] = "rich_positive"
                sentiment_variants = {
                    aspect: "name_and_description" for aspect in fold.evaluation_aspects
                }
                aspect_grid = build_aspect_grid(
                    evaluation, fold.evaluation_aspects, aspect_variants, resource
                )
                sentiment_grid = build_sentiment_grid(
                    evaluation, fold.evaluation_aspects, sentiment_variants, resource
                )
                scored = _score_grids(
                    aspect_grid, sentiment_grid, runtime=runtime, e5=e5
                )
                observed_seen = _score_hash(
                    scored[scored["candidate_aspect"].isin(fold.seen_aspects)].copy()
                )
                observed_sentiment = canonical_json_sha256(
                    scored[
                        [
                            "row_uid",
                            "candidate_aspect",
                            "candidate_sentiment",
                            "sentiment_score",
                        ]
                    ]
                    .sort_values(
                        ["row_uid", "candidate_aspect", "candidate_sentiment"],
                        kind="stable",
                    )
                    .to_dict(orient="records")
                )
                if seen_score_hash is None:
                    seen_score_hash = observed_seen
                    sentiment_score_hash = observed_sentiment
                elif (
                    seen_score_hash != observed_seen
                    or sentiment_score_hash != observed_sentiment
                ):
                    raise AssertionError(
                        "Seen or Stage 2 scores changed across D/R confirmation."
                    )
                conditions[condition] = evaluate_l2_condition(
                    scored,
                    fold,
                    aspect_threshold=aspect_selection.threshold,
                    second_sentiment_threshold=(
                        runner_selection.second_sentiment_threshold
                    ),
                )

            result: dict[str, object] = {
                "schema_version": "taxonomy_rich_description_confirmation_fold_v1",
                "protocol_id": PROTOCOL_ID,
                "selected_interface_id": SELECTED_INTERFACE_ID,
                "method_id": args.method,
                "fold_id": fold.fold_id,
                "training_scope_id": training_scope_id(fold),
                "heldout_aspect": fold.heldout_aspects[0],
                "train_rows": int(len(train)),
                "validation_rows": int(len(evaluation)),
                "aspect_threshold": float(aspect_selection.threshold),
                "second_sentiment_threshold": float(
                    runner_selection.second_sentiment_threshold
                ),
                "seen_score_sha256": seen_score_hash,
                "sentiment_score_sha256": sentiment_score_hash,
                "conditions": conditions,
                "failure_count": 0,
                "non_finite_value_count": 0,
                "resume_conflict_count": 0,
                "test_contract_count": 0,
                "seconds": float(time.perf_counter() - started),
            }
            _write_json(result_path, result)
            fold_results.append(result)
            print(f"COMPLETE {args.method} {fold.fold_id}", flush=True)
    finally:
        if e5 is not None:
            e5.close()

    if len(fold_results) != 12:
        raise AssertionError("Confirmation requires all twelve Level 2 folds.")
    summary: dict[str, object] = {
        "schema_version": "taxonomy_rich_description_confirmation_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "selected_interface_id": SELECTED_INTERFACE_ID,
        "method_id": args.method,
        "completed_folds": len(fold_results),
        "expected_folds": 12,
        "folds": [value["fold_id"] for value in fold_results],
        "aggregate": _aggregate(fold_results),
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "test_contract_count": 0,
        "seconds": float(time.perf_counter() - started_all),
    }
    summary["summary_sha256"] = canonical_json_sha256(
        {key: value for key, value in summary.items() if key != "seconds"}
    )
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_rich_description_validation_confirmation_v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
