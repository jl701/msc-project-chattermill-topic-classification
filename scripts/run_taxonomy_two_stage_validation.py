"""Run the preregistered validation-only true two-stage taxonomy study."""

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
    canonical_aspects,
    registered_folds,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    conditional_sentiment_metrics,
    evaluate_prediction_mask,
    select_two_stage_threshold,
    two_stage_prediction_mask,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    REPRESENTATION_VARIANTS,
    TfidfTrueTwoStageRuntime,
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
    representation_map,
)
from msc_project.experiments.taxonomy_methods import resolve_method_spec  # noqa: E402
from msc_project.llm.qwen_pair_classifier import load_frozen_qwen_pair  # noqa: E402
from msc_project.llm.qwen_two_stage_classifier import (  # noqa: E402
    qwen_two_stage_contract_sha256,
    score_two_stage_prompts,
    unique_missing_positions,
    validate_sentiment_verbalizer_token_ids,
)


METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "frozen_qwen_candidate_pair",
)


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
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _single_sentiment_frame(scored: pd.DataFrame) -> pd.DataFrame:
    gold_counts = (
        scored[scored["target"].astype(int).eq(1)]
        .groupby(["row_uid", "candidate_aspect"], sort=False)
        .size()
    )
    excluded = {
        (str(uid), str(aspect))
        for (uid, aspect), count in gold_counts.items()
        if int(count) > 1
    }
    keep = [
        (str(uid), str(aspect)) not in excluded
        for uid, aspect in zip(scored["row_uid"], scored["candidate_aspect"])
    ]
    return scored.loc[keep].copy()


def _evaluate(
    scored: pd.DataFrame,
    *,
    threshold: float,
    aspects: tuple[str, ...],
) -> dict[str, object]:
    subset = scored[scored["candidate_aspect"].isin(aspects)].copy()
    mask = two_stage_prediction_mask(subset, threshold)
    metrics = evaluate_prediction_mask(
        subset,
        mask.to_numpy(dtype=bool),
        aspects=aspects,
    )
    metrics.update(conditional_sentiment_metrics(subset))
    sensitivity = _single_sentiment_frame(subset)
    sensitivity_mask = two_stage_prediction_mask(sensitivity, threshold)
    sensitivity_metrics = evaluate_prediction_mask(
        sensitivity,
        sensitivity_mask.to_numpy(dtype=bool),
        aspects=aspects,
    )
    return {
        "metrics": metrics,
        "single_gold_sentiment_sensitivity": sensitivity_metrics,
        "evaluation_rows": int(subset["row_uid"].nunique()),
        "candidate_aspects": int(subset["candidate_aspect"].nunique()),
    }


class E5ScoreCache:
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
            uid: index for index, uid in enumerate(self.review_uids)
        }
        self.review_embeddings = self.encoder.encode(
            ordered["text"].astype(str).tolist(), role="review"
        )
        self.candidate_embeddings: dict[str, np.ndarray] = {}

    def _ensure_candidates(self, values: list[str]) -> None:
        missing = sorted(set(values) - set(self.candidate_embeddings))
        if not missing:
            return
        matrix = self.encoder.encode(missing, role="candidate")
        for candidate, embedding in zip(missing, matrix):
            self.candidate_embeddings[candidate] = embedding

    def score(self, grid: pd.DataFrame) -> np.ndarray:
        candidates = grid["candidate_text"].astype(str).tolist()
        self._ensure_candidates(candidates)
        values = np.asarray(
            [
                float(
                    self.review_embeddings[self.review_index[str(uid)]]
                    @ self.candidate_embeddings[str(candidate)]
                )
                for uid, candidate in zip(grid["row_uid"], candidates)
            ],
            dtype=float,
        )
        values = (np.clip(values, -1.0, 1.0) + 1.0) / 2.0
        if not np.isfinite(values).all():
            raise ValueError("E5 two-stage scores must be finite.")
        return values

    def close(self) -> None:
        self.encoder.close()


class QwenScoreCache:
    """Prompt-hash-only append cache for resumable frozen-Qwen inference."""

    def __init__(
        self,
        cache_path: Path,
        *,
        local_files_only: bool,
        max_length: int = 384,
        batch_size: int = 6,
        checkpoint_prompts: int = 96,
    ) -> None:
        self.cache_path = cache_path
        self.local_files_only = local_files_only
        self.max_length = max_length
        self.batch_size = batch_size
        self.checkpoint_prompts = checkpoint_prompts
        self.contract_sha256 = qwen_two_stage_contract_sha256(
            max_length=max_length
        )
        self.values: dict[str, list[float]] = {}
        self.tokenizer = None
        self.model = None
        self.aspect_ids = None
        self.sentiment_ids = None
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("contract_sha256") != self.contract_sha256:
                    raise ValueError("Frozen-Qwen cache prompt contract mismatch.")
                self.values[str(record["key"])] = [
                    float(value) for value in record["values"]
                ]

    def _key(self, review: str, candidate: str, mode: str) -> str:
        payload = json.dumps(
            [self.contract_sha256, mode, review, candidate],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.model is not None:
            return
        spec = resolve_method_spec("frozen_qwen_candidate_pair")
        if not spec.model_id or not spec.model_revision:
            raise ValueError("Frozen-Qwen registry entry is not pinned.")
        tokenizer, model, aspect_ids = load_frozen_qwen_pair(
            spec.model_id,
            revision=spec.model_revision,
            load_in_4bit=True,
            local_files_only=self.local_files_only,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.aspect_ids = aspect_ids
        self.sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)

    def score(self, grid: pd.DataFrame, *, mode: str) -> np.ndarray:
        if mode not in {"aspect", "sentiment"}:
            raise ValueError(f"Unknown Qwen cache mode: {mode!r}.")
        reviews = grid["text"].astype(str).tolist()
        candidates = grid["candidate_text"].astype(str).tolist()
        keys = [
            self._key(review, candidate, mode)
            for review, candidate in zip(reviews, candidates)
        ]
        # Multiple dataset rows may render byte-identical prompts.  Score each
        # missing prompt hash only once, including within the same checkpoint
        # chunk, then fan the cached value back out to every row.
        missing_positions = unique_missing_positions(keys, set(self.values))
        if missing_positions:
            self._load()
            assert self.model is not None and self.tokenizer is not None
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                for start in range(0, len(missing_positions), self.checkpoint_prompts):
                    positions = missing_positions[
                        start : start + self.checkpoint_prompts
                    ]
                    probabilities = score_two_stage_prompts(
                        self.model,
                        self.tokenizer,
                        [reviews[index] for index in positions],
                        [candidates[index] for index in positions],
                        mode=mode,
                        max_length=self.max_length,
                        batch_size=self.batch_size,
                        aspect_verbalizer_ids=self.aspect_ids,
                        sentiment_verbalizer_ids=self.sentiment_ids,
                    )
                    for index, row in zip(positions, probabilities):
                        key = keys[index]
                        values = [float(value) for value in row]
                        if not np.isfinite(values).all():
                            raise ValueError("Frozen-Qwen cache received non-finite scores.")
                        self.values[key] = values
                        handle.write(
                            json.dumps(
                                {
                                    "contract_sha256": self.contract_sha256,
                                    "key": key,
                                    "mode": mode,
                                    "values": values,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    handle.flush()
                    print(
                        f"QWEN_CACHE mode={mode} complete={min(start + len(positions), len(missing_positions))}/{len(missing_positions)} total={len(self.values)}",
                        flush=True,
                    )
        width = 2 if mode == "aspect" else 3
        output = np.asarray([self.values[key] for key in keys], dtype=float)
        if output.shape != (len(grid), width) or not np.isfinite(output).all():
            raise ValueError("Frozen-Qwen cached score matrix is invalid.")
        return output

    def close(self) -> None:
        if self.model is None:
            return
        import torch

        try:
            self.model.to("cpu")
        except (AttributeError, ValueError):
            pass
        self.model = None
        self.tokenizer = None
        self.aspect_ids = None
        self.sentiment_ids = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _aggregate(fold_results: list[dict[str, object]]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for fold in fold_results:
        for variant, variant_result in fold["variants"].items():  # type: ignore[union-attr]
            for partition, partition_result in variant_result["partitions"].items():  # type: ignore[union-attr]
                metrics = partition_result["metrics"]
                sensitivity = partition_result[
                    "single_gold_sentiment_sensitivity"
                ]
                records.append(
                    {
                        "fold_id": fold["fold_id"],
                        "heldout_aspect": fold["heldout_aspect"],
                        "variant": variant,
                        "partition": partition,
                        "pair_micro_f1": metrics["pair_micro_f1"],
                        "pair_micro_precision": metrics["pair_micro_precision"],
                        "pair_micro_recall": metrics["pair_micro_recall"],
                        "aspect_micro_f1": metrics["aspect_micro_f1"],
                        "conditional_sentiment_accuracy": metrics[
                            "conditional_sentiment_accuracy_all_gold_aspects"
                        ],
                        "single_gold_sentiment_pair_micro_f1": sensitivity[
                            "pair_micro_f1"
                        ],
                        "single_gold_sentiment_aspect_micro_f1": sensitivity[
                            "aspect_micro_f1"
                        ],
                        "prediction_cardinality": (
                            metrics["pair_predicted_label_count"]
                            / max(1, partition_result["evaluation_rows"])
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    grouped = (
        frame.groupby(["variant", "partition"], sort=True)
        .agg(
            folds=("fold_id", "nunique"),
            pair_micro_f1_mean=("pair_micro_f1", "mean"),
            pair_micro_f1_std=("pair_micro_f1", "std"),
            pair_micro_precision_mean=("pair_micro_precision", "mean"),
            pair_micro_recall_mean=("pair_micro_recall", "mean"),
            aspect_micro_f1_mean=("aspect_micro_f1", "mean"),
            conditional_sentiment_accuracy_mean=(
                "conditional_sentiment_accuracy",
                "mean",
            ),
            prediction_cardinality_mean=("prediction_cardinality", "mean"),
            single_gold_sentiment_pair_micro_f1_mean=(
                "single_gold_sentiment_pair_micro_f1",
                "mean",
            ),
            single_gold_sentiment_aspect_micro_f1_mean=(
                "single_gold_sentiment_aspect_micro_f1",
                "mean",
            ),
        )
        .reset_index()
    )
    return {
        "records": frame.to_dict(orient="records"),
        "aggregate": grouped.fillna(0.0).to_dict(orient="records"),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.method not in METHODS:
        raise ValueError(f"Unsupported method: {args.method!r}.")
    output_root = args.output_root.resolve()
    method_root = output_root / "study_c" / args.method
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    validation_rows = frame[frame["original_split"].eq("validation")].copy()
    resource = load_description_bundle(require_approved=True)
    aspects = canonical_aspects()
    folds = registered_folds("L2")
    if args.fold_id:
        folds = [fold for fold in folds if fold.fold_id == args.fold_id]
        if not folds:
            raise ValueError(f"Unknown L2 fold: {args.fold_id!r}.")

    pending_execution = any(
        not (
            args.resume
            and (method_root / "folds" / f"{fold.fold_id}.json").is_file()
        )
        for fold in folds
    )
    e5_cache = (
        E5ScoreCache(validation_rows, local_files_only=args.local_files_only)
        if args.method == "e5_base_v2" and pending_execution
        else None
    )
    qwen_cache = (
        QwenScoreCache(
            method_root / "raw_cache" / "prompt_scores.jsonl",
            local_files_only=args.local_files_only,
        )
        if args.method == "frozen_qwen_candidate_pair"
        else None
    )
    fold_results: list[dict[str, object]] = []
    try:
        for fold in folds:
            result_path = method_root / "folds" / f"{fold.fold_id}.json"
            if result_path.is_file() and args.resume:
                fold_results.append(json.loads(result_path.read_text(encoding="utf-8")))
                print(f"RESUME {fold.fold_id}", flush=True)
                continue
            started = time.perf_counter()
            splits = build_taxonomy_fold_splits(
                frame,
                fold,
                evaluation_splits=("validation",),
            )
            train_rows = splits["train"]
            eval_rows = splits["validation"]
            if args.max_validation_rows is not None:
                eval_rows = (
                    eval_rows.assign(_uid=eval_rows["row_uid"].astype(str))
                    .sort_values("_uid", kind="stable")
                    .head(args.max_validation_rows)
                    .drop(columns="_uid")
                )
            train_variants = {
                aspect: "name_and_description" for aspect in fold.seen_aspects
            }
            if args.method == "strict_train_only_tfidf":
                train_aspects = build_aspect_grid(
                    train_rows,
                    fold.seen_aspects,
                    train_variants,
                    resource,
                )
                train_sentiments = build_sentiment_grid(
                    train_rows,
                    fold.seen_aspects,
                    train_variants,
                    resource,
                    gold_aspects_only=True,
                )
                runtime = TfidfTrueTwoStageRuntime(seed=13).fit(
                    train_aspects, train_sentiments
                )
            else:
                runtime = None

            variant_results: dict[str, object] = {}
            threshold: float | None = None
            for variant in args.variant or REPRESENTATION_VARIANTS:
                variants = representation_map(
                    aspects, fold.heldout_aspects, variant
                )
                aspect_grid = build_aspect_grid(
                    eval_rows, aspects, variants, resource
                )
                sentiment_grid = build_sentiment_grid(
                    eval_rows, aspects, variants, resource
                )
                if runtime is not None:
                    aspect_scores = runtime.score_aspects(aspect_grid)
                    sentiment_scores = runtime.score_sentiments(sentiment_grid)
                elif e5_cache is not None:
                    assert e5_cache is not None
                    aspect_scores = e5_cache.score(aspect_grid)
                    sentiment_scores = e5_cache.score(sentiment_grid)
                else:
                    assert qwen_cache is not None
                    aspect_probabilities = qwen_cache.score(
                        aspect_grid, mode="aspect"
                    )
                    sentiment_probabilities = qwen_cache.score(
                        aspect_grid, mode="sentiment"
                    )
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
                        (str(uid), str(aspect), str(sentiment))
                        for (uid, aspect), probabilities in zip(
                            zip(
                                aspect_grid["row_uid"],
                                aspect_grid["candidate_aspect"],
                            ),
                            sentiment_probabilities,
                        )
                        for sentiment in ("negative", "neutral", "positive")
                    ]
                    if observed_keys != expected_keys:
                        raise AssertionError(
                            "Aspect and sentiment grid ordering is inconsistent."
                        )
                    sentiment_scores = sentiment_probabilities.reshape(-1)
                scored = join_two_stage_scores(
                    aspect_grid,
                    sentiment_grid,
                    aspect_scores,
                    sentiment_scores,
                )
                seen_scored = scored[
                    scored["candidate_aspect"].isin(fold.seen_aspects)
                ].copy()
                selected = select_two_stage_threshold(seen_scored)
                if threshold is None:
                    threshold = selected.threshold
                elif not np.isclose(threshold, selected.threshold, atol=0.0, rtol=0.0):
                    raise AssertionError(
                        "Held-out representation changed the seen-only threshold."
                    )
                variant_results[variant] = {
                    "threshold": selected.threshold,
                    "threshold_selection_rows": int(
                        seen_scored["row_uid"].nunique()
                    ),
                    "partitions": {
                        "overall": _evaluate(
                            scored, threshold=selected.threshold, aspects=aspects
                        ),
                        "seen": _evaluate(
                            scored,
                            threshold=selected.threshold,
                            aspects=fold.seen_aspects,
                        ),
                        "heldout": _evaluate(
                            scored,
                            threshold=selected.threshold,
                            aspects=fold.heldout_aspects,
                        ),
                    },
                }
            fold_result: dict[str, object] = {
                "schema_version": "taxonomy_true_two_stage_fold_v1",
                "method_id": args.method,
                "fold_id": fold.fold_id,
                "heldout_aspect": fold.heldout_aspects[0],
                "train_rows": int(len(train_rows)),
                "validation_rows": int(len(eval_rows)),
                "test_contract_count": 0,
                "seconds": float(time.perf_counter() - started),
                "variants": variant_results,
            }
            _write_json(result_path, fold_result)
            fold_results.append(fold_result)
            print(
                f"COMPLETE {fold.fold_id} seconds={fold_result['seconds']:.1f}",
                flush=True,
            )
    finally:
        if e5_cache is not None:
            e5_cache.close()
        if qwen_cache is not None:
            qwen_cache.close()

    aggregate = _aggregate(fold_results)
    summary: dict[str, object] = {
        "schema_version": "taxonomy_true_two_stage_summary_v1",
        "method_id": args.method,
        "level": "L2",
        "completed_folds": len(fold_results),
        "failed_folds": 0,
        "test_contract_count": 0,
        "selection_partition": "seen validation only",
        **aggregate,
    }
    _write_json(method_root / "summary.json", summary)
    pd.DataFrame.from_records(aggregate["records"]).to_csv(
        method_root / "per_fold.csv", index=False
    )
    print(json.dumps(_jsonable(summary["aggregate"]), indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_two_stage_validation_v1",
    )
    parser.add_argument("--fold-id")
    parser.add_argument("--variant", action="append", choices=REPRESENTATION_VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-validation-rows", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
