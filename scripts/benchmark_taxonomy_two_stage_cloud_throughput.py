"""Benchmark representative true-two-stage cloud workloads before formal runs.

This program is an administrative performance gate, not a scientific result.
It opens only the official train and validation splits, uses one registered
L2 scope, and records enough evidence to freeze safe batch sizes and estimate
the complete validation-only workload.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_methods import resolve_method_spec  # noqa: E402
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    candidate_representation_variants,
    registered_folds,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage_distilbert import (  # noqa: E402
    DistilBertTrueTwoStageRuntime,
    TwoStageDistilBertConfig,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    render_aspect_candidate,
    select_qwen_two_stage_demonstrations,
)
from msc_project.experiments.taxonomy_two_stage_training import (  # noqa: E402
    build_two_stage_training_manifests,
    training_manifest_summary,
)
from msc_project.llm.qwen_pair_classifier import (  # noqa: E402
    QwenPairQLoRAConfig,
    QwenPairTrainingConfig,
    load_frozen_qwen_pair,
    load_qwen_pair_qlora,
)
from msc_project.llm.qwen_two_stage_classifier import (  # noqa: E402
    score_two_stage_prompts,
    train_qwen_two_stage_adapter,
    validate_sentiment_verbalizer_token_ids,
)


DEFAULT_OUTPUT = Path(
    "outputs/experimental/taxonomy_two_stage_cloud_v1/"
    "representative_throughput_benchmark.json"
)
DEFAULT_BATCH_SIZES = (2, 4, 6, 8)
METHOD_PRICE_PER_HOUR_USD = 0.24
TRAINING_SCOPE_COUNT = 26
LEARNING_RATE_COUNT = 3
TRAINING_BUDGET = 4096


def candidate_contract_counts() -> dict[str, int]:
    """Count unique scope/aspect/representation contracts across L1--L4.

    The key deliberately includes the training scope because QLoRA adapters and
    few-shot demonstrations differ by scope.  This also deduplicates the L1/L2
    shared scopes and the L4 Value scope that is identical to one L3 scope.
    """

    full: set[tuple[str, str, str]] = set()
    selection: set[tuple[str, str, str]] = set()
    for level in ("L1", "L2", "L3", "L4"):
        for fold in registered_folds(level):
            scope = training_scope_id(fold)
            for aspect in fold.seen_aspects:
                selection.add((scope, str(aspect), "name_and_description"))
            for condition in fold.conditions:
                variants = candidate_representation_variants(fold, condition)
                for aspect in fold.evaluation_aspects:
                    variant = {
                        "minimal": "name_and_description",
                        "name_only": "name_only",
                        "rich": "rich",
                    }[str(variants[str(aspect)])]
                    full.add((scope, str(aspect), variant))
    full |= selection
    if not selection <= full:
        raise AssertionError("Seen-only selection contracts left the full suite.")
    return {
        "full_unique_candidate_contracts": len(full),
        "seen_selection_candidate_contracts": len(selection),
        "selected_model_additional_candidate_contracts": len(full - selection),
        "unique_training_scopes": len({value[0] for value in full}),
    }


def formal_workload(
    *,
    validation_rows: int,
    unique_validation_texts: int,
) -> dict[str, Any]:
    if validation_rows < 1 or unique_validation_texts < 1:
        raise ValueError("Validation workload counts must be positive.")
    if unique_validation_texts > validation_rows:
        raise ValueError("Unique validation texts cannot exceed validation rows.")
    counts = candidate_contract_counts()
    full = counts["full_unique_candidate_contracts"]
    selection = counts["seen_selection_candidate_contracts"]
    additional = counts["selected_model_additional_candidate_contracts"]
    stages = 2
    frozen_prompts = full * unique_validation_texts * stages
    trainable_tuning_prompts = (
        selection * unique_validation_texts * stages * LEARNING_RATE_COUNT
    )
    trainable_final_prompts = additional * unique_validation_texts * stages
    qlora_training_examples = (
        TRAINING_SCOPE_COUNT * LEARNING_RATE_COUNT * TRAINING_BUDGET
    )
    qlora_optimizer_steps = qlora_training_examples // 8
    distilbert_training_examples = qlora_training_examples * 3
    distilbert_optimizer_steps = (
        TRAINING_SCOPE_COUNT
        * LEARNING_RATE_COUNT
        * ((2048 // 32) + (2048 // 32))
        * 3
    )
    return {
        **counts,
        "validation_rows": validation_rows,
        "unique_validation_texts": unique_validation_texts,
        "deduplication_basis": "exact rendered prompt hash",
        "stages_per_candidate": stages,
        "frozen_prompts_per_arm": frozen_prompts,
        "trainable_tuning_prompts": trainable_tuning_prompts,
        "trainable_selected_model_additional_prompts": trainable_final_prompts,
        "trainable_total_prompts_per_method": (
            trainable_tuning_prompts + trainable_final_prompts
        ),
        "qlora_training_examples": qlora_training_examples,
        "qlora_optimizer_steps": qlora_optimizer_steps,
        "distilbert_training_examples_across_three_epochs": (
            distilbert_training_examples
        ),
        "distilbert_optimizer_steps": distilbert_optimizer_steps,
    }


def duration_and_cost_estimate(
    workload: Mapping[str, Any],
    *,
    frozen_prompts_per_second: float,
    qlora_prompts_per_second: float,
    qlora_training_examples_per_second: float,
    distilbert_prompts_per_second: float,
    distilbert_training_examples_per_second: float,
    price_per_hour_usd: float = METHOD_PRICE_PER_HOUR_USD,
) -> dict[str, Any]:
    rates = {
        "frozen_prompts_per_second": frozen_prompts_per_second,
        "qlora_prompts_per_second": qlora_prompts_per_second,
        "qlora_training_examples_per_second": qlora_training_examples_per_second,
        "distilbert_prompts_per_second": distilbert_prompts_per_second,
        "distilbert_training_examples_per_second": (
            distilbert_training_examples_per_second
        ),
    }
    if price_per_hour_usd <= 0 or any(value <= 0 for value in rates.values()):
        raise ValueError("Throughput rates and hourly price must be positive.")

    frozen_seconds = (
        float(workload["frozen_prompts_per_arm"])
        / frozen_prompts_per_second
    )
    trainable_prompts = float(workload["trainable_total_prompts_per_method"])
    qlora_seconds = (
        float(workload["qlora_training_examples"])
        / qlora_training_examples_per_second
        + trainable_prompts / qlora_prompts_per_second
    )
    distilbert_seconds = (
        float(workload["distilbert_training_examples_across_three_epochs"])
        / distilbert_training_examples_per_second
        + trainable_prompts / distilbert_prompts_per_second
    )
    raw = {
        "frozen_few_shot": frozen_seconds,
        "qlora": qlora_seconds,
        "distilbert": distilbert_seconds,
    }
    # Formal planning keeps a 25% margin for model reloads, checkpoint writes,
    # cache validation and variation in prompt lengths.
    return {
        "rates": rates,
        "hourly_price_usd": price_per_hour_usd,
        "overhead_multiplier": 1.25,
        "methods": {
            method: {
                "raw_seconds": seconds,
                "planned_hours_with_25pct_margin": seconds * 1.25 / 3600.0,
                "planned_cost_usd": (
                    seconds * 1.25 / 3600.0 * price_per_hour_usd
                ),
            }
            for method, seconds in raw.items()
        },
    }


def _sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_cuda_bytes() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def _reset_cuda_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _assert_probabilities(values: np.ndarray, *, width: int) -> None:
    if (
        values.ndim != 2
        or values.shape[1] != width
        or not np.isfinite(values).all()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError(f"Invalid {width}-way benchmark probabilities.")


def _benchmark_qwen_inference(
    scorer: Callable[[str, int], np.ndarray],
    *,
    batch_sizes: Sequence[int],
    prompt_count: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        _reset_cuda_peak()
        _sync_cuda()
        started = time.perf_counter()
        aspect = scorer("aspect", int(batch_size))
        sentiment = scorer("sentiment", int(batch_size))
        _sync_cuda()
        seconds = time.perf_counter() - started
        _assert_probabilities(aspect, width=2)
        _assert_probabilities(sentiment, width=3)
        total_prompts = prompt_count * 2
        results.append(
            {
                "batch_size": int(batch_size),
                "prompts": total_prompts,
                "seconds": seconds,
                "prompts_per_second": total_prompts / seconds,
                "cuda_peak_memory_bytes": _peak_cuda_bytes(),
                "aspect_probability_std": float(aspect[:, 0].std()),
                "sentiment_probability_std": float(sentiment.std()),
            }
        )
    best = max(results, key=lambda value: float(value["prompts_per_second"]))
    return {
        "trials": results,
        "recommended_batch_size": int(best["batch_size"]),
        "recommended_prompts_per_second": float(best["prompts_per_second"]),
        "maximum_cuda_peak_memory_bytes": max(
            int(value["cuda_peak_memory_bytes"]) for value in results
        ),
    }


def _release_model(model: Any) -> None:
    import torch

    try:
        model.to("cpu")
    except (AttributeError, RuntimeError, ValueError):
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _frozen_few_shot_benchmark(
    *,
    train: Any,
    validation: Any,
    fold: Any,
    resource: Mapping[str, Any],
    batch_sizes: Sequence[int],
    query_count: int,
    local_files_only: bool,
) -> dict[str, Any]:
    demonstrations = select_qwen_two_stage_demonstrations(
        train, fold.seen_aspects, resource, seed=13
    )
    queries = (
        validation.assign(_uid=validation["row_uid"].astype(str))
        .sort_values("_uid", kind="stable")
        .drop_duplicates("text", keep="first")
        .head(query_count)
    )
    if len(queries) != query_count:
        raise ValueError("Insufficient unique validation texts for the benchmark.")
    aspects = tuple(fold.heldout_aspects) + tuple(fold.seen_aspects)
    reviews = [str(row.text) for row in queries.itertuples(index=False)]
    candidates = [
        render_aspect_candidate(
            aspects[index % len(aspects)], "name_and_description", resource
        )
        for index in range(query_count)
    ]
    spec = resolve_method_spec("frozen_qwen_candidate_pair")
    _reset_cuda_peak()
    load_started = time.perf_counter()
    tokenizer, model, aspect_ids = load_frozen_qwen_pair(
        str(spec.model_id),
        revision=spec.model_revision,
        load_in_4bit=True,
        local_files_only=local_files_only,
    )
    _sync_cuda()
    load_seconds = time.perf_counter() - load_started
    sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)

    # Warm both prompt modes outside timed trials.
    score_two_stage_prompts(
        model,
        tokenizer,
        reviews[:2],
        candidates[:2],
        mode="aspect",
        max_length=1024,
        batch_size=2,
        aspect_verbalizer_ids=aspect_ids,
        demonstrations=demonstrations["aspect"],
    )
    score_two_stage_prompts(
        model,
        tokenizer,
        reviews[:2],
        candidates[:2],
        mode="sentiment",
        max_length=1024,
        batch_size=2,
        sentiment_verbalizer_ids=sentiment_ids,
        demonstrations=demonstrations["sentiment"],
    )

    def scorer(mode: str, batch_size: int) -> np.ndarray:
        return score_two_stage_prompts(
            model,
            tokenizer,
            reviews,
            candidates,
            mode=mode,
            max_length=1024,
            batch_size=batch_size,
            aspect_verbalizer_ids=aspect_ids,
            sentiment_verbalizer_ids=sentiment_ids,
            demonstrations=demonstrations[mode],
        )

    inference = _benchmark_qwen_inference(
        scorer,
        batch_sizes=batch_sizes,
        prompt_count=query_count,
    )
    result = {
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "max_length": 1024,
        "query_count_per_stage": query_count,
        "load_seconds": load_seconds,
        "inference": inference,
    }
    _release_model(model)
    return result


def _qlora_benchmark(
    *,
    train: Any,
    validation: Any,
    fold: Any,
    resource: Mapping[str, Any],
    batch_sizes: Sequence[int],
    query_count: int,
    training_budget: int,
    local_files_only: bool,
) -> dict[str, Any]:
    manifests = build_two_stage_training_manifests(
        train,
        fold.seen_aspects,
        resource,
        total_budget=training_budget,
        aspect_budget=training_budget // 2,
        seed=13,
    )
    examples = [
        (
            str(row.text),
            str(row.candidate_text),
            "aspect" if task == "aspect_presence" else "sentiment",
            str(row.answer),
        )
        for task, manifest in manifests.items()
        for row in manifest.itertuples(index=False)
    ]
    spec = resolve_method_spec("qwen_candidate_pair_qlora")
    _reset_cuda_peak()
    load_started = time.perf_counter()
    tokenizer, model, aspect_ids = load_qwen_pair_qlora(
        str(spec.model_id),
        QwenPairQLoRAConfig(
            load_in_4bit=True,
            gradient_checkpointing=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.05,
        ),
        revision=spec.model_revision,
        local_files_only=local_files_only,
    )
    _sync_cuda()
    load_seconds = time.perf_counter() - load_started
    _reset_cuda_peak()
    train_started = time.perf_counter()
    history = train_qwen_two_stage_adapter(
        model,
        tokenizer,
        examples,
        QwenPairTrainingConfig(
            max_length=384,
            batch_size=1,
            gradient_accumulation_steps=8,
            epochs=1,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            seed=13,
        ),
    )
    _sync_cuda()
    train_seconds = time.perf_counter() - train_started
    if not history or any(
        not math.isfinite(float(record["train_loss"])) for record in history
    ):
        raise RuntimeError("Representative QLoRA training loss is invalid.")
    training_peak = _peak_cuda_bytes()
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)
    queries = (
        validation.assign(_uid=validation["row_uid"].astype(str))
        .sort_values("_uid", kind="stable")
        .drop_duplicates("text", keep="first")
        .head(query_count)
    )
    aspects = tuple(fold.heldout_aspects) + tuple(fold.seen_aspects)
    reviews = [str(row.text) for row in queries.itertuples(index=False)]
    candidates = [
        render_aspect_candidate(
            aspects[index % len(aspects)], "name_and_description", resource
        )
        for index in range(query_count)
    ]

    def scorer(mode: str, batch_size: int) -> np.ndarray:
        return score_two_stage_prompts(
            model,
            tokenizer,
            reviews,
            candidates,
            mode=mode,
            max_length=384,
            batch_size=batch_size,
            aspect_verbalizer_ids=aspect_ids,
            sentiment_verbalizer_ids=sentiment_ids,
        )

    inference = _benchmark_qwen_inference(
        scorer,
        batch_sizes=batch_sizes,
        prompt_count=query_count,
    )
    result = {
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "max_length": 384,
        "query_count_per_stage": query_count,
        "load_seconds": load_seconds,
        "training_budget": training_budget,
        "training_seconds": train_seconds,
        "training_examples_per_second": training_budget / train_seconds,
        "training_cuda_peak_memory_bytes": training_peak,
        "history": history,
        "manifests": {
            task: training_manifest_summary(value)
            for task, value in manifests.items()
        },
        "inference": inference,
    }
    _release_model(model)
    return result


def _distilbert_benchmark(
    *,
    train: Any,
    fold: Any,
    resource: Mapping[str, Any],
    training_budget: int,
    local_files_only: bool,
) -> dict[str, Any]:
    import torch

    manifests = build_two_stage_training_manifests(
        train,
        fold.seen_aspects,
        resource,
        total_budget=training_budget,
        aspect_budget=training_budget // 2,
        seed=13,
    )
    config = TwoStageDistilBertConfig(
        max_length=256,
        batch_size=32,
        eval_batch_size=96,
        learning_rate=3e-5,
        weight_decay=0.01,
        epochs=1,
        warmup_ratio=0.1,
        use_amp=True,
        seed=13,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = DistilBertTrueTwoStageRuntime(
        config,
        device=device,
        local_files_only=local_files_only,
    )
    _reset_cuda_peak()
    started = time.perf_counter()
    runtime.fit(manifests["aspect_presence"], manifests["sentiment"])
    _sync_cuda()
    training_seconds = time.perf_counter() - started
    training_peak = _peak_cuda_bytes()
    score_rows = min(96, len(manifests["aspect_presence"]))
    _reset_cuda_peak()
    score_started = time.perf_counter()
    aspect = runtime.score_aspects(manifests["aspect_presence"].head(score_rows))
    sentiment = runtime.score_sentiments(manifests["sentiment"].head(score_rows))
    _sync_cuda()
    score_seconds = time.perf_counter() - score_started
    if (
        aspect.shape != (score_rows,)
        or not np.isfinite(aspect).all()
        or not ((aspect >= 0.0) & (aspect <= 1.0)).all()
    ):
        raise ValueError("Representative DistilBERT aspect scores are invalid.")
    _assert_probabilities(sentiment, width=3)
    result = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "max_length": config.max_length,
        "training_batch_size": config.batch_size,
        "evaluation_batch_size": config.eval_batch_size,
        "training_budget": training_budget,
        "benchmark_epochs": 1,
        "formal_epochs": 3,
        "training_seconds": training_seconds,
        "training_examples_per_second": training_budget / training_seconds,
        "training_cuda_peak_memory_bytes": training_peak,
        "scoring_prompts": score_rows * 2,
        "scoring_seconds": score_seconds,
        "scoring_prompts_per_second": score_rows * 2 / score_seconds,
        "scoring_cuda_peak_memory_bytes": _peak_cuda_bytes(),
        "history": runtime.history,
    }
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-id", default="l2-a01")
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--qlora-training-budget", type=int, default=128)
    parser.add_argument("--distilbert-training-budget", type=int, default=512)
    parser.add_argument(
        "--price-per-hour-usd",
        type=float,
        default=METHOD_PRICE_PER_HOUR_USD,
        help="Observed all-in running Pod price used only for cost planning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="append",
        dest="batch_sizes",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.query_count < 8:
        raise ValueError("Representative query count must be at least eight.")
    if args.qlora_training_budget < 32 or args.qlora_training_budget % 16:
        raise ValueError("QLoRA benchmark budget must be a multiple of 16 >= 32.")
    if args.distilbert_training_budget < 96 or args.distilbert_training_budget % 2:
        raise ValueError("DistilBERT benchmark budget must be even and >= 96.")
    if args.price_per_hour_usd <= 0:
        raise ValueError("Hourly price must be positive.")
    batch_sizes = tuple(args.batch_sizes or DEFAULT_BATCH_SIZES)
    if not batch_sizes or any(value < 1 or value > 8 for value in batch_sizes):
        raise ValueError(
            "Benchmark batch sizes must lie in [1, 8]; larger batches triggered "
            "repeated software thermal throttling on the measured RTX 3090."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Representative cloud throughput benchmark requires CUDA.")
    folds = [fold for fold in registered_folds("L2") if fold.fold_id == args.fold_id]
    if len(folds) != 1:
        raise ValueError(f"Expected one registered L2 fold for {args.fold_id!r}.")
    fold = folds[0]
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    splits = build_taxonomy_fold_splits(
        frame, fold, evaluation_splits=("validation",)
    )
    validation_all = frame[frame["original_split"].eq("validation")].copy()
    workload = formal_workload(
        validation_rows=int(validation_all["row_uid"].astype(str).nunique()),
        unique_validation_texts=int(validation_all["text"].astype(str).nunique()),
    )
    resource = load_description_bundle(require_approved=True)
    started = time.perf_counter()
    frozen = _frozen_few_shot_benchmark(
        train=splits["train"],
        validation=splits["validation"],
        fold=fold,
        resource=resource,
        batch_sizes=batch_sizes,
        query_count=args.query_count,
        local_files_only=args.local_files_only,
    )
    distilbert = _distilbert_benchmark(
        train=splits["train"],
        fold=fold,
        resource=resource,
        training_budget=args.distilbert_training_budget,
        local_files_only=args.local_files_only,
    )
    qlora = _qlora_benchmark(
        train=splits["train"],
        validation=splits["validation"],
        fold=fold,
        resource=resource,
        batch_sizes=batch_sizes,
        query_count=args.query_count,
        training_budget=args.qlora_training_budget,
        local_files_only=args.local_files_only,
    )
    estimates = duration_and_cost_estimate(
        workload,
        frozen_prompts_per_second=float(
            frozen["inference"]["recommended_prompts_per_second"]
        ),
        qlora_prompts_per_second=float(
            qlora["inference"]["recommended_prompts_per_second"]
        ),
        qlora_training_examples_per_second=float(
            qlora["training_examples_per_second"]
        ),
        distilbert_prompts_per_second=float(
            distilbert["scoring_prompts_per_second"]
        ),
        distilbert_training_examples_per_second=float(
            distilbert["training_examples_per_second"]
        ),
        price_per_hour_usd=float(args.price_per_hour_usd),
    )
    result = {
        "schema_version": "taxonomy_two_stage_cloud_throughput_benchmark_v1",
        "status": "pass",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "official_splits_opened": ["train", "validation"],
        "include_official_test": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "formal_result": False,
        "purpose": "administrative throughput, memory and cost gate",
        "fold_id": fold.fold_id,
        "training_scope_id": training_scope_id(fold),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(0).total_memory
            ),
        },
        "batch_sizes_tested": list(batch_sizes),
        "workload": workload,
        "benchmarks": {
            "frozen_qwen_few_shot": frozen,
            "distilbert": distilbert,
            "qlora": qlora,
        },
        "formal_estimates": estimates,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
