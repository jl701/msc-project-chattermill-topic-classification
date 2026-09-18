"""Run the preregistered sustained RTX 4090 two-stage hardware gate.

This is an administrative stability test, not a scientific experiment.  It
opens only train and validation, scores Frozen Qwen repeatedly at batch eight,
and performs one complete 4,096-example formal-length QLoRA training pass.  No
adapter checkpoint is retained.
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
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_methods import resolve_method_spec  # noqa: E402
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    registered_folds,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
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
    "outputs/experimental/taxonomy_two_stage_4090_gate_v1/"
    "sustained_hardware_soak.json"
)
SCORING_BATCH_SIZE = 8
QLORA_TRAINING_BUDGET = 4096
QLORA_MAX_LENGTH = 384
QLORA_GRADIENT_ACCUMULATION_STEPS = 8


def validate_soak_configuration(
    *,
    scoring_seconds: int,
    query_count: int,
    scoring_batch_size: int,
    qlora_training_budget: int,
) -> None:
    if scoring_seconds < 60:
        raise ValueError("Sustained scoring must run for at least 60 seconds.")
    if query_count < scoring_batch_size or query_count % scoring_batch_size:
        raise ValueError("Query count must be a positive multiple of batch eight.")
    if scoring_batch_size != SCORING_BATCH_SIZE:
        raise ValueError("The preregistered 4090 scoring batch size is exactly eight.")
    if qlora_training_budget != QLORA_TRAINING_BUDGET:
        raise ValueError("The preregistered QLoRA soak uses exactly 4,096 examples.")


def probability_summary(values: np.ndarray, *, width: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != width
        or not np.isfinite(array).all()
        or not np.allclose(array.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError(f"Invalid {width}-way soak probabilities.")
    return {
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "standard_deviation": float(array.std()),
    }


def _sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_cuda_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_cuda_bytes() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated())


def _release_model(model: Any) -> None:
    import torch

    try:
        model.to("cpu")
    except (AttributeError, RuntimeError, ValueError):
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-id", default="l2-a01")
    parser.add_argument("--scoring-seconds", type=int, default=600)
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--scoring-batch-size", type=int, default=8)
    parser.add_argument("--qlora-training-budget", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_soak_configuration(
        scoring_seconds=args.scoring_seconds,
        query_count=args.query_count,
        scoring_batch_size=args.scoring_batch_size,
        qlora_training_budget=args.qlora_training_budget,
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The sustained 4090 gate requires CUDA.")
    folds = [fold for fold in registered_folds("L2") if fold.fold_id == args.fold_id]
    if len(folds) != 1:
        raise ValueError(f"Expected one registered L2 fold for {args.fold_id!r}.")
    fold = folds[0]
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    splits = build_taxonomy_fold_splits(
        frame, fold, evaluation_splits=("validation",)
    )
    resource = load_description_bundle(require_approved=True)
    run_started = time.perf_counter()
    run_started_at = datetime.now(timezone.utc)

    demonstrations = select_qwen_two_stage_demonstrations(
        splits["train"], fold.seen_aspects, resource, seed=13
    )
    heldout = set(fold.heldout_aspects)
    if any(
        demonstration.candidate_aspect in heldout
        for examples in demonstrations.values()
        for demonstration in examples
    ):
        raise AssertionError("A held-out aspect entered soak demonstrations.")
    queries = (
        splits["validation"]
        .assign(_uid=splits["validation"]["row_uid"].astype(str))
        .sort_values("_uid", kind="stable")
        .drop_duplicates("text", keep="first")
        .head(args.query_count)
    )
    if len(queries) != args.query_count:
        raise ValueError("Insufficient unique validation texts for the soak.")
    aspects = tuple(fold.heldout_aspects) + tuple(fold.seen_aspects)
    reviews = [str(row.text) for row in queries.itertuples(index=False)]
    candidates = [
        render_aspect_candidate(
            aspects[index % len(aspects)], "name_and_description", resource
        )
        for index in range(args.query_count)
    ]
    frozen_spec = resolve_method_spec("frozen_qwen_candidate_pair")
    frozen_tokenizer, frozen_model, aspect_ids = load_frozen_qwen_pair(
        str(frozen_spec.model_id),
        revision=frozen_spec.model_revision,
        load_in_4bit=True,
        local_files_only=args.local_files_only,
    )
    sentiment_ids = validate_sentiment_verbalizer_token_ids(frozen_tokenizer)
    _reset_cuda_peak()
    scoring_started = time.perf_counter()
    scoring_loops = 0
    aspect_min = math.inf
    aspect_max = -math.inf
    aspect_std_max = 0.0
    sentiment_min = math.inf
    sentiment_max = -math.inf
    sentiment_std_max = 0.0
    while time.perf_counter() - scoring_started < args.scoring_seconds:
        aspect_probabilities = score_two_stage_prompts(
            frozen_model,
            frozen_tokenizer,
            reviews,
            candidates,
            mode="aspect",
            max_length=1024,
            batch_size=args.scoring_batch_size,
            aspect_verbalizer_ids=aspect_ids,
            demonstrations=demonstrations["aspect"],
        )
        sentiment_probabilities = score_two_stage_prompts(
            frozen_model,
            frozen_tokenizer,
            reviews,
            candidates,
            mode="sentiment",
            max_length=1024,
            batch_size=args.scoring_batch_size,
            sentiment_verbalizer_ids=sentiment_ids,
            demonstrations=demonstrations["sentiment"],
        )
        aspect_summary = probability_summary(aspect_probabilities, width=2)
        sentiment_summary = probability_summary(sentiment_probabilities, width=3)
        aspect_min = min(aspect_min, aspect_summary["minimum"])
        aspect_max = max(aspect_max, aspect_summary["maximum"])
        aspect_std_max = max(
            aspect_std_max, aspect_summary["standard_deviation"]
        )
        sentiment_min = min(sentiment_min, sentiment_summary["minimum"])
        sentiment_max = max(sentiment_max, sentiment_summary["maximum"])
        sentiment_std_max = max(
            sentiment_std_max, sentiment_summary["standard_deviation"]
        )
        scoring_loops += 1
    _sync_cuda()
    scoring_elapsed = time.perf_counter() - scoring_started
    scoring_peak = _peak_cuda_bytes()
    scoring_prompts = scoring_loops * args.query_count * 2
    frozen_result = {
        "batch_size": args.scoring_batch_size,
        "max_length": 1024,
        "requested_seconds": args.scoring_seconds,
        "elapsed_seconds": scoring_elapsed,
        "loops": scoring_loops,
        "prompts": scoring_prompts,
        "prompts_per_second": scoring_prompts / scoring_elapsed,
        "cuda_peak_memory_bytes": scoring_peak,
        "aspect_probability_min": aspect_min,
        "aspect_probability_max": aspect_max,
        "aspect_probability_std_max": aspect_std_max,
        "sentiment_probability_min": sentiment_min,
        "sentiment_probability_max": sentiment_max,
        "sentiment_probability_std_max": sentiment_std_max,
    }
    _release_model(frozen_model)

    manifests = build_two_stage_training_manifests(
        splits["train"],
        fold.seen_aspects,
        resource,
        total_budget=args.qlora_training_budget,
        aspect_budget=args.qlora_training_budget // 2,
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
    if len(examples) != QLORA_TRAINING_BUDGET:
        raise AssertionError("The QLoRA soak manifest is not exactly 4,096 rows.")
    qlora_spec = resolve_method_spec("qwen_candidate_pair_qlora")
    qlora_config = QwenPairQLoRAConfig(
        load_in_4bit=True,
        gradient_checkpointing=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
    )
    qlora_tokenizer, qlora_model, _ = load_qwen_pair_qlora(
        str(qlora_spec.model_id),
        qlora_config,
        revision=qlora_spec.model_revision,
        local_files_only=args.local_files_only,
    )
    trainable_parameters = sum(
        int(parameter.numel())
        for parameter in qlora_model.parameters()
        if parameter.requires_grad
    )
    training_config = QwenPairTrainingConfig(
        max_length=QLORA_MAX_LENGTH,
        batch_size=1,
        gradient_accumulation_steps=QLORA_GRADIENT_ACCUMULATION_STEPS,
        epochs=1,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        seed=13,
    )
    _reset_cuda_peak()
    training_started = time.perf_counter()
    history = train_qwen_two_stage_adapter(
        qlora_model,
        qlora_tokenizer,
        examples,
        training_config,
    )
    _sync_cuda()
    training_elapsed = time.perf_counter() - training_started
    training_peak = _peak_cuda_bytes()
    if not history or any(
        not math.isfinite(float(record["train_loss"])) for record in history
    ):
        raise RuntimeError("QLoRA soak training history is empty or non-finite.")
    qlora_result = {
        "training_examples": len(examples),
        "training_seconds": training_elapsed,
        "training_examples_per_second": len(examples) / training_elapsed,
        "cuda_peak_memory_bytes": training_peak,
        "trainable_parameters": trainable_parameters,
        "max_length": QLORA_MAX_LENGTH,
        "batch_size": 1,
        "gradient_accumulation_steps": QLORA_GRADIENT_ACCUMULATION_STEPS,
        "epochs": 1,
        "learning_rate": 5e-6,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "history": history,
        "manifests": {
            task: training_manifest_summary(manifest)
            for task, manifest in manifests.items()
        },
        "checkpoint_retained": False,
        "checkpoint_path": None,
    }
    _release_model(qlora_model)

    result = {
        "schema_version": "taxonomy_two_stage_4090_hardware_soak_v1",
        "status": "pass",
        "started_at": run_started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - run_started,
        "official_splits_opened": ["train", "validation"],
        "include_official_test": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "formal_result": False,
        "fold_id": fold.fold_id,
        "training_scope_id": training_scope_id(fold),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "model_id": qlora_spec.model_id,
        "model_revision": qlora_spec.model_revision,
        "frozen_scoring": frozen_result,
        "qlora_training": qlora_result,
    }
    _write_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
