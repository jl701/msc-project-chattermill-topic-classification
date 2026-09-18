from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
from msc_project.experiments.taxonomy_two_stage_training import (  # noqa: E402
    build_two_stage_training_manifests,
    training_manifest_summary,
)
from msc_project.llm.qwen_pair_classifier import (  # noqa: E402
    QwenPairQLoRAConfig,
    QwenPairTrainingConfig,
    load_qwen_pair_qlora,
)
from msc_project.llm.qwen_two_stage_classifier import (  # noqa: E402
    score_two_stage_prompts,
    train_qwen_two_stage_adapter,
    validate_sentiment_verbalizer_token_ids,
)


DEFAULT_OUTPUT = Path(
    "outputs/experimental/taxonomy_two_stage_precloud_v2/smoke/"
    "qwen_true_two_stage_qlora.json"
)


def _finite_history(history: list[dict[str, object]]) -> bool:
    return bool(history) and all(
        math.isfinite(float(record["train_loss"]))
        and int(record["global_step"]) >= 1
        for record in history
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-update real QLoRA smoke for the shared two-stage adapter."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-id", default="l2-a01")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    folds = [fold for fold in registered_folds("L2") if fold.fold_id == args.fold_id]
    if len(folds) != 1:
        raise ValueError(f"Expected one registered L2 fold for {args.fold_id!r}.")
    fold = folds[0]
    frame = load_official_fabsa_splits(args.data_dir, ("train",))
    train = build_taxonomy_fold_splits(
        frame, fold, evaluation_splits=()
    )["train"]
    resource = load_description_bundle(require_approved=False)
    manifests = build_two_stage_training_manifests(
        train,
        fold.seen_aspects,
        resource,
        total_budget=12,
        aspect_budget=6,
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
    if not spec.model_id or not spec.model_revision:
        raise ValueError("QLoRA method registry entry is not pinned.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, aspect_ids = load_qwen_pair_qlora(
        spec.model_id,
        QwenPairQLoRAConfig(
            load_in_4bit=True,
            gradient_checkpointing=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.05,
        ),
        revision=spec.model_revision,
        local_files_only=args.local_files_only,
    )
    trainable_parameters = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    history = train_qwen_two_stage_adapter(
        model,
        tokenizer,
        examples,
        QwenPairTrainingConfig(
            max_length=args.max_length,
            batch_size=1,
            gradient_accumulation_steps=len(examples),
            epochs=1,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.0,
            max_grad_norm=1.0,
            seed=13,
        ),
    )
    if not _finite_history(history):
        raise RuntimeError("QLoRA smoke history is empty or non-finite.")
    model.eval()
    sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)
    aspect_row = manifests["aspect_presence"].iloc[0]
    sentiment_row = manifests["sentiment"].iloc[0]
    aspect_probabilities = score_two_stage_prompts(
        model,
        tokenizer,
        [str(aspect_row["text"])],
        [str(aspect_row["candidate_text"])],
        mode="aspect",
        max_length=args.max_length,
        batch_size=1,
        aspect_verbalizer_ids=aspect_ids,
    )
    sentiment_probabilities = score_two_stage_prompts(
        model,
        tokenizer,
        [str(sentiment_row["text"])],
        [str(sentiment_row["candidate_text"])],
        mode="sentiment",
        max_length=args.max_length,
        batch_size=1,
        sentiment_verbalizer_ids=sentiment_ids,
    )
    probabilities = np.concatenate(
        [aspect_probabilities.reshape(-1), sentiment_probabilities.reshape(-1)]
    )
    if not np.isfinite(probabilities).all():
        raise ValueError("Post-update QLoRA smoke probabilities are non-finite.")
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "taxonomy_two_stage_qwen_qlora_smoke_v1",
        "status": "pass",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "official_splits_opened": ["train"],
        "test_contract_count": 0,
        "formal_result": False,
        "purpose": "memory and execution gate only",
        "fold_id": fold.fold_id,
        "training_scope_id": training_scope_id(fold),
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "manifests": {
            task: training_manifest_summary(value)
            for task, value in manifests.items()
        },
        "training_examples": len(examples),
        "trainable_parameters": trainable_parameters,
        "history": history,
        "post_update_score_shapes": {
            "aspect": list(aspect_probabilities.shape),
            "sentiment": list(sentiment_probabilities.shape),
        },
        "elapsed_seconds": elapsed,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2))
    try:
        model.to("cpu")
    except (AttributeError, ValueError):
        pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
