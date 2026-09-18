from __future__ import annotations

import argparse
import json
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
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    render_aspect_candidate,
    select_qwen_two_stage_demonstrations,
)
from msc_project.llm.qwen_pair_classifier import (  # noqa: E402
    load_frozen_qwen_pair,
)
from msc_project.llm.qwen_two_stage_classifier import (  # noqa: E402
    demonstrations_sha256,
    qwen_two_stage_few_shot_contract_sha256,
    score_two_stage_prompts,
    validate_sentiment_verbalizer_token_ids,
)


DEFAULT_OUTPUT = Path(
    "outputs/experimental/taxonomy_two_stage_precloud_v2/smoke/"
    "frozen_qwen_few_shot.json"
)


def _validate_probabilities(values: np.ndarray, *, width: int) -> None:
    if (
        values.ndim != 2
        or values.shape[1] != width
        or not np.isfinite(values).all()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError(f"Few-shot Qwen {width}-way probabilities are invalid.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-model train-only-demo Frozen-Qwen few-shot smoke."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-id", default="l2-a01")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    folds = [fold for fold in registered_folds("L2") if fold.fold_id == args.fold_id]
    if len(folds) != 1:
        raise ValueError(f"Expected one registered L2 fold for {args.fold_id!r}.")
    fold = folds[0]
    frame = load_official_fabsa_splits(
        args.data_dir, ("train", "validation")
    )
    splits = build_taxonomy_fold_splits(
        frame, fold, evaluation_splits=("validation",)
    )
    resource = load_description_bundle(require_approved=True)
    demonstrations = select_qwen_two_stage_demonstrations(
        splits["train"], fold.seen_aspects, resource, seed=13
    )
    heldout = set(fold.heldout_aspects)
    if any(
        value.candidate_aspect in heldout
        for values in demonstrations.values()
        for value in values
    ):
        raise AssertionError("A held-out aspect entered few-shot demonstrations.")
    queries = (
        splits["validation"]
        .assign(_uid=splits["validation"]["row_uid"].astype(str))
        .sort_values("_uid", kind="stable")
        .head(2)
    )
    query_aspects = (fold.heldout_aspects[0], fold.seen_aspects[0])
    reviews = [str(row.text) for row in queries.itertuples(index=False)]
    candidates = [
        render_aspect_candidate(aspect, "name_and_description", resource)
        for aspect in query_aspects
    ]
    spec = resolve_method_spec("frozen_qwen_candidate_pair")
    if not spec.model_id or not spec.model_revision:
        raise ValueError("Frozen-Qwen method registry entry is not pinned.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, aspect_ids = load_frozen_qwen_pair(
        spec.model_id,
        revision=spec.model_revision,
        load_in_4bit=True,
        local_files_only=args.local_files_only,
    )
    sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)
    aspect_probabilities = score_two_stage_prompts(
        model,
        tokenizer,
        reviews,
        candidates,
        mode="aspect",
        max_length=args.max_length,
        batch_size=args.batch_size,
        aspect_verbalizer_ids=aspect_ids,
        demonstrations=demonstrations["aspect"],
    )
    sentiment_probabilities = score_two_stage_prompts(
        model,
        tokenizer,
        reviews,
        candidates,
        mode="sentiment",
        max_length=args.max_length,
        batch_size=args.batch_size,
        sentiment_verbalizer_ids=sentiment_ids,
        demonstrations=demonstrations["sentiment"],
    )
    elapsed = time.perf_counter() - started
    _validate_probabilities(aspect_probabilities, width=2)
    _validate_probabilities(sentiment_probabilities, width=3)
    result = {
        "schema_version": "taxonomy_two_stage_qwen_few_shot_smoke_v1",
        "status": "pass",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "official_splits_opened": ["train", "validation"],
        "test_contract_count": 0,
        "fold_id": fold.fold_id,
        "training_scope_id": training_scope_id(fold),
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "prompt_contract_sha256": qwen_two_stage_few_shot_contract_sha256(
            max_length=args.max_length
        ),
        "demonstrations": {
            mode: {
                "count": len(values),
                "answers": [value.answer for value in values],
                "sha256": demonstrations_sha256(values),
                "heldout_overlap": [],
            }
            for mode, values in demonstrations.items()
        },
        "queries": {
            "count": len(reviews),
            "representations": ["name_and_description"] * len(reviews),
            "contains_one_heldout_and_one_seen_candidate": True,
        },
        "score_checks": {
            "aspect_shape": list(aspect_probabilities.shape),
            "aspect_min": float(aspect_probabilities.min()),
            "aspect_max": float(aspect_probabilities.max()),
            "sentiment_shape": list(sentiment_probabilities.shape),
            "sentiment_min": float(sentiment_probabilities.min()),
            "sentiment_max": float(sentiment_probabilities.max()),
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
