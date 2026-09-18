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
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
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
from msc_project.experiments.taxonomy_two_stage_training import (  # noqa: E402
    build_two_stage_training_manifests,
    training_manifest_summary,
)


DEFAULT_OUTPUT = Path(
    "outputs/experimental/taxonomy_two_stage_precloud_v2/smoke/"
    "distilbert_true_two_stage.json"
)


def _assert_probabilities(values: np.ndarray, *, width: int) -> None:
    if (
        values.ndim != 2
        or values.shape[1] != width
        or not np.isfinite(values).all()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError(f"Invalid {width}-class smoke probabilities.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and score both true-two-stage DistilBERT heads on train only."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-id", default="l2-a01")
    parser.add_argument("--total-budget", type=int, default=48)
    parser.add_argument("--aspect-budget", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
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
        total_budget=args.total_budget,
        aspect_budget=args.aspect_budget,
        seed=13,
    )
    config = TwoStageDistilBertConfig(
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        epochs=args.epochs,
        use_amp=True,
        seed=13,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    runtime = DistilBertTrueTwoStageRuntime(
        config,
        device=device,
        local_files_only=args.local_files_only,
    )
    started = time.perf_counter()
    runtime.fit(manifests["aspect_presence"], manifests["sentiment"])
    aspect_probabilities = runtime.score_aspects(
        manifests["aspect_presence"].head(8)
    )
    sentiment_probabilities = runtime.score_sentiments(
        manifests["sentiment"].head(8)
    )
    elapsed = time.perf_counter() - started
    if (
        aspect_probabilities.shape != (8,)
        or not np.isfinite(aspect_probabilities).all()
        or not ((aspect_probabilities >= 0) & (aspect_probabilities <= 1)).all()
    ):
        raise ValueError("Invalid binary aspect-presence smoke probabilities.")
    _assert_probabilities(sentiment_probabilities, width=3)
    losses = [
        float(record["train_loss"])
        for history in runtime.history.values()
        for record in history
    ]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("DistilBERT smoke produced a non-finite training loss.")
    result = {
        "schema_version": "taxonomy_two_stage_distilbert_smoke_v1",
        "status": "pass",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "official_splits_opened": ["train"],
        "test_contract_count": 0,
        "fold_id": fold.fold_id,
        "training_scope_id": training_scope_id(fold),
        "heldout_aspects": list(fold.heldout_aspects),
        "device": str(device),
        "config_contract_sha256": config.contract_sha256,
        "manifests": {
            task: training_manifest_summary(value)
            for task, value in manifests.items()
        },
        "history": runtime.history,
        "score_checks": {
            "aspect_shape": list(aspect_probabilities.shape),
            "aspect_min": float(aspect_probabilities.min()),
            "aspect_max": float(aspect_probabilities.max()),
            "sentiment_shape": list(sentiment_probabilities.shape),
            "sentiment_row_sum_min": float(
                sentiment_probabilities.sum(axis=1).min()
            ),
            "sentiment_row_sum_max": float(
                sentiment_probabilities.sum(axis=1).max()
            ),
        },
        "elapsed_seconds": elapsed,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
