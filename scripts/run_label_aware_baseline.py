from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.baselines.label_aware import (
    CrossEncoderConfig,
    build_optimizer_and_scheduler,
    build_pair_examples,
    make_tokenizer_and_cross_encoder,
    make_train_loader,
    predictions_from_scores,
    score_candidate_grid,
    train_one_epoch,
)
from msc_project.baselines.transformer import set_seed
from msc_project.data.fabsa import default_data_dir, unique_labels
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS, build_heldout_aspect_split, load_all_fabsa
from msc_project.evaluation.metrics import binarize_labels, evaluate_pair_and_aspect, per_label_report


THRESHOLDS = [round(value / 100, 2) for value in range(5, 96, 1)]
SELECTION_KEYS = ("pair_samples_f1", "pair_micro_f1", "pair_macro_f1")


def limit_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return frame
    return frame.head(limit).copy()


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def score_thresholds(
    eval_df,
    candidate_labels: list[str],
    scores,
    max_predictions_per_row: int | None,
) -> tuple[dict[str, float], list[list[str]]]:
    rows = []
    best_predictions: list[list[str]] = []
    for threshold in THRESHOLDS:
        predictions = predictions_from_scores(
            scores,
            candidate_labels,
            threshold,
            max_predictions_per_row=max_predictions_per_row,
        )
        metrics = evaluate_pair_and_aspect(eval_df["supervision_pair_labels"].tolist(), predictions, candidate_labels)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)

    rows.sort(key=lambda row: tuple(row[key] for key in SELECTION_KEYS), reverse=True)
    best = rows[0]
    best_predictions = predictions_from_scores(
        scores,
        candidate_labels,
        best["threshold"],
        max_predictions_per_row=max_predictions_per_row,
    )
    return best, best_predictions


def run_strategy(
    frame,
    heldout_aspects: list[str],
    strategy: str,
    output_dir: Path,
    args,
    device: torch.device,
) -> dict[str, object]:
    splits = build_heldout_aspect_split(
        frame,
        heldout_aspects,
        strategy=strategy,
        eval_label_scope="heldout",
    )
    train_df = limit_frame(splits["train"], args.train_limit)
    validation_df = limit_frame(splits["validation"], args.eval_limit)
    test_df = limit_frame(splits["test"], args.eval_limit)

    train_candidate_labels = unique_labels([train_df], "supervision_pair_labels")
    eval_candidate_labels = candidate_pair_labels(heldout_aspects)
    train_examples = build_pair_examples(
        train_df,
        train_candidate_labels,
        "supervision_pair_labels",
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )

    config = CrossEncoderConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    tokenizer, model = make_tokenizer_and_cross_encoder(config)
    model.to(device)
    train_loader = make_train_loader(train_examples, tokenizer, config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_epoch = None
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config)
        validation_scores = score_candidate_grid(model, tokenizer, validation_df, eval_candidate_labels, config, device)
        validation_metrics, _ = score_thresholds(
            validation_df,
            eval_candidate_labels,
            validation_scores,
            args.max_predictions_per_row,
        )
        validation_metrics["epoch"] = epoch
        validation_metrics["train_loss"] = float(train_loss)
        history.append(validation_metrics)
        print(json.dumps({"strategy": strategy, **validation_metrics}, indent=2), flush=True)

        if best_epoch is None or tuple(validation_metrics[key] for key in SELECTION_KEYS) > tuple(
            best_epoch[key] for key in SELECTION_KEYS
        ):
            best_epoch = validation_metrics
            best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_scores = score_candidate_grid(model, tokenizer, test_df, eval_candidate_labels, config, device)
    test_predictions = predictions_from_scores(
        test_scores,
        eval_candidate_labels,
        best_epoch["threshold"],
        max_predictions_per_row=args.max_predictions_per_row,
    )
    test_metrics = evaluate_pair_and_aspect(test_df["supervision_pair_labels"].tolist(), test_predictions, eval_candidate_labels)
    _, y_true = binarize_labels(test_df["supervision_pair_labels"].tolist(), eval_candidate_labels)
    _, y_pred = binarize_labels(test_predictions, eval_candidate_labels)
    per_label_report(y_true, y_pred, eval_candidate_labels, output_path=output_dir / "best_test_per_label.csv")

    best_test = {
        **test_metrics,
        "threshold": float(best_epoch["threshold"]),
        "best_epoch": int(best_epoch["epoch"]),
        "train_examples": int(len(train_df)),
        "training_pairs": int(len(train_examples)),
        "validation_examples": int(len(validation_df)),
        "test_examples": int(len(test_df)),
        "train_candidate_pair_labels": int(len(train_candidate_labels)),
        "eval_candidate_pair_labels": int(len(eval_candidate_labels)),
    }
    summary = {
        "model": "candidate_pair_cross_encoder",
        "strategy": strategy,
        "eval_label_scope": "heldout",
        "heldout_aspects": heldout_aspects,
        "selection_metric": "validation pair_samples_f1, with pair_micro_f1 and pair_macro_f1 tie-breakers",
        "config": config.__dict__,
        "negatives_per_positive": int(args.negatives_per_positive),
        "max_predictions_per_row": args.max_predictions_per_row,
        "best_validation": best_epoch,
        "best_test": best_test,
        "history": history,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label-aware held-out-aspect FABSA baselines.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines" / "label_aware")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered", "both"], default="both")
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--max-predictions-per-row", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    frame = load_all_fabsa(args.data_dir)
    heldout_aspects = args.heldout_aspect or DEFAULT_HELDOUT_ASPECTS
    strategies = ["label_masked", "example_filtered"] if args.strategy == "both" else [args.strategy]

    summaries = {}
    for strategy in strategies:
        summaries[strategy] = run_strategy(
            frame,
            heldout_aspects,
            strategy,
            args.output_dir / strategy,
            args,
            device,
        )

    write_json(summaries, args.output_dir / "summary.json")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Saved label-aware baseline outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
