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

from msc_project.baselines.candidate_label import (
    SENTIMENT_MODES,
    build_sentiment_feature_lookup,
    candidate_pair_labels,
    pair_predictions_from_aspects,
    sentiment_lookup_from_features,
    train_candidate_sentiment_model,
)
from msc_project.baselines.label_aware import (
    CrossEncoderConfig,
    aspect_predictions_from_scores,
    build_aspect_examples,
    build_optimizer_and_scheduler,
    make_aspect_train_loader,
    make_tokenizer_and_cross_encoder,
    score_aspect_grid,
    train_one_epoch,
)
from msc_project.baselines.transformer import set_seed
from msc_project.baselines.transformer_sentiment import (
    TransformerAspectSentimentConfig,
    train_transformer_aspect_sentiment_model,
)
from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS, build_heldout_aspect_split, load_all_fabsa
from msc_project.evaluation.metrics import binarize_labels, evaluate_pair_and_aspect, per_label_report


THRESHOLDS = [round(value / 100, 2) for value in range(5, 96, 1)]
SELECTION_KEYS = ("pair_samples_f1", "pair_micro_f1", "pair_macro_f1")


def limit_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return frame
    return frame.head(limit).copy()


def flatten(rows):
    values = []
    for row in rows:
        values.extend(row)
    return values


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def selection_keys(primary_metric: str) -> tuple[str, ...]:
    if primary_metric not in SELECTION_KEYS:
        raise ValueError(f"Unknown selection metric: {primary_metric}")
    return tuple([primary_metric] + [key for key in SELECTION_KEYS if key != primary_metric])


def row_score_features(
    row_scores,
    candidate_aspects: list[str],
    selected_aspects: list[str],
    threshold: float,
) -> dict[str, object]:
    ranked_indices = sorted(range(len(candidate_aspects)), key=lambda index: float(row_scores[index]), reverse=True)
    top_index = ranked_indices[0]
    second_index = ranked_indices[1] if len(ranked_indices) > 1 else ranked_indices[0]
    selected_indices = [candidate_aspects.index(aspect) for aspect in selected_aspects if aspect in candidate_aspects]
    selected_scores = [float(row_scores[index]) for index in selected_indices]
    distances = [abs(float(score) - threshold) for score in row_scores]

    return {
        "candidate_aspect_scores": {
            aspect: float(row_scores[index])
            for index, aspect in enumerate(candidate_aspects)
        },
        "threshold": float(threshold),
        "top_aspect": candidate_aspects[top_index],
        "top_score": float(row_scores[top_index]),
        "second_score": float(row_scores[second_index]),
        "score_margin": float(row_scores[top_index] - row_scores[second_index]),
        "min_abs_distance_to_threshold": float(min(distances) if distances else 0.0),
        "top_distance_to_threshold": float(row_scores[top_index] - threshold),
        "above_threshold_count": int(sum(float(score) >= threshold for score in row_scores)),
        "selected_count": int(len(selected_indices)),
        "selected_score_min": float(min(selected_scores)) if selected_scores else None,
        "selected_score_mean": float(sum(selected_scores) / len(selected_scores)) if selected_scores else None,
    }


def write_prediction_rows(
    frame: pd.DataFrame,
    predictions: list[list[str]],
    path: Path,
    aspect_scores=None,
    candidate_aspects: list[str] | None = None,
    selected_aspects: list[list[str]] | None = None,
    threshold: float | None = None,
    sentiment_features: list[dict[str, dict[str, object]]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_index, (_, row) in enumerate(frame.iterrows()):
            payload = {
                "row_index": row_index,
                "id": str(row["id"]),
                "row_uid": str(row.get("row_uid", "")),
                "original_split": str(row.get("original_split", "")),
                "org_index": int(row["org_index"]),
                "text": row["text"],
                "gold_pair_labels": row["supervision_pair_labels"],
                "pred_pair_labels": predictions[row_index],
            }
            if aspect_scores is not None and candidate_aspects is not None and selected_aspects is not None and threshold is not None:
                payload["score_features"] = row_score_features(
                    aspect_scores[row_index],
                    candidate_aspects,
                    selected_aspects[row_index],
                    float(threshold),
                )
            if sentiment_features is not None:
                payload["sentiment_features"] = sentiment_features[row_index]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def score_thresholds(
    eval_df,
    candidate_aspects: list[str],
    candidate_pairs: list[str],
    aspect_scores,
    sentiment_lookup: list[dict[str, str]],
    max_predictions_per_row: int | None,
    ensure_one: bool,
    selection_metric: str,
) -> tuple[dict[str, float], list[list[str]]]:
    rows = []
    for threshold in THRESHOLDS:
        aspect_predictions = aspect_predictions_from_scores(
            aspect_scores,
            candidate_aspects,
            threshold,
            ensure_one=ensure_one,
            max_predictions_per_row=max_predictions_per_row,
        )
        predictions = pair_predictions_from_aspects(aspect_predictions, sentiment_lookup)
        metrics = evaluate_pair_and_aspect(eval_df["supervision_pair_labels"].tolist(), predictions, candidate_pairs)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)

    selected_keys = selection_keys(selection_metric)
    rows.sort(key=lambda row: tuple(row[key] for key in selected_keys), reverse=True)
    best = rows[0]
    best_aspects = aspect_predictions_from_scores(
        aspect_scores,
        candidate_aspects,
        best["threshold"],
        ensure_one=ensure_one,
        max_predictions_per_row=max_predictions_per_row,
    )
    return best, pair_predictions_from_aspects(best_aspects, sentiment_lookup)


def run_strategy(
    frame,
    heldout_aspects: list[str],
    strategy: str,
    output_dir: Path,
    args,
    device: torch.device,
    eval_row_scope: str = "containing_heldout",
) -> dict[str, object]:
    splits = build_heldout_aspect_split(
        frame,
        heldout_aspects,
        strategy=strategy,
        eval_label_scope="heldout",
        eval_row_scope=eval_row_scope,
    )
    train_df = limit_frame(splits["train"], args.train_limit)
    validation_df = limit_frame(splits["validation"], args.eval_limit)
    test_df = limit_frame(splits["test"], args.eval_limit)
    ensure_one = not args.allow_empty_predictions

    train_aspects = sorted(set(flatten(train_df["supervision_aspect_labels"])))
    eval_candidate_pairs = candidate_pair_labels(heldout_aspects)
    train_examples = build_aspect_examples(
        train_df,
        train_aspects,
        "supervision_aspect_labels",
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    sentiment_summary = None
    if args.sentiment_mode == "transformer_aspect_conditioned":
        sentiment_config = TransformerAspectSentimentConfig(
            model_name=args.sentiment_model_name,
            max_length=args.sentiment_max_length,
            batch_size=args.sentiment_batch_size,
            eval_batch_size=args.sentiment_eval_batch_size,
            learning_rate=args.sentiment_learning_rate,
            weight_decay=args.sentiment_weight_decay,
            epochs=args.sentiment_epochs,
            warmup_ratio=args.sentiment_warmup_ratio,
            seed=args.seed,
            use_amp=not args.no_amp,
            class_weight=args.sentiment_class_weight,
            selection_metric=args.sentiment_selection_metric,
        )
        sentiment_model, sentiment_summary = train_transformer_aspect_sentiment_model(
            train_df,
            validation_df,
            output_dir / "sentiment_model",
            sentiment_config,
            device,
        )
    else:
        sentiment_model = train_candidate_sentiment_model(train_df, args.sentiment_mode)
    validation_sentiment_features = build_sentiment_feature_lookup(
        sentiment_model,
        validation_df["text"].tolist(),
        heldout_aspects,
        args.sentiment_mode,
    )
    validation_sentiment_lookup = sentiment_lookup_from_features(validation_sentiment_features)
    test_sentiment_features = build_sentiment_feature_lookup(
        sentiment_model,
        test_df["text"].tolist(),
        heldout_aspects,
        args.sentiment_mode,
    )
    test_sentiment_lookup = sentiment_lookup_from_features(test_sentiment_features)
    if args.sentiment_mode == "transformer_aspect_conditioned":
        del sentiment_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
    train_loader = make_aspect_train_loader(train_examples, tokenizer, config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_epoch = None
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config)
        validation_scores = score_aspect_grid(model, tokenizer, validation_df, heldout_aspects, config, device)
        validation_metrics, _ = score_thresholds(
            validation_df,
            heldout_aspects,
            eval_candidate_pairs,
            validation_scores,
            validation_sentiment_lookup,
            args.max_predictions_per_row,
            ensure_one,
            args.selection_metric,
        )
        validation_metrics["epoch"] = epoch
        validation_metrics["train_loss"] = float(train_loss)
        history.append(validation_metrics)
        print(json.dumps({"strategy": strategy, **validation_metrics}, indent=2), flush=True)

        selected_keys = selection_keys(args.selection_metric)
        if best_epoch is None or tuple(validation_metrics[key] for key in selected_keys) > tuple(
            best_epoch[key] for key in selected_keys
        ):
            best_epoch = validation_metrics
            best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    validation_scores = score_aspect_grid(model, tokenizer, validation_df, heldout_aspects, config, device)
    validation_aspects = aspect_predictions_from_scores(
        validation_scores,
        heldout_aspects,
        best_epoch["threshold"],
        ensure_one=ensure_one,
        max_predictions_per_row=args.max_predictions_per_row,
    )
    validation_predictions = pair_predictions_from_aspects(validation_aspects, validation_sentiment_lookup)
    write_prediction_rows(
        validation_df,
        validation_predictions,
        output_dir / "best_validation_predictions.jsonl",
        aspect_scores=validation_scores,
        candidate_aspects=heldout_aspects,
        selected_aspects=validation_aspects,
        threshold=float(best_epoch["threshold"]),
        sentiment_features=validation_sentiment_features,
    )

    test_scores = score_aspect_grid(model, tokenizer, test_df, heldout_aspects, config, device)
    test_aspects = aspect_predictions_from_scores(
        test_scores,
        heldout_aspects,
        best_epoch["threshold"],
        ensure_one=ensure_one,
        max_predictions_per_row=args.max_predictions_per_row,
    )
    test_predictions = pair_predictions_from_aspects(test_aspects, test_sentiment_lookup)
    write_prediction_rows(
        test_df,
        test_predictions,
        output_dir / "best_test_predictions.jsonl",
        aspect_scores=test_scores,
        candidate_aspects=heldout_aspects,
        selected_aspects=test_aspects,
        threshold=float(best_epoch["threshold"]),
        sentiment_features=test_sentiment_features,
    )
    test_metrics = evaluate_pair_and_aspect(test_df["supervision_pair_labels"].tolist(), test_predictions, eval_candidate_pairs)

    _, y_true = binarize_labels(test_df["supervision_pair_labels"].tolist(), eval_candidate_pairs)
    _, y_pred = binarize_labels(test_predictions, eval_candidate_pairs)
    per_label_report(y_true, y_pred, eval_candidate_pairs, output_path=output_dir / "best_test_per_label.csv")

    best_test = {
        **test_metrics,
        "threshold": float(best_epoch["threshold"]),
        "best_epoch": int(best_epoch["epoch"]),
        "train_examples": int(len(train_df)),
        "training_pairs": int(len(train_examples)),
        "validation_examples": int(len(validation_df)),
        "test_examples": int(len(test_df)),
        "train_candidate_aspects": int(len(train_aspects)),
        "eval_candidate_aspects": int(len(heldout_aspects)),
        "eval_candidate_pair_labels": int(len(eval_candidate_pairs)),
    }
    summary = {
        "model": f"candidate_aspect_cross_encoder_with_{args.sentiment_mode}_sentiment",
        "sentiment_mode": args.sentiment_mode,
        "strategy": strategy,
        "eval_label_scope": "heldout",
        "eval_row_scope": eval_row_scope,
        "heldout_aspects": heldout_aspects,
        "selection_metric": f"validation {args.selection_metric}, with remaining pair F1 metrics as tie-breakers",
        "config": config.__dict__,
        "negatives_per_positive": int(args.negatives_per_positive),
        "max_predictions_per_row": args.max_predictions_per_row,
        "ensure_one_prediction_per_row": bool(ensure_one),
        "sentiment_model": sentiment_summary,
        "best_validation": best_epoch,
        "best_test": best_test,
        "history": history,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aspect-label-aware held-out-aspect FABSA baselines.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines" / "aspect_label_aware")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered", "both"], default="both")
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--max-predictions-per-row", type=int, default=None)
    parser.add_argument("--allow-empty-predictions", action="store_true")
    parser.add_argument("--selection-metric", choices=SELECTION_KEYS, default="pair_samples_f1")
    parser.add_argument("--sentiment-mode", choices=SENTIMENT_MODES, default="aspect_conditioned")
    parser.add_argument("--sentiment-model-name", default="distilbert-base-uncased")
    parser.add_argument("--sentiment-epochs", type=int, default=3)
    parser.add_argument("--sentiment-batch-size", type=int, default=16)
    parser.add_argument("--sentiment-eval-batch-size", type=int, default=64)
    parser.add_argument("--sentiment-learning-rate", type=float, default=2e-5)
    parser.add_argument("--sentiment-weight-decay", type=float, default=0.01)
    parser.add_argument("--sentiment-max-length", type=int, default=256)
    parser.add_argument("--sentiment-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--sentiment-class-weight", choices=["none", "balanced", "sqrt"], default="balanced")
    parser.add_argument("--sentiment-selection-metric", choices=["accuracy", "macro_f1", "micro_f1"], default="macro_f1")
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
    print(f"Saved aspect-label-aware baseline outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
