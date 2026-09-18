from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.classical import predicted_label_lists, threshold_predictions
from msc_project.baselines.transformer import (
    TransformerConfig,
    build_optimizer_and_scheduler,
    make_dataloaders,
    make_tokenizer_and_model,
    predict_scores,
    set_seed,
    train_one_epoch,
)
from msc_project.data.fabsa import default_data_dir, load_split, unique_labels
from msc_project.data.splits import build_heldout_org_split, choose_org_split_candidates, load_all_fabsa
from msc_project.evaluation.metrics import evaluate_pair_and_aspect, per_label_report


THRESHOLDS = [round(value / 100, 2) for value in range(10, 76, 1)]
SELECTION_KEYS = ("pair_samples_f1", "pair_micro_f1", "pair_macro_f1")


def limit_frame(frame, limit: int | None):
    if limit is None:
        return frame
    return frame.head(limit).copy()


def score_thresholds(eval_df, labels, scores, binarizer, label_column: str) -> tuple[dict[str, float], np.ndarray]:
    rows = []
    best_pred = None
    for threshold in THRESHOLDS:
        y_pred = threshold_predictions(scores, threshold)
        pred_labels = predicted_label_lists(y_pred, binarizer)
        metrics = evaluate_pair_and_aspect(eval_df[label_column].tolist(), pred_labels, labels)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)

    rows.sort(key=lambda row: tuple(row[key] for key in SELECTION_KEYS), reverse=True)
    best = rows[0]
    best_pred = threshold_predictions(scores, best["threshold"])
    return best, best_pred


def make_pos_weight(
    train_df,
    labels: list[str],
    mode: str,
    max_weight: float,
    device: torch.device,
    label_column: str,
):
    if mode == "none":
        return None

    binarizer = MultiLabelBinarizer(classes=labels)
    y_train = binarizer.fit_transform(train_df[label_column])
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    weights = negatives / np.maximum(positives, 1)

    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode != "balanced":
        raise ValueError(f"Unknown pos-weight mode: {mode}")

    weights = np.clip(weights, 1.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_protocol_frames(data_dir: Path, protocol: str):
    if protocol == "closed-topic":
        train_df = load_split(data_dir, "train")
        validation_df = load_split(data_dir, "validation")
        test_df = load_split(data_dir, "test")
        labels = unique_labels([train_df], "pair_labels")
        return train_df, validation_df, test_df, labels, "pair_labels", {
            "protocol": "closed_topic",
            "split": "provided_fabsa",
        }

    if protocol == "heldout-org":
        frame = load_all_fabsa(data_dir)
        candidate = choose_org_split_candidates(frame, top_k=1)[0]
        splits = build_heldout_org_split(frame, candidate.validation_orgs, candidate.test_orgs)
        labels = unique_labels([splits["train"]], "supervision_pair_labels")
        return splits["train"], splits["validation"], splits["test"], labels, "supervision_pair_labels", {
            "protocol": "heldout_organisation",
            "validation_orgs": list(candidate.validation_orgs),
            "test_orgs": list(candidate.test_orgs),
        }

    raise ValueError(f"Unknown protocol: {protocol}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BERT-style closed-topic FABSA baseline.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--protocol", choices=["closed-topic", "heldout-org"], default="closed-topic")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines" / "transformer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--pos-weight", choices=["none", "sqrt", "balanced"], default="none")
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, validation_df, test_df, labels, label_column, split_info = load_protocol_frames(args.data_dir, args.protocol)
    train_df = limit_frame(train_df, args.train_limit)
    validation_df = limit_frame(validation_df, args.eval_limit)
    test_df = limit_frame(test_df, args.eval_limit)

    config = TransformerConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        grad_accumulation_steps=args.grad_accumulation_steps,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    tokenizer, model = make_tokenizer_and_model(config, labels)
    model.to(device)
    train_loader, validation_loader, binarizer = make_dataloaders(
        train_df,
        validation_df,
        tokenizer,
        labels,
        config,
        label_column=label_column,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, config)
    pos_weight = make_pos_weight(train_df, labels, args.pos_weight, args.max_pos_weight, device, label_column)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_epoch = None
    best_state = None
    best_validation_pred = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config, pos_weight)
        validation_scores, validation_true = predict_scores(model, validation_loader, device)
        validation_metrics, validation_pred = score_thresholds(
            validation_df,
            labels,
            validation_scores,
            binarizer,
            label_column,
        )
        validation_metrics["epoch"] = epoch
        validation_metrics["train_loss"] = float(train_loss)
        history.append(validation_metrics)
        print(json.dumps(validation_metrics, indent=2))

        if best_epoch is None or (
            tuple(validation_metrics[key] for key in SELECTION_KEYS)
        ) > (
            tuple(best_epoch[key] for key in SELECTION_KEYS)
        ):
            best_epoch = validation_metrics
            best_validation_pred = validation_pred
            best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_loader = make_dataloaders(train_df, test_df, tokenizer, labels, config, label_column=label_column)[1]
    test_scores, test_true = predict_scores(model, test_loader, device)
    y_test_pred = threshold_predictions(test_scores, best_epoch["threshold"])
    test_pred_labels = predicted_label_lists(y_test_pred, binarizer)
    test_metrics = evaluate_pair_and_aspect(test_df[label_column].tolist(), test_pred_labels, labels)
    test_metrics.update(
        {
            "threshold": float(best_epoch["threshold"]),
            "best_epoch": int(best_epoch["epoch"]),
            "train_examples": int(len(train_df)),
            "validation_examples": int(len(validation_df)),
            "test_examples": int(len(test_df)),
            "pair_labels": int(len(labels)),
            "label_column": label_column,
            "model_name": config.model_name,
            "learning_rate": float(config.learning_rate),
            "batch_size": int(config.batch_size),
            "max_length": int(config.max_length),
            "pos_weight": args.pos_weight,
            "max_pos_weight": float(args.max_pos_weight),
        }
    )

    per_label_report(
        test_true,
        y_test_pred,
        labels,
        output_path=args.output_dir / "best_test_per_label.csv",
    )

    summary = {
        **split_info,
        "config": config.__dict__,
        "pos_weight": args.pos_weight,
        "max_pos_weight": float(args.max_pos_weight),
        "selection_metric": "validation pair_samples_f1, with pair_micro_f1 and pair_macro_f1 tie-breakers",
        "history": history,
        "best_validation": best_epoch,
        "best_test": test_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Best validation")
    print(json.dumps(best_epoch, indent=2))
    print("Best model on test")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved transformer baseline outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
