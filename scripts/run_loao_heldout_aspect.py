from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import all_aspects, build_heldout_aspect_split, load_all_fabsa
from msc_project.baselines.candidate_label import SENTIMENT_MODES

from run_generalisation_baselines import run_candidate_label_baseline, write_json


METRIC_COLUMNS = [
    "pair_samples_f1",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
    "pair_false_positive_rows_per_100",
    "pair_false_positive_labels_per_100",
    "pair_false_negative_rows_per_100",
    "pair_exact_match_rate",
    "aspect_samples_f1",
    "aspect_micro_f1",
    "aspect_micro_precision",
    "aspect_micro_recall",
    "aspect_macro_f1",
    "aspect_false_positive_rows_per_100",
    "aspect_false_positive_labels_per_100",
    "aspect_false_negative_rows_per_100",
    "aspect_exact_match_rate",
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "presence_prevalence",
    "presence_false_positive_rows_per_100",
    "presence_false_negative_rows_per_100",
    "sentiment_accuracy_when_gold_aspect_predicted",
]


def aspect_slug(aspect: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", aspect).strip("_").lower()
    return slug or "aspect"


def selected_aspects(frame, requested_aspects: list[str]) -> list[str]:
    aspects = all_aspects(frame)
    if not requested_aspects:
        return aspects

    unknown = sorted(set(requested_aspects) - set(aspects))
    if unknown:
        raise ValueError(f"Unknown held-out aspects: {unknown}")
    return requested_aspects


def row_counts(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "validation_positive_rows": int(splits["validation"]["supervision_pair_labels"].apply(len).gt(0).sum()),
        "test_rows": int(len(splits["test"])),
        "test_positive_rows": int(splits["test"]["supervision_pair_labels"].apply(len).gt(0).sum()),
    }


def flatten_result(
    baseline: str,
    sentiment_mode: str,
    strategy: str,
    aspect: str,
    eval_row_scope: str,
    result: dict[str, object],
    counts: dict[str, int],
) -> list[dict[str, object]]:
    rows = []
    for split_key, split_name in [("best_validation", "validation"), ("best_test", "test")]:
        metrics = result[split_key]
        row = {
            "baseline": baseline,
            "sentiment_mode": sentiment_mode,
            "strategy": strategy,
            "heldout_aspect": aspect,
            "split": split_name,
            "eval_row_scope": eval_row_scope,
            **counts,
        }
        for column in METRIC_COLUMNS:
            row[column] = float(metrics.get(column, 0.0))
        row["threshold"] = float(metrics["threshold"])
        if "best_epoch" in metrics:
            row["best_epoch"] = int(metrics["best_epoch"])
        rows.append(row)
    return rows


def aggregate_spread(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (baseline, sentiment_mode, strategy, split_name), group in results.groupby(
        ["baseline", "sentiment_mode", "strategy", "split"]
    ):
        row: dict[str, object] = {
            "baseline": baseline,
            "sentiment_mode": sentiment_mode,
            "strategy": strategy,
            "split": split_name,
            "aspects": int(group["heldout_aspect"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["baseline", "strategy", "split"]).reset_index(drop=True)


def run_lexical_fold(
    frame,
    aspect: str,
    strategy: str,
    output_dir: Path,
    eval_row_scope: str,
    ensure_one: bool,
    selection_metric: str,
    sentiment_mode: str,
) -> tuple[dict[str, object], dict[str, int]]:
    splits = build_heldout_aspect_split(
        frame,
        [aspect],
        strategy=strategy,
        eval_label_scope="heldout",
        eval_row_scope=eval_row_scope,
    )
    counts = row_counts(splits)
    result = run_candidate_label_baseline(
        splits["train"],
        splits["validation"],
        splits["test"],
        [aspect],
        output_dir,
        ensure_one=ensure_one,
        selection_metric=selection_metric,
        sentiment_mode=sentiment_mode,
    )
    return result, counts


def cross_encoder_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        negatives_per_positive=args.negatives_per_positive,
        max_predictions_per_row=args.max_predictions_per_row,
        seed=args.seed,
        no_amp=args.no_amp,
        allow_empty_predictions=not args.ensure_one,
        selection_metric=args.selection_metric,
        sentiment_mode=args.sentiment_mode,
        sentiment_model_name=args.sentiment_model_name,
        sentiment_max_length=args.sentiment_max_length,
        sentiment_batch_size=args.sentiment_batch_size,
        sentiment_eval_batch_size=args.sentiment_eval_batch_size,
        sentiment_learning_rate=args.sentiment_learning_rate,
        sentiment_weight_decay=args.sentiment_weight_decay,
        sentiment_epochs=args.sentiment_epochs,
        sentiment_warmup_ratio=args.sentiment_warmup_ratio,
        sentiment_class_weight=args.sentiment_class_weight,
        sentiment_selection_metric=args.sentiment_selection_metric,
    )


def run_cross_encoder_fold(
    frame,
    aspect: str,
    strategy: str,
    output_dir: Path,
    eval_row_scope: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, int]]:
    import torch

    from run_aspect_label_aware_baseline import run_strategy as run_cross_encoder_strategy

    splits = build_heldout_aspect_split(
        frame,
        [aspect],
        strategy=strategy,
        eval_label_scope="heldout",
        eval_row_scope=eval_row_scope,
    )
    counts = row_counts(splits)

    fold_args = cross_encoder_args(args)
    result = run_cross_encoder_strategy(
        frame,
        [aspect],
        strategy,
        output_dir,
        fold_args,
        device,
        eval_row_scope=eval_row_scope,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, counts


def write_tables(rows: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
    results = pd.DataFrame(rows)
    spread = aggregate_spread(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "all_results.csv", index=False)
    spread.to_csv(output_dir / "spread.csv", index=False)
    test_results = results[results["split"] == "test"].copy()
    test_results.to_csv(output_dir / "test_results.csv", index=False)
    summary = {
        "results": results.to_dict(orient="records"),
        "spread": spread.to_dict(orient="records"),
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leave-one-aspect-out held-out-aspect FABSA evaluation.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines" / "loao_heldout_aspect")
    parser.add_argument("--baseline", choices=["lexical", "cross_encoder", "both"], default="lexical")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered", "both"], default="both")
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--eval-row-scope", choices=["containing_heldout", "all"], default="all")
    parser.add_argument("--ensure-one", action="store_true", help="Force at least one candidate aspect prediction per row.")
    parser.add_argument(
        "--selection-metric",
        choices=["pair_samples_f1", "pair_micro_f1", "pair_macro_f1"],
        default="pair_samples_f1",
        help="Primary validation metric for threshold/model selection.",
    )
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
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    baselines = ["lexical", "cross_encoder"] if args.baseline == "both" else [args.baseline]
    if args.sentiment_mode == "transformer_aspect_conditioned" and "lexical" in baselines:
        parser.error(
            "--sentiment-mode transformer_aspect_conditioned is only supported with "
            "--baseline cross_encoder in this LOAO runner."
        )

    frame = load_all_fabsa(args.data_dir)
    aspects = selected_aspects(frame, args.heldout_aspect)
    strategies = ["label_masked", "example_filtered"] if args.strategy == "both" else [args.strategy]

    device = None
    if "cross_encoder" in baselines:
        import torch

        from msc_project.baselines.transformer import set_seed

        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}", flush=True)
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    all_rows: list[dict[str, object]] = []
    nested: dict[str, object] = {
        "eval_row_scope": args.eval_row_scope,
        "ensure_one_prediction_per_row": bool(args.ensure_one),
        "selection_metric": args.selection_metric,
        "sentiment_mode": args.sentiment_mode,
        "heldout_aspects": aspects,
        "baselines": {},
    }

    total = len(baselines) * len(strategies) * len(aspects)
    step = 0
    for baseline in baselines:
        nested["baselines"][baseline] = {}
        for strategy in strategies:
            nested["baselines"][baseline][strategy] = {}
            for aspect in aspects:
                step += 1
                fold_dir = args.output_dir / baseline / strategy / aspect_slug(aspect)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "total": total,
                            "baseline": baseline,
                            "strategy": strategy,
                            "heldout_aspect": aspect,
                        }
                    ),
                    flush=True,
                )

                if baseline == "lexical":
                    result, counts = run_lexical_fold(
                        frame,
                        aspect,
                        strategy,
                        fold_dir,
                        args.eval_row_scope,
                        ensure_one=args.ensure_one,
                        selection_metric=args.selection_metric,
                        sentiment_mode=args.sentiment_mode,
                    )
                else:
                    if device is None:
                        raise RuntimeError("Cross-encoder execution requires a configured Torch device.")
                    result, counts = run_cross_encoder_fold(
                        frame,
                        aspect,
                        strategy,
                        fold_dir,
                        args.eval_row_scope,
                        args,
                        device,
                    )

                nested["baselines"][baseline][strategy][aspect] = {
                    "row_counts": counts,
                    "summary": result,
                }
                all_rows.extend(
                    flatten_result(
                        baseline,
                        args.sentiment_mode,
                        strategy,
                        aspect,
                        args.eval_row_scope,
                        result,
                        counts,
                    )
                )
                write_tables(all_rows, args.output_dir)

    summary = write_tables(all_rows, args.output_dir)
    nested["aggregate"] = summary
    write_json(nested, args.output_dir / "full_summary.json")
    print(json.dumps(summary["spread"], indent=2), flush=True)
    print(f"Saved LOAO outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
