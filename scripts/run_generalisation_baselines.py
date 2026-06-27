from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import (
    SENTIMENT_MODES,
    CandidateLexicalBaseline,
    candidate_pair_labels,
)
from msc_project.baselines.transformer_sentiment import (
    TransformerAspectSentimentConfig,
    train_transformer_aspect_sentiment_model,
)
from msc_project.baselines.classical import (
    ClassicalConfig,
    default_configs,
    predicted_label_lists,
    threshold_predictions,
    train_classical_model,
)
from msc_project.data.fabsa import default_data_dir, unique_labels
from msc_project.data.splits import (
    DEFAULT_HELDOUT_ASPECTS,
    build_heldout_aspect_split,
    build_heldout_org_split,
    choose_org_split_candidates,
    load_all_fabsa,
)
from msc_project.evaluation.metrics import binarize_labels, evaluate_pair_and_aspect, per_label_report


PROBABILITY_THRESHOLDS = [round(value / 100, 2) for value in range(15, 76, 2)]
DECISION_THRESHOLDS = [round(value / 100, 2) for value in range(-100, 101, 2)]
LEXICAL_THRESHOLDS = [round(value / 100, 2) for value in range(0, 51)]
SELECTION_COLUMNS = ["pair_samples_f1", "pair_micro_f1", "pair_macro_f1"]


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def selection_columns(primary_metric: str) -> list[str]:
    if primary_metric not in SELECTION_COLUMNS:
        raise ValueError(f"Unknown selection metric: {primary_metric}")
    return [primary_metric] + [column for column in SELECTION_COLUMNS if column != primary_metric]


def score_classical_predictions(eval_df, labels, result, threshold: float) -> dict[str, float]:
    y_pred = threshold_predictions(result.y_score, threshold)
    pred_labels = predicted_label_lists(y_pred, result.binarizer)
    scores = evaluate_pair_and_aspect(eval_df["supervision_pair_labels"].tolist(), pred_labels, labels)
    scores["threshold"] = float(threshold)
    return scores


def run_classical_validation_sweep(
    train_df,
    validation_df,
    labels: list[str],
    configs: list[ClassicalConfig],
) -> pd.DataFrame:
    rows = []
    for index, config in enumerate(configs, start=1):
        print(f"[cross-org {index}/{len(configs)}] {config.name}")
        result = train_classical_model(
            train_df,
            validation_df,
            labels,
            config,
            label_column="supervision_pair_labels",
        )
        thresholds = DECISION_THRESHOLDS if result.score_type == "decision" else PROBABILITY_THRESHOLDS
        for threshold in thresholds:
            rows.append(
                {
                    "config": config.name,
                    "model_type": config.model_type,
                    "score_type": result.score_type,
                    "feature_type": config.feature_type,
                    "c": config.c,
                    "class_weight": config.class_weight or "none",
                    "max_features": config.max_features,
                    "ngram_max": config.ngram_max,
                    **score_classical_predictions(validation_df, labels, result, threshold),
                }
            )

    return pd.DataFrame(rows).sort_values(SELECTION_COLUMNS, ascending=False)


def refined_cross_org_configs() -> list[ClassicalConfig]:
    configs = []
    for c in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for max_features in [60000, 100000]:
            for ngram_max in [2, 3]:
                configs.append(
                    ClassicalConfig(
                        name=f"refined_svm_word_char_tfidf_{max_features // 1000}k_1_{ngram_max}_balanced_c{c:g}",
                        feature_type="word_char_tfidf",
                        max_features=max_features,
                        ngram_max=ngram_max,
                        c=c,
                        class_weight="balanced",
                        model_type="linear_svm",
                    )
                )
    return configs


def evaluate_classical_test(
    train_df,
    test_df,
    labels: list[str],
    best_row,
    configs: list[ClassicalConfig],
    output_dir: Path,
) -> dict[str, object]:
    config = next(config for config in configs if config.name == best_row["config"])
    result = train_classical_model(
        train_df,
        test_df,
        labels,
        config,
        label_column="supervision_pair_labels",
    )
    y_pred = threshold_predictions(result.y_score, float(best_row["threshold"]))
    pred_labels = predicted_label_lists(y_pred, result.binarizer)
    scores = evaluate_pair_and_aspect(test_df["supervision_pair_labels"].tolist(), pred_labels, labels)
    scores.update(
        {
            "config": config.name,
            "threshold": float(best_row["threshold"]),
            "train_examples": int(len(train_df)),
            "eval_examples": int(len(test_df)),
            "pair_labels": int(len(labels)),
        }
    )
    per_label_report(result.y_true, y_pred, result.labels, output_path=output_dir / "best_test_per_label.csv")
    return scores


def run_heldout_org(frame, output_dir: Path, quick: bool, refined: bool) -> dict[str, object]:
    candidate = choose_org_split_candidates(frame, top_k=1)[0]
    splits = build_heldout_org_split(frame, candidate.validation_orgs, candidate.test_orgs)
    labels = unique_labels([splits["train"]], "supervision_pair_labels")
    if quick:
        configs = default_configs()[:3]
        grid = "quick"
    elif refined:
        configs = refined_cross_org_configs()
        grid = "refined_word_char_svm"
    else:
        configs = default_configs()
        grid = "default"

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_results = run_classical_validation_sweep(splits["train"], splits["validation"], labels, configs)
    validation_results.to_csv(output_dir / "validation_sweep.csv", index=False)

    best_validation = validation_results.iloc[0]
    best_test = evaluate_classical_test(splits["train"], splits["test"], labels, best_validation, configs, output_dir)
    summary = {
        "protocol": "heldout_organisation",
        "validation_orgs": list(candidate.validation_orgs),
        "test_orgs": list(candidate.test_orgs),
        "grid": grid,
        "selection_metric": "validation pair_samples_f1, with pair_micro_f1 and pair_macro_f1 tie-breakers",
        "best_validation": best_validation.to_dict(),
        "best_test": best_test,
        "top_validation": validation_results.head(10).to_dict(orient="records"),
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def score_candidate_predictions(eval_df, pair_classes: list[str], pred_labels: list[list[str]]) -> dict[str, float]:
    return evaluate_pair_and_aspect(eval_df["supervision_pair_labels"].tolist(), pred_labels, pair_classes)


def run_candidate_label_baseline(
    train_df,
    validation_df,
    test_df,
    heldout_aspects: list[str],
    output_dir: Path,
    ensure_one: bool = True,
    selection_metric: str = "pair_samples_f1",
    sentiment_mode: str = "aspect_conditioned",
    sentiment_model: object | None = None,
) -> dict[str, object]:
    pair_classes = candidate_pair_labels(heldout_aspects)
    baseline = CandidateLexicalBaseline(
        heldout_aspects,
        sentiment_mode=sentiment_mode,
        sentiment_model=sentiment_model,
    ).fit(train_df)

    rows = []
    for threshold in LEXICAL_THRESHOLDS:
        pred_labels = baseline.predict(validation_df, threshold=threshold, ensure_one=ensure_one)
        rows.append(
            {
                "threshold": threshold,
                **score_candidate_predictions(validation_df, pair_classes, pred_labels),
            }
        )

    validation_results = pd.DataFrame(rows).sort_values(selection_columns(selection_metric), ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(output_dir / "validation_sweep.csv", index=False)

    best_validation = validation_results.iloc[0]
    test_pred = baseline.predict(test_df, threshold=float(best_validation["threshold"]), ensure_one=ensure_one)
    test_scores = score_candidate_predictions(test_df, pair_classes, test_pred)
    _, y_true = binarize_labels(test_df["supervision_pair_labels"].tolist(), pair_classes)
    _, y_pred = binarize_labels(test_pred, pair_classes)
    per_label_report(y_true, y_pred, pair_classes, output_path=output_dir / "best_test_per_label.csv")

    best_test = {
        **test_scores,
        "threshold": float(best_validation["threshold"]),
        "train_examples": int(len(train_df)),
        "eval_examples": int(len(test_df)),
        "pair_labels": int(len(pair_classes)),
    }
    summary = {
        "model": f"candidate_label_lexical_tfidf_with_{sentiment_mode}_sentiment",
        "sentiment_mode": sentiment_mode,
        "selection_metric": f"validation {selection_metric}, with remaining pair F1 metrics as tie-breakers",
        "heldout_aspects": heldout_aspects,
        "ensure_one_prediction_per_row": bool(ensure_one),
        "best_validation": best_validation.to_dict(),
        "best_test": best_test,
        "top_validation": validation_results.head(10).to_dict(orient="records"),
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def run_heldout_aspect(
    frame,
    output_dir: Path,
    strategies: list[str],
    heldout_aspects: list[str],
    selection_metric: str,
    sentiment_mode: str,
    sentiment_config: TransformerAspectSentimentConfig,
    device: torch.device,
) -> dict[str, object]:
    summaries = {}
    for strategy in strategies:
        print(f"[heldout-aspect] {strategy}")
        splits = build_heldout_aspect_split(
            frame,
            heldout_aspects,
            strategy=strategy,
            eval_label_scope="heldout",
        )
        sentiment_model = None
        sentiment_summary = None
        if sentiment_mode == "transformer_aspect_conditioned":
            sentiment_model, sentiment_summary = train_transformer_aspect_sentiment_model(
                splits["train"],
                splits["validation"],
                output_dir / strategy / "sentiment_model",
                sentiment_config,
                device,
            )
        summaries[strategy] = run_candidate_label_baseline(
            splits["train"],
            splits["validation"],
            splits["test"],
            heldout_aspects,
            output_dir / strategy,
            selection_metric=selection_metric,
            sentiment_mode=sentiment_mode,
            sentiment_model=sentiment_model,
        )
        if sentiment_summary is not None:
            summaries[strategy]["sentiment_model"] = sentiment_summary
        if sentiment_mode == "transformer_aspect_conditioned":
            del sentiment_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first FABSA generalisation baselines.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines" / "generalisation")
    parser.add_argument("--protocol", choices=["heldout-org", "heldout-aspect", "all"], default="all")
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered", "both"], default="both")
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--selection-metric", choices=SELECTION_COLUMNS, default="pair_samples_f1")
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
    parser.add_argument("--quick", action="store_true", help="Run a small cross-org smoke-test grid.")
    parser.add_argument("--refined-cross-org", action="store_true", help="Run the narrower cross-org SVM refinement grid.")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    frame = load_all_fabsa(args.data_dir)
    summaries: dict[str, object] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_config = TransformerAspectSentimentConfig(
        model_name=args.sentiment_model_name,
        max_length=args.sentiment_max_length,
        batch_size=args.sentiment_batch_size,
        eval_batch_size=args.sentiment_eval_batch_size,
        learning_rate=args.sentiment_learning_rate,
        weight_decay=args.sentiment_weight_decay,
        epochs=args.sentiment_epochs,
        warmup_ratio=args.sentiment_warmup_ratio,
        seed=13,
        use_amp=not args.no_amp,
        class_weight=args.sentiment_class_weight,
        selection_metric=args.sentiment_selection_metric,
    )

    if args.protocol in {"heldout-org", "all"}:
        summaries["heldout_organisation"] = run_heldout_org(
            frame,
            args.output_dir / "heldout_organisation",
            quick=args.quick,
            refined=args.refined_cross_org,
        )

    if args.protocol in {"heldout-aspect", "all"}:
        strategies = ["label_masked", "example_filtered"] if args.strategy == "both" else [args.strategy]
        heldout_aspects = args.heldout_aspect or DEFAULT_HELDOUT_ASPECTS
        summaries["heldout_aspect"] = run_heldout_aspect(
            frame,
            args.output_dir / "heldout_aspect",
            strategies,
            heldout_aspects,
            args.selection_metric,
            args.sentiment_mode,
            sentiment_config,
            device,
        )

    write_json(summaries, args.output_dir / "summary.json")
    print(json.dumps(summaries, indent=2))
    print(f"Saved generalisation baseline results to {args.output_dir}")


if __name__ == "__main__":
    main()
