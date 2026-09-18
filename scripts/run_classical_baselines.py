from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.classical import (
    ClassicalConfig,
    default_configs,
    predicted_label_lists,
    threshold_predictions,
    train_classical_model,
)
from msc_project.data.fabsa import default_data_dir, load_split, unique_labels
from msc_project.evaluation.metrics import evaluate_pair_and_aspect, per_label_report


PROBABILITY_THRESHOLDS = [round(value / 100, 2) for value in range(15, 76, 2)]
DECISION_THRESHOLDS = [round(value / 100, 2) for value in range(-100, 101, 2)]


def score_predictions(eval_df, labels, result, threshold: float) -> dict[str, float]:
    y_pred = threshold_predictions(result.y_score, threshold)
    pred_labels = predicted_label_lists(y_pred, result.binarizer)
    scores = evaluate_pair_and_aspect(eval_df["pair_labels"].tolist(), pred_labels, labels)
    scores["threshold"] = float(threshold)
    return scores


def run_validation_sweep(
    train_df,
    validation_df,
    labels: list[str],
    configs: list[ClassicalConfig],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    trained_results = {}

    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config.name}")
        result = train_classical_model(train_df, validation_df, labels, config)
        trained_results[config.name] = result

        thresholds = DECISION_THRESHOLDS if result.score_type == "decision" else PROBABILITY_THRESHOLDS
        for threshold in thresholds:
            scores = score_predictions(validation_df, labels, result, threshold)
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
                    "svd_components": config.svd_components if config.feature_type == "lsa" else "",
                    **scores,
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["pair_micro_f1", "pair_macro_f1", "aspect_micro_f1"],
        ascending=False,
    )
    return results, trained_results


def evaluate_best_on_test(
    train_df,
    test_df,
    labels: list[str],
    best_row,
    configs: list[ClassicalConfig],
    output_dir: Path,
) -> dict[str, float]:
    config = next(config for config in configs if config.name == best_row["config"])
    result = train_classical_model(train_df, test_df, labels, config)
    y_pred = threshold_predictions(result.y_score, float(best_row["threshold"]))
    pred_labels = predicted_label_lists(y_pred, result.binarizer)

    scores = evaluate_pair_and_aspect(test_df["pair_labels"].tolist(), pred_labels, labels)
    scores.update(
        {
            "config": config.name,
            "threshold": float(best_row["threshold"]),
            "train_examples": int(len(train_df)),
            "eval_examples": int(len(test_df)),
            "pair_labels": int(len(labels)),
        }
    )
    per_label_report(
        result.y_true,
        y_pred,
        result.labels,
        output_path=output_dir / "best_test_per_label.csv",
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical FABSA baselines with threshold tuning.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines" / "classical")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test grid.")
    args = parser.parse_args()

    train_df = load_split(args.data_dir, "train")
    validation_df = load_split(args.data_dir, "validation")
    test_df = load_split(args.data_dir, "test")
    labels = unique_labels([train_df], "pair_labels")

    configs = default_configs()
    if args.quick:
        configs = configs[:3]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_results, _ = run_validation_sweep(train_df, validation_df, labels, configs)
    validation_results.to_csv(args.output_dir / "validation_sweep.csv", index=False)

    best_micro = validation_results.iloc[0]
    best_test = evaluate_best_on_test(train_df, test_df, labels, best_micro, configs, args.output_dir)

    summary = {
        "selection_metric": "validation pair_micro_f1, with pair_macro_f1 tie-breaker",
        "best_validation": best_micro.to_dict(),
        "best_test": best_test,
        "top_validation": validation_results.head(10).to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Best validation configuration")
    print(json.dumps(summary["best_validation"], indent=2))
    print("Best configuration evaluated on test")
    print(json.dumps(best_test, indent=2))
    print(f"Saved sweep results to {args.output_dir}")


if __name__ == "__main__":
    main()
