from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.tfidf_logreg import run_tfidf_logreg
from msc_project.data.fabsa import default_data_dir, load_split, unique_labels
from msc_project.evaluation.metrics import evaluate_pair_and_aspect, per_label_report


def evaluate_split(data_dir: Path, split: str, max_features: int, output_dir: Path) -> dict[str, float]:
    train_df = load_split(data_dir, "train")
    eval_df = load_split(data_dir, split)
    labels = unique_labels([train_df], "pair_labels")

    result = run_tfidf_logreg(
        train_df=train_df,
        eval_df=eval_df,
        labels=labels,
        max_features=max_features,
    )

    metrics = evaluate_pair_and_aspect(
        true_pairs=eval_df["pair_labels"].tolist(),
        pred_pairs=result.pred_labels,
        pair_classes=labels,
    )
    metrics.update(
        {
            "train_examples": int(len(train_df)),
            "eval_examples": int(len(eval_df)),
            "pair_labels": int(len(labels)),
            "max_features": int(max_features),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_label_report(
        result.y_true,
        result.y_pred,
        result.labels,
        output_path=output_dir / f"per_label_{split}.csv",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TF-IDF + Logistic Regression FABSA baseline.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--eval-split", choices=["validation", "test", "all"], default="validation")
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines" / "tfidf_logreg")
    args = parser.parse_args()

    splits = ["validation", "test"] if args.eval_split == "all" else [args.eval_split]
    all_metrics = {}
    for split in splits:
        metrics = evaluate_split(args.data_dir, split, args.max_features, args.output_dir)
        all_metrics[split] = metrics
        print(split)
        print(json.dumps(metrics, indent=2))

    if len(all_metrics) > 1:
        (args.output_dir / "metrics_all.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

