from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.unified_pair_scorers import (
    UnifiedPairCrossEncoderConfig,
    UnifiedTfidfPairConfig,
    UnifiedTfidfPairScorer,
    build_unified_pair_optimizer_and_scheduler,
    make_unified_pair_tokenizer_and_model,
    make_unified_pair_train_loader,
    score_unified_pair_manifest,
    set_unified_pair_seed,
    train_unified_pair_epoch,
)
from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import all_aspects, build_heldout_aspect_split, load_all_fabsa
from msc_project.experiments.unified_candidate_pairs import (
    CANDIDATE_SENTIMENTS,
    build_eval_grid,
    build_full_manifest,
    budget_sample,
    load_experiment_config,
    manifest_hash,
)
from msc_project.experiments.unified_pair_evaluation import (
    evaluate_score_matrix,
    paired_aspect_statistics,
    pilot_gate,
    score_matrix_from_grid,
    select_threshold,
)


PILOT_FOLDS = (
    "Company brand: Competitor",
    "Company brand: General satisfaction",
    "Staff support: Email",
)
MODELS = ("tfidf", "distilbert")
VARIANTS = ("control", "enhanced")


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialise {type(value).__name__}.")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def dataframe_jsonl(frame: pd.DataFrame, path: Path) -> None:
    write_jsonl(frame.to_dict(orient="records"), path)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines() if result.returncode == 0 else []


def package_versions() -> dict[str, str | None]:
    names = ("numpy", "pandas", "scikit-learn", "torch", "transformers", "pytest")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_summary() -> dict[str, object]:
    result: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_total_memory_bytes": int(properties.total_memory),
            }
        )
    return result


def ordered_eval_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    ordered = frame.assign(_ordered_uid=frame["row_uid"].astype(str)).sort_values(
        "_ordered_uid", kind="stable"
    )
    if limit is not None:
        ordered = ordered.head(limit)
    return ordered.drop(columns="_ordered_uid").reset_index(drop=True)


def seen_aspects(train_frame: pd.DataFrame) -> list[str]:
    values = {
        str(aspect)
        for labels in train_frame["supervision_labels"]
        for aspect, _ in labels
    }
    return sorted(values)


def make_training_manifest(
    train_frame: pd.DataFrame,
    heldout_aspect: str,
    variant: str,
    args: argparse.Namespace,
    fold_output: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates = seen_aspects(train_frame)
    if heldout_aspect in candidates:
        raise AssertionError("Held-out aspect entered the permitted training candidate set.")
    leaked_rows = train_frame["labels"].apply(
        lambda labels: any(str(aspect) == heldout_aspect for aspect, _ in labels)
    )
    if leaked_rows.any():
        raise AssertionError("example_filtered training still contains the held-out aspect.")

    full_manifest = build_full_manifest(train_frame, candidates, variant, seed=args.seed)
    budget_manifest = budget_sample(
        full_manifest,
        total_budget=args.train_budget,
        positive_budget=args.positive_budget,
        seed=args.seed,
    )
    if heldout_aspect in set(budget_manifest["candidate_aspect"]):
        raise AssertionError("Held-out aspect entered the budgeted manifest.")
    if budget_manifest.duplicated(["row_uid", "candidate_aspect", "candidate_sentiment"]).any():
        raise AssertionError("Duplicate pair survived budget sampling.")

    manifest_dir = fold_output / "manifests"
    dataframe_jsonl(budget_manifest, manifest_dir / f"{variant}_budget.jsonl")
    info = {
        "heldout_aspect": heldout_aspect,
        "variant": variant,
        "candidate_aspects": candidates,
        "candidate_aspect_count": int(len(candidates)),
        "full_pair_count": int(len(full_manifest)),
        "full_positive_count": int(full_manifest["target"].sum()),
        "full_manifest_hash": manifest_hash(full_manifest),
        "budget_pair_count": int(len(budget_manifest)),
        "budget_positive_count": int(budget_manifest["target"].sum()),
        "budget_negative_count": int((budget_manifest["target"] == 0).sum()),
        "budget_manifest_hash": manifest_hash(budget_manifest),
        "negative_type_counts": {
            str(key): int(value)
            for key, value in budget_manifest["negative_type"].value_counts().items()
        },
    }
    write_json(info, manifest_dir / f"{variant}_manifest_summary.json")
    return budget_manifest, info


def eval_inputs(
    eval_frame: pd.DataFrame,
    heldout_aspect: str,
    variant: str,
) -> tuple[pd.DataFrame, list[list[str]]]:
    grid = build_eval_grid(eval_frame, heldout_aspect, variant)
    expected = len(eval_frame) * len(CANDIDATE_SENTIMENTS)
    if len(grid) != expected:
        raise AssertionError(f"Expected {expected} evaluation pairs, found {len(grid)}.")
    true_pairs = [list(labels) for labels in eval_frame["supervision_pair_labels"]]
    return grid, true_pairs


def write_predictions(
    eval_frame: pd.DataFrame,
    scores: np.ndarray,
    predictions: list[list[str]],
    threshold: float,
    path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for row_index, row in eval_frame.iterrows():
        rows.append(
            {
                "row_index": int(row_index),
                "id": str(row["id"]),
                "row_uid": str(row["row_uid"]),
                "original_split": str(row["original_split"]),
                "text": str(row["text"]),
                "gold_pair_labels": list(row["supervision_pair_labels"]),
                "pred_pair_labels": predictions[row_index],
                "candidate_sentiment_scores": {
                    sentiment: float(scores[row_index, sentiment_index])
                    for sentiment_index, sentiment in enumerate(CANDIDATE_SENTIMENTS)
                },
                "threshold": float(threshold),
            }
        )
    write_jsonl(rows, path)


def selection_key(metrics: dict[str, object]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics["pair_micro_f1"]),
        float(metrics["pair_samples_f1"]),
        float(metrics["pair_micro_precision"]),
        -float(metrics["presence_false_positive_rows_per_100"]),
        -float(metrics.get("epoch", 0)),
    )


def run_tfidf_fold(
    train_manifest: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame | None,
    heldout_aspect: str,
    variant: str,
    model_output: Path,
    args: argparse.Namespace,
    manifest_info: dict[str, object],
) -> list[dict[str, object]]:
    started = time.time()
    config = UnifiedTfidfPairConfig(
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        min_df=1,
        max_word_features=60_000,
        max_char_features=60_000,
        max_iter=1_000,
        seed=args.seed,
    )
    scorer = UnifiedTfidfPairScorer(config).fit(train_manifest)
    validation_grid, validation_true = eval_inputs(validation_frame, heldout_aspect, variant)
    validation_flat_scores = scorer.score_manifest(validation_grid)
    validation_scores = score_matrix_from_grid(
        validation_grid,
        validation_flat_scores,
        row_count=len(validation_frame),
    )
    selected = select_threshold(validation_true, validation_scores, heldout_aspect)
    validation_metrics, validation_predictions = evaluate_score_matrix(
        validation_true,
        validation_scores,
        heldout_aspect,
        selected.threshold,
    )
    selected.sweep.to_csv(model_output / "validation_threshold_sweep.csv", index=False)
    score_rows = validation_grid.copy()
    score_rows["score"] = validation_flat_scores
    score_rows.to_csv(model_output / "validation_pair_scores.csv", index=False)
    write_predictions(
        validation_frame,
        validation_scores,
        validation_predictions,
        selected.threshold,
        model_output / "validation_predictions.jsonl",
    )

    common = {
        "model": "tfidf",
        "variant": variant,
        "heldout_aspect": heldout_aspect,
        "selected_threshold": float(selected.threshold),
        "best_epoch": None,
        "train_seconds": float(time.time() - started),
        **manifest_info,
    }
    results = [{**common, "split": "validation", **validation_metrics}]
    if test_frame is not None:
        test_grid, test_true = eval_inputs(test_frame, heldout_aspect, variant)
        test_flat_scores = scorer.score_manifest(test_grid)
        test_scores = score_matrix_from_grid(test_grid, test_flat_scores, row_count=len(test_frame))
        test_metrics, test_predictions = evaluate_score_matrix(
            test_true,
            test_scores,
            heldout_aspect,
            selected.threshold,
        )
        score_rows = test_grid.copy()
        score_rows["score"] = test_flat_scores
        score_rows.to_csv(model_output / "test_pair_scores.csv", index=False)
        write_predictions(
            test_frame,
            test_scores,
            test_predictions,
            selected.threshold,
            model_output / "test_predictions.jsonl",
        )
        results.append({**common, "split": "test", **test_metrics})

    write_json(
        {
            "protocol_id": "loao_unified_candidate_pair_experimental_v1",
            "config": config.__dict__,
            "results": results,
            "wall_seconds": float(time.time() - started),
        },
        model_output / "summary.json",
    )
    return results


def run_distilbert_fold(
    train_manifest: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame | None,
    heldout_aspect: str,
    variant: str,
    model_output: Path,
    args: argparse.Namespace,
    manifest_info: dict[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    started = time.time()
    config = UnifiedPairCrossEncoderConfig(
        model_name=args.distilbert_model_name,
        max_length=256,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=3e-5,
        weight_decay=0.01,
        epochs=args.epochs,
        warmup_ratio=0.1,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
    set_unified_pair_seed(config.seed)
    tokenizer, model = make_unified_pair_tokenizer_and_model(config)
    model.to(device)
    train_loader = make_unified_pair_train_loader(train_manifest, tokenizer, config)
    optimizer, scheduler = build_unified_pair_optimizer_and_scheduler(model, train_loader, config)
    validation_grid, validation_true = eval_inputs(validation_frame, heldout_aspect, variant)

    history: list[dict[str, object]] = []
    best_metrics: dict[str, object] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    best_scores: np.ndarray | None = None
    best_flat_scores: np.ndarray | None = None
    peak_memory = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.time()
        train_loss = train_unified_pair_epoch(model, train_loader, optimizer, scheduler, device, config)
        flat_scores = score_unified_pair_manifest(model, tokenizer, validation_grid, config, device)
        scores = score_matrix_from_grid(validation_grid, flat_scores, row_count=len(validation_frame))
        selected = select_threshold(validation_true, scores, heldout_aspect)
        metrics = {
            **selected.metrics,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "epoch_seconds": float(time.time() - epoch_started),
        }
        history.append(metrics)
        print(
            json.dumps(
                {
                    "event": "distilbert_epoch",
                    "variant": variant,
                    "heldout_aspect": heldout_aspect,
                    **metrics,
                },
                default=json_default,
            ),
            flush=True,
        )
        if best_metrics is None or selection_key(metrics) > selection_key(best_metrics):
            best_metrics = metrics
            best_threshold = float(selected.threshold)
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_scores = scores.copy()
            best_flat_scores = flat_scores.copy()
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))

    if best_metrics is None or best_state is None or best_scores is None or best_flat_scores is None:
        raise AssertionError("No DistilBERT epoch was selected.")
    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics, validation_predictions = evaluate_score_matrix(
        validation_true,
        best_scores,
        heldout_aspect,
        best_threshold,
    )
    validation_score_rows = validation_grid.copy()
    validation_score_rows["score"] = best_flat_scores
    validation_score_rows.to_csv(model_output / "validation_pair_scores.csv", index=False)
    write_predictions(
        validation_frame,
        best_scores,
        validation_predictions,
        best_threshold,
        model_output / "validation_predictions.jsonl",
    )

    common = {
        "model": "distilbert",
        "variant": variant,
        "heldout_aspect": heldout_aspect,
        "selected_threshold": float(best_threshold),
        "best_epoch": int(best_metrics["epoch"]),
        "train_seconds": float(sum(float(row["epoch_seconds"]) for row in history)),
        "peak_cuda_memory_bytes": int(peak_memory),
        **manifest_info,
    }
    results = [{**common, "split": "validation", **validation_metrics}]
    if test_frame is not None:
        test_grid, test_true = eval_inputs(test_frame, heldout_aspect, variant)
        test_flat_scores = score_unified_pair_manifest(model, tokenizer, test_grid, config, device)
        test_scores = score_matrix_from_grid(test_grid, test_flat_scores, row_count=len(test_frame))
        test_metrics, test_predictions = evaluate_score_matrix(
            test_true,
            test_scores,
            heldout_aspect,
            best_threshold,
        )
        test_score_rows = test_grid.copy()
        test_score_rows["score"] = test_flat_scores
        test_score_rows.to_csv(model_output / "test_pair_scores.csv", index=False)
        write_predictions(
            test_frame,
            test_scores,
            test_predictions,
            best_threshold,
            model_output / "test_predictions.jsonl",
        )
        results.append({**common, "split": "test", **test_metrics})

    write_json(
        {
            "protocol_id": "loao_unified_candidate_pair_experimental_v1",
            "config": config.__dict__,
            "history": history,
            "results": results,
            "wall_seconds": float(time.time() - started),
        },
        model_output / "summary.json",
    )
    del model, tokenizer, optimizer, scheduler, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def load_completed_results(path: Path) -> list[dict[str, object]] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("results")
    return rows if isinstance(rows, list) else None


def aggregate_results(
    results: list[dict[str, object]],
    output_dir: Path,
    stage: str,
) -> dict[str, object]:
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "all_results.csv", index=False)
    metric_columns = [
        "pair_micro_f1",
        "pair_micro_precision",
        "pair_micro_recall",
        "pair_samples_f1",
        "presence_f1",
        "presence_precision",
        "presence_recall",
        "presence_false_positive_rows_per_100",
        "presence_false_negative_rows_per_100",
        "conditional_sentiment_macro_f1",
        "conditional_sentiment_detection_coverage",
    ]
    spread_rows: list[dict[str, object]] = []
    for (model, variant, split), group in frame.groupby(["model", "variant", "split"]):
        row: dict[str, object] = {
            "model": model,
            "variant": variant,
            "split": split,
            "folds": int(group["heldout_aspect"].nunique()),
        }
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        spread_rows.append(row)
    spread = pd.DataFrame(spread_rows)
    spread.to_csv(output_dir / "spread.csv", index=False)

    gates: dict[str, object] = {}
    paired: dict[str, object] = {}
    for model in sorted(frame["model"].unique()):
        for split in sorted(frame["split"].unique()):
            subset = frame[(frame["model"] == model) & (frame["split"] == split)]
            control = subset[subset["variant"] == "control"]
            enhanced = subset[subset["variant"] == "enhanced"]
            if control.empty or enhanced.empty:
                continue
            if set(control["heldout_aspect"]) != set(enhanced["heldout_aspect"]):
                continue
            if stage == "pilot" and split == "validation":
                gates[model] = pilot_gate(enhanced, control)
            aligned = enhanced[["heldout_aspect", "pair_micro_f1"]].merge(
                control[["heldout_aspect", "pair_micro_f1"]],
                on="heldout_aspect",
                suffixes=("_enhanced", "_control"),
                validate="one_to_one",
            )
            paired[f"{model}_{split}"] = paired_aspect_statistics(
                aligned["pair_micro_f1_enhanced"],
                aligned["pair_micro_f1_control"],
            )

    summary = {
        "stage": stage,
        "results": results,
        "spread": spread.to_dict(orient="records"),
        "pilot_gates": gates,
        "paired_statistics": paired,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated, preregistered unified candidate-pair LOAO local experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experimental" / "loao_unified_candidate_pair_v1" / "local",
    )
    parser.add_argument("--stage", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--model", action="append", choices=MODELS, default=[])
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=[])
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-budget", type=int, default=4096)
    parser.add_argument("--positive-budget", type=int, default=2048)
    parser.add_argument("--train-row-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--distilbert-model-name", default="distilbert-base-uncased")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preregistration = load_experiment_config()
    if args.stage != "smoke":
        if args.seed != 13 or args.train_budget != 4096 or args.positive_budget != 2048 or args.epochs != 3:
            raise ValueError("Formal pilot/full runs must use the preregistered seed, budget, and epochs.")
        if args.train_row_limit is not None or args.eval_limit is not None:
            raise ValueError("Formal pilot/full runs cannot apply row limits.")

    models = args.model or list(MODELS)
    variants = args.variant or list(VARIANTS)
    frame = load_all_fabsa(args.data_dir)
    available_aspects = all_aspects(frame)
    if args.heldout_aspect:
        folds = args.heldout_aspect
    elif args.stage == "pilot":
        folds = list(PILOT_FOLDS)
    elif args.stage == "smoke":
        folds = [PILOT_FOLDS[0]]
    else:
        folds = available_aspects
    unknown = sorted(set(folds) - set(available_aspects))
    if unknown:
        raise ValueError(f"Unknown held-out aspects: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_manifest = {
        "protocol_id": preregistration["protocol_id"],
        "stage": args.stage,
        "formal_preregistered_run": args.stage in {"pilot", "full"},
        "models": models,
        "variants": variants,
        "folds": folds,
        "command": " ".join(sys.argv),
        "cwd": str(PROJECT_ROOT),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_status_at_start": git_status(),
        "packages": package_versions(),
        "hardware": hardware_summary(),
        "device": str(device),
        "arguments": vars(args),
    }
    write_json(run_manifest, args.output_dir / "run_manifest.json")
    print(json.dumps({"event": "run_start", **run_manifest}, default=json_default), flush=True)

    all_results: list[dict[str, object]] = []
    for fold_number, heldout_aspect in enumerate(folds, start=1):
        print(
            json.dumps(
                {
                    "event": "fold_start",
                    "fold": fold_number,
                    "folds": len(folds),
                    "heldout_aspect": heldout_aspect,
                }
            ),
            flush=True,
        )
        splits = build_heldout_aspect_split(
            frame,
            [heldout_aspect],
            strategy="example_filtered",
            eval_label_scope="heldout",
            eval_row_scope="all",
        )
        train_frame = splits["train"]
        if args.train_row_limit is not None:
            train_frame = train_frame.head(args.train_row_limit).copy()
        validation_frame = ordered_eval_frame(splits["validation"], args.eval_limit)
        test_frame = (
            ordered_eval_frame(splits["test"], args.eval_limit)
            if args.stage == "full"
            else None
        )
        fold_output = args.output_dir / slugify(heldout_aspect)

        for variant in variants:
            train_manifest, manifest_info = make_training_manifest(
                train_frame,
                heldout_aspect,
                variant,
                args,
                fold_output,
            )
            for model_name in models:
                model_output = fold_output / model_name / variant
                summary_path = model_output / "summary.json"
                if args.resume:
                    completed = load_completed_results(summary_path)
                    if completed is not None:
                        all_results.extend(completed)
                        print(
                            json.dumps(
                                {
                                    "event": "resume_completed_fold",
                                    "model": model_name,
                                    "variant": variant,
                                    "heldout_aspect": heldout_aspect,
                                }
                            ),
                            flush=True,
                        )
                        continue
                model_output.mkdir(parents=True, exist_ok=True)
                if model_name == "tfidf":
                    results = run_tfidf_fold(
                        train_manifest,
                        validation_frame,
                        test_frame,
                        heldout_aspect,
                        variant,
                        model_output,
                        args,
                        manifest_info,
                    )
                else:
                    results = run_distilbert_fold(
                        train_manifest,
                        validation_frame,
                        test_frame,
                        heldout_aspect,
                        variant,
                        model_output,
                        args,
                        manifest_info,
                        device,
                    )
                all_results.extend(results)
                aggregate_results(all_results, args.output_dir, args.stage)

    run_manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_manifest["git_status_at_finish"] = git_status()
    write_json(run_manifest, args.output_dir / "run_manifest.json")
    summary = aggregate_results(all_results, args.output_dir, args.stage)
    print(json.dumps({"event": "run_complete", "spread": summary["spread"], "gates": summary["pilot_gates"]}, default=json_default), flush=True)


if __name__ == "__main__":
    main()
