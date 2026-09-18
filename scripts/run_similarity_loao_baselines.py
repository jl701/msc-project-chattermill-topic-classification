from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import (
    build_sentiment_feature_lookup,
    candidate_aspect_predictions_from_scores,
    candidate_pair_labels,
    pair_predictions_from_aspects,
    sentiment_lookup_from_features,
    train_sentiment_model,
)
from msc_project.baselines.candidate_similarity import (
    REGISTERED_CONFIGS,
    CandidateSimilarityConfig,
    FrozenTransformerSentenceEncoder,
    SparseCandidateScorer,
    resolve_configs,
)
from msc_project.data.fabsa import default_data_dir, load_split
from msc_project.data.splits import all_aspects, build_heldout_aspect_split
from msc_project.evaluation.metrics import evaluate_pair_and_aspect


PROTOCOL_ID = "loao_open_topic_all_row_v1"
SCHEMA_VERSION = "loao_candidate_similarity/v1"
THRESHOLDS = [round(value / 100, 2) for value in range(-100, 101)]
SELECTION_COLUMNS = [
    "pair_micro_f1",
    "pair_samples_f1",
    "pair_macro_f1",
    "presence_f1",
    "presence_precision",
    "threshold",
]
AGGREGATE_METRICS = [
    "pair_samples_f1",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
    "pair_false_positive_rows_per_100",
    "pair_false_negative_rows_per_100",
    "pair_exact_match_rate",
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "presence_average_precision",
    "presence_prevalence",
    "presence_false_positive_rows_per_100",
    "presence_false_negative_rows_per_100",
    "sentiment_accuracy_when_gold_aspect_predicted",
    "sentiment_detection_coverage",
    "oracle_presence_sentiment_accuracy",
    "oracle_presence_sentiment_conflict_rows",
]


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable.")


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=json_default),
        encoding="utf-8",
    )


def aspect_slug(aspect: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", aspect).strip("_").lower() or "aspect"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(data: object) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_stage_frame(data_dir: Path, stage: str) -> pd.DataFrame:
    """Load train plus exactly one evaluation split to preserve stage isolation."""

    frames = []
    for split_name in ("train", stage):
        frame = load_split(data_dir, split_name).copy()
        frame["original_split"] = split_name
        frame["row_uid"] = frame["original_split"].astype(str) + ":" + frame["id"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def selected_aspects(frame: pd.DataFrame, requested: list[str]) -> list[str]:
    available = all_aspects(frame)
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown held-out aspects: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("Held-out aspects must be unique.")
    return requested


def build_fold(frame: pd.DataFrame, aspect: str, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = build_heldout_aspect_split(
        frame,
        [aspect],
        strategy="example_filtered",
        eval_label_scope="heldout",
        eval_row_scope="all",
    )
    train = splits["train"]
    evaluation = splits[stage]
    if train["labels"].apply(lambda labels: any(label_aspect == aspect for label_aspect, _ in labels)).any():
        raise RuntimeError(f"Held-out aspect leaked into example-filtered training fold: {aspect}")
    if evaluation["row_uid"].duplicated().any():
        raise RuntimeError(f"Duplicate evaluation row UID in {stage} fold: {aspect}")
    return train, evaluation


def positive_average_precision(gold_pairs: list[list[str]], scores: np.ndarray) -> float:
    target = np.asarray([bool(labels) for labels in gold_pairs], dtype=int)
    if target.min() == target.max():
        raise ValueError("Presence average precision requires both present and absent evaluation rows.")
    return float(average_precision_score(target, np.asarray(scores, dtype=float)))


def oracle_sentiment_metrics(
    eval_df: pd.DataFrame,
    aspect: str,
    sentiment_lookup: list[dict[str, str]],
) -> dict[str, float | int]:
    correct = 0
    total = 0
    conflict_rows = 0
    for labels, row_sentiments in zip(eval_df["supervision_labels"], sentiment_lookup):
        gold_sentiments = {sentiment for label_aspect, sentiment in labels if label_aspect == aspect}
        if not gold_sentiments:
            continue
        total += 1
        if len(gold_sentiments) > 1:
            conflict_rows += 1
        if row_sentiments.get(aspect) in gold_sentiments:
            correct += 1
    return {
        "oracle_presence_sentiment_accuracy": float(correct / total) if total else 0.0,
        "oracle_presence_sentiment_correct_rows": int(correct),
        "oracle_presence_sentiment_rows": int(total),
        "oracle_presence_sentiment_conflict_rows": int(conflict_rows),
    }


def evaluate_scores(
    eval_df: pd.DataFrame,
    aspect: str,
    scores: np.ndarray,
    threshold: float,
    sentiment_lookup: list[dict[str, str]],
) -> tuple[dict[str, float | int], list[list[str]], list[list[str]]]:
    score_matrix = np.asarray(scores, dtype=float).reshape(-1, 1)
    aspect_predictions = candidate_aspect_predictions_from_scores(
        score_matrix,
        [aspect],
        threshold,
        ensure_one=False,
    )
    pair_predictions = pair_predictions_from_aspects(aspect_predictions, sentiment_lookup)
    pair_classes = candidate_pair_labels([aspect])
    metrics: dict[str, float | int] = evaluate_pair_and_aspect(
        eval_df["supervision_pair_labels"].tolist(),
        pair_predictions,
        pair_classes,
    )
    metrics["presence_average_precision"] = positive_average_precision(
        eval_df["supervision_pair_labels"].tolist(),
        scores,
    )
    positive_rows = int(sum(bool(labels) for labels in eval_df["supervision_pair_labels"]))
    # Coverage is review-level: a present review counts once even if its held-out
    # aspect has more than one gold sentiment pair.
    detected_sentiment_rows = int(metrics["presence_tp_rows"])
    metrics["sentiment_detection_coverage"] = (
        float(detected_sentiment_rows / positive_rows) if positive_rows else 0.0
    )
    metrics.update(oracle_sentiment_metrics(eval_df, aspect, sentiment_lookup))
    metrics["threshold"] = float(threshold)
    return metrics, aspect_predictions, pair_predictions


def select_threshold(
    eval_df: pd.DataFrame,
    aspect: str,
    scores: np.ndarray,
    sentiment_lookup: list[dict[str, str]],
) -> tuple[float, pd.DataFrame, dict[str, float | int], list[list[str]], list[list[str]]]:
    rows: list[dict[str, float | int]] = []
    for threshold in THRESHOLDS:
        metrics, _, _ = evaluate_scores(eval_df, aspect, scores, threshold, sentiment_lookup)
        rows.append(metrics)
    sweep = pd.DataFrame(rows).sort_values(
        SELECTION_COLUMNS,
        ascending=[False] * len(SELECTION_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)
    selected_threshold = float(sweep.iloc[0]["threshold"])
    metrics, aspects, pairs = evaluate_scores(
        eval_df,
        aspect,
        scores,
        selected_threshold,
        sentiment_lookup,
    )
    return selected_threshold, sweep, metrics, aspects, pairs


def write_predictions(
    eval_df: pd.DataFrame,
    method: CandidateSimilarityConfig,
    stage: str,
    fold_index: int,
    aspect: str,
    scores: np.ndarray,
    threshold: float,
    aspect_predictions: list[list[str]],
    pair_predictions: list[list[str]],
    sentiment_features: list[dict[str, dict[str, object]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_index, (_, row) in enumerate(eval_df.iterrows()):
            payload = {
                "schema_version": SCHEMA_VERSION,
                "fold_index": int(fold_index),
                "method": method.name,
                "heldout_aspect": aspect,
                "split": stage,
                "row_index": int(row_index),
                "id": str(row["id"]),
                "row_uid": str(row["row_uid"]),
                "original_split": str(row["original_split"]),
                "org_index": int(row["org_index"]),
                "candidate_aspects": [aspect],
                "gold_pair_labels": list(row["supervision_pair_labels"]),
                "pred_pair_labels": pair_predictions[row_index],
                "presence_score": float(scores[row_index]),
                "selected_threshold": float(threshold),
                "predicted_present": bool(aspect_predictions[row_index]),
                "sentiment_features": sentiment_features[row_index][aspect],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=json_default) + "\n")


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    missing = [metric for metric in AGGREGATE_METRICS if metric not in results]
    if missing:
        raise ValueError(f"Required aggregate metrics are missing: {missing}")
    rows: list[dict[str, object]] = []
    for (method, stage), group in results.groupby(["method", "split"], sort=True):
        row: dict[str, object] = {
            "method": str(method),
            "split": str(stage),
            "aspects": int(group["heldout_aspect"].nunique()),
        }
        for metric in AGGREGATE_METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "split"]).reset_index(drop=True)


def read_selection_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen validation selection manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_test_selection(
    selection: dict[str, object],
    configs: list[CandidateSimilarityConfig],
    aspects: list[str],
    config_fingerprint: str,
) -> dict[str, dict[str, float]]:
    if selection.get("stage") != "validation":
        raise ValueError("Test selection manifest must come from the validation stage.")
    if selection.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Validation selection manifest uses a different protocol.")
    if selection.get("protocol_complete") is not True:
        raise ValueError("Test execution requires a protocol-complete validation selection manifest.")
    if selection.get("config_fingerprint") != config_fingerprint:
        raise ValueError("Validation selection manifest configuration does not match this test run.")
    expected_methods = [config.name for config in configs]
    if selection.get("methods") != expected_methods:
        raise ValueError("Validation selection manifest methods do not match this test run.")
    if selection.get("aspects") != aspects:
        raise ValueError("Validation selection manifest aspects do not match this test run.")
    thresholds = selection.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Validation selection manifest has no threshold mapping.")
    if list(thresholds) != expected_methods:
        raise ValueError("Validation threshold mapping has missing, extra, or reordered methods.")
    for method in expected_methods:
        method_thresholds = thresholds[method]
        if not isinstance(method_thresholds, dict) or list(method_thresholds) != aspects:
            raise ValueError(f"Validation threshold mapping for {method} does not cover all aspects exactly.")
        for aspect, threshold in method_thresholds.items():
            value = float(threshold)
            if not np.isfinite(value) or value < -1.0 or value > 1.0:
                raise ValueError(f"Invalid frozen threshold for {method} / {aspect}: {threshold}")
    return {
        str(method): {str(aspect): float(threshold) for aspect, threshold in values.items()}
        for method, values in thresholds.items()
    }


def runtime_manifest(
    args: argparse.Namespace,
    data_dir: Path,
    configs: list[CandidateSimilarityConfig],
    aspects: list[str],
    config_fingerprint: str,
    stage_started: float,
) -> dict[str, object]:
    import torch
    import transformers

    status = git_value("status", "--porcelain")
    data_files = {split_name: data_dir / f"{split_name}.csv" for split_name in ("train", args.stage)}
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "id": PROTOCOL_ID,
            "strategy": "example_filtered",
            "eval_row_scope": "all",
            "eval_label_scope": "heldout",
            "candidate_cardinality": 1,
            "candidate_text_policy": "canonical_exact",
            "prediction_representation": "pair_set",
            "allow_empty": True,
            "calibration_regime": "target_calibrated",
            "selection_split": "validation",
            "selection_metric": "pair_micro_f1",
            "threshold_grid": {"minimum": -1.0, "maximum": 1.0, "step": 0.01},
            "tie_breakers": SELECTION_COLUMNS[1:],
        },
        "stage": args.stage,
        "protocol_complete": bool(len(aspects) == 12 and len(configs) == len(REGISTERED_CONFIGS)),
        "methods": [config.to_dict() for config in configs],
        "config_fingerprint": config_fingerprint,
        "data": {
            "split_sha256": {name: sha256_file(path) for name, path in data_files.items()},
            "aspects": aspects,
            "expected_fold_count": 12,
        },
        "reproducibility": {
            "seed": 13,
            "command": sys.argv,
            "cwd": str(Path.cwd()),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(status),
            "git_status": status.splitlines(),
            "python": platform.python_version(),
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn_version,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "hardware": {
                "platform": platform.platform(),
                "device_argument": args.device,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        },
        "stage_started_unix": float(stage_started),
    }


def precompute_sentence_scores(
    configs: list[CandidateSimilarityConfig],
    eval_texts: list[str],
    aspects: list[str],
    device: str,
    local_files_only: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    score_matrices: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for config in configs:
        if config.model_id is None:
            continue
        print(json.dumps({"event": "load_encoder", "method": config.name}), flush=True)
        encoder = FrozenTransformerSentenceEncoder(
            config,
            device=device,
            local_files_only=local_files_only,
        )
        try:
            matrix, method_diagnostics = encoder.score_matrix(eval_texts, aspects)
        finally:
            encoder.close()
        if matrix.shape != (len(eval_texts), len(aspects)):
            raise RuntimeError(f"Unexpected sentence score matrix shape for {config.name}: {matrix.shape}")
        score_matrices[config.name] = matrix
        diagnostics[config.name] = method_diagnostics
    return score_matrices, diagnostics


def run(args: argparse.Namespace) -> dict[str, object]:
    stage_started = time.time()
    data_dir = args.data_dir.resolve()
    configs = resolve_configs(args.model)
    selection: dict[str, object] | None = None
    if args.stage == "test":
        expected_methods = list(REGISTERED_CONFIGS)
        if [config.name for config in configs] != expected_methods or args.heldout_aspect:
            raise ValueError("The frozen test stage must run all registered methods and all 12 aspects.")
        selection_path = args.selection_manifest or (args.output_dir / "validation" / "selection_manifest.json")
        selection = read_selection_manifest(selection_path)
        selection_aspects = selection.get("aspects")
        if (
            not isinstance(selection_aspects, list)
            or len(selection_aspects) != 12
            or len(set(str(value) for value in selection_aspects)) != 12
        ):
            raise ValueError("Test execution requires exactly 12 unique validation-selected aspects.")
        aspects = [str(value) for value in selection_aspects]
    else:
        frame = load_stage_frame(data_dir, args.stage)
        aspects = selected_aspects(frame, args.heldout_aspect)

    config_payload = {
        "protocol_id": PROTOCOL_ID,
        "methods": [config.to_dict() for config in configs],
        "aspects": aspects,
        "thresholds": THRESHOLDS,
        "selection_columns": SELECTION_COLUMNS,
        "sentiment_mode": "global",
    }
    config_fingerprint = canonical_json_hash(config_payload)
    thresholds: dict[str, dict[str, float]] = {config.name: {} for config in configs}
    if args.stage == "test":
        if selection is None:
            raise RuntimeError("Test selection was not loaded before the test-data gate.")
        thresholds = validate_test_selection(
            selection,
            configs,
            aspects,
            config_fingerprint,
        )
        # Test data is loaded only after the complete frozen selection passes.
        frame = load_stage_frame(data_dir, args.stage)
        available_aspects = all_aspects(frame)
        if available_aspects != aspects:
            raise ValueError("Test data aspects do not match the frozen validation aspect order.")

    stage_dir = args.output_dir / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    manifest = runtime_manifest(
        args,
        data_dir,
        configs,
        aspects,
        config_fingerprint,
        stage_started,
    )
    write_json(manifest, stage_dir / "manifest.json")

    reference_eval: pd.DataFrame | None = None
    for aspect in aspects:
        _, evaluation = build_fold(frame, aspect, args.stage)
        if reference_eval is None:
            reference_eval = evaluation
        elif evaluation["row_uid"].tolist() != reference_eval["row_uid"].tolist():
            raise RuntimeError("All-row evaluation UID order changed across held-out aspects.")
    if reference_eval is None:
        raise RuntimeError("No held-out aspects were selected.")

    sentence_scores, sentence_diagnostics = precompute_sentence_scores(
        configs,
        reference_eval["text"].tolist(),
        aspects,
        args.device,
        args.local_files_only,
    )

    result_rows: list[dict[str, object]] = []
    fold_diagnostics: dict[str, dict[str, object]] = {config.name: {} for config in configs}
    for fold_index, aspect in enumerate(aspects, start=1):
        print(
            json.dumps(
                {
                    "event": "fold",
                    "stage": args.stage,
                    "fold_index": fold_index,
                    "folds": len(aspects),
                    "heldout_aspect": aspect,
                }
            ),
            flush=True,
        )
        train_df, eval_df = build_fold(frame, aspect, args.stage)
        sentiment_model = train_sentiment_model(train_df)
        sentiment_features = build_sentiment_feature_lookup(
            sentiment_model,
            eval_df["text"].tolist(),
            [aspect],
            "global",
        )
        sentiment_lookup = sentiment_lookup_from_features(sentiment_features)

        for config in configs:
            if config.model_id is None:
                scorer = SparseCandidateScorer(config).fit(train_df["text"].tolist())
                scores, diagnostics = scorer.score(eval_df["text"].tolist(), aspect)
            else:
                scores = sentence_scores[config.name][:, fold_index - 1]
                diagnostics = dict(sentence_diagnostics[config.name])

            fold_dir = stage_dir / config.name / "folds" / f"{fold_index:02d}_{aspect_slug(aspect)}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            if args.stage == "validation":
                threshold, sweep, metrics, aspect_predictions, pair_predictions = select_threshold(
                    eval_df,
                    aspect,
                    scores,
                    sentiment_lookup,
                )
                thresholds[config.name][aspect] = threshold
                sweep.to_csv(fold_dir / "validation_threshold_sweep.csv", index=False)
            else:
                threshold = float(thresholds[config.name][aspect])
                metrics, aspect_predictions, pair_predictions = evaluate_scores(
                    eval_df,
                    aspect,
                    scores,
                    threshold,
                    sentiment_lookup,
                )

            write_predictions(
                eval_df,
                config,
                args.stage,
                fold_index,
                aspect,
                scores,
                threshold,
                aspect_predictions,
                pair_predictions,
                sentiment_features,
                fold_dir / f"predictions_{args.stage}.jsonl",
            )
            diagnostics.update(
                {
                    "train_rows": int(len(train_df)),
                    "eval_rows": int(len(eval_df)),
                    "eval_positive_rows": int(sum(bool(labels) for labels in eval_df["supervision_pair_labels"])),
                    "selected_threshold": float(threshold),
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                    "score_mean": float(np.mean(scores)),
                    "score_std": float(np.std(scores)),
                }
            )
            fold_diagnostics[config.name][aspect] = diagnostics
            write_json(diagnostics, fold_dir / "diagnostics.json")
            result_rows.append(
                {
                    "method": config.name,
                    "family": config.family,
                    "split": args.stage,
                    "heldout_aspect": aspect,
                    "fold_index": int(fold_index),
                    "train_rows": int(len(train_df)),
                    "eval_rows": int(len(eval_df)),
                    **metrics,
                }
            )

    results = pd.DataFrame(result_rows).sort_values(["method", "fold_index"]).reset_index(drop=True)
    spread = aggregate_results(results)
    results.to_csv(stage_dir / "all_results.csv", index=False)
    spread.to_csv(stage_dir / "spread.csv", index=False)
    write_json(fold_diagnostics, stage_dir / "fold_diagnostics.json")

    if args.stage == "validation":
        selection_manifest = {
            "schema_version": SCHEMA_VERSION,
            "stage": "validation",
            "protocol_id": PROTOCOL_ID,
            "config_fingerprint": config_fingerprint,
            "methods": [config.name for config in configs],
            "aspects": aspects,
            "thresholds": thresholds,
            "selection_columns": SELECTION_COLUMNS,
            "protocol_complete": bool(len(aspects) == 12 and len(configs) == len(REGISTERED_CONFIGS)),
        }
        write_json(selection_manifest, stage_dir / "selection_manifest.json")
        threshold_rows = [
            {"method": method, "heldout_aspect": aspect, "threshold": threshold}
            for method, method_thresholds in thresholds.items()
            for aspect, threshold in method_thresholds.items()
        ]
        pd.DataFrame(threshold_rows).to_csv(stage_dir / "selected_thresholds.csv", index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "protocol_complete": bool(len(aspects) == 12 and len(configs) == len(REGISTERED_CONFIGS)),
        "config_fingerprint": config_fingerprint,
        "fold_rows": results.to_dict(orient="records"),
        "aggregate": spread.to_dict(orient="records"),
        "runtime_seconds": float(time.time() - stage_started),
    }
    write_json(summary, stage_dir / "summary.json")

    if args.public_output_dir is not None:
        args.public_output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(
            args.public_output_dir / f"loao_bow_sentence_embedding_v1_{args.stage}_per_aspect.csv",
            index=False,
        )
        spread.to_csv(
            args.public_output_dir / f"loao_bow_sentence_embedding_v1_{args.stage}_aggregate.csv",
            index=False,
        )
        if args.stage == "validation":
            pd.DataFrame(
                [
                    {"method": method, "heldout_aspect": aspect, "threshold": threshold}
                    for method, method_thresholds in thresholds.items()
                    for aspect, threshold in method_thresholds.items()
                ]
            ).to_csv(
                args.public_output_dir / "loao_bow_sentence_embedding_v1_selected_thresholds.csv",
                index=False,
            )

    print(json.dumps(summary["aggregate"], indent=2, default=json_default), flush=True)
    print(f"Saved {args.stage} outputs to {stage_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run protocol-matched BoW, strict TF-IDF, and frozen sentence-embedding LOAO baselines."
    )
    parser.add_argument("--stage", choices=["validation", "test"], required=True)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baselines" / "loao_bow_sentence_embedding_v1",
    )
    parser.add_argument("--public-output-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        action="append",
        choices=list(REGISTERED_CONFIGS),
        default=[],
        help="Registered method to run; repeat as needed. Defaults to all four methods.",
    )
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--selection-manifest", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
