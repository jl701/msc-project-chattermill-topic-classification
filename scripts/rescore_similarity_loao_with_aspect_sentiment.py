from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from msc_project.evaluation.metrics import evaluate_pair_and_aspect, pair_to_components


METHODS = [
    "bow_count_1_2_train_vocab",
    "tfidf_char_3_5_train_vocab",
    "minilm_l6_v2",
    "e5_base_v2",
]
SENTIMENTS = ("negative", "neutral", "positive")
THRESHOLDS = [round(value / 100, 2) for value in range(-100, 101)]
SELECTION_COLUMNS = [
    "pair_micro_f1",
    "pair_samples_f1",
    "pair_macro_f1",
    "presence_f1",
    "presence_precision",
    "threshold",
]
SCENARIOS = [
    "global_original",
    "aspect_conditioned_fixed_threshold",
    "aspect_conditioned_validation_reselected",
]
AGGREGATE_METRICS = [
    "pair_samples_f1",
    "pair_micro_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "pair_macro_f1",
    "pair_exact_match_rate",
    "pair_false_positive_rows_per_100",
    "pair_false_negative_rows_per_100",
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "presence_false_positive_rows_per_100",
    "presence_false_negative_rows_per_100",
    "sentiment_accuracy_when_gold_aspect_predicted",
    "oracle_presence_sentiment_accuracy",
    "oracle_presence_sentiment_single_label_accuracy",
    "oracle_presence_sentiment_rows",
    "oracle_presence_sentiment_conflict_rows",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no prediction rows.")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_prediction_rows(
    strict_rows: list[dict[str, Any]],
    sentiment_rows: list[dict[str, Any]],
    *,
    expected_method: str,
    expected_stage: str,
) -> tuple[str, list[dict[str, Any]]]:
    if len(strict_rows) != len(sentiment_rows):
        raise ValueError(
            f"Row-count mismatch: strict={len(strict_rows)}, sentiment={len(sentiment_rows)}."
        )

    strict_uids = [str(row.get("row_uid")) for row in strict_rows]
    sentiment_uids = [str(row.get("row_uid")) for row in sentiment_rows]
    if len(set(strict_uids)) != len(strict_uids):
        raise ValueError("Duplicate row_uid in strict similarity predictions.")
    if len(set(sentiment_uids)) != len(sentiment_uids):
        raise ValueError("Duplicate row_uid in aspect-conditioned sentiment predictions.")
    if strict_uids != sentiment_uids:
        mismatch = next(
            index
            for index, (strict_uid, sentiment_uid) in enumerate(
                zip(strict_uids, sentiment_uids)
            )
            if strict_uid != sentiment_uid
        )
        raise ValueError(
            "row_uid sequence mismatch at row "
            f"{mismatch}: strict={strict_uids[mismatch]!r}, "
            f"sentiment={sentiment_uids[mismatch]!r}."
        )

    first_candidates = strict_rows[0].get("candidate_aspects")
    if not isinstance(first_candidates, list) or len(first_candidates) != 1:
        raise ValueError("Strict predictions must contain exactly one candidate aspect.")
    aspect = str(first_candidates[0])
    aligned: list[dict[str, Any]] = []

    for index, (strict_row, sentiment_row) in enumerate(
        zip(strict_rows, sentiment_rows)
    ):
        prefix = f"row {index} ({strict_uids[index]})"
        if strict_row.get("method") != expected_method:
            raise ValueError(f"{prefix}: unexpected strict method.")
        if strict_row.get("split") != expected_stage:
            raise ValueError(f"{prefix}: unexpected strict split.")
        if sentiment_row.get("original_split") != expected_stage:
            raise ValueError(f"{prefix}: unexpected sentiment split.")
        if strict_row.get("heldout_aspect") != aspect:
            raise ValueError(f"{prefix}: held-out aspect changed within the strict fold.")
        if strict_row.get("candidate_aspects") != [aspect]:
            raise ValueError(f"{prefix}: candidate aspect is not the singleton fold aspect.")
        if strict_row.get("gold_pair_labels") != sentiment_row.get("gold_pair_labels"):
            raise ValueError(f"{prefix}: gold pair labels differ between artifacts.")

        score = float(strict_row.get("presence_score"))
        threshold = float(strict_row.get("selected_threshold"))
        if not math.isfinite(score) or not math.isfinite(threshold):
            raise ValueError(f"{prefix}: non-finite presence score or threshold.")
        expected_present = score >= threshold
        if bool(strict_row.get("predicted_present")) != expected_present:
            raise ValueError(f"{prefix}: stored presence decision disagrees with score.")
        global_pairs = strict_row.get("pred_pair_labels")
        if not isinstance(global_pairs, list) or len(global_pairs) > 1:
            raise ValueError(f"{prefix}: global prediction is not a zero-or-one pair set.")
        if bool(global_pairs) != expected_present:
            raise ValueError(f"{prefix}: global pair presence disagrees with presence score.")

        strict_sentiment = strict_row.get("sentiment_features")
        if not isinstance(strict_sentiment, dict):
            raise ValueError(f"{prefix}: missing global sentiment features.")
        global_sentiment = str(strict_sentiment.get("predicted_sentiment"))
        if global_sentiment not in SENTIMENTS:
            raise ValueError(f"{prefix}: invalid global sentiment {global_sentiment!r}.")
        if global_pairs:
            predicted_aspect, predicted_sentiment = pair_to_components(
                str(global_pairs[0])
            )
            if predicted_aspect != aspect or predicted_sentiment != global_sentiment:
                raise ValueError(
                    f"{prefix}: global pair label disagrees with its sentiment features."
                )

        sentiment_features = sentiment_row.get("sentiment_features")
        if not isinstance(sentiment_features, dict) or list(sentiment_features) != [
            aspect
        ]:
            raise ValueError(
                f"{prefix}: aspect-conditioned features do not cover exactly {aspect!r}."
            )
        aspect_features = sentiment_features[aspect]
        if not isinstance(aspect_features, dict):
            raise ValueError(f"{prefix}: malformed aspect-conditioned sentiment features.")
        aspect_sentiment = str(aspect_features.get("predicted_sentiment"))
        if aspect_sentiment not in SENTIMENTS:
            raise ValueError(
                f"{prefix}: invalid aspect-conditioned sentiment {aspect_sentiment!r}."
            )

        aligned.append(
            {
                "row_uid": strict_uids[index],
                "gold_pair_labels": [str(value) for value in strict_row["gold_pair_labels"]],
                "presence_score": score,
                "original_threshold": threshold,
                "global_pred_pair_labels": [str(value) for value in global_pairs],
                "global_sentiment": global_sentiment,
                "aspect_sentiment": aspect_sentiment,
            }
        )

    thresholds = {row["original_threshold"] for row in aligned}
    if len(thresholds) != 1:
        raise ValueError("Strict prediction rows do not share one frozen fold threshold.")
    return aspect, aligned


def predicted_pairs(
    rows: list[dict[str, Any]],
    aspect: str,
    threshold: float,
    sentiment_key: str,
) -> list[list[str]]:
    return [
        [f"{aspect} | {row[sentiment_key]}"]
        if float(row["presence_score"]) >= threshold
        else []
        for row in rows
    ]


def oracle_presence_sentiment_metrics(
    rows: list[dict[str, Any]],
    aspect: str,
    sentiment_key: str,
) -> dict[str, float | int]:
    total = 0
    compatible_correct = 0
    single_total = 0
    single_correct = 0
    conflict_rows = 0
    for row in rows:
        gold_sentiments = {
            sentiment
            for label in row["gold_pair_labels"]
            for gold_aspect, sentiment in [pair_to_components(label)]
            if gold_aspect == aspect
        }
        if not gold_sentiments:
            continue
        total += 1
        predicted = str(row[sentiment_key])
        compatible_correct += int(predicted in gold_sentiments)
        if len(gold_sentiments) == 1:
            single_total += 1
            single_correct += int(predicted in gold_sentiments)
        else:
            conflict_rows += 1
    return {
        "oracle_presence_sentiment_accuracy": (
            float(compatible_correct / total) if total else 0.0
        ),
        "oracle_presence_sentiment_single_label_accuracy": (
            float(single_correct / single_total) if single_total else 0.0
        ),
        "oracle_presence_sentiment_rows": total,
        "oracle_presence_sentiment_single_label_rows": single_total,
        "oracle_presence_sentiment_conflict_rows": conflict_rows,
    }


def score_predictions(
    rows: list[dict[str, Any]],
    aspect: str,
    predictions: list[list[str]],
    sentiment_key: str,
) -> dict[str, float | int]:
    gold = [row["gold_pair_labels"] for row in rows]
    pair_classes = [f"{aspect} | {sentiment}" for sentiment in SENTIMENTS]
    metrics: dict[str, float | int] = evaluate_pair_and_aspect(
        gold,
        predictions,
        pair_classes,
    )
    metrics.update(oracle_presence_sentiment_metrics(rows, aspect, sentiment_key))
    return metrics


def score_global_original(
    rows: list[dict[str, Any]],
    aspect: str,
) -> dict[str, float | int]:
    return score_predictions(
        rows,
        aspect,
        [row["global_pred_pair_labels"] for row in rows],
        "global_sentiment",
    )


def score_aspect_conditioned(
    rows: list[dict[str, Any]],
    aspect: str,
    threshold: float,
) -> dict[str, float | int]:
    return score_predictions(
        rows,
        aspect,
        predicted_pairs(rows, aspect, threshold, "aspect_sentiment"),
        "aspect_sentiment",
    )


def select_threshold(
    rows: list[dict[str, Any]],
    aspect: str,
) -> tuple[float, pd.DataFrame, dict[str, float | int]]:
    sweep_rows: list[dict[str, float | int]] = []
    for threshold in THRESHOLDS:
        metrics = score_aspect_conditioned(rows, aspect, threshold)
        sweep_rows.append({**metrics, "threshold": threshold})
    sweep = pd.DataFrame(sweep_rows).sort_values(
        SELECTION_COLUMNS,
        ascending=[False] * len(SELECTION_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)
    threshold = float(sweep.iloc[0]["threshold"])
    return threshold, sweep, score_aspect_conditioned(rows, aspect, threshold)


def fold_paths(
    strict_root: Path,
    sentiment_root: Path,
    method: str,
    stage: str,
    strict_fold_name: str,
    sentiment_fold_name: str,
) -> tuple[Path, Path]:
    strict_path = (
        strict_root
        / stage
        / method
        / "folds"
        / strict_fold_name
        / f"predictions_{stage}.jsonl"
    )
    sentiment_path = (
        sentiment_root
        / sentiment_fold_name
        / f"best_{stage}_predictions.jsonl"
    )
    return strict_path, sentiment_path


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, stage, scenario), frame in results.groupby(
        ["method", "stage", "scenario"],
        sort=False,
    ):
        row: dict[str, Any] = {
            "method": method,
            "stage": stage,
            "scenario": scenario,
            "folds": int(len(frame)),
        }
        for metric in AGGREGATE_METRICS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_median"] = float(frame[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = [
        (
            "aspect_conditioned_fixed_threshold_minus_global",
            "aspect_conditioned_fixed_threshold",
            "global_original",
        ),
        (
            "aspect_conditioned_validation_reselected_minus_global",
            "aspect_conditioned_validation_reselected",
            "global_original",
        ),
        (
            "aspect_conditioned_validation_reselected_minus_fixed_threshold",
            "aspect_conditioned_validation_reselected",
            "aspect_conditioned_fixed_threshold",
        ),
    ]
    for (method, stage), frame in results.groupby(["method", "stage"], sort=False):
        for comparison, challenger, reference in comparisons:
            challenger_rows = (
                frame.loc[
                    frame["scenario"] == challenger,
                    ["heldout_aspect", "pair_micro_f1"],
                ]
                .set_index("heldout_aspect")
                .sort_index()
            )
            reference_rows = (
                frame.loc[
                    frame["scenario"] == reference,
                    ["heldout_aspect", "pair_micro_f1"],
                ]
                .set_index("heldout_aspect")
                .sort_index()
            )
            if (
                len(challenger_rows) != 12
                or len(reference_rows) != 12
                or not challenger_rows.index.equals(reference_rows.index)
            ):
                raise ValueError(
                    f"Incomplete paired folds for {method} / {stage} / {comparison}."
                )
            differences = (
                challenger_rows["pair_micro_f1"]
                - reference_rows["pair_micro_f1"]
            )
            tolerance = 1e-12
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "comparison": comparison,
                    "mean_pair_micro_f1_difference": float(differences.mean()),
                    "median_pair_micro_f1_difference": float(differences.median()),
                    "challenger_wins": int((differences > tolerance).sum()),
                    "ties": int((differences.abs() <= tolerance).sum()),
                    "reference_wins": int((differences < -tolerance).sum()),
                }
            )
    return pd.DataFrame(rows)


def validate_selection_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("stage") != "validation" or not payload.get("protocol_complete"):
        raise ValueError("Selection manifest is not a complete validation result.")
    if payload.get("methods") != METHODS:
        raise ValueError("Selection manifest method set or order is invalid.")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 12:
        raise ValueError("Selection manifest must contain exactly 12 folds.")
    aspects = [str(fold.get("heldout_aspect")) for fold in folds]
    if len(set(aspects)) != 12:
        raise ValueError("Selection manifest aspects are missing or duplicated.")
    for fold in folds:
        thresholds = fold.get("thresholds")
        if not isinstance(thresholds, dict) or list(thresholds) != METHODS:
            raise ValueError("Selection manifest does not contain all method thresholds.")
        for value in thresholds.values():
            if float(value) not in THRESHOLDS:
                raise ValueError("Selection manifest contains an unregistered threshold.")
    return folds


def write_stage_outputs(
    output_dir: Path,
    stage: str,
    results: pd.DataFrame,
    sweeps: pd.DataFrame | None,
    audit: list[dict[str, Any]],
) -> None:
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(stage_dir / "per_fold_results.csv", index=False)
    aggregate_results(results).to_csv(stage_dir / "aggregate_results.csv", index=False)
    comparison_table(results).to_csv(stage_dir / "comparisons.csv", index=False)
    if sweeps is not None:
        sweeps.to_csv(stage_dir / "validation_threshold_sweeps.csv", index=False)
    (stage_dir / "alignment_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )


def make_result_row(
    *,
    method: str,
    stage: str,
    aspect: str,
    scenario: str,
    threshold: float,
    metrics: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "method": method,
        "stage": stage,
        "heldout_aspect": aspect,
        "scenario": scenario,
        "threshold": threshold,
        **metrics,
    }


def run_validation(
    strict_root: Path,
    sentiment_root: Path,
    output_dir: Path,
) -> Path:
    results: list[dict[str, Any]] = []
    sweep_frames: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    fold_manifest: dict[str, dict[str, Any]] = {}

    reference_fold_dirs = sorted(
        (strict_root / "validation" / METHODS[0] / "folds").iterdir()
    )
    if len(reference_fold_dirs) != 12:
        raise ValueError("Validation source must contain exactly 12 strict folds.")

    for strict_fold_dir in reference_fold_dirs:
        strict_fold_name = strict_fold_dir.name
        sentiment_fold_name = strict_fold_name.split("_", maxsplit=1)[1]
        fold_record: dict[str, Any] | None = None
        for method in METHODS:
            strict_path, sentiment_path = fold_paths(
                strict_root,
                sentiment_root,
                method,
                "validation",
                strict_fold_name,
                sentiment_fold_name,
            )
            strict_rows = read_jsonl(strict_path)
            sentiment_rows = read_jsonl(sentiment_path)
            aspect, rows = align_prediction_rows(
                strict_rows,
                sentiment_rows,
                expected_method=method,
                expected_stage="validation",
            )
            original_threshold = float(rows[0]["original_threshold"])
            selected_threshold, sweep, selected_metrics = select_threshold(rows, aspect)
            global_metrics = score_global_original(rows, aspect)
            fixed_metrics = score_aspect_conditioned(
                rows,
                aspect,
                original_threshold,
            )
            results.extend(
                [
                    make_result_row(
                        method=method,
                        stage="validation",
                        aspect=aspect,
                        scenario="global_original",
                        threshold=original_threshold,
                        metrics=global_metrics,
                    ),
                    make_result_row(
                        method=method,
                        stage="validation",
                        aspect=aspect,
                        scenario="aspect_conditioned_fixed_threshold",
                        threshold=original_threshold,
                        metrics=fixed_metrics,
                    ),
                    make_result_row(
                        method=method,
                        stage="validation",
                        aspect=aspect,
                        scenario="aspect_conditioned_validation_reselected",
                        threshold=selected_threshold,
                        metrics=selected_metrics,
                    ),
                ]
            )
            sweep.insert(0, "heldout_aspect", aspect)
            sweep.insert(0, "method", method)
            sweep_frames.append(sweep)
            audit.append(
                {
                    "method": method,
                    "stage": "validation",
                    "heldout_aspect": aspect,
                    "rows": len(rows),
                    "strict_path": str(strict_path),
                    "strict_sha256": sha256_file(strict_path),
                    "sentiment_path": str(sentiment_path),
                    "sentiment_sha256": sha256_file(sentiment_path),
                    "alignment": "exact",
                }
            )

            if fold_record is None:
                fold_record = {
                    "strict_fold_name": strict_fold_name,
                    "sentiment_fold_name": sentiment_fold_name,
                    "heldout_aspect": aspect,
                    "thresholds": {},
                }
            elif fold_record["heldout_aspect"] != aspect:
                raise ValueError("Methods disagree on the held-out aspect for a fold.")
            fold_record["thresholds"][method] = selected_threshold
        assert fold_record is not None
        fold_manifest[strict_fold_name] = fold_record

    results_frame = pd.DataFrame(results)
    write_stage_outputs(
        output_dir,
        "validation",
        results_frame,
        pd.concat(sweep_frames, ignore_index=True),
        audit,
    )
    selection = {
        "experiment_id": "loao_similarity_aspect_conditioned_sentiment_rescore_v1",
        "stage": "validation",
        "protocol_complete": len(fold_manifest) == 12,
        "methods": METHODS,
        "selection_columns": SELECTION_COLUMNS,
        "threshold_grid": {"minimum": -1.0, "maximum": 1.0, "step": 0.01},
        "folds": list(fold_manifest.values()),
    }
    selection_path = output_dir / "validation" / "selection_manifest.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    validate_selection_manifest(selection)
    return selection_path


def run_test(
    strict_root: Path,
    sentiment_root: Path,
    output_dir: Path,
    selection_manifest: Path,
) -> None:
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    folds = validate_selection_manifest(selection)
    results: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for fold in folds:
        strict_fold_name = str(fold["strict_fold_name"])
        sentiment_fold_name = str(fold["sentiment_fold_name"])
        expected_aspect = str(fold["heldout_aspect"])
        for method in METHODS:
            strict_path, sentiment_path = fold_paths(
                strict_root,
                sentiment_root,
                method,
                "test",
                strict_fold_name,
                sentiment_fold_name,
            )
            strict_rows = read_jsonl(strict_path)
            sentiment_rows = read_jsonl(sentiment_path)
            aspect, rows = align_prediction_rows(
                strict_rows,
                sentiment_rows,
                expected_method=method,
                expected_stage="test",
            )
            if aspect != expected_aspect:
                raise ValueError("Test aspect disagrees with the frozen validation fold.")
            original_threshold = float(rows[0]["original_threshold"])
            selected_threshold = float(fold["thresholds"][method])
            results.extend(
                [
                    make_result_row(
                        method=method,
                        stage="test",
                        aspect=aspect,
                        scenario="global_original",
                        threshold=original_threshold,
                        metrics=score_global_original(rows, aspect),
                    ),
                    make_result_row(
                        method=method,
                        stage="test",
                        aspect=aspect,
                        scenario="aspect_conditioned_fixed_threshold",
                        threshold=original_threshold,
                        metrics=score_aspect_conditioned(
                            rows,
                            aspect,
                            original_threshold,
                        ),
                    ),
                    make_result_row(
                        method=method,
                        stage="test",
                        aspect=aspect,
                        scenario="aspect_conditioned_validation_reselected",
                        threshold=selected_threshold,
                        metrics=score_aspect_conditioned(
                            rows,
                            aspect,
                            selected_threshold,
                        ),
                    ),
                ]
            )
            audit.append(
                {
                    "method": method,
                    "stage": "test",
                    "heldout_aspect": aspect,
                    "rows": len(rows),
                    "strict_path": str(strict_path),
                    "strict_sha256": sha256_file(strict_path),
                    "sentiment_path": str(sentiment_path),
                    "sentiment_sha256": sha256_file(sentiment_path),
                    "alignment": "exact",
                }
            )
    write_stage_outputs(
        output_dir,
        "test",
        pd.DataFrame(results),
        None,
        audit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore strict similarity LOAO presence artifacts with a shared "
            "aspect-conditioned sentiment component."
        )
    )
    parser.add_argument("--stage", choices=["validation", "test"], required=True)
    parser.add_argument(
        "--strict-root",
        type=Path,
        default=Path("outputs/baselines/loao_bow_sentence_embedding_v1"),
    )
    parser.add_argument("--sentiment-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/analysis/"
            "loao_similarity_aspect_conditioned_sentiment_rescore_v1"
        ),
    )
    parser.add_argument("--selection-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "validation":
        if args.selection_manifest is not None:
            raise ValueError("--selection-manifest is only valid for the test stage.")
        selection_path = run_validation(
            args.strict_root,
            args.sentiment_root,
            args.output_dir,
        )
        print(selection_path)
        return
    if args.selection_manifest is None:
        raise ValueError("The test stage requires --selection-manifest.")
    run_test(
        args.strict_root,
        args.sentiment_root,
        args.output_dir,
        args.selection_manifest,
    )
    print(args.output_dir / "test")


if __name__ == "__main__":
    main()
