from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS
from msc_project.evaluation.cascade import (
    combine_predictions,
    label_reliability,
    prediction_features,
    row_key,
    validate_aligned_rows,
)
from msc_project.evaluation.metrics import pair_to_aspect, pair_to_components


DEFAULT_LOCAL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baselines"
    / "aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered"
    / "example_filtered"
)
DEFAULT_GEMINI_DIR = PROJECT_ROOT / "outputs" / "llm" / "gemini_candidate_label_20260701_0145_fixed_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "local_gemini_cascade"
COMBINATION_MODES = (
    "replace",
    "gemini_nonempty_else_local",
    "union",
    "intersection",
    "agreement_or_gemini",
    "agreement_or_local",
)
SELECTION_KEYS = ("pair_samples_f1", "pair_micro_f1", "pair_macro_f1")


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    combination_mode: str
    escalation_rate: float | None = None
    feature_name: str | None = None
    threshold: float | None = None
    triggers: tuple[str, ...] = ()
    weights: dict[str, float] | None = None
    selected_validation_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sorted_pairs(rows: list[dict[str, Any]], score_fn: Callable[[dict[str, Any]], float]) -> list[tuple[float, str]]:
    return sorted(
        [(float(score_fn(row)), row_key(row)) for row in rows],
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )


def normalise_feature_values(rows: list[dict[str, Any]], feature_name: str) -> dict[str, float]:
    values = [float(row["features"][feature_name]) for row in rows]
    minimum = min(values) if values else 0.0
    maximum = max(values) if values else 0.0
    denominator = maximum - minimum
    if denominator == 0:
        return {row_key(row): 0.0 for row in rows}
    return {row_key(row): (float(row["features"][feature_name]) - minimum) / denominator for row in rows}


def add_features(rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    feature_by_key = {row["row_key"]: row["features"] for row in feature_rows}
    for row in rows:
        row["features"] = feature_by_key[row_key(row)]


def build_feature_rows(
    rows: list[dict[str, Any]],
    pair_stats,
    aspect_stats,
    sentiment_stats,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        features = prediction_features(row, pair_stats, aspect_stats, sentiment_stats)
        predicted_pairs = list(row["pred_pair_labels"])
        predicted_aspects = sorted({label.rsplit(" | ", maxsplit=1)[0] for label in predicted_pairs})
        predicted_sentiments = sorted({pair_to_components(label)[1] for label in predicted_pairs})
        for label in predicted_pairs:
            features[f"pair::{label}"] = 1.0
        for aspect in predicted_aspects:
            features[f"aspect::{aspect}"] = 1.0
        for sentiment in predicted_sentiments:
            features[f"sentiment::{sentiment}"] = 1.0
        output.append({"row_key": row_key(row), "features": features})
    return output


def pair_classes() -> list[str]:
    return candidate_pair_labels(DEFAULT_HELDOUT_ASPECTS)


def aspect_classes() -> list[str]:
    return list(DEFAULT_HELDOUT_ASPECTS)


def sentiment_classes() -> list[str]:
    return ["negative", "neutral", "positive"]


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_pairs = [row["gold_pair_labels"] for row in rows]
    pred_pairs = [row["pred_pair_labels"] for row in rows]
    pairs = pair_classes()
    true_aspects = [sorted({pair_to_aspect(label) for label in labels}) for labels in true_pairs]
    pred_aspects = [sorted({pair_to_aspect(label) for label in labels}) for labels in pred_pairs]

    output: dict[str, Any] = {}
    for prefix, scores in [
        ("pair", evaluate_label_sets_fast(true_pairs, pred_pairs, pairs)),
        ("aspect", evaluate_label_sets_fast(true_aspects, pred_aspects, aspect_classes())),
    ]:
        for key, value in scores.items():
            output[f"{prefix}_{key}"] = value

    sentiment_total = 0
    sentiment_correct = 0
    for true_row, pred_row in zip(true_pairs, pred_pairs):
        predicted_sentiments_by_aspect: dict[str, set[str]] = {}
        for label in pred_row:
            aspect, sentiment = pair_to_components(label)
            predicted_sentiments_by_aspect.setdefault(aspect, set()).add(sentiment)
        for label in true_row:
            aspect, sentiment = pair_to_components(label)
            if aspect in predicted_sentiments_by_aspect:
                sentiment_total += 1
                if sentiment in predicted_sentiments_by_aspect[aspect]:
                    sentiment_correct += 1

    output["sentiment_accuracy_when_gold_aspect_predicted"] = (
        sentiment_correct / sentiment_total if sentiment_total else 0.0
    )
    output["sentiment_correct_when_gold_aspect_predicted"] = int(sentiment_correct)
    output["sentiment_evaluated_gold_aspects"] = int(sentiment_total)
    return output


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_label_sets_fast(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
    classes: list[str],
) -> dict[str, Any]:
    rows = len(true_labels)
    label_tp = 0
    label_fp = 0
    label_fn = 0
    exact_rows = 0
    empty_gold_rows = 0
    empty_prediction_rows = 0
    empty_gold_and_prediction_rows = 0
    false_positive_rows = 0
    false_negative_rows = 0
    samples_f1_sum = 0.0
    class_f1_sum = 0.0

    true_sets = [set(labels) for labels in true_labels]
    pred_sets = [set(labels) for labels in pred_labels]
    for true_set, pred_set in zip(true_sets, pred_sets):
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        label_tp += tp
        label_fp += fp
        label_fn += fn
        denominator = len(true_set) + len(pred_set)
        samples_f1_sum += 2 * tp / denominator if denominator else 0.0
        exact_rows += int(true_set == pred_set)
        empty_gold_rows += int(not true_set)
        empty_prediction_rows += int(not pred_set)
        empty_gold_and_prediction_rows += int(not true_set and not pred_set)
        false_positive_rows += int(not true_set and bool(pred_set))
        false_negative_rows += int(bool(true_set) and not pred_set)

    for label in classes:
        tp = fp = fn = 0
        for true_set, pred_set in zip(true_sets, pred_sets):
            tp += int(label in true_set and label in pred_set)
            fp += int(label not in true_set and label in pred_set)
            fn += int(label in true_set and label not in pred_set)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        class_f1_sum += f1

    precision, recall, micro_f1 = precision_recall_f1(label_tp, label_fp, label_fn)
    return {
        "micro_f1": micro_f1,
        "micro_precision": precision,
        "micro_recall": recall,
        "macro_f1": class_f1_sum / len(classes) if classes else 0.0,
        "samples_f1": samples_f1_sum / rows if rows else 0.0,
        "label_tp": int(label_tp),
        "label_fp": int(label_fp),
        "label_fn": int(label_fn),
        "predicted_label_count": int(sum(len(labels) for labels in pred_sets)),
        "gold_label_count": int(sum(len(labels) for labels in true_sets)),
        "empty_gold_rows": int(empty_gold_rows),
        "empty_prediction_rows": int(empty_prediction_rows),
        "empty_gold_and_prediction_rows": int(empty_gold_and_prediction_rows),
        "false_positive_rows": int(false_positive_rows),
        "false_negative_rows": int(false_negative_rows),
        "false_positive_rows_per_100": false_positive_rows * 100 / rows if rows else 0.0,
        "false_positive_labels_per_100": label_fp * 100 / rows if rows else 0.0,
        "false_negative_rows_per_100": false_negative_rows * 100 / rows if rows else 0.0,
        "exact_match_rate": exact_rows / rows if rows else 0.0,
    }


def token_cost(row: dict[str, Any], input_cost_per_1m: float, output_cost_per_1m: float) -> float:
    input_tokens = row.get("input_tokens") or 0
    output_tokens = row.get("output_tokens") or 0
    return float(input_tokens) * input_cost_per_1m / 1_000_000 + float(output_tokens) * output_cost_per_1m / 1_000_000


def apply_policy(
    local_rows: list[dict[str, Any]],
    gemini_by_key: dict[str, dict[str, Any]],
    policy: Policy,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = selected_keys_for_policy(local_rows, policy)
    selected_set = set(selected)
    rows: list[dict[str, Any]] = []
    called_gemini_rows = []
    for row in local_rows:
        output_row = {key: value for key, value in row.items() if key != "features"}
        if row_key(row) in selected_set:
            gemini_row = gemini_by_key[row_key(row)]
            called_gemini_rows.append(gemini_row)
            output_row["pred_pair_labels"] = combine_predictions(
                row["pred_pair_labels"],
                gemini_row["pred_pair_labels"],
                policy.combination_mode,
            )
        rows.append(output_row)

    metrics = evaluate_rows(rows)
    seconds = [float(row["seconds"]) for row in called_gemini_rows if row.get("seconds") is not None]
    cost = sum(token_cost(row, input_cost_per_1m, output_cost_per_1m) for row in called_gemini_rows)
    metrics.update(
        {
            "gemini_call_count": len(called_gemini_rows),
            "gemini_call_rate": len(called_gemini_rows) / max(1, len(local_rows)),
            "estimated_gemini_cost_usd": cost,
            "hosted_seconds_total": sum(seconds),
            "hosted_seconds_per_dataset_row": sum(seconds) / max(1, len(local_rows)),
            "mean_latency_called_seconds": sum(seconds) / len(seconds) if seconds else 0.0,
        }
    )
    return rows, metrics


def selected_keys_for_policy(rows: list[dict[str, Any]], policy: Policy) -> list[str]:
    if policy.kind == "none":
        return []
    if policy.kind == "all":
        return [row_key(row) for row in rows]
    if policy.kind == "rank":
        if policy.escalation_rate is None:
            raise ValueError("Rank policy is missing escalation_rate.")
        score_fn = score_function(policy, rows)
        scored = sorted_pairs(rows, score_fn)
        count = min(len(rows), max(0, round(policy.escalation_rate * len(rows))))
        return [key for _, key in scored[:count]]
    if policy.kind == "threshold":
        if policy.feature_name is None or policy.threshold is None:
            raise ValueError("Threshold policy is missing feature_name or threshold.")
        return [row_key(row) for row in rows if float(row["features"].get(policy.feature_name, 0.0)) >= policy.threshold]
    if policy.kind == "trigger_any":
        return [
            row_key(row)
            for row in rows
            if any(float(row["features"].get(trigger, 0.0)) > 0.0 for trigger in policy.triggers)
        ]
    if policy.kind == "trigger_all":
        return [
            row_key(row)
            for row in rows
            if all(float(row["features"].get(trigger, 0.0)) > 0.0 for trigger in policy.triggers)
        ]
    raise ValueError(f"Unknown policy kind: {policy.kind}")


def score_function(policy: Policy, rows: list[dict[str, Any]]) -> Callable[[dict[str, Any]], float]:
    if policy.kind != "rank":
        raise ValueError("score_function is only valid for rank policies.")
    if policy.weights:
        normalised = {
            feature: normalise_feature_values(rows, feature)
            for feature in policy.weights
        }

        def weighted_score(row: dict[str, Any]) -> float:
            key = row_key(row)
            return sum(weight * normalised[feature][key] for feature, weight in policy.weights.items())

        return weighted_score
    if not policy.feature_name:
        raise ValueError("Rank policy is missing feature_name.")
    return lambda row: float(row["features"].get(str(policy.feature_name), 0.0))


def policy_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["metrics"]
    return (
        float(metrics["pair_samples_f1"]),
        float(metrics["pair_micro_f1"]),
        float(metrics["pair_macro_f1"]),
        -float(metrics["gemini_call_rate"]),
    )


def build_policy_candidates(validation_rows: list[dict[str, Any]], rank_rate_step: int) -> list[Policy]:
    if rank_rate_step < 1 or rank_rate_step > 100:
        raise ValueError("rank_rate_step must be between 1 and 100.")

    numeric_features = [
        "pred_count",
        "has_multi_prediction",
        "has_neutral_prediction",
        "has_positive_prediction",
        "low_min_pair_precision",
        "low_mean_pair_precision",
        "low_min_pair_f1",
        "low_min_aspect_precision",
        "low_min_sentiment_precision",
        "low_min_sentiment_f1",
        "low_aspect_top_score",
        "low_aspect_score_margin",
        "score_near_threshold",
        "low_aspect_selected_score_min",
        "low_aspect_selected_score_mean",
        "aspect_above_threshold_count",
    ]
    weighted_feature_sets = [
        {"pred_count": 1.0, "low_min_pair_precision": 2.0},
        {"pred_count": 1.0, "low_min_pair_f1": 2.0},
        {"low_min_pair_precision": 1.0, "low_min_aspect_precision": 1.0},
        {"low_min_pair_precision": 1.0, "low_min_sentiment_precision": 1.0},
        {"pred_count": 1.0, "low_min_pair_precision": 1.0, "low_min_sentiment_precision": 1.0},
    ]
    trigger_features = sorted(
        {
            feature
            for row in validation_rows
            for feature, value in row["features"].items()
            if value > 0.0 and feature.startswith(("pair::", "aspect::", "sentiment::"))
        }
    )
    escalation_rates = [round(value / 100, 2) for value in range(0, 101, rank_rate_step)]
    if escalation_rates[-1] != 1.0:
        escalation_rates.append(1.0)

    candidates: list[Policy] = []
    for mode in COMBINATION_MODES:
        candidates.append(Policy(name=f"none::{mode}", kind="none", combination_mode=mode))
        candidates.append(Policy(name=f"all::{mode}", kind="all", combination_mode=mode))
        for feature in numeric_features:
            for rate in escalation_rates:
                candidates.append(
                    Policy(
                        name=f"rank::{feature}::{rate:.2f}::{mode}",
                        kind="rank",
                        combination_mode=mode,
                        escalation_rate=rate,
                        feature_name=feature,
                    )
                )
            thresholds = sorted({float(row["features"].get(feature, 0.0)) for row in validation_rows})
            for threshold in thresholds:
                candidates.append(
                    Policy(
                        name=f"threshold::{feature}>={threshold:.6f}::{mode}",
                        kind="threshold",
                        combination_mode=mode,
                        feature_name=feature,
                        threshold=threshold,
                    )
                )

        for trigger_count in [1, 2]:
            for triggers in itertools.combinations(trigger_features, trigger_count):
                candidates.append(
                    Policy(
                        name=f"trigger_any::{'+'.join(triggers)}::{mode}",
                        kind="trigger_any",
                        combination_mode=mode,
                        triggers=triggers,
                    )
                )
                if trigger_count > 1:
                    candidates.append(
                        Policy(
                            name=f"trigger_all::{'+'.join(triggers)}::{mode}",
                            kind="trigger_all",
                            combination_mode=mode,
                            triggers=triggers,
                        )
                    )

        for weight_map in weighted_feature_sets:
            for rate in escalation_rates:
                candidates.append(
                    Policy(
                        name=f"rank_weighted::{json.dumps(weight_map, sort_keys=True)}::{rate:.2f}::{mode}",
                        kind="rank",
                        combination_mode=mode,
                        escalation_rate=rate,
                        weights=weight_map,
                    )
                )
    return deduplicate_policies(candidates)


def deduplicate_policies(policies: list[Policy]) -> list[Policy]:
    seen = set()
    unique = []
    for policy in policies:
        key = json.dumps(policy.to_dict(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(policy)
    return unique


def evaluate_policy_candidates(
    policies: list[Policy],
    validation_rows: list[dict[str, Any]],
    validation_gemini_by_key: dict[str, dict[str, Any]],
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> list[dict[str, Any]]:
    results = []
    for policy in policies:
        if policy.kind == "rank" and policy.weights and policy.escalation_rate is None:
            continue
        selected_count = len(selected_keys_for_policy(validation_rows, policy))
        policy = Policy(**{**policy.to_dict(), "selected_validation_rows": selected_count})
        _, metrics = apply_policy(
            validation_rows,
            validation_gemini_by_key,
            policy,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )
        results.append({"policy": policy.to_dict(), "metrics": metrics})
    results.sort(key=policy_sort_key, reverse=True)
    return results


def summarise_pareto(
    validation_results: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    test_gemini_by_key: dict[str, dict[str, Any]],
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> list[dict[str, Any]]:
    budgets = [round(value / 100, 2) for value in range(0, 101, 5)]
    summary = []
    for budget in budgets:
        eligible = [
            result for result in validation_results if float(result["metrics"]["gemini_call_rate"]) <= budget
        ]
        if not eligible:
            continue
        selected = eligible[0]
        policy = Policy(**selected["policy"])
        _, test_metrics = apply_policy(
            test_rows,
            test_gemini_by_key,
            policy,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )
        summary.append(
            {
                "budget": budget,
                "policy": policy.to_dict(),
                "validation_metrics": selected["metrics"],
                "test_metrics": test_metrics,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune a local-to-Gemini held-out-aspect cascade from cached predictions.")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--gemini-dir", type=Path, default=DEFAULT_GEMINI_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-cost-per-1m", type=float, default=0.30)
    parser.add_argument("--output-cost-per-1m", type=float, default=2.50)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--rank-rate-step",
        type=int,
        default=1,
        help="Percentage-point step for ranked escalation-rate policies.",
    )
    args = parser.parse_args()

    local_validation = read_jsonl(args.local_dir / "best_validation_predictions.jsonl")
    local_test = read_jsonl(args.local_dir / "best_test_predictions.jsonl")
    gemini_validation = read_jsonl(args.gemini_dir / "predictions_validation_indexed.jsonl")
    gemini_test = read_jsonl(args.gemini_dir / "predictions_test_indexed.jsonl")

    validation_gemini_by_key = validate_aligned_rows(local_validation, gemini_validation)
    test_gemini_by_key = validate_aligned_rows(local_test, gemini_test)

    pair_stats = label_reliability(local_validation, pair_classes(), "pair")
    aspect_stats = label_reliability(local_validation, aspect_classes(), "aspect")
    sentiment_stats = label_reliability(local_validation, sentiment_classes(), "sentiment")
    validation_feature_rows = build_feature_rows(local_validation, pair_stats, aspect_stats, sentiment_stats)
    test_feature_rows = build_feature_rows(local_test, pair_stats, aspect_stats, sentiment_stats)
    add_features(local_validation, validation_feature_rows)
    add_features(local_test, test_feature_rows)

    local_validation_metrics = evaluate_rows(local_validation)
    local_test_metrics = evaluate_rows(local_test)
    gemini_validation_metrics = evaluate_rows(gemini_validation)
    gemini_test_metrics = evaluate_rows(gemini_test)

    policies = build_policy_candidates(local_validation, rank_rate_step=args.rank_rate_step)
    validation_results = evaluate_policy_candidates(
        policies,
        local_validation,
        validation_gemini_by_key,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )
    selected = validation_results[0]
    selected_policy = Policy(**selected["policy"])
    selected_test_rows, selected_test_metrics = apply_policy(
        local_test,
        test_gemini_by_key,
        selected_policy,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )
    pareto = summarise_pareto(
        validation_results,
        local_test,
        test_gemini_by_key,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )

    summary = {
        "task": "Local-to-Gemini fixed held-out-aspect cascade",
        "local_predictions": str(args.local_dir),
        "gemini_predictions": str(args.gemini_dir),
        "heldout_aspects": list(DEFAULT_HELDOUT_ASPECTS),
        "pair_classes": pair_classes(),
        "cost_rates": {
            "input_per_1m": args.input_cost_per_1m,
            "output_per_1m": args.output_cost_per_1m,
        },
        "rank_rate_step": args.rank_rate_step,
        "candidate_policy_count": len(policies),
        "baselines": {
            "local_validation": local_validation_metrics,
            "gemini_validation": gemini_validation_metrics,
            "local_test": local_test_metrics,
            "gemini_test": gemini_test_metrics,
        },
        "label_reliability": {
            "pair": {label: stats.to_dict() for label, stats in pair_stats.items()},
            "aspect": {label: stats.to_dict() for label, stats in aspect_stats.items()},
            "sentiment": {label: stats.to_dict() for label, stats in sentiment_stats.items()},
        },
        "feature_names": sorted(
            {
                feature
                for row in validation_feature_rows
                for feature in row["features"]
            }
        ),
        "selected_by_validation": {
            "policy": selected_policy.to_dict(),
            "validation_metrics": selected["metrics"],
            "test_metrics": selected_test_metrics,
        },
        "validation_top_policies": validation_results[: args.top_k],
        "pareto_by_validation_budget": pareto,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.output_dir / "summary.json")
    write_jsonl(selected_test_rows, args.output_dir / "selected_test_predictions.jsonl")
    compact_summary = {
        "output_dir": str(args.output_dir),
        "candidate_policy_count": len(policies),
        "selected_policy": selected_policy.name,
        "selected_validation_pair_samples_f1": selected["metrics"]["pair_samples_f1"],
        "selected_test_pair_samples_f1": selected_test_metrics["pair_samples_f1"],
        "selected_test_pair_micro_f1": selected_test_metrics["pair_micro_f1"],
        "selected_test_pair_macro_f1": selected_test_metrics["pair_macro_f1"],
        "selected_test_gemini_call_rate": selected_test_metrics["gemini_call_rate"],
        "selected_test_estimated_gemini_cost_usd": selected_test_metrics["estimated_gemini_cost_usd"],
    }
    print(json.dumps(compact_summary, indent=2, ensure_ascii=False))
    print(f"Saved local-to-Gemini cascade analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
