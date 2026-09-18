from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable

from msc_project.evaluation.metrics import PAIR_SEPARATOR, pair_to_aspect, pair_to_components


@dataclass(frozen=True)
class LabelReliability:
    label: str
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def support(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def predicted(self) -> int:
        return self.true_positive + self.false_positive

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output.update(
            {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "support": self.support,
                "predicted": self.predicted,
            }
        )
        return output


def row_key(row: dict[str, Any]) -> str:
    row_uid = row.get("row_uid")
    if row_uid:
        return str(row_uid)
    return f"{row.get('original_split', '')}:{row.get('id', '')}"


def sample_f1(gold_labels: Iterable[str], pred_labels: Iterable[str]) -> float:
    gold = set(gold_labels)
    predicted = set(pred_labels)
    denominator = len(gold) + len(predicted)
    return 2 * len(gold & predicted) / denominator if denominator else 0.0


def label_reliability(
    rows: list[dict[str, Any]],
    classes: list[str],
    label_type: str = "pair",
) -> dict[str, LabelReliability]:
    if label_type not in {"pair", "aspect", "sentiment"}:
        raise ValueError(f"Unknown label type: {label_type}")

    counters: dict[str, Counter[str]] = {
        "tp": Counter(),
        "fp": Counter(),
        "fn": Counter(),
    }
    for row in rows:
        gold_pairs = set(row["gold_pair_labels"])
        pred_pairs = set(row["pred_pair_labels"])
        if label_type == "pair":
            gold_values = gold_pairs
            pred_values = pred_pairs
        elif label_type == "aspect":
            gold_values = {pair_to_aspect(label) for label in gold_pairs}
            pred_values = {pair_to_aspect(label) for label in pred_pairs}
        else:
            gold_values = {pair_to_components(label)[1] for label in gold_pairs}
            pred_values = {pair_to_components(label)[1] for label in pred_pairs}

        for label in classes:
            if label in gold_values and label in pred_values:
                counters["tp"][label] += 1
            if label not in gold_values and label in pred_values:
                counters["fp"][label] += 1
            if label in gold_values and label not in pred_values:
                counters["fn"][label] += 1

    return {
        label: LabelReliability(
            label=label,
            true_positive=int(counters["tp"][label]),
            false_positive=int(counters["fp"][label]),
            false_negative=int(counters["fn"][label]),
        )
        for label in classes
    }


def safe_mean(values: list[float], default: float = 0.0) -> float:
    return float(mean(values)) if values else default


def prediction_features(
    row: dict[str, Any],
    pair_stats: dict[str, LabelReliability],
    aspect_stats: dict[str, LabelReliability],
    sentiment_stats: dict[str, LabelReliability],
) -> dict[str, float]:
    predicted_pairs = list(row["pred_pair_labels"])
    predicted_aspects = sorted({pair_to_aspect(label) for label in predicted_pairs})
    predicted_sentiments = sorted({pair_to_components(label)[1] for label in predicted_pairs})

    pair_precision = [pair_stats[label].precision for label in predicted_pairs if label in pair_stats]
    pair_f1 = [pair_stats[label].f1 for label in predicted_pairs if label in pair_stats]
    aspect_precision = [aspect_stats[label].precision for label in predicted_aspects if label in aspect_stats]
    aspect_f1 = [aspect_stats[label].f1 for label in predicted_aspects if label in aspect_stats]
    sentiment_precision = [sentiment_stats[label].precision for label in predicted_sentiments if label in sentiment_stats]
    sentiment_f1 = [sentiment_stats[label].f1 for label in predicted_sentiments if label in sentiment_stats]

    features = {
        "pred_count": float(len(predicted_pairs)),
        "aspect_count": float(len(predicted_aspects)),
        "sentiment_count": float(len(predicted_sentiments)),
        "has_multi_prediction": float(len(predicted_pairs) > 1),
        "has_neutral_prediction": float("neutral" in predicted_sentiments),
        "has_positive_prediction": float("positive" in predicted_sentiments),
        "has_negative_prediction": float("negative" in predicted_sentiments),
        "low_min_pair_precision": 1.0 - min(pair_precision, default=0.0),
        "low_mean_pair_precision": 1.0 - safe_mean(pair_precision),
        "low_min_pair_f1": 1.0 - min(pair_f1, default=0.0),
        "low_mean_pair_f1": 1.0 - safe_mean(pair_f1),
        "low_min_aspect_precision": 1.0 - min(aspect_precision, default=0.0),
        "low_mean_aspect_precision": 1.0 - safe_mean(aspect_precision),
        "low_min_aspect_f1": 1.0 - min(aspect_f1, default=0.0),
        "low_mean_aspect_f1": 1.0 - safe_mean(aspect_f1),
        "low_min_sentiment_precision": 1.0 - min(sentiment_precision, default=0.0),
        "low_mean_sentiment_precision": 1.0 - safe_mean(sentiment_precision),
        "low_min_sentiment_f1": 1.0 - min(sentiment_f1, default=0.0),
        "low_mean_sentiment_f1": 1.0 - safe_mean(sentiment_f1),
    }
    features.update(score_margin_features(row.get("score_features")))
    return features


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_margin_features(score_features: Any) -> dict[str, float]:
    if not isinstance(score_features, dict):
        return {}

    output: dict[str, float] = {}
    numeric_map = {
        "top_score": "aspect_top_score",
        "second_score": "aspect_second_score",
        "score_margin": "aspect_score_margin",
        "min_abs_distance_to_threshold": "aspect_min_abs_distance_to_threshold",
        "top_distance_to_threshold": "aspect_top_distance_to_threshold",
        "above_threshold_count": "aspect_above_threshold_count",
        "selected_count": "aspect_selected_count",
        "selected_score_min": "aspect_selected_score_min",
        "selected_score_mean": "aspect_selected_score_mean",
    }
    for source, target in numeric_map.items():
        value = score_features.get(source)
        if value is None:
            continue
        try:
            output[target] = float(value)
        except (TypeError, ValueError):
            continue

    if "aspect_top_score" in output:
        output["low_aspect_top_score"] = 1.0 - clamp_unit(output["aspect_top_score"])
    if "aspect_score_margin" in output:
        output["low_aspect_score_margin"] = 1.0 - clamp_unit(abs(output["aspect_score_margin"]))
    if "aspect_min_abs_distance_to_threshold" in output:
        output["score_near_threshold"] = 1.0 - clamp_unit(abs(output["aspect_min_abs_distance_to_threshold"]))
    if "aspect_selected_score_min" in output:
        output["low_aspect_selected_score_min"] = 1.0 - clamp_unit(output["aspect_selected_score_min"])
    if "aspect_selected_score_mean" in output:
        output["low_aspect_selected_score_mean"] = 1.0 - clamp_unit(output["aspect_selected_score_mean"])
    return output


def combine_predictions(local_labels: list[str], gemini_labels: list[str], mode: str) -> list[str]:
    local = set(local_labels)
    gemini = set(gemini_labels)
    if mode == "replace":
        output = gemini
    elif mode == "gemini_nonempty_else_local":
        output = gemini if gemini else local
    elif mode == "union":
        output = local | gemini
    elif mode == "intersection":
        output = local & gemini
    elif mode == "agreement_or_gemini":
        agreement = local & gemini
        output = agreement if agreement else gemini
    elif mode == "agreement_or_local":
        agreement = local & gemini
        output = agreement if agreement else local
    else:
        raise ValueError(f"Unknown cascade combination mode: {mode}")
    return sorted(output)


def validate_aligned_rows(
    local_rows: list[dict[str, Any]],
    gemini_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    gemini_by_key = {row_key(row): row for row in gemini_rows}
    missing = [row_key(row) for row in local_rows if row_key(row) not in gemini_by_key]
    if missing:
        raise ValueError(f"Gemini predictions are missing {len(missing)} rows; first missing row: {missing[0]}")

    for local_row in local_rows:
        gemini_row = gemini_by_key[row_key(local_row)]
        if sorted(local_row["gold_pair_labels"]) != sorted(gemini_row["gold_pair_labels"]):
            raise ValueError(f"Gold labels differ for row {row_key(local_row)}")
    return gemini_by_key
