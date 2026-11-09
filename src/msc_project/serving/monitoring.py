"""Review-level service summaries and delayed-label replay monitoring.

No model inference or target-label access is needed to collect service records.
Quality joins are an explicit later operation on independent label records.
All F1 values in this module use the 0--1 scale.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable


def percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not data:
        return None
    position = (len(data) - 1) * q / 100
    lo, hi = math.floor(position), math.ceil(position)
    return data[lo] + (data[hi] - data[lo]) * (position - lo)


def latency_summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "count": len(data),
        "mean_ms": sum(data) / len(data) if data else None,
        "p50_ms": percentile(data, 50),
        "p95_ms": percentile(data, 95),
        "p99_ms": percentile(data, 99),
        "max_ms": max(data) if data else None,
    }


def summarise_load(records: list[dict], wall_seconds: float | None = None) -> dict:
    """Count all attempts, including failures and unsent client-capacity drops."""
    if wall_seconds is None:
        wall_seconds = max((float(r.get("finished_s", 0)) for r in records), default=0)
    successful = [r for r in records if r.get("success") is True]
    sent = [r for r in records if r.get("started_s") is not None]
    status_counts = Counter(str(r.get("status_code")) for r in records)
    errors = Counter(r.get("error") or "none" for r in records if not r.get("success"))
    attempts = len(records)
    per_length = {}
    for value in sorted({str(r.get("length_bin", "unknown")) for r in records}):
        subset = [r for r in records if str(r.get("length_bin", "unknown")) == value]
        per_length[value] = {
            "attempts": len(subset),
            "successful": sum(r.get("success") is True for r in subset),
            "successful_latency": latency_summary(
                r["latency_ms"] for r in subset if r.get("success")
            ),
        }
    budget = {}
    for seconds in [1, 2, 5, 10, 30, 60]:
        met = sum(r.get("success") and r.get("scheduled_latency_ms", math.inf) <= seconds * 1000
                  for r in records)
        budget[str(seconds)] = {"met": met, "fraction_of_all_attempts": met / attempts if attempts else None}
    return {
        "attempts": attempts,
        "sent": len(sent),
        "successful": len(successful),
        "failed": attempts - len(successful),
        "success_rate": len(successful) / attempts if attempts else None,
        "error_rate": (attempts - len(successful)) / attempts if attempts else None,
        "wall_seconds_including_drain": wall_seconds,
        "successful_reviews_per_second": len(successful) / wall_seconds if wall_seconds > 0 else None,
        "sent_reviews_per_second": len(sent) / wall_seconds if wall_seconds > 0 else None,
        "all_attempt_latency": latency_summary(r["latency_ms"] for r in sent),
        "successful_latency": latency_summary(r["latency_ms"] for r in successful),
        "successful_latency_from_scheduled_arrival": latency_summary(
            r["scheduled_latency_ms"] for r in successful
        ),
        "schedule_lag": latency_summary(r["schedule_lag_ms"] for r in sent),
        "http_status_counts": dict(status_counts),
        "failure_counts": dict(errors),
        "response_budgets_seconds": budget,
        "length_groups": per_length,
        "latency_scope": "Client HTTP request to complete JSON response; successful percentiles exclude failures, which remain in the attempt denominator. Scheduled latency includes client scheduling lag.",
    }


def prediction_pairs(response: dict) -> set[tuple[str, str]]:
    out = set()
    for item in response.get("predictions", response.get("pairs", [])):
        aspect = item.get("aspect", item.get("aspect_id"))
        sentiments = item.get("sentiments")
        if sentiments is None:
            sentiments = [item["sentiment"]] if "sentiment" in item else []
        for sentiment in sentiments:
            if isinstance(sentiment, dict):
                sentiment = sentiment.get("sentiment", sentiment.get("label"))
            if aspect is not None and sentiment is not None:
                out.add((str(aspect), str(sentiment)))
    return out


def label_pairs(label: dict) -> set[tuple[str, str]]:
    return prediction_pairs({"pairs": label.get("pairs", label.get("labels", []))})


def _counts(predicted: set, gold: set) -> Counter:
    return Counter(tp=len(predicted & gold), fp=len(predicted - gold), fn=len(gold - predicted))


def _prf(counts: Counter) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def evaluate_quality(records: list[dict], labels: list[dict], heldout_aspects: Iterable[str] = (),
                     all_aspects: Iterable[str] = ()) -> dict:
    """Join unique replay samples after inference, and keep service failures visible.

    The first successful response for a sample is scored once. Samples attempted
    but never successful count as empty output in a second delivered-quality view.
    Model variants/folds must be separated by the caller.
    """
    label_map = {str(r["request_id"]): r for r in labels}
    if len(label_map) != len(labels):
        raise ValueError("Duplicate ground-truth request_id")
    variants = {r.get("response", {}).get("variant") for r in records if r.get("success")}
    variants.discard(None)
    versions = {r.get("response", {}).get("model_version") for r in records if r.get("success")}
    versions.discard(None)
    folds = {r.get("response", {}).get("fold_id") for r in records if r.get("success")}
    folds.discard(None)
    if len(variants) > 1 or len(versions) > 1 or len(folds) > 1:
        raise ValueError("Quality must be grouped by model version and variant")
    first = {}
    attempted = set()
    duplicate_successes = 0
    disagreements = 0
    for record in sorted(records, key=lambda r: (r.get("started_s") is None, r.get("started_s") or 0)):
        key = str(record.get("sample_id", record.get("request_id")))
        attempted.add(key)
        if record.get("success"):
            pairs = prediction_pairs(record.get("response", {}))
            if key in first:
                duplicate_successes += 1
                disagreements += pairs != first[key]
            else:
                first[key] = pairs
    unknown = attempted - label_map.keys()
    if unknown:
        raise ValueError(f"No independent labels for {len(unknown)} attempted samples")
    heldout = set(heldout_aspects)
    supplied_universe = set(all_aspects)
    predicted_aspects = {a for pairs in first.values() for a, _ in pairs}
    universe = supplied_universe or ({a for row in labels for a, _ in label_pairs(row)} | predicted_aspects)
    if supplied_universe and predicted_aspects - supplied_universe:
        raise ValueError("Prediction contains an aspect outside the frozen taxonomy")
    scopes = {"overall": universe, "heldout": heldout, "seen": universe - heldout}

    def score(keys: set[str]) -> dict:
        output = {}
        for name, scope in scopes.items():
            pc, ac = Counter(), Counter()
            exact = 0
            positive_reviews = 0
            for key in keys:
                pred = {(a, s) for a, s in first.get(key, set()) if a in scope}
                gold = {(a, s) for a, s in label_pairs(label_map[key]) if a in scope}
                pc.update(_counts(pred, gold))
                ac.update(_counts({a for a, _ in pred}, {a for a, _ in gold}))
                exact += pred == gold
                positive_reviews += bool(gold)
            output[name] = {"unique_reviews": len(keys), "pair": _prf(pc), "presence": _prf(ac),
                            "gold_presence_support": ac["tp"] + ac["fn"],
                            "gold_pair_support": pc["tp"] + pc["fn"],
                            "positive_review_count": positive_reviews,
                            "pair_set_exact_match": exact / len(keys) if keys else None}
        return output

    return {
        "label_source": "Independent delayed label join; never included in API requests",
        "f1_scale": "0-1",
        "variant": next(iter(variants), None),
        "fold_id": next(iter(folds), None),
        "unique_attempted_reviews": len(attempted),
        "unique_successful_reviews": len(first),
        "unique_reviews_without_success": len(attempted - first.keys()),
        "successful_attempts_omitted_as_repeats": duplicate_successes,
        "repeated_success_prediction_disagreements": disagreements,
        "successful_unique_quality": score(set(first)),
        "delivered_quality_failure_as_empty": score(attempted),
        "delivered_quality_interpretation": "Eventual unique-sample delivery: use the first successful attempt, or empty output if every attempt failed. Repeated successes can rescue earlier failed attempts; always report per-attempt HTTP success rate alongside this view.",
        "unattempted_labelled_reviews": len(label_map.keys() - attempted),
        "metric_scope": "Presence and aspect-sentiment pair micro-F1 for this single frozen model version. Not the historical 12-fold average. Repeated requests do not increase unique sample size.",
    }


def js_divergence(reference: dict[str, int], current: dict[str, int]) -> float | None:
    """Jensen-Shannon divergence in bits, 0--1, no statistical significance claim."""
    if not sum(reference.values()) or not sum(current.values()):
        return None
    keys = reference.keys() | current.keys()
    p = {k: reference.get(k, 0) / sum(reference.values()) for k in keys}
    q = {k: current.get(k, 0) / sum(current.values()) for k in keys}
    result = 0.0
    for key in keys:
        mid = (p[key] + q[key]) / 2
        for value in [p[key], q[key]]:
            if value:
                result += 0.5 * value * math.log2(value / mid)
    return result


def unlabelled_summary(records: list[dict]) -> dict:
    aspects, sentiments, sizes = Counter(), Counter(), Counter()
    score_bins = Counter()
    invalid_scores = 0
    successful = [r for r in records if r.get("success")]
    for row in records:
        sizes[str(row.get("length_bin", "unknown"))] += 1
    zero = 0
    for row in successful:
        response = row.get("response", {})
        pairs = prediction_pairs(response)
        zero += not pairs
        aspects.update({a for a, _ in pairs})
        sentiments.update(s for _, s in pairs)
        for item in response.get("aspect_scores", []):
            score = item.get("score")
            if isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1:
                score_bins[str(min(9, int(score * 10)))] += 1
            else:
                invalid_scores += 1
    length_summary = latency_summary(r.get("input_chars", 0) for r in records)
    length_summary = {key.removesuffix("_ms"): value for key, value in length_summary.items()}
    return {
        "attempts": len(records), "successful": len(successful),
        "input_character_length": length_summary,
        "length_bin_counts": dict(sizes), "predicted_aspect_counts": dict(aspects),
        "predicted_sentiment_pair_counts": dict(sentiments),
        "aspect_score_bin_counts": dict(score_bins),
        "aspect_score_bins": "Ten equal-width bins in [0,1]; final bin includes 1. All supplied candidates on successful requests. Uncalibrated model scores, not correctness probabilities.",
        "invalid_aspect_scores": invalid_scores,
        "no_aspect_fraction_of_successes": zero / len(successful) if successful else None,
        "warning": "Distribution changes are monitoring signals, not estimates of accuracy or F1. Model scores are not calibrated correctness probabilities.",
    }


def monitor_windows(records: list[dict], reference: list[dict], window_seconds: float = 60,
                    length_js_threshold: float = 0.1) -> dict:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    ref = unlabelled_summary(reference)
    groups = {}
    for row in records:
        slot = int(float(row.get("scheduled_s", 0)) // window_seconds)
        groups.setdefault(slot, []).append(row)
    windows = []
    for slot, rows in sorted(groups.items()):
        now = unlabelled_summary(rows)
        js = js_divergence(ref["length_bin_counts"], now["length_bin_counts"])
        aspect_js = js_divergence(ref["predicted_aspect_counts"], now["predicted_aspect_counts"])
        score_js = js_divergence(ref["aspect_score_bin_counts"], now["aspect_score_bin_counts"])
        windows.append({
            "window_start_s": slot * window_seconds, "window_end_s": (slot + 1) * window_seconds,
            "unlabelled": now,
            "successful_latency": latency_summary(r["latency_ms"] for r in rows if r.get("success")),
            "failure_fraction": sum(not r.get("success") for r in rows) / len(rows),
            "input_length_js_bits": js, "prediction_aspect_js_bits": aspect_js,
            "aspect_score_js_bits": score_js,
            "input_length_shift_signal": js is not None and js > length_js_threshold,
        })
    return {"window_assignment": "Scheduled-arrival cohort, not completion-time cohort",
            "window_seconds": window_seconds, "length_js_signal_threshold": length_js_threshold,
            "reference": ref, "windows": windows,
            "interpretation": "Threshold is a declared descriptive signal, not a significance test. Small windows may fluctuate; confirm quality changes with labels."}


def make_review_queues(records: list[dict], random_count: int = 20, targeted_count: int = 20,
                       seed: int = 20261109) -> dict:
    """Keep random auditing separate from targeted slow/empty-output inspection."""
    import random
    unique = {}
    for row in records:
        key = str(row.get("sample_id", row.get("request_id")))
        if row.get("success") and key not in unique:
            unique[key] = row
    keys = sorted(unique)
    chosen = random.Random(seed).sample(keys, min(random_count, len(keys)))
    remaining = [key for key in keys if key not in chosen]
    remaining.sort(key=lambda key: (not prediction_pairs(unique[key].get("response", {})),
                                   unique[key].get("latency_ms", 0)), reverse=True)
    return {
        "seed": seed,
        "random_audit_sample_ids": chosen,
        "targeted_inspection_sample_ids": remaining[:targeted_count],
        "targeted_rule": "Empty prediction first, then highest successful client latency; excludes random-audit samples",
        "interpretation": "The random audit samples successful unique replay samples only and estimates quality within that returned-output population, not delivery quality including failed or unattempted requests. The targeted queue diagnoses selected cases and is not an unbiased population estimate. Labels are joined later.",
    }
