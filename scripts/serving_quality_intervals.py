"""Descriptive paired review-cluster bootstrap for frozen serving replay quality.

No inference or training occurs. Repeated HTTP attempts are deduplicated using
monitoring's first-success-by-start-time rule. This validation replay is not a
new blind model test; percentile intervals are descriptive, not a significance
claim. Outputs contain aggregates only, never review IDs, text or input paths.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_final_test import canonical_sha256
from msc_project.serving.monitoring import evaluate_quality, label_pairs, prediction_pairs

SCOPES = ("overall", "seen", "heldout")
METRICS = ("presence", "pair")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def rng_for(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def deduplicate(records: list[dict]) -> tuple[set[str], dict[str, set]]:
    attempted, first = set(), {}
    # Keep exactly the monitoring module's ordering and first-success rule.
    for record in sorted(records, key=lambda r: (r.get("started_s") is None,
                                                r.get("started_s") or 0)):
        key = str(record.get("sample_id", record.get("request_id")))
        attempted.add(key)
        if record.get("success") and key not in first:
            first[key] = prediction_pairs(record.get("response", {}))
    return attempted, first


def count_tensor(keys: list[str], predictions: dict, gold: dict, scopes: dict) -> tuple:
    """One whole-review row: scope x metric x (TP, FP, FN)."""
    counts = np.zeros((len(keys), len(SCOPES), len(METRICS), 3), dtype=np.int64)
    positives = np.zeros((len(keys), len(SCOPES)), dtype=np.int64)
    for i, key in enumerate(keys):
        for j, scope_name in enumerate(SCOPES):
            scope = scopes[scope_name]
            pred_pairs = {(a, s) for a, s in predictions.get(key, set()) if a in scope}
            gold_pairs = {(a, s) for a, s in gold[key] if a in scope}
            positives[i, j] = bool(gold_pairs)
            sets = (({a for a, _ in pred_pairs}, {a for a, _ in gold_pairs}),
                    (pred_pairs, gold_pairs))
            for k, (pred, target) in enumerate(sets):
                counts[i, j, k] = (len(pred & target), len(pred - target), len(target - pred))
    return counts, positives


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    return np.divide(2.0 * tp, denominator, out=np.zeros(tp.shape, dtype=float),
                     where=denominator != 0)


def bootstrap_weights(n: int, replicates: int, rng: np.random.Generator) -> np.ndarray:
    # Repeated occurrences in this matrix are bootstrap draws of entire reviews,
    # not independent aspect candidates and not repeated HTTP attempts.
    indices = rng.integers(0, n, size=(replicates, n))
    weights = np.zeros((replicates, n), dtype=np.int32)
    np.add.at(weights, (np.arange(replicates)[:, None], indices), 1)
    return weights


def sample_totals(weights: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return (weights @ counts.reshape(len(counts), -1)).reshape(len(weights), 3, 2, 3)


def summarize_counts(counts: np.ndarray, positives: np.ndarray, replicates: int,
                     rng: np.random.Generator) -> dict:
    totals = counts.sum(axis=0)
    estimates = f1_from_counts(totals)
    sampled = (sample_totals(bootstrap_weights(len(counts), replicates, rng), counts)
               if len(counts) else None)
    distribution = f1_from_counts(sampled) if sampled is not None else None
    output = {"unique_reviews": len(counts), "scopes": {}}
    for j, scope in enumerate(SCOPES):
        entry = {"positive_review_count": int(positives[:, j].sum())}
        for k, metric in enumerate(METRICS):
            tp, fp, fn = (int(v) for v in totals[j, k])
            entry[metric] = {
                "micro_f1": float(estimates[j, k]),
                "percentile_95_interval": np.quantile(distribution[:, j, k], [.025, .975]).tolist()
                if distribution is not None else None,
                "tp": tp, "fp": fp, "fn": fn, "gold_support": tp + fn,
                "bootstrap_zero_denominator_replicates":
                    int(np.sum(2 * sampled[:, j, k, 0] + sampled[:, j, k, 1] + sampled[:, j, k, 2] == 0))
                    if sampled is not None else None,
            }
        output["scopes"][scope] = entry
    return output


def paired_difference(left: np.ndarray, right: np.ndarray, positives: np.ndarray,
                      replicates: int, rng: np.random.Generator) -> dict:
    output = {"unique_reviews": len(left), "difference_direction": "left_minus_right",
              "scopes": {}}
    if len(left) != len(right):
        raise ValueError("Paired systems must have the same review clusters")
    if len(left):
        weights = bootstrap_weights(len(left), replicates, rng)
        left_boot = f1_from_counts(sample_totals(weights, left))
        right_boot = f1_from_counts(sample_totals(weights, right))
        differences = left_boot - right_boot
        left_point, right_point = f1_from_counts(left.sum(axis=0)), f1_from_counts(right.sum(axis=0))
    for j, scope in enumerate(SCOPES):
        entry = {"positive_review_count": int(positives[:, j].sum())}
        for k, metric in enumerate(METRICS):
            entry[metric] = {
                "left_micro_f1": float(left_point[j, k]) if len(left) else None,
                "right_micro_f1": float(right_point[j, k]) if len(left) else None,
                "difference": float(left_point[j, k] - right_point[j, k]) if len(left) else None,
                "percentile_95_interval": np.quantile(differences[:, j, k], [.025, .975]).tolist()
                if len(left) else None,
            }
        output["scopes"][scope] = entry
    return output


def analyse(runs: dict[str, list[dict]], labels: list[dict], artifacts: dict,
            *, replicates: int = 2000, seed: int = 20261109) -> dict:
    if replicates < 1 or not runs:
        raise ValueError("At least one run and one bootstrap replicate are required")
    expected_fold = artifacts["fold_id"]
    taxonomy = artifacts["taxonomy"]
    universe = set(taxonomy["canonical_order"])
    heldout = set(taxonomy["heldout_aspects"])
    if not heldout or not heldout < universe:
        raise ValueError("Manifest must define nonempty held-out and seen scopes")
    expected = artifacts["replay"]["splits"]["evaluation"]
    label_ids = [str(row["request_id"]) for row in labels]
    if len(label_ids) != expected["count"] or len(set(label_ids)) != len(label_ids):
        raise ValueError("Labels do not cover the exact frozen evaluation set")
    if canonical_sha256(label_ids) != expected["request_id_sha256"]:
        raise ValueError("Label identities/order do not match frozen evaluation replay")
    gold = {str(row["request_id"]): label_pairs(row) for row in labels}
    valid_sentiments = {"negative", "neutral", "positive"}
    if any(a not in universe or s not in valid_sentiments for pairs in gold.values() for a, s in pairs):
        raise ValueError("Labels contain values outside the frozen taxonomy/sentiments")
    scopes = {"overall": universe, "seen": universe - heldout, "heldout": heldout}
    normalized, report, shared_attempted = {}, {}, None
    for name, records in sorted(runs.items()):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("Run names must be simple public labels, not paths")
        if not records:
            raise ValueError(f"Run {name} has no attempted reviews")
        quality = evaluate_quality(records, labels, heldout, universe)
        # Do not silently assume the supplied manifest describes responses whose
        # fold identity is absent. At least one successful response is needed.
        identities = [r.get("response", {}) for r in records if r.get("success")]
        if not identities or any(r.get("fold_id") != expected_fold for r in identities):
            raise ValueError(f"Run {name} lacks verified successful responses for the frozen fold")
        attempted, first = deduplicate(records)
        if any(s not in valid_sentiments for pairs in first.values() for _, s in pairs):
            raise ValueError(f"Run {name} predicts an unknown sentiment")
        if shared_attempted is None:
            shared_attempted = attempted
        elif attempted != shared_attempted:
            raise ValueError("Run attempted-review sets differ; paired comparison refuses implicit intersection")
        normalized[name] = first
        views = {}
        for view, keys in (("failure_as_empty", sorted(attempted)),
                           ("successful_only", sorted(first))):
            counts, positives = count_tensor(keys, first, gold, scopes)
            views[view] = summarize_counts(counts, positives, replicates, rng_for(seed, name + ":" + view))
            # Verify exact agreement with the existing monitoring point estimates.
            original = quality["delivered_quality_failure_as_empty" if view == "failure_as_empty"
                               else "successful_unique_quality"]
            for scope in SCOPES:
                for metric in METRICS:
                    if not np.isclose(views[view]["scopes"][scope][metric]["micro_f1"],
                                      original[scope][metric]["micro_f1"], rtol=0, atol=1e-15):
                        raise AssertionError("Bootstrap point estimate differs from monitoring")
        report[name] = {
            "variant": quality["variant"], "attempts": len(records),
            "successful_attempts": sum(r.get("success") is True for r in records),
            "unique_attempted_reviews": len(attempted), "unique_successful_reviews": len(first),
            "unique_reviews_without_success": len(attempted - first.keys()),
            "successful_repeats_excluded": quality["successful_attempts_omitted_as_repeats"],
            "repeat_prediction_disagreements": quality["repeated_success_prediction_disagreements"],
            "views": views,
        }
    comparisons = []
    for left, right in itertools.combinations(sorted(normalized), 2):
        result = {"left": left, "right": right, "views": {}}
        common_success = set(normalized[left]) & set(normalized[right])
        for view, keys in (("failure_as_empty", sorted(shared_attempted)),
                           ("successful_only_common_subset", sorted(common_success))):
            lc, positive = count_tensor(keys, normalized[left], gold, scopes)
            rc, _ = count_tensor(keys, normalized[right], gold, scopes)
            result["views"][view] = paired_difference(
                lc, rc, positive, replicates, rng_for(seed, left + "-" + right + ":" + view))
        comparisons.append(result)
    return {
        "schema_version": "chattermill_serving_quality_intervals_v1",
        "fold_id": expected_fold, "f1_scale": "0-1", "seed": seed,
        "bootstrap_replicates": replicates, "interval_method": "review-cluster percentile 2.5/97.5",
        "coverage": {"frozen_evaluation_reviews": len(labels),
                     "common_unique_attempted_reviews": len(shared_attempted),
                     "unattempted_evaluation_reviews": len(labels) - len(shared_attempted),
                     "complete_evaluation_coverage": len(shared_attempted) == len(labels)},
        "semantics": {
            "cluster": "Whole unique review, preserving all aspects and sentiments together",
            "deduplication": "First successful attempt ordered by start time, matching monitoring; no HTTP repeat is a new sample",
            "failure_as_empty": "All common attempted unique reviews; never-successful samples have empty predictions; retries may rescue earlier failures",
            "successful_only": "Per-system intervals use that system's successful unique samples",
            "successful_only_difference": "Paired differences use the two systems' common successful samples; these conditional point estimates can differ from each system's own successful-subset result",
            "zero_denominator": "Micro-F1 is zero, matching monitoring; zero-denominator bootstrap counts are reported",
            "limitations": "Existing validation engineering replay, not a new blind test or the historical 12-fold mean. Intervals are descriptive; no p-values, significance claim or multiplicity correction. Partial coverage is not extrapolated to all evaluation reviews.",
        },
        "systems": report, "paired_differences": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8-sig"))
    expected_label_hash = artifacts["replay"]["splits"]["evaluation"]["files"]["labels"]["sha256"]
    if hashlib.sha256(args.labels.read_bytes()).hexdigest() != expected_label_hash:
        parser.error("--labels content hash differs from the frozen evaluation label file")
    runs = {}
    for entry in args.run:
        if "=" not in entry:
            parser.error("Each --run must be NAME=DIR")
        name, directory = entry.split("=", 1)
        if name in runs:
            parser.error("Run names must be unique")
        runs[name] = read_jsonl(Path(directory) / "requests.jsonl")
    result = analyse(runs, read_jsonl(args.labels), artifacts,
                     replicates=args.replicates, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"systems": len(runs), "replicates": args.replicates,
                      "coverage": result["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
