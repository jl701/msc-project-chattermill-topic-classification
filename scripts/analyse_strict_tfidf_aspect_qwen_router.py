from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from itertools import product
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from scripts.rescore_similarity_loao_with_aspect_sentiment import (
    align_prediction_rows,
    read_jsonl,
)


EXPERIMENT_ID = "loao_strict_tfidf_aspect_qwen_router_v1"
LOCAL_METHOD = "tfidf_char_3_5_train_vocab"
SENTIMENTS = ("negative", "neutral", "positive")
AGGREGATE_METRICS = (
    "pair_micro_f1",
    "pair_samples_f1",
    "pair_micro_precision",
    "pair_micro_recall",
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "pair_false_positive_rows_per_100",
    "pair_false_negative_rows_per_100",
    "sentiment_accuracy_when_gold_aspect_predicted",
    "qwen_call_rate",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(config: dict[str, Any]) -> list[float]:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Router config has the wrong experiment_id.")
    if config.get("status") != "preregistered_before_router_execution":
        raise ValueError("Router config is not marked as preregistered.")
    scope = config.get("scope", {})
    if (
        scope.get("folds") != 12
        or scope.get("information_regime") != "target-calibrated LOAO"
    ):
        raise ValueError("Router config does not declare the locked 12-fold regime.")
    artifacts = config.get("artifact_policy", {})
    if (
        artifacts.get("new_model_training") is not False
        or artifacts.get("new_model_inference") is not False
        or artifacts.get("legacy_router_artifacts_permitted") is not False
        or artifacts.get("historical_json_prompt_qwen_permitted") is not False
    ):
        raise ValueError("Router config permits an unregistered artifact path.")
    fractions = [float(value) for value in config["policy_search"]["fraction_grid"]]
    if fractions != [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        raise ValueError("Router fraction grid differs from the preregistered grid.")
    return fractions


def validate_similarity_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        payload.get("experiment_id")
        != "loao_similarity_aspect_conditioned_sentiment_rescore_v1"
        or payload.get("stage") != "validation"
        or payload.get("protocol_complete") is not True
    ):
        raise ValueError("Similarity selection manifest is not the frozen validation result.")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 12:
        raise ValueError("Similarity selection manifest must contain twelve folds.")
    aspects = [str(fold.get("heldout_aspect")) for fold in folds]
    if len(set(aspects)) != 12:
        raise ValueError("Similarity selection manifest has duplicate or missing aspects.")
    for fold in folds:
        thresholds = fold.get("thresholds")
        if not isinstance(thresholds, dict) or LOCAL_METHOD not in thresholds:
            raise ValueError(f"Missing frozen {LOCAL_METHOD} threshold.")
        if not math.isfinite(float(thresholds[LOCAL_METHOD])):
            raise ValueError("Non-finite strict TF-IDF threshold.")
    return folds


def fold_paths(
    *,
    strict_root: Path,
    sentiment_root: Path,
    qwen_full_root: Path,
    qwen_pilot_root: Path,
    fold: dict[str, Any],
    stage: str,
) -> tuple[Path, Path, Path]:
    strict_fold = str(fold["strict_fold_name"])
    sentiment_fold = str(fold["sentiment_fold_name"])
    strict_path = (
        strict_root
        / stage
        / LOCAL_METHOD
        / "folds"
        / strict_fold
        / f"predictions_{stage}.jsonl"
    )
    sentiment_path = (
        sentiment_root / sentiment_fold / f"best_{stage}_predictions.jsonl"
    )
    qwen_path = qwen_full_root / sentiment_fold / f"{stage}_predictions.jsonl"
    if stage == "validation" and not qwen_path.is_file():
        qwen_path = qwen_pilot_root / sentiment_fold / "validation_predictions.jsonl"
    return strict_path, sentiment_path, qwen_path


def candidate_pair_classes(aspect: str) -> list[str]:
    return [f"{aspect} | {sentiment}" for sentiment in SENTIMENTS]


def align_router_rows(
    strict_rows: list[dict[str, Any]],
    sentiment_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    *,
    stage: str,
    threshold: float,
) -> tuple[str, list[dict[str, Any]]]:
    aspect, aligned_local = align_prediction_rows(
        strict_rows,
        sentiment_rows,
        expected_method=LOCAL_METHOD,
        expected_stage=stage,
    )
    qwen_uids = [str(row.get("row_uid")) for row in qwen_rows]
    if len(set(qwen_uids)) != len(qwen_uids):
        raise ValueError("Duplicate row_uid in frozen candidate-pair Qwen predictions.")
    local_uids = [str(row["row_uid"]) for row in aligned_local]
    if set(qwen_uids) != set(local_uids):
        missing = sorted(set(local_uids) - set(qwen_uids))
        extra = sorted(set(qwen_uids) - set(local_uids))
        raise ValueError(
            f"Strict/Qwen row_uid set mismatch; missing={missing[:3]}, extra={extra[:3]}."
        )
    qwen_by_uid = {str(row["row_uid"]): row for row in qwen_rows}
    classes = set(candidate_pair_classes(aspect))
    output: list[dict[str, Any]] = []
    for local in aligned_local:
        uid = str(local["row_uid"])
        qwen = qwen_by_uid[uid]
        local_gold = sorted(str(label) for label in local["gold_pair_labels"])
        qwen_gold = sorted(str(label) for label in qwen.get("gold_pair_labels", []))
        if local_gold != qwen_gold:
            raise ValueError(f"Gold pair labels differ for {uid}.")
        qwen_pred = sorted(str(label) for label in qwen.get("pred_pair_labels", []))
        if any(label not in classes for label in qwen_pred):
            raise ValueError(f"Frozen Qwen emitted an out-of-scope pair for {uid}.")
        score = float(local["presence_score"])
        present = score >= threshold
        local_pred = [f"{aspect} | {local['aspect_sentiment']}"] if present else []
        output.append(
            {
                "row_uid": uid,
                "gold_pair_labels": local_gold,
                "local_pred_pair_labels": local_pred,
                "qwen_pred_pair_labels": qwen_pred,
                "presence_score": score,
                "threshold": threshold,
                "local_present": present,
                "distance": abs(score - threshold),
            }
        )
    return aspect, output


def ranked_cutoff(
    rows: list[dict[str, Any]],
    *,
    present: bool,
    fraction: float,
) -> float | None:
    if fraction == 0:
        return None
    eligible = sorted(
        (
            (float(row["distance"]), str(row["row_uid"]))
            for row in rows
            if bool(row["local_present"]) is present
        ),
        key=lambda value: (value[0], value[1]),
    )
    if not eligible:
        return None
    count = math.ceil(fraction * len(eligible))
    return float(eligible[count - 1][0])


def policy_id(rescue_fraction: float, confirm_fraction: float) -> str:
    return f"rescue_{rescue_fraction:.2f}_confirm_{confirm_fraction:.2f}"


def route_predictions(
    rows: list[dict[str, Any]],
    *,
    rescue_cutoff: float | None,
    confirm_cutoff: float | None,
) -> tuple[list[list[str]], list[bool]]:
    predictions: list[list[str]] = []
    calls: list[bool] = []
    for row in rows:
        cutoff = confirm_cutoff if row["local_present"] else rescue_cutoff
        call_qwen = cutoff is not None and float(row["distance"]) <= cutoff
        calls.append(call_qwen)
        predictions.append(
            list(
                row["qwen_pred_pair_labels"]
                if call_qwen
                else row["local_pred_pair_labels"]
            )
        )
    return predictions, calls


def score_predictions(
    rows: list[dict[str, Any]],
    *,
    aspect: str,
    predictions: list[list[str]],
    calls: list[bool],
    stage: str,
    scenario: str,
    rescue_fraction: float | None = None,
    confirm_fraction: float | None = None,
    rescue_cutoff: float | None = None,
    confirm_cutoff: float | None = None,
) -> dict[str, Any]:
    if len(rows) != len(predictions) or len(rows) != len(calls):
        raise ValueError("Prediction or call vector length mismatch.")
    metrics = evaluate_pair_and_aspect(
        [list(row["gold_pair_labels"]) for row in rows],
        predictions,
        candidate_pair_classes(aspect),
    )
    return {
        "stage": stage,
        "heldout_aspect": aspect,
        "scenario": scenario,
        "rows": len(rows),
        "qwen_call_rows": sum(calls),
        "qwen_call_rate": sum(calls) / len(rows),
        "rescue_fraction": rescue_fraction,
        "confirm_fraction": confirm_fraction,
        "rescue_cutoff": rescue_cutoff,
        "confirm_cutoff": confirm_cutoff,
        **metrics,
    }


def score_policy(
    rows: list[dict[str, Any]],
    *,
    aspect: str,
    stage: str,
    rescue_fraction: float,
    confirm_fraction: float,
    rescue_cutoff: float | None = None,
    confirm_cutoff: float | None = None,
) -> dict[str, Any]:
    if stage == "validation":
        rescue_cutoff = ranked_cutoff(
            rows, present=False, fraction=rescue_fraction
        )
        confirm_cutoff = ranked_cutoff(
            rows, present=True, fraction=confirm_fraction
        )
    predictions, calls = route_predictions(
        rows,
        rescue_cutoff=rescue_cutoff,
        confirm_cutoff=confirm_cutoff,
    )
    return score_predictions(
        rows,
        aspect=aspect,
        predictions=predictions,
        calls=calls,
        stage=stage,
        scenario=policy_id(rescue_fraction, confirm_fraction),
        rescue_fraction=rescue_fraction,
        confirm_fraction=confirm_fraction,
        rescue_cutoff=rescue_cutoff,
        confirm_cutoff=confirm_cutoff,
    )


def score_endpoint(
    rows: list[dict[str, Any]],
    *,
    aspect: str,
    stage: str,
    endpoint: str,
) -> dict[str, Any]:
    if endpoint == "strict_local_only":
        predictions = [list(row["local_pred_pair_labels"]) for row in rows]
        calls = [False] * len(rows)
    elif endpoint == "frozen_qwen_only":
        predictions = [list(row["qwen_pred_pair_labels"]) for row in rows]
        calls = [True] * len(rows)
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}.")
    return score_predictions(
        rows,
        aspect=aspect,
        predictions=predictions,
        calls=calls,
        stage=stage,
        scenario=endpoint,
    )


def aggregate(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    selected = [row for row in rows if row["scenario"] == scenario]
    if len(selected) != 12:
        raise ValueError(f"{scenario} has {len(selected)} folds rather than 12.")
    output: dict[str, Any] = {
        "stage": selected[0]["stage"],
        "scenario": scenario,
        "folds": 12,
    }
    for metric in AGGREGATE_METRICS:
        values = [float(row[metric]) for row in selected]
        output[f"{metric}_mean"] = mean(values)
        output[f"{metric}_median"] = median(values)
        output[f"{metric}_std"] = pstdev(values)
    for field in ("rescue_fraction", "confirm_fraction"):
        values = {row.get(field) for row in selected}
        if len(values) == 1:
            output[field] = values.pop()
    return output


def public_summary_row(row: dict[str, Any], scenario: str | None = None) -> dict[str, Any]:
    return {
        "stage": row["stage"],
        "scenario": scenario or row["scenario"],
        "folds": row["folds"],
        "pair_micro_f1_mean": row["pair_micro_f1_mean"],
        "pair_samples_f1_mean": row["pair_samples_f1_mean"],
        "pair_micro_precision_mean": row["pair_micro_precision_mean"],
        "pair_micro_recall_mean": row["pair_micro_recall_mean"],
        "presence_precision_mean": row["presence_precision_mean"],
        "presence_recall_mean": row["presence_recall_mean"],
        "presence_f1_mean": row["presence_f1_mean"],
        "pair_false_positive_rows_per_100_mean": row[
            "pair_false_positive_rows_per_100_mean"
        ],
        "pair_false_negative_rows_per_100_mean": row[
            "pair_false_negative_rows_per_100_mean"
        ],
        "qwen_call_rate_mean": row["qwen_call_rate_mean"],
        "rescue_fraction": row.get("rescue_fraction"),
        "confirm_fraction": row.get("confirm_fraction"),
    }


def select_global_policy(
    aggregate_rows: list[dict[str, Any]],
    *,
    maximum_call_rate: float,
) -> dict[str, Any]:
    eligible = [
        row
        for row in aggregate_rows
        if float(row["qwen_call_rate_mean"]) <= maximum_call_rate + 1e-12
    ]
    if not eligible:
        raise ValueError("No router policy satisfies the validation call-rate budget.")
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["pair_micro_f1_mean"]),
            -float(row["pair_samples_f1_mean"]),
            float(row["pair_false_positive_rows_per_100_mean"]),
            float(row["qwen_call_rate_mean"]),
            -float(row["pair_micro_precision_mean"]),
            str(row["scenario"]),
        ),
    )[0]


def load_fold_rows(
    *,
    strict_root: Path,
    sentiment_root: Path,
    qwen_full_root: Path,
    qwen_pilot_root: Path,
    fold: dict[str, Any],
    stage: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    strict_path, sentiment_path, qwen_path = fold_paths(
        strict_root=strict_root,
        sentiment_root=sentiment_root,
        qwen_full_root=qwen_full_root,
        qwen_pilot_root=qwen_pilot_root,
        fold=fold,
        stage=stage,
    )
    threshold = float(fold["thresholds"][LOCAL_METHOD])
    aspect, rows = align_router_rows(
        read_jsonl(strict_path),
        read_jsonl(sentiment_path),
        read_jsonl(qwen_path),
        stage=stage,
        threshold=threshold,
    )
    if aspect != str(fold["heldout_aspect"]):
        raise ValueError("Artifact aspect disagrees with the frozen fold manifest.")
    audit = {
        "stage": stage,
        "heldout_aspect": aspect,
        "rows": len(rows),
        "strict_path": str(strict_path),
        "strict_sha256": sha256_file(strict_path),
        "sentiment_path": str(sentiment_path),
        "sentiment_sha256": sha256_file(sentiment_path),
        "qwen_path": str(qwen_path),
        "qwen_sha256": sha256_file(qwen_path),
        "row_uid_alignment": "exact_set_joined_in_strict_artifact_order",
        "gold_alignment": "exact",
        "frozen_local_threshold": threshold,
    }
    return aspect, rows, audit


def run_validation(args: argparse.Namespace) -> Path:
    config = read_json(args.config)
    fractions = validate_protocol(config)
    similarity_payload = read_json(args.similarity_selection_manifest)
    folds = validate_similarity_manifest(similarity_payload)
    by_aspect: dict[str, list[dict[str, Any]]] = {}
    audits: list[dict[str, Any]] = []
    for fold in folds:
        aspect, rows, audit = load_fold_rows(
            strict_root=args.strict_root,
            sentiment_root=args.sentiment_root,
            qwen_full_root=args.qwen_full_root,
            qwen_pilot_root=args.qwen_pilot_root,
            fold=fold,
            stage="validation",
        )
        by_aspect[aspect] = rows
        audits.append(audit)

    per_fold: list[dict[str, Any]] = []
    cutoff_manifest: dict[str, dict[str, dict[str, float | None]]] = {}
    for rescue_fraction, confirm_fraction in product(fractions, repeat=2):
        scenario = policy_id(rescue_fraction, confirm_fraction)
        cutoff_manifest[scenario] = {}
        for fold in folds:
            aspect = str(fold["heldout_aspect"])
            result = score_policy(
                by_aspect[aspect],
                aspect=aspect,
                stage="validation",
                rescue_fraction=rescue_fraction,
                confirm_fraction=confirm_fraction,
            )
            per_fold.append(result)
            cutoff_manifest[scenario][aspect] = {
                "rescue_cutoff": result["rescue_cutoff"],
                "confirm_cutoff": result["confirm_cutoff"],
            }
    aggregate_rows = [
        aggregate(per_fold, policy_id(rescue, confirm))
        for rescue, confirm in product(fractions, repeat=2)
    ]
    maximum_call_rate = float(
        config["policy_search"]["hard_mean_validation_qwen_call_rate_max"]
    )
    selected = select_global_policy(
        aggregate_rows, maximum_call_rate=maximum_call_rate
    )
    selected_policy = str(selected["scenario"])
    selected_folds = [
        row for row in per_fold if row["scenario"] == selected_policy
    ]
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "validation",
        "protocol_complete": True,
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "similarity_selection_manifest_path": str(args.similarity_selection_manifest),
        "similarity_selection_manifest_sha256": sha256_file(
            args.similarity_selection_manifest
        ),
        "selection_order": config["policy_search"]["selection_order"],
        "maximum_mean_validation_qwen_call_rate": maximum_call_rate,
        "selected_policy": selected_policy,
        "selected_rescue_fraction": selected["rescue_fraction"],
        "selected_confirm_fraction": selected["confirm_fraction"],
        "selected_validation_metrics": selected,
        "folds": [
            {
                "strict_fold_name": fold["strict_fold_name"],
                "sentiment_fold_name": fold["sentiment_fold_name"],
                "heldout_aspect": fold["heldout_aspect"],
                "local_threshold": fold["thresholds"][LOCAL_METHOD],
                **cutoff_manifest[selected_policy][str(fold["heldout_aspect"])],
            }
            for fold in folds
        ],
        "test_artifacts_read": False,
    }
    validation_dir = args.output_dir / "validation"
    write_csv(validation_dir / "policy_results.csv", aggregate_rows)
    write_csv(validation_dir / "per_fold_results.csv", per_fold)
    write_csv(validation_dir / "selected_per_fold_results.csv", selected_folds)
    write_json(validation_dir / "alignment_audit.json", audits)
    manifest_path = validation_dir / "selection_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def validate_router_manifest(
    manifest: dict[str, Any],
    *,
    config_path: Path,
) -> list[dict[str, Any]]:
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("stage") != "validation"
        or manifest.get("protocol_complete") is not True
        or manifest.get("test_artifacts_read") is not False
    ):
        raise ValueError("Router selection manifest is not a frozen validation result.")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Router config changed after validation selection.")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != 12:
        raise ValueError("Router selection manifest must contain twelve frozen folds.")
    for fold in folds:
        for field in (
            "strict_fold_name",
            "sentiment_fold_name",
            "heldout_aspect",
            "local_threshold",
        ):
            if field not in fold:
                raise ValueError(f"Router fold is missing {field}.")
    return folds


def paired_diagnostics(
    per_fold_rows: list[dict[str, Any]],
    *,
    seed: int = 20260723,
    samples: int = 100_000,
) -> dict[str, Any]:
    local = {
        str(row["heldout_aspect"]): float(row["pair_micro_f1"])
        for row in per_fold_rows
        if row["scenario"] == "strict_local_only"
    }
    router = {
        str(row["heldout_aspect"]): float(row["pair_micro_f1"])
        for row in per_fold_rows
        if row["scenario"] == "selected_router"
    }
    if local.keys() != router.keys() or len(local) != 12:
        raise ValueError("Incomplete paired router/local test results.")
    aspects = list(local)
    deltas = [router[aspect] - local[aspect] for aspect in aspects]
    tolerance = 1e-12
    rng = random.Random(seed)
    bootstrap = sorted(
        mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(samples)
    )
    low_index = int(0.025 * samples)
    high_index = int(0.975 * samples) - 1
    observed = abs(sum(deltas))
    extreme = 0
    for mask in range(1 << len(deltas)):
        signed_sum = sum(
            delta if mask & (1 << index) else -delta
            for index, delta in enumerate(deltas)
        )
        if abs(signed_sum) >= observed - tolerance:
            extreme += 1
    return {
        "aspects": aspects,
        "deltas": deltas,
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "wins": sum(delta > tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "losses": sum(delta < -tolerance for delta in deltas),
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
        "bootstrap_95_percent_interval": [
            bootstrap[low_index],
            bootstrap[high_index],
        ],
        "exact_two_sided_sign_flip_p_value": extreme / (1 << len(deltas)),
    }


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    validate_protocol(config)
    manifest = read_json(args.router_selection_manifest)
    frozen_folds = validate_router_manifest(manifest, config_path=args.config)
    similarity_folds = validate_similarity_manifest(
        read_json(args.similarity_selection_manifest)
    )
    similarity_by_aspect = {
        str(fold["heldout_aspect"]): fold for fold in similarity_folds
    }
    per_fold: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    public_per_aspect: list[dict[str, Any]] = []
    selected_rescue = float(manifest["selected_rescue_fraction"])
    selected_confirm = float(manifest["selected_confirm_fraction"])

    for frozen in frozen_folds:
        aspect = str(frozen["heldout_aspect"])
        source_fold = similarity_by_aspect.get(aspect)
        if source_fold is None:
            raise ValueError(f"Frozen router aspect is missing from similarity manifest: {aspect}.")
        if not math.isclose(
            float(frozen["local_threshold"]),
            float(source_fold["thresholds"][LOCAL_METHOD]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Frozen router threshold differs from similarity manifest.")
        loaded_aspect, rows, audit = load_fold_rows(
            strict_root=args.strict_root,
            sentiment_root=args.sentiment_root,
            qwen_full_root=args.qwen_full_root,
            qwen_pilot_root=args.qwen_pilot_root,
            fold=source_fold,
            stage="test",
        )
        if loaded_aspect != aspect:
            raise ValueError("Test artifact aspect differs from router manifest.")
        local_result = score_endpoint(
            rows,
            aspect=aspect,
            stage="test",
            endpoint="strict_local_only",
        )
        qwen_result = score_endpoint(
            rows,
            aspect=aspect,
            stage="test",
            endpoint="frozen_qwen_only",
        )
        router_result = score_policy(
            rows,
            aspect=aspect,
            stage="test",
            rescue_fraction=selected_rescue,
            confirm_fraction=selected_confirm,
            rescue_cutoff=(
                None
                if frozen.get("rescue_cutoff") is None
                else float(frozen["rescue_cutoff"])
            ),
            confirm_cutoff=(
                None
                if frozen.get("confirm_cutoff") is None
                else float(frozen["confirm_cutoff"])
            ),
        )
        router_result["scenario"] = "selected_router"
        per_fold.extend([local_result, qwen_result, router_result])
        public_per_aspect.append(
            {
                "heldout_aspect": aspect,
                "strict_local_pair_micro_f1": local_result["pair_micro_f1"],
                "frozen_qwen_pair_micro_f1": qwen_result["pair_micro_f1"],
                "selected_router_pair_micro_f1": router_result["pair_micro_f1"],
                "router_minus_strict_local": (
                    float(router_result["pair_micro_f1"])
                    - float(local_result["pair_micro_f1"])
                ),
                "router_qwen_call_rate": router_result["qwen_call_rate"],
            }
        )
        audits.append(audit)

    aggregate_rows = [
        aggregate(per_fold, scenario)
        for scenario in (
            "strict_local_only",
            "frozen_qwen_only",
            "selected_router",
        )
    ]
    by_scenario = {str(row["scenario"]): row for row in aggregate_rows}
    local = by_scenario["strict_local_only"]
    router = by_scenario["selected_router"]
    gain = float(router["pair_micro_f1_mean"]) - float(local["pair_micro_f1_mean"])
    call_rate = float(router["qwen_call_rate_mean"])
    gate = config["admission_gate"]
    passed = (
        gain
        >= float(gate["minimum_pair_micro_f1_gain_over_strict_local"]) - 1e-12
        and call_rate <= float(gate["maximum_test_mean_qwen_call_rate"]) + 1e-12
    )
    diagnostics = paired_diagnostics(per_fold)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "test",
        "selected_policy": manifest["selected_policy"],
        "selected_rescue_fraction": selected_rescue,
        "selected_confirm_fraction": selected_confirm,
        "aggregate_results": aggregate_rows,
        "router_minus_strict_local_pair_micro_f1": gain,
        "admission_gate_passed": passed,
        "admission_decision": (
            gate["decision_if_passed"] if passed else gate["decision_if_failed"]
        ),
        "paired_diagnostics": diagnostics,
        "test_policy_scope": [
            "strict_local_only",
            "frozen_qwen_only",
            "selected_router",
        ],
        "test_boundary_tuning": False,
    }
    test_dir = args.output_dir / "test"
    write_csv(test_dir / "per_fold_results.csv", per_fold)
    write_csv(test_dir / "aggregate_results.csv", aggregate_rows)
    write_csv(test_dir / "per_aspect_comparison.csv", public_per_aspect)
    write_json(test_dir / "alignment_audit.json", audits)
    write_json(test_dir / "summary.json", summary)
    write_csv(
        args.public_summary_csv,
        [
            public_summary_row(
                manifest["selected_validation_metrics"],
                "validation_selected_local_only",
            ),
            *(public_summary_row(row) for row in aggregate_rows),
        ],
    )
    write_csv(args.public_per_aspect_csv, public_per_aspect)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-select and test the preregistered strict TF-IDF plus "
            "aspect-conditioned sentiment to frozen candidate-pair Qwen router."
        )
    )
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/loao_strict_tfidf_aspect_qwen_router_v1.json"
        ),
    )
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--sentiment-root", type=Path, required=True)
    parser.add_argument("--qwen-full-root", type=Path, required=True)
    parser.add_argument("--qwen-pilot-root", type=Path, required=True)
    parser.add_argument("--similarity-selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/analysis/loao_strict_tfidf_aspect_qwen_router_v1"
        ),
    )
    parser.add_argument("--router-selection-manifest", type=Path)
    parser.add_argument(
        "--public-summary-csv",
        type=Path,
        default=Path(
            "docs/thesis_figure_data/"
            "loao_strict_tfidf_aspect_qwen_router_v1_summary.csv"
        ),
    )
    parser.add_argument(
        "--public-per-aspect-csv",
        type=Path,
        default=Path(
            "docs/thesis_figure_data/"
            "loao_strict_tfidf_aspect_qwen_router_v1_per_aspect.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "validation":
        if args.router_selection_manifest is not None:
            raise ValueError("--router-selection-manifest is only valid for test.")
        path = run_validation(args)
        print(f"Wrote frozen validation selection: {path}")
    else:
        if args.router_selection_manifest is None:
            raise ValueError("Test requires --router-selection-manifest.")
        summary = run_test(args)
        print(
            "Test router pair F1 delta: "
            f"{summary['router_minus_strict_local_pair_micro_f1']:+.6f}; "
            f"admission gate passed={summary['admission_gate_passed']}"
        )


if __name__ == "__main__":
    main()
