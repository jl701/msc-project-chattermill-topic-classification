from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "loao_unified_candidate_pair_experimental_v1"
PILOT_FOLDS = (
    "Company brand: Competitor",
    "Company brand: General satisfaction",
    "Staff support: Email",
)
FULL_FOLDS = (
    "Account management: Account access",
    "Company brand: Competitor",
    "Company brand: General satisfaction",
    "Company brand: Reviews",
    "Logistics rides: Speed",
    "Online experience: App website",
    "Purchase booking experience: Ease of use",
    "Staff support: Attitude of staff",
    "Staff support: Email",
    "Staff support: Phone",
    "Value: Discounts promotions",
    "Value: Price value for money",
)
TIE_TOLERANCE = 1e-12
SUMMARY_FILENAME = "unified_improvement_summary.json"
PER_FOLD_FILENAME = "unified_improvement_per_fold.csv"


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} must contain a JSON object: {path}")
    return payload


def _require_exact(value: object, expected: object, description: str) -> None:
    if value != expected:
        raise ValueError(f"{description} must be {expected!r}; got {value!r}.")


def _require_exact_fold_set(
    folds: Iterable[object], expected: tuple[str, ...], description: str
) -> list[str]:
    values = list(folds)
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{description} must contain only fold names.")
    names = [str(value) for value in values]
    if len(names) != len(expected) or len(set(names)) != len(expected):
        raise ValueError(
            f"{description} must contain exactly {len(expected)} unique folds; "
            f"found {len(names)} rows and {len(set(names))} unique folds."
        )
    if set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(f"{description} has the wrong fold set; missing={missing}, extra={extra}.")
    return names


def _metric(row: dict[str, Any], key: str, description: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} {key} must be numeric; got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} {key} must be finite and in [0, 1]; got {value!r}.")
    return result


def _result_rows(payload: dict[str, Any], description: str) -> list[dict[str, Any]]:
    raw_rows = payload.get("results")
    if not isinstance(raw_rows, list) or not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError(f"{description} must contain a list of result objects.")
    return raw_rows


def extract_local_pilot(
    payload: dict[str, Any], *, model: str, description: str
) -> tuple[dict[str, float], dict[str, float]]:
    _require_exact(payload.get("stage"), "pilot", f"{description} stage")
    rows = [row for row in _result_rows(payload, description) if row.get("model") == model]
    expected_keys = {(fold, variant) for fold in PILOT_FOLDS for variant in ("control", "enhanced")}
    keys: list[tuple[str, str]] = []
    by_variant: dict[str, dict[str, float]] = {"control": {}, "enhanced": {}}
    for row in rows:
        _require_exact(row.get("split"), "validation", f"{description} split")
        aspect = row.get("heldout_aspect")
        variant = row.get("variant")
        if not isinstance(aspect, str) or variant not in {"control", "enhanced"}:
            raise ValueError(f"{description} has an invalid heldout_aspect or variant row.")
        keys.append((aspect, str(variant)))
        by_variant[str(variant)][aspect] = _metric(row, "pair_micro_f1", description)

    if len(keys) != len(expected_keys) or len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError(
            f"{description} must contain exactly one control and one enhanced validation row "
            f"for each of the {len(PILOT_FOLDS)} preregistered pilot folds."
        )
    _require_exact_fold_set(by_variant["control"], PILOT_FOLDS, f"{description} control folds")
    _require_exact_fold_set(by_variant["enhanced"], PILOT_FOLDS, f"{description} enhanced folds")
    return by_variant["control"], by_variant["enhanced"]


def extract_full_candidate_pair(
    payload: dict[str, Any], *, mode: str, model: str, description: str
) -> dict[str, float]:
    _require_exact(payload.get("protocol_id"), PROTOCOL_ID, f"{description} protocol_id")
    _require_exact(payload.get("mode"), mode, f"{description} mode")
    expected_from_summary = payload.get("expected_folds")
    if not isinstance(expected_from_summary, list):
        raise ValueError(f"{description} must declare expected_folds.")
    _require_exact_fold_set(expected_from_summary, FULL_FOLDS, f"{description} expected_folds")

    rows = [row for row in _result_rows(payload, description) if row.get("model") == model]
    expected_keys = {(fold, split) for fold in FULL_FOLDS for split in ("validation", "test")}
    keys: list[tuple[str, str]] = []
    test_scores: dict[str, float] = {}
    for row in rows:
        _require_exact(row.get("variant"), "enhanced", f"{description} variant")
        aspect = row.get("heldout_aspect")
        split = row.get("split")
        if not isinstance(aspect, str) or split not in {"validation", "test"}:
            raise ValueError(f"{description} has an invalid heldout_aspect or split row.")
        keys.append((aspect, str(split)))
        score = _metric(row, "pair_micro_f1", description)
        if split == "test":
            test_scores[aspect] = score

    if len(keys) != len(expected_keys) or len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError(
            f"{description} must be complete: exactly one validation and one test row for each "
            f"of the {len(FULL_FOLDS)} canonical LOAO folds."
        )
    _require_exact_fold_set(test_scores, FULL_FOLDS, f"{description} test folds")
    runtime_contract = payload.get("qwen_runtime_contract")
    if not isinstance(runtime_contract, dict) or not runtime_contract:
        raise ValueError(f"{description} must declare a non-empty qwen_runtime_contract.")
    return test_scores


def extract_historical_qwen(payload: dict[str, Any], description: str) -> dict[str, float]:
    declared_folds = payload.get("heldout_aspects")
    if not isinstance(declared_folds, list):
        raise ValueError(f"{description} must declare heldout_aspects.")
    _require_exact_fold_set(declared_folds, FULL_FOLDS, f"{description} heldout_aspects")
    _require_exact(payload.get("strategy"), "label_masked", f"{description} strategy")
    _require_exact(payload.get("eval_label_scope"), "heldout", f"{description} eval_label_scope")
    _require_exact(payload.get("eval_row_scope"), "all", f"{description} eval_row_scope")

    rows = _result_rows(payload, description)
    if len(rows) != len(FULL_FOLDS):
        raise ValueError(f"{description} must contain exactly {len(FULL_FOLDS)} result rows.")
    scores: dict[str, float] = {}
    for row in rows:
        _require_exact(row.get("split"), "test", f"{description} split")
        aspect = row.get("heldout_aspect")
        if not isinstance(aspect, str) or aspect in scores:
            raise ValueError(f"{description} contains an invalid or duplicate heldout_aspect.")
        scores[aspect] = _metric(row, "pair_micro_f1", description)
    _require_exact_fold_set(scores, FULL_FOLDS, f"{description} result folds")
    return scores


def require_matched_full_protocols(
    frozen_payload: dict[str, Any], qlora_payload: dict[str, Any]
) -> None:
    for key in ("protocol_id", "expected_folds", "qwen_runtime_contract"):
        if frozen_payload.get(key) != qlora_payload.get(key):
            raise ValueError(f"QLoRA/frozen matched comparison requires identical {key}.")


def build_comparison(
    *,
    comparison_id: str,
    reference_label: str,
    candidate_label: str,
    reference_scores: dict[str, float],
    candidate_scores: dict[str, float],
    folds: tuple[str, ...],
    stage: str,
    matched_protocol: bool,
    interpretation: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_exact_fold_set(reference_scores, folds, f"{comparison_id} reference folds")
    _require_exact_fold_set(candidate_scores, folds, f"{comparison_id} candidate folds")
    per_fold: list[dict[str, Any]] = []
    deltas: list[float] = []
    for aspect in folds:
        reference = float(reference_scores[aspect])
        candidate = float(candidate_scores[aspect])
        delta = candidate - reference
        deltas.append(delta)
        per_fold.append(
            {
                "comparison_id": comparison_id,
                "stage": stage,
                "split": "validation" if stage == "pilot" else "test",
                "matched_protocol": matched_protocol,
                "heldout_aspect": aspect,
                "reference_label": reference_label,
                "candidate_label": candidate_label,
                "reference_pair_micro_f1": reference,
                "candidate_pair_micro_f1": candidate,
                "pair_micro_f1_delta": delta,
                "delta_percentage_points": 100.0 * delta,
                "relative_delta_percent": None if reference == 0.0 else 100.0 * delta / reference,
            }
        )

    reference_mean = statistics.fmean(reference_scores[fold] for fold in folds)
    candidate_mean = statistics.fmean(candidate_scores[fold] for fold in folds)
    mean_delta = statistics.fmean(deltas)
    aggregate = {
        "comparison_id": comparison_id,
        "metric": "pair_micro_f1",
        "stage": stage,
        "split": "validation" if stage == "pilot" else "test",
        "fold_count": len(folds),
        "folds": list(folds),
        "matched_protocol": matched_protocol,
        "interpretation": interpretation,
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "reference_mean_pair_micro_f1": reference_mean,
        "candidate_mean_pair_micro_f1": candidate_mean,
        "mean_pair_micro_f1_delta": mean_delta,
        "mean_delta_percentage_points": 100.0 * mean_delta,
        "relative_change_of_means_percent": (
            None if reference_mean == 0.0 else 100.0 * mean_delta / reference_mean
        ),
        "median_pair_micro_f1_delta": statistics.median(deltas),
        "minimum_pair_micro_f1_delta": min(deltas),
        "maximum_pair_micro_f1_delta": max(deltas),
        "wins": sum(delta > TIE_TOLERANCE for delta in deltas),
        "ties": sum(abs(delta) <= TIE_TOLERANCE for delta in deltas),
        "losses": sum(delta < -TIE_TOLERANCE for delta in deltas),
    }
    return aggregate, per_fold


def build_result_summary(
    *,
    tfidf_pilot_path: Path,
    distilbert_pilot_path: Path,
    frozen_full_path: Path,
    historical_qwen_path: Path,
    qlora_full_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tfidf_payload = read_json_object(tfidf_pilot_path, "corrected TF-IDF pilot summary")
    distilbert_payload = read_json_object(distilbert_pilot_path, "DistilBERT pilot summary")
    frozen_payload = read_json_object(frozen_full_path, "frozen candidate-pair Qwen full summary")
    historical_payload = read_json_object(historical_qwen_path, "historical Qwen JSON summary")
    qlora_payload = read_json_object(qlora_full_path, "QLoRA candidate-pair Qwen full summary")

    tfidf_control, tfidf_enhanced = extract_local_pilot(
        tfidf_payload, model="tfidf", description="corrected TF-IDF pilot"
    )
    distilbert_control, distilbert_enhanced = extract_local_pilot(
        distilbert_payload, model="distilbert", description="DistilBERT pilot"
    )
    frozen = extract_full_candidate_pair(
        frozen_payload,
        mode="frozen",
        model="qwen_frozen_candidate_pair",
        description="frozen candidate-pair Qwen full summary",
    )
    historical = extract_historical_qwen(historical_payload, "historical Qwen JSON summary")
    qlora = extract_full_candidate_pair(
        qlora_payload,
        mode="qlora",
        model="qwen_qlora_candidate_pair",
        description="QLoRA candidate-pair Qwen full summary",
    )
    require_matched_full_protocols(frozen_payload, qlora_payload)

    specifications = (
        {
            "comparison_id": "corrected_tfidf_pilot_enhanced_vs_control",
            "reference_label": "corrected_tfidf_control",
            "candidate_label": "corrected_tfidf_enhanced",
            "reference_scores": tfidf_control,
            "candidate_scores": tfidf_enhanced,
            "folds": PILOT_FOLDS,
            "stage": "pilot",
            "matched_protocol": True,
            "interpretation": (
                "Matched enhanced-versus-control comparison on the three preregistered validation "
                "pilot folds; it is a pilot result, not a 12-fold test claim."
            ),
        },
        {
            "comparison_id": "distilbert_pilot_enhanced_vs_control",
            "reference_label": "distilbert_control",
            "candidate_label": "distilbert_enhanced",
            "reference_scores": distilbert_control,
            "candidate_scores": distilbert_enhanced,
            "folds": PILOT_FOLDS,
            "stage": "pilot",
            "matched_protocol": True,
            "interpretation": (
                "Matched enhanced-versus-control comparison on the three preregistered validation "
                "pilot folds; it is a pilot result, not a 12-fold test claim."
            ),
        },
        {
            "comparison_id": "frozen_candidate_pair_qwen_full_vs_historical_json_qwen",
            "reference_label": "historical_qwen_json",
            "candidate_label": "frozen_candidate_pair_qwen",
            "reference_scores": historical,
            "candidate_scores": frozen,
            "folds": FULL_FOLDS,
            "stage": "full",
            "matched_protocol": False,
            "interpretation": (
                "Descriptive non-matched protocol comparison only: prompt, output space, and "
                "thresholding differ, so the delta cannot be attributed solely to candidate-pair scoring."
            ),
        },
        {
            "comparison_id": "qlora_full_vs_frozen_candidate_pair_qwen",
            "reference_label": "frozen_candidate_pair_qwen",
            "candidate_label": "qlora_candidate_pair_qwen",
            "reference_scores": frozen,
            "candidate_scores": qlora,
            "folds": FULL_FOLDS,
            "stage": "full",
            "matched_protocol": True,
            "interpretation": (
                "Protocol-matched paired 12-fold test comparison; this is the primary estimate of "
                "the QLoRA fine-tuning effect within the candidate-pair design."
            ),
        },
    )

    comparisons: list[dict[str, Any]] = []
    per_fold_rows: list[dict[str, Any]] = []
    for specification in specifications:
        aggregate, rows = build_comparison(**specification)  # type: ignore[arg-type]
        comparisons.append(aggregate)
        per_fold_rows.extend(rows)

    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "experimental_only_pending_user_approval",
        "metric": "pair_micro_f1",
        "comparison_count": len(comparisons),
        "per_fold_row_count": len(per_fold_rows),
        "per_fold_csv": PER_FOLD_FILENAME,
        "inputs": {
            "corrected_tfidf_pilot_summary": str(tfidf_pilot_path),
            "distilbert_pilot_summary": str(distilbert_pilot_path),
            "frozen_candidate_pair_qwen_full_summary": str(frozen_full_path),
            "historical_qwen_json_summary": str(historical_qwen_path),
            "qlora_candidate_pair_qwen_full_summary": str(qlora_full_path),
        },
        "comparisons": comparisons,
    }
    return summary, per_fold_rows


def require_experimental_output_dir(
    output_dir: Path, experimental_root: Path | None = None
) -> Path:
    root = (experimental_root or PROJECT_ROOT / "outputs" / "experimental").resolve()
    resolved = output_dir.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Output directory must be inside the experimental root {root}.") from exc
    if relative == Path("."):
        raise ValueError("Choose a dedicated directory below outputs/experimental, not its root.")
    return resolved


def write_outputs(
    summary: dict[str, Any], per_fold_rows: list[dict[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    per_fold_path = output_dir / PER_FOLD_FILENAME
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "comparison_id",
        "stage",
        "split",
        "matched_protocol",
        "heldout_aspect",
        "reference_label",
        "candidate_label",
        "reference_pair_micro_f1",
        "candidate_pair_micro_f1",
        "pair_micro_f1_delta",
        "delta_percentage_points",
        "relative_delta_percent",
    ]
    with per_fold_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_fold_rows)
    return summary_path, per_fold_path


def parse_args() -> argparse.Namespace:
    experiment_root = PROJECT_ROOT / "outputs" / "experimental" / "loao_unified_candidate_pair_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Read completed isolated LOAO summaries and write a descriptive improvement summary; "
            "this script does not load or run any model."
        )
    )
    parser.add_argument(
        "--tfidf-pilot-summary",
        type=Path,
        default=experiment_root / "local_pilot_tfidf_prereg_corrected_20260718" / "summary.json",
    )
    parser.add_argument(
        "--distilbert-pilot-summary",
        type=Path,
        default=experiment_root / "local_pilot_preregistered_20260717" / "summary.json",
    )
    parser.add_argument(
        "--frozen-full-summary",
        type=Path,
        default=experiment_root / "qwen_frozen_full_preregistered_20260718" / "summary.json",
    )
    parser.add_argument(
        "--historical-qwen-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "llm"
            / "qwen_loao_heldout_aspect_all_rows_test_20260701"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--qlora-full-summary",
        type=Path,
        default=experiment_root / "qwen_qlora_full_preregistered_20260718" / "summary.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_dir = require_experimental_output_dir(args.output_dir)
        summary, per_fold_rows = build_result_summary(
            tfidf_pilot_path=args.tfidf_pilot_summary,
            distilbert_pilot_path=args.distilbert_pilot_summary,
            frozen_full_path=args.frozen_full_summary,
            historical_qwen_path=args.historical_qwen_summary,
            qlora_full_path=args.qlora_full_summary,
        )
        summary_path, per_fold_path = write_outputs(summary, per_fold_rows, output_dir)
    except ValueError as exc:
        raise SystemExit(f"Result summary validation failed: {exc}") from exc
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "per_fold_csv": str(per_fold_path),
                "comparisons": summary["comparison_count"],
                "per_fold_rows": summary["per_fold_row_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
