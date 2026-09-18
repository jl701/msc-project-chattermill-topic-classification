from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS
from msc_project.experiments.unified_pair_evaluation import (
    evaluate_score_matrix,
    select_threshold,
)


ANALYSIS_PROTOCOL_ID = "loao_duplicate_text_posthoc_sensitivity_v1"
ANALYSIS_CLASSIFICATION = "post_hoc_descriptive_sensitivity_only"
SOURCE_PROTOCOL_ID = "loao_unified_candidate_pair_experimental_v1"
PILOT_FOLDS = frozenset(
    {
        "Company brand: Competitor",
        "Company brand: General satisfaction",
        "Staff support: Email",
    }
)
SENTIMENT_INDEX = {
    sentiment: index for index, sentiment in enumerate(CANDIDATE_SENTIMENTS)
}
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ScoreArtifact:
    matrix: np.ndarray
    identity_sha256: str
    path: Path


@dataclass(frozen=True)
class FoldSensitivity:
    result: dict[str, object]
    duplicate_rows: pd.DataFrame
    original_threshold_sweep: pd.DataFrame
    retained_threshold_sweep: pd.DataFrame


def normalise_review_text(value: object) -> str:
    """Conservatively normalise review text for exact-duplicate detection."""

    if value is None or bool(pd.isna(value)):
        return ""
    normalised = unicodedata.normalize("NFKC", str(value)).casefold()
    return _WHITESPACE.sub(" ", normalised).strip()


def ordered_eval_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"row_uid", "text", "supervision_pair_labels"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation frame is missing columns: {sorted(missing)}")
    if frame["row_uid"].astype(str).duplicated().any():
        raise ValueError("Evaluation row_uid values must be unique.")
    return (
        frame.assign(_row_uid=frame["row_uid"].astype(str))
        .sort_values("_row_uid", kind="stable")
        .drop(columns="_row_uid")
        .reset_index(drop=True)
    )


def duplicate_mask_against_training(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    if "text" not in train_frame or "text" not in eval_frame:
        raise ValueError("Training and evaluation frames must contain text.")
    train_text = train_frame["text"].map(normalise_review_text)
    counts = train_text[train_text.ne("")].value_counts().to_dict()
    eval_text = eval_frame["text"].map(normalise_review_text)
    mask = eval_text.ne("") & eval_text.isin(counts)
    return mask.astype(bool), {str(key): int(value) for key, value in counts.items()}


def _expected_target(labels: Iterable[str], aspect: str, sentiment: str) -> int:
    return int(f"{aspect} | {sentiment}" in set(str(value) for value in labels))


def load_validated_score_artifact(
    path: Path,
    expected_frame: pd.DataFrame,
    heldout_aspect: str,
) -> ScoreArtifact:
    """Load scores only after exact FABSA row/pair identity validation."""

    if not path.is_file():
        raise ValueError(f"Missing pair-score artifact: {path}")
    scores = pd.read_csv(path, keep_default_na=False)
    required = {
        "row_index",
        "row_uid",
        "text",
        "candidate_aspect",
        "candidate_sentiment",
        "target",
        "variant",
        "score",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Pair-score artifact is missing columns: {sorted(missing)}")

    expected = ordered_eval_frame(expected_frame)
    expected_rows = len(expected)
    expected_score_rows = expected_rows * len(CANDIDATE_SENTIMENTS)
    if len(scores) != expected_score_rows:
        raise ValueError(
            f"Pair-score artifact has {len(scores)} rows; expected {expected_score_rows}."
        )

    row_indices = pd.to_numeric(scores["row_index"], errors="coerce")
    if row_indices.isna().any() or not np.equal(row_indices, np.floor(row_indices)).all():
        raise ValueError("Pair-score row_index values must be finite integers.")
    scores = scores.copy()
    scores["row_index"] = row_indices.astype(int)
    if not scores["row_index"].between(0, expected_rows - 1).all():
        raise ValueError("Pair-score row_index is outside the expected evaluation frame.")

    key_columns = ["row_index", "candidate_sentiment"]
    if scores.duplicated(key_columns).any():
        raise ValueError("Pair-score artifact contains duplicate row/sentiment keys.")
    if set(scores["candidate_sentiment"].astype(str)) != set(CANDIDATE_SENTIMENTS):
        raise ValueError("Pair-score artifact has an unexpected sentiment set.")
    if set(scores["candidate_aspect"].astype(str)) != {heldout_aspect}:
        raise ValueError("Pair-score artifact held-out aspect does not match the fold.")
    if set(scores["variant"].astype(str)) != {"enhanced"}:
        raise ValueError("Pair-score artifact must use the enhanced candidate variant.")

    score_values = pd.to_numeric(scores["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(score_values).all() or not ((score_values >= 0.0) & (score_values <= 1.0)).all():
        raise ValueError("Pair-score values must be finite probabilities in [0, 1].")
    target_values = pd.to_numeric(scores["target"], errors="coerce")
    if target_values.isna().any() or not target_values.isin([0, 1]).all():
        raise ValueError("Pair-score target values must be binary.")

    matrix = np.full((expected_rows, len(CANDIDATE_SENTIMENTS)), np.nan, dtype=float)
    identity_records: list[list[object]] = []
    for position, row in enumerate(scores.itertuples(index=False)):
        row_index = int(row.row_index)
        sentiment = str(row.candidate_sentiment)
        expected_row = expected.iloc[row_index]
        if str(row.row_uid) != str(expected_row["row_uid"]):
            raise ValueError(f"Pair-score row_uid mismatch at row_index {row_index}.")
        if str(row.text) != str(expected_row["text"]):
            raise ValueError(f"Pair-score text mismatch at row_index {row_index}.")
        expected_target = _expected_target(
            expected_row["supervision_pair_labels"], heldout_aspect, sentiment
        )
        if int(target_values.iloc[position]) != expected_target:
            raise ValueError(
                f"Pair-score target mismatch at row_index {row_index}, sentiment {sentiment}."
            )
        column = SENTIMENT_INDEX[sentiment]
        matrix[row_index, column] = score_values[position]
        identity_records.append(
            [
                row_index,
                str(row.row_uid),
                str(row.text),
                heldout_aspect,
                sentiment,
                expected_target,
            ]
        )
    if np.isnan(matrix).any():
        raise ValueError("Pair-score artifact does not contain the complete evaluation grid.")

    encoded = json.dumps(
        sorted(identity_records, key=lambda item: (int(item[0]), str(item[4]))),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return ScoreArtifact(
        matrix=matrix,
        identity_sha256=hashlib.sha256(encoded).hexdigest(),
        path=path.resolve(),
    )


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def resolve_pair_score_path(
    *,
    full_root: Path,
    pilot_root: Path,
    heldout_aspect: str,
    split: str,
) -> Path:
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported score split: {split!r}")
    fold_name = slugify(heldout_aspect)
    full_path = full_root / fold_name / f"{split}_pair_scores.csv"
    if full_path.is_file():
        return full_path
    if split == "validation" and heldout_aspect in PILOT_FOLDS:
        pilot_path = pilot_root / fold_name / "validation_pair_scores.csv"
        if pilot_path.is_file():
            return pilot_path
        raise ValueError(
            f"Missing reused pilot validation scores for {heldout_aspect!r}: {pilot_path}"
        )
    raise ValueError(f"Missing full-run {split} scores for {heldout_aspect!r}: {full_path}")


def validate_run_root(
    root: Path,
    *,
    expected_mode: str,
    expected_stage: str,
    expected_folds: Iterable[str],
    require_finished: bool,
) -> dict[str, object]:
    """Verify that a score root has the declared source-experiment provenance."""

    path = root / "run_manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing source run manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid source run manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Source run manifest must be a JSON object: {path}")
    expected = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "mode": expected_mode,
        "stage": expected_stage,
        "variant": "enhanced",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Source run manifest {key} must be {value!r}; got {payload.get(key)!r}."
            )
    actual_folds = payload.get("folds")
    fold_set = set(str(value) for value in expected_folds)
    if (
        not isinstance(actual_folds, list)
        or len(actual_folds) != len(fold_set)
        or len(set(actual_folds)) != len(actual_folds)
        or set(actual_folds) != fold_set
    ):
        raise ValueError("Source run manifest folds do not match the required fold set.")
    if require_finished and not isinstance(payload.get("finished_at"), str):
        raise ValueError(f"Source full run is not marked finished: {path}")
    return payload


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _duplicate_rows(
    train_counts: dict[str, int],
    frames: dict[str, pd.DataFrame],
    masks: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("validation", "test"):
        frame = frames[split]
        for row_index in np.flatnonzero(masks[split].to_numpy(dtype=bool)):
            normalised = normalise_review_text(frame.iloc[row_index]["text"])
            rows.append(
                {
                    "split": split,
                    "row_index": int(row_index),
                    "row_uid": str(frame.iloc[row_index]["row_uid"]),
                    "normalised_text_sha256": hashlib.sha256(
                        normalised.encode("utf-8")
                    ).hexdigest(),
                    "matching_training_rows": int(train_counts[normalised]),
                    "heldout_gold_pair_count": int(
                        len(frame.iloc[row_index]["supervision_pair_labels"])
                    ),
                    "heldout_gold_present": bool(
                        frame.iloc[row_index]["supervision_pair_labels"]
                    ),
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "split",
            "row_index",
            "row_uid",
            "normalised_text_sha256",
            "matching_training_rows",
            "heldout_gold_pair_count",
            "heldout_gold_present",
        ],
    )


def analyse_model_fold(
    *,
    model_name: str,
    heldout_aspect: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    full_root: Path,
    pilot_root: Path,
) -> FoldSensitivity:
    """Run the no-inference sensitivity analysis for one saved-score fold."""

    frames = {
        "validation": ordered_eval_frame(validation_frame),
        "test": ordered_eval_frame(test_frame),
    }
    masks: dict[str, pd.Series] = {}
    train_counts: dict[str, int] | None = None
    for split, frame in frames.items():
        masks[split], counts = duplicate_mask_against_training(train_frame, frame)
        if train_counts is None:
            train_counts = counts
        elif counts != train_counts:
            raise AssertionError("Internal training-text counts changed between splits.")
    assert train_counts is not None

    artifacts = {
        split: load_validated_score_artifact(
            resolve_pair_score_path(
                full_root=full_root,
                pilot_root=pilot_root,
                heldout_aspect=heldout_aspect,
                split=split,
            ),
            frame,
            heldout_aspect,
        )
        for split, frame in frames.items()
    }
    retained = {split: ~mask.to_numpy(dtype=bool) for split, mask in masks.items()}
    if not retained["validation"].any():
        raise ValueError(f"No validation rows remain after duplicate removal for {heldout_aspect!r}.")
    if not retained["test"].any():
        raise ValueError(f"No test rows remain after duplicate removal for {heldout_aspect!r}.")

    true_pairs = {
        split: [list(value) for value in frame["supervision_pair_labels"]]
        for split, frame in frames.items()
    }
    original_selection = select_threshold(
        true_pairs["validation"], artifacts["validation"].matrix, heldout_aspect
    )
    original_test_metrics, _ = evaluate_score_matrix(
        true_pairs["test"],
        artifacts["test"].matrix,
        heldout_aspect,
        original_selection.threshold,
    )

    retained_validation_pairs = [
        labels
        for labels, keep in zip(true_pairs["validation"], retained["validation"])
        if keep
    ]
    retained_validation_scores = artifacts["validation"].matrix[retained["validation"]]
    retained_selection = select_threshold(
        retained_validation_pairs, retained_validation_scores, heldout_aspect
    )
    retained_test_pairs = [
        labels for labels, keep in zip(true_pairs["test"], retained["test"]) if keep
    ]
    retained_test_scores = artifacts["test"].matrix[retained["test"]]
    retained_at_original_metrics, _ = evaluate_score_matrix(
        retained_test_pairs,
        retained_test_scores,
        heldout_aspect,
        original_selection.threshold,
    )
    retained_reselected_metrics, _ = evaluate_score_matrix(
        retained_test_pairs,
        retained_test_scores,
        heldout_aspect,
        retained_selection.threshold,
    )

    result: dict[str, object] = {
        "analysis_protocol_id": ANALYSIS_PROTOCOL_ID,
        "analysis_classification": ANALYSIS_CLASSIFICATION,
        "model": model_name,
        "heldout_aspect": heldout_aspect,
        "train_rows": int(len(train_frame)),
        "train_unique_nonempty_normalised_texts": int(len(train_counts)),
        "validation_score_path": str(artifacts["validation"].path),
        "test_score_path": str(artifacts["test"].path),
        "validation_score_identity_sha256": artifacts["validation"].identity_sha256,
        "test_score_identity_sha256": artifacts["test"].identity_sha256,
        "original_selected_threshold": float(original_selection.threshold),
        "retained_selected_threshold": float(retained_selection.threshold),
    }
    for split in ("validation", "test"):
        total = int(len(frames[split]))
        removed = int(masks[split].sum())
        gold_present = frames[split]["supervision_pair_labels"].map(bool).to_numpy(dtype=bool)
        duplicate_values = masks[split].to_numpy(dtype=bool)
        result.update(
            {
                f"{split}_rows_original": total,
                f"{split}_duplicate_rows_removed": removed,
                f"{split}_rows_retained": total - removed,
                f"{split}_duplicate_fraction": float(removed / total) if total else 0.0,
                f"{split}_gold_present_rows_original": int(gold_present.sum()),
                f"{split}_gold_present_rows_removed": int(
                    np.sum(gold_present & duplicate_values)
                ),
                f"{split}_gold_present_rows_retained": int(
                    np.sum(gold_present & ~duplicate_values)
                ),
            }
        )
    result.update(_prefix_metrics("original_all_test", original_test_metrics))
    result.update(
        _prefix_metrics("retained_test_at_original_threshold", retained_at_original_metrics)
    )
    result.update(
        _prefix_metrics("retained_test_reselected", retained_reselected_metrics)
    )
    result["retained_test_pair_micro_f1_threshold_effect"] = float(
        retained_reselected_metrics["pair_micro_f1"]
        - retained_at_original_metrics["pair_micro_f1"]
    )

    return FoldSensitivity(
        result=result,
        duplicate_rows=_duplicate_rows(train_counts, frames, masks),
        original_threshold_sweep=original_selection.sweep,
        retained_threshold_sweep=retained_selection.sweep,
    )


def compare_complete_models(
    frozen: list[FoldSensitivity],
    qlora: list[FoldSensitivity],
    expected_aspects: Iterable[str],
) -> dict[str, object]:
    """Compare models only when both contain the exact complete fold set."""

    expected = list(expected_aspects)
    if len(expected) != 12 or len(set(expected)) != 12:
        raise ValueError("A full post-hoc comparison requires 12 unique canonical folds.")
    by_model: dict[str, dict[str, FoldSensitivity]] = {}
    for name, values in (("frozen", frozen), ("qlora", qlora)):
        mapping = {str(value.result["heldout_aspect"]): value for value in values}
        if len(mapping) != len(values) or set(mapping) != set(expected):
            raise ValueError(f"{name} sensitivity results do not match the complete fold set.")
        by_model[name] = mapping

    rows: list[dict[str, object]] = []
    for aspect in expected:
        frozen_result = by_model["frozen"][aspect].result
        qlora_result = by_model["qlora"][aspect].result
        for split in ("validation", "test"):
            if (
                frozen_result[f"{split}_score_identity_sha256"]
                != qlora_result[f"{split}_score_identity_sha256"]
            ):
                raise ValueError(f"Frozen and QLoRA {split} score identities differ for {aspect!r}.")
            for key in (
                f"{split}_rows_original",
                f"{split}_duplicate_rows_removed",
                f"{split}_rows_retained",
            ):
                if frozen_result[key] != qlora_result[key]:
                    raise ValueError(f"Frozen and QLoRA duplicate scopes differ at {key!r}.")
        frozen_f1 = float(frozen_result["retained_test_reselected_pair_micro_f1"])
        qlora_f1 = float(qlora_result["retained_test_reselected_pair_micro_f1"])
        rows.append(
            {
                "heldout_aspect": aspect,
                "frozen_retained_test_pair_micro_f1": frozen_f1,
                "qlora_retained_test_pair_micro_f1": qlora_f1,
                "qlora_minus_frozen_retained_test_pair_micro_f1": qlora_f1 - frozen_f1,
                "validation_duplicate_rows_removed": frozen_result[
                    "validation_duplicate_rows_removed"
                ],
                "test_duplicate_rows_removed": frozen_result["test_duplicate_rows_removed"],
            }
        )
    comparison = pd.DataFrame.from_records(rows)
    deltas = comparison["qlora_minus_frozen_retained_test_pair_micro_f1"].to_numpy(
        dtype=float
    )
    tolerance = 1e-12
    paired = {
        "folds": int(len(deltas)),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "sample_standard_deviation": float(np.std(deltas, ddof=1)),
        "minimum_delta": float(deltas.min()),
        "maximum_delta": float(deltas.max()),
        "wins": int(np.sum(deltas > tolerance)),
        "ties": int(np.sum(np.abs(deltas) <= tolerance)),
        "losses": int(np.sum(deltas < -tolerance)),
        "inferential_test_performed": False,
    }
    return {
        "analysis_classification": ANALYSIS_CLASSIFICATION,
        "not_confirmatory": True,
        "not_for_main_story": True,
        "per_fold": rows,
        "frozen_mean_retained_test_pair_micro_f1": float(
            comparison["frozen_retained_test_pair_micro_f1"].mean()
        ),
        "qlora_mean_retained_test_pair_micro_f1": float(
            comparison["qlora_retained_test_pair_micro_f1"].mean()
        ),
        "paired_descriptive_statistics": paired,
    }


def assert_experimental_output_dir(output_dir: Path, project_root: Path) -> Path:
    allowed = (project_root / "outputs" / "experimental").resolve()
    resolved = output_dir.resolve()
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Output must be under {allowed}.") from exc
    if not relative.parts:
        raise ValueError("Output must be a child directory of outputs/experimental.")
    return resolved
