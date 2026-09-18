"""Create the final validation-only scientific-freeze comparison tables."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.taxonomy_scientific_freeze import (  # noqa: E402
    pair_micro_f1_from_counts,
)


METHODS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "description_conditioned_weight_transfer_kernel_ridge",
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_candidate_pair",
    "frozen_qwen_few_shot",
    "qwen_candidate_pair_qlora",
)
CONDITIONS = ("N", "D")
FOLDS = tuple(f"l2-a{index:02d}" for index in range(1, 13))
COUNT_COLUMNS = ("pair_tp", "pair_fp", "pair_fn")
L4_PARENT_GROUPS = {
    "l4-g01": "Company brand",
    "l4-g02": "Staff support",
    "l4-g03": "Value",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _load_l2_evidence(formal_root: Path, local_root: Path) -> pd.DataFrame:
    paths = list(formal_root.glob("*/L2/*/[ND].csv"))
    paths.extend(local_root.glob("*/*/[ND].csv"))
    frames = [pd.read_csv(path) for path in sorted(paths)]
    if not frames:
        raise FileNotFoundError("No L2 row evidence was found.")
    evidence = pd.concat(frames, ignore_index=True)
    identities = evidence[
        ["method_id", "fold_id", "condition", "row_uid"]
    ]
    if identities.duplicated().any():
        raise ValueError("Duplicate row-evidence identity.")
    expected = {
        (method, fold, condition)
        for method in METHODS
        for fold in FOLDS
        for condition in CONDITIONS
    }
    observed = set(
        evidence[["method_id", "fold_id", "condition"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(
            f"Incomplete seven-method L2 evidence; missing={sorted(expected-observed)}; "
            f"extra={sorted(observed-expected)}"
        )
    uid_sets = {
        key: frozenset(group["row_uid"].astype(str))
        for key, group in evidence.groupby(
            ["method_id", "fold_id", "condition"], sort=False
        )
    }
    first = next(iter(uid_sets.values()))
    if len(first) != 1057 or any(values != first for values in uid_sets.values()):
        raise ValueError("Review identities are not exactly paired across all methods/folds.")
    return evidence


def _evidence_cube(
    evidence: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    row_uids = tuple(sorted(evidence["row_uid"].astype(str).unique()))
    cube = np.zeros(
        (len(METHODS), len(CONDITIONS), len(FOLDS), len(row_uids), 3),
        dtype=np.int16,
    )
    for method_index, method in enumerate(METHODS):
        for condition_index, condition in enumerate(CONDITIONS):
            for fold_index, fold in enumerate(FOLDS):
                group = evidence[
                    evidence["method_id"].eq(method)
                    & evidence["condition"].eq(condition)
                    & evidence["fold_id"].eq(fold)
                ].copy()
                group["row_uid"] = group["row_uid"].astype(str)
                group = group.set_index("row_uid").loc[list(row_uids)]
                cube[method_index, condition_index, fold_index] = group[
                    list(COUNT_COLUMNS)
                ].to_numpy(dtype=np.int16)
    return cube, row_uids


def _point_metrics(counts: np.ndarray) -> tuple[float, float, np.ndarray]:
    fold_counts = counts.sum(axis=1, dtype=np.int64)
    fold_f1 = pair_micro_f1_from_counts(
        fold_counts[:, 0], fold_counts[:, 1], fold_counts[:, 2]
    )
    pooled = counts.sum(axis=(0, 1), dtype=np.int64)
    pooled_f1 = float(pair_micro_f1_from_counts(*pooled))
    return float(np.mean(fold_f1)), pooled_f1, np.asarray(fold_f1, dtype=float)


def _synchronised_bootstrap(
    cube: np.ndarray,
    *,
    draws: int,
    seed: int,
    batch_size: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Use one review-cluster resample for every method/fold/N-D comparison."""

    rng = np.random.default_rng(seed)
    method_count, condition_count, fold_count, review_count, _ = cube.shape
    balanced = np.empty((draws, method_count, condition_count), dtype=float)
    pooled = np.empty_like(balanced)
    probabilities = np.full(review_count, 1.0 / review_count)
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        weights = rng.multinomial(review_count, probabilities, size=stop - start)
        counts = np.einsum("br,mcfrk->bmcfk", weights, cube, optimize=True)
        fold_f1 = pair_micro_f1_from_counts(
            counts[..., 0], counts[..., 1], counts[..., 2]
        )
        balanced[start:stop] = fold_f1.mean(axis=3)
        pooled_counts = counts.sum(axis=3)
        pooled[start:stop] = pair_micro_f1_from_counts(
            pooled_counts[..., 0], pooled_counts[..., 1], pooled_counts[..., 2]
        )
    return balanced, pooled


def _interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _fold_sign_sensitivity(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    distribution = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(distribution) >= observed - 1e-15))


def _holm(values: pd.Series) -> pd.Series:
    ordered = values.sort_values(kind="stable")
    adjusted = pd.Series(index=values.index, dtype=float)
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted.loc[index] = running
    return adjusted


def _build_l2_tables(
    cube: np.ndarray,
    balanced_boot: np.ndarray,
    pooled_boot: np.ndarray,
    *,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    effects = []
    point_balanced = np.zeros((len(METHODS), len(CONDITIONS)))
    point_pooled = np.zeros_like(point_balanced)
    fold_points: dict[tuple[int, int], np.ndarray] = {}
    for method_index, method in enumerate(METHODS):
        for condition_index, condition in enumerate(CONDITIONS):
            balanced, pooled, per_fold = _point_metrics(
                cube[method_index, condition_index]
            )
            point_balanced[method_index, condition_index] = balanced
            point_pooled[method_index, condition_index] = pooled
            fold_points[(method_index, condition_index)] = per_fold
            balanced_low, balanced_high = _interval(
                balanced_boot[:, method_index, condition_index]
            )
            pooled_low, pooled_high = _interval(
                pooled_boot[:, method_index, condition_index]
            )
            summaries.append(
                {
                    "method_id": method,
                    "condition": condition,
                    "fold_count": 12,
                    "validation_review_clusters": 1057,
                    "aspect_balanced_heldout_pair_micro_f1": balanced,
                    "aspect_balanced_review_bootstrap_95_ci_low": balanced_low,
                    "aspect_balanced_review_bootstrap_95_ci_high": balanced_high,
                    "pooled_heldout_pair_micro_f1_sensitivity": pooled,
                    "pooled_review_bootstrap_95_ci_low": pooled_low,
                    "pooled_review_bootstrap_95_ci_high": pooled_high,
                    "bootstrap_draws": len(balanced_boot),
                    "bootstrap_seed": bootstrap_seed,
                    "inference_scope": (
                        "conditional on one trained realisation per neural system "
                        "and the fixed 12-aspect taxonomy"
                    ),
                }
            )
        balanced_delta = (
            balanced_boot[:, method_index, 1]
            - balanced_boot[:, method_index, 0]
        )
        pooled_delta = (
            pooled_boot[:, method_index, 1] - pooled_boot[:, method_index, 0]
        )
        balanced_low, balanced_high = _interval(balanced_delta)
        pooled_low, pooled_high = _interval(pooled_delta)
        fold_delta = fold_points[(method_index, 1)] - fold_points[(method_index, 0)]
        effects.append(
            {
                "method_id": method,
                "primary_estimand": "aspect-balanced heldout pair micro-F1",
                "N": point_balanced[method_index, 0],
                "D": point_balanced[method_index, 1],
                "D_minus_N": (
                    point_balanced[method_index, 1]
                    - point_balanced[method_index, 0]
                ),
                "paired_review_bootstrap_95_ci_low": balanced_low,
                "paired_review_bootstrap_95_ci_high": balanced_high,
                "pooled_N_sensitivity": point_pooled[method_index, 0],
                "pooled_D_sensitivity": point_pooled[method_index, 1],
                "pooled_D_minus_N_sensitivity": (
                    point_pooled[method_index, 1]
                    - point_pooled[method_index, 0]
                ),
                "pooled_paired_review_bootstrap_95_ci_low": pooled_low,
                "pooled_paired_review_bootstrap_95_ci_high": pooled_high,
                "improved_folds": int(np.count_nonzero(fold_delta > 1e-15)),
                "tied_folds": int(np.count_nonzero(np.abs(fold_delta) <= 1e-15)),
                "worsened_folds": int(np.count_nonzero(fold_delta < -1e-15)),
                "enumerated_fold_sign_sensitivity_p": _fold_sign_sensitivity(
                    fold_delta
                ),
            }
        )
    effects_frame = pd.DataFrame(effects)
    effects_frame["holm_adjusted_across_all_7_methods"] = _holm(
        effects_frame["enumerated_fold_sign_sensitivity_p"]
    )

    pairwise = []
    for left_index, right_index in itertools.combinations(range(len(METHODS)), 2):
        difference = (
            balanced_boot[:, left_index, 1] - balanced_boot[:, right_index, 1]
        )
        low, high = _interval(difference)
        pairwise.append(
            {
                "condition": "D",
                "left_method": METHODS[left_index],
                "right_method": METHODS[right_index],
                "left_minus_right_aspect_balanced_heldout_pair_micro_f1": (
                    point_balanced[left_index, 1] - point_balanced[right_index, 1]
                ),
                "shared_review_bootstrap_95_ci_low": low,
                "shared_review_bootstrap_95_ci_high": high,
                "interpretation": "descriptive systems comparison; not a causal adaptation effect",
            }
        )
    return pd.DataFrame(summaries), effects_frame, pd.DataFrame(pairwise)


def _build_l4_table(formal_root: Path, *, draws: int, seed: int) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(formal_root.glob("*/L4/*/D.csv"))]
    evidence = pd.concat(frames, ignore_index=True)
    rows = []
    rng = np.random.default_rng(seed)
    for (method, fold), group in evidence.groupby(["method_id", "fold_id"], sort=True):
        group = group.sort_values("row_uid", kind="stable")
        counts = group[list(COUNT_COLUMNS)].to_numpy(dtype=np.int64)
        point_counts = counts.sum(axis=0)
        point = float(pair_micro_f1_from_counts(*point_counts))
        weights = rng.multinomial(
            len(group), np.full(len(group), 1.0 / len(group)), size=draws
        )
        boot_counts = weights @ counts
        values = pair_micro_f1_from_counts(
            boot_counts[:, 0], boot_counts[:, 1], boot_counts[:, 2]
        )
        low, high = _interval(values)
        rows.append(
            {
                "method_id": method,
                "parent_group_fold": fold,
                "heldout_parent_group": L4_PARENT_GROUPS[str(fold)],
                "condition": "D",
                "validation_reviews": len(group),
                "positive_review_support": int((group["pair_gold_count"] > 0).sum()),
                "heldout_pair_micro_f1": point,
                "review_bootstrap_95_ci_low": low,
                "review_bootstrap_95_ci_high": high,
                "bootstrap_draws": draws,
                "bootstrap_seed": seed,
                "cross_group_p_value_reported": False,
            }
        )
    return pd.DataFrame(rows)


def _build_fold_error_diagnostics(cube: np.ndarray) -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(METHODS):
        for condition_index, condition in enumerate(CONDITIONS):
            for fold_index, fold in enumerate(FOLDS):
                per_review = cube[method_index, condition_index, fold_index].astype(
                    np.int64
                )
                tp, fp, fn = per_review.sum(axis=0)
                precision = float(tp / (tp + fp)) if tp + fp else 0.0
                recall = float(tp / (tp + fn)) if tp + fn else 0.0
                f1 = float(pair_micro_f1_from_counts(tp, fp, fn))
                review_count = len(per_review)
                rows.append(
                    {
                        "method_id": method,
                        "fold_id": fold,
                        "condition": condition,
                        "validation_reviews": review_count,
                        "heldout_pair_tp": int(tp),
                        "heldout_pair_fp": int(fp),
                        "heldout_pair_fn": int(fn),
                        "heldout_pair_precision": precision,
                        "heldout_pair_recall": recall,
                        "heldout_pair_micro_f1": f1,
                        "heldout_pair_gold_label_count": int(tp + fn),
                        "heldout_pair_predicted_label_count": int(tp + fp),
                        "false_positive_rows_per_100": float(
                            100.0 * np.count_nonzero(per_review[:, 1]) / review_count
                        ),
                        "false_negative_rows_per_100": float(
                            100.0 * np.count_nonzero(per_review[:, 2]) / review_count
                        ),
                        "mean_heldout_prediction_set_size": float(
                            np.mean(per_review[:, 0] + per_review[:, 1])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence = _load_l2_evidence(
        args.formal_evidence_root.resolve(), args.local_evidence_root.resolve()
    )
    cube, row_uids = _evidence_cube(evidence)
    balanced_boot, pooled_boot = _synchronised_bootstrap(
        cube, draws=args.bootstrap_draws, seed=args.bootstrap_seed
    )
    summary, effects, pairwise = _build_l2_tables(
        cube,
        balanced_boot,
        pooled_boot,
        bootstrap_seed=args.bootstrap_seed,
    )
    l4 = _build_l4_table(
        args.formal_evidence_root.resolve(),
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )
    diagnostics = _build_fold_error_diagnostics(cube)
    tables = {
        "l2_model_condition_review_bootstrap.csv": summary,
        "l2_description_effects_review_bootstrap.csv": effects,
        "l2_pairwise_model_differences_review_bootstrap.csv": pairwise,
        "l4_parent_group_review_bootstrap.csv": l4,
        "l2_fold_pair_error_diagnostics.csv": diagnostics,
    }
    hashes = {}
    for name, frame in tables.items():
        path = args.table_dir.resolve() / name
        _atomic_csv(path, frame)
        hashes[name] = _sha256(path)
    audit = {
        "schema_version": "taxonomy_scientific_freeze_inference_v1",
        "status": "pass",
        "allowed_splits": ["train", "validation"],
        "include_official_test": False,
        "official_test_opened": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "primary_estimand": (
            "unweighted mean across the 12 held-out-aspect fold pair micro-F1 values"
        ),
        "sensitivity_estimand": "pooled held-out pair micro-F1 across all folds",
        "bootstrap_unit": "row_uid review cluster",
        "synchronisation": (
            "the same review-cluster resample is used for every method, fold, and N/D condition"
        ),
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "validation_review_clusters": len(row_uids),
        "validation_row_uid_sha256": hashlib.sha256(
            ("\n".join(row_uids) + "\n").encode("utf-8")
        ).hexdigest(),
        "method_count": len(METHODS),
        "methods": list(METHODS),
        "fold_pair_error_diagnostic_rows": len(diagnostics),
        "holm_family": (
            "all seven methods for the enumerated fold-sign sensitivity "
            "associated with the primary D-minus-N metric"
        ),
        "fold_sign_test_label": "enumerated fold sign sensitivity",
        "conditional_inference_scope": (
            "one trained realisation per neural system and the fixed 12-aspect taxonomy"
        ),
        "table_sha256": hashes,
    }
    _write_json(args.audit_output.resolve(), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-evidence-root",
        type=Path,
        default=Path(
            "outputs/experimental/taxonomy_scientific_freeze_v1/formal_row_evidence"
        ),
    )
    parser.add_argument(
        "--local-evidence-root",
        type=Path,
        default=Path(
            "outputs/experimental/taxonomy_scientific_freeze_v1/local_row_evidence"
        ),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("docs/thesis_figure_data/taxonomy_scientific_freeze_v1"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "docs/experiments/taxonomy_scientific_freeze_inference_audit_20260822.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
