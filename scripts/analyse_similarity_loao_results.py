from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "bow_count_1_2_train_vocab",
    "tfidf_char_3_5_train_vocab",
    "minilm_l6_v2",
    "e5_base_v2",
)
COMPARISONS = (
    ("e5_base_v2", "tfidf_char_3_5_train_vocab"),
    ("minilm_l6_v2", "tfidf_char_3_5_train_vocab"),
    ("e5_base_v2", "minilm_l6_v2"),
    ("tfidf_char_3_5_train_vocab", "bow_count_1_2_train_vocab"),
    ("e5_base_v2", "bow_count_1_2_train_vocab"),
    ("minilm_l6_v2", "bow_count_1_2_train_vocab"),
)


def exact_sign_flip_p_value(differences: np.ndarray) -> float:
    """Return the exact two-sided paired sign-flip p-value for the mean."""

    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional array")
    if not np.isfinite(differences).all():
        raise ValueError("differences must all be finite")

    observed = abs(float(differences.mean()))
    permuted = np.asarray(
        [
            np.mean(differences * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
        ]
    )
    return float(np.mean(np.abs(permuted) >= observed - 1e-12))


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile interval for the mean paired-aspect difference."""

    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional array")
    if not np.isfinite(differences).all():
        raise ValueError("differences must all be finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def holm_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjust p-values while preserving their original order."""

    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or p_values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional array")
    if not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must all be finite and between zero and one")

    order = np.argsort(p_values, kind="stable")
    sorted_values = p_values[order]
    adjusted_sorted = np.maximum.accumulate(
        (p_values.size - np.arange(p_values.size)) * sorted_values
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def validated_pivot(frame: pd.DataFrame, split: str, metric: str) -> pd.DataFrame:
    required = {"method", "split", "heldout_aspect", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{split} per-aspect CSV is missing columns: {missing}")

    selected = frame.loc[frame["split"] == split, list(required)].copy()
    if len(selected) != len(METHODS) * 12:
        raise ValueError(
            f"{split} must contain exactly {len(METHODS) * 12} registered per-aspect rows; "
            f"found {len(selected)}"
        )
    if selected.duplicated(["method", "heldout_aspect"]).any():
        raise ValueError(f"{split} contains duplicate method/aspect rows")
    observed_methods = set(selected["method"])
    if observed_methods != set(METHODS):
        raise ValueError(
            f"{split} methods do not match the registered set: {sorted(observed_methods)}"
        )

    pivot = selected.pivot(index="heldout_aspect", columns="method", values=metric)
    if pivot.shape != (12, len(METHODS)) or pivot.isna().any().any():
        raise ValueError(f"{split} methods do not share exactly twelve complete aspects")
    values = pivot.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{split} {metric} values must all be finite")
    return pivot.sort_index()


def comparison_table(
    frame: pd.DataFrame,
    *,
    split: str,
    metric: str = "pair_micro_f1",
    bootstrap_resamples: int = 100_000,
    seed: int = 13,
) -> pd.DataFrame:
    pivot = validated_pivot(frame, split, metric)
    rows: list[dict[str, object]] = []
    for challenger, reference in COMPARISONS:
        challenger_values = pivot[challenger].to_numpy(dtype=float)
        reference_values = pivot[reference].to_numpy(dtype=float)
        differences = challenger_values - reference_values
        lower, upper = paired_bootstrap_interval(
            differences,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        tolerance = 1e-12
        rows.append(
            {
                "split": split,
                "metric": metric,
                "challenger": challenger,
                "reference": reference,
                "aspects": int(differences.size),
                "challenger_mean": float(challenger_values.mean()),
                "reference_mean": float(reference_values.mean()),
                "mean_paired_difference": float(differences.mean()),
                "median_paired_difference": float(np.median(differences)),
                "bootstrap_95_ci_lower": lower,
                "bootstrap_95_ci_upper": upper,
                "exact_sign_flip_p_value_two_sided": exact_sign_flip_p_value(differences),
                "challenger_wins": int(np.sum(differences > tolerance)),
                "ties": int(np.sum(np.abs(differences) <= tolerance)),
                "challenger_losses": int(np.sum(differences < -tolerance)),
                "bootstrap_resamples": bootstrap_resamples,
                "seed": seed,
            }
        )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value_within_six_comparisons"] = holm_adjusted_p_values(
        result["exact_sign_flip_p_value_two_sided"].to_numpy(dtype=float)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired-aspect uncertainty tables for frozen similarity LOAO results."
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=PROJECT_ROOT
        / "docs"
        / "thesis_figure_data"
        / "loao_bow_sentence_embedding_v1_validation_per_aspect.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=PROJECT_ROOT
        / "docs"
        / "thesis_figure_data"
        / "loao_bow_sentence_embedding_v1_test_per_aspect.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT
        / "docs"
        / "thesis_figure_data"
        / "loao_bow_sentence_embedding_v1_paired_comparisons.csv",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = pd.read_csv(args.validation_csv)
    test = pd.read_csv(args.test_csv)
    table = pd.concat(
        [
            comparison_table(
                validation,
                split="validation",
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed,
            ),
            comparison_table(
                test,
                split="test",
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed,
            ),
        ],
        ignore_index=True,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
