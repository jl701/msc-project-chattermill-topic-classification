"""Run the preregistered TF-IDF/E5 multi-sentiment decoder ablation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    canonical_aspects,
    registered_folds,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    evaluate_prediction_mask,
    multi_sentiment_prediction_mask,
    select_multi_sentiment_thresholds,
    select_top_k_aspect_threshold,
    select_two_stage_threshold,
    top_k_sentiment_prediction_mask,
    two_stage_prediction_mask,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    TfidfTrueTwoStageRuntime,
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
)
from run_taxonomy_two_stage_validation import E5ScoreCache  # noqa: E402


METHODS = ("strict_train_only_tfidf", "e5_base_v2")
VARIANT = "name_and_description"
SENTIMENT_QUANTILES = 65


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _decoder_metrics(
    scored: pd.DataFrame,
    mask: pd.Series,
    *,
    aspects: tuple[str, ...],
) -> dict[str, object]:
    subset = scored[scored["candidate_aspect"].astype(str).isin(aspects)].copy()
    subset_mask = pd.Series(mask, index=scored.index, dtype=bool).loc[subset.index]
    metrics = evaluate_prediction_mask(
        subset,
        subset_mask.to_numpy(dtype=bool),
        aspects=aspects,
    )
    predicted = subset.loc[subset_mask]
    predicted_counts = predicted.groupby(
        ["row_uid", "candidate_aspect"], sort=False
    ).size()
    selected_aspects = int(len(predicted_counts))
    multi_selected = int(predicted_counts.gt(1).sum())

    gold_counts = (
        subset[subset["target"].astype(int).eq(1)]
        .groupby(["row_uid", "candidate_aspect"], sort=False)
        .size()
    )
    multi_gold_keys = set(gold_counts[gold_counts.gt(1)].index)
    in_multi_gold = pd.Series(
        [
            (str(uid), str(aspect)) in multi_gold_keys
            for uid, aspect in zip(subset["row_uid"], subset["candidate_aspect"])
        ],
        index=subset.index,
        dtype=bool,
    )
    multi_gold_labels = int(
        (in_multi_gold & subset["target"].astype(int).eq(1)).sum()
    )
    recovered_multi_gold = int(
        (
            in_multi_gold
            & subset["target"].astype(int).eq(1)
            & subset_mask
        ).sum()
    )
    return {
        "metrics": metrics,
        "selected_aspect_instances": selected_aspects,
        "sentiments_per_selected_aspect": (
            float(len(predicted) / selected_aspects) if selected_aspects else 0.0
        ),
        "multi_sentiment_selected_aspect_rate": (
            float(multi_selected / selected_aspects) if selected_aspects else 0.0
        ),
        "multi_gold_aspect_instances": int(len(multi_gold_keys)),
        "multi_gold_sentiment_labels": multi_gold_labels,
        "recovered_multi_gold_sentiment_labels": recovered_multi_gold,
        "multi_gold_sentiment_recall": (
            float(recovered_multi_gold / multi_gold_labels)
            if multi_gold_labels
            else 0.0
        ),
    }


def _assert_parent_control(
    parent_path: Path,
    observed: dict[str, object],
) -> None:
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent Study C fold is missing: {parent_path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    expected = parent["variants"][VARIANT]["partitions"]["heldout"]["metrics"]
    actual = observed["metrics"]
    for key in ("pair_micro_f1", "pair_micro_precision", "pair_micro_recall"):
        if not np.isclose(
            float(actual[key]), float(expected[key]), atol=1e-12, rtol=0.0
        ):
            raise AssertionError(
                f"Argmax control failed to reproduce parent {key}: "
                f"observed={actual[key]} expected={expected[key]}"
            )


def _aggregate(folds: list[dict[str, object]]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for fold in folds:
        control_f1 = float(
            fold["decoders"]["argmax"]["heldout"]["metrics"]["pair_micro_f1"]
        )
        for decoder, result in fold["decoders"].items():
            heldout = result["heldout"]
            metrics = heldout["metrics"]
            records.append(
                {
                    "fold_id": fold["fold_id"],
                    "heldout_aspect": fold["heldout_aspect"],
                    "decoder": decoder,
                    "pair_micro_f1": metrics["pair_micro_f1"],
                    "pair_micro_precision": metrics["pair_micro_precision"],
                    "pair_micro_recall": metrics["pair_micro_recall"],
                    "pair_micro_f1_delta_vs_argmax": (
                        float(metrics["pair_micro_f1"]) - control_f1
                    ),
                    "sentiments_per_selected_aspect": heldout[
                        "sentiments_per_selected_aspect"
                    ],
                    "multi_sentiment_selected_aspect_rate": heldout[
                        "multi_sentiment_selected_aspect_rate"
                    ],
                    "multi_gold_sentiment_labels": heldout[
                        "multi_gold_sentiment_labels"
                    ],
                    "recovered_multi_gold_sentiment_labels": heldout[
                        "recovered_multi_gold_sentiment_labels"
                    ],
                }
            )
    frame = pd.DataFrame.from_records(records)
    aggregate: list[dict[str, object]] = []
    for decoder, group in frame.groupby("decoder", sort=True):
        deltas = group["pair_micro_f1_delta_vs_argmax"].to_numpy(dtype=float)
        multi_gold = int(group["multi_gold_sentiment_labels"].sum())
        recovered = int(group["recovered_multi_gold_sentiment_labels"].sum())
        aggregate.append(
            {
                "decoder": str(decoder),
                "folds": int(group["fold_id"].nunique()),
                "pair_micro_f1_mean": float(group["pair_micro_f1"].mean()),
                "pair_micro_precision_mean": float(
                    group["pair_micro_precision"].mean()
                ),
                "pair_micro_recall_mean": float(group["pair_micro_recall"].mean()),
                "pair_micro_f1_delta_vs_argmax_mean": float(deltas.mean()),
                "wins_ties_losses_vs_argmax": {
                    "wins": int((deltas > 1e-12).sum()),
                    "ties": int((np.abs(deltas) <= 1e-12).sum()),
                    "losses": int((deltas < -1e-12).sum()),
                },
                "sentiments_per_selected_aspect_mean": float(
                    group["sentiments_per_selected_aspect"].mean()
                ),
                "multi_sentiment_selected_aspect_rate_mean": float(
                    group["multi_sentiment_selected_aspect_rate"].mean()
                ),
                "multi_gold_sentiment_labels": multi_gold,
                "recovered_multi_gold_sentiment_labels": recovered,
                "multi_gold_sentiment_recall": (
                    float(recovered / multi_gold) if multi_gold else 0.0
                ),
            }
        )
    return {
        "records": frame.to_dict(orient="records"),
        "aggregate": aggregate,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.resolve()
    method_root = output_root / args.method
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    validation_rows = frame[frame["original_split"].eq("validation")].copy()
    resource = load_description_bundle(require_approved=True)
    aspects = canonical_aspects()
    folds = registered_folds("L2")
    if args.fold_id:
        folds = [fold for fold in folds if fold.fold_id == args.fold_id]
        if not folds:
            raise ValueError(f"Unknown L2 fold: {args.fold_id!r}")
    pending = any(
        not (
            args.resume
            and (method_root / "folds" / f"{fold.fold_id}.json").is_file()
        )
        for fold in folds
    )
    e5_cache = (
        E5ScoreCache(validation_rows, local_files_only=args.local_files_only)
        if args.method == "e5_base_v2" and pending
        else None
    )
    fold_results: list[dict[str, object]] = []
    try:
        for fold in folds:
            result_path = method_root / "folds" / f"{fold.fold_id}.json"
            if result_path.is_file() and args.resume:
                fold_results.append(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
                print(f"RESUME {args.method} {fold.fold_id}", flush=True)
                continue
            started = time.perf_counter()
            splits = build_taxonomy_fold_splits(
                frame,
                fold,
                evaluation_splits=("validation",),
            )
            train_rows = splits["train"]
            eval_rows = splits["validation"]
            variants = {aspect: VARIANT for aspect in aspects}
            if args.method == "strict_train_only_tfidf":
                train_variants = {aspect: VARIANT for aspect in fold.seen_aspects}
                runtime = TfidfTrueTwoStageRuntime(seed=13).fit(
                    build_aspect_grid(
                        train_rows, fold.seen_aspects, train_variants, resource
                    ),
                    build_sentiment_grid(
                        train_rows,
                        fold.seen_aspects,
                        train_variants,
                        resource,
                        gold_aspects_only=True,
                    ),
                )
            else:
                runtime = None
            aspect_grid = build_aspect_grid(eval_rows, aspects, variants, resource)
            sentiment_grid = build_sentiment_grid(
                eval_rows, aspects, variants, resource
            )
            if runtime is not None:
                aspect_scores = runtime.score_aspects(aspect_grid)
                sentiment_scores = runtime.score_sentiments(sentiment_grid)
            else:
                assert e5_cache is not None
                aspect_scores = e5_cache.score(aspect_grid)
                sentiment_scores = e5_cache.score(sentiment_grid)
            scored = join_two_stage_scores(
                aspect_grid,
                sentiment_grid,
                aspect_scores,
                sentiment_scores,
            )
            seen = scored[
                scored["candidate_aspect"].astype(str).isin(fold.seen_aspects)
            ].copy()

            control_selection = select_two_stage_threshold(seen)
            multi_selection = select_multi_sentiment_thresholds(
                seen,
                sentiment_quantiles=SENTIMENT_QUANTILES,
            )
            top2_selection = select_top_k_aspect_threshold(seen, top_k=2)
            control_mask = two_stage_prediction_mask(
                scored, control_selection.threshold
            )
            assert multi_selection.sentiment_threshold is not None
            multi_mask = multi_sentiment_prediction_mask(
                scored,
                aspect_threshold=multi_selection.aspect_threshold,
                sentiment_threshold=multi_selection.sentiment_threshold,
            )
            top2_mask = top_k_sentiment_prediction_mask(
                scored,
                aspect_threshold=top2_selection.aspect_threshold,
                top_k=2,
            )
            decoders = {
                "argmax": {
                    "selection": {
                        "aspect_threshold": control_selection.threshold,
                        "selection_partition": "seen_validation",
                    },
                    "heldout": _decoder_metrics(
                        scored,
                        control_mask,
                        aspects=fold.heldout_aspects,
                    ),
                },
                "multi_threshold": {
                    "selection": {
                        "aspect_threshold": multi_selection.aspect_threshold,
                        "sentiment_threshold": multi_selection.sentiment_threshold,
                        "selection_metrics": multi_selection.selection_metrics,
                        "candidates_evaluated": multi_selection.candidates_evaluated,
                        "sentiment_thresholds_evaluated": (
                            multi_selection.sentiment_thresholds_evaluated
                        ),
                        "selection_partition": "seen_validation",
                    },
                    "heldout": _decoder_metrics(
                        scored,
                        multi_mask,
                        aspects=fold.heldout_aspects,
                    ),
                },
                "forced_top2": {
                    "selection": {
                        "aspect_threshold": top2_selection.aspect_threshold,
                        "selection_metrics": top2_selection.selection_metrics,
                        "candidates_evaluated": top2_selection.candidates_evaluated,
                        "selection_partition": "seen_validation",
                    },
                    "heldout": _decoder_metrics(
                        scored,
                        top2_mask,
                        aspects=fold.heldout_aspects,
                    ),
                },
            }
            parent_path = (
                args.parent_output_root
                / "study_c"
                / args.method
                / "folds"
                / f"{fold.fold_id}.json"
            )
            _assert_parent_control(parent_path, decoders["argmax"]["heldout"])
            result: dict[str, object] = {
                "schema_version": "taxonomy_multi_sentiment_decoder_fold_v1",
                "protocol_id": "taxonomy_multi_sentiment_decoder_validation_v1",
                "method_id": args.method,
                "fold_id": fold.fold_id,
                "heldout_aspect": fold.heldout_aspects[0],
                "representation": VARIANT,
                "train_rows": int(len(train_rows)),
                "validation_rows": int(len(eval_rows)),
                "test_contract_count": 0,
                "decoders": decoders,
                "seconds": time.perf_counter() - started,
            }
            _write_json(result_path, result)
            fold_results.append(result)
            print(
                "COMPLETE "
                f"{args.method} {fold.fold_id} "
                "argmax="
                f"{decoders['argmax']['heldout']['metrics']['pair_micro_f1']:.6f} "
                "multi="
                f"{decoders['multi_threshold']['heldout']['metrics']['pair_micro_f1']:.6f} "
                "top2="
                f"{decoders['forced_top2']['heldout']['metrics']['pair_micro_f1']:.6f}",
                flush=True,
            )
    finally:
        if e5_cache is not None:
            e5_cache.close()

    aggregate = _aggregate(fold_results)
    summary: dict[str, object] = {
        "schema_version": "taxonomy_multi_sentiment_decoder_summary_v1",
        "protocol_id": "taxonomy_multi_sentiment_decoder_validation_v1",
        "method_id": args.method,
        "representation": VARIANT,
        "completed_folds": len(fold_results),
        "failed_folds": 0,
        "test_contract_count": 0,
        "selection_partition": "seen validation only",
        "evaluation_partition": "held-out validation only",
        **aggregate,
    }
    _write_json(method_root / "summary.json", summary)
    pd.DataFrame.from_records(aggregate["records"]).to_csv(
        method_root / "per_fold.csv", index=False
    )
    print(json.dumps(_jsonable(summary["aggregate"]), indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_multi_sentiment_decoder_validation_v1",
    )
    parser.add_argument(
        "--parent-output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_two_stage_validation_v1",
    )
    parser.add_argument("--fold-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
