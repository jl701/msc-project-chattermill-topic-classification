"""Run the preregistered matched one-stage versus two-stage validation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for source in (SRC_ROOT, SCRIPTS_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_execution import (  # noqa: E402
    dataframe_sha256,
)
from msc_project.experiments.taxonomy_methods import (  # noqa: E402
    StrictTfidfRuntime,
    resolve_method_spec,
    tfidf_config_from_parameters,
)
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_budgeted_training_manifest,
    build_taxonomy_fold_splits,
    canonical_aspects,
    pair_identity_hash,
    registered_folds,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    capped_two_sentiment_prediction_mask,
    pair_prediction_mask,
    select_pair_threshold,
    select_second_sentiment_threshold,
    select_two_stage_threshold,
    two_stage_prediction_mask,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    TfidfTrueTwoStageRuntime,
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
    render_sentiment_candidate,
)
from msc_project.experiments.unified_candidate_pairs import (  # noqa: E402
    manifest_hash,
)
from run_taxonomy_multi_sentiment_decoder_validation import (  # noqa: E402
    _decoder_metrics,
    _jsonable,
    _write_json,
)
from run_taxonomy_two_stage_validation import E5ScoreCache  # noqa: E402


METHODS = ("strict_train_only_tfidf", "e5_base_v2")
DECODERS = (
    "one_stage_independent_pairs",
    "two_stage_argmax",
    "two_stage_capped_two",
)
VARIANT = "name_and_description"
PROTOCOL_ID = "taxonomy_matched_one_vs_two_stage_validation_v1"
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_matched_one_vs_two_stage_validation_v1.json"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rerender_one_stage_training_manifest(
    manifest: pd.DataFrame,
    resource: Mapping[str, object],
) -> pd.DataFrame:
    """Keep the original sampled identities but enforce matched candidate text."""

    required = {"candidate_aspect", "candidate_sentiment", "candidate_text"}
    missing = sorted(required - set(manifest.columns))
    if missing or manifest.empty:
        raise ValueError(f"One-stage training manifest is invalid; missing={missing}.")
    result = manifest.copy()
    result["candidate_text"] = [
        render_sentiment_candidate(
            str(aspect), str(sentiment), VARIANT, resource
        )
        for aspect, sentiment in zip(
            result["candidate_aspect"], result["candidate_sentiment"]
        )
    ]
    result["representation_variant"] = VARIANT
    if not result["candidate_text"].astype(str).str.contains(
        "Aspect: ", regex=False
    ).all() or not result["candidate_text"].astype(str).str.contains(
        "Definition: ", regex=False
    ).all():
        raise AssertionError("Matched one-stage candidates lack name or description.")
    return result


def _assert_matched_evaluation_grids(
    one_stage: pd.DataFrame,
    two_stage: pd.DataFrame,
) -> None:
    columns = [
        "row_uid",
        "candidate_aspect",
        "candidate_sentiment",
        "candidate_text",
        "target",
    ]
    left = one_stage[columns].sort_values(columns[:3], kind="stable").reset_index(
        drop=True
    )
    right = two_stage[columns].sort_values(columns[:3], kind="stable").reset_index(
        drop=True
    )
    if not left.equals(right):
        raise AssertionError(
            "One-stage and two-stage evaluation grids are not matched."
        )


def _selected_aspects(
    scored: pd.DataFrame, mask: pd.Series
) -> set[tuple[str, str]]:
    selected = scored.loc[pd.Series(mask, index=scored.index, dtype=bool)]
    return set(
        zip(
            selected["row_uid"].astype(str),
            selected["candidate_aspect"].astype(str),
        )
    )


def _sentiment_cardinality(
    scored: pd.DataFrame, mask: pd.Series
) -> dict[str, int]:
    selected = scored.loc[pd.Series(mask, index=scored.index, dtype=bool)]
    counts = selected.groupby(["row_uid", "candidate_aspect"]).size()
    return {
        "minimum": int(counts.min()) if len(counts) else 0,
        "maximum": int(counts.max()) if len(counts) else 0,
    }


def _score_hash(frame: pd.DataFrame, score_columns: list[str]) -> str:
    columns = [
        "row_uid",
        "candidate_aspect",
        "candidate_sentiment",
        "target",
        "candidate_text",
        *score_columns,
    ]
    return dataframe_sha256(
        frame,
        columns,
        sort_columns=["row_uid", "candidate_aspect", "candidate_sentiment"],
    )


def _aggregate(folds: list[dict[str, object]]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for fold in folds:
        decoders = fold["decoders"]  # type: ignore[assignment]
        one_stage_f1 = float(
            decoders["one_stage_independent_pairs"]["heldout"]["metrics"][
                "pair_micro_f1"
            ]  # type: ignore[index]
        )
        argmax_f1 = float(
            decoders["two_stage_argmax"]["heldout"]["metrics"][
                "pair_micro_f1"
            ]  # type: ignore[index]
        )
        for decoder in DECODERS:
            heldout = decoders[decoder]["heldout"]  # type: ignore[index]
            metrics = heldout["metrics"]
            records.append(
                {
                    "fold_id": fold["fold_id"],
                    "heldout_aspect": fold["heldout_aspect"],
                    "decoder": decoder,
                    "pair_micro_f1": metrics["pair_micro_f1"],
                    "pair_micro_precision": metrics["pair_micro_precision"],
                    "pair_micro_recall": metrics["pair_micro_recall"],
                    "aspect_micro_f1": metrics["aspect_micro_f1"],
                    "pair_micro_f1_delta_vs_one_stage": (
                        float(metrics["pair_micro_f1"]) - one_stage_f1
                    ),
                    "pair_micro_f1_delta_vs_two_stage_argmax": (
                        float(metrics["pair_micro_f1"]) - argmax_f1
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
    for decoder in DECODERS:
        group = frame[frame["decoder"].eq(decoder)].copy()
        delta_one = group["pair_micro_f1_delta_vs_one_stage"].to_numpy(dtype=float)
        delta_argmax = group[
            "pair_micro_f1_delta_vs_two_stage_argmax"
        ].to_numpy(dtype=float)
        multi_gold = int(group["multi_gold_sentiment_labels"].sum())
        recovered = int(group["recovered_multi_gold_sentiment_labels"].sum())
        aggregate.append(
            {
                "decoder": decoder,
                "folds": int(group["fold_id"].nunique()),
                "pair_micro_f1_mean": float(group["pair_micro_f1"].mean()),
                "pair_micro_precision_mean": float(
                    group["pair_micro_precision"].mean()
                ),
                "pair_micro_recall_mean": float(group["pair_micro_recall"].mean()),
                "aspect_micro_f1_mean": float(group["aspect_micro_f1"].mean()),
                "pair_micro_f1_delta_vs_one_stage_mean": float(delta_one.mean()),
                "pair_micro_f1_delta_vs_two_stage_argmax_mean": float(
                    delta_argmax.mean()
                ),
                "wins_ties_losses_vs_one_stage": {
                    "wins": int((delta_one > 1e-12).sum()),
                    "ties": int((np.abs(delta_one) <= 1e-12).sum()),
                    "losses": int((delta_one < -1e-12).sum()),
                },
                "wins_ties_losses_vs_two_stage_argmax": {
                    "wins": int((delta_argmax > 1e-12).sum()),
                    "ties": int((np.abs(delta_argmax) <= 1e-12).sum()),
                    "losses": int((delta_argmax < -1e-12).sum()),
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
    if args.method not in METHODS:
        raise ValueError(f"Unsupported method: {args.method!r}.")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "preregistered_before_execution"
        or protocol["data_contract"].get("include_official_test") is not False
    ):
        raise ValueError("Matched comparison protocol is not safely preregistered.")
    protocol_sha256 = _file_sha256(CONFIG_PATH)
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
            raise ValueError(f"Unknown L2 fold: {args.fold_id!r}.")

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
    results: list[dict[str, object]] = []
    try:
        for fold in folds:
            result_path = method_root / "folds" / f"{fold.fold_id}.json"
            if result_path.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"Refusing to overwrite matched result: {result_path}"
                    )
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    existing.get("protocol_sha256") != protocol_sha256
                    or existing.get("method_id") != args.method
                    or existing.get("fold_id") != fold.fold_id
                ):
                    raise ValueError(f"Resume contract conflict: {result_path}")
                results.append(existing)
                print(f"RESUME {args.method} {fold.fold_id}", flush=True)
                continue

            started = time.perf_counter()
            splits = build_taxonomy_fold_splits(
                frame, fold, evaluation_splits=("validation",)
            )
            train_rows = splits["train"]
            eval_rows = splits["validation"]
            if args.max_training_rows is not None:
                train_rows = (
                    train_rows.assign(_uid=train_rows["row_uid"].astype(str))
                    .sort_values("_uid", kind="stable")
                    .head(args.max_training_rows)
                    .drop(columns="_uid")
                )
            if args.max_validation_rows is not None:
                eval_rows = (
                    eval_rows.assign(_uid=eval_rows["row_uid"].astype(str))
                    .sort_values("_uid", kind="stable")
                    .head(args.max_validation_rows)
                    .drop(columns="_uid")
                )
            variants = {aspect: VARIANT for aspect in aspects}
            aspect_grid = build_aspect_grid(eval_rows, aspects, variants, resource)
            sentiment_grid = build_sentiment_grid(
                eval_rows, aspects, variants, resource
            )

            one_stage_training_sha256 = "frozen_no_task_specific_training"
            two_stage_training_sha256: dict[str, str] | str = (
                "frozen_no_task_specific_training"
            )
            if args.method == "strict_train_only_tfidf":
                one_stage_train = _rerender_one_stage_training_manifest(
                    build_budgeted_training_manifest(
                        train_rows,
                        fold,
                        resource,
                        total_budget=4096,
                        positive_budget=2048,
                        seed=13,
                    ),
                    resource,
                )
                one_stage_training_sha256 = manifest_hash(one_stage_train)
                spec = resolve_method_spec(args.method)
                one_runtime = StrictTfidfRuntime(
                    tfidf_config_from_parameters(spec.starting_recipe, seed=13)
                ).fit(one_stage_train)

                seen_variants = {
                    aspect: VARIANT for aspect in fold.seen_aspects
                }
                aspect_train = build_aspect_grid(
                    train_rows, fold.seen_aspects, seen_variants, resource
                )
                sentiment_train = build_sentiment_grid(
                    train_rows,
                    fold.seen_aspects,
                    seen_variants,
                    resource,
                    gold_aspects_only=True,
                )
                two_runtime = TfidfTrueTwoStageRuntime(seed=13).fit(
                    aspect_train, sentiment_train
                )
                two_stage_training_sha256 = {
                    "aspect": manifest_hash(aspect_train),
                    "sentiment": manifest_hash(sentiment_train),
                }
                try:
                    one_scores = one_runtime.score(sentiment_grid)
                    aspect_scores = two_runtime.score_aspects(aspect_grid)
                    sentiment_scores = two_runtime.score_sentiments(sentiment_grid)
                finally:
                    one_runtime.close()
            else:
                assert e5_cache is not None
                one_scores = e5_cache.score(sentiment_grid)
                aspect_scores = e5_cache.score(aspect_grid)
                sentiment_scores = one_scores.copy()

            one_scored = sentiment_grid.copy()
            one_scored["score"] = np.asarray(one_scores, dtype=float)
            two_scored = join_two_stage_scores(
                aspect_grid,
                sentiment_grid,
                aspect_scores,
                sentiment_scores,
            )
            _assert_matched_evaluation_grids(one_scored, two_scored)
            if not np.isfinite(
                one_scored["score"].to_numpy(dtype=float)
            ).all() or not np.isfinite(
                two_scored[["aspect_score", "sentiment_score"]].to_numpy(
                    dtype=float
                )
            ).all():
                raise ValueError("Matched comparison produced non-finite scores.")

            one_seen = one_scored[
                one_scored["candidate_aspect"].astype(str).isin(fold.seen_aspects)
            ].copy()
            two_seen = two_scored[
                two_scored["candidate_aspect"].astype(str).isin(fold.seen_aspects)
            ].copy()
            one_selection = select_pair_threshold(one_seen)
            aspect_selection = select_two_stage_threshold(two_seen)
            second_selection = select_second_sentiment_threshold(
                two_seen, aspect_threshold=aspect_selection.threshold
            )

            one_mask = pair_prediction_mask(one_scored, one_selection.threshold)
            argmax_mask = two_stage_prediction_mask(
                two_scored, aspect_selection.threshold
            )
            capped_mask = capped_two_sentiment_prediction_mask(
                two_scored,
                aspect_threshold=aspect_selection.threshold,
                second_sentiment_threshold=(
                    second_selection.second_sentiment_threshold
                ),
            )
            if _selected_aspects(two_scored, argmax_mask) != _selected_aspects(
                two_scored, capped_mask
            ):
                raise AssertionError("Capped-two changed the selected aspects.")
            capped_cardinality = _sentiment_cardinality(two_scored, capped_mask)
            if capped_cardinality["maximum"] > 2:
                raise AssertionError("Capped-two emitted a third sentiment.")

            fold_result: dict[str, object] = {
                "schema_version": "taxonomy_matched_one_vs_two_stage_fold_v1",
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": protocol_sha256,
                "method_id": args.method,
                "fold_id": fold.fold_id,
                "heldout_aspect": fold.heldout_aspects[0],
                "seen_aspects": list(fold.seen_aspects),
                "train_rows": int(len(train_rows)),
                "validation_rows": int(len(eval_rows)),
                "representation": VARIANT,
                "selection_partition": "seen_validation_only",
                "evaluation_partition": "heldout_validation_only",
                "test_contract_count": 0,
                "hashes": {
                    "validation_rows_sha256": dataframe_sha256(
                        eval_rows,
                        ["row_uid"],
                        sort_columns=["row_uid"],
                    ),
                    "evaluation_pair_identity_sha256": pair_identity_hash(
                        sentiment_grid
                    ),
                    "candidate_representation_sha256": dataframe_sha256(
                        sentiment_grid.drop_duplicates(
                            ["candidate_aspect", "candidate_sentiment"]
                        ),
                        [
                            "candidate_aspect",
                            "candidate_sentiment",
                            "candidate_text",
                            "representation_variant",
                        ],
                        sort_columns=["candidate_aspect", "candidate_sentiment"],
                    ),
                    "one_stage_training_sha256": one_stage_training_sha256,
                    "two_stage_training_sha256": two_stage_training_sha256,
                    "one_stage_scores_sha256": _score_hash(
                        one_scored, ["score"]
                    ),
                    "two_stage_scores_sha256": _score_hash(
                        two_scored, ["aspect_score", "sentiment_score"]
                    ),
                },
                "decoders": {
                    "one_stage_independent_pairs": {
                        "selection": {
                            "pair_threshold": one_selection.threshold,
                            "selection_partition": "seen_validation",
                        },
                        "heldout": _decoder_metrics(
                            one_scored,
                            one_mask,
                            aspects=fold.heldout_aspects,
                        ),
                        "sentiment_cardinality": _sentiment_cardinality(
                            one_scored, one_mask
                        ),
                    },
                    "two_stage_argmax": {
                        "selection": {
                            "aspect_threshold": aspect_selection.threshold,
                            "selection_partition": "seen_validation",
                        },
                        "heldout": _decoder_metrics(
                            two_scored,
                            argmax_mask,
                            aspects=fold.heldout_aspects,
                        ),
                        "sentiment_cardinality": _sentiment_cardinality(
                            two_scored, argmax_mask
                        ),
                    },
                    "two_stage_capped_two": {
                        "selection": {
                            "aspect_threshold": aspect_selection.threshold,
                            "second_sentiment_threshold": (
                                second_selection.second_sentiment_threshold
                            ),
                            "second_threshold_selection_metrics": (
                                second_selection.selection_metrics
                            ),
                            "selection_partition": "seen_validation",
                            "aspect_threshold_frozen": True,
                        },
                        "heldout": _decoder_metrics(
                            two_scored,
                            capped_mask,
                            aspects=fold.heldout_aspects,
                        ),
                        "sentiment_cardinality": capped_cardinality,
                    },
                },
                "matched_checks": {
                    "same_validation_rows": True,
                    "same_pair_identities": True,
                    "same_name_and_description_representation": True,
                    "seen_only_selection": True,
                    "capped_two_preserved_aspect_decisions": True,
                },
                "seconds": float(time.perf_counter() - started),
            }
            _write_json(result_path, fold_result)
            results.append(fold_result)
            print(
                f"COMPLETE {args.method} {fold.fold_id} "
                f"seconds={fold_result['seconds']:.1f}",
                flush=True,
            )
    finally:
        if e5_cache is not None:
            e5_cache.close()

    aggregate = _aggregate(results)
    summary: dict[str, object] = {
        "schema_version": "taxonomy_matched_one_vs_two_stage_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "method_id": args.method,
        "level": "L2",
        "completed_folds": len(results),
        "failed_folds": 0,
        "test_contract_count": 0,
        "selection_partition": "seen validation only",
        "evaluation_partition": "held-out validation only",
        "representation": VARIANT,
        "primary_comparison": (
            "one_stage_independent_pairs versus two_stage_capped_two"
        ),
        "capped_two_is_default_sentiment_decoder": True,
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
        / PROTOCOL_ID,
    )
    parser.add_argument("--fold-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-training-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
