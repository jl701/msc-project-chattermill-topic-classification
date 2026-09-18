"""Run the preregistered dataset-faithful capped-two sentiment validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
    capped_two_sentiment_prediction_mask,
    select_second_sentiment_threshold,
    select_two_stage_threshold,
    two_stage_prediction_mask,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    TfidfTrueTwoStageRuntime,
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
)
from run_taxonomy_multi_sentiment_decoder_validation import (  # noqa: E402
    E5ScoreCache,
    METHODS,
    VARIANT,
    _aggregate,
    _assert_parent_control,
    _decoder_metrics,
    _jsonable,
    _write_json,
)


def _selected_aspect_keys(scored: pd.DataFrame, mask: pd.Series) -> set[tuple[str, str]]:
    selected = scored.loc[pd.Series(mask, index=scored.index, dtype=bool)]
    return set(
        zip(
            selected["row_uid"].astype(str),
            selected["candidate_aspect"].astype(str),
        )
    )


def _assert_decoder_structure(
    scored: pd.DataFrame,
    control_mask: pd.Series,
    capped_mask: pd.Series,
) -> None:
    control_aspects = _selected_aspect_keys(scored, control_mask)
    capped_aspects = _selected_aspect_keys(scored, capped_mask)
    if capped_aspects != control_aspects:
        raise AssertionError("Capped-two decoder changed parent aspect decisions.")
    selected = scored.loc[pd.Series(capped_mask, index=scored.index, dtype=bool)]
    cardinality = selected.groupby(["row_uid", "candidate_aspect"]).size()
    if cardinality.empty or int(cardinality.min()) < 1 or int(cardinality.max()) > 2:
        raise AssertionError("Capped-two decoder violated the one-or-two contract.")


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
                        train_rows,
                        fold.seen_aspects,
                        train_variants,
                        resource,
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
                eval_rows,
                aspects,
                variants,
                resource,
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
            capped_selection = select_second_sentiment_threshold(
                seen,
                aspect_threshold=control_selection.threshold,
            )
            if capped_selection.aspect_threshold != control_selection.threshold:
                raise AssertionError("Second-sentiment search changed aspect threshold.")
            control_mask = two_stage_prediction_mask(
                scored,
                control_selection.threshold,
            )
            capped_mask = capped_two_sentiment_prediction_mask(
                scored,
                aspect_threshold=control_selection.threshold,
                second_sentiment_threshold=(
                    capped_selection.second_sentiment_threshold
                ),
            )
            _assert_decoder_structure(scored, control_mask, capped_mask)

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
                "capped_two_threshold": {
                    "selection": {
                        "aspect_threshold": control_selection.threshold,
                        "second_sentiment_threshold": (
                            capped_selection.second_sentiment_threshold
                        ),
                        "selection_metrics": capped_selection.selection_metrics,
                        "candidates_evaluated": (
                            capped_selection.candidates_evaluated
                        ),
                        "selection_partition": "seen_validation",
                        "aspect_threshold_frozen": True,
                    },
                    "heldout": _decoder_metrics(
                        scored,
                        capped_mask,
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
                "schema_version": "taxonomy_capped_two_sentiment_fold_v2",
                "protocol_id": "taxonomy_capped_two_sentiment_validation_v2",
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
                "capped2="
                f"{decoders['capped_two_threshold']['heldout']['metrics']['pair_micro_f1']:.6f}",
                flush=True,
            )
    finally:
        if e5_cache is not None:
            e5_cache.close()

    aggregate = _aggregate(fold_results)
    summary: dict[str, object] = {
        "schema_version": "taxonomy_capped_two_sentiment_summary_v2",
        "protocol_id": "taxonomy_capped_two_sentiment_validation_v2",
        "method_id": args.method,
        "representation": VARIANT,
        "completed_folds": len(fold_results),
        "failed_folds": 0,
        "test_contract_count": 0,
        "selection_partition": "seen validation only",
        "evaluation_partition": "held-out validation only",
        "aspect_threshold_frozen": True,
        "maximum_sentiments_per_aspect": 2,
        **aggregate,
    }
    _write_json(method_root / "summary.json", summary)
    pd.DataFrame.from_records(aggregate["records"]).to_csv(
        method_root / "per_fold.csv",
        index=False,
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
        / "taxonomy_capped_two_sentiment_validation_v2",
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
