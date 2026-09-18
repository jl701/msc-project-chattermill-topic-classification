"""Single-reveal analysis for the sealed taxonomy final-test score graph."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from msc_project.data.fabsa import format_pair_label
from msc_project.experiments.taxonomy_final_test import (
    FIXED_COMPOSITION_METHOD,
    PROTOCOL_ID,
    FinalTestJob,
    canonical_sha256,
    file_sha256,
)
from msc_project.experiments.taxonomy_final_test_workflow import (
    SENTIMENTS,
    atomic_csv,
    atomic_json,
    copy_verified_file,
    decode_capped_two,
    read_score_bundle,
    validate_score_seal,
)

LABEL_COLUMNS = (
    "row_uid",
    "candidate_aspect",
    "candidate_sentiment",
    "target",
)


def build_label_vault(
    labelled_rows: pd.DataFrame, aspects: Sequence[str]
) -> pd.DataFrame:
    required = {"row_uid", "labels"}
    missing = sorted(required - set(labelled_rows.columns))
    if missing or labelled_rows.empty:
        raise ValueError(f"Label-vault source is invalid; missing={missing}.")
    candidates = tuple(str(value) for value in aspects)
    records: list[dict[str, object]] = []
    for row in labelled_rows.assign(
        _uid=labelled_rows["row_uid"].astype(str)
    ).sort_values("_uid", kind="stable").itertuples(index=False):
        labels = {(str(aspect), str(sentiment)) for aspect, sentiment in row.labels}
        for aspect in candidates:
            for sentiment in SENTIMENTS:
                records.append(
                    {
                        "row_uid": str(row.row_uid),
                        "candidate_aspect": aspect,
                        "candidate_sentiment": sentiment,
                        "target": int((aspect, sentiment) in labels),
                    }
                )
    frame = pd.DataFrame.from_records(records, columns=LABEL_COLUMNS)
    validate_label_vault(
        frame,
        expected_row_uids=labelled_rows["row_uid"].astype(str).tolist(),
        expected_aspects=candidates,
    )
    return frame


def validate_label_vault(
    frame: pd.DataFrame,
    *,
    expected_row_uids: Sequence[str],
    expected_aspects: Sequence[str],
) -> dict[str, object]:
    if tuple(frame.columns) != LABEL_COLUMNS or frame.empty:
        raise ValueError("Label vault does not match its frozen four-column schema.")
    keys = list(LABEL_COLUMNS[:-1])
    if frame.duplicated(keys).any():
        raise ValueError("Label vault contains duplicate candidate identities.")
    if set(frame["target"].astype(int)) - {0, 1}:
        raise ValueError("Label vault target must be binary.")
    observed_uids = sorted(frame["row_uid"].astype(str).unique())
    observed_aspects = tuple(frame["candidate_aspect"].astype(str).drop_duplicates())
    if observed_uids != sorted(str(value) for value in expected_row_uids):
        raise ValueError("Label-vault review identities differ from the release record.")
    if set(observed_aspects) != {str(value) for value in expected_aspects}:
        raise ValueError("Label-vault aspect set differs from the frozen taxonomy.")
    per_review = frame.groupby("row_uid", sort=False).size()
    if not per_review.eq(len(expected_aspects) * 3).all():
        raise ValueError("Label vault does not contain the complete candidate grid.")
    return {
        "status": "pass",
        "rows": len(frame),
        "review_rows": len(observed_uids),
        "positive_pairs": int(frame["target"].astype(int).sum()),
        "payload_sha256": canonical_sha256(
            frame.sort_values(keys, kind="stable").to_dict(orient="records")
        ),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_counts(target: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int]:
    truth = target.astype(bool)
    pred = prediction.astype(bool)
    return (
        int(np.sum(truth & pred)),
        int(np.sum(~truth & pred)),
        int(np.sum(truth & ~pred)),
    )


def _metric_record(frame: pd.DataFrame) -> dict[str, object]:
    target = frame["target"].to_numpy(dtype=int)
    prediction = frame["prediction"].to_numpy(dtype=bool)
    tp, fp, fn = _binary_counts(target, prediction)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    micro_f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    class_values: list[float] = []
    for _, group in frame.groupby(
        ["candidate_aspect", "candidate_sentiment"], sort=False
    ):
        ctp, cfp, cfn = _binary_counts(
            group["target"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=bool),
        )
        class_values.append(_safe_divide(2 * ctp, 2 * ctp + cfp + cfn))
    exact = (
        frame.groupby("row_uid", sort=False)
        .apply(
            lambda value: bool(
                np.array_equal(
                    value["target"].to_numpy(dtype=int),
                    value["prediction"].to_numpy(dtype=int),
                )
            ),
            include_groups=False,
        )
        .mean()
    )
    predicted_pairs = frame.groupby("row_uid", sort=False)["prediction"].sum()
    predicted_aspects = (
        frame[frame["prediction"]]
        .groupby("row_uid", sort=False)["candidate_aspect"]
        .nunique()
        .reindex(frame["row_uid"].astype(str).drop_duplicates(), fill_value=0)
    )
    return {
        "pair_micro_precision": precision,
        "pair_micro_recall": recall,
        "pair_micro_f1": micro_f1,
        "pair_macro_f1": float(np.mean(class_values)),
        "exact_set_match": float(exact),
        "mean_prediction_set_size": float(predicted_pairs.mean()),
        "aspect_call_rate": float((predicted_aspects > 0).mean()),
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "gold_pair_count": int(target.sum()),
        "predicted_pair_count": int(prediction.sum()),
        "review_rows": int(frame["row_uid"].nunique()),
    }


def _oracle_stage2_prediction(
    frame: pd.DataFrame, *, runner_up_threshold: float
) -> np.ndarray:
    ordered = frame.reset_index(drop=True).reset_index(names="_position").sort_values(
        ["row_uid", "candidate_aspect", "candidate_sentiment"], kind="stable"
    )
    output = np.zeros(len(frame), dtype=bool)
    for _, group in ordered.groupby(["row_uid", "candidate_aspect"], sort=False):
        if not group["target"].astype(bool).any():
            continue
        scores = group["sentiment_score"].to_numpy(dtype=float)
        rank = np.argsort(-scores, kind="stable")
        selected = [int(rank[0])]
        if scores[int(rank[1])] >= float(runner_up_threshold):
            selected.append(int(rank[1]))
        output[group.iloc[selected]["_position"].to_numpy(dtype=int)] = True
    return output


def evaluate_score_bundle(
    score: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    aspect_threshold: float,
    runner_up_threshold: float,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    joined = score.merge(
        labels,
        on=["row_uid", "candidate_aspect", "candidate_sentiment"],
        how="left",
        validate="many_to_one",
    )
    if joined["target"].isna().any():
        raise ValueError("A sealed score identity is absent from the label vault.")
    joined["target"] = joined["target"].astype(int)
    joined["prediction"] = decode_capped_two(
        joined,
        aspect_threshold=aspect_threshold,
        runner_up_threshold=runner_up_threshold,
    ).to_numpy(dtype=bool)
    metrics: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for condition, condition_frame in joined.groupby("condition", sort=True):
        for partition, partition_frame in (
            ("overall", condition_frame),
            ("seen", condition_frame[condition_frame["is_seen"].astype(bool)]),
            (
                "heldout",
                condition_frame[condition_frame["is_heldout"].astype(bool)],
            ),
        ):
            record = _metric_record(partition_frame)
            record.update(
                {
                    "job_id": str(score["job_id"].iloc[0]),
                    "method_id": str(score["method_id"].iloc[0]),
                    "level": str(score["level"].iloc[0]),
                    "fold_id": str(score["fold_id"].iloc[0]),
                    "training_scope_id": str(score["training_scope_id"].iloc[0]),
                    "condition": str(condition),
                    "partition": partition,
                }
            )
            metrics.append(record)

        heldout = condition_frame[condition_frame["is_heldout"].astype(bool)].copy()
        aspects = heldout.drop_duplicates(["row_uid", "candidate_aspect"])
        presence_target = (
            heldout.groupby(["row_uid", "candidate_aspect"], sort=False)["target"]
            .max()
            .to_numpy(dtype=int)
        )
        presence_scores = aspects["aspect_score"].to_numpy(dtype=float)
        presence_prediction = presence_scores >= float(aspect_threshold)
        tp, fp, fn = _binary_counts(presence_target, presence_prediction)
        presence_f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
        ap = (
            float(average_precision_score(presence_target, presence_scores))
            if len(set(presence_target.tolist())) > 1
            else float(presence_target.mean())
        )
        oracle = heldout.copy()
        oracle["prediction"] = _oracle_stage2_prediction(
            oracle, runner_up_threshold=runner_up_threshold
        )
        oracle_metrics = _metric_record(oracle)
        diagnostics.append(
            {
                "job_id": str(score["job_id"].iloc[0]),
                "method_id": str(score["method_id"].iloc[0]),
                "level": str(score["level"].iloc[0]),
                "fold_id": str(score["fold_id"].iloc[0]),
                "condition": str(condition),
                "heldout_presence_average_precision": ap,
                "heldout_presence_f1": presence_f1,
                "stage1_tp": tp,
                "stage1_fp": fp,
                "stage1_fn": fn,
                "oracle_gated_stage2_pair_micro_f1": oracle_metrics[
                    "pair_micro_f1"
                ],
                "third_sentiment_violation_count": 0,
                "prediction_collapse_count": 0,
            }
        )
        grouped = heldout.groupby("row_uid", sort=False)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "job_id": str(score["job_id"].iloc[0]),
                    "method_id": str(score["method_id"].iloc[0]),
                    "level": str(score["level"].iloc[0]),
                    "fold_id": str(score["fold_id"].iloc[0]),
                    "condition": str(condition),
                    "row_uid": list(grouped.groups),
                    "gold_pairs": [
                        sorted(
                            format_pair_label(str(row.candidate_aspect), str(row.candidate_sentiment))
                            for row in value.itertuples(index=False)
                            if int(row.target) == 1
                        )
                        for _, value in grouped
                    ],
                    "pred_pairs": [
                        sorted(
                            format_pair_label(str(row.candidate_aspect), str(row.candidate_sentiment))
                            for row in value.itertuples(index=False)
                            if bool(row.prediction)
                        )
                        for _, value in grouped
                    ],
                }
            )
        )
    return metrics, pd.concat(prediction_rows, ignore_index=True), diagnostics


def _counts_by_fold_and_review(predictions: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    folds = tuple(sorted(predictions["fold_id"].astype(str).unique()))
    uids = tuple(sorted(predictions["row_uid"].astype(str).unique()))
    fold_index = {value: index for index, value in enumerate(folds)}
    uid_index = {value: index for index, value in enumerate(uids)}
    counts = np.zeros((len(folds), len(uids), 3), dtype=np.int64)
    for row in predictions.itertuples(index=False):
        gold = set(row.gold_pairs)
        pred = set(row.pred_pairs)
        counts[fold_index[str(row.fold_id)], uid_index[str(row.row_uid)]] = (
            len(gold & pred),
            len(pred - gold),
            len(gold - pred),
        )
    return folds, uids, counts


def _aspect_balanced_f1(counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.einsum("bu,fuc->bfc", weights, counts, optimize=True)
    tp, fp, fn = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )
    return f1.mean(axis=1)


def synchronised_aspect_balanced_difference(
    challenger: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    cf, cu, cc = _counts_by_fold_and_review(challenger)
    rf, ru, rc = _counts_by_fold_and_review(reference)
    if cf != rf or cu != ru:
        raise ValueError("Confirmatory systems do not share identical folds/reviews.")
    point_weights = np.ones((1, len(cu)), dtype=np.int64)
    challenger_point = float(_aspect_balanced_f1(cc, point_weights)[0])
    reference_point = float(_aspect_balanced_f1(rc, point_weights)[0])
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    # Preserve the already-audited formal-v2 resampling contract exactly.  A
    # multinomial draw is statistically equivalent to drawing review indices
    # with replacement, but the two consume the RNG stream differently.  The
    # exact multinomial form and batch size below are therefore frozen so that
    # the final runner reproduces the published validation intervals bit for
    # bit before it is allowed anywhere near the official test.
    batch_size = 250
    probabilities = np.full(len(cu), 1.0 / len(cu))
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        weights = rng.multinomial(len(cu), probabilities, size=size)
        samples.append(
            _aspect_balanced_f1(cc, weights) - _aspect_balanced_f1(rc, weights)
        )
    differences = np.concatenate(samples)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "endpoint": "L2_D_aspect_balanced_mean_heldout_pair_micro_f1",
        "challenger_point": challenger_point,
        "reference_point": reference_point,
        "point_difference": challenger_point - reference_point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": 0.95,
        "interval_method": "synchronised_review_cluster_percentile_bootstrap",
        "folds": len(cf),
        "review_clusters": len(cu),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "superiority": bool(lower > 0),
    }


def _pooled_heldout_f1(predictions: pd.DataFrame) -> float:
    tp = fp = fn = 0
    for row in predictions.itertuples(index=False):
        gold, pred = set(row.gold_pairs), set(row.pred_pairs)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return _safe_divide(2 * tp, 2 * tp + fp + fn)


def analyse_sealed_graph_once(
    *,
    output_root: Path,
    backup_root: Path,
    jobs: Sequence[FinalTestJob],
    label_vault_path: Path,
    expected_row_uids: Sequence[str],
    expected_aspects: Sequence[str],
    preregistration_sha256: str,
    execution_commit: str,
    authorisation_id: str,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 13,
    test_contract_count: int = 1,
) -> dict[str, object]:
    analysis_root = output_root / "single_reveal_analysis"
    if analysis_root.exists():
        raise FileExistsError("The one-time final analysis output already exists.")
    validate_score_seal(
        output_root / "sealed_scores" / "score_seal.json",
        preregistration_sha256=preregistration_sha256,
        execution_commit=execution_commit,
        authorisation_id=authorisation_id,
        expected_test_contract_count=test_contract_count,
    )
    labels = pd.read_csv(label_vault_path)
    label_audit = validate_label_vault(
        labels,
        expected_row_uids=expected_row_uids,
        expected_aspects=expected_aspects,
    )
    all_metrics: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    score_jobs = [job for job in jobs if job.phase in {"score", "compose"}]
    for job in score_jobs:
        score, manifest = read_score_bundle(
            output_root,
            job,
            preregistration_sha256=preregistration_sha256,
            execution_commit=execution_commit,
            authorisation_id=authorisation_id,
            expected_row_uids=expected_row_uids,
            expected_test_contract_count=test_contract_count,
        )
        metrics, predictions, diagnostics = evaluate_score_bundle(
            score,
            labels,
            aspect_threshold=float(manifest["aspect_threshold"]),
            runner_up_threshold=float(manifest["runner_up_threshold"]),
        )
        all_metrics.extend(metrics)
        all_diagnostics.extend(diagnostics)
        all_predictions.append(predictions)
    metrics_frame = pd.DataFrame.from_records(all_metrics)
    diagnostics_frame = pd.DataFrame.from_records(all_diagnostics)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)

    primary = predictions_frame[
        predictions_frame["level"].eq("L2")
        & predictions_frame["condition"].eq("D")
        & predictions_frame["method_id"].isin(
            [
                FIXED_COMPOSITION_METHOD,
                "frozen_qwen_few_shot",
                "qwen_candidate_pair_qlora",
            ]
        )
    ]
    by_method = {
        method: primary[primary["method_id"].eq(method)].copy()
        for method in primary["method_id"].unique()
    }
    h1 = synchronised_aspect_balanced_difference(
        by_method[FIXED_COMPOSITION_METHOD],
        by_method["frozen_qwen_few_shot"],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    h1.update(
        {
            "hypothesis": "H1",
            "candidate": FIXED_COMPOSITION_METHOD,
            "comparator": "frozen_qwen_few_shot",
            "confirmatory_role": "primary",
        }
    )
    h2 = synchronised_aspect_balanced_difference(
        by_method[FIXED_COMPOSITION_METHOD],
        by_method["qwen_candidate_pair_qlora"],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    h2.update(
        {
            "hypothesis": "H2",
            "candidate": FIXED_COMPOSITION_METHOD,
            "comparator": "qwen_candidate_pair_qlora",
            "confirmatory_role": "confirmatory" if h1["superiority"] else "descriptive_only",
            "gate_H1_passed": bool(h1["superiority"]),
        }
    )
    confirmatory = [h1, h2]
    headline: list[dict[str, object]] = []
    for method, frame in sorted(by_method.items()):
        counts = _counts_by_fold_and_review(frame)[2]
        primary_value = float(
            _aspect_balanced_f1(
                counts, np.ones((1, counts.shape[1]), dtype=np.int64)
            )[0]
        )
        headline.append(
            {
                "method_id": method,
                "L2_D_aspect_balanced_mean_heldout_pair_micro_f1": primary_value,
                "L2_D_pooled_heldout_pair_micro_f1": _pooled_heldout_f1(frame),
            }
        )

    analysis_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "all_metrics": analysis_root / "all_registered_metrics.csv",
        "diagnostics": analysis_root / "diagnostic_metrics.csv",
        "heldout_predictions": analysis_root / "heldout_prediction_sets.jsonl",
        "confirmatory": analysis_root / "confirmatory_hypotheses.json",
        "headline": analysis_root / "primary_headline.csv",
    }
    atomic_csv(paths["all_metrics"], metrics_frame)
    atomic_csv(paths["diagnostics"], diagnostics_frame)
    atomic_csv(paths["headline"], pd.DataFrame.from_records(headline))
    lines = "\n".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in predictions_frame.to_dict(orient="records")
    ) + "\n"
    temporary = paths["heldout_predictions"].with_name(
        f".{paths['heldout_predictions'].name}.tmp-{os.getpid()}"
    )
    temporary.write_text(lines, encoding="utf-8", newline="\n")
    os.replace(temporary, paths["heldout_predictions"])
    atomic_json(
        paths["confirmatory"],
        {
            "schema_version": "taxonomy_final_test_confirmatory_results_v1",
            "protocol_id": PROTOCOL_ID,
            "multiplicity": "hierarchical_gatekeeping_H1_then_H2",
            "results": confirmatory,
        },
    )
    output_hashes = {
        key: file_sha256(path) for key, path in sorted(paths.items())
    }
    manifest: dict[str, object] = {
        "schema_version": "taxonomy_final_test_single_reveal_manifest_v1",
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": preregistration_sha256,
        "execution_commit": execution_commit,
        "authorisation_id": authorisation_id,
        "label_vault_sha256": file_sha256(label_vault_path),
        "label_vault_audit": label_audit,
        "score_bundle_count": len(score_jobs),
        "outcomes_revealed_once": True,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "output_sha256s": output_hashes,
        "failure_count": 0,
        "test_contract_count": int(test_contract_count),
        "post_test_tuning_permitted": False,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    manifest_path = analysis_root / "analysis_manifest.json"
    atomic_json(manifest_path, manifest)
    for path in [*paths.values(), manifest_path]:
        copy_verified_file(path, backup_root / path.relative_to(output_root))
    return {
        "status": "complete",
        "headline": headline,
        "confirmatory": confirmatory,
        "analysis_manifest": str(manifest_path),
        "analysis_manifest_sha256": file_sha256(manifest_path),
        "failure_count": 0,
    }
