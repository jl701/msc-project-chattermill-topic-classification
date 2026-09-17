"""Run validation-only description-to-classifier-weight transfer (DCWT)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.baselines.candidate_similarity import (  # noqa: E402
    FrozenTransformerSentenceEncoder,
    REGISTERED_CONFIGS,
)
from msc_project.baselines.unified_pair_scorers import (  # noqa: E402
    UnifiedTfidfPairConfig,
    UnifiedTfidfPairScorer,
)
from msc_project.data.fabsa import default_data_dir  # noqa: E402
from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_description_weight_transfer import (  # noqa: E402
    GENERATOR_FAMILIES,
    parameter_direction_cosine,
    pseudo_unseen_scores,
    select_presence_threshold,
    sigmoid,
    synthesise_weight,
)
from msc_project.experiments.taxonomy_post_supervisor import (  # noqa: E402
    aggregate_l2_folds,
    evaluate_l2_condition,
    post_supervisor_l2_folds,
)
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    candidate_representation_variants,
    canonical_aspects,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    capped_two_sentiment_prediction_mask,
    select_second_sentiment_threshold,
)
from msc_project.experiments.taxonomy_scientific_freeze import (  # noqa: E402
    row_pair_confusions,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
    render_aspect_candidate,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_description_to_classifier_weight_transfer_v1.json"
)


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels_matrix(rows: pd.DataFrame, aspects: tuple[str, ...]) -> np.ndarray:
    records = []
    for labels in rows["supervision_labels"]:
        present = {str(aspect) for aspect, _ in labels}
        records.append([int(aspect in present) for aspect in aspects])
    output = np.asarray(records, dtype=np.uint8)
    if output.shape != (len(rows), len(aspects)):
        raise AssertionError("Aspect target matrix has an unexpected shape.")
    return output


def _fit_review_space(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    components: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    vectoriser = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    train_sparse = vectoriser.fit_transform(train["text"].astype(str).tolist())
    validation_sparse = vectoriser.transform(validation["text"].astype(str).tolist())
    output: dict[int, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    for requested in components:
        effective = min(int(requested), int(train_sparse.shape[1] - 1))
        if effective < 2:
            raise ValueError("TF-IDF feature space is too small for registered SVD.")
        projector = TruncatedSVD(
            n_components=effective,
            algorithm="randomized",
            random_state=13,
        )
        train_latent = projector.fit_transform(train_sparse)
        validation_latent = projector.transform(validation_sparse)
        scaler = StandardScaler().fit(train_latent)
        output[int(requested)] = (
            scaler.transform(train_latent),
            scaler.transform(validation_latent),
            {
                "requested_components": int(requested),
                "effective_components": effective,
                "tfidf_features": int(train_sparse.shape[1]),
                "explained_variance_ratio_sum": float(
                    projector.explained_variance_ratio_.sum()
                ),
            },
        )
    return output


def _fit_seen_weights(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    classifier_c: float,
) -> np.ndarray:
    parameters: list[np.ndarray] = []
    for column in range(targets.shape[1]):
        target = targets[:, column].astype(int)
        if set(np.unique(target)) != {0, 1}:
            raise ValueError("Every seen-aspect classifier requires both classes.")
        classifier = LogisticRegression(
            C=float(classifier_c),
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=13,
        ).fit(features, target)
        parameters.append(
            np.concatenate(
                [classifier.coef_.reshape(-1), classifier.intercept_.reshape(-1)]
            )
        )
    matrix = np.vstack(parameters)
    if not np.isfinite(matrix).all():
        raise ValueError("A seen classifier produced non-finite parameters.")
    return matrix


def _configuration_id(
    family: str,
    components: int,
    classifier_c: float,
    parameter: float | None,
) -> str:
    suffix = "none" if parameter is None else f"{parameter:g}"
    return f"{family}-svd{components}-c{classifier_c:g}-p{suffix}"


def _rank_configuration(record: dict[str, object]) -> tuple[object, ...]:
    family = str(record["family"])
    parameter = record["generator_parameter"]
    complexity = -int(record["svd_components"])
    stronger_classifier_regularisation = -float(record["classifier_c"])
    if family == "kernel_ridge":
        stronger_generator_regularisation = float(parameter)
    elif family == "cosine_barycentric_weight":
        stronger_generator_regularisation = float(parameter)
    else:
        stronger_generator_regularisation = 0.0
    return (
        float(record["pseudo_unseen_f1"]),
        float(record["pseudo_unseen_average_precision"]),
        complexity,
        stronger_classifier_regularisation,
        stronger_generator_regularisation,
        str(record["configuration_id"]),
    )


def _target_descriptor_text(
    aspect: str,
    condition: str,
    resource: dict[str, object],
) -> str:
    variants = {"N": "name_only", "D": "name_and_description", "R": "rich"}
    return render_aspect_candidate(aspect, variants[condition], resource)


def run(args: argparse.Namespace) -> dict[str, object]:
    started_all = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["status"] != "preregistered_design_pending_implementation":
        raise ValueError("Unexpected DCWT preregistration state.")
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    resource = load_description_bundle(require_approved=True)
    aspects = canonical_aspects()
    validation = frame[frame["original_split"].eq("validation")].copy()
    validation["supervision_labels"] = validation["labels"]
    validation = validation.assign(_uid=validation["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    ).drop(columns="_uid").reset_index(drop=True)

    folds = list(post_supervisor_l2_folds())
    if args.fold_id:
        folds = [fold for fold in folds if fold.fold_id == args.fold_id]
        if not folds:
            raise ValueError(f"Unknown Level 2 fold: {args.fold_id!r}.")

    minimal_descriptor_texts = [
        render_aspect_candidate(aspect, "name_and_description", resource)
        for aspect in aspects
    ]
    target_texts = [
        _target_descriptor_text(aspect, condition, resource)
        for aspect in aspects
        for condition in ("N", "D", "R")
    ]
    encoder = FrozenTransformerSentenceEncoder(
        REGISTERED_CONFIGS["e5_base_v2"],
        device="auto",
        local_files_only=args.local_files_only,
    )
    try:
        minimal_embeddings = encoder.encode(minimal_descriptor_texts, role="candidate")
        target_embeddings_matrix = encoder.encode(target_texts, role="candidate")
    finally:
        encoder.close()
    target_embeddings = {
        (aspect, condition): target_embeddings_matrix[index * 3 + offset]
        for index, aspect in enumerate(aspects)
        for offset, condition in enumerate(("N", "D", "R"))
    }

    components = tuple(int(value) for value in config["stage_1"]["review_representation"]["svd_components_grid"])
    c_grid = tuple(float(value) for value in config["stage_1"]["seen_aspect_classifiers"]["c_grid"])
    alpha_grid = tuple(float(value) for value in config["stage_1"]["primary_generator"]["ridge_alpha_grid"])
    temperature_grid = tuple(float(value) for value in config["stage_1"]["mandatory_transfer_baselines"][2]["temperature_grid"])
    output_root = args.output_root.resolve()
    fold_results: list[dict[str, object]] = []

    for fold in folds:
        result_path = output_root / "folds" / f"{fold.fold_id}.json"
        if args.resume and result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("protocol_sha256") != _sha256(CONFIG_PATH)
                or existing.get("training_scope_id") != training_scope_id(fold)
                or existing.get("failure_count") != 0
                or existing.get("test_contract_count") != 0
            ):
                raise ValueError(f"DCWT resume conflict for {fold.fold_id}.")
            fold_results.append(existing)
            print(f"RESUME dcwt {fold.fold_id}", flush=True)
            continue

        started = time.perf_counter()
        splits = build_taxonomy_fold_splits(
            frame, fold, evaluation_splits=("validation",)
        )
        train = splits["train"].copy()
        train = train.assign(_uid=train["row_uid"].astype(str)).sort_values(
            "_uid", kind="stable"
        ).drop(columns="_uid").reset_index(drop=True)
        evaluation = splits["validation"].copy()
        evaluation = evaluation.assign(_uid=evaluation["row_uid"].astype(str)).sort_values(
            "_uid", kind="stable"
        ).drop(columns="_uid").reset_index(drop=True)
        if evaluation["row_uid"].astype(str).tolist() != validation["row_uid"].astype(str).tolist():
            raise AssertionError("DCWT validation rows differ from the fixed Level 2 rows.")

        seen = tuple(fold.seen_aspects)
        heldout = fold.heldout_aspects[0]
        train_targets = _labels_matrix(train, seen)
        validation_targets = _labels_matrix(validation, seen)
        spaces = _fit_review_space(train, validation, components)
        model_cache: dict[tuple[int, float], np.ndarray] = {}
        selection_records: dict[str, list[dict[str, object]]] = {
            family: [] for family in GENERATOR_FAMILIES
        }
        seen_indices = np.asarray([aspects.index(aspect) for aspect in seen])
        seen_descriptors = minimal_embeddings[seen_indices]

        for component_count in components:
            train_features, validation_features, diagnostics = spaces[component_count]
            for classifier_c in c_grid:
                weights = _fit_seen_weights(
                    train_features,
                    train_targets,
                    classifier_c=classifier_c,
                )
                model_cache[(component_count, classifier_c)] = weights
                parameter_values: dict[str, tuple[float | None, ...]] = {
                    "kernel_ridge": alpha_grid,
                    "mean_seen_weight": (None,),
                    "nearest_description_weight": (None,),
                    "cosine_barycentric_weight": temperature_grid,
                }
                for family, values in parameter_values.items():
                    for parameter in values:
                        kwargs: dict[str, float] = {}
                        if family == "kernel_ridge":
                            kwargs["alpha"] = float(parameter)
                        elif family == "cosine_barycentric_weight":
                            kwargs["temperature"] = float(parameter)
                        pseudo_targets, pseudo_scores, direction_cosine = pseudo_unseen_scores(
                            seen_descriptors,
                            weights,
                            validation_features,
                            validation_targets,
                            family=family,
                            **kwargs,
                        )
                        threshold = select_presence_threshold(
                            pseudo_targets, pseudo_scores
                        )
                        selection_records[family].append(
                            {
                                "configuration_id": _configuration_id(
                                    family,
                                    component_count,
                                    classifier_c,
                                    parameter,
                                ),
                                "family": family,
                                "svd_components": component_count,
                                "effective_components": diagnostics["effective_components"],
                                "classifier_c": classifier_c,
                                "generator_parameter": parameter,
                                "aspect_threshold": threshold.threshold,
                                "pseudo_unseen_f1": threshold.f1,
                                "pseudo_unseen_average_precision": threshold.average_precision,
                                "pseudo_unseen_precision": threshold.precision,
                                "pseudo_unseen_recall": threshold.recall,
                                "parameter_direction_cosine": direction_cosine,
                                "threshold_candidates": threshold.candidates_evaluated,
                            }
                        )

        selected = {
            family: max(records, key=_rank_configuration)
            for family, records in selection_records.items()
        }
        train_variants = {aspect: "name_and_description" for aspect in seen}
        sentiment_scorer = UnifiedTfidfPairScorer(
            UnifiedTfidfPairConfig(
                classifier_c=1.0,
                feature_ablation="all_six",
                max_iter=1000,
                seed=13,
            )
        ).fit(
            build_sentiment_grid(
                train,
                seen,
                train_variants,
                resource,
                gold_aspects_only=True,
            )
        )

        generator_results: dict[str, object] = {}
        for family, selected_config in selected.items():
            component_count = int(selected_config["svd_components"])
            classifier_c = float(selected_config["classifier_c"])
            _, validation_features, space_diagnostics = spaces[component_count]
            weights = model_cache[(component_count, classifier_c)]
            kwargs = {}
            if family == "kernel_ridge":
                kwargs["alpha"] = float(selected_config["generator_parameter"])
            elif family == "cosine_barycentric_weight":
                kwargs["temperature"] = float(selected_config["generator_parameter"])

            direct_seen_scores = np.column_stack(
                [
                    sigmoid(validation_features @ weight[:-1] + weight[-1])
                    for weight in weights
                ]
            )
            minimal_target_weight = synthesise_weight(
                seen_descriptors,
                weights,
                target_embeddings[(heldout, "D")],
                family=family,
                **kwargs,
            )
            parameter_diagnostics = {
                "predicted_weight_l2_norm": float(np.linalg.norm(minimal_target_weight)),
                "mean_seen_weight_l2_norm": float(np.mean(np.linalg.norm(weights, axis=1))),
                "minimum_seen_direction_cosine": float(
                    min(parameter_direction_cosine(weight, minimal_target_weight) for weight in weights)
                ),
                "maximum_seen_direction_cosine": float(
                    max(parameter_direction_cosine(weight, minimal_target_weight) for weight in weights)
                ),
            }

            selection_aspect_grid = build_aspect_grid(
                validation, seen, train_variants, resource
            )
            selection_sentiment_grid = build_sentiment_grid(
                validation, seen, train_variants, resource
            )
            selection_aspect_scores = direct_seen_scores.reshape(-1)
            selection_sentiment_scores = sentiment_scorer.score_manifest(
                selection_sentiment_grid
            )
            selection_scored = join_two_stage_scores(
                selection_aspect_grid,
                selection_sentiment_grid,
                selection_aspect_scores,
                selection_sentiment_scores,
            )
            runner_selection = select_second_sentiment_threshold(
                selection_scored,
                aspect_threshold=float(selected_config["aspect_threshold"]),
            )

            condition_results: dict[str, object] = {}
            seen_score_hash: str | None = None
            for condition in ("N", "D", "R"):
                variants_raw = candidate_representation_variants(fold, condition)
                variants = {
                    aspect: {
                        "name_only": "name_only",
                        "minimal": "name_and_description",
                        "rich": "rich",
                    }[variant]
                    for aspect, variant in variants_raw.items()
                }
                aspect_grid = build_aspect_grid(
                    evaluation, fold.evaluation_aspects, variants, resource
                )
                sentiment_grid = build_sentiment_grid(
                    evaluation, fold.evaluation_aspects, variants, resource
                )
                target_weight = synthesise_weight(
                    seen_descriptors,
                    weights,
                    target_embeddings[(heldout, condition)],
                    family=family,
                    **kwargs,
                )
                target_scores = sigmoid(
                    validation_features @ target_weight[:-1] + target_weight[-1]
                )
                score_by_aspect = {
                    aspect: direct_seen_scores[:, index]
                    for index, aspect in enumerate(seen)
                }
                score_by_aspect[heldout] = target_scores
                aspect_score_matrix = np.column_stack(
                    [score_by_aspect[aspect] for aspect in fold.evaluation_aspects]
                )
                sentiment_scores = sentiment_scorer.score_manifest(sentiment_grid)
                scored = join_two_stage_scores(
                    aspect_grid,
                    sentiment_grid,
                    aspect_score_matrix.reshape(-1),
                    sentiment_scores,
                )
                seen_scores = scored[
                    scored["candidate_aspect"].isin(seen)
                ][["row_uid", "candidate_aspect", "aspect_score", "sentiment_score"]]
                observed_seen_hash = hashlib.sha256(
                    pd.util.hash_pandas_object(
                        seen_scores.sort_values(
                            ["row_uid", "candidate_aspect"], kind="stable"
                        ),
                        index=False,
                    ).to_numpy(dtype=np.uint64).tobytes()
                ).hexdigest()
                if seen_score_hash is None:
                    seen_score_hash = observed_seen_hash
                elif seen_score_hash != observed_seen_hash:
                    raise AssertionError("Seen scores changed across N/D/R conditions.")
                primary_mask = capped_two_sentiment_prediction_mask(
                    scored,
                    aspect_threshold=float(selected_config["aspect_threshold"]),
                    second_sentiment_threshold=float(
                        runner_selection.second_sentiment_threshold
                    ),
                )
                if (
                    args.evidence_output_root is not None
                    and family == "kernel_ridge"
                    and condition in {"N", "D"}
                ):
                    evidence = row_pair_confusions(
                        scored,
                        primary_mask,
                        aspects=fold.heldout_aspects,
                        method_id=(
                            "description_conditioned_weight_transfer_kernel_ridge"
                        ),
                        fold_id=fold.fold_id,
                        condition=condition,
                    )
                    _write_csv(
                        args.evidence_output_root.resolve()
                        / "description_conditioned_weight_transfer_kernel_ridge"
                        / fold.fold_id
                        / f"{condition}.csv",
                        evidence,
                    )
                condition_results[condition] = evaluate_l2_condition(
                    scored,
                    fold,
                    aspect_threshold=float(selected_config["aspect_threshold"]),
                    second_sentiment_threshold=float(
                        runner_selection.second_sentiment_threshold
                    ),
                )

            generator_results[family] = {
                "selection": selected_config,
                "second_sentiment_threshold": float(
                    runner_selection.second_sentiment_threshold
                ),
                "second_sentiment_selection": runner_selection.selection_metrics,
                "space_diagnostics": space_diagnostics,
                "parameter_diagnostics": parameter_diagnostics,
                "selection_candidate_count": len(selection_records[family]),
                "seen_score_sha256": seen_score_hash,
                "conditions": condition_results,
            }

        result: dict[str, object] = {
            "schema_version": "taxonomy_description_weight_transfer_fold_v1",
            "protocol_id": config["protocol_id"],
            "protocol_sha256": _sha256(CONFIG_PATH),
            "fold_id": fold.fold_id,
            "training_scope_id": training_scope_id(fold),
            "heldout_aspect": heldout,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "target_labels_used_for_selection": False,
            "target_classifier_weight_used_for_selection": False,
            "generators": generator_results,
            "failure_count": 0,
            "non_finite_value_count": 0,
            "resume_conflict_count": 0,
            "test_contract_count": 0,
            "seconds": float(time.perf_counter() - started),
        }
        _write_json(result_path, result)
        fold_results.append(result)
        print(
            f"COMPLETE dcwt {fold.fold_id} seconds={result['seconds']:.1f}",
            flush=True,
        )

    summary: dict[str, object] = {
        "schema_version": "taxonomy_description_weight_transfer_summary_v1",
        "protocol_id": config["protocol_id"],
        "protocol_sha256": _sha256(CONFIG_PATH),
        "completed_folds": len(fold_results),
        "expected_folds": len(folds),
        "folds": [result["fold_id"] for result in fold_results],
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "test_contract_count": 0,
        "seconds": float(time.perf_counter() - started_all),
    }
    if len(fold_results) == 12:
        summary["generator_aggregates"] = {
            family: aggregate_l2_folds(
                [
                    {
                        "fold_id": fold_result["fold_id"],
                        "conditions": fold_result["generators"][family]["conditions"],
                    }
                    for fold_result in fold_results
                ]
            )
            for family in GENERATOR_FAMILIES
        }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--fold-id")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evidence-output-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_post_supervisor_local_v1"
        / "dcwt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
