"""Frozen method adapters for final-test score generation.

The adapters consume unlabelled evaluation rows and train-only labels.  They do
not decode predictions or compute metrics; their sole output is the immutable,
label-free score grid used by the score-seal-reveal workflow.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from msc_project.experiments.taxonomy_final_test import FinalTestJob
from msc_project.experiments.taxonomy_final_test_artifacts import FrozenJobArtifacts
from msc_project.experiments.taxonomy_final_test_workflow import (
    build_unlabelled_grids,
    canonical_l2_nd_scores,
    score_frame_from_grids,
)
from msc_project.experiments.taxonomy_two_stage_formal import (
    representative_fold,
    scope_folds,
)


def final_fold(job: FinalTestJob):
    folds = scope_folds().get(job.training_scope_id)
    if folds is None:
        raise ValueError(f"Unknown frozen training scope: {job.training_scope_id!r}.")
    matches = [value for value in folds if value.fold_id == job.fold_id]
    fold = matches[0] if len(matches) == 1 else representative_fold(folds)
    if fold.level != job.level or fold.fold_id != job.fold_id:
        raise ValueError("Final-test job does not match its registered fold.")
    return replace(fold, conditions=tuple(job.conditions))


def example_filtered_train_rows(
    train_rows: pd.DataFrame, heldout_aspects: Sequence[str]
) -> pd.DataFrame:
    required = {"row_uid", "text", "labels"}
    missing = sorted(required - set(train_rows.columns))
    if missing or train_rows.empty:
        raise ValueError(f"Train-only source is invalid; missing={missing}.")
    if "original_split" in train_rows and set(
        train_rows["original_split"].astype(str)
    ) != {"train"}:
        raise ValueError("Final-test training source may contain only the train split.")
    heldout = {str(value) for value in heldout_aspects}
    keep = ~train_rows["labels"].map(
        lambda values: any(str(aspect) in heldout for aspect, _ in values)
    )
    result = train_rows.loc[keep].copy()
    result["supervision_labels"] = result["labels"]
    survived = {
        str(aspect)
        for values in result["supervision_labels"]
        for aspect, _ in values
    }
    if survived.intersection(heldout):
        raise AssertionError("A held-out aspect survived example-filtered training.")
    if result.empty or result["row_uid"].astype(str).duplicated().any():
        raise ValueError("Example-filtered training rows are empty or duplicated.")
    return result.assign(_uid=result["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    ).drop(columns="_uid").reset_index(drop=True)


class _E5Scorer:
    def __init__(self, reviews: pd.DataFrame, *, local_files_only: bool) -> None:
        from msc_project.baselines.candidate_similarity import (
            REGISTERED_CONFIGS,
            FrozenTransformerSentenceEncoder,
        )

        self.encoder = FrozenTransformerSentenceEncoder(
            REGISTERED_CONFIGS["e5_base_v2"],
            device="auto",
            local_files_only=local_files_only,
        )
        ordered = reviews.assign(_uid=reviews["row_uid"].astype(str)).sort_values(
            "_uid", kind="stable"
        )
        self.review_index = {
            str(uid): index for index, uid in enumerate(ordered["row_uid"])
        }
        self.review_embeddings = self.encoder.encode(
            ordered["text"].astype(str).tolist(), role="review"
        )
        self.candidates: dict[str, np.ndarray] = {}

    def score(self, grid: pd.DataFrame) -> np.ndarray:
        cards = grid["candidate_text"].astype(str).tolist()
        missing = sorted(set(cards) - set(self.candidates))
        if missing:
            values = self.encoder.encode(missing, role="candidate")
            self.candidates.update(dict(zip(missing, values)))
        similarities = np.asarray(
            [
                float(
                    self.review_embeddings[self.review_index[str(uid)]]
                    @ self.candidates[card]
                )
                for uid, card in zip(grid["row_uid"], cards)
            ],
            dtype=float,
        )
        return (np.clip(similarities, -1.0, 1.0) + 1.0) / 2.0

    def close(self) -> None:
        self.encoder.close()


def _release_runtime(value: Any) -> None:
    try:
        if isinstance(value, tuple):
            for item in value:
                del item
        elif hasattr(value, "close"):
            value.close()
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _qwen_scores(
    runtime: tuple[Any, Any, Any],
    aspect_grid: pd.DataFrame,
    *,
    maximum_length: int,
    batch_size: int,
    demonstrations: Mapping[str, Sequence[Any]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    from msc_project.llm.qwen_two_stage_classifier import (
        score_two_stage_prompts,
        validate_sentiment_verbalizer_token_ids,
    )

    tokenizer, model, aspect_ids = runtime
    sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)
    reviews = aspect_grid["text"].astype(str).tolist()
    candidates = aspect_grid["candidate_text"].astype(str).tolist()
    aspect = score_two_stage_prompts(
        model,
        tokenizer,
        reviews,
        candidates,
        mode="aspect",
        max_length=maximum_length,
        batch_size=batch_size,
        aspect_verbalizer_ids=aspect_ids,
        demonstrations=None if demonstrations is None else demonstrations["aspect"],
    )
    sentiment = score_two_stage_prompts(
        model,
        tokenizer,
        reviews,
        candidates,
        mode="sentiment",
        max_length=maximum_length,
        batch_size=batch_size,
        sentiment_verbalizer_ids=sentiment_ids,
        demonstrations=(
            None if demonstrations is None else demonstrations["sentiment"]
        ),
    )
    if np.asarray(aspect).shape != (len(aspect_grid), 2) or np.asarray(
        sentiment
    ).shape != (len(aspect_grid), 3):
        raise ValueError("Frozen Qwen returned an unexpected two-stage score shape.")
    return np.asarray(aspect, dtype=float)[:, 0], np.asarray(
        sentiment, dtype=float
    ).reshape(-1)


def _assert_few_shot_contract(
    train: pd.DataFrame,
    seen_aspects: Sequence[str],
    resource: Mapping[str, object],
    selection: Mapping[str, object],
) -> Mapping[str, Sequence[Any]]:
    from msc_project.experiments.taxonomy_two_stage_runtime import (
        select_qwen_two_stage_demonstrations,
    )
    from msc_project.llm.qwen_two_stage_classifier import demonstrations_sha256

    demonstrations = select_qwen_two_stage_demonstrations(
        train, seen_aspects, resource, seed=13
    )
    expected = selection.get("demonstration_sha256s")
    if not isinstance(expected, Mapping):
        raise TypeError("Frozen few-shot selection lacks demonstration hashes.")
    observed = {
        mode: demonstrations_sha256(values)
        for mode, values in demonstrations.items()
    }
    if observed != dict(expected):
        raise ValueError("Train-only few-shot demonstrations changed since validation.")
    return demonstrations


def _load_neural_runtime(
    method_id: str,
    artifacts: FrozenJobArtifacts,
    *,
    local_files_only: bool,
) -> Any:
    if method_id == "distilbert_review_candidate_cross_encoder":
        import torch

        from msc_project.experiments.taxonomy_two_stage_distilbert import (
            DistilBertTrueTwoStageRuntime,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Final DistilBERT scoring requires CUDA.")
        if artifacts.checkpoint_path is None:
            raise ValueError("DistilBERT checkpoint is missing.")
        return DistilBertTrueTwoStageRuntime.from_pretrained(
            artifacts.checkpoint_path,
            device=torch.device("cuda"),
            local_files_only=True,
        )
    from msc_project.experiments.taxonomy_methods import resolve_method_spec
    from msc_project.llm.qwen_pair_classifier import (
        load_frozen_qwen_pair,
        load_saved_qwen_pair_adapter,
    )

    source_id = (
        "frozen_qwen_candidate_pair"
        if method_id == "frozen_qwen_few_shot"
        else method_id
    )
    spec = resolve_method_spec(source_id)
    if not spec.model_id or not spec.model_revision:
        raise ValueError("Frozen Qwen model identity is not pinned.")
    if method_id in {"frozen_qwen_candidate_pair", "frozen_qwen_few_shot"}:
        return load_frozen_qwen_pair(
            spec.model_id,
            revision=spec.model_revision,
            load_in_4bit=True,
            local_files_only=local_files_only,
        )
    if artifacts.checkpoint_path is None:
        raise ValueError("QLoRA checkpoint is missing.")
    return load_saved_qwen_pair_adapter(
        spec.model_id,
        artifacts.checkpoint_path / "adapter",
        revision=spec.model_revision,
        local_files_only=local_files_only,
    )


def _dcwt_runtime(
    train: pd.DataFrame,
    reviews: pd.DataFrame,
    fold: Any,
    resource: Mapping[str, object],
    artifacts: FrozenJobArtifacts,
    *,
    local_files_only: bool,
) -> tuple[Any, dict[str, Any]]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import StandardScaler

    from msc_project.baselines.candidate_similarity import (
        REGISTERED_CONFIGS,
        FrozenTransformerSentenceEncoder,
    )
    from msc_project.baselines.unified_pair_scorers import (
        UnifiedTfidfPairConfig,
        UnifiedTfidfPairScorer,
    )
    from msc_project.experiments.taxonomy_description_weight_transfer import (
        synthesise_weight,
    )
    from msc_project.experiments.taxonomy_two_stage_runtime import (
        build_sentiment_grid,
        render_aspect_candidate,
    )

    generators = artifacts.selection.get("generators")
    primary = generators.get("kernel_ridge") if isinstance(generators, Mapping) else None
    selected = primary.get("selection") if isinstance(primary, Mapping) else None
    if not isinstance(selected, Mapping):
        raise TypeError("DCWT frozen kernel-ridge selection is unavailable.")
    components = int(selected["svd_components"])
    classifier_c = float(selected["classifier_c"])
    alpha = float(selected["generator_parameter"])

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
    review_sparse = vectoriser.transform(reviews["text"].astype(str).tolist())
    effective = min(components, int(train_sparse.shape[1] - 1))
    projector = TruncatedSVD(
        n_components=effective, algorithm="randomized", random_state=13
    )
    train_latent = projector.fit_transform(train_sparse)
    review_latent = projector.transform(review_sparse)
    scaler = StandardScaler().fit(train_latent)
    train_features = scaler.transform(train_latent)
    review_features = scaler.transform(review_latent)

    seen = tuple(fold.seen_aspects)
    targets = np.asarray(
        [
            [
                int(aspect in {str(value) for value, _ in labels})
                for aspect in seen
            ]
            for labels in train["supervision_labels"]
        ],
        dtype=np.uint8,
    )
    weights: list[np.ndarray] = []
    for column in range(targets.shape[1]):
        classifier = LogisticRegression(
            C=classifier_c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=13,
        ).fit(train_features, targets[:, column].astype(int))
        weights.append(
            np.concatenate(
                [classifier.coef_.reshape(-1), classifier.intercept_.reshape(-1)]
            )
        )
    weight_matrix = np.vstack(weights)
    encoder = FrozenTransformerSentenceEncoder(
        REGISTERED_CONFIGS["e5_base_v2"],
        device="auto",
        local_files_only=local_files_only,
    )
    minimal_cards = [
        render_aspect_candidate(aspect, "name_and_description", resource)
        for aspect in seen
    ]
    seen_embeddings = encoder.encode(minimal_cards, role="candidate")
    target_cards = {
        (aspect, condition): render_aspect_candidate(
            aspect,
            "name_and_description" if condition == "D" else "name_only",
            resource,
        )
        for aspect in fold.heldout_aspects
        for condition in ("D", "N")
    }
    target_values = encoder.encode(list(target_cards.values()), role="candidate")
    target_embeddings = dict(zip(target_cards, target_values))
    encoder.close()
    direct_seen = np.column_stack(
        [1.0 / (1.0 + np.exp(-(review_features @ weight[:-1] + weight[-1]))) for weight in weight_matrix]
    )
    target_weights = {
        key: synthesise_weight(
            seen_embeddings,
            weight_matrix,
            embedding,
            family="kernel_ridge",
            alpha=alpha,
        )
        for key, embedding in target_embeddings.items()
    }
    train_variants = {aspect: "name_and_description" for aspect in seen}
    sentiment = UnifiedTfidfPairScorer(
        UnifiedTfidfPairConfig(
            classifier_c=1.0,
            feature_ablation="all_six",
            max_iter=1000,
            seed=13,
        )
    ).fit(
        build_sentiment_grid(
            train, seen, train_variants, resource, gold_aspects_only=True
        )
    )
    review_order = reviews.assign(_uid=reviews["row_uid"].astype(str)).sort_values(
        "_uid", kind="stable"
    )["row_uid"].astype(str).tolist()
    return sentiment, {
        "seen": seen,
        "review_order": review_order,
        "direct_seen": direct_seen,
        "target_weights": target_weights,
        "review_features": review_features,
    }


def _dcwt_scores(
    runtime: tuple[Any, Mapping[str, Any]],
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    *,
    condition: str,
    heldout_aspects: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    sentiment, values = runtime
    seen = tuple(values["seen"])
    review_order = list(values["review_order"])
    direct = np.asarray(values["direct_seen"], dtype=float)
    review_features = np.asarray(values["review_features"], dtype=float)
    score_by_aspect = {
        aspect: direct[:, index] for index, aspect in enumerate(seen)
    }
    for aspect in heldout_aspects:
        weight = values["target_weights"][(aspect, condition)]
        score_by_aspect[aspect] = 1.0 / (
            1.0 + np.exp(-(review_features @ weight[:-1] + weight[-1]))
        )
    review_index = {uid: index for index, uid in enumerate(review_order)}
    aspect_scores = np.asarray(
        [
            score_by_aspect[str(aspect)][review_index[str(uid)]]
            for uid, aspect in zip(
                aspect_grid["row_uid"], aspect_grid["candidate_aspect"]
            )
        ],
        dtype=float,
    )
    sentiment_scores = sentiment.score_manifest(sentiment_grid)
    return aspect_scores, np.asarray(sentiment_scores, dtype=float)


def score_base_job(
    job: FinalTestJob,
    artifacts: FrozenJobArtifacts,
    train_rows: pd.DataFrame,
    unlabelled_reviews: pd.DataFrame,
    resource: Mapping[str, object],
    *,
    local_files_only: bool,
) -> pd.DataFrame:
    """Score one frozen base job without accessing an evaluation target."""

    if job.phase != "score" or artifacts.job != job:
        raise ValueError("Scoring artifacts do not match the frozen score job.")
    fold = final_fold(job)
    train = example_filtered_train_rows(train_rows, fold.heldout_aspects)
    runtime: Any = None
    demonstrations: Mapping[str, Sequence[Any]] | None = None
    method = job.method_id
    if method == "strict_train_only_tfidf":
        from msc_project.experiments.taxonomy_two_stage_runtime import (
            TfidfTrueTwoStageRuntime,
            build_aspect_grid,
            build_sentiment_grid,
        )

        variants = {aspect: "name_and_description" for aspect in fold.seen_aspects}
        runtime = TfidfTrueTwoStageRuntime(seed=13).fit(
            build_aspect_grid(train, fold.seen_aspects, variants, resource),
            build_sentiment_grid(
                train,
                fold.seen_aspects,
                variants,
                resource,
                gold_aspects_only=True,
            ),
        )
    elif method == "e5_base_v2":
        runtime = _E5Scorer(unlabelled_reviews, local_files_only=local_files_only)
    elif method == "description_to_classifier_weight_transfer":
        runtime = _dcwt_runtime(
            train,
            unlabelled_reviews,
            fold,
            resource,
            artifacts,
            local_files_only=local_files_only,
        )
    elif method in {
        "distilbert_review_candidate_cross_encoder",
        "frozen_qwen_candidate_pair",
        "frozen_qwen_few_shot",
        "qwen_candidate_pair_qlora",
    }:
        if method == "frozen_qwen_few_shot":
            demonstrations = _assert_few_shot_contract(
                train, fold.seen_aspects, resource, artifacts.selection
            )
        runtime = _load_neural_runtime(
            method, artifacts, local_files_only=local_files_only
        )
    else:
        raise ValueError(f"Unregistered final-test score method: {method!r}.")

    def score_condition(condition: str, *, heldout_only: bool) -> pd.DataFrame:
        aspect_grid, sentiment_grid = build_unlabelled_grids(
            unlabelled_reviews, fold, condition, resource
        )
        if heldout_only:
            aspect_grid = aspect_grid[aspect_grid["is_heldout"]].reset_index(drop=True)
            sentiment_grid = sentiment_grid[
                sentiment_grid["is_heldout"]
            ].reset_index(drop=True)
        if method == "strict_train_only_tfidf":
            aspect_scores = runtime.score_aspects(aspect_grid)
            sentiment_scores = runtime.score_sentiments(sentiment_grid)
        elif method == "e5_base_v2":
            aspect_scores = runtime.score(aspect_grid)
            sentiment_scores = runtime.score(sentiment_grid)
        elif method == "description_to_classifier_weight_transfer":
            aspect_scores, sentiment_scores = _dcwt_scores(
                runtime,
                aspect_grid,
                sentiment_grid,
                condition=condition,
                heldout_aspects=fold.heldout_aspects,
            )
        elif method == "distilbert_review_candidate_cross_encoder":
            aspect_scores = runtime.score_aspects(aspect_grid)
            sentiment_scores = np.asarray(
                runtime.score_sentiments(aspect_grid), dtype=float
            ).reshape(-1)
        else:
            maximum_length = 1024 if method == "frozen_qwen_few_shot" else 384
            batch_size = 6 if method == "frozen_qwen_few_shot" else 8
            aspect_scores, sentiment_scores = _qwen_scores(
                runtime,
                aspect_grid,
                maximum_length=maximum_length,
                batch_size=batch_size,
                demonstrations=demonstrations,
            )
        return score_frame_from_grids(
            job,
            condition,
            aspect_grid,
            sentiment_grid,
            aspect_scores,
            sentiment_scores,
        )

    try:
        d = score_condition("D", heldout_only=False)
        if job.level == "L2":
            n = score_condition("N", heldout_only=True)
            return canonical_l2_nd_scores(job, d, n)
        return d
    finally:
        _release_runtime(runtime)
