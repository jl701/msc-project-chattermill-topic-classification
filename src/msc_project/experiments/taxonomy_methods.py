"""Common probability-scoring interfaces and frozen method registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from msc_project.baselines.candidate_similarity import (
    FrozenTransformerSentenceEncoder,
    REGISTERED_CONFIGS as SIMILARITY_CONFIGS,
)
from msc_project.baselines.unified_pair_scorers import (
    UnifiedPairCrossEncoderConfig,
    UnifiedTfidfPairConfig,
    UnifiedTfidfPairScorer,
    fit_unified_pair_cross_encoder,
    score_unified_pair_manifest,
    validate_pair_manifest,
)
from msc_project.llm.qwen_pair_classifier import (
    QwenPairTrainingConfig,
    VerbalizerTokenIds,
    score_candidate_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METHOD_REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_method_registry_v1.json"
)
METHOD_IDS = (
    "strict_train_only_tfidf",
    "e5_base_v2",
    "distilbert_review_candidate_cross_encoder",
    "frozen_qwen_candidate_pair",
    "qwen_candidate_pair_qlora",
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_method_registry(path: Path | None = None) -> dict[str, object]:
    source = path or METHOD_REGISTRY_PATH
    if not source.is_file():
        raise FileNotFoundError(source)
    registry = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Method registry must contain a JSON object.")
    if registry.get("registry_id") != "taxonomy_method_registry_v1":
        raise ValueError("Unexpected taxonomy method registry_id.")
    methods = registry.get("methods")
    if not isinstance(methods, dict) or tuple(methods) != METHOD_IDS:
        raise ValueError("Method registry must preserve the five canonical methods in order.")
    orders = [methods[method_id].get("order") for method_id in METHOD_IDS]
    if orders != list(range(1, len(METHOD_IDS) + 1)):
        raise ValueError("Method registry order fields are inconsistent.")
    for method_id, payload in methods.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Method {method_id!r} must be an object.")
        if not payload.get("family") or not payload.get("thesis_role"):
            raise ValueError(f"Method {method_id!r} lacks family or thesis role.")
        if "starting_recipe" not in payload or "seen_only_tuning" not in payload:
            raise ValueError(f"Method {method_id!r} lacks its recipe or tuning contract.")
    qwen_frozen = methods["frozen_qwen_candidate_pair"]
    qwen_qlora = methods["qwen_candidate_pair_qlora"]
    for field in ("model_id", "model_revision"):
        if qwen_frozen.get(field) != qwen_qlora.get(field):
            raise ValueError(f"Frozen Qwen and QLoRA must share {field}.")
    return registry


def method_registry_sha256(registry: Mapping[str, object] | None = None) -> str:
    return _canonical_sha256(dict(registry or load_method_registry()))


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    family: str
    thesis_role: str
    requires_pair_training: bool
    model_id: str | None
    model_revision: str | None
    starting_recipe: dict[str, object]
    seen_only_tuning: dict[str, object]

    @property
    def spec_sha256(self) -> str:
        return _canonical_sha256(
            {
                "method_id": self.method_id,
                "family": self.family,
                "thesis_role": self.thesis_role,
                "requires_pair_training": self.requires_pair_training,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "starting_recipe": self.starting_recipe,
                "seen_only_tuning": self.seen_only_tuning,
            }
        )


def resolve_method_spec(
    method_id: str,
    registry: Mapping[str, object] | None = None,
) -> MethodSpec:
    source = dict(registry or load_method_registry())
    methods = source["methods"]
    if method_id not in methods:  # type: ignore[operator]
        raise ValueError(f"Unknown taxonomy method: {method_id!r}")
    payload = methods[method_id]  # type: ignore[index]
    return MethodSpec(
        method_id=method_id,
        family=str(payload["family"]),
        thesis_role=str(payload["thesis_role"]),
        requires_pair_training=bool(payload["requires_pair_training"]),
        model_id=str(payload["model_id"]) if payload.get("model_id") else None,
        model_revision=(
            str(payload["model_revision"]) if payload.get("model_revision") else None
        ),
        starting_recipe=dict(payload["starting_recipe"]),
        seen_only_tuning=dict(payload["seen_only_tuning"]),
    )


def tfidf_config_from_parameters(
    parameters: Mapping[str, object],
    *,
    seed: int = 13,
) -> UnifiedTfidfPairConfig:
    return UnifiedTfidfPairConfig(
        word_ngram_range=tuple(int(value) for value in parameters["word_ngram_range"]),
        char_ngram_range=tuple(int(value) for value in parameters["char_ngram_range"]),
        min_df=int(parameters["min_df"]),
        max_word_features=(
            int(parameters["max_word_features"])
            if parameters.get("max_word_features") is not None
            else None
        ),
        max_char_features=(
            int(parameters["max_char_features"])
            if parameters.get("max_char_features") is not None
            else None
        ),
        classifier_c=float(parameters["classifier_c"]),
        feature_ablation=str(parameters["feature_ablation"]),
        max_iter=int(parameters["max_iter"]),
        seed=seed,
    )


def distilbert_config_from_parameters(
    parameters: Mapping[str, object],
    spec: MethodSpec,
    *,
    seed: int = 13,
) -> UnifiedPairCrossEncoderConfig:
    if not spec.model_id or not spec.model_revision:
        raise ValueError("DistilBERT method spec must pin its model and revision.")
    selected_epoch = int(
        parameters.get("selected_checkpoint_epoch", parameters["epochs"])
    )
    if selected_epoch < 1 or selected_epoch > int(parameters["epochs"]):
        raise ValueError("Selected DistilBERT checkpoint epoch is outside the run.")
    return UnifiedPairCrossEncoderConfig(
        model_name=spec.model_id,
        model_revision=spec.model_revision,
        max_length=int(parameters["max_length"]),
        batch_size=int(parameters["batch_size"]),
        eval_batch_size=int(parameters["eval_batch_size"]),
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        epochs=selected_epoch,
        warmup_ratio=float(parameters["warmup_ratio"]),
        seed=seed,
        use_amp=bool(parameters["amp"]),
    )


def qlora_training_config_from_parameters(
    parameters: Mapping[str, object],
    *,
    seed: int = 13,
) -> QwenPairTrainingConfig:
    selected_epoch = int(
        parameters.get("selected_checkpoint_epoch", parameters["epochs"])
    )
    if selected_epoch < 1 or selected_epoch > int(parameters["epochs"]):
        raise ValueError("Selected QLoRA checkpoint epoch is outside the run.")
    return QwenPairTrainingConfig(
        max_length=int(parameters["max_length"]),
        batch_size=int(parameters["batch_size"]),
        gradient_accumulation_steps=int(
            parameters["gradient_accumulation_steps"]
        ),
        epochs=selected_epoch,
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        seed=seed,
    )


class PairProbabilityRuntime(Protocol):
    """Runtime boundary shared by all method families."""

    method_id: str

    def fit(self, train_manifest: pd.DataFrame) -> "PairProbabilityRuntime":
        ...

    def score(self, pair_manifest: pd.DataFrame) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


def validate_probability_scores(
    scores: Sequence[float] | np.ndarray,
    *,
    expected_rows: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError(
            f"Method scores must have shape ({expected_rows},), got {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Method scores must all be finite.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Method scores must lie in [0,1].")
    return values


class StrictTfidfRuntime:
    method_id = "strict_train_only_tfidf"

    def __init__(self, config: UnifiedTfidfPairConfig | None = None) -> None:
        self.scorer = UnifiedTfidfPairScorer(config)
        self._fitted = False

    def fit(self, train_manifest: pd.DataFrame) -> "StrictTfidfRuntime":
        self.scorer.fit(train_manifest)
        self._fitted = True
        return self

    def score(self, pair_manifest: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Strict TF-IDF runtime must be fitted before scoring.")
        validate_pair_manifest(pair_manifest, require_target=False)
        return validate_probability_scores(
            self.scorer.score_manifest(pair_manifest),
            expected_rows=len(pair_manifest),
        )

    def close(self) -> None:
        return None


class E5PairRuntime:
    method_id = "e5_base_v2"

    def __init__(
        self,
        encoder: Any | None = None,
        *,
        config: Any | None = None,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.encoder = encoder or FrozenTransformerSentenceEncoder(
            config or SIMILARITY_CONFIGS["e5_base_v2"],
            device=device,
            local_files_only=local_files_only,
        )

    def fit(self, train_manifest: pd.DataFrame) -> "E5PairRuntime":
        # This is a frozen bi-encoder. The manifest is validated but never fitted.
        validate_pair_manifest(train_manifest, require_target=True)
        return self

    def score(self, pair_manifest: pd.DataFrame) -> np.ndarray:
        validate_pair_manifest(pair_manifest, require_target=False)
        review_values = list(dict.fromkeys(pair_manifest["text"].astype(str)))
        candidate_values = list(
            dict.fromkeys(pair_manifest["candidate_text"].astype(str))
        )
        matrix, _ = self.encoder.score_matrix(review_values, candidate_values)
        if matrix.shape != (len(review_values), len(candidate_values)):
            raise ValueError("E5 encoder returned an unexpected score-matrix shape.")
        review_index = {value: index for index, value in enumerate(review_values)}
        candidate_index = {
            value: index for index, value in enumerate(candidate_values)
        }
        # Cosine lies in [-1,1]. Map monotonically to the shared probability-like
        # [0,1] scale; threshold selection remains seen-only.
        cosine = np.asarray(
            [
                matrix[review_index[str(review)], candidate_index[str(candidate)]]
                for review, candidate in zip(
                    pair_manifest["text"],
                    pair_manifest["candidate_text"],
                )
            ],
            dtype=float,
        )
        return validate_probability_scores(
            (np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0,
            expected_rows=len(pair_manifest),
        )

    def close(self) -> None:
        close = getattr(self.encoder, "close", None)
        if callable(close):
            close()


class DistilBertPairRuntime:
    method_id = "distilbert_review_candidate_cross_encoder"

    def __init__(
        self,
        config: UnifiedPairCrossEncoderConfig | None = None,
        *,
        device: Any,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or UnifiedPairCrossEncoderConfig()
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.history: list[float] = []

    def fit(self, train_manifest: pd.DataFrame) -> "DistilBertPairRuntime":
        self.tokenizer, self.model, self.history = fit_unified_pair_cross_encoder(
            train_manifest,
            self.config,
            self.device,
            tokenizer=self.tokenizer,
            model=self.model,
        )
        return self

    def score(self, pair_manifest: pd.DataFrame) -> np.ndarray:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("DistilBERT runtime must be fitted before scoring.")
        validate_pair_manifest(pair_manifest, require_target=False)
        return validate_probability_scores(
            score_unified_pair_manifest(
                self.model,
                self.tokenizer,
                pair_manifest,
                self.config,
                self.device,
            ),
            expected_rows=len(pair_manifest),
        )

    def close(self) -> None:
        if self.model is not None:
            try:
                self.model.to("cpu")
            except (AttributeError, RuntimeError, ValueError):
                pass


class QwenPairRuntime:
    """Shared scorer for frozen Qwen and an already-trained QLoRA adapter."""

    def __init__(
        self,
        *,
        method_id: str,
        model: Any,
        tokenizer: Any,
        verbalizer_ids: VerbalizerTokenIds,
        max_length: int = 384,
        batch_size: int = 6,
        fit_callback: Callable[[pd.DataFrame], tuple[Any, Any, VerbalizerTokenIds]]
        | None = None,
        trained_adapter: bool = False,
    ) -> None:
        if method_id not in {
            "frozen_qwen_candidate_pair",
            "qwen_candidate_pair_qlora",
        }:
            raise ValueError("Qwen runtime received a non-Qwen method_id.")
        if (
            method_id == "qwen_candidate_pair_qlora"
            and fit_callback is None
            and not trained_adapter
        ):
            raise ValueError("QLoRA runtime requires an explicit fit_callback.")
        if method_id == "frozen_qwen_candidate_pair" and fit_callback is not None:
            raise ValueError("Frozen Qwen must not receive a training callback.")
        self.method_id = method_id
        self.model = model
        self.tokenizer = tokenizer
        self.verbalizer_ids = verbalizer_ids
        self.max_length = max_length
        self.batch_size = batch_size
        self.fit_callback = fit_callback
        self._ready = (
            method_id == "frozen_qwen_candidate_pair" or trained_adapter
        )

    def fit(self, train_manifest: pd.DataFrame) -> "QwenPairRuntime":
        validate_pair_manifest(train_manifest, require_target=True)
        if self.fit_callback is not None:
            self.model, self.tokenizer, self.verbalizer_ids = self.fit_callback(
                train_manifest
            )
        self._ready = True
        return self

    def score(self, pair_manifest: pd.DataFrame) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("QLoRA runtime must be fitted before scoring.")
        validate_pair_manifest(pair_manifest, require_target=False)
        scores = score_candidate_pairs(
            self.model,
            self.tokenizer,
            pair_manifest["text"].astype(str).tolist(),
            pair_manifest["candidate_text"].astype(str).tolist(),
            max_length=self.max_length,
            batch_size=self.batch_size,
            verbalizer_ids=self.verbalizer_ids,
        )
        return validate_probability_scores(scores, expected_rows=len(pair_manifest))

    def close(self) -> None:
        try:
            self.model.to("cpu")
        except (AttributeError, RuntimeError, ValueError):
            pass
