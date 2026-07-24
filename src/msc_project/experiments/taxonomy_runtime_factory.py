"""Construct, train, checkpoint, and reload the five registered runtimes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from msc_project.experiments.taxonomy_checkpoints import (
    TrainingContract,
    validate_checkpoint,
    write_checkpoint_atomic,
)
from msc_project.baselines.candidate_similarity import (
    REGISTERED_CONFIGS as SIMILARITY_CONFIGS,
)
from msc_project.experiments.taxonomy_methods import (
    DistilBertPairRuntime,
    E5PairRuntime,
    PairProbabilityRuntime,
    QwenPairRuntime,
    StrictTfidfRuntime,
    distilbert_config_from_parameters,
    qlora_training_config_from_parameters,
    resolve_method_spec,
    tfidf_config_from_parameters,
)
from msc_project.llm.qwen_pair_classifier import (
    QwenPairQLoRAConfig,
    load_frozen_qwen_pair,
    load_qwen_pair_qlora,
    load_saved_qwen_pair_adapter,
    train_qwen_pair_adapter,
)


def create_runtime_for_training(
    method_id: str,
    parameters: Mapping[str, object],
    *,
    seed: int,
    device: Any,
    local_files_only: bool = False,
) -> PairProbabilityRuntime:
    spec = resolve_method_spec(method_id)
    if method_id == "strict_train_only_tfidf":
        return StrictTfidfRuntime(
            tfidf_config_from_parameters(parameters, seed=seed)
        )
    if method_id == "e5_base_v2":
        e5_config = replace(
            SIMILARITY_CONFIGS["e5_base_v2"],
            max_length=int(parameters["max_length"]),
            batch_size=int(parameters["batch_size"]),
            review_prefix=str(parameters["review_prefix"]),
            candidate_prefix=str(parameters["candidate_prefix"]),
        )
        return E5PairRuntime(
            config=e5_config,
            device=str(device),
            local_files_only=local_files_only,
        )
    if method_id == "distilbert_review_candidate_cross_encoder":
        return DistilBertPairRuntime(
            distilbert_config_from_parameters(parameters, spec, seed=seed),
            device=device,
        )
    if method_id == "frozen_qwen_candidate_pair":
        if not spec.model_id or not spec.model_revision:
            raise ValueError("Frozen Qwen spec must pin model ID and revision.")
        tokenizer, model, ids = load_frozen_qwen_pair(
            spec.model_id,
            revision=spec.model_revision,
            load_in_4bit=bool(parameters["load_in_4bit"]),
        )
        return QwenPairRuntime(
            method_id=method_id,
            model=model,
            tokenizer=tokenizer,
            verbalizer_ids=ids,
            max_length=int(parameters["max_length"]),
            batch_size=int(parameters["eval_batch_size"]),
        )
    if method_id == "qwen_candidate_pair_qlora":
        if not spec.model_id or not spec.model_revision:
            raise ValueError("QLoRA spec must pin model ID and revision.")
        adapter_config = QwenPairQLoRAConfig(
            load_in_4bit=bool(parameters["load_in_4bit"]),
            gradient_checkpointing=True,
            lora_r=int(parameters["lora_r"]),
            lora_alpha=int(parameters["lora_alpha"]),
            lora_dropout=float(parameters["lora_dropout"]),
            target_modules=tuple(
                str(value) for value in parameters["lora_target_modules"]
            ),
        )
        tokenizer, model, ids = load_qwen_pair_qlora(
            spec.model_id,
            adapter_config,
            revision=spec.model_revision,
        )
        training_config = qlora_training_config_from_parameters(
            parameters,
            seed=seed,
        )
        runtime: QwenPairRuntime

        def fit_callback(
            manifest: pd.DataFrame,
        ) -> tuple[Any, Any, Any]:
            pairs = [
                (str(row.text), str(row.candidate_text), int(row.target))
                for row in manifest.itertuples(index=False)
            ]
            history = train_qwen_pair_adapter(
                model,
                tokenizer,
                ids,
                pairs,
                training_config,
            )
            runtime.training_history = history
            model.eval()
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True
            return model, tokenizer, ids

        runtime = QwenPairRuntime(
            method_id=method_id,
            model=model,
            tokenizer=tokenizer,
            verbalizer_ids=ids,
            max_length=int(parameters["max_length"]),
            batch_size=int(parameters.get("eval_batch_size", 6)),
            fit_callback=fit_callback,
        )
        runtime.training_history = []
        return runtime
    raise ValueError(f"Unknown taxonomy method: {method_id!r}")


def save_runtime_checkpoint(
    runtime: PairProbabilityRuntime,
    checkpoint_dir: Path,
    contract: TrainingContract,
) -> Path:
    if runtime.method_id != contract.method_id:
        raise ValueError("Runtime and training-contract methods differ.")

    def writer(path: Path) -> None:
        if runtime.method_id == "strict_train_only_tfidf":
            import joblib

            joblib.dump(runtime.scorer, path / "tfidf_scorer.joblib")  # type: ignore[attr-defined]
        elif runtime.method_id == "distilbert_review_candidate_cross_encoder":
            runtime.model.save_pretrained(path / "model")  # type: ignore[attr-defined]
            runtime.tokenizer.save_pretrained(path / "model")  # type: ignore[attr-defined]
        elif runtime.method_id == "qwen_candidate_pair_qlora":
            runtime.model.save_pretrained(path / "adapter")  # type: ignore[attr-defined]
            runtime.tokenizer.save_pretrained(path / "adapter")  # type: ignore[attr-defined]
        else:
            (path / "reload.json").write_text(
                json.dumps(
                    {
                        "method_id": runtime.method_id,
                        "reload_from_pinned_registry": True,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )

    history = getattr(runtime, "history", None)
    if history is None:
        history = getattr(runtime, "training_history", [])
    return write_checkpoint_atomic(
        checkpoint_dir,
        contract,
        writer,
        evidence={"training_history": history},
    )


def save_frozen_registry_checkpoint(
    method_id: str,
    checkpoint_dir: Path,
    contract: TrainingContract,
) -> Path:
    """Create a reload marker for a method with no task-specific fitting."""

    spec = resolve_method_spec(method_id)
    if spec.requires_pair_training:
        raise ValueError("A trainable method cannot use a frozen reload marker.")
    if method_id != contract.method_id:
        raise ValueError("Method and training-contract IDs differ.")

    def writer(path: Path) -> None:
        (path / "reload.json").write_text(
            json.dumps(
                {
                    "method_id": method_id,
                    "reload_from_pinned_registry": True,
                    "task_specific_fit_performed": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return write_checkpoint_atomic(
        checkpoint_dir,
        contract,
        writer,
        evidence={
            "training_history": [],
            "task_specific_fit_performed": False,
        },
    )


def load_runtime_checkpoint(
    method_id: str,
    parameters: Mapping[str, object],
    checkpoint_dir: Path,
    contract: TrainingContract,
    *,
    seed: int,
    device: Any,
    local_files_only: bool = False,
) -> PairProbabilityRuntime:
    validate_checkpoint(checkpoint_dir, contract)
    spec = resolve_method_spec(method_id)
    if method_id == "strict_train_only_tfidf":
        import joblib

        runtime = StrictTfidfRuntime(
            tfidf_config_from_parameters(parameters, seed=seed)
        )
        runtime.scorer = joblib.load(checkpoint_dir / "tfidf_scorer.joblib")
        runtime._fitted = True
        return runtime
    if method_id == "distilbert_review_candidate_cross_encoder":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = checkpoint_dir / "model"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )
        runtime = DistilBertPairRuntime(
            distilbert_config_from_parameters(parameters, spec, seed=seed),
            device=device,
            tokenizer=tokenizer,
            model=model,
        )
        model.to(device)
        return runtime
    if method_id == "qwen_candidate_pair_qlora":
        if not spec.model_id or not spec.model_revision:
            raise ValueError("QLoRA spec must pin model ID and revision.")
        tokenizer, model, ids = load_saved_qwen_pair_adapter(
            spec.model_id,
            checkpoint_dir / "adapter",
            revision=spec.model_revision,
        )
        return QwenPairRuntime(
            method_id=method_id,
            model=model,
            tokenizer=tokenizer,
            verbalizer_ids=ids,
            max_length=int(parameters["max_length"]),
            batch_size=int(parameters.get("eval_batch_size", 6)),
            trained_adapter=True,
        )
    # Frozen methods intentionally reload their pinned immutable weights.
    return create_runtime_for_training(
        method_id,
        parameters,
        seed=seed,
        device=device,
        local_files_only=local_files_only,
    )
