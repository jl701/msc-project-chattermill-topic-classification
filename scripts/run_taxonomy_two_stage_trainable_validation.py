"""Execute the preregistered trainable genuine-two-stage validation graph.

Only official train and validation splits are loadable.  Each learning-rate
candidate produces an immutable checkpoint and seen-validation selection
record.  The selected checkpoint then writes content-addressed validation
score shards for every registered fold/condition sharing that training scope.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.data.fabsa import default_data_dir  # noqa: E402
from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_checkpoints import (  # noqa: E402
    TrainingContract,
    checkpoint_resume_state,
    validate_checkpoint,
    write_checkpoint_atomic,
)
from msc_project.experiments.taxonomy_execution import (  # noqa: E402
    RunContract,
    canonical_sha256,
    dataframe_sha256,
    select_pair_shard,
)
from msc_project.experiments.taxonomy_methods import (  # noqa: E402
    method_registry_sha256,
    resolve_method_spec,
)
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    pair_identity_hash,
)
from msc_project.experiments.taxonomy_post_supervisor import (  # noqa: E402
    evaluate_l2_condition,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage import (  # noqa: E402
    capped_two_sentiment_prediction_mask,
    conditional_sentiment_metrics,
    evaluate_prediction_mask,
    select_second_sentiment_threshold,
    select_two_stage_threshold,
)
from msc_project.experiments.taxonomy_two_stage_artifacts import (  # noqa: E402
    build_two_stage_score_artifact,
    merge_two_stage_score_shards,
    two_stage_shard_resume_state,
    write_two_stage_score_shard,
)
from msc_project.experiments.taxonomy_two_stage_distilbert import (  # noqa: E402
    DistilBertTrueTwoStageRuntime,
    TwoStageDistilBertConfig,
)
from msc_project.experiments.taxonomy_two_stage_formal import (  # noqa: E402
    PROTOCOL_ID,
    TRAINABLE_METHODS,
    build_formal_jobs,
    candidate_result_path,
    learning_rate_token,
    learning_rates,
    plan_payload,
    representative_fold,
    scope_folds,
    selection_path,
    variant_map,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    build_aspect_grid,
    build_sentiment_grid,
    join_two_stage_scores,
)
from msc_project.experiments.taxonomy_two_stage_training import (  # noqa: E402
    build_two_stage_training_manifests,
    training_manifest_sha256,
    training_manifest_summary,
)
from msc_project.experiments.verified_artifact_sync import (  # noqa: E402
    file_sha256,
    publish_artifact_unit,
)
from msc_project.llm.qwen_pair_classifier import (  # noqa: E402
    QwenPairQLoRAConfig,
    QwenPairTrainingConfig,
    load_qwen_pair_qlora,
    load_saved_qwen_pair_adapter,
)
from msc_project.llm.qwen_two_stage_classifier import (  # noqa: E402
    qwen_two_stage_contract_sha256,
    score_two_stage_prompts,
    train_qwen_two_stage_adapter,
    validate_sentiment_verbalizer_token_ids,
)


DEFAULT_SAFETY_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "taxonomy_two_stage_cloud_execution_safety_v1.json"
)
PHASES = ("plan", "train-candidate", "select-scope", "score-selected")


def formal_conditions(fold):
    """Default legacy condition set; post-supervisor wrappers may narrow it."""

    return fold.conditions


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary JSON file: {temporary}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _seal_payload(value: Mapping[str, object], field: str) -> dict[str, object]:
    payload = dict(value)
    payload.pop(field, None)
    payload[field] = canonical_sha256(payload)
    return payload


def _validate_sealed_payload(value: Mapping[str, object], field: str) -> None:
    expected = str(value.get(field, ""))
    payload = dict(value)
    payload.pop(field, None)
    if expected != canonical_sha256(payload):
        raise RuntimeError(f"JSON artifact content hash mismatch: {field}")


ALLOWED_SAFETY_STATUSES = {
    "preregistered_before_formal_execution",
    "preregistered_after_rich_gate_before_formal_execution",
}


def _validate_safety_config(config: Mapping[str, object]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected formal two-stage protocol ID.")
    if config.get("status") not in ALLOWED_SAFETY_STATUSES:
        raise ValueError("Formal execution safety contract is not preregistered.")
    data = config.get("sealed_data_contract")
    if not isinstance(data, Mapping):
        raise ValueError("Formal execution safety contract lacks its data boundary.")
    if list(data.get("allowed_splits", [])) != ["train", "validation"]:
        raise ValueError("Formal executor may load only train and validation.")
    if data.get("include_official_test") is not False:
        raise ValueError("Official test must remain disabled.")


def _protocol_hash(config: Mapping[str, object]) -> str:
    scientific = dict(config)
    scientific.pop("status", None)
    scientific.pop("registered_at", None)
    return canonical_sha256(scientific)


def _combined_training_hash(manifests: Mapping[str, pd.DataFrame]) -> str:
    return canonical_sha256(
        {
            task: training_manifest_sha256(frame)
            for task, frame in sorted(manifests.items())
        }
    )


def _checkpoint_directory(
    output_root: Path,
    method_id: str,
    scope_id: str,
    learning_rate: float,
    contract: TrainingContract,
) -> Path:
    return (
        output_root
        / "checkpoints"
        / method_id
        / scope_id
        / f"lr-{learning_rate_token(learning_rate)}"
        / contract.contract_sha256[:16]
    )


def _training_parameters(method_id: str, learning_rate: float) -> dict[str, object]:
    if method_id == "distilbert_review_candidate_cross_encoder":
        return asdict(TwoStageDistilBertConfig(learning_rate=learning_rate))
    if method_id == "qwen_candidate_pair_qlora":
        return {
            "qlora": asdict(QwenPairQLoRAConfig()),
            "training": asdict(
                QwenPairTrainingConfig(
                    max_length=384,
                    batch_size=1,
                    gradient_accumulation_steps=8,
                    epochs=1,
                    learning_rate=learning_rate,
                    seed=13,
                )
            ),
            "prompt_contract_sha256": qwen_two_stage_contract_sha256(max_length=384),
            "shared_adapter_for_both_stages": True,
        }
    raise ValueError(f"Unsupported trainable two-stage method: {method_id!r}")


def _training_contract(
    *,
    config: Mapping[str, object],
    method_id: str,
    scope_id: str,
    manifests: Mapping[str, pd.DataFrame],
    learning_rate: float,
) -> TrainingContract:
    spec = resolve_method_spec(method_id)
    return TrainingContract(
        protocol_id=PROTOCOL_ID,
        scientific_protocol_sha256=_protocol_hash(config),
        method_id=method_id,
        method_spec_sha256=spec.spec_sha256,
        method_registry_sha256=method_registry_sha256(),
        training_scope_id=scope_id,
        seed=13,
        training_manifest_sha256=_combined_training_hash(manifests),
        scientific_parameters_sha256=canonical_sha256(
            _training_parameters(method_id, learning_rate)
        ),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        training_pairs=sum(len(frame) for frame in manifests.values()),
    )


def _load_scope_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], Any, tuple[Any, ...]]:
    folds = scope_folds().get(args.scope_id)
    if folds is None:
        raise ValueError(f"Unknown formal training scope: {args.scope_id!r}")
    fold = representative_fold(folds)
    frame = load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    if set(frame["original_split"].astype(str)) != {"train", "validation"}:
        raise AssertionError("Formal executor loaded an unregistered split.")
    splits = build_taxonomy_fold_splits(
        frame,
        fold,
        evaluation_splits=("validation",),
    )
    resource = load_description_bundle(require_approved=True)
    manifests = build_two_stage_training_manifests(
        splits["train"], fold.seen_aspects, resource, seed=13
    )
    return splits["train"], splits["validation"], manifests, resource, folds


def _release_runtime(runtime: Any) -> None:
    try:
        if hasattr(runtime, "close"):
            runtime.close()
        elif isinstance(runtime, tuple):
            model = runtime[1]
            try:
                model.to("cpu")
            except (AttributeError, ValueError):
                pass
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _train_runtime(
    method_id: str,
    manifests: Mapping[str, pd.DataFrame],
    learning_rate: float,
    *,
    local_files_only: bool,
) -> tuple[Any, dict[str, object]]:
    if method_id == "distilbert_review_candidate_cross_encoder":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Formal DistilBERT training requires CUDA.")
        runtime = DistilBertTrueTwoStageRuntime(
            TwoStageDistilBertConfig(learning_rate=learning_rate),
            device=device,
            local_files_only=local_files_only,
        ).fit(manifests["aspect_presence"], manifests["sentiment"])
        return runtime, {"history": runtime.history}

    spec = resolve_method_spec(method_id)
    if not spec.model_id or not spec.model_revision:
        raise ValueError("QLoRA model ID and revision must be pinned.")
    tokenizer, model, _ = load_qwen_pair_qlora(
        spec.model_id,
        QwenPairQLoRAConfig(),
        revision=spec.model_revision,
        local_files_only=local_files_only,
    )
    examples = [
        (
            str(row.text),
            str(row.candidate_text),
            "aspect" if str(row.task) == "aspect_presence" else "sentiment",
            str(row.answer),
        )
        for manifest in (manifests["aspect_presence"], manifests["sentiment"])
        for row in manifest.itertuples(index=False)
    ]
    history = train_qwen_two_stage_adapter(
        model,
        tokenizer,
        examples,
        QwenPairTrainingConfig(
            max_length=384,
            batch_size=1,
            gradient_accumulation_steps=8,
            epochs=1,
            learning_rate=learning_rate,
            seed=13,
        ),
    )
    return (tokenizer, model), {"history": history}


def _write_runtime_checkpoint(
    runtime: Any,
    method_id: str,
    destination: Path,
    parameters: Mapping[str, object],
) -> None:
    if method_id == "distilbert_review_candidate_cross_encoder":
        runtime.save_pretrained(destination)
        return
    tokenizer, model = runtime
    adapter = destination / "adapter"
    tokenizer_dir = destination / "tokenizer"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(tokenizer_dir)
    (destination / "two_stage_qlora_contract.json").write_text(
        json.dumps(dict(parameters), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _contract_from_dict(value: Mapping[str, object]) -> TrainingContract:
    names = {field.name for field in fields(TrainingContract)}
    return TrainingContract(**{name: value[name] for name in names})


def _load_checkpoint_runtime(
    method_id: str,
    checkpoint_dir: Path,
    *,
    local_files_only: bool,
) -> Any:
    if method_id == "distilbert_review_candidate_cross_encoder":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Formal DistilBERT scoring requires CUDA.")
        return DistilBertTrueTwoStageRuntime.from_pretrained(
            checkpoint_dir,
            device=torch.device("cuda"),
            local_files_only=True,
        )
    spec = resolve_method_spec(method_id)
    tokenizer, model, _ = load_saved_qwen_pair_adapter(
        str(spec.model_id),
        checkpoint_dir / "adapter",
        revision=spec.model_revision,
        local_files_only=local_files_only,
    )
    return tokenizer, model


def _score_runtime(method_id: str, runtime: Any, aspect_grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if method_id == "distilbert_review_candidate_cross_encoder":
        aspect_scores = runtime.score_aspects(aspect_grid)
        sentiment_matrix = runtime.score_sentiments(aspect_grid)
    else:
        tokenizer, model = runtime
        sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)
        aspect_matrix = score_two_stage_prompts(
            model,
            tokenizer,
            aspect_grid["text"].astype(str).tolist(),
            aspect_grid["candidate_text"].astype(str).tolist(),
            mode="aspect",
            max_length=384,
            batch_size=8,
        )
        sentiment_matrix = score_two_stage_prompts(
            model,
            tokenizer,
            aspect_grid["text"].astype(str).tolist(),
            aspect_grid["candidate_text"].astype(str).tolist(),
            mode="sentiment",
            max_length=384,
            batch_size=8,
            sentiment_verbalizer_ids=sentiment_ids,
        )
        aspect_scores = aspect_matrix[:, 0]
    aspect_scores = np.asarray(aspect_scores, dtype=float)
    sentiment_matrix = np.asarray(sentiment_matrix, dtype=float)
    if sentiment_matrix.shape != (len(aspect_grid), 3):
        raise ValueError("Two-stage sentiment head must return three probabilities per aspect.")
    if not np.isfinite(aspect_scores).all() or not np.isfinite(sentiment_matrix).all():
        raise RuntimeError("Non-finite formal two-stage probability.")
    return aspect_scores, sentiment_matrix.reshape(-1)


def _calibration_grid(
    validation_rows: pd.DataFrame,
    seen_aspects: tuple[str, ...],
    resource: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {aspect: "name_and_description" for aspect in seen_aspects}
    return (
        build_aspect_grid(validation_rows, seen_aspects, variants, resource),
        build_sentiment_grid(validation_rows, seen_aspects, variants, resource),
    )


def _checkpoint_files(output_root: Path, checkpoint_dir: Path) -> list[Path]:
    return sorted(
        path.relative_to(output_root)
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
    )


def train_candidate(args: argparse.Namespace, config: Mapping[str, object]) -> dict[str, object]:
    if args.learning_rate is None or args.learning_rate not in learning_rates(args.method):
        raise ValueError("--learning-rate must be one registered candidate.")
    result_path = candidate_result_path(
        args.output_root, args.method, args.scope_id, args.learning_rate
    )
    if result_path.is_file() and args.resume:
        value = _read_object(result_path)
        _validate_sealed_payload(value, "candidate_payload_sha256")
        if (
            value.get("protocol_id") != PROTOCOL_ID
            or value.get("method_id") != args.method
            or value.get("training_scope_id") != args.scope_id
            or float(value.get("learning_rate", -1)) != args.learning_rate
            or value.get("test_contract_count") != 0
        ):
            raise RuntimeError("Candidate resume record conflicts with the requested job.")
        contract = _contract_from_dict(value["training_contract"])
        validate_checkpoint(args.output_root / str(value["checkpoint_relative_path"]), contract)
        return value

    _, validation_rows, manifests, resource, folds = _load_scope_data(args)
    contract = _training_contract(
        config=config,
        method_id=args.method,
        scope_id=args.scope_id,
        manifests=manifests,
        learning_rate=args.learning_rate,
    )
    checkpoint_dir = _checkpoint_directory(
        args.output_root, args.method, args.scope_id, args.learning_rate, contract
    )
    state, _ = checkpoint_resume_state(checkpoint_dir, contract)
    runtime = None
    try:
        if state == "complete":
            runtime = _load_checkpoint_runtime(
                args.method,
                checkpoint_dir,
                local_files_only=args.local_files_only,
            )
        else:
            runtime, evidence = _train_runtime(
                args.method,
                manifests,
                args.learning_rate,
                local_files_only=args.local_files_only,
            )
            parameters = _training_parameters(args.method, args.learning_rate)
            write_checkpoint_atomic(
                checkpoint_dir,
                contract,
                lambda target: _write_runtime_checkpoint(
                    runtime, args.method, target, parameters
                ),
                evidence={
                    **evidence,
                    "training_manifests": {
                        task: training_manifest_summary(frame)
                        for task, frame in manifests.items()
                    },
                    "test_contract_count": 0,
                },
            )
            publish_artifact_unit(
                args.output_root,
                _checkpoint_files(args.output_root, checkpoint_dir),
                unit_id=(
                    f"checkpoint-{args.method}-{args.scope_id}-"
                    f"lr-{learning_rate_token(args.learning_rate)}-"
                    f"{contract.contract_sha256[:12]}"
                ),
                protocol_id=PROTOCOL_ID,
                contract_sha256=contract.contract_sha256,
            )

        fold = representative_fold(folds)
        aspect_grid, sentiment_grid = _calibration_grid(
            validation_rows, fold.seen_aspects, resource
        )
        aspect_scores, sentiment_scores = _score_runtime(
            args.method, runtime, aspect_grid
        )
        scored = join_two_stage_scores(
            aspect_grid, sentiment_grid, aspect_scores, sentiment_scores
        )
        if (
            scored["aspect_score"].nunique() < 2
            or scored["sentiment_score"].nunique() < 2
        ):
            raise RuntimeError(
                "Trainable two-stage prediction collapse on the complete "
                "seen-validation calibration grid."
            )
        aspect_selection = select_two_stage_threshold(scored)
        second_selection = select_second_sentiment_threshold(
            scored, aspect_threshold=aspect_selection.threshold
        )
        mask = capped_two_sentiment_prediction_mask(
            scored,
            aspect_threshold=aspect_selection.threshold,
            second_sentiment_threshold=second_selection.second_sentiment_threshold,
        )
        metrics = evaluate_prediction_mask(
            scored, mask, aspects=fold.seen_aspects
        )
        result = _seal_payload({
            "schema_version": "taxonomy_two_stage_formal_candidate_v1",
            "protocol_id": PROTOCOL_ID,
            "method_id": args.method,
            "training_scope_id": args.scope_id,
            "learning_rate": args.learning_rate,
            "selection_partition": "seen_validation_only",
            "training_contract": contract.to_dict(),
            "checkpoint_relative_path": checkpoint_dir.relative_to(
                args.output_root
            ).as_posix(),
            "checkpoint_manifest_sha256": file_sha256(
                checkpoint_dir / "checkpoint.manifest.json"
            ),
            "thresholds": {
                "aspect": aspect_selection.threshold,
                "runner_up_sentiment": second_selection.second_sentiment_threshold,
            },
            "selection_metrics": metrics,
            "conditional_sentiment_metrics": conditional_sentiment_metrics(scored),
            "calibration_pair_identity_sha256": pair_identity_hash(sentiment_grid),
            "calibration_score_sha256": dataframe_sha256(
                scored,
                ["row_uid", "candidate_aspect", "candidate_sentiment", "aspect_score", "sentiment_score"],
                sort_columns=["row_uid", "candidate_aspect", "candidate_sentiment"],
            ),
            "test_contract_count": 0,
            "failure_count": 0,
        }, "candidate_payload_sha256")
        _write_json_atomic(result_path, result)
        publish_artifact_unit(
            args.output_root,
            [result_path.relative_to(args.output_root)],
            unit_id=(
                f"candidate-{args.method}-{args.scope_id}-"
                f"lr-{learning_rate_token(args.learning_rate)}-"
                f"{contract.contract_sha256[:12]}"
            ),
            protocol_id=PROTOCOL_ID,
            contract_sha256=contract.contract_sha256,
        )
        return result
    finally:
        if runtime is not None:
            _release_runtime(runtime)


def select_scope(args: argparse.Namespace) -> dict[str, object]:
    candidates = []
    for rate in learning_rates(args.method):
        path = candidate_result_path(args.output_root, args.method, args.scope_id, rate)
        if not path.is_file():
            raise FileNotFoundError(f"Missing learning-rate candidate: {path}")
        candidate = _read_object(path)
        _validate_sealed_payload(candidate, "candidate_payload_sha256")
        if (
            candidate.get("protocol_id") != PROTOCOL_ID
            or candidate.get("method_id") != args.method
            or candidate.get("training_scope_id") != args.scope_id
            or float(candidate.get("learning_rate", -1)) != rate
            or candidate.get("test_contract_count") != 0
            or candidate.get("failure_count") != 0
        ):
            raise ValueError(f"Candidate identity or safety contract mismatch: {path}")
        metrics = candidate.get("selection_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Candidate lacks selection metrics: {path}")
        candidates.append(candidate)
    selected = max(
        candidates,
        key=lambda value: (
            float(value["selection_metrics"]["pair_micro_f1"]),
            float(value["selection_metrics"]["pair_samples_f1"]),
            float(value["selection_metrics"]["pair_micro_precision"]),
            -float(value["learning_rate"]),
        ),
    )
    output = selection_path(args.output_root, args.method, args.scope_id)
    if output.exists():
        observed = _read_object(output)
        _validate_sealed_payload(observed, "selection_payload_sha256")
        if observed.get("selected_candidate_contract_sha256") != selected[
            "training_contract"
        ]["contract_sha256"]:
            raise FileExistsError("Existing selection conflicts with the registered candidates.")
        return observed
    value = _seal_payload({
        "schema_version": "taxonomy_two_stage_formal_selection_v1",
        "protocol_id": PROTOCOL_ID,
        "method_id": args.method,
        "training_scope_id": args.scope_id,
        "selected_learning_rate": selected["learning_rate"],
        "selected_candidate_contract_sha256": selected["training_contract"][
            "contract_sha256"
        ],
        "selected_checkpoint_relative_path": selected["checkpoint_relative_path"],
        "selected_thresholds": selected["thresholds"],
        "selected_metrics": selected["selection_metrics"],
        "tie_breakers": [
            "pair_micro_f1",
            "pair_samples_f1",
            "pair_micro_precision",
            "lower_learning_rate",
        ],
        "candidate_contract_sha256s": [
            value["training_contract"]["contract_sha256"] for value in candidates
        ],
        "test_contract_count": 0,
        "failure_count": 0,
    }, "selection_payload_sha256")
    _write_json_atomic(output, value)
    publish_artifact_unit(
        args.output_root,
        [output.relative_to(args.output_root)],
        unit_id=f"selection-{args.method}-{args.scope_id}",
        protocol_id=PROTOCOL_ID,
        contract_sha256=str(value["selected_candidate_contract_sha256"]),
    )
    return value


def _augment_grids(
    aspect_grid: pd.DataFrame,
    sentiment_grid: pd.DataFrame,
    *,
    fold: Any,
    condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_uids = sorted(sentiment_grid["row_uid"].astype(str).unique())
    row_indices = {uid: index for index, uid in enumerate(row_uids)}
    for frame in (aspect_grid, sentiment_grid):
        frame["fold_id"] = fold.fold_id
        frame["condition"] = condition
        frame["row_index"] = frame["row_uid"].astype(str).map(row_indices).astype(int)
        frame["is_seen"] = frame["candidate_aspect"].astype(str).isin(fold.seen_aspects)
        frame["is_heldout"] = frame["candidate_aspect"].astype(str).isin(
            fold.heldout_aspects
        )
    return aspect_grid, sentiment_grid


def _run_contract(
    *,
    config: Mapping[str, object],
    method_id: str,
    fold: Any,
    condition: str,
    training_contract: TrainingContract,
    selection: Mapping[str, object],
    sentiment_grid: pd.DataFrame,
    resource: Mapping[str, object],
) -> RunContract:
    candidate_rows = sentiment_grid.drop_duplicates(
        ["candidate_aspect", "candidate_sentiment"]
    )
    return RunContract(
        protocol_id=PROTOCOL_ID,
        scientific_protocol_sha256=_protocol_hash(config),
        method_id=method_id,
        method_spec_sha256=resolve_method_spec(method_id).spec_sha256,
        method_registry_sha256=method_registry_sha256(),
        description_resource_sha256=canonical_sha256(resource),
        candidate_representation_sha256=dataframe_sha256(
            candidate_rows,
            [
                "candidate_aspect",
                "candidate_sentiment",
                "candidate_text",
                "representation_variant",
            ],
            sort_columns=["candidate_aspect", "candidate_sentiment"],
        ),
        level=fold.level,
        fold_id=fold.fold_id,
        condition=condition,
        split="validation",
        seed=13,
        training_manifest_sha256=training_contract.training_manifest_sha256,
        evaluation_data_sha256=dataframe_sha256(
            sentiment_grid,
            ["row_uid", "candidate_aspect", "candidate_sentiment", "target"],
            sort_columns=["row_uid", "candidate_aspect", "candidate_sentiment"],
        ),
        evaluation_pair_identity_sha256=pair_identity_hash(sentiment_grid),
        scientific_parameters_sha256=canonical_sha256(
            {
                "training_contract_sha256": training_contract.contract_sha256,
                "selected_learning_rate": selection["selected_learning_rate"],
                "thresholds": selection["selected_thresholds"],
                "decoder": "top_one_plus_thresholded_runner_up",
                "maximum_sentiments_per_aspect": 2,
            }
        ),
        shard_count=8,
        formal=True,
    )


def score_selected(args: argparse.Namespace, config: Mapping[str, object]) -> dict[str, object]:
    selection = select_scope(args)
    selected_rate = float(selection["selected_learning_rate"])
    candidate = _read_object(
        candidate_result_path(
            args.output_root, args.method, args.scope_id, selected_rate
        )
    )
    _validate_sealed_payload(candidate, "candidate_payload_sha256")
    training_contract = _contract_from_dict(candidate["training_contract"])
    checkpoint_dir = args.output_root / str(selection["selected_checkpoint_relative_path"])
    validate_checkpoint(checkpoint_dir, training_contract)
    _, validation_rows, _, resource, folds = _load_scope_data(args)
    runtime = _load_checkpoint_runtime(
        args.method,
        checkpoint_dir,
        local_files_only=args.local_files_only,
    )
    results: list[dict[str, object]] = []
    try:
        for fold in folds:
            for condition in formal_conditions(fold):
                variants = variant_map(fold, condition)
                aspect_grid = build_aspect_grid(
                    validation_rows, fold.evaluation_aspects, variants, resource
                )
                sentiment_grid = build_sentiment_grid(
                    validation_rows, fold.evaluation_aspects, variants, resource
                )
                aspect_grid, sentiment_grid = _augment_grids(
                    aspect_grid, sentiment_grid, fold=fold, condition=condition
                )
                contract = _run_contract(
                    config=config,
                    method_id=args.method,
                    fold=fold,
                    condition=condition,
                    training_contract=training_contract,
                    selection=selection,
                    sentiment_grid=sentiment_grid,
                    resource=resource,
                )
                artifacts: dict[int, pd.DataFrame] = {}
                for shard_index in range(contract.shard_count):
                    pair_shard = select_pair_shard(
                        sentiment_grid, shard_index, contract.shard_count
                    )
                    if pair_shard.empty:
                        continue
                    state, artifact = two_stage_shard_resume_state(
                        args.output_root,
                        contract,
                        sentiment_grid,
                        shard_index=shard_index,
                    )
                    if state == "complete":
                        assert artifact is not None
                        artifacts[shard_index] = artifact
                        continue
                    shard_uids = set(pair_shard["row_uid"].astype(str))
                    aspect_shard = aspect_grid[
                        aspect_grid["row_uid"].astype(str).isin(shard_uids)
                    ].copy()
                    aspect_scores, sentiment_scores = _score_runtime(
                        args.method, runtime, aspect_shard
                    )
                    scored = join_two_stage_scores(
                        aspect_shard,
                        pair_shard,
                        aspect_scores,
                        sentiment_scores,
                    )
                    artifact = build_two_stage_score_artifact(
                        scored, contract, shard_index=shard_index
                    )
                    csv_path, manifest_path = write_two_stage_score_shard(
                        artifact,
                        args.output_root,
                        contract,
                        shard_index=shard_index,
                    )
                    publish_artifact_unit(
                        args.output_root,
                        [
                            csv_path.relative_to(args.output_root),
                            manifest_path.relative_to(args.output_root),
                        ],
                        unit_id=(
                            f"score-{args.method}-{fold.fold_id}-{condition}-"
                            f"s{shard_index:02d}-{contract.contract_sha256[:12]}"
                        ),
                        protocol_id=PROTOCOL_ID,
                        contract_sha256=contract.contract_sha256,
                    )
                    artifacts[shard_index] = artifact
                merged = merge_two_stage_score_shards(
                    artifacts, contract, sentiment_grid
                )
                if (
                    merged["aspect_score"].nunique() < 2
                    or merged["sentiment_score"].nunique() < 2
                ):
                    raise RuntimeError(
                        "Trainable two-stage score collapse in a complete formal "
                        "condition."
                    )
                thresholds = selection["selected_thresholds"]
                mask = capped_two_sentiment_prediction_mask(
                    merged,
                    aspect_threshold=float(thresholds["aspect"]),
                    second_sentiment_threshold=float(
                        thresholds["runner_up_sentiment"]
                    ),
                )
                per_group = (
                    pd.DataFrame(
                        {
                            "row_uid": merged["row_uid"].astype(str),
                            "candidate_aspect": merged["candidate_aspect"].astype(str),
                            "selected": mask.astype(int),
                        }
                    )
                    .groupby(["row_uid", "candidate_aspect"], sort=False)["selected"]
                    .sum()
                )
                if int(per_group.max()) > 2:
                    raise RuntimeError("Formal decoder emitted a third sentiment.")
                if per_group.gt(0).astype(int).nunique() < 2:
                    raise RuntimeError(
                        "Trainable two-stage prediction collapse in a complete "
                        "formal condition."
                    )
                partitions = {
                    "heldout": evaluate_prediction_mask(
                        merged, mask, aspects=fold.heldout_aspects
                    ),
                    "full": evaluate_prediction_mask(
                        merged, mask, aspects=fold.evaluation_aspects
                    ),
                }
                post_supervisor_views = (
                    {
                        "post_supervisor_views": evaluate_l2_condition(
                            merged,
                            fold,
                            aspect_threshold=float(thresholds["aspect"]),
                            second_sentiment_threshold=float(
                                thresholds["runner_up_sentiment"]
                            ),
                        )
                    }
                    if PROTOCOL_ID == "taxonomy_two_stage_formal_v2"
                    and fold.level == "L2"
                    else {}
                )
                result_path = (
                    args.output_root
                    / "results"
                    / args.method
                    / fold.level
                    / fold.fold_id
                    / f"{condition}.json"
                )
                result = _seal_payload({
                    "schema_version": "taxonomy_two_stage_formal_fold_condition_v1",
                    "protocol_id": PROTOCOL_ID,
                    "method_id": args.method,
                    "training_scope_id": args.scope_id,
                    "fold_id": fold.fold_id,
                    "level": fold.level,
                    "condition": condition,
                    "heldout_aspects": list(fold.heldout_aspects),
                    "selection_partition": "seen_validation_only",
                    "evaluation_partition": "validation_only",
                    "selected_learning_rate": selected_rate,
                    "thresholds": thresholds,
                    "training_contract_sha256": training_contract.contract_sha256,
                    "run_contract_sha256": contract.contract_sha256,
                    "score_shards": len(artifacts),
                    "score_rows": len(merged),
                    "score_sha256": dataframe_sha256(
                        merged,
                        [
                            "row_uid",
                            "candidate_aspect",
                            "candidate_sentiment",
                            "aspect_score",
                            "sentiment_score",
                        ],
                        sort_columns=[
                            "row_uid",
                            "candidate_aspect",
                            "candidate_sentiment",
                        ],
                    ),
                    "partitions": partitions,
                    **post_supervisor_views,
                    "conditional_sentiment_metrics": conditional_sentiment_metrics(
                        merged
                    ),
                    "maximum_sentiments_per_aspect": int(per_group.max()),
                    "test_contract_count": 0,
                    "failure_count": 0,
                }, "result_payload_sha256")
                if result_path.exists() and args.resume:
                    observed = _read_object(result_path)
                    _validate_sealed_payload(observed, "result_payload_sha256")
                    if observed.get("run_contract_sha256") != contract.contract_sha256:
                        raise RuntimeError("Existing result belongs to another run contract.")
                    result = observed
                elif result_path.exists():
                    raise FileExistsError(result_path)
                else:
                    _write_json_atomic(result_path, result)
                    publish_artifact_unit(
                        args.output_root,
                        [result_path.relative_to(args.output_root)],
                        unit_id=(
                            f"result-{args.method}-{fold.fold_id}-{condition}-"
                            f"{contract.contract_sha256[:12]}"
                        ),
                        protocol_id=PROTOCOL_ID,
                        contract_sha256=contract.contract_sha256,
                    )
                results.append(result)
        return {
            "status": "pass",
            "method_id": args.method,
            "training_scope_id": args.scope_id,
            "completed_fold_conditions": len(results),
            "test_contract_count": 0,
            "failure_count": 0,
        }
    finally:
        _release_runtime(runtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run immutable validation-only trainable two-stage jobs."
    )
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--method", choices=TRAINABLE_METHODS)
    parser.add_argument("--scope-id")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--config", type=Path, default=DEFAULT_SAFETY_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/experimental/taxonomy_two_stage_formal_v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _read_object(args.config)
    _validate_safety_config(config)
    args.output_root = args.output_root.resolve()
    if args.phase == "plan":
        value = plan_payload(
            build_formal_jobs(
                output_root=str(args.output_root), data_dir=str(args.data_dir.resolve())
            )
        )
        _write_json_atomic(args.output_root / "formal_job_plan.json", value)
        printed: Mapping[str, object] = {
            "status": "pass",
            "protocol_id": value["protocol_id"],
            "unique_training_scopes": value["unique_training_scopes"],
            "fold_condition_count_per_method": value[
                "fold_condition_count_per_method"
            ],
            "job_counts": value["job_counts"],
            "test_contract_count": value["test_contract_count"],
        }
    else:
        if args.method is None or args.scope_id is None:
            raise ValueError("Execution phases require --method and --scope-id.")
        if args.scope_id not in scope_folds():
            raise ValueError(f"Unknown training scope: {args.scope_id!r}")
        if args.phase == "train-candidate":
            value = train_candidate(args, config)
        elif args.phase == "select-scope":
            value = select_scope(args)
        else:
            value = score_selected(args, config)
        printed = value
    print(json.dumps(printed, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
