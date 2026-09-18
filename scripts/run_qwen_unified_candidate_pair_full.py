from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import all_aspects, build_heldout_aspect_split, load_all_fabsa
from msc_project.experiments.unified_candidate_pairs import (
    CANDIDATE_SENTIMENTS,
    build_eval_grid,
    build_full_manifest,
    budget_sample,
    load_experiment_config,
    manifest_hash,
)
from msc_project.experiments.unified_pair_evaluation import (
    evaluate_score_matrix,
    paired_aspect_statistics,
    score_matrix_from_grid,
    select_threshold,
)
from msc_project.llm.qwen_pair_classifier import QwenPairQLoRAConfig, load_qwen_pair_qlora, score_candidate_pairs
from run_qwen_unified_candidate_pair_loao import (
    PILOT_FOLDS,
    git_commit,
    historical_frozen_gate,
    json_default,
    load_frozen_qwen,
    load_saved_qwen_pair_adapter,
    ordered_eval_frame,
    package_versions,
    prediction_rows,
    set_seed,
    seen_aspects,
    slugify,
    train_qwen_adapter,
    write_json,
    write_jsonl,
)


PROTOCOL_ID = "loao_unified_candidate_pair_experimental_v1"
INTEGRITY_SCHEMA_VERSION = 2
FULL_FOLD_COUNT = 12
ENHANCED_VARIANT = "enhanced"
QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
FROZEN_RESULT_MODEL = "qwen_frozen_candidate_pair"
QLORA_RESULT_MODEL = "qwen_qlora_candidate_pair"
QLORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} must contain a JSON object: {path}")
    return payload


def _require_exact(value: object, expected: object, description: str) -> None:
    if value != expected:
        raise ValueError(f"{description} must be {expected!r}; got {value!r}.")


def _result_model(mode: str) -> str:
    if mode == "frozen":
        return FROZEN_RESULT_MODEL
    if mode == "qlora":
        return QLORA_RESULT_MODEL
    raise ValueError(f"Unsupported full-run mode: {mode!r}")


def preregistered_scientific_parameters(preregistration: dict[str, object]) -> dict[str, object]:
    """Return every model/training value frozen for formal Qwen runs.

    Evaluation batch size is intentionally absent: it is an operational memory
    control and was changed during the preregistered pilot's OOM recovery.
    """

    qwen = preregistration.get("models", {}).get("qwen", {})  # type: ignore[union-attr]
    manifest = preregistration.get("training_manifest", {})
    if not isinstance(qwen, dict) or not isinstance(manifest, dict):
        raise ValueError("Preregistration is missing Qwen or training-manifest parameters.")
    return {
        "model_name": qwen.get("model_name"),
        "seed": preregistration.get("seed"),
        "max_length": qwen.get("max_length"),
        "batch_size": qwen.get("batch_size"),
        "gradient_accumulation_steps": qwen.get("gradient_accumulation_steps"),
        "epochs": qwen.get("epochs"),
        "learning_rate": qwen.get("learning_rate"),
        "weight_decay": qwen.get("weight_decay"),
        # These optimiser controls were fixed in the preregistered runner even
        # though the first config revision did not duplicate them in models.qwen.
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "log_every_steps": 64,
        "train_budget": manifest.get("model_budget_pairs_per_fold"),
        "positive_budget": manifest.get("budget_positive_pairs"),
    }


def preregistered_qwen_runtime_contract(
    preregistration: dict[str, object],
) -> dict[str, object]:
    qwen = preregistration.get("models", {}).get("qwen", {})  # type: ignore[union-attr]
    if not isinstance(qwen, dict):
        raise ValueError("Preregistration is missing models.qwen.")
    expected = {
        "task_format": "single candidate aspect-sentiment claim; one-token Y/N answer",
        "score": "softmax probability of Y over Y and N next-token logits",
        "load_in_4bit": True,
        "quantisation": "NF4 with double quantisation and float16 compute",
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "lora_target_modules": list(QLORA_TARGET_MODULES),
    }
    for key, value in expected.items():
        _require_exact(qwen.get(key), value, f"Preregistered Qwen contract {key}")
    return {**expected, "gradient_checkpointing": True}


def validate_scientific_parameters(
    values: argparse.Namespace | dict[str, object],
    preregistration: dict[str, object],
) -> None:
    preregistered_qwen_runtime_contract(preregistration)
    actual = vars(values) if isinstance(values, argparse.Namespace) else values
    for key, expected in preregistered_scientific_parameters(preregistration).items():
        _require_exact(actual.get(key), expected, f"Formal Qwen parameter {key}")
    eval_batch_size = actual.get("eval_batch_size")
    if not isinstance(eval_batch_size, int) or eval_batch_size < 1:
        raise ValueError("eval_batch_size must be a positive operational batch size.")


def validate_full_fold_set(requested: list[str], dataset_folds: list[str]) -> list[str]:
    """Require a formal run to contain the complete canonical 12-fold set."""

    if len(dataset_folds) != FULL_FOLD_COUNT or len(set(dataset_folds)) != FULL_FOLD_COUNT:
        raise ValueError("The dataset must expose exactly 12 unique canonical LOAO folds.")
    folds = list(requested) if requested else list(dataset_folds)
    if len(folds) != FULL_FOLD_COUNT or len(set(folds)) != FULL_FOLD_COUNT:
        raise ValueError("A formal full confirmation must request exactly 12 unique folds.")
    if set(folds) != set(dataset_folds):
        missing = sorted(set(dataset_folds) - set(folds))
        extra = sorted(set(folds) - set(dataset_folds))
        raise ValueError(f"Formal full fold set differs from the dataset; missing={missing}, extra={extra}.")
    return folds


def _validate_same_full_fold_set(value: object, expected: list[str], description: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{description} must be a list of fold names.")
    if len(value) != FULL_FOLD_COUNT or len(set(value)) != FULL_FOLD_COUNT:
        raise ValueError(f"{description} must contain exactly 12 unique folds.")
    if set(value) != set(expected):
        raise ValueError(f"{description} is not the same preregistered 12-fold set.")
    return value


def dataset_identity(frame: pd.DataFrame, folds: list[str]) -> dict[str, object]:
    """Fingerprint the immutable inputs that determine the LOAO split contents."""

    required = ("row_uid", "split", "text", "labels_json")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot fingerprint dataset; missing columns: {sorted(missing)}")
    records = []
    ordered = frame.sort_values("row_uid", kind="stable")
    for row in ordered.loc[:, required].itertuples(index=False, name=None):
        records.append([None if pd.isna(value) else str(value) for value in row])
    encoded = json.dumps(
        {"folds": sorted(folds), "records": records},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "rows": int(len(frame)),
        "folds": sorted(folds),
    }


def validate_existing_run_manifest(
    path: Path,
    *,
    mode: str,
    folds: list[str],
    preregistration: dict[str, object],
    expected_dataset_identity: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Validate resume identity and identify a narrowly supported frozen legacy run."""

    payload = _read_json_object(path, "run manifest")
    _require_exact(payload.get("protocol_id"), PROTOCOL_ID, "Run-manifest protocol_id")
    _require_exact(payload.get("mode"), mode, "Run-manifest mode")
    _require_exact(payload.get("stage"), "full_12_fold_confirmation", "Run-manifest stage")
    _require_exact(payload.get("folds"), folds, "Run-manifest folds")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Run manifest is missing its arguments object.")
    _require_exact(arguments.get("mode"), mode, "Run-manifest argument mode")
    validate_scientific_parameters(arguments, preregistration)

    schema_version = payload.get("integrity_schema_version")
    identity = payload.get("dataset_identity")
    legacy_frozen = schema_version is None and identity is None and mode == "frozen"
    if legacy_frozen:
        # Compatibility is deliberately limited to the already-running frozen
        # job. Its completed folds are still deep-validated before reuse and
        # rewritten with v2 identity metadata.
        return payload, True
    _require_exact(schema_version, INTEGRITY_SCHEMA_VERSION, "Run-manifest integrity schema")
    _require_exact(identity, expected_dataset_identity, "Run-manifest dataset identity")
    _require_exact(payload.get("variant"), ENHANCED_VARIANT, "Run-manifest variant")
    _require_exact(
        payload.get("qwen_runtime_contract"),
        preregistered_qwen_runtime_contract(preregistration),
        "Run-manifest Qwen runtime contract",
    )
    return payload, False


def _normalised_run_manifest(
    *,
    existing: dict[str, object] | None,
    args: argparse.Namespace,
    folds: list[str],
    identity: dict[str, object],
    preregistration: dict[str, object],
) -> dict[str, object]:
    now = datetime.now().isoformat(timespec="seconds")
    current_commit = git_commit()
    payload = dict(existing or {})
    resumed_at = payload.get("resumed_at", [])
    if not isinstance(resumed_at, list):
        resumed_at = []
    if existing is not None:
        resumed_at = [*resumed_at, now]
    payload.update(
        {
            "integrity_schema_version": INTEGRITY_SCHEMA_VERSION,
            "protocol_id": preregistration["protocol_id"],
            "mode": args.mode,
            "stage": "full_12_fold_confirmation",
            "variant": ENHANCED_VARIANT,
            "folds": folds,
            "dataset_identity": identity,
            "qwen_runtime_contract": preregistered_qwen_runtime_contract(preregistration),
            "command": " ".join(sys.argv),
            "started_at": payload.get("started_at", now),
            "resumed_at": resumed_at,
            "git_commit": payload.get("git_commit", current_commit),
            "current_execution_code_git_commit": current_commit,
            "packages": payload.get("packages", package_versions()),
            "cuda_device": payload.get(
                "cuda_device",
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            ),
            "arguments": vars(args),
        }
    )
    if existing is not None and payload.get("integrity_schema_version") is None:
        payload["integrity_upgrade_git_commit"] = current_commit
    payload.pop("finished_at", None)
    return payload


def score_split(
    model: Any,
    tokenizer: Any,
    verbalizer_ids: Any,
    frame: pd.DataFrame,
    heldout_aspect: str,
    split_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    threshold: float | None = None,
) -> tuple[dict[str, object], float]:
    grid = build_eval_grid(frame, heldout_aspect, "enhanced")
    started = time.time()
    flat_scores = score_candidate_pairs(
        model,
        tokenizer,
        [str(value) for value in grid["text"]],
        [str(value) for value in grid["candidate_text"]],
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        verbalizer_ids=verbalizer_ids,
        padding_side="right",
    )
    seconds = time.time() - started
    scores = score_matrix_from_grid(grid, flat_scores, row_count=len(frame))
    true_pairs = [list(labels) for labels in frame["supervision_pair_labels"]]
    if split_name == "validation":
        if threshold is not None:
            raise ValueError("Validation must select, not receive, a threshold.")
        selection = select_threshold(true_pairs, scores, heldout_aspect)
        threshold = selection.threshold
        selection.sweep.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)
    elif threshold is None:
        raise ValueError("Test scoring requires the frozen validation threshold.")

    metrics, predictions = evaluate_score_matrix(
        true_pairs,
        scores,
        heldout_aspect,
        float(threshold),
    )
    score_rows = grid.copy()
    score_rows["score"] = flat_scores
    score_rows.to_csv(output_dir / f"{split_name}_pair_scores.csv", index=False)
    write_jsonl(
        prediction_rows(frame, scores, predictions, float(threshold)),
        output_dir / f"{split_name}_predictions.jsonl",
    )
    result = {
        "heldout_aspect": heldout_aspect,
        "split": split_name,
        "variant": "enhanced",
        "selected_threshold": float(threshold),
        "score_seconds": float(seconds),
        "seconds_per_pair": float(seconds / len(grid)),
        **metrics,
    }
    return result, float(threshold)


def _validate_score_row(
    row: object,
    *,
    heldout_aspect: str,
    split: str,
    model: str,
    examples: int,
    expected_manifest_hash: str | None = None,
) -> dict[str, object]:
    if not isinstance(row, dict):
        raise ValueError("A result row must be a JSON object.")
    expected = {
        "heldout_aspect": heldout_aspect,
        "split": split,
        "model": model,
        "variant": ENHANCED_VARIANT,
        "examples": examples,
    }
    for key, value in expected.items():
        _require_exact(row.get(key), value, f"Result {key}")
    for key in ("selected_threshold", "pair_micro_f1", "pair_micro_precision", "pair_micro_recall"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Result {key} must be finite.")
    threshold = float(row["selected_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("selected_threshold must lie in [0, 1].")
    if "threshold" in row and not math.isclose(
        float(row["threshold"]), threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Result threshold differs from selected_threshold.")
    if expected_manifest_hash is not None:
        _require_exact(
            row.get("training_manifest_hash"),
            expected_manifest_hash,
            "Result training_manifest_hash",
        )
        _require_exact(row.get("training_pairs"), 4096, "Result training_pairs")
        _require_exact(row.get("training_positive_pairs"), 2048, "Result training_positive_pairs")
    return row


def _same_result(expected: dict[str, object], actual: dict[str, object], description: str) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"{description} differs at {key!r}.")


def _validate_pilot_gate(
    payload: dict[str, object],
    *,
    mode: str,
    results: list[dict[str, object]],
    frozen_results: list[dict[str, object]] | None,
) -> None:
    """Recompute the applicable gate instead of trusting a stored boolean."""

    if mode == "frozen":
        gate = payload.get("frozen_gate")
        if payload.get("qlora_gate") is not None:
            raise ValueError("Frozen pilot summary unexpectedly contains a QLoRA gate.")
        recomputed = historical_frozen_gate(results)
    elif mode == "qlora":
        gate = payload.get("qlora_gate")
        if payload.get("frozen_gate") is not None:
            raise ValueError("QLoRA pilot summary unexpectedly contains a frozen gate.")
        if frozen_results is None:
            raise ValueError("QLoRA pilot validation requires frozen pilot results.")
        from msc_project.experiments.unified_pair_evaluation import pilot_gate

        recomputed = pilot_gate(pd.DataFrame(results), pd.DataFrame(frozen_results))
    else:
        raise ValueError(f"Unsupported pilot mode: {mode!r}")
    if not isinstance(gate, dict):
        raise ValueError(f"{mode} pilot summary is missing its preregistered gate.")
    if gate.get("passed") is not True or recomputed.get("passed") is not True:
        raise ValueError(f"The {mode} pilot did not pass its recomputed preregistered gate.")
    for key, value in recomputed.items():
        if isinstance(value, (bool, str, int, float)) and gate.get(key) != value:
            raise ValueError(f"Stored {mode} pilot gate differs from recomputation at {key!r}.")


def _load_jsonl_frame(path: Path, description: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{description} line {line_number} is not an object.")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}") from exc
    return pd.DataFrame(rows)


def validate_training_manifest_file(
    path: Path,
    *,
    expected_manifest: pd.DataFrame,
    expected_hash: str,
) -> None:
    stored = _load_jsonl_frame(path, "training manifest")
    _require_exact(len(stored), len(expected_manifest), "Stored training-manifest row count")
    _require_exact(int(stored["target"].sum()), int(expected_manifest["target"].sum()), "Stored positive count")
    _require_exact(manifest_hash(stored), expected_hash, "Stored training-manifest hash")
    _require_exact(manifest_hash(expected_manifest), expected_hash, "Expected training-manifest hash")


def _expected_adapter_parameters() -> dict[str, object]:
    return {
        "base_model_name_or_path": QWEN_MODEL_NAME,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "bias": "none",
    }


def validate_adapter_artifacts(
    adapter_dir: Path,
    *,
    expected_heldout_aspect: str,
    expected_manifest_hash: str,
    expected_result: dict[str, object],
    training_summary_path: Path | None,
    allow_legacy_pilot_summary: bool,
) -> dict[str, object]:
    """Validate adapter identity, LoRA shape, base model and training evidence."""

    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size <= 0:
        raise ValueError(f"Missing or empty adapter weights: {weights}")
    adapter_config = _read_json_object(adapter_dir / "adapter_config.json", "adapter config")
    for key, value in _expected_adapter_parameters().items():
        _require_exact(adapter_config.get(key), value, f"Adapter config {key}")
    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, list) or set(target_modules) != set(QLORA_TARGET_MODULES):
        raise ValueError("Adapter target_modules differ from the preregistered QLoRA modules.")
    _require_exact(adapter_config.get("inference_mode"), True, "Adapter inference_mode")

    _require_exact(
        expected_result.get("training_manifest_hash"),
        expected_manifest_hash,
        "Adapter result manifest hash",
    )
    _require_exact(expected_result.get("training_pairs"), 4096, "Adapter training pair count")
    _require_exact(expected_result.get("training_positive_pairs"), 2048, "Adapter positive pair count")
    embedded_history = expected_result.get("history")
    nested_training = expected_result.get("training_summary")
    if embedded_history is None and isinstance(nested_training, dict):
        embedded_history = nested_training.get("history")
    if (
        not isinstance(embedded_history, list)
        or not embedded_history
        or not isinstance(embedded_history[-1], dict)
    ):
        raise ValueError("Adapter result is missing training history.")
    _require_exact(embedded_history[-1].get("global_step"), 512, "Adapter final global step")

    summary: dict[str, object]
    if training_summary_path is not None and training_summary_path.is_file():
        summary = _read_json_object(training_summary_path, "training summary")
        _require_exact(
            summary.get("training_manifest_hash"),
            expected_manifest_hash,
            "Training-summary manifest hash",
        )
        history = summary.get("history")
        if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
            raise ValueError("Training summary is missing history.")
        _require_exact(history[-1].get("global_step"), 512, "Training-summary final global step")
        if summary.get("integrity_schema_version") is not None:
            _require_exact(
                summary.get("integrity_schema_version"),
                INTEGRITY_SCHEMA_VERSION,
                "Training-summary integrity schema",
            )
            _require_exact(summary.get("protocol_id"), PROTOCOL_ID, "Training-summary protocol")
            _require_exact(summary.get("mode"), "qlora", "Training-summary mode")
            _require_exact(
                summary.get("heldout_aspect"),
                expected_heldout_aspect,
                "Training-summary heldout aspect",
            )
            _require_exact(summary.get("base_model_name_or_path"), QWEN_MODEL_NAME, "Training-summary base model")
            _require_exact(summary.get("adapter_parameters"), _expected_adapter_parameters() | {"target_modules": list(QLORA_TARGET_MODULES)}, "Training-summary adapter parameters")
            _require_exact(
                summary.get("scientific_parameters"),
                preregistered_scientific_parameters(load_experiment_config()),
                "Training-summary scientific parameters",
            )
            _require_exact(
                summary.get("qwen_runtime_contract"),
                preregistered_qwen_runtime_contract(load_experiment_config()),
                "Training-summary Qwen runtime contract",
            )
            _require_exact(summary.get("training_pairs"), 4096, "Training-summary pair count")
            _require_exact(summary.get("training_positive_pairs"), 2048, "Training-summary positive count")
        elif not allow_legacy_pilot_summary:
            raise ValueError("Legacy training summary is not accepted for a formal full-run adapter.")
    elif allow_legacy_pilot_summary:
        # Two preregistered pilot adapters survived a scoring OOM before the
        # sidecar was written. Their root result has the captured training
        # history/hash and is cross-checked against the manifest and PEFT config.
        summary = {"legacy_embedded_pilot_training_summary": True, **expected_result}
    else:
        raise ValueError(f"Missing training summary: {training_summary_path}")
    return summary


def validate_pilot_results(
    root: Path,
    *,
    mode: str,
    preregistration: dict[str, object],
    expected_examples: dict[str, int],
    expected_validation_frames: dict[str, pd.DataFrame] | None = None,
    expected_manifests: dict[str, pd.DataFrame] | None = None,
    frozen_results: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    manifest = _read_json_object(root / "run_manifest.json", f"{mode} pilot run manifest")
    _require_exact(manifest.get("protocol_id"), PROTOCOL_ID, "Pilot protocol_id")
    _require_exact(manifest.get("mode"), mode, "Pilot mode")
    _require_exact(manifest.get("stage"), "pilot", "Pilot stage")
    _require_exact(manifest.get("variant"), ENHANCED_VARIANT, "Pilot variant")
    _require_exact(manifest.get("folds"), list(PILOT_FOLDS), "Pilot folds")
    if not manifest.get("finished_at"):
        raise ValueError(f"{mode} pilot run manifest is not marked finished.")
    arguments = manifest.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"{mode} pilot run manifest is missing arguments.")
    _require_exact(arguments.get("mode"), mode, "Pilot argument mode")
    _require_exact(arguments.get("stage"), "pilot", "Pilot argument stage")
    _require_exact(arguments.get("variant"), ENHANCED_VARIANT, "Pilot argument variant")
    _require_exact(arguments.get("eval_limit"), None, "Pilot eval_limit")
    _require_exact(arguments.get("train_row_limit"), None, "Pilot train_row_limit")
    validate_scientific_parameters(arguments, preregistration)

    payload = _read_json_object(root / "summary.json", f"{mode} pilot summary")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(PILOT_FOLDS):
        raise ValueError(f"{mode} pilot summary must contain exactly three result rows.")
    expected_model = _result_model(mode)
    by_fold: dict[str, dict[str, object]] = {}
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise ValueError("Pilot result rows must be objects.")
        aspect = raw.get("heldout_aspect")
        if aspect not in PILOT_FOLDS or not isinstance(aspect, str) or aspect in by_fold:
            raise ValueError(f"Invalid or duplicate {mode} pilot fold: {aspect!r}")
        digest = None
        if mode == "qlora":
            if expected_manifests is None:
                raise ValueError("QLoRA pilot validation requires expected manifests.")
            digest = manifest_hash(expected_manifests[aspect])
        result = _validate_score_row(
            raw,
            heldout_aspect=aspect,
            split="validation",
            model=expected_model,
            examples=expected_examples[aspect],
            expected_manifest_hash=digest,
        )
        fold_payload = _read_json_object(
            root / slugify(aspect) / "summary.json",
            f"{mode} pilot fold summary",
        )
        fold_result = fold_payload.get("result")
        if not isinstance(fold_result, dict):
            raise ValueError(f"{mode} pilot fold summary lacks its result object.")
        _same_result(result, fold_result, f"{mode} pilot fold/root result")
        by_fold[aspect] = result
        _validate_scored_artifact_rows(
            root / slugify(aspect),
            split="validation",
            examples=expected_examples[aspect],
            require_files=True,
            heldout_aspect=aspect,
            expected_frame=(
                None
                if expected_validation_frames is None
                else expected_validation_frames[aspect]
            ),
        )

        if mode == "qlora":
            assert expected_manifests is not None and digest is not None
            validate_training_manifest_file(
                root / slugify(aspect) / "training_manifest.jsonl",
                expected_manifest=expected_manifests[aspect],
                expected_hash=digest,
            )
            validate_adapter_artifacts(
                root / slugify(aspect) / "adapter",
                expected_heldout_aspect=aspect,
                expected_manifest_hash=digest,
                expected_result=result,
                training_summary_path=root / slugify(aspect) / "training_summary.json",
                allow_legacy_pilot_summary=True,
            )
    _require_exact(set(by_fold), set(PILOT_FOLDS), f"{mode} pilot fold set")

    frozen_list = None if frozen_results is None else [frozen_results[fold] for fold in PILOT_FOLDS]
    if mode == "qlora":
        references = payload.get("frozen_reference")
        if not isinstance(references, list) or frozen_list is None or len(references) != len(frozen_list):
            raise ValueError("QLoRA pilot summary has incomplete frozen-reference rows.")
        for expected, actual in zip(frozen_list, references):
            if not isinstance(actual, dict):
                raise ValueError("QLoRA frozen-reference rows must be objects.")
            _same_result(expected, actual, "QLoRA frozen pilot reference")
    _validate_pilot_gate(
        payload,
        mode=mode,
        results=[by_fold[fold] for fold in PILOT_FOLDS],
        frozen_results=frozen_list,
    )
    return by_fold


def load_pilot_result(
    root: Path,
    heldout_aspect: str,
    *,
    expected_mode: str,
    preregistration: dict[str, object],
    expected_examples: dict[str, int],
    expected_validation_frames: dict[str, pd.DataFrame] | None = None,
    expected_manifests: dict[str, pd.DataFrame] | None = None,
    frozen_results: dict[str, dict[str, object]] | None = None,
) -> dict[str, object] | None:
    """Strict compatibility wrapper used by tests and one-off recovery tools."""

    if heldout_aspect not in PILOT_FOLDS:
        return None
    return validate_pilot_results(
        root,
        mode=expected_mode,
        preregistration=preregistration,
        expected_examples=expected_examples,
        expected_validation_frames=expected_validation_frames,
        expected_manifests=expected_manifests,
        frozen_results=frozen_results,
    )[heldout_aspect]


def _validate_scored_artifact_rows(
    fold_dir: Path,
    *,
    split: str,
    examples: int,
    require_files: bool,
    heldout_aspect: str | None = None,
    expected_frame: pd.DataFrame | None = None,
) -> None:
    scores_path = fold_dir / f"{split}_pair_scores.csv"
    predictions_path = fold_dir / f"{split}_predictions.jsonl"
    if not require_files and not scores_path.exists() and not predictions_path.exists():
        return
    if not scores_path.is_file() or not predictions_path.is_file():
        raise ValueError(f"Incomplete {split} scoring artifacts in {fold_dir}.")
    scores = pd.read_csv(scores_path)
    _require_exact(len(scores), examples * len(CANDIDATE_SENTIMENTS), f"{split} pair-score rows")
    if heldout_aspect is not None:
        if "candidate_aspect" not in scores.columns:
            raise ValueError(f"{split} pair scores lack candidate_aspect.")
        _require_exact(
            set(scores["candidate_aspect"].astype(str)),
            {heldout_aspect},
            f"{split} pair-score aspect",
        )
        if "candidate_sentiment" not in scores.columns:
            raise ValueError(f"{split} pair scores lack candidate_sentiment.")
        _require_exact(
            set(scores["candidate_sentiment"].astype(str)),
            set(CANDIDATE_SENTIMENTS),
            f"{split} pair-score sentiments",
        )
    predictions = _load_jsonl_frame(predictions_path, f"{split} predictions")
    _require_exact(len(predictions), examples, f"{split} prediction rows")
    if expected_frame is not None:
        if len(expected_frame) != examples:
            raise ValueError(f"Internal {split} expected-frame size differs from result examples.")
        for column in ("row_uid", "text", "gold_pair_labels"):
            if column not in predictions.columns:
                raise ValueError(f"{split} predictions lack {column}.")
        _require_exact(
            predictions["row_uid"].astype(str).tolist(),
            expected_frame["row_uid"].astype(str).tolist(),
            f"{split} prediction row_uid sequence",
        )
        _require_exact(
            predictions["text"].astype(str).tolist(),
            expected_frame["text"].astype(str).tolist(),
            f"{split} prediction text sequence",
        )
        expected_gold = [list(value) for value in expected_frame["supervision_pair_labels"]]
        actual_gold = [list(value) for value in predictions["gold_pair_labels"]]
        _require_exact(actual_gold, expected_gold, f"{split} prediction gold labels")


def completed_fold_results(
    path: Path,
    *,
    heldout_aspect: str,
    mode: str,
    validation_examples: int,
    test_examples: int,
    expected_manifest_hash: str | None = None,
    expected_pilot_validation: dict[str, object] | None = None,
    validation_frame: pd.DataFrame | None = None,
    test_frame: pd.DataFrame | None = None,
    allow_legacy_frozen: bool = False,
) -> list[dict[str, object]] | None:
    """Return a completed fold only after strict identity and artifact checks."""

    if not path.exists():
        return None
    payload = _read_json_object(path, "completed fold summary")
    metadata_present = payload.get("integrity_schema_version") is not None
    if metadata_present:
        _require_exact(payload.get("integrity_schema_version"), INTEGRITY_SCHEMA_VERSION, "Fold integrity schema")
        _require_exact(payload.get("protocol_id"), PROTOCOL_ID, "Fold protocol_id")
        _require_exact(payload.get("mode"), mode, "Fold mode")
        _require_exact(payload.get("heldout_aspect"), heldout_aspect, "Fold heldout_aspect")
        _require_exact(payload.get("variant"), ENHANCED_VARIANT, "Fold variant")
        _require_exact(payload.get("base_model_name_or_path"), QWEN_MODEL_NAME, "Fold base model")
        _require_exact(
            payload.get("qwen_runtime_contract"),
            preregistered_qwen_runtime_contract(load_experiment_config()),
            "Fold Qwen runtime contract",
        )
    elif not (allow_legacy_frozen and mode == "frozen"):
        raise ValueError("Legacy fold summaries are accepted only for the in-flight frozen recovery run.")

    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("A completed full fold must contain exactly two result rows.")
    by_split: dict[str, dict[str, object]] = {}
    for raw in results:
        if not isinstance(raw, dict):
            raise ValueError("Completed fold result rows must be objects.")
        split = raw.get("split")
        if split not in {"validation", "test"} or split in by_split:
            raise ValueError("A completed fold needs one unique validation row and one unique test row.")
        by_split[str(split)] = _validate_score_row(
            raw,
            heldout_aspect=heldout_aspect,
            split=str(split),
            model=_result_model(mode),
            examples=validation_examples if split == "validation" else test_examples,
            expected_manifest_hash=expected_manifest_hash,
        )
    if not math.isclose(
        float(by_split["validation"]["selected_threshold"]),
        float(by_split["test"]["selected_threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Completed fold test threshold differs from its validation threshold.")

    fold_dir = path.parent
    reused_pilot = by_split["validation"].get("reused_preregistered_pilot_validation") is True
    if expected_pilot_validation is not None:
        if not reused_pilot:
            raise ValueError("Pilot-fold validation is not marked as reused.")
        _same_result(expected_pilot_validation, by_split["validation"], "Reused pilot validation")
    elif reused_pilot:
        raise ValueError("A non-pilot fold claims to reuse pilot validation.")
    _validate_scored_artifact_rows(
        fold_dir,
        split="validation",
        examples=validation_examples,
        require_files=not reused_pilot,
        heldout_aspect=heldout_aspect,
        expected_frame=validation_frame,
    )
    _validate_scored_artifact_rows(
        fold_dir,
        split="test",
        examples=test_examples,
        require_files=True,
        heldout_aspect=heldout_aspect,
        expected_frame=test_frame,
    )
    return [by_split["validation"], by_split["test"]]


def write_fold_summary(
    path: Path,
    *,
    mode: str,
    heldout_aspect: str,
    results: list[dict[str, object]],
) -> None:
    write_json(
        {
            "integrity_schema_version": INTEGRITY_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "mode": mode,
            "heldout_aspect": heldout_aspect,
            "variant": ENHANCED_VARIANT,
            "base_model_name_or_path": QWEN_MODEL_NAME,
            "qwen_runtime_contract": preregistered_qwen_runtime_contract(
                load_experiment_config()
            ),
            "results": results,
        },
        path,
    )


def _is_complete_full_frame(
    frame: pd.DataFrame,
    expected_folds: list[str],
    *,
    expected_model: str,
) -> bool:
    required = {"heldout_aspect", "split", "model", "variant"}
    if not required.issubset(frame.columns) or len(frame) != FULL_FOLD_COUNT * 2:
        return False
    if set(frame["model"].astype(str)) != {expected_model}:
        return False
    if set(frame["variant"].astype(str)) != {ENHANCED_VARIANT}:
        return False
    keys = list(zip(frame["heldout_aspect"].astype(str), frame["split"].astype(str)))
    expected = {(fold, split) for fold in expected_folds for split in ("validation", "test")}
    return len(set(keys)) == len(keys) and set(keys) == expected


def aggregate(
    rows: list[dict[str, object]],
    output_dir: Path,
    mode: str,
    frozen_summary: Path | None,
    expected_folds: list[str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "all_results.csv", index=False)
    metrics = [
        "pair_micro_f1",
        "pair_micro_precision",
        "pair_micro_recall",
        "pair_samples_f1",
        "presence_f1",
        "presence_false_positive_rows_per_100",
        "presence_false_negative_rows_per_100",
        "conditional_sentiment_macro_f1",
    ]
    spread: list[dict[str, object]] = []
    for split_name, group in frame.groupby("split") if not frame.empty else []:
        summary: dict[str, object] = {
            "model": str(frame["model"].iloc[0]),
            "split": str(split_name),
            "folds": int(group["heldout_aspect"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_median"] = float(values.median())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        spread.append(summary)

    confirmation = None
    if mode == "qlora" and frozen_summary is not None:
        frozen_payload = _read_json_object(frozen_summary, "full frozen-Qwen summary")
        _require_exact(
            frozen_payload.get("integrity_schema_version"),
            INTEGRITY_SCHEMA_VERSION,
            "Frozen confirmation-reference integrity schema",
        )
        _require_exact(
            frozen_payload.get("protocol_id"),
            PROTOCOL_ID,
            "Frozen confirmation-reference protocol_id",
        )
        _require_exact(frozen_payload.get("mode"), "frozen", "Frozen confirmation-reference mode")
        _require_exact(
            frozen_payload.get("qwen_runtime_contract"),
            preregistered_qwen_runtime_contract(load_experiment_config()),
            "Frozen confirmation-reference Qwen runtime contract",
        )
        _validate_same_full_fold_set(
            frozen_payload.get("expected_folds"),
            expected_folds,
            "Frozen confirmation-reference folds",
        )
        frozen_raw = frozen_payload.get("results")
        if not isinstance(frozen_raw, list):
            raise ValueError("Full frozen-Qwen summary is missing result rows.")
        frozen_rows = pd.DataFrame(frozen_raw)
        qwen_test = frame[frame["split"] == "test"]
        frozen_test = frozen_rows[frozen_rows["split"] == "test"]
        expected_set = set(expected_folds)
        qwen_complete = _is_complete_full_frame(
            frame,
            expected_folds,
            expected_model=QLORA_RESULT_MODEL,
        )
        frozen_complete = _is_complete_full_frame(
            frozen_rows,
            expected_folds,
            expected_model=FROZEN_RESULT_MODEL,
        )
        if qwen_complete:
            if not frozen_complete:
                raise ValueError(
                    "QLoRA full confirmation requires a complete frozen reference on the same 12 folds."
                )
            aligned = qwen_test[["heldout_aspect", "pair_micro_f1"]].merge(
                frozen_test[["heldout_aspect", "pair_micro_f1"]],
                on="heldout_aspect",
                suffixes=("_qlora", "_frozen"),
                validate="one_to_one",
            )
            if len(aligned) != FULL_FOLD_COUNT or set(aligned["heldout_aspect"]) != expected_set:
                raise ValueError("QLoRA/frozen alignment is not the exact preregistered 12-fold set.")
            statistics = paired_aspect_statistics(
                aligned["pair_micro_f1_qlora"],
                aligned["pair_micro_f1_frozen"],
            )
            confirmation = {
                "passed": bool(statistics["mean_delta"] >= 0.02 and statistics["wins"] >= 8),
                "required_mean_delta": 0.02,
                "required_wins": 8,
                **statistics,
                "per_fold": aligned.assign(
                    delta=aligned["pair_micro_f1_qlora"] - aligned["pair_micro_f1_frozen"]
                ).to_dict(orient="records"),
            }

    summary = {
        "integrity_schema_version": INTEGRITY_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "expected_folds": expected_folds,
        "qwen_runtime_contract": preregistered_qwen_runtime_contract(load_experiment_config()),
        "results": rows,
        "spread": spread,
        "full_confirmation": confirmation,
        "historical_qwen_json_test_mean_pair_micro_f1": 0.33778765,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def validate_full_reference_summary(
    path: Path,
    *,
    expected_folds: list[str],
    expected_examples: dict[str, dict[str, int]],
) -> dict[str, object]:
    """Require a completed, identity-upgraded frozen 12-fold reference."""

    payload = _read_json_object(path, "full frozen-Qwen summary")
    _require_exact(payload.get("integrity_schema_version"), INTEGRITY_SCHEMA_VERSION, "Frozen-summary integrity schema")
    _require_exact(payload.get("protocol_id"), PROTOCOL_ID, "Frozen-summary protocol_id")
    _require_exact(payload.get("mode"), "frozen", "Frozen-summary mode")
    _require_exact(
        payload.get("qwen_runtime_contract"),
        preregistered_qwen_runtime_contract(load_experiment_config()),
        "Frozen-summary Qwen runtime contract",
    )
    _validate_same_full_fold_set(
        payload.get("expected_folds"), expected_folds, "Frozen-summary expected folds"
    )
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != FULL_FOLD_COUNT * 2:
        raise ValueError("Frozen full reference must contain exactly 24 result rows.")
    seen: set[tuple[str, str]] = set()
    for raw in results:
        if not isinstance(raw, dict):
            raise ValueError("Frozen full-reference rows must be objects.")
        aspect = raw.get("heldout_aspect")
        split = raw.get("split")
        if aspect not in expected_folds or split not in {"validation", "test"}:
            raise ValueError("Frozen full-reference row has an unexpected fold or split.")
        key = (str(aspect), str(split))
        if key in seen:
            raise ValueError("Frozen full reference contains duplicate fold/split rows.")
        seen.add(key)
        _validate_score_row(
            raw,
            heldout_aspect=str(aspect),
            split=str(split),
            model=FROZEN_RESULT_MODEL,
            examples=expected_examples[str(aspect)][str(split)],
        )
    expected_keys = {(fold, split) for fold in expected_folds for split in ("validation", "test")}
    _require_exact(seen, expected_keys, "Frozen full-reference fold/split set")
    return payload


def _training_summary_payload(
    *,
    heldout_aspect: str,
    manifest_digest: str,
    train_seconds: float,
    peak_memory: int,
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "integrity_schema_version": INTEGRITY_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "mode": "qlora",
        "heldout_aspect": heldout_aspect,
        "base_model_name_or_path": QWEN_MODEL_NAME,
        "scientific_parameters": preregistered_scientific_parameters(load_experiment_config()),
        "qwen_runtime_contract": preregistered_qwen_runtime_contract(load_experiment_config()),
        "training_manifest_hash": manifest_digest,
        "training_pairs": 4096,
        "training_positive_pairs": 2048,
        "adapter_parameters": _expected_adapter_parameters()
        | {"target_modules": list(QLORA_TARGET_MODULES)},
        "train_seconds": float(train_seconds),
        "peak_cuda_memory_bytes": int(peak_memory),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 12-fold Qwen candidate-pair confirmation.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["frozen", "qlora"], required=True)
    parser.add_argument("--model-name", default=QWEN_MODEL_NAME)
    parser.add_argument("--pilot-frozen-root", type=Path, required=True)
    parser.add_argument("--pilot-qlora-root", type=Path, default=None)
    parser.add_argument("--frozen-full-summary", type=Path, default=None)
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=64)
    parser.add_argument("--train-budget", type=int, default=4096)
    parser.add_argument("--positive-budget", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preregistration = load_experiment_config()
    validate_scientific_parameters(args, preregistration)
    frame = load_all_fabsa(args.data_dir)
    folds = validate_full_fold_set(args.heldout_aspect, all_aspects(frame))
    identity = dataset_identity(frame, folds)

    expected_examples: dict[str, dict[str, int]] = {}
    pilot_manifests: dict[str, pd.DataFrame] = {}
    pilot_validation_frames: dict[str, pd.DataFrame] = {}
    for aspect in folds:
        identity_splits = build_heldout_aspect_split(
            frame,
            [aspect],
            strategy="example_filtered",
            eval_label_scope="heldout",
            eval_row_scope="all",
        )
        expected_examples[aspect] = {
            "validation": int(len(identity_splits["validation"])),
            "test": int(len(identity_splits["test"])),
        }
        if args.mode == "qlora" and aspect in PILOT_FOLDS:
            train = identity_splits["train"]
            pilot_manifests[aspect] = budget_sample(
                build_full_manifest(train, seen_aspects(train), ENHANCED_VARIANT, seed=args.seed),
                total_budget=args.train_budget,
                positive_budget=args.positive_budget,
                seed=args.seed,
            )
        if aspect in PILOT_FOLDS:
            pilot_validation_frames[aspect] = ordered_eval_frame(
                identity_splits["validation"],
                None,
            )

    pilot_example_counts = {
        aspect: expected_examples[aspect]["validation"] for aspect in PILOT_FOLDS
    }
    frozen_pilot_results = validate_pilot_results(
        args.pilot_frozen_root,
        mode="frozen",
        preregistration=preregistration,
        expected_examples=pilot_example_counts,
        expected_validation_frames=pilot_validation_frames,
    )
    qlora_pilot_results: dict[str, dict[str, object]] | None = None
    if args.mode == "qlora":
        if args.pilot_qlora_root is None:
            raise ValueError("Full QLoRA requires the preregistered QLoRA pilot root.")
        qlora_pilot_results = validate_pilot_results(
            args.pilot_qlora_root,
            mode="qlora",
            preregistration=preregistration,
            expected_examples=pilot_example_counts,
            expected_validation_frames=pilot_validation_frames,
            expected_manifests=pilot_manifests,
            frozen_results=frozen_pilot_results,
        )
        if args.frozen_full_summary is None or not args.frozen_full_summary.exists():
            raise ValueError("Full QLoRA requires the completed full frozen-Qwen summary.")
        reference_manifest_path = args.frozen_full_summary.parent / "run_manifest.json"
        reference_manifest_preview = _read_json_object(
            reference_manifest_path,
            "full frozen-Qwen run manifest",
        )
        reference_folds = _validate_same_full_fold_set(
            reference_manifest_preview.get("folds"),
            folds,
            "Frozen-reference run-manifest folds",
        )
        _, reference_is_legacy = validate_existing_run_manifest(
            reference_manifest_path,
            mode="frozen",
            folds=reference_folds,
            preregistration=preregistration,
            expected_dataset_identity=identity,
        )
        if reference_is_legacy:
            raise ValueError(
                "Frozen full reference must be resumed once with the integrity runner before QLoRA starts."
            )
        frozen_reference_payload = validate_full_reference_summary(
            args.frozen_full_summary,
            expected_folds=folds,
            expected_examples=expected_examples,
        )
        frozen_reference_rows = {
            (str(row["heldout_aspect"]), str(row["split"])): row
            for row in frozen_reference_payload["results"]  # type: ignore[index]
        }
        for aspect in folds:
            reference_splits = build_heldout_aspect_split(
                frame,
                [aspect],
                strategy="example_filtered",
                eval_label_scope="heldout",
                eval_row_scope="all",
            )
            reference_validation = ordered_eval_frame(reference_splits["validation"], None)
            reference_test = ordered_eval_frame(reference_splits["test"], None)
            fold_rows = completed_fold_results(
                args.frozen_full_summary.parent / slugify(aspect) / "summary.json",
                heldout_aspect=aspect,
                mode="frozen",
                validation_examples=expected_examples[aspect]["validation"],
                test_examples=expected_examples[aspect]["test"],
                expected_pilot_validation=frozen_pilot_results.get(aspect),
                validation_frame=reference_validation,
                test_frame=reference_test,
            )
            if fold_rows is None:
                raise ValueError(f"Frozen full reference is missing fold evidence for {aspect!r}.")
            for row in fold_rows:
                _same_result(
                    frozen_reference_rows[(aspect, str(row["split"]))],
                    row,
                    "Frozen full root/fold result",
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = args.output_dir / "run_manifest.json"
    existing_run_manifest: dict[str, object] | None = None
    legacy_frozen_resume = False
    if args.resume:
        existing_run_manifest, legacy_frozen_resume = validate_existing_run_manifest(
            run_manifest_path,
            mode=args.mode,
            folds=folds,
            preregistration=preregistration,
            expected_dataset_identity=identity,
        )
    elif any(args.output_dir.iterdir()):
        raise ValueError("Non-resume full run requires an empty output directory.")
    run_manifest = _normalised_run_manifest(
        existing=existing_run_manifest,
        args=args,
        folds=folds,
        identity=identity,
        preregistration=preregistration,
    )
    # Do not label the in-flight legacy frozen run upgraded until every reused
    # fold has passed the v2 deep checks below.
    if not legacy_frozen_resume:
        write_json(run_manifest, run_manifest_path)
    all_results: list[dict[str, object]] = []

    frozen_model = frozen_tokenizer = frozen_ids = None

    for fold_number, heldout_aspect in enumerate(folds, start=1):
        fold_output = args.output_dir / slugify(heldout_aspect)
        fold_summary = fold_output / "summary.json"
        splits = build_heldout_aspect_split(
            frame,
            [heldout_aspect],
            strategy="example_filtered",
            eval_label_scope="heldout",
            eval_row_scope="all",
        )
        validation = ordered_eval_frame(splits["validation"], None)
        test = ordered_eval_frame(splits["test"], None)
        pilot_validation = (
            frozen_pilot_results.get(heldout_aspect)
            if args.mode == "frozen"
            else qlora_pilot_results.get(heldout_aspect)  # type: ignore[union-attr]
        )

        manifest: pd.DataFrame | None = None
        digest: str | None = None
        adapter_dir: Path | None = None
        training_summary_path: Path | None = None
        if args.mode == "qlora":
            train = splits["train"]
            manifest = pilot_manifests.get(heldout_aspect)
            if manifest is None:
                manifest = budget_sample(
                    build_full_manifest(
                        train,
                        seen_aspects(train),
                        ENHANCED_VARIANT,
                        seed=args.seed,
                    ),
                    total_budget=args.train_budget,
                    positive_budget=args.positive_budget,
                    seed=args.seed,
                )
            digest = manifest_hash(manifest)
            if pilot_validation is not None:
                assert args.pilot_qlora_root is not None
                adapter_dir = args.pilot_qlora_root / slugify(heldout_aspect) / "adapter"
                training_summary_path = (
                    args.pilot_qlora_root / slugify(heldout_aspect) / "training_summary.json"
                )
            else:
                adapter_dir = fold_output / "adapter"
                training_summary_path = fold_output / "training_summary.json"

        if args.resume:
            completed = completed_fold_results(
                fold_summary,
                heldout_aspect=heldout_aspect,
                mode=args.mode,
                validation_examples=len(validation),
                test_examples=len(test),
                expected_manifest_hash=digest,
                expected_pilot_validation=pilot_validation,
                validation_frame=validation,
                test_frame=test,
                allow_legacy_frozen=legacy_frozen_resume,
            )
            if completed is not None:
                if args.mode == "qlora":
                    assert manifest is not None and digest is not None and adapter_dir is not None
                    if pilot_validation is None:
                        validate_training_manifest_file(
                            fold_output / "training_manifest.jsonl",
                            expected_manifest=manifest,
                            expected_hash=digest,
                        )
                    validate_adapter_artifacts(
                        adapter_dir,
                        expected_heldout_aspect=heldout_aspect,
                        expected_manifest_hash=digest,
                        expected_result=completed[0],
                        training_summary_path=training_summary_path,
                        allow_legacy_pilot_summary=pilot_validation is not None,
                    )
                write_fold_summary(
                    fold_summary,
                    mode=args.mode,
                    heldout_aspect=heldout_aspect,
                    results=completed,
                )
                all_results.extend(completed)
                print(json.dumps({"event": "full_resume_fold", "heldout_aspect": heldout_aspect}), flush=True)
                continue

        fold_output.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "event": "full_fold_start",
                    "mode": args.mode,
                    "fold": fold_number,
                    "folds": len(folds),
                    "heldout_aspect": heldout_aspect,
                }
            ),
            flush=True,
        )

        training_info: dict[str, object] = {}
        if args.mode == "frozen":
            if frozen_model is None:
                frozen_tokenizer, frozen_model, frozen_ids = load_frozen_qwen(
                    args.model_name,
                    load_in_4bit=True,
                )
            model, tokenizer, verbalizer_ids = frozen_model, frozen_tokenizer, frozen_ids
        else:
            assert manifest is not None and digest is not None and adapter_dir is not None
            training_info = {
                "training_manifest_hash": digest,
                "training_pairs": int(len(manifest)),
                "training_positive_pairs": int(manifest["target"].sum()),
            }
            if pilot_validation is not None:
                training_info["training_summary"] = validate_adapter_artifacts(
                    adapter_dir,
                    expected_heldout_aspect=heldout_aspect,
                    expected_manifest_hash=digest,
                    expected_result=pilot_validation,
                    training_summary_path=training_summary_path,
                    allow_legacy_pilot_summary=True,
                )
            else:
                training_manifest_path = fold_output / "training_manifest.jsonl"
                if training_manifest_path.exists():
                    if not args.resume:
                        raise ValueError("Training manifest already exists outside resume mode.")
                    validate_training_manifest_file(
                        training_manifest_path,
                        expected_manifest=manifest,
                        expected_hash=digest,
                    )
                else:
                    write_jsonl(manifest.to_dict(orient="records"), training_manifest_path)
                    validate_training_manifest_file(
                        training_manifest_path,
                        expected_manifest=manifest,
                        expected_hash=digest,
                    )
                adapter_complete = (adapter_dir / "adapter_model.safetensors").is_file()
                if adapter_complete:
                    if not args.resume:
                        raise ValueError("Adapter already exists outside resume mode.")
                    assert training_summary_path is not None
                    summary_evidence = _read_json_object(training_summary_path, "training summary")
                    adapter_evidence = {
                        "training_manifest_hash": digest,
                        "training_pairs": int(len(manifest)),
                        "training_positive_pairs": int(manifest["target"].sum()),
                        "history": summary_evidence.get("history"),
                    }
                    training_info["training_summary"] = validate_adapter_artifacts(
                        adapter_dir,
                        expected_heldout_aspect=heldout_aspect,
                        expected_manifest_hash=digest,
                        expected_result=adapter_evidence,
                        training_summary_path=training_summary_path,
                        allow_legacy_pilot_summary=False,
                    )
                else:
                    if (adapter_dir.exists() and any(adapter_dir.iterdir())) or (
                        training_summary_path is not None and training_summary_path.exists()
                    ):
                        raise ValueError(
                            f"Partial adapter artifacts cannot be trusted or overwritten: {adapter_dir}"
                        )
                    set_seed(args.seed)
                    qlora_config = QwenPairQLoRAConfig(
                        load_in_4bit=True,
                        gradient_checkpointing=True,
                        lora_r=4,
                        lora_alpha=8,
                        lora_dropout=0.05,
                        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                    )
                    train_tokenizer, train_model, train_ids = load_qwen_pair_qlora(
                        args.model_name,
                        qlora_config,
                    )
                    train_started = time.time()
                    history, peak_memory = train_qwen_adapter(
                        train_model,
                        train_tokenizer,
                        train_ids,
                        manifest,
                        args,
                    )
                    train_seconds = time.time() - train_started
                    train_model.save_pretrained(adapter_dir)
                    summary_evidence = _training_summary_payload(
                        heldout_aspect=heldout_aspect,
                        manifest_digest=digest,
                        train_seconds=train_seconds,
                        peak_memory=peak_memory,
                        history=history,
                    )
                    assert training_summary_path is not None
                    write_json(summary_evidence, training_summary_path)
                    del train_model, train_tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    adapter_evidence = {
                        "training_manifest_hash": digest,
                        "training_pairs": int(len(manifest)),
                        "training_positive_pairs": int(manifest["target"].sum()),
                        "history": history,
                    }
                    training_info["training_summary"] = validate_adapter_artifacts(
                        adapter_dir,
                        expected_heldout_aspect=heldout_aspect,
                        expected_manifest_hash=digest,
                        expected_result=adapter_evidence,
                        training_summary_path=training_summary_path,
                        allow_legacy_pilot_summary=False,
                    )
            tokenizer, model, verbalizer_ids = load_saved_qwen_pair_adapter(
                args.model_name,
                adapter_dir,
            )

        if pilot_validation is None:
            validation_result, threshold = score_split(
                model,
                tokenizer,
                verbalizer_ids,
                validation,
                heldout_aspect,
                "validation",
                fold_output,
                args,
            )
        else:
            validation_result = dict(pilot_validation)
            threshold = float(validation_result["selected_threshold"])
            validation_result["reused_preregistered_pilot_validation"] = True
        test_result, _ = score_split(
            model,
            tokenizer,
            verbalizer_ids,
            test,
            heldout_aspect,
            "test",
            fold_output,
            args,
            threshold=threshold,
        )
        model_name = _result_model(args.mode)
        fold_results = [
            {"model": model_name, **training_info, **validation_result},
            {"model": model_name, **training_info, **test_result},
        ]
        write_fold_summary(
            fold_summary,
            mode=args.mode,
            heldout_aspect=heldout_aspect,
            results=fold_results,
        )
        all_results.extend(fold_results)
        aggregate(
            all_results,
            args.output_dir,
            args.mode,
            args.frozen_full_summary,
            folds,
        )
        if args.mode == "qlora":
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    if frozen_model is not None:
        del frozen_model, frozen_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    if len(all_results) != FULL_FOLD_COUNT * 2:
        raise RuntimeError("Formal full run ended without exactly 24 validation/test rows.")
    run_manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_manifest, run_manifest_path)
    summary = aggregate(
        all_results,
        args.output_dir,
        args.mode,
        args.frozen_full_summary,
        folds,
    )
    print(json.dumps({"event": "full_run_complete", "spread": summary["spread"], "confirmation": summary["full_confirmation"]}, default=json_default), flush=True)


if __name__ == "__main__":
    main()
