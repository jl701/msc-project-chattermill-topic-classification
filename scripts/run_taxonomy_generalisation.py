from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.experiments.taxonomy_analysis import (
    condition_cluster_uncertainty,
    level3_condition_diagnostics,
    per_aspect_metrics,
    per_sentiment_metrics,
)
from msc_project.experiments.taxonomy_artifacts import (
    build_training_contract as training_contract,
    checkpoint_dir_for,
    resolve_parameters,
    summary_path_for,
    threshold_path_for,
)
from msc_project.experiments.taxonomy_audit import (
    TestUseLedger,
    assert_formal_run_gates,
    audit_strict_calibration,
)
from msc_project.experiments.taxonomy_checkpoints import (
    checkpoint_resume_state,
)
from msc_project.experiments.taxonomy_execution import (
    merge_score_shards,
    row_shard_index,
    score_shard_paths,
    validate_score_shard,
)
from msc_project.experiments.frozen_raw_score_cache import (
    FrozenRawScoreCache,
    build_frozen_raw_score_cache_contract,
    frozen_raw_score_cache_path,
)
from msc_project.experiments.taxonomy_methods import (
    METHOD_IDS,
    resolve_method_spec,
)
from msc_project.experiments.taxonomy_pipeline import (
    assert_condition_pair_identity,
    build_run_contract,
    prepare_fold_evaluation,
    prepare_fold_training,
    prepare_strict_seen_calibration,
    score_prepared_shard,
    score_prepared_conditions_shard,
)
from msc_project.experiments.taxonomy_protocol import (
    TaxonomyFold,
    evaluate_scored_grid,
    load_precloud_config,
    pair_identity_hash,
    registered_folds,
    select_strict_seen_threshold,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import load_description_bundle
from msc_project.experiments.taxonomy_runtime_factory import (
    create_runtime_for_training,
    load_runtime_checkpoint,
    save_frozen_registry_checkpoint,
    save_runtime_checkpoint,
)
from msc_project.experiments.taxonomy_selection import (
    build_threshold_transfer_artifact,
    load_threshold_transfer_artifact,
    write_threshold_transfer_artifact,
)


PHASES = (
    "train",
    "score-validation",
    "select-threshold",
    "score-test",
    "analyse-test",
)

WINDOWS_PROCESS_PRIORITY_CLASSES = {
    "normal": 0x00000020,
    "above-normal": 0x00008000,
    "high": 0x00000080,
}


def configure_windows_process_priority(
    method_id: str,
    requested: str = "auto",
    *,
    platform_name: str | None = None,
    kernel32: object | None = None,
) -> dict[str, object]:
    """Apply the opt-out Windows priority policy for QLoRA workers.

    ``auto`` leaves every other method and non-Windows host unchanged, while
    QLoRA workers on Windows select ``HIGH_PRIORITY_CLASS`` before model or
    dataset loading begins.  The explicit choices remain available for
    diagnostics and for hosts where interactive responsiveness matters more
    than training throughput.
    """

    if requested not in {"auto", *WINDOWS_PROCESS_PRIORITY_CLASSES}:
        raise ValueError(f"Unknown Windows process priority: {requested!r}.")
    platform_value = platform_name or sys.platform
    effective = None
    if requested == "auto" and method_id == "qwen_candidate_pair_qlora":
        effective = "high"
    elif requested != "auto":
        effective = requested
    evidence: dict[str, object] = {
        "event": "windows_process_priority",
        "method_id": method_id,
        "requested": requested,
        "effective": effective or "unchanged",
        "platform": platform_value,
        "applied": False,
    }
    if platform_value != "win32" or effective is None:
        return evidence

    if kernel32 is None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
    process_handle = kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
    applied = kernel32.SetPriorityClass(  # type: ignore[attr-defined]
        process_handle,
        WINDOWS_PROCESS_PRIORITY_CLASSES[effective],
    )
    if not applied:
        import ctypes

        raise ctypes.WinError(ctypes.get_last_error())
    evidence["applied"] = True
    return evidence


def _write_json_new(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale result temporary file: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def resolve_fold(level: str, fold_id: str) -> TaxonomyFold:
    matches = [
        fold for fold in registered_folds(level) if fold.fold_id == fold_id
    ]
    if len(matches) != 1:
        valid = [fold.fold_id for fold in registered_folds(level)]
        raise ValueError(f"Unknown fold_id {fold_id!r}; valid values are {valid}.")
    return matches[0]


def _prepared_conditions(
    frame: pd.DataFrame,
    fold: TaxonomyFold,
    conditions: Sequence[str],
    split_name: str,
    resource: Mapping[str, object],
    *,
    seed: int,
) -> list:
    values = [
        prepare_fold_evaluation(
            frame,
            fold,
            condition,
            split_name,
            resource,
            seed=seed,
        )
        for condition in conditions
    ]
    assert_condition_pair_identity(values)
    return values


def _load_all_shards(prepared, contract, output_root: Path) -> pd.DataFrame:
    artifacts = {}
    for shard_index in range(contract.shard_count):
        expected = prepared.evaluation_grid[
            prepared.evaluation_grid["row_uid"].astype(str).map(
                lambda value: row_shard_index(value, contract.shard_count)
                == shard_index
            )
        ]
        if expected.empty:
            continue
        csv_path, manifest_path = score_shard_paths(
            output_root,
            contract,
            shard_index,
        )
        artifacts[shard_index] = validate_score_shard(
            csv_path,
            manifest_path,
            contract,
            prepared.evaluation_grid,
            shard_index=shard_index,
        )
    return merge_score_shards(artifacts, contract, prepared.evaluation_grid)


def _validated_existing_threshold_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    fold: TaxonomyFold,
    parameters_sha256: str,
    training_contract_sha256: str,
) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("event") != "threshold_selected":
        raise ValueError("Existing validation summary has the wrong event.")
    observation = value.get("tuning_observation")
    if not isinstance(observation, dict):
        raise ValueError("Existing validation summary lacks a tuning observation.")
    expected = {
        "method_id": args.method,
        "candidate_id": args.candidate_id,
        "parameters_sha256": parameters_sha256,
        "fold_id": fold.fold_id,
        "calibration_scope": "seen_aspects_only",
    }
    mismatches = {
        key: {"expected": expected_value, "observed": observation.get(key)}
        for key, expected_value in expected.items()
        if observation.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Existing validation summary contract mismatch: {mismatches}"
        )
    condition_metrics = value.get("per_condition_selection_metrics")
    if not isinstance(condition_metrics, dict) or set(condition_metrics) != set(
        fold.conditions
    ):
        raise ValueError("Existing validation summary has wrong fold conditions.")
    thresholds = {
        float(metrics["threshold"])
        for metrics in condition_metrics.values()
    }
    if len(thresholds) != 1:
        raise ValueError("Existing validation summary does not share one threshold.")

    threshold_artifact_sha256 = None
    if args.purpose == "final":
        artifact = load_threshold_transfer_artifact(
            threshold_path_for(
                args.output_root,
                args.method,
                fold,
                parameters_sha256,
                args.seed,
            ),
            fold=fold,
            method_id=args.method,
            scientific_parameters_sha256=parameters_sha256,
            training_contract_sha256=training_contract_sha256,
        )
        if value.get("threshold_artifact") != artifact.to_dict():
            raise ValueError(
                "Existing validation summary and threshold artifact disagree."
            )
        threshold_artifact_sha256 = artifact.artifact_sha256

    return {
        "event": "threshold_selection_resumed",
        "summary_path": str(path),
        "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "threshold": next(iter(thresholds)),
        "threshold_artifact_sha256": threshold_artifact_sha256,
    }


def _validate_purpose(
    purpose: str,
    phase: str,
    method_id: str,
    fold: TaxonomyFold,
    candidate_id: str | None,
) -> None:
    if purpose == "tuning":
        if phase in {"score-test", "analyse-test"}:
            raise ValueError("Tuning jobs may never access the test split.")
        if (
            resolve_method_spec(method_id)
            .seen_only_tuning.get("selection_scope")
            != "nested_per_training_scope"
        ):
            raise ValueError("This method has no registered nested tuning grid.")
        if candidate_id is None:
            raise ValueError("Tuning requires a registered candidate_id.")
    elif purpose == "final":
        if candidate_id is not None:
            raise ValueError("Final runs require a frozen parameter-selection artifact.")
    else:
        raise ValueError("purpose must be tuning or final.")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.method not in METHOD_IDS:
        raise ValueError(f"Unknown method: {args.method}")
    fold = resolve_fold(args.level, args.fold_id)
    default_conditions = (
        ("D",)
        if args.purpose == "tuning" and "D" in fold.conditions
        else fold.conditions
    )
    conditions = tuple(args.condition or default_conditions)
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("Conditions must be non-empty and unique.")
    if set(conditions) - set(fold.conditions):
        raise ValueError("A requested condition is not registered for the fold.")
    if args.purpose == "final" and set(conditions) != set(fold.conditions):
        raise ValueError("Final jobs must cover every registered fold condition.")
    _validate_purpose(
        args.purpose,
        args.phase,
        args.method,
        fold,
        args.candidate_id,
    )
    parameters, parameters_sha256 = resolve_parameters(
        args.method,
        candidate_id=args.candidate_id,
        selection_path=args.parameter_selection,
        expected_training_scope_id=training_scope_id(fold),
    )
    resource = load_description_bundle(require_approved=False)
    config = load_precloud_config()
    assert_formal_run_gates(resource, config)

    if args.phase == "train":
        frame = load_official_fabsa_splits(args.data_dir, ("train",))
        _, training_manifest = prepare_fold_training(
            frame,
            fold,
            resource,
            seed=args.seed,
        )
        train_contract = training_contract(
            args.method,
            fold,
            training_manifest,
            parameters_sha256,
            seed=args.seed,
        )
        checkpoint_dir = checkpoint_dir_for(args.output_root, train_contract)
        state, _ = checkpoint_resume_state(checkpoint_dir, train_contract)
        if state == "complete":
            if not args.resume:
                raise FileExistsError("Checkpoint exists; use --resume to validate reuse.")
            return {
                "event": "checkpoint_resumed",
                "checkpoint_dir": str(checkpoint_dir),
                "training_contract_sha256": train_contract.contract_sha256,
            }
        spec = resolve_method_spec(args.method)
        if not spec.requires_pair_training:
            save_frozen_registry_checkpoint(
                args.method,
                checkpoint_dir,
                train_contract,
            )
            return {
                "event": "frozen_checkpoint_created",
                "checkpoint_dir": str(checkpoint_dir),
                "training_contract_sha256": train_contract.contract_sha256,
                "task_specific_fit_performed": False,
            }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runtime = create_runtime_for_training(
            args.method,
            parameters,
            seed=args.seed,
            device=device,
            local_files_only=args.local_files_only,
        )
        runtime.fit(training_manifest)
        save_runtime_checkpoint(runtime, checkpoint_dir, train_contract)
        runtime.close()
        return {
            "event": "checkpoint_created",
            "checkpoint_dir": str(checkpoint_dir),
            "training_contract_sha256": train_contract.contract_sha256,
        }

    split_name = "test" if args.phase in {"score-test", "analyse-test"} else "validation"
    frame = load_official_fabsa_splits(
        args.data_dir,
        ("train", split_name),
    )
    calibration_prepared = None
    if split_name == "validation":
        calibration_prepared = prepare_strict_seen_calibration(
            frame,
            fold,
            resource,
            seed=args.seed,
        )
        prepared = []
        training_manifest = calibration_prepared.training_manifest
    else:
        prepared = _prepared_conditions(
            frame,
            fold,
            conditions,
            split_name,
            resource,
            seed=args.seed,
        )
        training_manifest = prepared[0].training_manifest
    train_contract = training_contract(
        args.method,
        fold,
        training_manifest,
        parameters_sha256,
        seed=args.seed,
    )
    checkpoint_dir = checkpoint_dir_for(args.output_root, train_contract)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime = None
    if args.phase in {"score-validation", "score-test"}:
        runtime = load_runtime_checkpoint(
            args.method,
            parameters,
            checkpoint_dir,
            train_contract,
            seed=args.seed,
            device=device,
            local_files_only=args.local_files_only,
            lazy_frozen_qwen=(
                args.method == "frozen_qwen_candidate_pair"
            ),
        )

    score_contracts = {
        value.condition: build_run_contract(
            value,
            args.method,
            parameters,
            shard_count=args.shard_count,
            formal=True,
            seed=args.seed,
        )
        for value in prepared
    }
    calibration_contract = None
    if calibration_prepared is not None:
        calibration_contract = build_run_contract(
            calibration_prepared,
            args.method,
            parameters,
            shard_count=args.shard_count,
            formal=True,
            seed=args.seed,
        )

    if args.phase in {"score-validation", "score-test"}:
        assert runtime is not None
        if args.all_shards:
            shard_indices = tuple(range(args.shard_count))
        else:
            if args.shard_index < 0 or args.shard_index >= args.shard_count:
                raise ValueError("--shard-index must be inside --shard-count.")
            shard_indices = (args.shard_index,)
        ledger = TestUseLedger(args.output_root / "_governance" / "test_use.json")
        if args.phase == "score-test":
            load_threshold_transfer_artifact(
                threshold_path_for(
                    args.output_root,
                    args.method,
                    fold,
                    parameters_sha256,
                    args.seed,
                ),
                fold=fold,
                method_id=args.method,
                scientific_parameters_sha256=parameters_sha256,
                training_contract_sha256=train_contract.contract_sha256,
            )
        raw_score_cache = None
        results = []
        unique_pairs_scored = 0
        cache_summary = None
        try:
            for value in prepared:
                contract = score_contracts[value.condition]
                if args.phase == "score-test":
                    ledger.claim(contract)
            if args.method == "frozen_qwen_candidate_pair":
                cache_run_contract = (
                    calibration_contract
                    if calibration_contract is not None
                    else next(iter(score_contracts.values()))
                )
                assert cache_run_contract is not None
                cache_contract = build_frozen_raw_score_cache_contract(
                    cache_run_contract,
                    parameters,
                )
                raw_score_cache = FrozenRawScoreCache(
                    frozen_raw_score_cache_path(
                        args.output_root,
                        cache_contract,
                    ),
                    cache_contract,
                )
            for shard_index in shard_indices:
                # Strict validation never scores held-out candidates or labels.
                if calibration_prepared is None:
                    artifacts, states, scored_count = score_prepared_conditions_shard(
                        prepared,
                        runtime,
                        score_contracts,
                        args.output_root,
                        shard_index=shard_index,
                        resume=args.resume,
                        raw_score_cache=raw_score_cache,
                    )
                    unique_pairs_scored += scored_count
                    results.extend(
                        {
                            "condition": value.condition,
                            "shard_index": shard_index,
                            "state": states[value.condition],
                            "rows": int(len(artifacts[value.condition])),
                            "score_contract_sha256": score_contracts[
                                value.condition
                            ].contract_sha256,
                        }
                        for value in prepared
                    )
                if calibration_prepared is not None:
                    assert calibration_contract is not None
                    calibration_artifact, calibration_state = score_prepared_shard(
                        calibration_prepared,
                        runtime,
                        calibration_contract,
                        args.output_root,
                        shard_index=shard_index,
                        resume=args.resume,
                        raw_score_cache=raw_score_cache,
                    )
                    results.append(
                        {
                            "condition": "seen-calibration",
                            "shard_index": shard_index,
                            "state": calibration_state,
                            "rows": int(len(calibration_artifact)),
                            "score_contract_sha256": calibration_contract.contract_sha256,
                        }
                    )
                    if calibration_state == "scored":
                        unique_pairs_scored += len(calibration_artifact)
            if raw_score_cache is not None:
                cache_summary = raw_score_cache.session_summary()
                unique_pairs_scored = int(
                    cache_summary["model_scored_unique_inputs"]
                )
        finally:
            if raw_score_cache is not None:
                raw_score_cache.close()
            runtime.close()
        return {
            "event": args.phase,
            "checkpoint_dir": str(checkpoint_dir),
            "unique_pairs_scored": unique_pairs_scored,
            "raw_score_cache": cache_summary,
            "results": results,
        }

    if args.phase == "select-threshold":
        validation_summary_path = summary_path_for(
            args.output_root,
            args.purpose,
            "validation",
            args.method,
            fold,
            parameters_sha256,
            args.seed,
        )
        if args.resume and validation_summary_path.exists():
            return _validated_existing_threshold_summary(
                validation_summary_path,
                args=args,
                fold=fold,
                parameters_sha256=parameters_sha256,
                training_contract_sha256=train_contract.contract_sha256,
            )
        selections = {}
        merged_by_condition = {}
        assert calibration_prepared is not None
        assert calibration_contract is not None
        calibration_scores = _load_all_shards(
            calibration_prepared,
            calibration_contract,
            args.output_root,
        )
        selection = select_strict_seen_threshold(
            calibration_scores,
            seen_aspects=fold.seen_aspects,
            heldout_aspects=fold.heldout_aspects,
        )
        audit = audit_strict_calibration(selection.calibration_aspects, fold)
        if not audit.passed:
            raise ValueError(f"Strict calibration audit failed: {audit.details}")
        selections = {
            condition: selection
            for condition in fold.conditions
        }
        validation_score_contracts = {
            condition: calibration_contract.contract_sha256
            for condition in fold.conditions
        }
        validation_pair_identity_sha256 = pair_identity_hash(
            calibration_prepared.evaluation_grid
        )
        selected_condition = "D" if "D" in selections else fold.conditions[-1]
        selected_metrics = selections[selected_condition].metrics["seen"]
        tuning_observation = {
            "method_id": args.method,
            "candidate_id": args.candidate_id,
            "parameters_sha256": parameters_sha256,
            "fold_id": fold.fold_id,
            "calibration_scope": "seen_aspects_only",
            **{
                key: float(selected_metrics[key])
                for key in (
                    "pair_micro_f1",
                    "pair_samples_f1",
                    "pair_micro_precision",
                    "presence_false_positive_rows_per_100",
                )
            },
        }
        summary: dict[str, object] = {
            "event": "threshold_selected",
            "tuning_observation": tuning_observation,
            "per_condition_selection_metrics": {
                condition: selection.metrics
                for condition, selection in selections.items()
            },
        }
        if args.purpose == "final":
            artifact = build_threshold_transfer_artifact(
                fold,
                args.method,
                selections,
                protocol_id="taxonomy_generalisation_precloud_v1",
                scientific_parameters_sha256=parameters_sha256,
                training_contract_sha256=train_contract.contract_sha256,
                validation_score_contracts=validation_score_contracts,
                validation_pair_identity_sha256=validation_pair_identity_sha256,
            )
            path = threshold_path_for(
                args.output_root,
                args.method,
                fold,
                parameters_sha256,
                args.seed,
            )
            write_threshold_transfer_artifact(path, artifact)
            summary["threshold_artifact"] = artifact.to_dict()
        _write_json_new(
            validation_summary_path,
            summary,
        )
        return summary

    threshold = load_threshold_transfer_artifact(
        threshold_path_for(
            args.output_root,
            args.method,
            fold,
            parameters_sha256,
            args.seed,
        ),
        fold=fold,
        method_id=args.method,
        scientific_parameters_sha256=parameters_sha256,
        training_contract_sha256=train_contract.contract_sha256,
    )
    ledger = TestUseLedger(args.output_root / "_governance" / "test_use.json")
    condition_results = {}
    for value in prepared:
        contract = score_contracts[value.condition]
        merged = _load_all_shards(value, contract, args.output_root)
        metrics = evaluate_scored_grid(
            merged,
            threshold.threshold,
            seen_aspects=fold.seen_aspects,
            heldout_aspects=fold.heldout_aspects,
        )
        result = {
            "metrics": metrics,
            "uncertainty": condition_cluster_uncertainty(
                merged,
                threshold.threshold,
                fold,
                replicates=int(config["statistics"]["bootstrap_replicates"]),
                seed=int(config["statistics"]["bootstrap_seed"]),
            ),
            "per_aspect": per_aspect_metrics(
                merged, threshold.threshold
            ).to_dict(orient="records"),
            "per_sentiment": per_sentiment_metrics(
                merged, threshold.threshold
            ).to_dict(orient="records"),
        }
        if fold.level == "L3":
            result["level3_diagnostics"] = level3_condition_diagnostics(
                merged,
                fold,
                value.condition,
                threshold.threshold,
            )
        condition_results[value.condition] = result
        ledger.complete(contract)
    summary = {
        "event": "test_analysis_complete",
        "method_id": args.method,
        "fold_id": fold.fold_id,
        "parameters_sha256": parameters_sha256,
        "training_contract_sha256": train_contract.contract_sha256,
        "threshold_artifact_sha256": threshold.artifact_sha256,
        "conditions": condition_results,
    }
    _write_json_new(
        summary_path_for(
            args.output_root,
            args.purpose,
            "test",
            args.method,
            fold,
            parameters_sha256,
            args.seed,
        ),
        summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one guarded stage of the taxonomy-generalisation suite."
    )
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--purpose", choices=["tuning", "final"], required=True)
    parser.add_argument("--method", choices=METHOD_IDS, required=True)
    parser.add_argument("--level", choices=["L1", "L2", "L3", "L4"], required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--candidate-id")
    parser.add_argument("--parameter-selection", type=Path)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--all-shards",
        action="store_true",
        help="Process every registered shard after loading the model once.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--windows-process-priority",
        choices=("auto", *WINDOWS_PROCESS_PRIORITY_CLASSES),
        default="auto",
        help=(
            "Windows worker priority. 'auto' (default) uses High for QLoRA "
            "and leaves every other method or operating system unchanged."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    priority_evidence = configure_windows_process_priority(
        args.method,
        args.windows_process_priority,
    )
    print(json.dumps(priority_evidence, ensure_ascii=False), flush=True)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
