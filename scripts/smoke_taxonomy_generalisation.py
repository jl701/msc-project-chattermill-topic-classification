from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments.taxonomy_analysis import assert_shared_seen_threshold
from msc_project.experiments.taxonomy_checkpoints import (
    TrainingContract,
    checkpoint_resume_state,
)
from msc_project.experiments.taxonomy_execution import (
    canonical_sha256,
    merge_score_shards,
    select_pair_shard,
)
from msc_project.experiments.taxonomy_methods import (
    METHOD_IDS,
    method_registry_sha256,
    resolve_method_spec,
)
from msc_project.experiments.taxonomy_pipeline import (
    assert_condition_pair_identity,
    build_run_contract,
    prepare_fold_evaluation,
    score_prepared_conditions_shard,
    unique_rendered_claim_count,
)
from msc_project.experiments.taxonomy_protocol import (
    registered_folds,
    select_strict_seen_threshold,
    scientific_protocol_sha256,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions
from msc_project.experiments.taxonomy_runtime_factory import (
    create_runtime_for_training,
    load_runtime_checkpoint,
    save_runtime_checkpoint,
)
from msc_project.experiments.unified_candidate_pairs import manifest_hash


def synthetic_frame() -> pd.DataFrame:
    fold = registered_folds("L3")[0]
    rows = []
    row_id = 1
    for index, aspect in enumerate(fold.seen_aspects):
        sentiment = ("negative", "neutral", "positive")[index % 3]
        rows.append(
            {
                "id": row_id,
                "original_split": "train",
                "row_uid": f"train:{row_id}",
                "text": f"Synthetic training review about {aspect} with {sentiment} sentiment.",
                "labels": [(aspect, sentiment)],
            }
        )
        row_id += 1
    rows.extend(
        [
            {
                "id": row_id,
                "original_split": "validation",
                "row_uid": f"validation:{row_id}",
                "text": "Synthetic review containing both held-out candidates.",
                "labels": [
                    (fold.heldout_aspects[0], "positive"),
                    (fold.heldout_aspects[1], "negative"),
                ],
            },
            {
                "id": row_id + 1,
                "original_split": "validation",
                "row_uid": f"validation:{row_id + 1}",
                "text": "Synthetic review containing neither held-out candidate.",
                "labels": [],
            },
            {
                "id": row_id + 2,
                "original_split": "validation",
                "row_uid": f"validation:{row_id + 2}",
                "text": "Synthetic review containing only the second held-out candidate.",
                "labels": [(fold.heldout_aspects[1], "positive")],
            },
        ]
    )
    frame = pd.DataFrame(rows)
    frame["org_index"] = range(1, len(frame) + 1)
    frame["industry"] = "synthetic"
    frame["data_source"] = "synthetic"
    frame["pair_labels"] = frame["labels"].apply(
        lambda labels: [f"{aspect} | {sentiment}" for aspect, sentiment in labels]
    )
    frame["aspect_labels"] = frame["labels"].apply(
        lambda labels: sorted({aspect for aspect, _ in labels})
    )
    return frame


def smoke_parameters(method_id: str) -> dict[str, object]:
    parameters = dict(resolve_method_spec(method_id).starting_recipe)
    if method_id == "distilbert_review_candidate_cross_encoder":
        parameters.update(
            {
                "max_length": 64,
                "batch_size": 8,
                "eval_batch_size": 16,
                "epochs": 1,
                "selected_checkpoint_epoch": 1,
            }
        )
    elif method_id == "frozen_qwen_candidate_pair":
        parameters.update({"max_length": 128, "eval_batch_size": 2})
    elif method_id == "qwen_candidate_pair_qlora":
        parameters.update(
            {
                "max_length": 128,
                "gradient_accumulation_steps": 2,
                "epochs": 1,
                "selected_checkpoint_epoch": 1,
                "eval_batch_size": 2,
            }
        )
    elif method_id == "e5_base_v2":
        parameters["batch_size"] = 8
    return parameters


def run_smoke(
    method_id: str,
    output_root: Path,
    *,
    conditions: tuple[str, ...] = ("NN", "DN", "ND", "DD"),
    seed: int = 13,
    local_files_only: bool = True,
) -> dict[str, object]:
    fold = registered_folds("L3")[0]
    if set(conditions) - set(fold.conditions):
        raise ValueError("Smoke condition is not registered for Level 3.")
    resource = load_minimal_descriptions(require_approved=False)
    frame = synthetic_frame()
    prepared = [
        prepare_fold_evaluation(
            frame,
            fold,
            condition,
            "validation",
            resource,
            total_budget=64,
            positive_budget=32,
            seed=seed,
        )
        for condition in conditions
    ]
    pair_hash = assert_condition_pair_identity(prepared)
    parameters = smoke_parameters(method_id)
    parameters_sha256 = canonical_sha256(parameters)
    spec = resolve_method_spec(method_id)
    train_contract = TrainingContract(
        protocol_id="taxonomy_generalisation_precloud_v1_smoke",
        scientific_protocol_sha256=scientific_protocol_sha256(),
        method_id=method_id,
        method_spec_sha256=spec.spec_sha256,
        method_registry_sha256=method_registry_sha256(),
        training_scope_id=training_scope_id(fold),
        seed=seed,
        training_manifest_sha256=manifest_hash(prepared[0].training_manifest),
        scientific_parameters_sha256=parameters_sha256,
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        training_pairs=len(prepared[0].training_manifest),
    )
    checkpoint_dir = (
        output_root
        / method_id
        / parameters_sha256[:16]
        / "checkpoint"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_state, _ = checkpoint_resume_state(checkpoint_dir, train_contract)
    if checkpoint_state == "missing":
        started = time.perf_counter()
        runtime = create_runtime_for_training(
            method_id,
            parameters,
            seed=seed,
            device=device,
            local_files_only=local_files_only,
        )
        runtime.fit(prepared[0].training_manifest)
        fit_seconds = time.perf_counter() - started
        save_runtime_checkpoint(runtime, checkpoint_dir, train_contract)
        runtime.close()
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        fit_seconds = 0.0

    runtime = load_runtime_checkpoint(
        method_id,
        parameters,
        checkpoint_dir,
        train_contract,
        seed=seed,
        device=device,
        local_files_only=local_files_only,
    )
    selections = {}
    condition_results = {}
    score_started = time.perf_counter()
    score_contracts = {
        value.condition: build_run_contract(
            value,
            method_id,
            parameters,
            shard_count=2,
            formal=False,
            protocol_id="taxonomy_generalisation_precloud_v1_smoke",
            seed=seed,
        )
        for value in prepared
    }
    artifacts_by_condition = {value.condition: {} for value in prepared}
    shard_states_by_condition = {value.condition: {} for value in prepared}
    unique_pairs_scored = 0
    for shard_index in range(2):
        representative = prepared[0]
        representative_contract = score_contracts[representative.condition]
        shard = select_pair_shard(
            representative.evaluation_grid,
            shard_index,
            representative_contract.shard_count,
        )
        if shard.empty:
            continue
        artifacts, states, scored_count = score_prepared_conditions_shard(
            prepared,
            runtime,
            score_contracts,
            output_root / "scores",
            shard_index=shard_index,
            resume=True,
        )
        resumed, resume_states, resumed_count = score_prepared_conditions_shard(
            prepared,
            runtime,
            score_contracts,
            output_root / "scores",
            shard_index=shard_index,
            resume=True,
        )
        if resumed_count != 0:
            raise AssertionError("A complete condition shard was unexpectedly rescored.")
        for condition in conditions:
            pd.testing.assert_frame_equal(
                artifacts[condition],
                resumed[condition],
            )
            artifacts_by_condition[condition][shard_index] = artifacts[condition]
            shard_states_by_condition[condition][str(shard_index)] = {
                "score_or_resume_state": states[condition],
                "resume_validated": resume_states[condition] == "resumed",
            }
        unique_pairs_scored += scored_count

    for value in prepared:
        contract = score_contracts[value.condition]
        merged = merge_score_shards(
            artifacts_by_condition[value.condition],
            contract,
            value.evaluation_grid,
        )
        selection = select_strict_seen_threshold(
            merged,
            seen_aspects=fold.seen_aspects,
            heldout_aspects=fold.heldout_aspects,
        )
        selections[value.condition] = selection
        condition_results[value.condition] = {
            "score_contract_sha256": contract.contract_sha256,
            "pairs": len(merged),
            "threshold": selection.threshold,
            "seen_validation_pair_micro_f1": selection.metrics["seen"][
                "pair_micro_f1"
            ],
            "shard_states": shard_states_by_condition[value.condition],
        }
    score_seconds = time.perf_counter() - score_started
    shared_threshold = assert_shared_seen_threshold(selections)
    runtime.close()
    report = {
        "schema_version": "taxonomy_precloud_real_model_smoke_v1",
        "method_id": method_id,
        "device": str(device),
        "fold_id": fold.fold_id,
        "conditions": list(conditions),
        "training_pairs": len(prepared[0].training_manifest),
        "evaluation_pairs_per_condition": len(prepared[0].evaluation_grid),
        "pair_identity_sha256": pair_hash,
        "parameters": parameters,
        "parameters_sha256": parameters_sha256,
        "training_contract_sha256": train_contract.contract_sha256,
        "checkpoint_dir": str(checkpoint_dir),
        "shared_seen_threshold": shared_threshold,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "unique_pairs_across_conditions": unique_rendered_claim_count(prepared),
        "new_pairs_scored_this_run": unique_pairs_scored,
        "condition_results": condition_results,
        "official_data_read": False,
    }
    report_path = (
        output_root
        / method_id
        / parameters_sha256[:16]
        / "smoke_summary.json"
    )
    if report_path.exists():
        existing_raw = json.loads(report_path.read_text(encoding="utf-8"))
        existing = json.loads(json.dumps(existing_raw))
        # Normalise via a JSON round trip so removing volatile runtime fields
        # cannot mutate the report returned to the caller.
        comparison = json.loads(json.dumps(report))
        for value in (existing, comparison):
            value.pop("fit_seconds", None)
            value.pop("score_seconds", None)
            value.pop("new_pairs_scored_this_run", None)
            # Backward-compatible normalisation for reports produced before
            # the invariant/dynamic unique-count fields were separated.
            if (
                "unique_pairs_across_conditions" not in value
                and "unique_pairs_scored_across_conditions" in value
            ):
                value["unique_pairs_across_conditions"] = value.pop(
                    "unique_pairs_scored_across_conditions"
                )
            for condition in value["condition_results"].values():
                for shard in condition["shard_states"].values():
                    shard.pop("score_or_resume_state", None)
        if existing != comparison:
            raise ValueError("Existing smoke report differs from the resumed run.")
        if "unique_pairs_scored_across_conditions" in existing_raw:
            migrated = json.loads(json.dumps(existing_raw))
            count = migrated.pop("unique_pairs_scored_across_conditions")
            migrated["unique_pairs_across_conditions"] = count
            migrated["new_pairs_scored_this_run"] = count
            report_path.write_text(
                json.dumps(migrated, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    else:
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded real-model taxonomy smoke on synthetic data only."
    )
    parser.add_argument("--method", choices=METHOD_IDS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    result = run_smoke(
        args.method,
        args.output_root,
        conditions=tuple(args.condition or ("NN", "DN", "ND", "DD")),
        local_files_only=not args.allow_download,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
