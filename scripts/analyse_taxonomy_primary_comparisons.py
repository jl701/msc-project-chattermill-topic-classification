from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.experiments.taxonomy_artifacts import (
    build_training_contract,
    resolve_parameters,
    threshold_path_for,
)
from msc_project.experiments.taxonomy_audit import assert_formal_run_gates
from msc_project.experiments.taxonomy_comparisons import (
    ProtocolResult,
    aggregate_protocol_predictions,
    canonical_pair_classes,
    cross_protocol_degradation,
    holm_adjust_comparison_family,
    matched_protocol_comparison,
)
from msc_project.experiments.statistical_inference import (
    build_cluster_sufficient_statistics,
    cluster_bootstrap_interval,
)
from msc_project.experiments.taxonomy_execution import (
    canonical_sha256,
    merge_score_shards,
    row_shard_index,
    score_shard_paths,
    validate_score_shard,
)
from msc_project.experiments.taxonomy_methods import METHOD_IDS
from msc_project.experiments.taxonomy_pipeline import (
    build_run_contract,
    prepare_fold_evaluation,
)
from msc_project.experiments.taxonomy_protocol import (
    load_precloud_config,
    registered_folds,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions
from msc_project.experiments.taxonomy_selection import (
    load_threshold_transfer_artifact,
)
from msc_project.experiments.taxonomy_tuning import registered_tuning_candidates


ENDPOINT_CONDITIONS = {
    "L1": "D",
    "L2": "D",
    "L3": "DD",
    "L4": "D",
}


def parameter_selection_path(
    selection_dir: Path,
    method_id: str,
    fold,
) -> Path:
    if len(registered_tuning_candidates(method_id)) == 1:
        return selection_dir / f"{method_id}.json"
    return (
        selection_dir
        / method_id
        / f"{training_scope_id(fold)}.json"
    )


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


def load_protocol_results(
    frame: pd.DataFrame,
    resource: dict[str, object],
    *,
    method_id: str,
    level: str,
    condition: str,
    seed: int,
    shard_count: int,
    output_root: Path,
    parameter_selection_dir: Path,
) -> list[ProtocolResult]:
    results = []
    for fold in registered_folds(level):
        parameter_selection = parameter_selection_path(
            parameter_selection_dir,
            method_id,
            fold,
        )
        parameters, parameters_sha256 = resolve_parameters(
            method_id,
            candidate_id=None,
            selection_path=parameter_selection,
            expected_training_scope_id=training_scope_id(fold),
        )
        if condition not in fold.conditions:
            raise ValueError(f"{condition} is not registered for {fold.fold_id}.")
        prepared = prepare_fold_evaluation(
            frame,
            fold,
            condition,
            "test",
            resource,
            seed=seed,
        )
        training = build_training_contract(
            method_id,
            fold,
            prepared.training_manifest,
            parameters_sha256,
            seed=seed,
        )
        threshold = load_threshold_transfer_artifact(
            threshold_path_for(
                output_root,
                method_id,
                fold,
                parameters_sha256,
                seed,
            ),
            fold=fold,
            method_id=method_id,
            scientific_parameters_sha256=parameters_sha256,
            training_contract_sha256=training.contract_sha256,
        )
        contract = build_run_contract(
            prepared,
            method_id,
            parameters,
            shard_count=shard_count,
            formal=True,
            seed=seed,
        )
        results.append(
            ProtocolResult(
                unit_id=fold.fold_id,
                fold=fold,
                condition=condition,
                scored_grid=_load_all_shards(prepared, contract, output_root),
                threshold=threshold.threshold,
            )
        )
    return results


def build_primary_comparisons(
    endpoints: dict[tuple[str, str, str], list[ProtocolResult]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    adaptation = []
    for level, condition in ENDPOINT_CONDITIONS.items():
        value = matched_protocol_comparison(
            endpoints[("qwen_candidate_pair_qlora", level, condition)],
            endpoints[("frozen_qwen_candidate_pair", level, condition)],
            replicates=replicates,
            seed=seed,
        )
        adaptation.append(
            {
                "comparison_id": f"{level}-{condition}-qlora-minus-frozen",
                "level": level,
                "condition": condition,
                **value,
            }
        )

    description = []
    crossover = {}
    for method_id in METHOD_IDS:
        value = matched_protocol_comparison(
            endpoints[(method_id, "L3", "DD")],
            endpoints[(method_id, "L3", "NN")],
            replicates=replicates,
            seed=seed,
        )
        description.append(
            {
                "comparison_id": f"L3-{method_id}-DD-minus-NN",
                "method_id": method_id,
                **value,
            }
        )
        directional = []
        for challenger_condition, reference_condition, interpretation in (
            ("DN", "NN", "add definition to held-out aspect A"),
            ("ND", "NN", "add definition to held-out aspect B"),
            ("DD", "ND", "add definition to aspect A when B is described"),
            ("DD", "DN", "add definition to aspect B when A is described"),
        ):
            value = matched_protocol_comparison(
                endpoints[(method_id, "L3", challenger_condition)],
                endpoints[(method_id, "L3", reference_condition)],
                replicates=replicates,
                seed=seed,
            )
            directional.append(
                {
                    "comparison_id": (
                        f"L3-{method_id}-{challenger_condition}"
                        f"-minus-{reference_condition}"
                    ),
                    "method_id": method_id,
                    "interpretation": interpretation,
                    **value,
                }
            )
        crossover[method_id] = holm_adjust_comparison_family(directional)

    degradation = []
    for method_id in METHOD_IDS:
        for easier_level, harder_level in (("L1", "L2"), ("L2", "L3"), ("L3", "L4")):
            easier_condition = ENDPOINT_CONDITIONS[easier_level]
            harder_condition = ENDPOINT_CONDITIONS[harder_level]
            value = cross_protocol_degradation(
                endpoints[(method_id, harder_level, harder_condition)],
                endpoints[(method_id, easier_level, easier_condition)],
                replicates=replicates,
                seed=seed,
            )
            degradation.append(
                {
                    "comparison_id": (
                        f"{method_id}-{harder_level}-{harder_condition}"
                        f"-minus-{easier_level}-{easier_condition}"
                    ),
                    "method_id": method_id,
                    "harder_level": harder_level,
                    "easier_level": easier_level,
                    **value,
                }
            )
    return {
        "qlora_minus_frozen_family": holm_adjust_comparison_family(adaptation),
        "level3_dd_minus_nn_family": holm_adjust_comparison_family(description),
        "level3_directional_crossover_by_method": crossover,
        "cross_level_degradation": degradation,
    }


def build_level1_qlora_seed_sensitivity(
    seed_results: dict[int, list[ProtocolResult]],
    *,
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if set(seed_results) != {13, 23, 42}:
        raise ValueError("Level 1 QLoRA seed sensitivity requires seeds 13, 23 and 42.")
    per_seed = {}
    unit_frames = []
    for training_seed, results in sorted(seed_results.items()):
        predictions, units = aggregate_protocol_predictions(
            results,
            partition="overall",
        )
        statistics = build_cluster_sufficient_statistics(
            predictions,
            canonical_pair_classes(),
            observation_key_columns=("experiment_unit", "row_uid"),
        )
        per_seed[str(training_seed)] = cluster_bootstrap_interval(
            statistics,
            "pair_micro_f1",
            replicates=replicates,
            seed=bootstrap_seed,
        )
        unit_frames.append(units.assign(training_seed=training_seed))

    sensitivity = []
    for training_seed in (23, 42):
        value = matched_protocol_comparison(
            seed_results[training_seed],
            seed_results[13],
            replicates=replicates,
            seed=bootstrap_seed,
        )
        sensitivity.append(
            {
                "comparison_id": f"qlora-seed{training_seed}-minus-seed13",
                "training_seed": training_seed,
                **value,
            }
        )
    units = pd.concat(unit_frames, ignore_index=True)
    fold_summary = (
        units.groupby("experiment_unit", as_index=False)["pair_micro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"experiment_unit": "fold_id"})
    )
    return {
        "training_seeds": [13, 23, 42],
        "review_sampling_intervals_by_seed": per_seed,
        "paired_seed_sensitivity_vs_seed13": holm_adjust_comparison_family(
            sensitivity
        ),
        "per_fold_across_seed_summary": fold_summary.to_dict(orient="records"),
        "uncertainty_scope": (
            "Review-cluster intervals are conditional on each training seed. "
            "Three-seed spread is reported descriptively and is not treated as "
            "a precise population interval over random initialisations."
        ),
    }


def _write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite comparison report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the three preregistered primary taxonomy comparisons."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parameter-selection-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    resource = load_minimal_descriptions(require_approved=False)
    config = load_precloud_config()
    assert_formal_run_gates(resource, config)
    replicates = int(config["statistics"]["bootstrap_replicates"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    frame = load_official_fabsa_splits(args.data_dir, ("train", "test"))

    endpoints = {}
    source_selections = {}
    required_conditions = {
        "L1": ("D",),
        "L2": ("D",),
        "L3": ("NN", "DN", "ND", "DD"),
        "L4": ("D",),
    }
    for method_id in METHOD_IDS:
        method_selection_paths = set()
        for level, conditions in required_conditions.items():
            for condition in conditions:
                for fold in registered_folds(level):
                    method_selection_paths.add(
                        parameter_selection_path(
                            args.parameter_selection_dir,
                            method_id,
                            fold,
                        )
                    )
                endpoints[(method_id, level, condition)] = load_protocol_results(
                    frame,
                    resource,
                    method_id=method_id,
                    level=level,
                    condition=condition,
                    seed=args.seed,
                    shard_count=args.shard_count,
                    output_root=args.output_root,
                    parameter_selection_dir=args.parameter_selection_dir,
                )
        source_selections[method_id] = [
            {
                "path": str(selection),
                "sha256": canonical_sha256(
                    json.loads(selection.read_text(encoding="utf-8"))
                ),
            }
            for selection in sorted(method_selection_paths)
        ]
    qlora_seed_results = {
        13: endpoints[("qwen_candidate_pair_qlora", "L1", "D")]
    }
    for training_seed in (23, 42):
        qlora_seed_results[training_seed] = load_protocol_results(
            frame,
            resource,
            method_id="qwen_candidate_pair_qlora",
            level="L1",
            condition="D",
            seed=training_seed,
            shard_count=args.shard_count,
            output_root=args.output_root,
            parameter_selection_dir=args.parameter_selection_dir,
        )

    report = {
        "schema_version": "taxonomy_primary_comparisons_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "seed": args.seed,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "description_resource_sha256": resource["content_sha256"],
        "source_parameter_selections": source_selections,
        "families": build_primary_comparisons(
            endpoints,
            replicates=replicates,
            seed=bootstrap_seed,
        ),
        "level1_qlora_seed_sensitivity": build_level1_qlora_seed_sensitivity(
            qlora_seed_results,
            replicates=replicates,
            bootstrap_seed=bootstrap_seed,
        ),
    }
    _write_new(args.output, report)
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
