"""Materialise and audit the reduced formal-v2 graph without launching a GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments.taxonomy_post_supervisor_cloud import (  # noqa: E402
    audit_worker_union,
    build_formal_jobs,
    build_worker_manifest,
    expected_result_assignments,
    load_parallel_plan,
    plan_payload,
    runtime_cost_forecast,
    validate_safety_config,
    workload_contract_counts,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallel-plan",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_three_gpu_parallel_v2.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/experimental/taxonomy_post_supervisor_local_v1/cloud_preflight",
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_cloud_execution_safety_v2.json",
    )
    parser.add_argument("--cloud-output-root", default="/workspace/taxonomy_two_stage_formal_v2")
    parser.add_argument("--cloud-data-dir", default="/workspace/data/fabsa")
    parser.add_argument(
        "--hardware-gate-result",
        type=Path,
        default=PROJECT_ROOT
        / "docs/experiments/taxonomy_two_stage_4090_hardware_gate_v1_result.json",
    )
    args = parser.parse_args()
    plan = load_parallel_plan(args.parallel_plan)
    safety = json.loads(args.safety_config.read_text(encoding="utf-8"))
    safety_audit = validate_safety_config(safety)
    hardware = json.loads(args.hardware_gate_result.read_text(encoding="utf-8"))
    rates = hardware["representative_benchmark"]
    price = float(hardware["qualified_host"]["observed_total_price_usd_per_hour"])
    forecast = runtime_cost_forecast(
        workload_contract_counts(),
        unique_validation_texts=1042,
        frozen_prompts_per_second=float(rates["frozen_qwen_prompts_per_second_batch_6"]),
        qlora_prompts_per_second=float(rates["qlora_prompts_per_second_batch_8"]),
        qlora_training_examples_per_second=float(rates["qlora_training_examples_per_second"]),
        distilbert_prompts_per_second=float(rates["distilbert_scoring_prompts_per_second"]),
        distilbert_training_examples_per_second=float(rates["distilbert_training_examples_per_second"]),
        price_per_hour_usd=price,
    )
    worker_forecast: dict[str, object] = {}
    for worker in plan["workers"]:
        worker_counts = workload_contract_counts(worker["training_scope_ids"])
        worker_runtime = runtime_cost_forecast(
            worker_counts,
            unique_validation_texts=1042,
            frozen_prompts_per_second=float(rates["frozen_qwen_prompts_per_second_batch_6"]),
            qlora_prompts_per_second=float(rates["qlora_prompts_per_second_batch_8"]),
            qlora_training_examples_per_second=float(rates["qlora_training_examples_per_second"]),
            distilbert_prompts_per_second=float(rates["distilbert_scoring_prompts_per_second"]),
            distilbert_training_examples_per_second=float(rates["distilbert_training_examples_per_second"]),
            price_per_hour_usd=price,
        )
        method_key = (
            "qlora"
            if worker["method_id"] == "qwen_candidate_pair_qlora"
            else "distilbert"
        )
        hours = float(
            worker_runtime["methods"][method_key]["planned_hours_with_25pct_margin"]
        )
        components = {method_key: hours}
        if worker.get("frozen_qwen_few_shot_scope_ids"):
            frozen_hours = float(
                worker_runtime["methods"]["frozen_few_shot"][
                    "planned_hours_with_25pct_margin"
                ]
            )
            components["frozen_few_shot"] = frozen_hours
            hours += frozen_hours
        worker_forecast[str(worker["worker_id"])] = {
            "counts": worker_counts,
            "components_planned_hours": components,
            "sequential_planned_hours": hours,
        }
    forecast["three_worker_schedule"] = worker_forecast
    forecast["planned_parallel_wall_hours"] = max(
        float(value["sequential_planned_hours"])
        for value in worker_forecast.values()
    )
    formal = plan_payload(
        build_formal_jobs(
            output_root=args.cloud_output_root,
            data_dir=args.cloud_data_dir,
        )
    )
    manifests = [
        build_worker_manifest(
            plan,
            str(worker["worker_id"]),
            output_root=f"{args.cloud_output_root}/{worker['worker_id']}",
            data_dir=args.cloud_data_dir,
        )
        for worker in plan["workers"]
    ]
    audit = audit_worker_union(manifests, plan)
    audit["safety_config_sha256"] = safety_audit["safety_config_sha256"]
    assignments = expected_result_assignments(plan)
    assignment_payload = {
        "schema_version": "taxonomy_post_supervisor_result_assignments_v2",
        "protocol_id": formal["protocol_id"],
        "assignment_count": len(assignments),
        "assignments": [
            {
                "method_id": key[0],
                "level": key[1],
                "fold_id": key[2],
                "condition": key[3],
                "worker_id": value[0],
                "training_scope_id": value[1],
            }
            for key, value in sorted(assignments.items())
        ],
        "failure_count": 0,
        "test_contract_count": 0,
    }
    _write(args.output_root / "formal_job_plan.json", formal)
    for manifest in manifests:
        _write(
            args.output_root / "worker_manifests" / f"{manifest['worker_id']}.json",
            manifest,
        )
    _write(args.output_root / "result_assignments.json", assignment_payload)
    _write(args.output_root / "runtime_cost_forecast.json", forecast)
    _write(args.output_root / "cloud_plan_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
