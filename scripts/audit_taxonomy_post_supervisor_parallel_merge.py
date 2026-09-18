"""Audit and seal the downloaded three-worker formal-v2 artifact union."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _formal_v2_module():
    return importlib.import_module(
        "msc_project.experiments.taxonomy_post_supervisor_cloud"
    )


def _configured_merge_module(formal_v2):
    merge = importlib.import_module(
        "msc_project.experiments.taxonomy_two_stage_parallel_merge"
    )
    merge.PROTOCOL_ID = formal_v2.PROTOCOL_ID
    merge.audit_worker_union = formal_v2.audit_worker_union
    merge.candidate_result_path = formal_v2.candidate_result_path
    merge.expected_result_assignments = formal_v2.expected_result_assignments
    merge.load_parallel_plan = formal_v2.load_parallel_plan
    merge.scope_folds = formal_v2.scope_folds
    merge.selection_path = formal_v2.selection_path
    merge.trainable_worker_jobs = formal_v2.trainable_worker_jobs
    merge.validate_worker_manifest = formal_v2.validate_worker_manifest
    merge.worker_record = formal_v2.worker_record
    return merge


def _finalise_receipt(
    receipt: dict[str, object],
    assignments: Mapping[tuple[str, str, str, str], tuple[str, str]],
    *,
    protocol_id: str,
    training_scope_count: int,
) -> dict[str, object]:
    """Replace historical formal-v1 counts with the exact formal-v2 graph."""

    method_counts = Counter(key[0] for key in assignments)
    if set(method_counts) != set(receipt["methods"]):  # type: ignore[arg-type]
        raise ValueError("Formal-v2 receipt method set is inconsistent.")
    per_method_counts = set(method_counts.values())
    if len(per_method_counts) != 1:
        raise ValueError("Formal-v2 methods do not have equal result counts.")
    value = dict(receipt)
    value["schema_version"] = "taxonomy_post_supervisor_parallel_merge_audit_v2"
    value["protocol_id"] = protocol_id
    value["training_scope_count_per_method"] = training_scope_count
    value["result_count_per_method"] = per_method_counts.pop()
    value["result_count"] = len(assignments)
    value.pop("merge_audit_sha256", None)
    value["merge_audit_sha256"] = canonical_sha256(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--parallel-plan",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_three_gpu_parallel_v2.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    formal_v2 = _formal_v2_module()
    merge = _configured_merge_module(formal_v2)
    receipt = merge.audit_parallel_outputs(args.campaign_root, args.parallel_plan)
    plan = formal_v2.load_parallel_plan(args.parallel_plan)
    receipt = _finalise_receipt(
        receipt,
        formal_v2.expected_result_assignments(plan),
        protocol_id=formal_v2.PROTOCOL_ID,
        training_scope_count=len(formal_v2.scope_folds()),
    )
    output = args.output or args.campaign_root / "parallel_merge_audit_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
