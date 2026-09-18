from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _script_module():
    path = PROJECT_ROOT / "scripts/audit_taxonomy_post_supervisor_parallel_merge.py"
    spec = importlib.util.spec_from_file_location("formal_v2_merge_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_v2_receipt_uses_exact_reduced_result_graph() -> None:
    module = _script_module()
    methods = [
        "distilbert_review_candidate_cross_encoder",
        "qwen_candidate_pair_qlora",
        "frozen_qwen_few_shot",
    ]
    assignments = {
        (method, "L2", f"fold-{index:02d}", "D"): ("worker", "scope")
        for method in methods
        for index in range(27)
    }
    receipt = {
        "schema_version": "historical",
        "protocol_id": "taxonomy_two_stage_formal_v2",
        "methods": methods,
        "result_count_per_method": 99,
        "result_count": 297,
        "merge_audit_sha256": "stale",
    }

    final = module._finalise_receipt(
        receipt,
        assignments,
        protocol_id="taxonomy_two_stage_formal_v2",
        training_scope_count=15,
    )

    assert final["training_scope_count_per_method"] == 15
    assert final["result_count_per_method"] == 27
    assert final["result_count"] == 81
    assert final["merge_audit_sha256"] == module.canonical_sha256(
        {key: value for key, value in final.items() if key != "merge_audit_sha256"}
    )
