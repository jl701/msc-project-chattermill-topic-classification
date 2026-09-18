"""Bind the proven trainable executor to the frozen post-supervisor v2 graph."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments import taxonomy_post_supervisor_cloud as formal_v2  # noqa: E402


def _load_legacy_executor():
    path = PROJECT_ROOT / "scripts" / "run_taxonomy_two_stage_trainable_validation.py"
    spec = importlib.util.spec_from_file_location("_taxonomy_trainable_v1_executor", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load executor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    executor = _load_legacy_executor()
    executor.PROTOCOL_ID = formal_v2.PROTOCOL_ID
    executor.build_formal_jobs = formal_v2.build_formal_jobs
    executor.candidate_result_path = formal_v2.candidate_result_path
    executor.plan_payload = formal_v2.plan_payload
    executor.representative_fold = formal_v2.representative_fold
    executor.formal_conditions = formal_v2.formal_conditions
    executor.scope_folds = formal_v2.scope_folds
    executor.selection_path = formal_v2.selection_path
    executor.variant_map = formal_v2.variant_map
    executor.DEFAULT_SAFETY_CONFIG = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "taxonomy_two_stage_cloud_execution_safety_v2.json"
    )
    if "--output-root" not in sys.argv:
        sys.argv.extend(
            [
                "--output-root",
                str(PROJECT_ROOT / "outputs/experimental/taxonomy_two_stage_formal_v2"),
            ]
        )
    return int(executor.main())


if __name__ == "__main__":
    raise SystemExit(main())
