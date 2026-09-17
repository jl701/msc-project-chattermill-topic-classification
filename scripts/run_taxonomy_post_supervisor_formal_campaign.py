"""Run one dependency-closed worker from the three-GPU formal-v2 plan."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.experiments import taxonomy_post_supervisor_cloud as formal_v2  # noqa: E402


def _load_campaign():
    path = PROJECT_ROOT / "scripts" / "run_taxonomy_two_stage_formal_campaign.py"
    spec = importlib.util.spec_from_file_location("_taxonomy_formal_v1_campaign", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load campaign: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    campaign = _load_campaign()
    campaign.build_formal_jobs = formal_v2.build_formal_jobs
    campaign.build_worker_manifest = formal_v2.build_worker_manifest
    campaign.load_parallel_plan = formal_v2.load_parallel_plan
    campaign.trainable_worker_jobs = formal_v2.trainable_worker_jobs
    campaign.validate_formal_job_graph = formal_v2.validate_formal_job_graph
    campaign.validate_formal_job_subset = formal_v2.validate_formal_job_subset
    campaign.DEFAULT_PARALLEL_PLAN = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "taxonomy_two_stage_three_gpu_parallel_v2.json"
    )
    if "--output-root" not in sys.argv:
        sys.argv.extend(["--output-root", "/workspace/taxonomy_two_stage_formal_v2"])
    return int(campaign.main())


if __name__ == "__main__":
    raise SystemExit(main())
