"""Run the formal-v2 Frozen-Qwen few-shot scopes on their assigned worker."""

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
    path = PROJECT_ROOT / "scripts" / "run_taxonomy_two_stage_frozen_few_shot_campaign.py"
    spec = importlib.util.spec_from_file_location("_taxonomy_few_shot_v1_campaign", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load campaign: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    campaign = _load_campaign()
    campaign.PROTOCOL_ID = formal_v2.PROTOCOL_ID
    campaign.scope_folds = formal_v2.scope_folds
    campaign.build_worker_manifest = formal_v2.build_worker_manifest
    campaign.load_parallel_plan = formal_v2.load_parallel_plan
    campaign.worker_record = formal_v2.worker_record
    campaign.DEFAULT_PARALLEL_PLAN = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "taxonomy_two_stage_three_gpu_parallel_v2.json"
    )

    def scope_command(args, scope_id: str) -> list[str]:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_taxonomy_post_supervisor_frozen_few_shot_validation.py"),
            "--scope-id",
            scope_id,
            "--config",
            str(args.config.resolve()),
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
            "--resume",
        ]
        if args.local_files_only:
            command.append("--local-files-only")
        return command

    campaign._scope_command = scope_command
    if "--parallel-plan" not in sys.argv:
        sys.argv.extend(["--parallel-plan", str(campaign.DEFAULT_PARALLEL_PLAN)])
    if "--config" not in sys.argv:
        sys.argv.extend(
            [
                "--config",
                str(
                    PROJECT_ROOT
                    / "configs"
                    / "experiments"
                    / "taxonomy_two_stage_cloud_execution_safety_v2.json"
                ),
            ]
        )
    if "--output-root" not in sys.argv:
        sys.argv.extend(
            [
                "--output-root",
                "/workspace/taxonomy_two_stage_formal_v2/worker-distil-frozen",
            ]
        )
    return int(campaign.main())


if __name__ == "__main__":
    raise SystemExit(main())
