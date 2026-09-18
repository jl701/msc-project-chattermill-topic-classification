from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(*arguments: str) -> str:
    result = subprocess.run(
        [PYTHON, *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def test_trainable_v2_plan_entrypoint() -> None:
    output = _run(
        "scripts/run_taxonomy_post_supervisor_trainable_validation.py",
        "--phase",
        "plan",
        "--output-root",
        "outputs/experimental/_post_supervisor_wrapper_test",
    )
    assert '"protocol_id": "taxonomy_two_stage_formal_v2"' in output
    assert '"score-selected": 30' in output
    assert '"test_contract_count": 0' in output


def test_three_trainable_worker_dry_runs_cover_150_jobs() -> None:
    counts = []
    for worker in ("worker-qlora-a", "worker-qlora-b", "worker-distil-frozen"):
        output = _run(
            "scripts/run_taxonomy_post_supervisor_formal_campaign.py",
            "--worker-id",
            worker,
            "--hf-home",
            ".",
            "--dry-run",
        )
        import json

        counts.append(json.loads(output)["planned_count"])
    assert counts == [40, 35, 75]
    assert sum(counts) == 150


def test_frozen_v2_campaign_dry_run_has_15_scopes() -> None:
    import json

    output = _run(
        "scripts/run_taxonomy_post_supervisor_frozen_few_shot_campaign.py",
        "--hf-home",
        ".",
        "--dry-run",
    )
    payload = json.loads(output)
    assert payload["protocol_id"] == "taxonomy_two_stage_formal_v2"
    assert payload["planned_count"] == 15
    assert payload["test_contract_count"] == 0
