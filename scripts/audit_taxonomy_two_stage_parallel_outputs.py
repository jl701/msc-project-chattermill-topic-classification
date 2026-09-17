"""Audit complete isolated worker outputs and write a content-bound receipt."""

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

from msc_project.experiments.taxonomy_two_stage_parallel_merge import (  # noqa: E402
    audit_parallel_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of the three-GPU formal output union."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--parallel-plan",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "taxonomy_two_stage_three_gpu_parallel_v1.json"
        ),
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    value = audit_parallel_outputs(
        args.campaign_root.resolve(), args.parallel_plan.resolve()
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(f".{args.receipt.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, args.receipt)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
