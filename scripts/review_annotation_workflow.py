"""Versioned blind audit/adjudication/feedback CLI; never overwrite live labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.annotation_workflow import compare_blind, export_feedback, prepare_blind, release_labels
from msc_project.serving.adaptation_use import (
    PURPOSES, add_policy_arguments, export_adaptation_use, policy_from_args,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-blind")
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--group", choices=("pilot", "adaptation", "calibration", "locked", "all"), default="all")
    prepare.add_argument("--mode", choices=("independent", "delayed_same_reviewer"), default="independent")
    prepare.add_argument("--delay-hours", type=float, default=24)
    prepare.add_argument("--seed", type=int, default=20261111)
    compare = commands.add_parser("compare")
    compare.add_argument("--queue", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    release = commands.add_parser("release")
    release.add_argument("--data", type=Path, required=True)
    release.add_argument("--comparison", type=Path, required=True)
    release.add_argument("--decisions", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    feedback = commands.add_parser("export-feedback")
    feedback.add_argument("--data", type=Path, required=True)
    feedback.add_argument("--release", type=Path, required=True)
    feedback.add_argument("--output", type=Path, required=True)
    add_policy_arguments(feedback)
    use = commands.add_parser("export-adaptation-use", help="Private eligible error-analysis/model-improvement view")
    for name in ("data", "release", "output"):
        use.add_argument("--" + name, type=Path, required=True)
    use.add_argument("--purpose", choices=PURPOSES, required=True)
    use.add_argument("--predictions", type=Path)
    add_policy_arguments(use)
    args = parser.parse_args()
    if args.command == "prepare-blind":
        result = prepare_blind(args.data, args.output, group=args.group, seed=args.seed, mode=args.mode, delay_hours=args.delay_hours)
    elif args.command == "compare":
        result = compare_blind(args.queue, args.output)
    elif args.command == "release":
        result = release_labels(args.data, args.comparison, args.decisions, args.output)
    elif args.command == "export-feedback":
        result = export_feedback(args.data, args.release, args.output, policy=policy_from_args(args))
    else:
        result = export_adaptation_use(args.data, args.release, args.output,
            purpose=args.purpose, policy=policy_from_args(args), predictions=args.predictions)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
