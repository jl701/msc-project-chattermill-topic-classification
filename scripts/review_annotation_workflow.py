"""Versioned blind audit/adjudication/feedback CLI; never overwrite live labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.annotation_workflow import compare_blind, export_feedback, prepare_blind, release_labels


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
    args = parser.parse_args()
    if args.command == "prepare-blind":
        result = prepare_blind(args.data, args.output, group=args.group, seed=args.seed, mode=args.mode, delay_hours=args.delay_hours)
    elif args.command == "compare":
        result = compare_blind(args.queue, args.output)
    elif args.command == "release":
        result = release_labels(args.data, args.comparison, args.decisions, args.output)
    else:
        result = export_feedback(args.data, args.release, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
