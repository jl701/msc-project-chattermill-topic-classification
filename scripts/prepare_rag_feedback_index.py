"""Create an immutable reviewed-adaptation index, then optionally encode a new version."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msc_project.serving.feedback_index import append_feedback_index, encode_immutable_index
from msc_project.serving.adaptation_use import add_policy_arguments, policy_from_args


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    append = commands.add_parser("append", help="Validate reviewed adaptation feedback and build a new unencoded index")
    for name in ("parent", "data", "release", "feedback", "artifacts", "output"):
        append.add_argument("--"+name, type=Path, required=True)
    append.add_argument("--parent-exclusion-replay", type=Path, action="append", default=[])
    append.add_argument("--extra-protected-replay", type=Path, action="append", default=[])
    add_policy_arguments(append)
    encode = commands.add_parser("encode", help="Encode a validated pool into another immutable index version")
    for name in ("source", "output", "encoder-path"):
        encode.add_argument("--"+name, type=Path, required=True)
    encode.add_argument("--encoder-device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if args.command == "append":
        result = append_feedback_index(parent=args.parent, data=args.data, release=args.release,
            feedback=args.feedback, artifacts=args.artifacts, output=args.output,
            policy=policy_from_args(args),
            parent_replays=args.parent_exclusion_replay, extra_protected_replays=args.extra_protected_replay)
        report = {k: result[k] for k in ("pool_pairs", "legal_reviews", "reviews_by_origin", "encoded",
                                       "adaptation_evaluation_certified_disjoint", "parent_index_modified")}
    else:
        result = encode_immutable_index(source=args.source, output=args.output,
                                       encoder_path=args.encoder_path, device=args.encoder_device)
        report = {k: result[k] for k in ("pool_pairs", "pool_sha256", "encoded", "index_encoding_device")}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
