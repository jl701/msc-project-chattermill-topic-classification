from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import (
    DEFAULT_HELDOUT_ASPECTS,
    build_heldout_aspect_split,
    build_heldout_org_split,
    choose_org_split_candidates,
    load_all_fabsa,
    split_manifest,
)


def write_manifest(manifest: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def build_org(args, frame) -> None:
    if args.validation_org:
        validation_orgs = args.validation_org
        test_orgs = args.test_org
    else:
        candidate = choose_org_split_candidates(frame, top_k=1)[0]
        validation_orgs = list(candidate.validation_orgs)
        test_orgs = list(candidate.test_orgs)

    splits = build_heldout_org_split(frame, validation_orgs, test_orgs)
    manifest = split_manifest(
        "heldout_organisation",
        splits,
        {
            "validation_orgs": validation_orgs,
            "test_orgs": test_orgs,
            "selection": "manual" if args.validation_org else "auto_min_score",
        },
    )
    write_manifest(manifest, args.output_dir / "heldout_organisation" / "manifest.json")


def build_aspect(args, frame) -> None:
    heldout_aspects = args.heldout_aspect or DEFAULT_HELDOUT_ASPECTS
    strategies = ["label_masked", "example_filtered"] if args.strategy == "both" else [args.strategy]

    for strategy in strategies:
        splits = build_heldout_aspect_split(
            frame,
            heldout_aspects,
            strategy=strategy,
            eval_label_scope=args.eval_label_scope,
        )
        manifest = split_manifest(
            "heldout_aspect",
            splits,
            {
                "heldout_aspects": heldout_aspects,
                "strategy": strategy,
                "eval_label_scope": args.eval_label_scope,
                "candidate_labels_at_inference": True,
            },
        )
        write_manifest(manifest, args.output_dir / "heldout_aspect" / strategy / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible FABSA generalisation split manifests.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "splits")
    parser.add_argument("--split-type", choices=["heldout-org", "heldout-aspect", "all"], default="all")
    parser.add_argument("--validation-org", type=int, action="append", default=[])
    parser.add_argument("--test-org", type=int, action="append", default=[])
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered", "both"], default="both")
    parser.add_argument("--eval-label-scope", choices=["heldout", "full"], default="heldout")
    args = parser.parse_args()

    if bool(args.validation_org) != bool(args.test_org):
        raise ValueError("Provide both --validation-org and --test-org, or neither.")

    frame = load_all_fabsa(args.data_dir)
    if args.split_type in {"heldout-org", "all"}:
        build_org(args, frame)
    if args.split_type in {"heldout-aspect", "all"}:
        build_aspect(args, frame)


if __name__ == "__main__":
    main()

