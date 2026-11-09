"""Join delayed labels and summarise local serving load-test runs.

Outputs contain aggregates plus optional sample IDs for local review queues.
They must be reviewed before publication. This script does not tune models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msc_project.serving.monitoring import (
    evaluate_quality, make_review_queues, monitor_windows, summarise_load, unlabelled_summary,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--labels", type=Path, action="append", default=[])
    p.add_argument("--artifacts", type=Path, help="Frozen private manifest with taxonomy")
    p.add_argument("--reference-run", type=Path)
    p.add_argument("--window-seconds", type=float, default=60)
    p.add_argument("--length-js-threshold", type=float, default=0.1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.labels and not args.artifacts:
        p.error("--artifacts is required with --labels to define the held-out and seen scopes")
    labels = [r for path in args.labels for r in read_jsonl(path)]
    taxonomy = json.loads(args.artifacts.read_text(encoding="utf-8-sig"))["taxonomy"] if args.artifacts else {}
    heldout = taxonomy.get("heldout_aspects", [])
    universe = taxonomy.get("all_aspects", taxonomy.get("candidate_aspects", []))
    if not universe:
        universe = list(taxonomy.get("seen_aspects", [])) + list(heldout)
    reference = read_jsonl(args.reference_run / "requests.jsonl") if args.reference_run else None
    outputs = []
    for run in args.run:
        records = read_jsonl(run / "requests.jsonl")
        existing_path = run / "summary.json"
        existing = json.loads(existing_path.read_text(encoding="utf-8-sig")) if existing_path.exists() else {}
        result = {"run": str(run), "performance": summarise_load(records, existing.get("wall_seconds_including_drain")),
                  "unlabelled": unlabelled_summary(records), "review_queues": make_review_queues(records)}
        if labels:
            result["delayed_label_quality"] = evaluate_quality(records, labels, heldout, universe)
        if reference is not None:
            result["monitoring"] = monitor_windows(records, reference, args.window_seconds, args.length_js_threshold)
        outputs.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": outputs,
        "scope": "Local single-version API replay. Quality counts unique samples, not HTTP repeats. Unlabelled signals do not prove F1 decline.",
        "labels_available": bool(labels)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analysed {len(outputs)} runs; independent labels supplied: {bool(labels)}")


if __name__ == "__main__":
    main()
