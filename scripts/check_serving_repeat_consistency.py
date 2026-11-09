"""Count repeated-delivery label-set disagreements in a completed soak run.

CPU only, without label access. Export aggregate counts and evidence hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.monitoring import prediction_pairs


def consistency_counts(rows):
    first, counts, different = {}, {}, set()
    disagreements = 0
    for row in sorted(rows, key=lambda r: r["attempt_index"]):
        if not row.get("success"):
            continue
        key = str(row.get("sample_id", row["request_id"]))
        pairs = prediction_pairs(row["response"])
        if key not in first:
            first[key] = pairs
        elif pairs != first[key]:
            different.add(key)
            disagreements += 1
        counts[key] = counts.get(key, 0) + 1
    return {
        "attempts": len(rows),
        "successful_attempts": sum(counts.values()),
        "failed_attempts": len(rows) - sum(counts.values()),
        "unique_attempted_reviews": len({str(r.get("sample_id", r["request_id"])) for r in rows}),
        "unique_successful_reviews": len(first),
        "reviews_with_repeated_successful_deliveries": sum(n > 1 for n in counts.values()),
        "repeated_successful_deliveries_after_first": sum(n - 1 for n in counts.values()),
        "reviews_with_differing_prediction_sets": len(different),
        "repeated_deliveries_differing_from_first": disagreements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    campaign = args.campaign.resolve()
    state, plan = read(campaign / "state.json"), read(campaign / "plan.json")
    if state["status"] != "completed":
        raise ValueError("The resilience campaign must be completed")
    targets = [s["name"] + "__" + r["name"] for s in plan["sessions"] for r in s["runs"]
               if s["server"]["variant"] == "hybrid" and r["name"] == "soak_60min"]
    if len(targets) != 1 or targets[0] not in state["completed"]:
        raise ValueError("Expected exactly one completed hybrid soak run")
    directory = campaign / "runs" / targets[0]
    summary = read(directory / "summary.json")
    journal = directory / "requests.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    if summary.get("interrupted") is not False or len(rows) != summary["attempts"]:
        raise ValueError("Incomplete request journal")
    registration = ROOT / "configs/experiments/serving_repeat_consistency_20261109.json"
    registered = read(registration)
    if datetime.fromisoformat(registered["registered_utc"]) >= datetime.fromisoformat(state["started_utc"]):
        raise ValueError("Consistency analysis registration must precede resilience execution")
    result = {
        "schema_version": "serving_repeat_consistency_v1",
        "campaign_status": "completed", "labels_read": False,
        "plan_sha256": hashlib.sha256((campaign / "plan.json").read_bytes()).hexdigest(),
        "registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
        "requests_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        **consistency_counts(rows),
        "interpretation": registered["boundaries"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Preserve the existing analysis result")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
