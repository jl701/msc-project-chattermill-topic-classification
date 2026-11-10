"""Prepare disjoint text-only capacity and benchmark pools from public reserves."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.retrieval import exclude_duplicates


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--quality-replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing frozen pools; use a new output directory")
    reserves = read(args.reserve)
    if any("split" in row or "labels" in row for row in reserves):
        raise ValueError("Use unassigned reserve inputs, never quality split annotations")
    if len({r["request_id"] for r in reserves}) != len(reserves):
        raise ValueError("Duplicate reserve IDs")
    rows = [dict(row, row_uid=row["request_id"]) for row in reserves]
    references = list({row["review_text"] for row in read(args.index / "pool.jsonl")})
    quality_ids = set()
    for path in args.quality_replay:
        reviews = read(path)
        if any(set(row) != {"request_id", "text"} for row in reviews):
            raise ValueError("Quality exclusions must be label-free replay files")
        references.extend(row["text"] for row in reviews)
        quality_ids.update(row["request_id"] for row in reviews)
    if quality_ids & {r["request_id"] for r in reserves}:
        raise ValueError("Reserve and quality IDs overlap")
    accepted, exclusions = exclude_duplicates(rows, references)
    capacity, benchmark = [], []
    for source in sorted({row["app"] for row in accepted}):
        group = sorted([row for row in accepted if row["app"] == source], key=lambda row:
            hashlib.sha256(("rag_performance_20261111|" + row["request_id"]).encode()).hexdigest())
        capacity.extend(group[::2])
        benchmark.extend(group[1::2])
    if min(len(capacity), len(benchmark)) < 30:
        raise ValueError("At least30 unique reviews required in each performance pool")
    # Cross-pool near matches are forbidden even if a prior collection omitted grouping.
    kept, cross = exclude_duplicates(benchmark, [row["text"] for row in capacity])
    if len(kept) != len(benchmark):
        raise ValueError(f"Cross-pool duplicates require regrouping before freeze: {cross}")
    args.output.mkdir(parents=True)
    manifest = {"source_reserve_count": len(reserves), "eligible_count": len(accepted),
                "exclusion_counts": exclusions, "selection_uses_labels": False,
                "split_rule": "per app SHA256 stable ordering, alternating capacity/benchmark",
                "capacity_used_for": "3x30 serialrequests for commonload calibration only",
                "benchmark_used_for": "closed/open/soak/burst comparisons",
                "sha256_inputs": {"reserve":sha(args.reserve),
                    "index_pool":sha(args.index / "pool.jsonl"),
                    "quality_replays": [sha(path) for path in args.quality_replay]}, "pools": {}}
    for name, population in [("capacity", capacity), ("benchmark", benchmark)]:
        population.sort(key=lambda row: row["request_id"])
        path = args.output / f"replay_{name}.jsonl"
        path.write_text("".join(json.dumps({"request_id":row["request_id"], "text":row["text"]},
             ensure_ascii=False) + "\n" for row in population), encoding="utf-8")
        manifest["pools"][name] = {"count":len(population), "sources":dict(Counter(
            row["app"] for row in population)), "sha256":sha(path)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
