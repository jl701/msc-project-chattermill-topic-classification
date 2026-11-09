"""Freeze a resilience campaign from a completed primary hybrid serving study.

This is CPU-only planning. It reads aggregate capacity and label-free development
input metadata, writes immutable local plans, and never starts a service or load.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CAPACITY_RUNS = ["closed_c1_r1", "closed_c1_r2", "closed_c1_r3"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(root, path):
    path = Path(path)
    result = path.resolve() if path.is_absolute() else (root / path).resolve()
    # Campaign controller executes at the repository root. Keep every generated
    # reference relative to that root and reject accidental unrelated outputs.
    result.relative_to(root)
    return result


def create_plan(primary, output, development_replay, development_metadata, session_name=None, *, root=ROOT):
    root = Path(root).resolve()
    primary, output = resolve(root, primary), resolve(root, output)
    replay_path, metadata_path = resolve(root, development_replay), resolve(root, development_metadata)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing plan directory")
    state = read_json(primary / "state.json")
    if state.get("status") != "completed":
        raise ValueError("Primary campaign must be completed before freezing resilience rates")
    original = read_json(primary / "plan.json")
    candidates = [s for s in original["sessions"] if s["server"].get("variant") == "hybrid"]
    if session_name is not None:
        candidates = [s for s in candidates if s["name"] == session_name]
    if len(candidates) != 1:
        raise ValueError("Select exactly one primary hybrid session with --session")
    session = candidates[0]
    server = deepcopy(session["server"])
    # Fix method identity before reading quality: no labels or F1 are consulted.
    if server["variant"] != "hybrid":
        raise ValueError("Resilience target must be the fixed two-stage hybrid")
    registered = {r["name"]: r for r in session["runs"]}
    references = []
    for name in CAPACITY_RUNS:
        if name not in registered:
            raise ValueError("Missing registered capacity run " + name)
        config = registered[name]
        if config.get("mode") != "closed" or config.get("concurrency") != 1:
            raise ValueError("Capacity reference must be closed concurrency 1: " + name)
        path = primary / "runs" / (session["name"] + "__" + name) / "summary.json"
        summary = read_json(path)
        capacity = summary.get("successful_reviews_per_second")
        if (summary.get("interrupted") or not summary.get("attempts") or
            summary.get("success_rate", 0) < 0.99 or not isinstance(capacity, (int, float)) or
            not math.isfinite(capacity) or capacity <= 0):
            raise ValueError("Capacity reference is incomplete, failed, or nonpositive: " + name)
        references.append({"run": session["name"] + "__" + name,
                           "reviews_per_second": capacity, "success_rate": summary["success_rate"],
                           "summary_sha256": sha(path)})
    capacity = statistics.median(r["reviews_per_second"] for r in references)
    replay, metadata = read_jsonl(replay_path), read_jsonl(metadata_path)
    if not replay:
        raise ValueError("Development replay is empty")
    if any(set(row) != {"request_id", "text"} or not isinstance(row["text"], str) for row in replay):
        raise ValueError("Development replay must contain only request_id and text, never labels")
    lookup = {row["request_id"]: row for row in replay}
    meta = {row["request_id"]: row for row in metadata}
    if len(lookup) != len(replay) or len(meta) != len(metadata) or lookup.keys() != meta.keys():
        raise ValueError("Replay and metadata identities must be unique and exactly matched")
    for sample, row in meta.items():
        if set(row) & {"labels", "pairs", "aspects", "sentiment", "sentiments", "gold"}:
            raise ValueError("Selection metadata must be label-free")
        if row.get("character_count") != len(lookup[sample]["text"]):
            raise ValueError("Recorded character length differs from the replay text")
    long_count = max(1, math.ceil(len(replay) * 0.10))
    ranked = sorted(meta, key=lambda key: (-meta[key]["character_count"], key))
    long_keys = ranked[:long_count]
    long_replay = [lookup[key] for key in long_keys]
    long_metadata = [meta[key] for key in long_keys]
    rel = lambda path: Path(path).relative_to(root).as_posix()
    normal_replay, normal_metadata = rel(replay_path), rel(metadata_path)
    long_replay_ref = rel(output / "replay_long_development.jsonl")
    long_metadata_ref = rel(output / "metadata_long_development.jsonl")
    phase_plans = {}
    normal_rate, burst_rate = 0.5 * capacity, 2.0 * capacity
    for repetition in range(1, 4):
        name = f"burst_recovery_r{repetition}"
        phase_plans[name] = {"scope": "Continuous independent arrivals, no intermediate drain",
            "phases": [
                {"name": "normal", "duration": 60, "rate": normal_rate},
                {"name": "burst", "duration": 30, "rate": burst_rate},
                {"name": "recovery", "duration": 180, "rate": normal_rate},
            ]}
    phase_plans["input_length_shift"] = {
        "scope": "Input-length intervention only. Selection uses development character length, never labels.",
        "phases": [
            {"name": "normal_before", "duration": 120, "rate": normal_rate},
            {"name": "long_input", "duration": 120, "rate": normal_rate,
             "replay": long_replay_ref, "metadata": long_metadata_ref},
            {"name": "normal_after", "duration": 120, "rate": normal_rate},
        ]}
    defaults = {"replay": normal_replay, "metadata": normal_metadata, "seed": 20261109,
                "metrics_interval": 5}
    runs = [
        {**defaults, "name": "warmup", "mode": "closed", "requests": 3, "concurrency": 1,
         "quality_evaluation": False, "purpose": "warmup_excluded_from_headline"},
        {**defaults, "name": "soak_60min", "mode": "open", "duration": 3600,
         "rate": 0.7 * capacity, "max_inflight": 64, "quality_evaluation": False,
         "purpose": "60_minute_continuous_service_stability", "capacity_fraction": 0.7},
    ]
    for name in phase_plans:
        runs.append({**defaults, "name": name, "mode": "open", "max_inflight": 64,
                     "phase_plan": rel(output / (name + ".json")), "quality_evaluation": False,
                     "purpose": "input_length_shift_diagnostic" if name == "input_length_shift" else "burst_and_recovery_diagnostic"})
    artifacts = resolve(root, original["artifacts"])
    plan = {
        "study_id": "serving_resilience_20261109",
        "scope": "Local single-GPU genuine-inference service resilience; fixed hybrid, not selected by evaluation F1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": rel(artifacts),
        "artifacts_sha256": sha(artifacts),
        "capacity_rule": "Median successful complete reviews/second of the same primary hybrid session closed_c1_r1/r2/r3; all success rates at least 99%",
        "primary_plan_sha256": sha(primary / "plan.json"),
        "primary_capacity_references": references,
        "reference_capacity_reviews_per_second": capacity,
        "server_configuration_rule": "Exact copy of the primary hybrid session including prefix-cache, batching, queue and timeout configuration",
        "development_inputs": {"replay_sha256": sha(replay_path), "metadata_sha256": sha(metadata_path),
                               "unique_reviews": len(replay), "labels_read": False},
        "long_input_selection": {"rule": "Top ceil(10%) of development reviews by character_count descending; ties by stable request_id",
                                 "count": long_count, "minimum_selected_characters": min(meta[k]["character_count"] for k in long_keys),
                                 "maximum_selected_characters": max(meta[k]["character_count"] for k in long_keys),
                                 "labels_read": False, "evaluation_reviews_used": False},
        "planned_arrival_seconds_excluding_warmup_and_drain": 3600 + 3 * 270 + 360,
        "phase_drain_rule": "No drain at phase transitions; controller drains once after each complete run",
        "quality_scope": "No primary quality evaluation in this stage. Any later label join is diagnostic only and reports unique development reviews separately from repeated HTTP requests.",
        "sessions": [{"name": "hybrid_resilience", "server": server, "runs": runs}],
    }
    # All validation occurs before creating the directory. Existing evidence is
    # never overwritten; reruns require a new output directory.
    output.mkdir(parents=True, exist_ok=False)
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, value in phase_plans.items():
        write(output / (name + ".json"), value)
    for name, rows in [("replay_long_development.jsonl", long_replay), ("metadata_long_development.jsonl", long_metadata)]:
        (output / name).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    plan["long_input_selection"].update(replay_sha256=sha(output / "replay_long_development.jsonl"),
                                         metadata_sha256=sha(output / "metadata_long_development.jsonl"))
    plan["phase_plan_sha256"] = {name: sha(output / (name + ".json")) for name in phase_plans}
    write(output / "plan.json", plan)
    write(output / "generation_manifest.json", {
        "generator_sha256": sha(Path(__file__)), "plan_sha256": sha(output / "plan.json"),
        "files": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        "status": "frozen_plan_only_not_executed", "raw_inputs_local_only": True,
    })
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-replay", type=Path, default=Path("outputs/serving_20261109/replay_development.jsonl"))
    parser.add_argument("--development-metadata", type=Path, default=Path("outputs/serving_20261109/metadata_development.jsonl"))
    parser.add_argument("--session", help="Choose a primary hybrid session if more than one exists")
    args = parser.parse_args()
    plan = create_plan(args.primary_campaign, args.output_dir, args.development_replay,
                       args.development_metadata, args.session)
    print(json.dumps({"status": "plan_only_not_executed", "capacity": plan["reference_capacity_reviews_per_second"],
                      "arrival_minutes": plan["planned_arrival_seconds_excluding_warmup_and_drain"] / 60,
                      "long_development_reviews": plan["long_input_selection"]["count"]}, indent=2))


if __name__ == "__main__":
    main()
