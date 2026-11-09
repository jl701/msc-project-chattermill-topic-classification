"""Register and sequentially validate the exploratory 50%-capacity operating point.

This follow-up is selected after inspecting the 70% one-hour soak. It preserves
that experiment and waits for its controller to stop all GPU work before running.
Raw evidence is private; the tracked registration is aggregate-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("outputs/serving_20261109"))
    args = parser.parse_args()
    root = args.study_root.resolve()
    source = root / "resilience_plan/plan.json"
    previous = read(source)
    session = deepcopy(previous["sessions"][0])
    session["name"] = "hybrid_operating_point_50pct"
    session["runs"] = [r for r in session["runs"] if r["name"] in ("warmup", "soak_60min")]
    soak = next(r for r in session["runs"] if r["name"] == "soak_60min")
    soak["rate"] = previous["reference_capacity_reviews_per_second"] * 0.5
    soak["capacity_fraction"] = 0.5
    now = datetime.now(timezone.utc).isoformat()
    plan = {k: deepcopy(previous[k]) for k in (
        "artifacts", "artifacts_sha256", "capacity_rule", "primary_plan_sha256",
        "primary_capacity_references", "reference_capacity_reviews_per_second",
        "server_configuration_rule", "development_inputs", "quality_scope")}
    plan.update(study_id="serving_operating_point_50pct_20261110", created_utc=now,
                scope="Exploratory long-run operational validation selected after the 70% soak",
                sessions=[session], planned_arrival_seconds_excluding_warmup_and_drain=3600,
                source_resilience_plan_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    directory = root / "operating_point_50pct_controller"
    directory.mkdir(parents=True, exist_ok=False)
    plan_path = root / "operating_point_50pct_plan/plan.json"
    if plan_path.exists():
        raise FileExistsError("Preserve the existing operating-point plan")
    write(plan_path, plan)
    registration = {
        "registered_utc": now,
        "selection": "After observing the original 70%-capacity one-hour soak: 2827/2827 successful, HTTP P95 2136.98 ms. Earlier 50% short runs motivated this separate validation.",
        "objective": "Measure whether 50% of the frozen primary hybrid C1 capacity sustains the illustrative 95% of all attempts within 2 seconds over a full hour.",
        "candidate_target": "At least 95% of all scheduled attempts succeed within 2 seconds from scheduled arrival. This is an illustrative engineering target, not a contractual business SLA.",
        "selection_rule": "One new one-hour run at 50%; no model/prompt/threshold or runtime tuning; retain and report both 70% and 50% outcomes regardless of target attainment.",
        "dataset": "Replay the same 100 historical validation-derived development reviews; seed 20261109. No label access or evaluation-based model selection during execution.",
        "pipeline": "Fixed hybrid: few-shot aspect presence, QLoRA sentiment, Qwen3-4B-Instruct-2507 NF4 double quantization with FP16 compute; prefix cache on.",
        "server": session["server"],
        "duration_seconds": 3600, "warmup_requests_excluded": 3,
        "reference_capacity_reviews_per_second": previous["reference_capacity_reviews_per_second"],
        "offered_reviews_per_second": soak["rate"], "capacity_fraction": 0.5,
        "client_max_inflight": soak["max_inflight"], "telemetry_interval_seconds": soak["metrics_interval"],
        "analysis": "Full-hour and 5-minute latency/queue/error/memory windows; all-attempt response budget; label-free repeated prediction-set consistency using the existing pre-registered checker.",
        "sequential_gate": "The original followup_controller must complete all campaigns and aggregates, after its GPU servers stop.",
        "private_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_resilience_plan_sha256": plan["source_resilience_plan_sha256"],
        "publication": "Safe aggregate statistics and hashes only; no texts, IDs, labels, per-review outputs, weights or workstation paths.",
    }
    registration_path = ROOT / "configs/experiments/serving_operating_point_20261110.json"
    if registration_path.exists():
        raise FileExistsError("Preserve the existing registration")
    write(registration_path, registration)
    state = {"status": "waiting_original_followups", "completed": [], "started_utc": now}
    def record(**updates):
        state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
        write(directory / "state.json", state)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    def child(name, script, *options):
        argv = [sys.executable, "-u", "-X", "utf8", "scripts/" + script, *map(str, options)]
        record(status="running", current=name)
        with (directory / "commands.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"name": name, "argv": argv, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        print("START " + name, flush=True)
        with (directory / (name + ".log")).open("w", encoding="utf-8") as handle:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}; inspect retained log")
        state["completed"].append(name)
        record()
        print("DONE " + name, flush=True)
    try:
        record()
        print("REGISTERED; waiting for original followup completion", flush=True)
        while True:
            try:
                prior = read(root / "followup_controller/state.json")
            except json.JSONDecodeError:
                time.sleep(0.1)
                continue
            if prior["status"] == "completed":
                break
            if prior["status"] == "failed":
                raise RuntimeError("Original followup failed; no extra GPU work started")
            time.sleep(10)
        campaign = root / "operating_point_50pct"
        child("operating_point_50pct", "run_serving_study.py", "--plan", plan_path, "--output", campaign)
        child("aggregate", "summarise_serving_campaign.py", "--campaign", campaign,
              "--labels", root / "labels_evaluation.jsonl", "--artifacts", root / "artifacts.json",
              "--output-dir", root / "aggregate_operating_point_50pct")
        child("window_analysis", "analyse_serving_resilience.py", "--campaign", campaign,
              "--primary-campaign", root / "primary", "--development-labels", root / "labels_development.jsonl",
              "--artifacts", root / "artifacts.json", "--output-dir", root / "operating_point_50pct_analysis")
        child("repeat_consistency", "check_serving_repeat_consistency.py", "--campaign", campaign,
              "--output", root / "operating_point_50pct_analysis/repeat_consistency.json")
        record(status="completed")
    except Exception as error:
        record(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
