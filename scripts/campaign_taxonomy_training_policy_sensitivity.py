"""Bounded local worker with GPU guard and verified per-job backup."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID, sha256_file, verify_receipt, write_json, canonical_sha256,
)


def snapshot():
    fields = ["utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw", "pstate",
              "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.hw_thermal_slowdown",
              "clocks_event_reasons.hw_slowdown", "clocks_event_reasons.hw_power_brake_slowdown"]
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    values = result.stdout.strip().split(",")
    if len(values) != len(fields):
        raise RuntimeError("GPU telemetry schema mismatch")
    return dict(zip(fields, (v.strip() for v in values)))


def backup_job(source: Path, destination: Path):
    payload = verify_receipt(source)
    destination.mkdir(parents=True, exist_ok=True)
    for relative, expected in payload["files"].items():
        src, dst = source / relative, destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha256_file(dst) != expected["sha256"]:
                raise RuntimeError("Backup content conflict: " + str(dst))
        else:
            temporary = dst.with_name(dst.name + ".receiving")
            if temporary.exists():
                raise RuntimeError("Incomplete backup file requires audit")
            shutil.copy2(src, temporary)
            if sha256_file(temporary) != expected["sha256"]:
                raise RuntimeError("Backup SHA256 mismatch")
            os.replace(temporary, dst)
    shutil.copy2(source / "receipt.json", destination / "receipt.json")
    verify_receipt(destination, payload["contract"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads((ROOT / f"configs/experiments/{STUDY_ID}.json").read_text())
    output = ROOT / config["output_root"]
    backup = ROOT.parent / "cloud_backups" / STUDY_ID
    campaign = output / "campaign" / args.worker
    campaign.mkdir(parents=True, exist_ok=True)
    lock = campaign / "worker.lock"
    # An existing lock is never silently discarded or used to resume a worker.
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    jobs = []
    for fold in config["folds"]:
        for policy in config["policies"]:
            jobs.append(("tfidf" if args.worker == "cpu" else "distilbert", policy, fold, None))
        if args.worker == "cpu":
            jobs.extend(("tfidf", "size_matched_label_masked", fold, seed)
                        for seed in config["tfidf"]["subset_seeds"])
    if args.limit is not None:
        jobs = jobs[:args.limit]
    state = {"worker_pid": os.getpid(), "worker": args.worker, "planned": len(jobs), "completed": 0,
             "failure_count": 0, "test_contract_count": 0, "status": "running", "started_unix": time.time(),
             "config_sha256": sha256_file(ROOT / f"configs/experiments/{STUDY_ID}.json")}
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    child = None
    try:
        for method, policy, fold, seed in jobs:
            policy_name = policy + (f"_s{seed}" if seed is not None else "")
            job_id = f"{method}__{policy_name}__{fold}"
            source = output / method / policy_name / fold
            state.update(current_job=job_id, updated_unix=time.time(), child_pid=None)
            write_json(campaign / "state.json", state)
            if not (source / "receipt.json").exists():
                command = [sys.executable, "scripts/run_taxonomy_training_policy_sensitivity.py", "--phase", "run",
                           "--method", method, "--policy", policy, "--fold-id", fold]
                if seed is not None:
                    command += ["--subset-seed", str(seed)]
                if shutil.disk_usage(output).free / 1024**3 < 20:
                    raise RuntimeError("Insufficient disk headroom")
                with (campaign / f"{job_id}.log").open("x", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    state.update(child_pid=child.pid, command=command)
                    write_json(campaign / "state.json", state)
                    thermal_events = 0
                    telemetry_errors = 0
                    while child.poll() is None:
                        sample = {"at": time.time(), "job": job_id, "child_pid": child.pid}
                        if args.worker == "gpu":
                            try:
                                gpu = snapshot()
                                sample["gpu"] = gpu
                                telemetry_errors = 0
                                if float(gpu["temperature.gpu"]) >= config["safety"]["maximum_temperature_c"]:
                                    raise RuntimeError("Registered GPU temperature stop")
                                active = any(str(gpu[k]).lower() == "active" for k in gpu if "slowdown" in k)
                                thermal_events += int(active)
                                if thermal_events >= 2:
                                    raise RuntimeError("Repeated thermal/hardware slowdown")
                            except subprocess.SubprocessError:
                                telemetry_errors += 1
                                if telemetry_errors >= 2:
                                    raise RuntimeError("Repeated GPU telemetry failure")
                        if shutil.disk_usage(output).free / 1024**3 < 20:
                            raise RuntimeError("Disk headroom stop")
                        with (campaign / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(sample) + "\n")
                        state.update(updated_unix=time.time())
                        write_json(campaign / "state.json", state)
                        deadline = time.monotonic() + 30
                        while child.poll() is None and time.monotonic() < deadline:
                            time.sleep(1)
                    if child.returncode != 0:
                        raise RuntimeError(f"Job failed: {job_id}, exit={child.returncode}")
                    child = None
            observed_receipt = verify_receipt(source)
            expected_fields = {"config_sha256": canonical_sha256(config), "method": method,
                               "policy": policy_name, "fold_id": fold}
            if any(observed_receipt["contract"].get(k) != v for k, v in expected_fields.items()):
                raise RuntimeError("Campaign receipt identity conflict")
            backup_job(source, backup / method / policy_name / fold)
            state.update(completed=state["completed"] + 1, last_backup_job=job_id, updated_unix=time.time())
            write_json(campaign / "state.json", state)
        state.update(status="complete", current_job=None, child_pid=None, updated_unix=time.time())
        write_json(campaign / "state.json", state)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=15)
        state.update(status="stopped_failure", failure_count=1, error=repr(error), updated_unix=time.time())
        write_json(campaign / "state.json", state)
        write_json(campaign / "FAILED.json", state)
        raise
    finally:
        # Lock is ours: precise nonrecursive removal only after no child remains.
        lock.unlink()


if __name__ == "__main__":
    main()
