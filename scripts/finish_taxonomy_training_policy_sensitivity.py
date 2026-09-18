"""Continue the bounded local study after existing CPU/GPU campaigns finish.

This is an experiment dependency runner, not a cloud or test-data monitor.
It never restarts a failed process and only terminates children it launched.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID, canonical_sha256, receipt, sha256_file, verify_receipt, write_json,
)
from campaign_taxonomy_training_policy_sensitivity import backup_job, snapshot


def campaign_complete(state, expected):
    if state.get("failure_count") != 0 or state.get("test_contract_count") != 0:
        raise RuntimeError("Campaign failure or data boundary violation")
    if state.get("planned") != expected or not 0 <= state.get("completed", -1) <= expected:
        raise RuntimeError("Campaign completion boundary mismatch")
    if state.get("status") == "complete":
        if state["completed"] != expected or state.get("child_pid") is not None:
            raise RuntimeError("Incomplete campaign marked complete")
        return True
    if state.get("status") != "running":
        raise RuntimeError("Campaign is not safely running")
    return False


def guard_gpu(gpu, previous_events, limit):
    if float(gpu["temperature.gpu"]) >= limit:
        raise RuntimeError("Registered GPU temperature stop")
    events = previous_events + int(any(str(v).lower() == "active" for k, v in gpu.items() if "slowdown" in k))
    if events >= 2:
        raise RuntimeError("Repeated thermal/hardware slowdown")
    return events


def terminate_owned(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=15)


def main():
    config_path = ROOT / f"configs/experiments/{STUDY_ID}.json"
    config = json.loads(config_path.read_text())
    if config["include_official_test"] is not False or config["test_contract_count"] != 0:
        raise RuntimeError("Forbidden study boundary")
    output = ROOT / config["output_root"]
    backup = ROOT.parent / "cloud_backups" / STUDY_ID
    destination = output / "campaign" / "finish"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "FAILED.json").exists() or (destination / "COMPLETE.json").exists():
        raise RuntimeError("Existing terminal state requires inspection, not automatic restart")
    lock = destination / "worker.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    monitored = [config_path, Path(__file__),
        ROOT / "scripts/campaign_taxonomy_training_policy_sensitivity.py",
        ROOT / "scripts/run_taxonomy_training_policy_sensitivity.py",
        ROOT / "scripts/prepare_taxonomy_policy_qwen_sources.py",
        ROOT / "scripts/run_taxonomy_policy_qwen_sensitivity.py",
        ROOT / "scripts/analyse_taxonomy_training_policy_sensitivity.py"]
    monitored += list((ROOT / "src/msc_project/experiments").glob("taxonomy_*.py"))
    monitored += list((ROOT / "src/msc_project/llm").glob("qwen*.py"))
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in monitored}
    state = {"pid": os.getpid(), "status": "running", "phase": "initialising", "failure_count": 0,
             "test_contract_count": 0, "started_unix": time.time(), "code_and_config_hashes": hashes}
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    child = None

    def persist(**fields):
        state.update(fields, updated_unix=time.time())
        write_json(destination / "state.json", state)

    def check_boundary():
        for relative, expected_hash in hashes.items():
            if sha256_file(ROOT / relative) != expected_hash:
                raise RuntimeError("In-flight code/config change: " + relative)
        if shutil.disk_usage(output).free / 1024**3 < config["safety"]["minimum_free_gib"]:
            raise RuntimeError("Disk headroom stop")

    def inspect_campaign(worker, expected):
        campaign = output / "campaign" / worker
        observed = json.loads((campaign / "state.json").read_text())
        if observed["config_sha256"] != hashes[str(config_path.relative_to(ROOT))]:
            raise RuntimeError("Campaign config identity conflict")
        complete = campaign_complete(observed, expected)
        if not complete:
            pid = observed.get("worker_pid")
            if not pid or not psutil.pid_exists(pid):
                raise RuntimeError(worker + " campaign process died")
            command = psutil.Process(pid).cmdline()
            if not any("campaign_taxonomy_training_policy_sensitivity.py" in token for token in command):
                raise RuntimeError(worker + " campaign PID identity conflict")
            if time.time() - observed["updated_unix"] > 1800:
                raise RuntimeError(worker + " campaign state stale for thirty minutes")
        return complete and not (campaign / "worker.lock").exists(), observed

    def run_phase(phase, command, *, gpu=False):
        nonlocal child
        check_boundary()
        persist(phase=phase, command=command, child_pid=None)
        events = 0
        with (destination / f"{phase}.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            persist(child_pid=child.pid)
            while child.poll() is None:
                check_boundary()
                sample = {"at": time.time(), "phase": phase, "child_pid": child.pid}
                if gpu:
                    sample["gpu"] = snapshot()
                    events = guard_gpu(sample["gpu"], events, config["safety"]["maximum_temperature_c"])
                with (destination / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample) + "\n")
                persist(last_telemetry=sample)
                deadline = time.monotonic() + config["safety"]["gpu_poll_seconds"]
                while child.poll() is None and time.monotonic() < deadline:
                    time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"{phase} failed with exit {child.returncode}")
            child = None
            persist(child_pid=None)

    try:
        check_boundary()
        prepared = output / "qwen_prepared"
        preparation = verify_receipt(prepared)
        if preparation["contract"]["config_sha256"] != canonical_sha256(config):
            raise RuntimeError("Preparation config identity conflict")
        backup_job(prepared, backup / "qwen_prepared")
        # Immutable data-audit manifests were produced before training began.
        audit_root = output / "data_audit"
        if not (audit_root / "receipt.json").exists():
            receipt(audit_root, {"study_id": STUDY_ID, "config_sha256": canonical_sha256(config),
                                "role": "train_validation_only_policy_and_demonstration_manifests"})
        backup_job(audit_root, backup / "data_audit")
        persist(phase="waiting_for_core_gpu", preparation_receipt_sha256=sha256_file(prepared / "receipt.json"))
        while True:
            check_boundary()
            cpu_done, cpu = inspect_campaign("cpu", 60)
            gpu_done, gpu = inspect_campaign("gpu", 24)
            persist(core_cpu_completed=cpu["completed"], core_gpu_completed=gpu["completed"])
            if gpu_done:
                break
            time.sleep(30)
        run_phase("qwen_inference", ["scripts/run_taxonomy_policy_qwen_sensitivity.py", "--phase", "infer"], gpu=True)
        backup_job(output / "qwen_inference", backup / "qwen_inference")
        run_phase("qwen_evaluation", ["scripts/run_taxonomy_policy_qwen_sensitivity.py", "--phase", "evaluate"])
        persist(phase="waiting_for_core_cpu")
        while True:
            check_boundary()
            complete, cpu = inspect_campaign("cpu", 60)
            persist(core_cpu_completed=cpu["completed"])
            if complete:
                break
            time.sleep(30)
        run_phase("complete_case_analysis", ["scripts/analyse_taxonomy_training_policy_sensitivity.py"])
        analysis = verify_receipt(output / "analysis")
        persist(status="complete", phase="complete", analysis_receipt_sha256=sha256_file(output / "analysis/receipt.json"),
                analysis_file_count=len(analysis["files"]), child_pid=None)
        write_json(destination / "COMPLETE.json", state)
        print("ALL_LOCAL_POLICY_SENSITIVITY_WORK_COMPLETE", flush=True)
    except BaseException as error:
        terminate_owned(child)
        persist(status="stopped_failure", failure_count=1, error=repr(error), child_pid=None)
        write_json(destination / "FAILED.json", state)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
