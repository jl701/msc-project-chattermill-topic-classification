"""Bounded CPU extension with hash guards, per-fold backups and dependency finish."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import psutil
import pandas as pd
import run_taxonomy_training_policy_extension as run
from analyse_taxonomy_training_policy_sensitivity import audit_score_grid
from finish_taxonomy_training_policy_sensitivity import terminate_owned

ROOT = run.ROOT


def check_dependency(path):
    state = json.loads(path.read_text())
    if state.get("failure_count") != 0 or state.get("test_contract_count") != 0:
        raise RuntimeError("Parent dependency has failed")
    if state["status"] == "complete":
        return True
    if state["status"] != "running":
        raise RuntimeError("Unsafe parent status")
    pid = state.get("worker_pid", state.get("pid"))
    if not pid or not psutil.pid_exists(pid):
        raise RuntimeError("Parent process no longer exists")
    return False


def main():
    config, _, _, folds, _ = run.load_extension()
    output = ROOT / config["output_root"]
    campaign = output / "campaign"
    campaign.mkdir(parents=True, exist_ok=True)
    if (campaign / "state.json").exists():
        raise RuntimeError("Existing campaign requires inspection, not automatic restart")
    lock = campaign / "worker.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    run.verify_receipt(output / "descriptor_cache", run.contract(config, "frozen_dcwt_descriptors"))
    run.verify_receipt(output / "invariance_summary", run.contract(config, "frozen_invariance_audit"))
    monitored = [run.CONFIG, Path(__file__), Path(run.__file__),
                 ROOT / "scripts/analyse_taxonomy_training_policy_extension.py",
                 ROOT / "scripts/run_taxonomy_description_weight_transfer.py",
                 ROOT / "scripts/campaign_taxonomy_training_policy_sensitivity.py",
                 ROOT / "scripts/analyse_taxonomy_training_policy_sensitivity.py"]
    monitored += list((ROOT / "src/msc_project/experiments").glob("taxonomy_*.py"))
    hashes = {str(p): run.sha256_file(p) for p in monitored}
    state = {"status": "running", "pid": os.getpid(), "planned": 24, "completed": 0,
             "failure_count": 0, "test_contract_count": 0, "phase": "dcwt", "started_unix": time.time(),
             "code_and_config_hashes": hashes}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    child = None

    def persist(**fields):
        state.update(fields, updated_unix=time.time())
        run.write_json(campaign / "state.json", state)

    def guard():
        if shutil.disk_usage(output).free / 1024**3 < 20:
            raise RuntimeError("Disk headroom below 20 GiB")
        if any(run.sha256_file(Path(p)) != digest for p, digest in hashes.items()):
            raise RuntimeError("Immutable execution code changed")

    def execute(command, job, maximum_seconds):
        nonlocal child
        guard()
        with (campaign / f"{job}.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            persist(current_job=job, child_pid=child.pid, command=command)
            started = time.monotonic()
            while child.poll() is None:
                guard()
                if time.monotonic() - started > maximum_seconds:
                    raise RuntimeError("Registered execution time cap exceeded")
                persist()
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f"Child failed: {job}, exit={child.returncode}")
            child = None
            persist(child_pid=None)

    try:
        persist()
        for fold in folds:
            for policy in config["policies"]:
                job = f"dcwt__{policy}__{fold.fold_id}"
                source = output / "dcwt" / policy / fold.fold_id
                execute([sys.executable, "scripts/run_taxonomy_training_policy_extension.py", "--phase", "dcwt",
                         "--fold-id", fold.fold_id, "--policy", policy], job, config["dcwt"]["maximum_seconds_per_fold_policy"])
                run.verify_receipt(source)
                # The first and every later full fold must pass score reconstruction
                # before the next expensive job can start.
                for family in run.legacy.GENERATOR_FAMILIES:
                    family_root = source / "generators" / family
                    selection = json.loads((family_root / "selection.json").read_text())
                    digests = [audit_score_grid(family_root, condition, fold.fold_id, selection,
                               pd.read_csv(family_root / f"{condition}_row_counts.csv")) for condition in ("D", "N")]
                    if digests[0] != digests[1]:
                        raise ValueError("N/D invariant seen score conflict")
                run.backup_job(source, ROOT.parent / "cloud_backups" / run.EXT_ID / "dcwt" / policy / fold.fold_id)
                persist(completed=state["completed"] + 1, last_backup_job=job)
        persist(phase="waiting_for_parent_analysis", current_job=None)
        parent_campaign = ROOT / "outputs/experimental" / run.PARENT_ID / "campaign"
        while True:
            guard()
            ready = [check_dependency(parent_campaign / worker / "state.json") for worker in ("cpu", "gpu", "finish")]
            if all(ready):
                break
            persist()
            time.sleep(30)
        persist(phase="combined_analysis")
        execute([sys.executable, "scripts/analyse_taxonomy_training_policy_extension.py"], "combined_analysis", 7200)
        run.verify_receipt(output / "analysis")
        persist(status="complete", phase="complete", current_job=None, child_pid=None)
    except BaseException as error:
        terminate_owned(child)
        persist(status="stopped_failure", failure_count=1, error=repr(error))
        run.write_json(campaign / "FAILED.json", state)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
