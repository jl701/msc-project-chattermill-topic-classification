"""Operational-only completion of the registered validation analysis. No inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.experiments.taxonomy_training_policy_sensitivity import verify_receipt

PARENT = "taxonomy_training_policy_sensitivity_v1"
EXTENSION = "taxonomy_training_policy_extension_v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parent = ROOT / "outputs/experimental" / PARENT
    extension = ROOT / "outputs/experimental" / EXTENSION
    configs = [ROOT / "configs/experiments" / (name + ".json") for name in (PARENT, EXTENSION)]
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        if config["include_official_test"] is not False or config["test_contract_count"] != 0:
            raise RuntimeError("Forbidden official-test configuration")
    if digest(configs[0]) != "a7775e0d98de47969226d1be938889723bf9145e0268e691ccf531a1ee368ebc":
        raise RuntimeError("Original study config changed")
    state = json.loads((parent / "campaign/gpu/state.json").read_text())
    if any(state.get(k) != v for k, v in {"status": "complete", "completed": 24, "planned": 24,
                                        "failure_count": 0, "test_contract_count": 0}.items()):
        raise RuntimeError("DistilBERT completion gate failed")
    for path in (parent / "fewshot", parent / "mixed_component_diagnostic", parent / "analysis", extension / "analysis"):
        if path.exists():
            raise RuntimeError(f"Existing postprocessing output requires inspection: {path}")
    run = parent / "postprocess_20260908"
    run.mkdir(exist_ok=False)
    stages = [
        ["scripts/run_taxonomy_policy_qwen_sensitivity.py", "--phase", "evaluate"],
        ["scripts/analyse_taxonomy_training_policy_sensitivity.py"],
        ["scripts/analyse_taxonomy_training_policy_extension.py"],
    ]
    sources = set(configs)
    sources.update((ROOT / "src").rglob("*.py"))
    sources.update(ROOT / command[0] for command in stages)
    sources.update(ROOT / "scripts" / name for name in (
        "campaign_taxonomy_training_policy_sensitivity.py", "run_taxonomy_training_policy_extension.py"))
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}
    source_hashes[str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    manifest = {"started_unix": time.time(), "commands": stages, "source_hashes": source_hashes,
                "scope": "train_validation_only", "new_inference": False, "new_training": False,
                "test_contract_count": 0, "git_commit_push": False,
                "statistics": {"bootstrap_draws": 20000, "seed": 13,
                               "unit": "synchronised validation row_uid review clusters"}}
    write(run / "manifest.json", manifest)
    environment = dict(os.environ)
    environment.update({"PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": "",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    try:
        for index, command in enumerate(stages, 1):
            started = time.time()
            command = [sys.executable, *command]
            print(f"START_STAGE {index}/3 {Path(command[1]).name}", flush=True)
            with (run / f"stage_{index}.log").open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    write(run / "state.json", {"status": "running", "stage": index, "pid": child.pid,
                          "command": command, "updated_unix": time.time(), "elapsed_seconds": time.time()-started})
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f"Stage {index} failed with exit {child.returncode}. Inspect retained log.")
            print(f"COMPLETE_STAGE {index}/3 elapsed={time.time()-started:.1f}s", flush=True)
        print("VERIFY_ALL_LOCAL_AND_BACKUP_RECEIPTS", flush=True)
        roots = []
        config = json.loads(configs[0].read_text())
        for method in ("tfidf", "distilbert", "fewshot", "mixed_component_diagnostic"):
            policies = list(config["policies"])
            if method == "tfidf":
                policies += [f"size_matched_label_masked_s{seed}" for seed in config["tfidf"]["subset_seeds"]]
            roots += [parent / method / policy / fold for policy in policies for fold in config["folds"]]
        roots += [extension / "dcwt" / policy / fold for policy in config["policies"] for fold in config["folds"]]
        ext_config = json.loads(configs[1].read_text())
        roots += [extension / "invariance" / method / policy / fold
                  for method in ext_config["invariance"]["methods"] for policy in config["policies"] for fold in config["folds"]]
        roots += [parent / "qwen_inference", parent / "analysis", extension / "analysis"]
        verified = []
        for index, path in enumerate(roots, 1):
            if (path / "FAILED.json").exists():
                raise RuntimeError(f"Active failure artifact: {path}")
            source = verify_receipt(path)
            relative = path.relative_to(ROOT / "outputs/experimental")
            backup = ROOT.parent / "cloud_backups" / relative
            target = verify_receipt(backup)
            receipt_hash = digest(path / "receipt.json")
            if source != target or digest(backup / "receipt.json") != receipt_hash:
                raise RuntimeError(f"Backup identity mismatch: {relative}")
            verified.append({"unit": str(relative), "receipt_sha256": receipt_hash, "files": len(source["files"])})
            if index % 12 == 0:
                print(f"BACKUP_AUDIT {index}/{len(roots)}", flush=True)
        for relative, expected in source_hashes.items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f"Frozen source changed: {relative}")
        audit = {"status": "pass", "completed_unix": time.time(), "elapsed_seconds": time.time()-manifest["started_unix"],
                 "units_verified_in_both_locations": len(verified), "units": verified,
                 "source_hashes_unchanged": True, "failure_count": 0, "test_contract_count": 0,
                 "official_test_accessed": False, "new_inference": False, "new_training": False}
        write(run / "completion_audit.json", audit)
        write(run / "state.json", {"status": "complete", "completed_stages": 3, "updated_unix": time.time(),
                                   "failure_count": 0, "test_contract_count": 0})
        print("ALL_REGISTERED_EVALUATION_ANALYSIS_BACKUP_AUDITS_PASS", flush=True)
    except BaseException as error:
        write(run / "FAILED.json", {"error": repr(error), "at": time.time(), "rerun_authorised": False})
        raise


if __name__ == "__main__":
    main()
