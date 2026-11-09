"""Verify frozen Qwen artifacts and prepare private validation-only serving replay.

This script never opens the official test split, downloads models, or loads GPU
weights. Review text, gold labels and machine-specific paths stay in outputs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.experiments.taxonomy_final_test import (
    build_final_test_jobs,
    canonical_sha256,
    file_sha256,
)
from msc_project.experiments.taxonomy_final_test_artifacts import (
    ArtifactRoots,
    resolve_job_artifacts,
)
from msc_project.experiments.taxonomy_final_test_scoring import (
    _assert_few_shot_contract,
    example_filtered_train_rows,
    final_fold,
)
from msc_project.experiments.taxonomy_resources import load_minimal_descriptions
from msc_project.experiments.taxonomy_two_stage_runtime import render_aspect_candidate
from msc_project.llm.qwen_two_stage_classifier import (
    qwen_two_stage_few_shot_contract_sha256,
    qwen_two_stage_contract_sha256,
)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SAMPLING_SALT = "chattermill-serving-validation-replay-20261109-v1"
FEW_SHOT = "frozen_qwen_few_shot"
QLORA = "qwen_candidate_pair_qlora"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def check_selection_seal(value: dict) -> None:
    payload = dict(value)
    expected = payload.pop("selection_payload_sha256")
    if canonical_sha256(payload) != expected:
        raise ValueError("Frozen selection payload seal mismatch")


def check_snapshot(path: Path) -> dict:
    """Check required local files and shard identities, without model loading."""
    required = ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required if not (path / name).is_file()]
    files = {}
    if not missing:
        index = read_json(path / "model.safetensors.index.json")
        for name in sorted(set(index["weight_map"].values())):
            shard = path / name
            if not shard.is_file() or shard.stat().st_size == 0:
                missing.append(name)
            else:
                files[name] = {"bytes": shard.stat().st_size}
        for name in required:
            files[name] = {"bytes": (path / name).stat().st_size, "sha256": file_sha256(path / name)}
    return {
        "path": str(path), "status": "present_not_loaded" if not missing else "incomplete",
        "missing_files": missing, "files": files,
        "weight_hashes_verified": False,
        "note": "Presence and metadata checks only; runtime separately verifies loadability. No model download.",
    }


def prepare(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = PROJECT_ROOT / "docs/experiments/taxonomy_final_test_v1_artifact_inventory.json"
    preregistration_path = PROJECT_ROOT / "configs/experiments/taxonomy_final_test_v1.json"
    taxonomy_path = PROJECT_ROOT / "configs/experiments/fabsa_aspect_descriptions_minimal_v2.json"
    inventory = read_json(inventory_path)
    config = read_json(preregistration_path)
    resource = load_minimal_descriptions(taxonomy_path, require_approved=True)
    cloud = args.cloud_root.resolve()
    roots = ArtifactRoots(
        project_root=PROJECT_ROOT,
        local_validation_root=args.source_repo.resolve() / "outputs/experimental/taxonomy_post_supervisor_local_v1",
        cloud_worker_distil_frozen=cloud / "worker-distil-frozen",
        cloud_worker_qlora_a=cloud / "worker-qlora-a",
        cloud_worker_qlora_b=cloud / "worker-qlora-b",
    )
    jobs = [job for job in build_final_test_jobs(config)
            if job.level == "L2" and job.method_id in (FEW_SHOT, QLORA)]
    if not args.all_folds:
        jobs = [job for job in jobs if job.fold_id == args.fold_id]
    resolved = {}
    audit_rows = []
    for job in jobs:
        try:
            artifacts = resolve_job_artifacts(job, inventory, roots)
            check_selection_seal(dict(artifacts.selection))
            row = {"fold_id": job.fold_id, "method_id": job.method_id, "status": "pass",
                   "source_artifact_sha256s": dict(artifacts.source_artifact_sha256s)}
            resolved[(job.fold_id, job.method_id)] = artifacts
        except (OSError, ValueError, KeyError, TypeError) as error:
            row = {"fold_id": job.fold_id, "method_id": job.method_id, "status": "fail", "error": str(error)}
        audit_rows.append(row)
    write_json(output / "artifact_verification.json", {"rows": audit_rows})
    fs = resolved.get((args.fold_id, FEW_SHOT))
    qlora = resolved.get((args.fold_id, QLORA))
    if fs is None or qlora is None:
        raise ValueError("Selected fold failed frozen artifact verification; see artifact_verification.json")
    fold = final_fold(fs.job)
    checkpoint = qlora.checkpoint_path
    manifest = read_json(checkpoint / "checkpoint.manifest.json")
    adapter_config = read_json(checkpoint / "adapter/adapter_config.json")
    training_contract = manifest["training_contract"]
    prompt_contract = read_json(checkpoint / "two_stage_qlora_contract.json")
    fs_contract = fs.selection["scope_contract"]
    for value in (training_contract, fs_contract):
        if value["model_id"] != MODEL_ID or value["model_revision"] != MODEL_REVISION:
            raise ValueError("Frozen base model identity differs")
    if adapter_config["base_model_name_or_path"] != MODEL_ID:
        raise ValueError("Adapter config has a different base model")
    if prompt_contract["prompt_contract_sha256"] != qwen_two_stage_contract_sha256(max_length=384):
        raise ValueError("QLoRA prompt code no longer matches frozen training contract")
    if fs_contract["prompt_contract_sha256"] != qwen_two_stage_few_shot_contract_sha256(max_length=1024):
        raise ValueError("Few-shot prompt code no longer matches frozen selection")
    # No helper that opens all three official splits is used here.
    train = load_official_fabsa_splits(args.data_dir, ("train",))
    validation = load_official_fabsa_splits(args.data_dir, ("validation",))
    if len(train) != 7930 or len(validation) != 1057:
        raise ValueError("Unexpected official train/validation row counts")
    if not 0 < args.development_count < len(validation):
        raise ValueError("Development count must leave a nonempty evaluation replay")
    filtered_train = example_filtered_train_rows(train, fold.heldout_aspects)
    demos = _assert_few_shot_contract(filtered_train, fold.seen_aspects, resource, fs.selection)
    # Entirely label-independent sampling. Character length is metadata, not a filter.
    validation = validation.copy()
    validation["sampling_hash"] = validation["row_uid"].map(
        lambda uid: hashlib.sha256(f"{SAMPLING_SALT}|{uid}".encode()).hexdigest())
    validation["character_count"] = validation["text"].map(len)
    lengths = validation["character_count"].to_numpy()
    boundaries = np.quantile(lengths, np.arange(0.1, 1.0, 0.1)).tolist()
    validation["length_decile"] = validation["character_count"].map(
        lambda length: int(np.searchsorted(boundaries, length, side="right") + 1))
    ordered = validation.sort_values(["sampling_hash", "row_uid"], kind="stable")
    parts = {"development": ordered.iloc[:args.development_count],
             "evaluation": ordered.iloc[args.development_count:]}
    replay = {
        "source_split": "validation", "source_rows": len(validation),
        "sampling_salt": SAMPLING_SALT,
        "sampling_rule": "sort sha256(salt + '|' + row_uid), then row_uid; first N development, remainder evaluation",
        "development_count": args.development_count,
        "evaluation_use": "Frozen engineering replay only; do not tune thresholds/prompts/model using evaluation results",
        "blind_evaluation": False,
        "limitation": "Both subsets come from validation already used for historical selection; they are not a new blind test.",
        "official_test_opened": False,
        "validation_character_length_decile_boundaries": boundaries,
        "length_definition": "Python len(text), Unicode code points, before tokenizer truncation; not token counts",
        "splits": {},
    }
    all_ids = []
    for split, frame in parts.items():
        requests, labels, metadata = [], [], []
        for row in frame.itertuples(index=False):
            uid = str(row.row_uid)
            request_id = "replay-" + hashlib.sha256(uid.encode()).hexdigest()[:20]
            requests.append({"request_id": request_id, "text": str(row.text)})
            pairs = [{"aspect": str(a), "sentiment": str(s)} for a, s in row.labels]
            labels.append({"request_id": request_id, "row_uid": uid, "pairs": pairs,
                           "aspects": sorted({p["aspect"] for p in pairs})})
            metadata.append({"request_id": request_id, "row_uid": uid,
                             "sampling_hash": row.sampling_hash,
                             "character_count": int(row.character_count),
                             "length_decile": int(row.length_decile),
                             "text_sha256": hashlib.sha256(str(row.text).encode()).hexdigest()})
            all_ids.append(request_id)
        files = {}
        for kind, rows in (("replay", requests), ("labels", labels), ("metadata", metadata)):
            path = output / f"{kind}_{split}.jsonl"
            write_jsonl(path, rows)
            files[kind] = {"path": str(path), "sha256": file_sha256(path)}
        replay["splits"][split] = {
            "count": len(frame), "files": files,
            "request_id_sha256": canonical_sha256([r["request_id"] for r in requests]),
            "character_length_quantiles": {str(q): float(np.quantile(frame["character_count"], q))
                                           for q in (0, .1, .25, .5, .75, .9, .95, .99, 1)},
            "length_decile_counts": {str(k): int(v) for k, v in frame["length_decile"].value_counts().sort_index().items()},
        }
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Replay request IDs overlap")
    write_json(output / "replay_manifest.json", replay)
    snapshot_audit = check_snapshot(args.model_snapshot.resolve())
    variants = {
        "few_shot": {"aspect_threshold": fs.aspect_threshold, "runner_up_threshold": fs.runner_up_threshold},
        "qlora": {"aspect_threshold": qlora.aspect_threshold, "runner_up_threshold": qlora.runner_up_threshold},
        "hybrid": {"aspect_threshold": fs.aspect_threshold, "runner_up_threshold": qlora.runner_up_threshold},
    }
    artifacts = {
        "schema_version": "chattermill_serving_artifacts_v1",
        "fold_id": args.fold_id, "training_scope_id": fs.job.training_scope_id, "condition": "D",
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                  "local_snapshot_path": str(args.model_snapshot.resolve()), "snapshot_audit": snapshot_audit},
        "taxonomy": {"version": resource["version"], "path": str(taxonomy_path),
                     "content_sha256": resource["content_sha256"],
                     "canonical_order": resource["canonical_order"],
                     "heldout_aspects": list(fold.heldout_aspects), "seen_aspects": list(fold.seen_aspects),
                     "candidates": [{"aspect": a, "text": render_aspect_candidate(a, "name_and_description", resource),
                                     "is_heldout": a in fold.heldout_aspects} for a in resource["canonical_order"]]},
        "adapter": {"path": str(checkpoint / "adapter"), "checkpoint_path": str(checkpoint),
                    "config": adapter_config, "checkpoint_tree_sha256": qlora.source_artifact_sha256s["checkpoint_tree"]},
        "few_shot": {"selection_path": str(fs.selection_path), "max_length": 1024,
                     "demonstrations": {mode: [value.as_hash_payload() for value in values] for mode, values in demos.items()},
                     "demonstration_sha256s": fs.selection["demonstration_sha256s"],
                     "prompt_contract_sha256": fs_contract["prompt_contract_sha256"]},
        "qlora": {"selection_path": str(qlora.selection_path), "max_length": 384,
                  "prompt_contract_sha256": prompt_contract["prompt_contract_sha256"]},
        "variants": variants,
        "sources": {"inventory": {"path": str(inventory_path), "sha256": file_sha256(inventory_path)},
                    "preregistration": {"path": str(preregistration_path), "sha256": file_sha256(preregistration_path)},
                    "train": {"path": str(args.data_dir / "train.csv"), "sha256": file_sha256(args.data_dir / "train.csv")},
                    "validation": {"path": str(args.data_dir / "validation.csv"), "sha256": file_sha256(args.data_dir / "validation.csv")}},
        "audit": {"frozen_artifacts": audit_rows, "selected_fold_passed": True,
                  "demonstrations_match_frozen_hashes": True, "prompt_contracts_match": True,
                  "filtered_training_rows": len(filtered_train), "model_loaded": False,
                  "official_test_opened": False},
        "replay": replay,
    }
    write_json(output / "artifacts.json", artifacts)
    return {"output": str(output), "selected_fold": args.fold_id,
            "artifact_checks_passed": sum(row["status"] == "pass" for row in audit_rows),
            "artifact_checks_failed": sum(row["status"] == "fail" for row in audit_rows),
            "development_rows": len(parts["development"]), "evaluation_rows": len(parts["evaluation"]),
            "base_snapshot_status": snapshot_audit["status"], "official_test_opened": False}


def main() -> None:
    default_hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--cloud-root", type=Path, default=PROJECT_ROOT.parent / "cloud_backups/taxonomy_two_stage_formal_v2_r2")
    parser.add_argument("--source-repo", type=Path, default=PROJECT_ROOT.parent / "msc-project-chattermill-topic-classification-mainline-integration")
    parser.add_argument("--model-snapshot", type=Path, default=default_hub / "models--Qwen--Qwen3-4B-Instruct-2507/snapshots" / MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/serving_20261109")
    parser.add_argument("--fold-id", default="l2-a01")
    parser.add_argument("--development-count", type=int, default=100)
    parser.add_argument("--all-folds", action="store_true")
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
