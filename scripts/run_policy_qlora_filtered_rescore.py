"""Prepare and score the audited filtered adapters without training or selection.

The preparation phase is CPU-only. Scoring is an explicit separate invocation.
Historical checkpoints, candidate records and thresholds remain byte-identical.
New N/D scores are written to an isolated policy-labelled output root.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath

import run_taxonomy_policy_qlora_completion as masked

ROOT = Path(__file__).resolve().parents[1]
POLICY = "review_filtered"
ORIGIN = "aa84212976a652d62cfca31ed8bf0516a216c485"
MANIFEST = "policy_provenance/filtered_rescore_inputs.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_path(root, relative):
    value = PurePosixPath(str(relative))
    if value.is_absolute() or ".." in value.parts or "\\" in str(relative) or ":" in str(relative):
        raise ValueError("Unsafe artifact path")
    path = (root / str(value)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes root")
    return path


def code_hashes():
    paths = [
        "scripts/run_policy_qlora_filtered_rescore.py",
        "scripts/run_taxonomy_policy_qlora_completion.py",
        "scripts/run_taxonomy_two_stage_trainable_validation.py",
        "src/msc_project/experiments/taxonomy_two_stage_training.py",
        "src/msc_project/experiments/taxonomy_two_stage_runtime.py",
        "src/msc_project/experiments/taxonomy_two_stage.py",
        "src/msc_project/experiments/taxonomy_protocol.py",
        "src/msc_project/experiments/taxonomy_resources.py",
        "src/msc_project/experiments/taxonomy_post_supervisor_cloud.py",
        "src/msc_project/llm/qwen_pair_classifier.py",
        "src/msc_project/llm/qwen_two_stage_classifier.py",
    ]
    return {name: masked.sha256_file(ROOT / name) for name in paths}


def copy_verified(source, target, expected):
    if masked.sha256_file(source) != expected:
        raise ValueError(f"Source hash mismatch: {source}")
    if target.exists():
        if masked.sha256_file(target) != expected:
            raise FileExistsError(f"Refusing to overwrite: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if masked.sha256_file(target) != expected:
        raise ValueError(f"Copy hash mismatch: {target}")


def validate_audit(audit, selected):
    if (audit.get("status") != "PASS_ADAPTER_TRAINING_REUSE"
            or audit.get("artifact_origin_commit") != ORIGIN
            or audit.get("failed_checks") != []
            or audit.get("candidate_count") != 36 or audit.get("scope_count") != 12
            or audit.get("scopes") != selected):
        raise ValueError("Selected sources must equal the complete passed reuse audit")
    expected = set(masked.l2_scopes())
    scopes = [item["training_scope_id"] for item in selected]
    if len(scopes) != 12 or set(scopes) != expected:
        raise ValueError("Exactly twelve unique selected filtered scopes required")


def expected_identity(executor, manifest, config):
    return {"study_id": masked.STUDY_ID, "policy": POLICY,
            "canonical_seen_policy": masked.CANONICAL_POLICY,
            "artifact_origin_commit": ORIGIN,
            "configuration_sha256": executor.canonical_sha256(config),
            "filtered_input_manifest_sha256": manifest["manifest_sha256"],
            "recipe": masked.frozen_recipe(executor), "retraining": False,
            "reselection": False, "test_contract_count": 0}


def validate_identity(observed, expected):
    # JSON normalisation makes tuple/list representation irrelevant, not values.
    if observed != json.loads(json.dumps(expected)):
        raise ValueError("Prepared run identity/configuration/recipe mismatch")


def prepare(args):
    audit, selected, config = read(args.reuse_audit), read(args.selected_map), read(args.config)
    validate_audit(audit, selected)
    masked.validate_config(config)
    executor = masked.load_executor()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()) and not (root / MANIFEST).exists():
        raise FileExistsError("Preparation requires a new empty root or completed matching manifest")
    files, sources = {}, []
    for item in selected:
        scope = item["training_scope_id"]
        owner = Path(item["worker"]).resolve()
        if root == owner or root.is_relative_to(owner) or owner.is_relative_to(root):
            raise ValueError("Prepared root must not overlap historical evidence")
        selection_path = executor.selection_path(owner, masked.METHOD, scope)
        if selection_path.resolve() != Path(item["selection_path"]).resolve():
            raise ValueError("Selection source path mismatch")
        selection = read(selection_path)
        executor._validate_sealed_payload(selection, "selection_payload_sha256")
        if (selection["selected_learning_rate"] != item["selected_learning_rate"]
                or selection["selected_thresholds"] != item["selected_thresholds"]
                or selection.get("failure_count") != 0 or selection.get("test_contract_count") != 0):
            raise ValueError("Selection differs from audited frozen record")
        to_copy = {selection_path: item["selection_sha256"]}
        candidates = []
        for rate in masked.RATES:
            candidate_path = executor.candidate_result_path(owner, masked.METHOD, scope, rate)
            candidate = read(candidate_path)
            executor._validate_sealed_payload(candidate, "candidate_payload_sha256")
            if (candidate.get("protocol_id") != "taxonomy_two_stage_formal_v2"
                    or candidate.get("training_scope_id") != scope
                    or candidate.get("method_id") != masked.METHOD
                    or candidate.get("learning_rate") != rate
                    or candidate.get("failure_count") != 0 or candidate.get("test_contract_count") != 0):
                raise ValueError("Historical candidate identity/safety mismatch")
            candidates.append(candidate)
            to_copy[candidate_path] = masked.sha256_file(candidate_path)
        if [c["training_contract"]["contract_sha256"] for c in candidates] != selection["candidate_contract_sha256s"]:
            raise ValueError("Frozen candidate set changed")
        chosen = next(c for c in candidates if c["learning_rate"] == selection["selected_learning_rate"])
        if (chosen["training_contract"]["contract_sha256"] != selection["selected_candidate_contract_sha256"]
                or chosen["thresholds"] != selection["selected_thresholds"]
                or chosen["checkpoint_relative_path"] != selection["selected_checkpoint_relative_path"]):
            raise ValueError("Selected checkpoint or thresholds changed")
        checkpoint = safe_path(owner, selection["selected_checkpoint_relative_path"])
        if checkpoint != Path(item["selected_checkpoint"]).resolve():
            raise ValueError("Audited checkpoint path mismatch")
        contract = executor._contract_from_dict(chosen["training_contract"])
        executor.validate_checkpoint(checkpoint, contract)
        checkpoint_manifest = checkpoint / "checkpoint.manifest.json"
        if masked.sha256_file(checkpoint_manifest) != chosen["checkpoint_manifest_sha256"]:
            raise ValueError("Checkpoint manifest binding mismatch")
        to_copy[checkpoint_manifest] = chosen["checkpoint_manifest_sha256"]
        for relative, digest in read(checkpoint_manifest)["files"].items():
            to_copy[safe_path(checkpoint, relative)] = digest
        for source, digest in to_copy.items():
            relative = source.relative_to(owner).as_posix()
            copy_verified(source, safe_path(root, relative), digest)
            files[relative] = digest
        sources.append({"fold_id": item["fold_id"], "scope_id": scope,
                        "selection_relative_path": selection_path.relative_to(owner).as_posix(),
                        "selection_sha256": item["selection_sha256"],
                        "selected_checkpoint_relative_path": selection["selected_checkpoint_relative_path"],
                        "selected_learning_rate": selection["selected_learning_rate"],
                        "selected_thresholds": selection["selected_thresholds"]})
    for path, name in [(args.reuse_audit, "reuse_audit.json"), (args.selected_map, "selected_sources.json"),
                       (args.config, "masked_reference_config.json")]:
        relative = "policy_provenance/" + name
        digest = masked.sha256_file(path)
        copy_verified(path, root / relative, digest)
        files[relative] = digest
    manifest = {"study_id": masked.STUDY_ID, "policy": POLICY, "artifact_origin_commit": ORIGIN,
                "canonical_seen_policy": masked.CANONICAL_POLICY, "sources": sources,
                "files": files, "code_files": code_hashes(),
                "reuse_audit_sha256": masked.sha256_file(args.reuse_audit),
                "reference_config_sha256": executor.canonical_sha256(config),
                "retraining": False, "reselection": False,
                "include_official_test": False, "test_contract_count": 0,
                "score_status": "prepared_not_run"}
    manifest = executor._seal_payload(manifest, "manifest_sha256")
    masked._write_immutable_json(executor, root / MANIFEST, manifest)
    identity = expected_identity(executor, manifest, config)
    masked._write_immutable_json(executor, root / "policy_provenance/run_identity.json", identity)
    return verify(root)


def verify(root):
    root = root.resolve()
    executor = masked.load_executor()
    manifest = read(root / MANIFEST)
    executor._validate_sealed_payload(manifest, "manifest_sha256")
    if (manifest.get("policy") != POLICY or manifest.get("retraining") is not False
            or manifest.get("reselection") is not False or manifest.get("include_official_test") is not False
            or manifest.get("test_contract_count") != 0 or manifest.get("artifact_origin_commit") != ORIGIN
            or manifest.get("canonical_seen_policy") != masked.CANONICAL_POLICY
            or manifest.get("code_files") != code_hashes()):
        raise ValueError("Prepared inference contract/code mismatch")
    for relative, digest in manifest["files"].items():
        if masked.sha256_file(safe_path(root, relative)) != digest:
            raise ValueError(f"Prepared input hash mismatch: {relative}")
    if len(manifest["sources"]) != 12 or {s["scope_id"] for s in manifest["sources"]} != set(masked.l2_scopes()):
        raise ValueError("Prepared scope coverage mismatch")
    identity = read(root / "policy_provenance/run_identity.json")
    config = read(root / "policy_provenance/masked_reference_config.json")
    masked.validate_config(config)
    if manifest.get("reference_config_sha256") != executor.canonical_sha256(config):
        raise ValueError("Prepared configuration digest mismatch")
    validate_identity(identity, expected_identity(executor, manifest, config))
    return {"status": "PASS_PREPARED_INPUTS", "scopes": 12, "selected_adapters": 12,
            "candidate_records": 36, "files_verified": len(manifest["files"]),
            "manifest_sha256": manifest["manifest_sha256"], "gpu_scoring_performed": False,
            "test_contract_count": 0}


def load_scope_inputs(args, executor):
    root = args.output_root.resolve()
    config = read(root / "policy_provenance/masked_reference_config.json")
    masked.validate_config(config)
    for name, digest in config["data_hashes"].items():
        if masked.sha256_file(args.data_dir / name) != digest:
            raise ValueError("Pinned train/validation data changed")
    data = executor._load_scope_data(args)
    train, validation, manifests, resource, folds = data
    if len(validation) != 1057 or not validation.row_uid.str.startswith("validation:").all():
        raise ValueError("Validation-only row boundary failed")
    selected_source = next(s for s in read(root / "policy_provenance/selected_sources.json")
                           if s["training_scope_id"] == args.scope_id)
    if (len(train) != selected_source["training_reviews"]
            or masked.train_pool_hash(train) != selected_source["training_pool_sha256"]
            or {k: executor.training_manifest_sha256(v) for k, v in manifests.items()}
            != selected_source["stage_manifest_sha256"]):
        raise ValueError("Filtered review/sample policy changed since reuse audit")
    return config, data


def score(args):
    verify(args.output_root)
    root = args.output_root.resolve()
    manifest = read(root / MANIFEST)
    item = next(row for row in manifest["sources"] if row["scope_id"] == args.scope_id)
    executor = masked.load_executor()
    selection = read(root / item["selection_relative_path"])
    executor._validate_sealed_payload(selection, "selection_payload_sha256")
    # Freeze the already-audited decision. Never recompute thresholds/LR rankings.
    executor.select_scope = lambda unused: selection
    args.method = masked.METHOD
    args.output_root = root
    args.resume = True
    args.local_files_only = True
    config, data = load_scope_inputs(args, executor)
    _, validation, _, resource, folds = data
    executor._load_scope_data = lambda unused: data
    scorer = masked.CanonicalNDScorer(executor._score_runtime)
    executor._score_runtime = scorer
    fold = folds[0]
    canonical = executor.build_aspect_grid(validation, fold.evaluation_aspects,
                                           executor.variant_map(fold, "N"), resource)
    original_resume = executor.two_stage_shard_resume_state
    def restore(*inputs, **kwargs):
        state, artifact = original_resume(*inputs, **kwargs)
        if state == "complete" and inputs[1].condition == "N":
            scorer.restore(artifact, canonical)
        return state, artifact
    executor.two_stage_shard_resume_state = restore
    scoring_config = {**config, "policy": POLICY, "retraining": False, "reselection": False,
                      "frozen_filtered_input_manifest_sha256": manifest["manifest_sha256"]}
    result = executor.score_selected(args, scoring_config)
    return {**result, "policy": POLICY, "retraining": False, "reselection": False,
            "new_prompt_rows": scorer.new_prompt_rows, "reused_prompt_rows": scorer.reused_prompt_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "verify", "validate-scope", "score"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-audit", type=Path)
    parser.add_argument("--selected-map", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scope-id", choices=list(masked.l2_scopes()))
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        if not all((args.reuse_audit, args.selected_map, args.config)):
            parser.error("prepare requires --reuse-audit --selected-map --config")
        result = prepare(args)
    elif args.phase == "verify":
        result = verify(args.output_root)
    elif args.phase == "validate-scope":
        if not args.scope_id or not args.data_dir:
            parser.error("validate-scope requires --scope-id --data-dir")
        verify(args.output_root)
        _, data = load_scope_inputs(args, masked.load_executor())
        result = {"status": "PASS_FILTERED_SCOPE_PREFLIGHT", "scope_id": args.scope_id,
                  "training_reviews": len(data[0]), "validation_reviews": len(data[1]),
                  "training_instance_manifests": "exact_original_match",
                  "gpu_scoring_performed": False, "test_contract_count": 0}
    else:
        if not args.scope_id or not args.data_dir:
            parser.error("score requires --scope-id --data-dir")
        result = score(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
