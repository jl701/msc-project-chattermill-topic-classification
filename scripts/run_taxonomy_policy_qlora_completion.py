"""Complete masked-only QLoRA using the original two-stage training executor.

Only the training review policy and the explicitly registered canonical N/D
score reuse differ. The shared adapter, three learning-rate candidates, training
budgets, prompts, decoder and seen-only selection functions are reused unchanged.
Official-test data are not loadable through this command.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.experiments import taxonomy_post_supervisor_cloud as formal_v2
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    sha256_file, train_pool_hash, training_rows,
)
from msc_project.experiments.unified_candidate_pairs import CANDIDATE_SENTIMENTS

STUDY_ID = "taxonomy_policy_qlora_completion_v1"
POLICY = "label_masked_all_reviews"
METHOD = "qwen_candidate_pair_qlora"
CANONICAL_POLICY = "registered_first_N_prompt_identity_reuse"
FOLDS = tuple(f"l2-a{i:02d}" for i in range(1, 13))
RATES = (2e-6, 5e-6, 1e-5)
PHASES = ("plan", "prepare-scope", "train-candidate", "select-scope", "score-selected", "scope")
PAIR_KEY = ["row_uid", "candidate_aspect", "candidate_sentiment"]
SCORE_COLUMNS = [*PAIR_KEY, "aspect_score", "sentiment_score"]


def validate_config(config: Mapping[str, Any]) -> None:
    if (config.get("study_id") != STUDY_ID or config.get("policy") != POLICY
            or config.get("allowed_splits") != ["train", "validation"]
            or config.get("include_official_test") is not False
            or config.get("test_contract_count") != 0
            or tuple(config.get("folds", ())) != FOLDS
            or config.get("canonical_seen_policy") != CANONICAL_POLICY):
        raise ValueError("Unregistered masked-only study or data boundary")
    hashes = config.get("data_hashes", {})
    if set(hashes) != {"train.csv", "validation.csv"} or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes.values()
    ):
        raise ValueError("Only pinned train.csv and validation.csv inputs are allowed")
    if config.get("learning_rates", list(RATES)) != list(RATES):
        raise ValueError("The original three learning rates must not change")
    if config.get("method", METHOD) != METHOD:
        raise ValueError("This runner is QLoRA-only")


def l2_scopes():
    scopes = {key: value for key, value in formal_v2.scope_folds().items()
              if re.fullmatch(r"heldout-a\d{2}", key)}
    if len(scopes) != 12 or any(len(value) != 1 or value[0].level != "L2"
                               for value in scopes.values()):
        raise ValueError("Expected exactly twelve single-aspect scopes")
    return scopes


def load_executor():
    path = PROJECT_ROOT / "scripts/run_taxonomy_two_stage_trainable_validation.py"
    spec = importlib.util.spec_from_file_location("_masked_qlora_proven_executor", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    executor.PROTOCOL_ID = STUDY_ID
    for name in ("candidate_result_path", "selection_path", "representative_fold",
                 "formal_conditions", "variant_map"):
        setattr(executor, name, getattr(formal_v2, name))
    executor.scope_folds = l2_scopes
    return executor


def frozen_recipe(executor):
    if tuple(executor.learning_rates(METHOD)) != RATES:
        raise ValueError("Original executor learning-rate registry changed")
    parameters = executor._training_parameters(METHOD, RATES[0])
    training = parameters["training"]
    required = {"max_length": 384, "batch_size": 1,
                "gradient_accumulation_steps": 8, "epochs": 1, "seed": 13}
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError("Original two-stage training recipe changed")
    if parameters.get("shared_adapter_for_both_stages") is not True:
        raise ValueError("A shared two-stage adapter is required")
    return {"training_candidates": [executor._training_parameters(METHOD, rate)
                                    for rate in RATES],
            "total_budget": 4096, "aspect_budget": 2048,
            "scoring_batch_size": 8, "scoring_max_length": 384,
            "canonical_seen_policy": CANONICAL_POLICY}


def _write_immutable_json(executor, path: Path, value: Mapping[str, Any]):
    # Refuse changed configuration/provenance on resume, not just changed paths.
    normalised = json.loads(json.dumps(dict(value), allow_nan=False))
    if path.exists():
        if executor._read_object(path) != normalised:
            raise FileExistsError(f"Immutable JSON conflict: {path}")
    else:
        executor._write_json_atomic(path, normalised)


def _write_immutable_frame(path: Path, frame: pd.DataFrame):
    payload = gzip.compress(frame.to_csv(index=False, lineterminator="\n",
                                        float_format="%.17g").encode(), mtime=0)
    if path.exists():
        if gzip.decompress(path.read_bytes()) != gzip.decompress(payload):
            raise FileExistsError(f"Immutable frame conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
    os.replace(temporary, path)


def _publish(executor, root, paths, unit_id, contract_hash):
    executor.publish_artifact_unit(
        root, [path.relative_to(root) for path in paths], unit_id=unit_id,
        protocol_id=STUDY_ID, contract_sha256=contract_hash,
    )


def load_masked_scope(args, config, executor):
    folds = l2_scopes().get(args.scope_id)
    if folds is None:
        raise ValueError("Only a registered single-aspect scope may run")
    for name, expected in config["data_hashes"].items():
        if sha256_file(args.data_dir / name) != expected:
            raise ValueError(f"Data hash mismatch: {name}")
    frame = executor.load_official_fabsa_splits(args.data_dir, ("train", "validation"))
    if set(frame.original_split.astype(str)) != {"train", "validation"}:
        raise ValueError("Unregistered data split loaded")
    if frame.row_uid.astype(str).duplicated().any():
        raise ValueError("Duplicate row identities")
    original_train = frame.loc[frame.original_split.eq("train")].copy()
    validation = frame.loc[frame.original_split.eq("validation")].copy()
    if len(original_train) != 7930 or len(validation) != 1057:
        raise ValueError("Registered train/validation row counts changed")
    if (not original_train.row_uid.astype(str).str.startswith("train:").all()
            or not validation.row_uid.astype(str).str.startswith("validation:").all()):
        raise ValueError("Unexpected partition-specific row identifiers")
    validation["supervision_labels"] = validation["labels"]
    fold = executor.representative_fold(folds)
    train = training_rows(original_train, fold, POLICY)
    resource = executor.load_description_bundle(require_approved=True)
    manifests = executor.build_two_stage_training_manifests(
        train, fold.seen_aspects, resource, total_budget=4096, aspect_budget=2048, seed=13)
    if set(manifests) != {"aspect_presence", "sentiment"}:
        raise ValueError("Unexpected two-stage manifests")
    for task, manifest in manifests.items():
        if len(manifest) != 2048 or set(manifest.candidate_aspect) - set(fold.seen_aspects):
            raise ValueError("Training budget or held-out exclusion failed")
        if not manifest.row_uid.astype(str).str.startswith("train:").all():
            raise ValueError("Training manifest contains non-training rows")
    folder = args.output_root / "policy_provenance" / args.scope_id
    paths = []
    for task, manifest in manifests.items():
        path = folder / f"{task}.csv.gz"
        _write_immutable_frame(path, manifest)
        paths.append(path)
    provenance = {
        "study_id": STUDY_ID, "policy": POLICY, "scope_id": args.scope_id,
        "configuration_sha256": executor.canonical_sha256(config),
        "train_pool_sha256": train_pool_hash(train), "training_review_count": len(train),
        "validation_review_count": len(validation), "seen_aspects": list(fold.seen_aspects),
        "heldout_aspects": list(fold.heldout_aspects),
        "manifests": {task: executor.training_manifest_summary(manifest)
                      for task, manifest in manifests.items()},
        "files": {path.name: sha256_file(path) for path in paths},
        "recipe": frozen_recipe(executor), "test_contract_count": 0,
    }
    provenance_path = folder / "scope.json"
    _write_immutable_json(executor, provenance_path, provenance)
    _publish(executor, args.output_root, [*paths, provenance_path],
             f"policy-provenance-{args.scope_id}", executor.canonical_sha256(provenance))
    return train, validation, manifests, resource, folds


def calibration_paths(args, executor):
    folder = args.output_root / "calibration_scores" / args.scope_id / (
        "lr-" + executor.learning_rate_token(args.learning_rate))
    return folder / "seen_scores.csv.gz", folder / "manifest.json"


def save_calibration(args, config, executor, scored, contract, fold):
    if (set(scored.candidate_aspect) != set(fold.seen_aspects)
            or len(scored) != 1057 * 11 * 3
            or scored.duplicated(PAIR_KEY).any()
            or not scored.row_uid.astype(str).str.startswith("validation:").all()
            or not np.isfinite(scored[["aspect_score", "sentiment_score"]]).all().all()):
        raise ValueError("Calibration is not the complete finite seen-only validation grid")
    score_path, manifest_path = calibration_paths(args, executor)
    _write_immutable_frame(score_path, scored)
    record = executor._seal_payload({
        "study_id": STUDY_ID, "policy": POLICY, "training_contract_sha256": contract.contract_sha256,
        "configuration_sha256": executor.canonical_sha256(config), "learning_rate": args.learning_rate,
        "scope_id": args.scope_id, "rows": len(scored), "test_contract_count": 0,
        "selection_partition": "seen_validation_only", "csv_sha256": sha256_file(score_path),
        "calibration_pair_identity_sha256": executor.pair_identity_hash(scored),
        "calibration_score_sha256": executor.dataframe_sha256(scored, SCORE_COLUMNS, sort_columns=PAIR_KEY),
    }, "payload_sha256")
    _write_immutable_json(executor, manifest_path, record)
    _publish(executor, args.output_root, [score_path, manifest_path],
             f"calibration-{args.scope_id}-lr-{executor.learning_rate_token(args.learning_rate)}",
             contract.contract_sha256)


def validate_candidate(args, config, executor, manifests):
    expected = executor._training_contract(config=config, method_id=METHOD,
        scope_id=args.scope_id, manifests=manifests, learning_rate=args.learning_rate)
    path = executor.candidate_result_path(args.output_root, METHOD, args.scope_id, args.learning_rate)
    candidate = executor._read_object(path)
    executor._validate_sealed_payload(candidate, "candidate_payload_sha256")
    if (candidate.get("training_contract") != expected.to_dict()
            or candidate.get("failure_count") != 0 or candidate.get("test_contract_count") != 0
            or candidate.get("selection_partition") != "seen_validation_only"):
        raise ValueError("Candidate does not match this masked study's exact contract")
    checkpoint = (args.output_root / candidate["checkpoint_relative_path"]).resolve()
    if not checkpoint.is_relative_to(args.output_root):
        raise ValueError("Checkpoint escapes the isolated output root")
    executor.validate_checkpoint(checkpoint, expected)
    if sha256_file(checkpoint / "checkpoint.manifest.json") != candidate["checkpoint_manifest_sha256"]:
        raise ValueError("Candidate checkpoint-manifest binding mismatch")
    score_path, manifest_path = calibration_paths(args, executor)
    record = executor._read_object(manifest_path)
    executor._validate_sealed_payload(record, "payload_sha256")
    if (record.get("training_contract_sha256") != expected.contract_sha256
            or record.get("configuration_sha256") != executor.canonical_sha256(config)
            or record.get("csv_sha256") != sha256_file(score_path)
            or any(record.get(key) != candidate.get(key) for key in
                   ("calibration_pair_identity_sha256", "calibration_score_sha256"))):
        raise ValueError("Raw candidate calibration evidence conflict")
    return candidate


class CanonicalNDScorer:
    """Compute registered-first N, then reuse identical seen prompt scores for D."""

    def __init__(self, base_score):
        self.base_score = base_score
        self.cache = {}
        self.new_prompt_rows = 0
        self.reused_prompt_rows = 0

    @staticmethod
    def key(row):
        return (str(row.row_uid), str(row.candidate_aspect), str(row.text), str(row.candidate_text))

    def restore(self, artifact, canonical_aspects):
        indexed = artifact.set_index(PAIR_KEY)
        identities = set(zip(artifact.row_uid.astype(str), artifact.candidate_aspect.astype(str)))
        for row in canonical_aspects.itertuples(index=False):
            if (str(row.row_uid), str(row.candidate_aspect)) not in identities:
                continue
            values = indexed.loc[(str(row.row_uid), str(row.candidate_aspect))]
            vector = np.array([float(values.loc[s, "sentiment_score"]) for s in CANDIDATE_SENTIMENTS])
            presence = values.aspect_score.to_numpy(float)
            if not np.isfinite(vector).all() or not np.isfinite(presence).all() or np.ptp(presence) != 0:
                raise ValueError("Invalid canonical resumed N scores")
            self.cache[self.key(row)] = (float(presence[0]), vector)

    def __call__(self, method, runtime, grid):
        rows = list(grid.itertuples(index=False))
        missing = [i for i, row in enumerate(rows) if self.key(row) not in self.cache]
        if any(str(getattr(rows[i], "condition", "N")) == "D"
               and bool(getattr(rows[i], "is_seen", False)) for i in missing):
            raise ValueError("D seen scores require the validated canonical N source")
        if missing:
            aspect, sentiment = self.base_score(method, runtime, grid.iloc[missing].copy())
            vectors = np.asarray(sentiment, dtype=float).reshape(len(missing), 3)
            if not np.isfinite(aspect).all() or not np.isfinite(vectors).all():
                raise ValueError("Non-finite canonical inference")
            for index, presence, vector in zip(missing, aspect, vectors):
                self.cache[self.key(rows[index])] = (float(presence), vector.copy())
        self.new_prompt_rows += len(missing)
        self.reused_prompt_rows += len(rows) - len(missing)
        return (np.asarray([self.cache[self.key(row)][0] for row in rows]),
                np.asarray([self.cache[self.key(row)][1] for row in rows]).reshape(-1))


def execute_phase(args, config):
    validate_config(config)
    if args.scope_id not in l2_scopes() or args.phase not in PHASES:
        raise ValueError("A registered scope and phase are required")
    if args.phase == "scope":
        for rate in RATES:
            execute_phase(argparse.Namespace(**{**vars(args), "phase": "train-candidate", "learning_rate": rate}), config)
        execute_phase(argparse.Namespace(**{**vars(args), "phase": "select-scope"}), config)
        return execute_phase(argparse.Namespace(**{**vars(args), "phase": "score-selected"}), config)
    executor = load_executor()
    recipe = frozen_recipe(executor)
    if args.phase == "plan":
        return {"study_id": STUDY_ID, "policy": POLICY, "scopes": list(l2_scopes()),
                "training_candidates": 36, "selected_fold_conditions": 24,
                "recipe": recipe, "include_official_test": False, "test_contract_count": 0}
    args.output_root = args.output_root.resolve()
    marker = {"study_id": STUDY_ID, "policy": POLICY,
              "configuration_sha256": executor.canonical_sha256(config),
              "canonical_seen_policy": CANONICAL_POLICY, "recipe": recipe}
    marker_path = args.output_root / "policy_provenance" / "run_identity.json"
    if args.output_root.exists() and not marker_path.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("Refusing an existing output root without this study identity")
    _write_immutable_json(executor, marker_path, marker)
    _publish(executor, args.output_root, [marker_path], "policy-run-identity", executor.canonical_sha256(marker))
    data = load_masked_scope(args, config, executor)
    executor._load_scope_data = lambda ignored: data
    _, validation, manifests, resource, folds = data
    fold = executor.representative_fold(folds)
    args.method = METHOD
    if args.phase == "prepare-scope":
        return {"status": "prepared", "scope_id": args.scope_id,
                "training_review_count": len(data[0]), "test_contract_count": 0}
    if args.phase == "train-candidate":
        if args.learning_rate not in RATES:
            raise ValueError("Choose one of the three original learning-rate candidates")
        expected = executor._training_contract(config=config, method_id=METHOD, scope_id=args.scope_id,
                                              manifests=manifests, learning_rate=args.learning_rate)
        base_join = executor.join_two_stage_scores
        def capture_join(*inputs, **kwargs):
            scored = base_join(*inputs, **kwargs)
            save_calibration(args, config, executor, scored, expected, fold)
            return scored
        executor.join_two_stage_scores = capture_join
        candidate_path = executor.candidate_result_path(args.output_root, METHOD, args.scope_id, args.learning_rate)
        if candidate_path.exists():
            validate_candidate(args, config, executor, manifests)
        executor.train_candidate(args, config)
        validate_candidate(args, config, executor, manifests)
        return {"status": "complete", "phase": args.phase, "scope_id": args.scope_id,
                "learning_rate": args.learning_rate, "failure_count": 0, "test_contract_count": 0}
    for rate in RATES:
        validate_candidate(argparse.Namespace(**{**vars(args), "learning_rate": rate}), config, executor, manifests)
    if args.phase == "select-scope":
        executor.select_scope(args)
        return {"status": "complete", "phase": args.phase, "scope_id": args.scope_id,
                "failure_count": 0, "test_contract_count": 0}
    scorer = CanonicalNDScorer(executor._score_runtime)
    executor._score_runtime = scorer
    canonical_grid = executor.build_aspect_grid(validation, fold.evaluation_aspects,
                                                executor.variant_map(fold, "N"), resource)
    base_resume = executor.two_stage_shard_resume_state
    def restore_resume(*inputs, **kwargs):
        state, artifact = base_resume(*inputs, **kwargs)
        contract = inputs[1]
        if state == "complete" and contract.condition == "N":
            scorer.restore(artifact, canonical_grid)
        return state, artifact
    executor.two_stage_shard_resume_state = restore_resume
    result = executor.score_selected(args, config)
    return {**result, "canonical_seen_policy": CANONICAL_POLICY,
            "new_prompt_rows": scorer.new_prompt_rows, "reused_prompt_rows": scorer.reused_prompt_rows}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--scope-id", default="heldout-a01", choices=tuple(l2_scopes()))
    parser.add_argument("--learning-rate", type=float, choices=RATES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    result = execute_phase(args, config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
