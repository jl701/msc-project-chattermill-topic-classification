"""Execute only the preregistered local construction-sensitivity study."""
from __future__ import annotations
import argparse
import gc
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from msc_project.data.fabsa import default_data_dir
from msc_project.experiments.taxonomy_execution import canonical_sha256
from msc_project.experiments.taxonomy_resources import load_description_bundle
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID, IDENTITY, load_study, training_rows, train_pool_hash, grids, score,
    select_seen, evaluate_and_save, write_json, sha256_file, receipt, verify_receipt,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (
    TfidfTrueTwoStageRuntime, build_aspect_grid, build_sentiment_grid,
    select_qwen_two_stage_demonstrations,
)
from msc_project.experiments.taxonomy_two_stage_training import (
    build_two_stage_training_manifests, training_manifest_sha256,
)
from msc_project.llm.qwen_two_stage_classifier import demonstrations_sha256


def audit(config, original, validation, folds, resource, output):
    records, demos, manifest_records = [], [], []
    for fold in folds:
        filtered = training_rows(original, fold, "review_filtered")
        masked = training_rows(original, fold, "label_masked_all_reviews")
        removed = original.loc[~original.row_uid.isin(filtered.row_uid)]
        seen = set(fold.seen_aspects)
        lost = sum(sum(a in seen for a, _ in labels) for labels in removed.labels)
        target_only = sum(not any(a in seen for a, _ in labels) for labels in removed.labels)
        records.append({"fold_id": fold.fold_id, "heldout_aspect": fold.heldout_aspects[0],
                        "original_reviews": len(original), "filtered_reviews": len(filtered),
                        "masked_reviews": len(masked), "removed_reviews": len(removed),
                        "removed_fraction": len(removed) / len(original),
                        "lost_seen_pair_annotations": lost, "heldout_only_reviews_retained_by_masking": target_only})
        previous_manifests = None
        previous_demos = None
        for policy, train in (("review_filtered", filtered), ("label_masked_all_reviews", masked)):
            destination = output / "data_audit" / fold.fold_id / policy
            destination.mkdir(parents=True, exist_ok=True)
            train.to_json(destination / "training_pool.jsonl", orient="records", lines=True, force_ascii=False)
            manifests = build_two_stage_training_manifests(train, fold.seen_aspects, resource, seed=13)
            selected = select_qwen_two_stage_demonstrations(train, fold.seen_aspects, resource, seed=13)
            for task, manifest in manifests.items():
                if set(manifest.candidate_aspect) - seen or len(manifest) != 2048:
                    raise AssertionError("Training manifest scope or budget mismatch")
                manifest.to_json(destination / f"{task}.jsonl", orient="records", lines=True, force_ascii=False)
                keys = ["row_uid", "candidate_aspect", "candidate_sentiment"]
                identities = set(map(tuple, manifest[keys].astype(str).to_numpy()))
                old = set(map(tuple, previous_manifests[task][keys].astype(str).to_numpy())) if previous_manifests else identities
                manifest_records.append({"fold_id": fold.fold_id, "policy": policy, "task": task,
                                         "examples": len(manifest), "unique_reviews": manifest.row_uid.nunique(),
                                         "replaced_instances_vs_filtered": len(identities - old),
                                         "sha256": training_manifest_sha256(manifest)})
            write_json(destination / "demonstrations.json", {mode: [d.as_hash_payload() for d in values]
                                                              for mode, values in selected.items()})
            for mode, values in selected.items():
                digest = demonstrations_sha256(values)
                demos.append({"fold_id": fold.fold_id, "policy": policy, "stage": mode, "count": len(values),
                              "demonstrations_sha256": digest,
                              "changed_vs_filtered": bool(previous_demos and digest != demonstrations_sha256(previous_demos[mode]))})
            previous_manifests, previous_demos = manifests, selected
            write_json(destination / "contract.json", {"pool_sha256": train_pool_hash(train), "policy": policy,
                                                       "fold": asdict(fold), "config_sha256": canonical_sha256(config)})
        print(f"AUDIT {fold.fold_id} removed={len(removed)} lost_seen_annotations={lost}", flush=True)
    report = ROOT / config["report_root"]
    report.mkdir(parents=True, exist_ok=True)
    for name, values in (("data_impact", records), ("demonstration_changes", demos), ("training_manifest_changes", manifest_records)):
        pd.DataFrame(values).to_csv(report / f"{name}.csv", index=False)
        pd.DataFrame(values).to_csv(output / f"{name}.csv", index=False)
    write_json(output / "data_audit_summary.json", {"folds": 12, "failure_count": 0, "test_contract_count": 0,
                                                   "changed_demo_stages": sum(d["changed_vs_filtered"] for d in demos)})


def selected_candidate(candidates):
    return max(candidates, key=lambda c: (c["selection"]["selection_metrics"]["pair_micro_f1"],
                                         c["selection"]["selection_metrics"]["pair_samples_f1"],
                                         c["selection"]["selection_metrics"]["pair_micro_precision"],
                                         -c["learning_rate"]))


def run_fold(args, config, original, validation, fold, resource):
    train = training_rows(original, fold, args.policy, args.subset_seed)
    policy_name = args.policy + (f"_s{args.subset_seed}" if args.subset_seed is not None else "")
    destination = args.output_root / args.method / policy_name / fold.fold_id
    contract = {"study_id": STUDY_ID, "config_sha256": canonical_sha256(config), "method": args.method,
                "policy": policy_name, "fold_id": fold.fold_id, "training_pool_sha256": train_pool_hash(train),
                "runner_sha256": sha256_file(Path(__file__)),
                "primitive_sha256": sha256_file(ROOT / "src/msc_project/experiments/taxonomy_training_policy_sensitivity.py")}
    if (destination / "receipt.json").exists():
        verify_receipt(destination, contract)
        print(f"VERIFIED_RESUME {args.method} {policy_name} {fold.fold_id}", flush=True)
        return
    if (destination / "FAILED.json").exists():
        raise RuntimeError("Existing failed job requires explicit audited recovery")
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(destination / "RUNNING.json", {"pid": os.getpid(), "contract": contract, "started_unix": time.time()})
    calibration_ag, calibration_sg = grids(validation, fold, resource, "D", seen_only=True)
    runtime = None
    try:
        if args.method == "tfidf":
            import joblib
            variants = {a: "name_and_description" for a in fold.seen_aspects}
            runtime = TfidfTrueTwoStageRuntime(seed=13).fit(
                build_aspect_grid(train, fold.seen_aspects, variants, resource),
                build_sentiment_grid(train, fold.seen_aspects, variants, resource, gold_aspects_only=True))
            selection_scored = score(runtime, "tfidf", calibration_ag, calibration_sg)
            selection = select_seen(selection_scored, fold)
            joblib.dump(runtime, destination / "tfidf_runtime.joblib", compress=3)
        elif args.method == "distilbert":
            import torch
            from msc_project.experiments.taxonomy_two_stage_distilbert import (
                DistilBertTrueTwoStageRuntime, TwoStageDistilBertConfig, _set_seed)
            if not torch.cuda.is_available():
                raise RuntimeError("DistilBERT sensitivity requires CUDA")
            manifests = build_two_stage_training_manifests(train, fold.seen_aspects, resource, seed=13)
            candidates = []
            for rate in config["distilbert"]["learning_rates"]:
                candidate_dir = destination / "candidates" / f"lr-{rate:.0e}"
                candidate_contract = {**contract, "learning_rate": rate}
                if (candidate_dir / "receipt.json").exists():
                    verify_receipt(candidate_dir, candidate_contract)
                    candidate = json.loads((candidate_dir / "selection.json").read_text())
                else:
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    _set_seed(13)
                    runtime = DistilBertTrueTwoStageRuntime(TwoStageDistilBertConfig(learning_rate=rate),
                                                          device=torch.device("cuda"), local_files_only=True)
                    print(f"TRAIN {fold.fold_id} {policy_name} lr={rate}", flush=True)
                    runtime.fit(manifests["aspect_presence"], manifests["sentiment"])
                    selection_scored = score(runtime, "distilbert", calibration_ag, calibration_sg)
                    candidate = {"learning_rate": rate, "selection": select_seen(selection_scored, fold),
                                 "history": runtime.history, "config": asdict(runtime.config)}
                    runtime.save_pretrained(candidate_dir / "checkpoint")
                    selection_scored.to_csv(candidate_dir / "seen_scores.csv.gz", index=False,
                                            compression={"method": "gzip", "mtime": 0})
                    write_json(candidate_dir / "selection.json", candidate)
                    receipt(candidate_dir, candidate_contract)
                    runtime.close()
                    runtime = None
                    gc.collect()
                    torch.cuda.empty_cache()
                candidates.append(candidate)
            chosen = selected_candidate(candidates)
            selection = chosen["selection"]
            selected_path = destination / "candidates" / f"lr-{chosen['learning_rate']:.0e}"
            selection_scored = pd.read_csv(selected_path / "seen_scores.csv.gz")
            runtime = DistilBertTrueTwoStageRuntime.from_pretrained(selected_path / "checkpoint", device=torch.device("cuda"))
            write_json(destination / "model_selection.json", chosen)
        else:
            raise ValueError("Unsupported core method")
        write_json(destination / "selection.json", selection)
        seen_digest = None
        for condition in ("D", "N"):
            # Shared seen grid is scored once. Only the held-out candidate changes.
            ag, sg = grids(validation, fold, resource, condition)
            target_ag = ag[ag.candidate_aspect.isin(fold.heldout_aspects)].copy()
            target_sg = sg[sg.candidate_aspect.isin(fold.heldout_aspects)].copy()
            target = score(runtime, args.method, target_ag, target_sg)
            full = pd.concat([selection_scored, target], ignore_index=True).sort_values(IDENTITY).reset_index(drop=True)
            observed = canonical_sha256(full.loc[full.candidate_aspect.isin(fold.seen_aspects),
                                                IDENTITY + ["aspect_score", "sentiment_score"]].to_dict("records"))
            if seen_digest is not None and observed != seen_digest:
                raise AssertionError("N/D identical seen candidates changed scores")
            seen_digest = observed
            evaluate_and_save(full, fold, selection, destination, condition)
        write_json(destination / "runtime.json", {"elapsed_seconds": time.monotonic() - started,
                                                  "pid": os.getpid(), "python": sys.version,
                                                  "platform": platform.platform(), "seen_ND_sha256": seen_digest})
        receipt(destination, contract)
        print(f"COMPLETE {args.method} {policy_name} {fold.fold_id} seconds={time.monotonic()-started:.1f}", flush=True)
    except BaseException as error:
        write_json(destination / "FAILED.json", {"error": repr(error), "traceback": traceback.format_exc(),
                                                 "contract": contract, "at": time.time()})
        raise
    finally:
        if runtime is not None and hasattr(runtime, "close"):
            runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit", "run"], required=True)
    parser.add_argument("--method", choices=["tfidf", "distilbert"])
    parser.add_argument("--policy", choices=["review_filtered", "label_masked_all_reviews", "size_matched_label_masked"])
    parser.add_argument("--subset-seed", type=int)
    parser.add_argument("--fold-id")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments" / f"{STUDY_ID}.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experimental" / STUDY_ID)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = parser.parse_args()
    config, original, validation, folds = load_study(args.config, args.data_dir)
    resource = load_description_bundle(require_approved=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "audit":
        audit(config, original, validation, folds, resource, args.output_root)
    else:
        if not args.method or not args.policy:
            parser.error("run requires --method and --policy")
        if args.method != "tfidf" and args.policy == "size_matched_label_masked":
            parser.error("Size controls are TF-IDF only")
        selected = [f for f in folds if not args.fold_id or f.fold_id == args.fold_id]
        if not selected:
            parser.error("Unregistered fold")
        for fold in selected:
            run_fold(args, config, original, validation, fold, resource)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
