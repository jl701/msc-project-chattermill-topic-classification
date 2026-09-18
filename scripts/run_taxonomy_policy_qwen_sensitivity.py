"""Score missing exact few-shot prompts, then evaluate paired policy replays."""
from __future__ import annotations
import argparse
import gc
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from msc_project.data.fabsa import default_data_dir
from msc_project.experiments.taxonomy_resources import load_description_bundle
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID, IDENTITY, canonical_sha256, load_study, grids, select_seen, evaluate_and_save,
    sha256_file, write_json, receipt, verify_receipt,
)
from msc_project.experiments.taxonomy_two_stage_runtime import join_two_stage_scores
from msc_project.llm.qwen_pair_classifier import load_frozen_qwen_pair
from msc_project.llm.qwen_two_stage_classifier import (
    TwoStageDemonstration, score_two_stage_prompts, validate_sentiment_verbalizer_token_ids,
)
from campaign_taxonomy_training_policy_sensitivity import backup_job


def validate_probabilities(mode, values):
    array = np.asarray(values, dtype=float)
    width = 2 if mode == "aspect" else 3
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise ValueError("Non-finite or malformed Qwen probabilities")
    if (array < 0).any() or (array > 1).any() or not np.allclose(array.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid Qwen probability normalisation")
    return array


def load_chunks(root, preparation_hash):
    values = {}
    for path in sorted(root.glob("chunk-*.json")):
        value = json.loads(path.read_text())
        expected = value.pop("payload_sha256")
        if canonical_sha256(value) != expected or value["preparation_sha256"] != preparation_hash:
            raise ValueError("New few-shot cache integrity/resume conflict")
        probabilities = validate_probabilities(value["mode"], value["probabilities"])
        if len(probabilities) != len(value["keys"]):
            raise ValueError("Chunk identity length mismatch")
        for key, row in zip(value["keys"], probabilities):
            if key in values:
                raise ValueError("Duplicated inferred prompt identity")
            values[key] = row.tolist()
    return values


def infer(config, output, prepared):
    preparation_hash = sha256_file(prepared / "receipt.json")
    destination = output / "qwen_inference"
    destination.mkdir(parents=True, exist_ok=True)
    contract = {"study_id": STUDY_ID, "config_sha256": canonical_sha256(config),
                "preparation_sha256": preparation_hash, "runner_sha256": sha256_file(Path(__file__))}
    if (destination / "receipt.json").exists():
        verify_receipt(destination, contract)
        return
    if (destination / "FAILED.json").exists():
        raise RuntimeError("Existing inference failure requires audit")
    cache = load_chunks(destination, preparation_hash)
    pending = [json.loads(line) for line in (prepared / "pending.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = {}
    for row in pending:
        if row["key"] not in cache:
            groups.setdefault((row["mode"], row["demonstrations"]), []).append(row)
    demos = {k: tuple(TwoStageDemonstration(**d) for d in v) for k, v in
             json.loads((prepared / "demonstrations.json").read_text(encoding="utf-8")).items()}
    started = time.monotonic()
    model = None
    try:
        if groups:
            tokenizer, model, aspect_ids = load_frozen_qwen_pair(config["few_shot"]["model_id"],
                revision=config["few_shot"]["revision"], load_in_4bit=True, local_files_only=True)
            sentiment_ids = validate_sentiment_verbalizer_token_ids(tokenizer)
        chunk_index = len(list(destination.glob("chunk-*.json")))
        for (mode, digest), rows in groups.items():
            for offset in range(0, len(rows), 96):
                if time.monotonic() - started > config["few_shot"]["maximum_local_inference_hours"] * 3600:
                    raise RuntimeError("Registered local few-shot inference time budget reached")
                batch = rows[offset:offset + 96]
                values = score_two_stage_prompts(model, tokenizer,
                    [r["review"] for r in batch], [r["candidate"] for r in batch], mode=mode,
                    max_length=1024, batch_size=6, aspect_verbalizer_ids=aspect_ids,
                    sentiment_verbalizer_ids=sentiment_ids, demonstrations=demos[digest])
                values = validate_probabilities(mode, values)
                payload = {"mode": mode, "demonstrations": digest, "keys": [r["key"] for r in batch],
                           "probabilities": values.tolist(), "preparation_sha256": preparation_hash,
                           "test_contract_count": 0}
                payload["payload_sha256"] = canonical_sha256(payload)
                write_json(destination / f"chunk-{chunk_index:05d}.json", payload)
                for row, value in zip(batch, values):
                    cache[row["key"]] = value.tolist()
                chunk_index += 1
                write_json(destination / "state.json", {"status": "running", "completed_prompts": len(cache),
                    "planned_prompts": len(pending), "elapsed_seconds": time.monotonic() - started,
                    "updated_unix": time.time(), "failure_count": 0, "test_contract_count": 0})
                print(f"QWEN_CHUNK {chunk_index} complete={len(cache)}/{len(pending)}", flush=True)
        if set(cache) != {r["key"] for r in pending}:
            raise ValueError("Incomplete or extra inferred prompt identities")
        write_json(destination / "state.json", {"status": "complete", "completed_prompts": len(cache),
            "planned_prompts": len(pending), "elapsed_seconds": time.monotonic() - started,
            "failure_count": 0, "test_contract_count": 0})
        receipt(destination, contract)
    except BaseException as error:
        write_json(destination / "FAILED.json", {"error": repr(error), "at": time.time(), "contract": contract})
        raise
    finally:
        if model is not None:
            del model
            gc.collect()
            import torch
            torch.cuda.empty_cache()


def evaluate(config, validation, folds, resource, output, prepared):
    preparation_hash = sha256_file(prepared / "receipt.json")
    verify_receipt(output / "qwen_inference")
    inferred = load_chunks(output / "qwen_inference", preparation_hash)
    db = sqlite3.connect((prepared / "source_cache.sqlite").as_uri() + "?mode=ro", uri=True)
    old_thresholds = json.loads((prepared / "source_thresholds.json").read_text())
    backup = ROOT.parent / "cloud_backups" / STUDY_ID
    try:
        for fold in folds:
            for policy in config["policies"]:
                destination = output / "fewshot" / policy / fold.fold_id
                contract = {"study_id": STUDY_ID, "config_sha256": canonical_sha256(config), "policy": policy,
                            "fold_id": fold.fold_id, "method": "fewshot", "preparation_sha256": preparation_hash,
                            "inference_receipt_sha256": sha256_file(output / "qwen_inference/receipt.json")}
                frames = {}
                for condition in ("D", "N"):
                    ag, sg = grids(validation, fold, resource, condition)
                    query = pd.read_csv(prepared / "queries" / policy / fold.fold_id / f"{condition}.csv")
                    if not ag[["row_uid", "candidate_aspect"]].equals(query[["row_uid", "candidate_aspect"]]):
                        raise ValueError("Qwen query order/identity mismatch")
                    probabilities = {}
                    for mode in ("aspect", "sentiment"):
                        values = []
                        for key in query[mode + "_key"]:
                            source = db.execute("SELECT value FROM scores WHERE key=?", (key,)).fetchone()
                            if source is None and key not in inferred:
                                raise ValueError("Missing exact query key")
                            values.append(json.loads(source[0]) if source is not None else inferred[key])
                        probabilities[mode] = validate_probabilities(mode, values)
                    frames[condition] = join_two_stage_scores(ag, sg, probabilities["aspect"][:, 0],
                                                              probabilities["sentiment"].reshape(-1))
                selection = select_seen(frames["D"].loc[frames["D"].candidate_aspect.isin(fold.seen_aspects)], fold)
                destination.mkdir(parents=True, exist_ok=True)
                if not (destination / "receipt.json").exists():
                    write_json(destination / "selection.json", selection)
                    for condition in ("D", "N"):
                        evaluate_and_save(frames[condition], fold, selection, destination, condition)
                    receipt(destination, contract)
                else:
                    verify_receipt(destination, contract)
                backup_job(destination, backup / "fewshot" / policy / fold.fold_id)
                mixed_root = output / "mixed_component_diagnostic" / policy / fold.fold_id
                mixed_selection = {"aspect_threshold": selection["aspect_threshold"],
                                   "second_sentiment_threshold": old_thresholds[f"qwen_candidate_pair_qlora/{fold.fold_id}"]["runner_up_sentiment"]}
                mixed_contract = {**contract, "method": "mixed_component_diagnostic", "no_joint_retuning": True,
                                  "stage_2": "original_review_filtered_qlora", "thresholds": mixed_selection}
                if not (mixed_root / "receipt.json").exists():
                    mixed_root.mkdir(parents=True, exist_ok=True)
                    write_json(mixed_root / "selection.json", mixed_selection)
                    for condition in ("D", "N"):
                        qlora = pd.read_csv(prepared / "sources/qwen_candidate_pair_qlora" / fold.fold_id / f"{condition}.csv.gz",
                                            float_precision="round_trip").sort_values(IDENTITY).reset_index(drop=True)
                        mixed = frames[condition].sort_values(IDENTITY).reset_index(drop=True).copy()
                        if not mixed[IDENTITY].equals(qlora[IDENTITY]) or not np.array_equal(mixed.target, qlora.target):
                            raise ValueError("Fixed QLoRA source grid mismatch")
                        mixed["sentiment_score"] = qlora.sentiment_score.to_numpy()
                        evaluate_and_save(mixed, fold, mixed_selection, mixed_root, condition)
                    receipt(mixed_root, mixed_contract)
                else:
                    verify_receipt(mixed_root, mixed_contract)
                backup_job(mixed_root, backup / "mixed_component_diagnostic" / policy / fold.fold_id)
                print(f"QWEN_EVALUATED {fold.fold_id} {policy}", flush=True)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["infer", "evaluate"], required=True)
    args = parser.parse_args()
    config, original, validation, folds = load_study(ROOT / f"configs/experiments/{STUDY_ID}.json", default_data_dir())
    output = ROOT / config["output_root"]
    prepared = output / "qwen_prepared"
    verify_receipt(prepared)
    if args.phase == "infer":
        infer(config, output, prepared)
    else:
        evaluate(config, validation, folds, load_description_bundle(require_approved=True), output, prepared)


if __name__ == "__main__":
    main()
