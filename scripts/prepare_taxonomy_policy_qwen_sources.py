"""Verify validation artifacts and enumerate exact frozen-Qwen prompt reuse."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import analyse_taxonomy_two_stage_global_router as router
import analyse_taxonomy_post_supervisor_formal_v2 as legacy
from msc_project.data.fabsa import default_data_dir
from msc_project.experiments.taxonomy_resources import load_description_bundle
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID, IDENTITY, canonical_sha256, load_study, grids, sha256_file, write_json, receipt,
)
from msc_project.llm.qwen_two_stage_classifier import (
    TwoStageDemonstration, demonstrations_sha256, render_two_stage_few_shot_prompt,
    qwen_two_stage_few_shot_contract_sha256,
)

FEWSHOT = "frozen_qwen_few_shot"
QLORA = "qwen_candidate_pair_qlora"


def source_receipts(root):
    """Read receipt identities. Hash only the explicitly selected source artifacts."""
    published = {}
    for worker in legacy.worker_roots(root).values():
        for path in sorted((worker / "_sync/received").glob("*.json")):
            value = json.loads(path.read_text())
            if value.get("schema_version") != "verified_artifact_unit_v1" or value.get("protocol_id") != legacy.PROTOCOL_ID:
                raise ValueError("Invalid source receipt identity")
            for entry in value["files"]:
                file = (worker / entry["path"]).resolve()
                if not file.is_relative_to(worker.resolve()):
                    raise ValueError("Source receipt path escape")
                if file in published and published[file] != entry["sha256"]:
                    raise ValueError("Conflicting source receipt hashes")
                published[file] = entry["sha256"]
    return published


def main():
    from transformers import AutoTokenizer
    config_path = ROOT / f"configs/experiments/{STUDY_ID}.json"
    config, original, validation, folds = load_study(config_path, default_data_dir())
    output = ROOT / config["output_root"]
    prepared = output / "qwen_prepared"
    if prepared.exists():
        raise FileExistsError("Qwen preparation is immutable. Audit any existing directory before resuming.")
    prepared.mkdir(parents=True)
    write_json(prepared / "RUNNING.json", {"status": "preparing", "config_sha256": canonical_sha256(config)})
    resource = load_description_bundle(require_approved=True)
    tokenizer = AutoTokenizer.from_pretrained(config["few_shot"]["model_id"],
                                              revision=config["few_shot"]["revision"], local_files_only=True)
    prompt_contract = qwen_two_stage_few_shot_contract_sha256(max_length=1024)
    demos_by_hash, demo_ids = {}, {}
    for fold in folds:
        for policy in config["policies"]:
            raw = json.loads((output / "data_audit" / fold.fold_id / policy / "demonstrations.json").read_text(encoding="utf-8"))
            for mode, values in raw.items():
                demos = tuple(TwoStageDemonstration(**v) for v in values)
                if any(d.candidate_aspect in fold.heldout_aspects for d in demos):
                    raise ValueError("Held-out demonstration candidate")
                digest = demonstrations_sha256(demos)
                demos_by_hash[digest] = demos
                demo_ids[(fold.fold_id, policy, mode)] = digest
    computational_contract = {"model_id": config["few_shot"]["model_id"], "revision": config["few_shot"]["revision"],
                              "prompt_contract": prompt_contract, "max_length": 1024, "batch_size": 6,
                              "quantization": "NF4_double_quant_float16_compute", "load_in_4bit": True}

    @lru_cache(maxsize=180000)
    def key(mode, digest, review, candidate):
        rendered = render_two_stage_few_shot_prompt(tokenizer, review, candidate, mode=mode,
                                                   demonstrations=demos_by_hash[digest])
        return canonical_sha256({**computational_contract, "mode": mode, "demonstrations": digest,
                                 "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest()})

    backup = ROOT.parent / "cloud_backups/taxonomy_two_stage_formal_v2_r2"
    published = source_receipts(backup)
    entries = router._entry_map(backup)
    verified = {}
    db = sqlite3.connect(prepared / "source_cache.sqlite")
    db.execute("CREATE TABLE scores (key TEXT PRIMARY KEY, mode TEXT NOT NULL, value TEXT NOT NULL, origin TEXT NOT NULL)")
    selection_records, repairs = {}, {}
    for fold in folds:
        for method in (FEWSHOT, QLORA):
            scope = "heldout-a" + fold.fold_id[-2:]
            selection_paths = list(backup.glob(f"worker-*/selections/{method}/{scope}.json"))
            if len(selection_paths) != 1:
                raise ValueError("Source selection is missing or duplicated")
            spath = selection_paths[0]
            selection = json.loads(spath.read_text())
            legacy.validate_sealed(selection, "selection_payload_sha256", spath)
            if selection.get("failure_count") != 0 or selection.get("test_contract_count") != 0:
                raise ValueError("Unsafe source selection")
            thresholds = selection.get("selected_thresholds", selection.get("thresholds"))
            selection_records[f"{method}/{fold.fold_id}"] = thresholds
            if method == FEWSHOT:
                contract = selection["scope_contract"]
                for field, expected in (("model_id", computational_contract["model_id"]),
                                        ("model_revision", computational_contract["revision"]),
                                        ("prompt_contract_sha256", prompt_contract), ("maximum_length", 1024), ("batch_size", 6)):
                    if contract.get(field) != expected:
                        raise ValueError("Few-shot computation contract mismatch: " + field)
                for mode in ("aspect", "sentiment"):
                    if selection["demonstration_sha256s"][mode] != demo_ids[(fold.fold_id, "review_filtered", mode)]:
                        raise ValueError("Original demonstration reconstruction mismatch")
            for condition in ("D", "N"):
                worker, result_path, result = entries[(method, fold.fold_id, condition)]
                for path in router._critical_score_paths(worker, result_path, result, spath):
                    if path.resolve() not in published or sha256_file(path) != published[path.resolve()]:
                        raise ValueError("Missing or mismatched receipt for " + str(path))
                    verified[str(path)] = published[path.resolve()]
                scored = router._load_frame(entries, method=method, fold=fold.fold_id,
                                           condition=condition, repair_audit=repairs).sort_values(IDENTITY).reset_index(drop=True)
                ag, sg = grids(validation, fold, resource, condition)
                expected = sg.sort_values(IDENTITY).reset_index(drop=True)
                if not scored[IDENTITY].equals(expected[IDENTITY]) or not np.array_equal(scored.target, expected.target):
                    raise ValueError("Source row/grid/target alignment mismatch")
                saved = prepared / "sources" / method / fold.fold_id
                saved.mkdir(parents=True, exist_ok=True)
                scored.to_csv(saved / f"{condition}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
                if method == FEWSHOT:
                    ordered_aspects = scored[["row_uid", "candidate_aspect"]].drop_duplicates().reset_index(drop=True)
                    if not ordered_aspects.equals(ag[["row_uid", "candidate_aspect"]].reset_index(drop=True)):
                        raise ValueError("Aspect/sentiment source matrix ordering mismatch")
                    aspect_values = scored.groupby(["row_uid", "candidate_aspect"], sort=True).aspect_score.first().to_numpy()
                    sentiment_values = scored.sentiment_score.to_numpy().reshape(-1, 3)
                    for mode in ("aspect", "sentiment"):
                        digest = demo_ids[(fold.fold_id, "review_filtered", mode)]
                        insert = []
                        for index, row in enumerate(ag.itertuples(index=False)):
                            values = [float(aspect_values[index]), float(1 - aspect_values[index])] if mode == "aspect" else sentiment_values[index].tolist()
                            cache_key = key(mode, digest, str(row.text), str(row.candidate_text))
                            insert.append((cache_key, mode, json.dumps(values), f"{fold.fold_id}/{condition}/{verified[str(result_path)]}"))
                        # Fixed pre-outcome source priority: ascending fold ID, D before N.
                        # Reuse one exact identified source, never average or choose by score.
                        db.executemany("INSERT OR IGNORE INTO scores VALUES (?,?,?,?)", insert)
                    db.commit()
            print(f"VERIFIED_SOURCE {method} {fold.fold_id}", flush=True)
    pending, reuse_counts = {}, []
    for fold in folds:
        for policy in config["policies"]:
            for condition in ("D", "N"):
                ag, _ = grids(validation, fold, resource, condition)
                columns = {"row_uid": ag.row_uid.tolist(), "candidate_aspect": ag.candidate_aspect.tolist()}
                for mode in ("aspect", "sentiment"):
                    digest = demo_ids[(fold.fold_id, policy, mode)]
                    keys, reused = [], 0
                    for row in ag.itertuples(index=False):
                        cache_key = key(mode, digest, str(row.text), str(row.candidate_text))
                        keys.append(cache_key)
                        if db.execute("SELECT 1 FROM scores WHERE key=?", (cache_key,)).fetchone():
                            reused += 1
                        elif cache_key not in pending:
                            pending[cache_key] = {"key": cache_key, "mode": mode, "demonstrations": digest,
                                                  "review": str(row.text), "candidate": str(row.candidate_text)}
                    columns[mode + "_key"] = keys
                    reuse_counts.append({"fold_id": fold.fold_id, "policy": policy, "condition": condition,
                                         "mode": mode, "queries": len(keys), "reused_queries": reused})
                destination = prepared / "queries" / policy / fold.fold_id
                destination.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(columns).to_csv(destination / f"{condition}.csv", index=False)
    db.close()
    with (prepared / "pending.jsonl").open("w", encoding="utf-8") as stream:
        for item in pending.values():
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_json(prepared / "demonstrations.json", {k: [d.as_hash_payload() for d in v] for k, v in demos_by_hash.items()})
    write_json(prepared / "source_thresholds.json", selection_records)
    write_json(prepared / "original_canonicalisation_provenance.json", repairs)
    write_json(prepared / "verified_source_files.json", verified)
    pd.DataFrame(reuse_counts).to_csv(prepared / "reuse_counts.csv", index=False)
    summary = {"status": "prepared", "config_sha256": canonical_sha256(config), "computational_contract": computational_contract,
               "unique_new_prompts": len(pending), "verified_source_files": len(verified),
               "source_priority": "ascending fold, D before N, independent of numerical outcomes",
               "source_role": "canonical exact-source replay for paired policy comparison, historical main results remain untouched",
               "failure_count": 0, "test_contract_count": 0}
    write_json(prepared / "summary.json", summary)
    receipt(prepared, {"study_id": STUDY_ID, "config_sha256": canonical_sha256(config),
                       "preparer_sha256": sha256_file(Path(__file__))})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
