"""CPU-only DCWT paired replay and exact frozen-method invariance audit."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_taxonomy_description_weight_transfer as legacy
from campaign_taxonomy_training_policy_sensitivity import backup_job
from msc_project.data.fabsa import default_data_dir
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID as PARENT_ID, IDENTITY, canonical_sha256, load_study, training_rows,
    train_pool_hash, grids, evaluate_and_save, receipt, verify_receipt, write_json, sha256_file,
)

EXT_ID = "taxonomy_training_policy_extension_v1"
CONFIG = ROOT / f"configs/experiments/{EXT_ID}.json"


def load_extension():
    config = json.loads(CONFIG.read_text())
    if (config["study_id"] != EXT_ID or config["allowed_splits"] != ["train", "validation"]
            or config["include_official_test"] is not False or config["test_contract_count"] != 0):
        raise ValueError("Forbidden extension boundary")
    checks = {ROOT / f"configs/experiments/{PARENT_ID}.json": config["parent_config_sha256"],
              ROOT / config["dcwt"]["recipe"]: config["dcwt"]["recipe_sha256"],
              ROOT / "scripts/run_taxonomy_description_weight_transfer.py": config["dcwt"]["reference_implementation_sha256"],
              ROOT / config["invariance"]["source_audit"]: config["invariance"]["source_audit_sha256"]}
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError("Extension authority hash conflict: " + str(path))
    parent, train, validation, folds = load_study(ROOT / f"configs/experiments/{PARENT_ID}.json", default_data_dir())
    if config["folds"] != [fold.fold_id for fold in folds] or config["policies"] != parent["policies"]:
        raise ValueError("Extension fold/policy scope changed")
    return config, train, validation, folds, legacy.load_description_bundle(require_approved=True)


def contract(config, method, policy=None, fold=None, **extra):
    return {"study_id": EXT_ID, "config_sha256": canonical_sha256(config), "method": method,
            "policy": policy, "fold_id": fold, "runner_sha256": sha256_file(Path(__file__)), **extra}


def prepare(config, resource, output):
    destination = output / "descriptor_cache"
    identity = contract(config, "frozen_dcwt_descriptors")
    if (destination / "receipt.json").exists():
        verify_receipt(destination, identity)
        return
    if destination.exists():
        raise RuntimeError("Incomplete descriptor preparation requires inspection")
    destination.mkdir(parents=True)
    aspects = tuple(legacy.canonical_aspects())
    texts = sorted({legacy.render_aspect_candidate(a, v, resource) for a in aspects
                    for v in ("name_only", "name_and_description")})
    encoder = legacy.FrozenTransformerSentenceEncoder(legacy.REGISTERED_CONFIGS["e5_base_v2"],
                                                      device="cpu", local_files_only=True)
    try:
        matrix = encoder.encode(texts, role="candidate")
    finally:
        encoder.close()
    if len(matrix) != len(texts) or not np.isfinite(matrix).all():
        raise ValueError("Invalid frozen description cache")
    np.savez_compressed(destination / "embeddings.npz", vectors=matrix)
    write_json(destination / "inputs.json", {"texts": texts, "encoder": asdict(legacy.REGISTERED_CONFIGS["e5_base_v2"]),
                                             "device": "cpu", "resource_sha256": canonical_sha256(resource)})
    receipt(destination, identity)
    backup_job(destination, ROOT.parent / "cloud_backups" / EXT_ID / "descriptor_cache")
    print(f"DESCRIPTORS_VERIFIED {len(texts)} frozen inputs, no GPU", flush=True)


def fit_spaces(train, validation, components):
    """Exact legacy feature algorithm, with fitted objects retained for recovery."""
    vectoriser = legacy.FeatureUnion([
        ("word", legacy.TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), min_df=2,
                                       max_features=30000, sublinear_tf=True)),
        ("character", legacy.TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                            max_features=30000, sublinear_tf=True)),
    ])
    train_sparse = vectoriser.fit_transform(train.text.astype(str).tolist())
    val_sparse = vectoriser.transform(validation.text.astype(str).tolist())
    values, fitted = {}, {"vectoriser": vectoriser, "transforms": {}}
    for requested in components:
        effective = min(int(requested), train_sparse.shape[1] - 1)
        if effective < 2:
            raise ValueError("Feature space too small")
        projector = legacy.TruncatedSVD(n_components=effective, algorithm="randomized", random_state=13)
        a = projector.fit_transform(train_sparse)
        b = projector.transform(val_sparse)
        scaler = legacy.StandardScaler().fit(a)
        values[requested] = (scaler.transform(a), scaler.transform(b),
                             {"effective_components": effective, "tfidf_features": train_sparse.shape[1],
                              "explained_variance_ratio_sum": float(projector.explained_variance_ratio_.sum())})
        fitted["transforms"][requested] = {"projector": projector, "scaler": scaler}
    return values, fitted


def dcwt_fold(config, original, validation, fold, resource, output, policy):
    destination = output / "dcwt" / policy / fold.fold_id
    train = training_rows(original, fold, policy).sort_values("row_uid", kind="stable").reset_index(drop=True)
    validation = validation.sort_values("row_uid", kind="stable").reset_index(drop=True)
    identity = contract(config, "dcwt", policy, fold.fold_id, training_pool_sha256=train_pool_hash(train))
    if (destination / "receipt.json").exists():
        verify_receipt(destination, identity)
        print("VERIFIED_DCWT_RESUME", policy, fold.fold_id, flush=True)
        return
    if destination.exists():
        raise RuntimeError("Partial DCWT fold requires inspection, not automatic overwrite")
    destination.mkdir(parents=True)
    started = time.monotonic()
    try:
        write_json(destination / "RUNNING.json", identity)
        train.to_json(destination / "training_rows.jsonl", orient="records", lines=True, force_ascii=False)
        recipe = json.loads((ROOT / config["dcwt"]["recipe"]).read_text())
        cached = output / "descriptor_cache"
        verify_receipt(cached, contract(config, "frozen_dcwt_descriptors"))
        text_list = json.loads((cached / "inputs.json").read_text())["texts"]
        with np.load(cached / "embeddings.npz", allow_pickle=False) as archive:
            embeddings = dict(zip(text_list, archive["vectors"]))
        seen = tuple(fold.seen_aspects)
        descriptors = np.stack([embeddings[legacy.render_aspect_candidate(a, "name_and_description", resource)] for a in seen])
        components = tuple(recipe["stage_1"]["review_representation"]["svd_components_grid"])
        c_grid = recipe["stage_1"]["seen_aspect_classifiers"]["c_grid"]
        alphas = recipe["stage_1"]["primary_generator"]["ridge_alpha_grid"]
        temperatures = recipe["stage_1"]["mandatory_transfer_baselines"][2]["temperature_grid"]
        target_train = legacy._labels_matrix(train, seen)
        target_validation = legacy._labels_matrix(validation, seen)
        spaces, fitted = fit_spaces(train, validation, components)
        candidates = {family: [] for family in legacy.GENERATOR_FAMILIES}
        weight_cache = {}
        for component_count in components:
            features, val_features, diagnostics = spaces[component_count]
            for classifier_c in c_grid:
                weights = legacy._fit_seen_weights(features, target_train, classifier_c=classifier_c)
                weight_cache[(component_count, classifier_c)] = weights
                options = {"kernel_ridge": alphas, "mean_seen_weight": [None],
                           "nearest_description_weight": [None], "cosine_barycentric_weight": temperatures}
                for family, parameters in options.items():
                    for parameter in parameters:
                        kwargs = {"alpha": parameter} if family == "kernel_ridge" else (
                            {"temperature": parameter} if family == "cosine_barycentric_weight" else {})
                        targets, scores, direction = legacy.pseudo_unseen_scores(descriptors, weights, val_features,
                            target_validation, family=family, **kwargs)
                        chosen_threshold = legacy.select_presence_threshold(targets, scores)
                        candidates[family].append({
                            "configuration_id": legacy._configuration_id(family, component_count, classifier_c, parameter),
                            "family": family, "svd_components": component_count, "classifier_c": classifier_c,
                            "generator_parameter": parameter, "aspect_threshold": chosen_threshold.threshold,
                            "pseudo_unseen_f1": chosen_threshold.f1,
                            "pseudo_unseen_average_precision": chosen_threshold.average_precision,
                            "pseudo_unseen_precision": chosen_threshold.precision, "pseudo_unseen_recall": chosen_threshold.recall,
                            "parameter_direction_cosine": direction,
                        })
                print(f"DCWT_FIT {fold.fold_id}/{policy} svd={component_count} C={classifier_c}", flush=True)
        selected = {family: max(records, key=legacy._rank_configuration) for family, records in candidates.items()}
        variants = {a: "name_and_description" for a in seen}
        sentiment = legacy.UnifiedTfidfPairScorer(legacy.UnifiedTfidfPairConfig(
            classifier_c=1.0, feature_ablation="all_six", max_iter=1000, seed=13)).fit(
                legacy.build_sentiment_grid(train, seen, variants, resource, gold_aspects_only=True))
        fitted.update(weights=weight_cache, selected=selected, seen_aspects=seen, sentiment_scorer=sentiment)
        joblib.dump(fitted, destination / "fitted_components.joblib", compress=3)
        write_json(destination / "all_candidate_selections.json", candidates)
        write_json(destination / "selected_families.json", selected)
        selection_ag, selection_sg = grids(validation, fold, resource, "D", seen_only=True)
        seen_sentiment = sentiment.score_manifest(selection_sg)
        full_grids = {condition: grids(validation, fold, resource, condition) for condition in ("D", "N")}
        target_sentiments = {condition: sentiment.score_manifest(sg.loc[sg.candidate_aspect.isin(fold.heldout_aspects)])
                             for condition, (ag, sg) in full_grids.items()}
        for family, selected_config in selected.items():
            weights = weight_cache[(selected_config["svd_components"], selected_config["classifier_c"])]
            val_features = spaces[selected_config["svd_components"]][1]
            direct_seen = np.column_stack([legacy.sigmoid(val_features @ w[:-1] + w[-1]) for w in weights])
            seen_scored = legacy.join_two_stage_scores(selection_ag, selection_sg, direct_seen.reshape(-1), seen_sentiment)
            aspect_threshold = float(selected_config["aspect_threshold"])
            second = legacy.select_second_sentiment_threshold(seen_scored, aspect_threshold=aspect_threshold)
            selection = {"aspect_threshold": aspect_threshold,
                         "second_sentiment_threshold": float(second.second_sentiment_threshold),
                         "selected_configuration": selected_config, "second_selection": asdict(second)}
            family_root = destination / "generators" / family
            write_json(family_root / "selection.json", selection)
            parameter = selected_config["generator_parameter"]
            kwargs = {"alpha": parameter} if family == "kernel_ridge" else (
                {"temperature": parameter} if family == "cosine_barycentric_weight" else {})
            seen_hash = None
            for condition in ("D", "N"):
                ag, sg = full_grids[condition]
                ag = ag.loc[ag.candidate_aspect.isin(fold.heldout_aspects)]
                sg = sg.loc[sg.candidate_aspect.isin(fold.heldout_aspects)]
                descriptor_text = legacy.render_aspect_candidate(fold.heldout_aspects[0],
                    "name_and_description" if condition == "D" else "name_only", resource)
                generated = legacy.synthesise_weight(descriptors, weights, embeddings[descriptor_text], family=family, **kwargs)
                presence = legacy.sigmoid(val_features @ generated[:-1] + generated[-1])
                heldout_scored = legacy.join_two_stage_scores(ag, sg, presence, target_sentiments[condition])
                complete = pd.concat([seen_scored, heldout_scored]).sort_values(IDENTITY).reset_index(drop=True)
                observed_hash = canonical_sha256(complete.loc[complete.candidate_aspect.isin(seen),
                    IDENTITY + ["aspect_score", "sentiment_score"]].to_dict("records"))
                if seen_hash is not None and observed_hash != seen_hash:
                    raise ValueError("DCWT N/D seen scores changed")
                seen_hash = observed_hash
                evaluate_and_save(complete, fold, selection, family_root, condition)
            receipt(family_root, contract(config, "dcwt_" + family, policy, fold.fold_id,
                parent_training_pool_sha256=identity["training_pool_sha256"], seen_ND_sha256=seen_hash))
        write_json(destination / "runtime.json", {"elapsed_seconds": time.monotonic() - started,
            "train_rows": len(train), "seen_aspects": len(seen), "families": list(selected),
            "candidate_count": sum(map(len, candidates.values())), "selection_uses_heldout_labels": False})
        receipt(destination, identity)
        backup_job(destination, ROOT.parent / "cloud_backups" / EXT_ID / "dcwt" / policy / fold.fold_id)
        print(f"COMPLETE_DCWT {fold.fold_id}/{policy} seconds={time.monotonic()-started:.1f}", flush=True)
    except BaseException as error:
        write_json(destination / "FAILED.json", {"error": repr(error), "contract": identity, "at": time.time()})
        raise
    finally:
        gc.collect()


def frozen_invariance(config, original, validation, folds, resource, output):
    audit = json.loads((ROOT / config["invariance"]["source_audit"]).read_text())
    if audit["status"] != "pass" or audit["failure_count"] != 0 or audit["test_contract_count"] != 0:
        raise ValueError("Invalid original replay audit")
    records = {(r["method_id"], r["fold_id"]): r for r in audit["records"]}
    source_root = ROOT / config["invariance"]["source_root"]
    summaries = []
    for method in config["invariance"]["methods"]:
        for fold in folds:
            source = source_root / "local_replay" / method / "folds" / f"{fold.fold_id}.json"
            if sha256_file(source) != records[(method, fold.fold_id)]["replay_sha256"]:
                raise ValueError("Frozen result source hash conflict")
            result = json.loads(source.read_text())
            if (result["failure_count"] != 0 or result["test_contract_count"] != 0
                    or result["fold_id"] != fold.fold_id or result["method_id"] != method
                    or result["validation_rows"] != 1057 or result["selection_partition"] != "seen validation candidates only"):
                raise ValueError("Frozen result source scope conflict")
            paired_input_hashes = {}
            for policy in config["policies"]:
                training = training_rows(original, fold, policy)
                destination = output / "invariance" / method / policy / fold.fold_id
                identity = contract(config, method, policy, fold.fold_id, source_result_sha256=sha256_file(source),
                    role=config["invariance"]["evidence_role"])
                if (destination / "receipt.json").exists():
                    verify_receipt(destination, identity)
                    raise RuntimeError("Partial invariance campaign must be inspected before replay")
                selection = {"aspect_threshold": result["aspect_threshold"],
                             "second_sentiment_threshold": result["second_sentiment_threshold"]}
                write_json(destination / "selection.json", selection)
                input_hashes = {}
                for condition in ("D", "N"):
                    ag, sg = grids(validation, fold, resource, condition)
                    sel_ag, sel_sg = grids(validation, fold, resource, "D", seen_only=True)
                    columns = ["row_uid", "candidate_aspect", "text", "candidate_text", "target"]
                    input_hashes[condition] = canonical_sha256({"aspect": ag[columns].to_dict("records"),
                        "sentiment": sg[columns + ["candidate_sentiment"]].to_dict("records"),
                        "selection_aspect": sel_ag[columns].to_dict("records"),
                        "selection_sentiment": sel_sg[columns + ["candidate_sentiment"]].to_dict("records"),
                        "thresholds": selection, "scorer_uses_train_rows": False})
                    old_evidence = source_root / "local_row_evidence" / method / fold.fold_id / f"{condition}.csv"
                    if sha256_file(old_evidence) != audit["local_row_evidence_sha256"][str(old_evidence.resolve())]:
                        raise ValueError("Frozen row evidence hash conflict")
                    rows = pd.read_csv(old_evidence).sort_values("row_uid")
                    if (len(rows) != 1057 or rows.row_uid.duplicated().any()
                            or rows.row_uid.tolist() != sorted(validation.row_uid.tolist())
                            or set(rows.fold_id) != {fold.fold_id} or set(rows.method_id) != {method}
                            or set(rows.condition) != {condition}):
                        raise ValueError("Frozen evidence identities mismatch")
                    expected_gold = sg.loc[sg.candidate_aspect.isin(fold.heldout_aspects)].groupby("row_uid").target.sum().sort_index()
                    if not np.array_equal(expected_gold.to_numpy(), rows.pair_gold_count.to_numpy()):
                        raise ValueError("Frozen evidence gold-label alignment mismatch")
                    tp, fp, fn = rows[["pair_tp", "pair_fp", "pair_fn"]].sum().to_numpy()
                    score = float(2 * tp / (2 * tp + fp + fn))
                    metrics = copy.deepcopy(result["conditions"][condition])
                    if not np.isclose(score, metrics["L2_E"]["partitions"]["heldout"]["pair_micro_f1"], atol=1e-12):
                        raise ValueError("Frozen F1 evidence reconstruction mismatch")
                    selected_aspects = int(metrics["L2_E"]["selected_aspect_instances"])
                    predicted_pairs = int(metrics["L2_E"]["partitions"]["overall"]["pair_predicted_label_count"])
                    if not 0 <= predicted_pairs - selected_aspects <= selected_aspects:
                        raise ValueError("Invalid frozen capped-two usage")
                    metrics["decoder_usage"] = {"selected_review_aspects": selected_aspects,
                                               "two_sentiment_instances": predicted_pairs - selected_aspects}
                    write_json(destination / f"{condition}_metrics.json", metrics)
                    count_frame = rows[["row_uid", "pair_tp", "pair_fp", "pair_fn"]].rename(
                        columns={"pair_tp": "tp", "pair_fp": "fp", "pair_fn": "fn"})
                    count_frame["partition"] = "heldout"
                    count_frame.to_csv(destination / f"{condition}_row_counts.csv", index=False)
                    summaries.append({"method": method, "policy": policy, "fold_id": fold.fold_id,
                        "condition": condition, "heldout_pair_f1": score, "new_inference_count": 0,
                        "source_result_sha256": identity["source_result_sha256"]})
                if paired_input_hashes and paired_input_hashes != input_hashes:
                    raise ValueError("Frozen scorer/selection inputs changed between training policies")
                paired_input_hashes = input_hashes
                write_json(destination / "invariance.json", {"status": "pass", "model_inputs": input_hashes,
                    "model_and_selection_depend_on_training_reviews": False, "training_pool_sha256": train_pool_hash(training),
                    "new_inference_count": 0, "interpretation": "Same source result, not independent replication",
                    "original_source": str(source), "source_audit_sha256": config["invariance"]["source_audit_sha256"]})
                receipt(destination, identity)
                backup_job(destination, ROOT.parent / "cloud_backups" / EXT_ID / "invariance" / method / policy / fold.fold_id)
            print("INVARIANCE_VERIFIED", method, fold.fold_id, flush=True)
    root = output / "invariance_summary"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(root / "folds.csv", index=False)
    write_json(root / "audit.json", {"status": "pass", "condition_records": len(summaries),
        "fold_policy_receipts": 48, "new_inference_count": 0, "failure_count": 0, "test_contract_count": 0,
        "interpretation": "Architectural invariance under fixed evaluation and selection inputs"})
    receipt(root, contract(config, "frozen_invariance_audit"))
    backup_job(root, ROOT.parent / "cloud_backups" / EXT_ID / "invariance_summary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["prepare", "invariance", "dcwt"])
    parser.add_argument("--fold-id")
    parser.add_argument("--policy", choices=["review_filtered", "label_masked_all_reviews"])
    args = parser.parse_args()
    config, train, validation, folds, resource = load_extension()
    output = ROOT / config["output_root"]
    if args.phase == "prepare":
        prepare(config, resource, output)
    elif args.phase == "invariance":
        frozen_invariance(config, train, validation, folds, resource, output)
    else:
        selected = [f for f in folds if f.fold_id == args.fold_id]
        if len(selected) != 1 or args.policy is None:
            parser.error("dcwt requires one registered fold and policy")
        dcwt_fold(config, train, validation, selected[0], resource, output, args.policy)


if __name__ == "__main__":
    main()
