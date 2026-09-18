"""Complete-case, local-only policy comparison. Never reads official-test data.

`template` records the exact source inventory before results arrive. `inventory`
only checks file presence and prints no performance. `analyse` requires all 12
folds, both policies and both conditions, validates sealed scores, then composes
each policy's own components with their unchanged thresholds. Missing inputs
produce PENDING inventory only, never a partial headline table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from msc_project.evaluation.resampling import (
    review_cluster_bootstrap as bootstrap,
    f1_from_counts as f1,
)
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.experiments.taxonomy_execution import RunContract, canonical_sha256, dataframe_sha256
from msc_project.experiments.taxonomy_post_supervisor import evaluate_l2_condition
from msc_project.experiments.taxonomy_post_supervisor_cloud import selection_path, candidate_result_path
from msc_project.experiments.taxonomy_protocol import registered_folds
from msc_project.experiments.taxonomy_training_policy_sensitivity import sha256_file, verify_receipt, write_json
from msc_project.experiments.taxonomy_two_stage import capped_two_sentiment_prediction_mask
from msc_project.experiments.taxonomy_two_stage_artifacts import validate_two_stage_score_shard, merge_two_stage_score_shards

STUDY = "taxonomy_policy_qlora_completion_v1"
FOLDS = tuple(f"l2-a{i:02d}" for i in range(1, 13))
POLICIES = ("review_filtered", "label_masked_all_reviews")
METHODS = ("fewshot", "qlora")
CONDITIONS = ("N", "D")
QLORA = "qwen_candidate_pair_qlora"
KEY = ["row_uid", "candidate_aspect", "candidate_sentiment"]
CANONICAL = "registered_first_N_prompt_identity_reuse"
VALIDATION_SHA = "3a32afda9c4ba1d59e76ee8fcfb480011c7178d8201d0025b254a40c2f81203c"
FEWSHOT_CONFIG_SHA = "a7cd5e1b3e10e312634604b508396492ec50914bb555b0dd8409a6f97e9f52dd"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def template(fewshot_root, filtered_root, masked_root, data_dir):
    return {"study_id": STUDY, "allowed_splits": ["validation"], "include_official_test": False,
            "test_contract_count": 0, "validation_sha256": VALIDATION_SHA,
            "data_dir": str(Path(data_dir).resolve()), "folds": list(FOLDS),
            "bootstrap": {"draws": 20000, "seed": 13, "unit": "synchronised_review_uid"},
            "composition": "same_policy_fewshot_stage1_qlora_stage2_no_retuning",
            "evidence_role": "post_test_validation_only_training_policy_sensitivity",
            "sources": [{"method": method, "policy": policy, "fold_id": fold,
                         "root": str((Path(fewshot_root) / policy / fold if method == "fewshot"
                                      else Path(filtered_root if policy == POLICIES[0] else masked_root)).resolve())}
                        for method in METHODS for policy in POLICIES for fold in FOLDS]}


def validate_manifest(manifest):
    if (manifest.get("study_id") != STUDY or manifest.get("allowed_splits") != ["validation"]
            or manifest.get("include_official_test") is not False or manifest.get("test_contract_count") != 0
            or manifest.get("validation_sha256") != VALIDATION_SHA or manifest.get("folds") != list(FOLDS)
            or manifest.get("bootstrap") != {"draws": 20000, "seed": 13, "unit": "synchronised_review_uid"}
            or manifest.get("composition") != "same_policy_fewshot_stage1_qlora_stage2_no_retuning"
            or manifest.get("evidence_role") != "post_test_validation_only_training_policy_sensitivity"):
        raise ValueError("Unregistered analysis protocol or validation-only boundary")
    expected = {(m, p, f) for m in METHODS for p in POLICIES for f in FOLDS}
    observed = [(s["method"], s["policy"], s["fold_id"]) for s in manifest["sources"]]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Exactly 48 distinct method-policy-fold sources are required")
    for source in manifest["sources"]:
        if not Path(source["root"]).is_absolute():
            raise ValueError("Explicit absolute source roots required")


def selection_file(source):
    root = Path(source["root"])
    return (root / "selection.json" if source["method"] == "fewshot" else
            selection_path(root, QLORA, source["fold_id"].replace("l2-", "heldout-")))


def inventory(manifest):
    validate_manifest(manifest)
    rows = []
    for source in manifest["sources"]:
        root = Path(source["root"])
        required = [selection_file(source)]
        if source["method"] == "fewshot":
            required += [root / "receipt.json"]
            required += [root / f"{c}_scores.csv.gz" for c in CONDITIONS]
        else:
            required += [root / "policy_provenance/run_identity.json"]
            required += [candidate_result_path(root, QLORA, source["fold_id"].replace("l2-", "heldout-"), rate)
                         for rate in (2e-6, 5e-6, 1e-5)]
            required += [root / "results" / QLORA / "L2" / source["fold_id"] / f"{c}.json" for c in CONDITIONS]
        missing = [str(p) for p in required if not p.is_file()]
        shard_counts = {}
        if source["method"] == "qlora":
            for condition in CONDITIONS:
                directory = root / "scores" / QLORA / "L2" / source["fold_id"] / condition
                manifests = list(directory.glob("*/shard-*.manifest.json"))
                shard_counts[condition] = len(manifests)
                if len(manifests) != 8:
                    missing.append(f"{directory}: expected exactly 8 shard manifests, found {len(manifests)}")
        rows.append({**source, "status": "PENDING" if missing else "PRESENT_NOT_YET_AUDITED",
                     "missing": missing, "shard_manifests": shard_counts})
    pending = sum(r["status"] == "PENDING" for r in rows)
    return {"status": "PENDING" if pending else "READY_FOR_INTEGRITY_AUDIT", "sources": rows,
            "required_source_folds": 48, "present_source_folds": 48-pending,
            "pending_source_folds": pending, "headline_metrics_generated": False,
            "test_contract_count": 0, "manifest_sha256": canonical_sha256(manifest)}


def sealed(payload, field):
    value = dict(payload)
    digest = value.pop(field, None)
    if digest != canonical_sha256(value):
        raise ValueError(f"Sealed JSON hash mismatch: {field}")


def thresholds(value, formal=False):
    source = value["selected_thresholds"] if formal else value
    result = {"aspect_threshold": float(source["aspect"] if formal else source["aspect_threshold"]),
              "second_sentiment_threshold": float(source["runner_up_sentiment"] if formal else source["second_sentiment_threshold"])}
    # The frozen decoder permits nextafter(1, +inf) to select no runner-up.
    if not all(np.isfinite(list(result.values()))) or any(v < 0 or v > np.nextafter(1.0, np.inf) for v in result.values()):
        raise ValueError("Invalid frozen thresholds")
    return result


def validate_grid(scores, expected, fold, condition):
    columns = KEY + ["target"]
    scores = scores.sort_values(KEY, kind="stable").reset_index(drop=True)
    reference = expected.sort_values(KEY, kind="stable").reset_index(drop=True)
    if (scores.duplicated(KEY).any() or len(scores) != len(reference)
            or not scores.row_uid.astype(str).str.startswith("validation:").all()
            or not set(scores.target.unique()) <= {0, 1, False, True}):
        raise ValueError("Invalid validation grid identity or target")
    observed = scores[columns].copy()
    observed["target"] = observed.target.astype(int)
    if not observed.equals(reference[columns]):
        raise ValueError("Scores do not match the pinned validation labels and full candidate grid")
    if "split" in scores and set(scores.split.astype(str)) != {"validation"}:
        raise ValueError("Non-validation score split")
    if "condition" in scores and set(scores.condition.astype(str)) != {condition}:
        raise ValueError("Wrong source condition")
    if "fold_id" in scores and set(scores.fold_id.astype(str)) != {fold.fold_id}:
        raise ValueError("Wrong source fold")
    values = scores[["aspect_score", "sentiment_score"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probability scores")
    if scores.groupby(KEY[:2]).aspect_score.nunique().max() != 1:
        raise ValueError("Aspect presence must be identical across the three sentiment rows")
    if scores.aspect_score.nunique() < 2 or scores.sentiment_score.nunique() < 2:
        raise ValueError("Constant raw score collapse")
    return scores


def checkpoint_files(root, candidate, hashes):
    checkpoint = (root / candidate["checkpoint_relative_path"]).resolve()
    if not checkpoint.is_relative_to(root.resolve()):
        raise ValueError("Checkpoint escapes source root")
    path = checkpoint / "checkpoint.manifest.json"
    if sha256_file(path) != candidate["checkpoint_manifest_sha256"]:
        raise ValueError("Checkpoint manifest hash mismatch")
    record = read_json(path)
    if record["training_contract"] != candidate["training_contract"] or record["evidence"].get("test_contract_count") != 0:
        raise ValueError("Checkpoint training contract mismatch")
    hashes[str(path)] = sha256_file(path)
    for relative, digest in record["files"].items():
        file = (checkpoint / relative).resolve()
        if not file.is_relative_to(checkpoint) or sha256_file(file) != digest:
            raise ValueError("Checkpoint file hash/path mismatch")
        hashes[str(file)] = digest


def validate_selection_sources(root, selection, fold_id, hashes, policy="review_filtered"):
    scope = fold_id.replace("l2-", "heldout-")
    training_digest = None
    configuration_digest = canonical_sha256(read_json(ROOT / "configs/experiments/taxonomy_policy_qlora_completion_v1.json"))
    if policy == POLICIES[1]:
        from msc_project.experiments.taxonomy_two_stage_training import training_manifest_sha256
        folder = root / "policy_provenance" / scope
        provenance = read_json(folder / "scope.json")
        definition = next(f for f in registered_folds("L2") if f.fold_id == fold_id)
        if (provenance.get("study_id") != STUDY or provenance.get("policy") != policy
                or provenance.get("scope_id") != scope or provenance.get("test_contract_count") != 0
                or provenance.get("configuration_sha256") != configuration_digest
                or provenance.get("training_review_count") != 7930 or provenance.get("validation_review_count") != 1057
                or provenance.get("seen_aspects") != list(definition.seen_aspects)
                or provenance.get("heldout_aspects") != list(definition.heldout_aspects)):
            raise ValueError("Masked training scope supervision/data boundary mismatch")
        manifest_hashes = {}
        for task in ("aspect_presence", "sentiment"):
            file = folder / f"{task}.csv.gz"
            if sha256_file(file) != provenance["files"][file.name]:
                raise ValueError("Masked sampled training manifest file hash mismatch")
            frame = pd.read_csv(file, float_precision="round_trip", keep_default_na=False)
            if (len(frame) != 2048 or set(frame.candidate_aspect) - set(definition.seen_aspects)
                    or not frame.row_uid.str.startswith("train:").all()):
                raise ValueError("Masked manifest includes held-out/nontraining supervision or wrong budget")
            digest = training_manifest_sha256(frame)
            if digest != provenance["manifests"][task]["manifest_sha256"]:
                raise ValueError("Masked sampled training manifest semantic hash mismatch")
            manifest_hashes[task] = digest
            hashes[str(file)] = provenance["files"][file.name]
        training_digest = canonical_sha256(manifest_hashes)
        hashes[str(folder / "scope.json")] = sha256_file(folder / "scope.json")
    candidates = []
    for rate in (2e-6, 5e-6, 1e-5):
        path = candidate_result_path(root, QLORA, scope, rate)
        value = read_json(path)
        sealed(value, "candidate_payload_sha256")
        if (value.get("failure_count") != 0 or value.get("test_contract_count") != 0
                or value.get("selection_partition") != "seen_validation_only"
                or value.get("training_scope_id") != scope or value.get("method_id") != QLORA
                or value.get("learning_rate") != rate):
            raise ValueError("Three-candidate selection provenance mismatch")
        hashes[str(path)] = sha256_file(path)
        if policy == POLICIES[1]:
            if value["training_contract"]["training_manifest_sha256"] != training_digest:
                raise ValueError("Candidate is not bound to audited masked sampled supervision")
            checkpoint_files(root, value, hashes)
            directory = root / "calibration_scores" / scope / ("lr-" + f"{rate:.0e}".replace("-0", "-"))
            record = read_json(directory / "manifest.json")
            sealed(record, "payload_sha256")
            csv = directory / "seen_scores.csv.gz"
            if (record.get("study_id") != STUDY or record.get("policy") != policy
                    or record.get("scope_id") != scope or record.get("learning_rate") != rate
                    or record.get("configuration_sha256") != configuration_digest
                    or record.get("test_contract_count") != 0 or record.get("selection_partition") != "seen_validation_only"
                    or record.get("training_contract_sha256") != value["training_contract"]["contract_sha256"]
                    or record.get("csv_sha256") != sha256_file(csv)
                    or any(record.get(key) != value.get(key) for key in ("calibration_pair_identity_sha256", "calibration_score_sha256"))):
                raise ValueError("Raw seen-only calibration provenance mismatch")
            calibration = pd.read_csv(csv, float_precision="round_trip")
            definition = next(f for f in registered_folds("L2") if f.fold_id == fold_id)
            if (len(calibration) != 1057*11*3 or calibration.duplicated(KEY).any()
                    or calibration.row_uid.nunique() != 1057 or set(calibration.candidate_aspect) != set(definition.seen_aspects)
                    or not calibration.row_uid.str.startswith("validation:").all()
                    or not np.isfinite(calibration[["aspect_score", "sentiment_score"]]).all().all()
                    or dataframe_sha256(calibration, KEY + ["aspect_score", "sentiment_score"], sort_columns=KEY) != record["calibration_score_sha256"]):
                raise ValueError("Raw calibration grid is not complete finite seen-only evidence")
            hashes[str(csv)] = record["csv_sha256"]
            hashes[str(directory / "manifest.json")] = sha256_file(directory / "manifest.json")
        candidates.append(value)
    contract_ids = [v["training_contract"]["contract_sha256"] for v in candidates]
    if selection.get("candidate_contract_sha256s") != contract_ids:
        raise ValueError("Selection does not bind all three learning-rate candidates")
    winner = max(candidates, key=lambda v: (v["selection_metrics"]["pair_micro_f1"],
                  v["selection_metrics"]["pair_samples_f1"], v["selection_metrics"]["pair_micro_precision"], -v["learning_rate"]))
    if (selection["selected_candidate_contract_sha256"] != winner["training_contract"]["contract_sha256"]
            or selection["selected_thresholds"] != winner["thresholds"]
            or selection["selected_learning_rate"] != winner["learning_rate"]
            or selection["selected_checkpoint_relative_path"] != winner["checkpoint_relative_path"]):
        raise ValueError("Frozen seen-only selection/tie rule does not reconstruct")
    if policy == POLICIES[0]:
        checkpoint_files(root, winner, hashes)


def verify_policy_identity(root, identity, policy, hashes):
    import run_taxonomy_policy_qlora_completion as masked
    config_path = ROOT / "configs/experiments/taxonomy_policy_qlora_completion_v1.json"
    config = read_json(config_path)
    masked.validate_config(config)
    if identity.get("configuration_sha256") != canonical_sha256(config) or identity.get("test_contract_count", 0) != 0:
        raise ValueError("Completion source configuration/safety binding mismatch")
    if policy == POLICIES[0]:
        import run_policy_qlora_filtered_rescore as filtered
        path = root / filtered.MANIFEST
        if str(path) not in hashes:
            filtered.verify(root)
            hashes[str(path)] = sha256_file(path)
            for relative, digest in read_json(path)["files"].items():
                hashes[str(filtered.safe_path(root, relative))] = digest
    else:
        expected = {"study_id": STUDY, "policy": policy, "configuration_sha256": canonical_sha256(config),
                    "canonical_seen_policy": CANONICAL, "recipe": masked.frozen_recipe(masked.load_executor())}
        if identity != json.loads(json.dumps(expected)):
            raise ValueError("Masked source full run identity mismatch")
    hashes[str(config_path)] = sha256_file(config_path)


def load_source(source, expected, fold, hashes):
    root = Path(source["root"])
    selection_path_ = selection_file(source)
    selection = read_json(selection_path_)
    hashes[str(selection_path_)] = sha256_file(selection_path_)
    grids = {}
    if (root / "FAILED.json").exists():
        raise ValueError("Source contains failure evidence")
    if source["method"] == "fewshot":
        record = verify_receipt(root)
        contract = record["contract"]
        if any(contract.get(k) != source[k] for k in ("method", "policy", "fold_id")):
            raise ValueError("Few-shot receipt policy/fold identity mismatch")
        if (contract.get("study_id") != "taxonomy_training_policy_sensitivity_v1"
                or contract.get("config_sha256") != FEWSHOT_CONFIG_SHA):
            raise ValueError("Unexpected few-shot provenance")
        hashes[str(root / "receipt.json")] = sha256_file(root / "receipt.json")
        for condition in CONDITIONS:
            path = root / f"{condition}_scores.csv.gz"
            hashes[str(path)] = sha256_file(path)
            grids[condition] = validate_grid(pd.read_csv(path, float_precision="round_trip"), expected, fold, condition)
        chosen = thresholds(selection)
    else:
        identity_path = root / "policy_provenance/run_identity.json"
        identity = read_json(identity_path)
        if identity.get("policy") != source["policy"] or identity.get("canonical_seen_policy") != CANONICAL:
            raise ValueError("QLoRA source policy/canonical scorer identity mismatch")
        if identity.get("study_id") != STUDY:
            raise ValueError("QLoRA result is not from the registered completion study")
        verify_policy_identity(root, identity, source["policy"], hashes)
        hashes[str(identity_path)] = sha256_file(identity_path)
        sealed(selection, "selection_payload_sha256")
        if (selection.get("failure_count") != 0 or selection.get("test_contract_count") != 0
                or selection.get("training_scope_id") != fold.fold_id.replace("l2-", "heldout-")
                or selection.get("method_id") != QLORA):
            raise ValueError("QLoRA selected candidate safety/scope conflict")
        validate_selection_sources(root, selection, fold.fold_id, hashes, source["policy"])
        chosen = thresholds(selection, formal=True)
        for condition in CONDITIONS:
            result_path = root / "results" / QLORA / "L2" / fold.fold_id / f"{condition}.json"
            result = read_json(result_path)
            sealed(result, "result_payload_sha256")
            if (result.get("failure_count") != 0 or result.get("test_contract_count") != 0
                    or result.get("evaluation_partition") != "validation_only"
                    or result.get("selection_partition") != "seen_validation_only"
                    or result.get("protocol_id") != STUDY or result.get("fold_id") != fold.fold_id or result.get("condition") != condition
                    or result.get("thresholds") != selection["selected_thresholds"]
                    or result.get("training_contract_sha256") != selection["selected_candidate_contract_sha256"]):
                raise ValueError("QLoRA result identity or selection binding mismatch")
            directory = root / "scores" / QLORA / "L2" / fold.fold_id / condition
            shard_paths = sorted(directory.glob("*/shard-*.manifest.json"))
            if len(shard_paths) != 8 or len({p.parent for p in shard_paths}) != 1:
                raise ValueError("Exactly one complete eight-shard score contract required")
            artifacts, contract = {}, None
            for path in shard_paths:
                manifest = read_json(path)
                fields = dict(manifest["contract"])
                fields.pop("contract_sha256", None)
                current = RunContract(**fields)
                if (current.protocol_id != STUDY or current.level != "L2" or not current.formal
                        or current.shard_count != 8 or current.split != "validation" or current.fold_id != fold.fold_id
                        or current.condition != condition or current.method_id != QLORA
                        or current.contract_sha256 != result["run_contract_sha256"]
                        or (contract is not None and current != contract)):
                    raise ValueError("Score contract identity conflict")
                contract = current
                index = manifest["shard_index"]
                if index in artifacts:
                    raise ValueError("Duplicate shard")
                csv = path.with_name(path.name.replace(".manifest.json", ".csv"))
                artifacts[index] = validate_two_stage_score_shard(csv, path, contract, expected, shard_index=index)
                hashes[str(path)] = sha256_file(path)
                hashes[str(csv)] = sha256_file(csv)
            merged = merge_two_stage_score_shards(artifacts, contract, expected)
            if result["score_sha256"] != dataframe_sha256(merged, KEY + ["aspect_score", "sentiment_score"], sort_columns=KEY):
                raise ValueError("Merged scores do not reproduce sealed result hash")
            hashes[str(result_path)] = sha256_file(result_path)
            grids[condition] = validate_grid(merged, expected, fold, condition)
    seen_columns = KEY + ["aspect_score", "sentiment_score"]
    n = grids["N"].loc[grids["N"].candidate_aspect.isin(fold.seen_aspects), seen_columns].reset_index(drop=True)
    d = grids["D"].loc[grids["D"].candidate_aspect.isin(fold.seen_aspects), seen_columns].reset_index(drop=True)
    if not n.equals(d):
        raise ValueError("N/D representation-invariant seen scores differ")
    return grids, chosen


def compose(first, second, first_policy, second_policy):
    if first_policy != second_policy:
        raise ValueError("Cross-policy composition is excluded")
    if not first[KEY + ["target"]].equals(second[KEY + ["target"]]):
        raise ValueError("Component identity/target ordering mismatch")
    combined = first.copy()
    combined["sentiment_score"] = second.sentiment_score.to_numpy(copy=True)
    return combined


def summarise_grid(scores, fold, selected):
    result = evaluate_l2_condition(scores, fold, **selected)
    predicted = capped_two_sentiment_prediction_mask(scores, **selected).to_numpy(bool)
    chosen = scores.loc[predicted].groupby(KEY[:2]).size()
    if chosen.empty or chosen.max() > 2 or len(chosen) == scores.row_uid.nunique() * 12:
        raise ValueError("Prediction collapse or third sentiment violation")
    truth = scores.target.to_numpy(bool)
    evidence = scores[KEY].copy()
    for name, values in {"tp": truth & predicted, "fp": ~truth & predicted, "fn": truth & ~predicted}.items():
        evidence[name] = values.astype(np.int64)
    heldout = evidence.loc[evidence.candidate_aspect.isin(fold.heldout_aspects)]
    counts = heldout.groupby("row_uid")[["tp", "fp", "fn"]].sum().sort_index()
    partitions = result["L2_E"]["partitions"]
    row = {f"{part}_{name}": metric[name] for part, metric in partitions.items()
           for name in ("pair_micro_f1", "pair_micro_precision", "pair_micro_recall")}
    row.update({"heldout_presence_f1": result["L2_S"]["aspect_presence"]["f1"],
                "heldout_presence_ap": result["L2_S"]["aspect_presence"]["average_precision"],
                "oracle_stage2_f1": result["L2_S"]["oracle_aspect_gated_sentiment"]["capped_two_pair_metrics"]["pair_micro_f1"],
                "selected_review_aspects": int(len(chosen)), "two_sentiment_instances": int((chosen == 2).sum()),
                **selected})
    if not np.isclose(f1(counts.to_numpy().sum(axis=0)), row["heldout_pair_micro_f1"], atol=1e-12):
        raise ValueError("Row-confusion F1 reconstruction failure")
    return row, counts


def comparison_terms():
    terms = []
    for c in CONDITIONS:
        for m in ("fewshot", "qlora", "composition"):
            terms.append((f"{m}_{c}_masked_minus_filtered", [((m, POLICIES[1], c), 1), ((m, POLICIES[0], c), -1)]))
        for p in POLICIES:
            for comparator in METHODS:
                terms.append((f"{p}_{c}_composition_minus_{comparator}", [(("composition", p, c), 1), ((comparator, p, c), -1)]))
        for comparator in METHODS:
            terms.append((f"{c}_change_in_composition_gain_over_{comparator}",
                          [(("composition", POLICIES[1], c), 1), ((comparator, POLICIES[1], c), -1),
                           (("composition", POLICIES[0], c), -1), ((comparator, POLICIES[0], c), 1)]))
    return terms


def expected_validation(manifest):
    validate_manifest(manifest)
    data_dir = Path(manifest["data_dir"])
    if sha256_file(data_dir / "validation.csv") != VALIDATION_SHA:
        raise ValueError("Pinned validation hash mismatch")
    # Explicit one-file split whitelist. No glob or official-test input path.
    validation = load_official_fabsa_splits(data_dir, ("validation",))
    order = sorted(validation.row_uid.astype(str))
    if len(order) != 1057 or len(set(order)) != 1057 or not all(uid.startswith("validation:") for uid in order):
        raise ValueError("Pinned validation row identity/count mismatch")
    definitions = {f.fold_id: f for f in registered_folds("L2")}
    aspects = definitions[FOLDS[0]].evaluation_aspects
    expected = pd.DataFrame([(row.row_uid, aspect, sentiment, int((aspect, sentiment) in row.labels))
                             for row in validation.itertuples() for aspect in aspects
                             for sentiment in ("negative", "neutral", "positive")], columns=KEY + ["target"])
    return expected, order, definitions


def audit_pilot(manifest):
    """Validate a01 masked component compatibility without computing any F1."""
    status = inventory(manifest)
    wanted = [s for s in status["sources"] if s["policy"] == POLICIES[1] and s["fold_id"] == FOLDS[0]]
    if any(s["status"] == "PENDING" for s in wanted):
        return {"status": "PENDING", "scope": "masked_a01_pilot_only", "sources": wanted,
                "headline_metrics_generated": False, "test_contract_count": 0}
    expected, _, definitions = expected_validation(manifest)
    fold = definitions[FOLDS[0]]
    hashes, loaded = {}, {}
    for source in wanted:
        loaded[source["method"]] = load_source(source, expected, fold, hashes)
    first, ft = loaded["fewshot"]
    second, st = loaded["qlora"]
    for condition in CONDITIONS:
        combined = compose(first[condition], second[condition], POLICIES[1], POLICIES[1])
        selected = {"aspect_threshold": ft["aspect_threshold"], "second_sentiment_threshold": st["second_sentiment_threshold"]}
        prediction = capped_two_sentiment_prediction_mask(combined, **selected)
        counts = combined.loc[prediction].groupby(KEY[:2]).size()
        if counts.empty or counts.max() > 2 or len(counts) == 1057 * 12:
            raise ValueError("Pilot composition decoder collapse or third sentiment")
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("Pilot sources changed during integrity audit")
    return {"status": "PASS", "scope": "masked_a01_pilot_only", "complete_study": False,
            "source_hashes": hashes, "same_policy_components": True, "N_D_seen_scores_identical": True,
            "third_sentiment_violations": 0, "headline_metrics_generated": False, "test_contract_count": 0}


def analyse(manifest, target):
    status = inventory(manifest)
    if status["status"] == "PENDING":
        return status
    target = Path(target)
    if target.exists():
        raise FileExistsError("Use a new immutable analysis output directory")
    expected, order, definitions = expected_validation(manifest)
    data_dir = Path(manifest["data_dir"])
    index = {(s["method"], s["policy"], s["fold_id"]): s for s in manifest["sources"]}
    hashes, rows, counts_by_system = {}, [], {}
    hashes[str(data_dir / "validation.csv")] = VALIDATION_SHA
    for relative in ("scripts/analyse_taxonomy_policy_qlora_completion.py", "scripts/analyse_taxonomy_training_policy_sensitivity.py",
                     "src/msc_project/experiments/taxonomy_two_stage.py", "src/msc_project/experiments/taxonomy_post_supervisor.py",
                     "src/msc_project/experiments/taxonomy_two_stage_artifacts.py", "src/msc_project/experiments/taxonomy_execution.py"):
        path = ROOT / relative
        hashes[str(path)] = sha256_file(path)
    for fold_id in FOLDS:
        fold = definitions[fold_id]
        for policy in POLICIES:
            first, ft = load_source(index[("fewshot", policy, fold_id)], expected, fold, hashes)
            second, st = load_source(index[("qlora", policy, fold_id)], expected, fold, hashes)
            for c in CONDITIONS:
                combined = compose(first[c], second[c], policy, policy)
                ht = {"aspect_threshold": ft["aspect_threshold"], "second_sentiment_threshold": st["second_sentiment_threshold"]}
                for method, grid, chosen in (("fewshot", first[c], ft), ("qlora", second[c], st), ("composition", combined, ht)):
                    row, counts = summarise_grid(grid, fold, chosen)
                    if counts.index.tolist() != order:
                        raise ValueError("Review-cluster order differs across systems/folds")
                    rows.append({"method": method, "policy": policy, "fold_id": fold_id, "condition": c, **row})
                    counts_by_system.setdefault((method, policy, c), {})[fold_id] = counts.to_numpy(np.int64)
    table = pd.DataFrame(rows)
    systems = sorted(counts_by_system)
    tensor = np.stack([np.stack([counts_by_system[k][f] for f in FOLDS]) for k in systems])
    balanced, pooled = bootstrap(tensor, draws=20000, seed=13)
    points = f1(tensor.sum(axis=2)).mean(axis=1)
    pooled_points = f1(tensor.sum(axis=(1, 2)))
    aggregate = []
    for i, (method, policy, condition) in enumerate(systems):
        subset = table.loc[table.method.eq(method) & table.policy.eq(policy) & table.condition.eq(condition)]
        aggregate.append({"method": method, "policy": policy, "condition": condition, "folds": 12,
                          "heldout_pair_f1": points[i], "pooled_heldout_pair_f1": pooled_points[i],
                          "nominal_ci_low": np.quantile(balanced[:, i], .025), "nominal_ci_high": np.quantile(balanced[:, i], .975),
                          "pooled_nominal_ci_low": np.quantile(pooled[:, i], .025), "pooled_nominal_ci_high": np.quantile(pooled[:, i], .975),
                          **{k: subset[k].mean() for k in ("heldout_pair_micro_precision", "heldout_pair_micro_recall", "overall_pair_micro_f1", "seen_pair_micro_f1", "heldout_presence_f1", "heldout_presence_ap", "oracle_stage2_f1")},
                          "two_sentiment_instances": int(subset.two_sentiment_instances.sum()),
                          "selected_review_aspects": int(subset.selected_review_aspects.sum())})
    comparisons = []
    for name, terms in comparison_terms():
        estimate, draws = 0.0, np.zeros(20000)
        for key, coefficient in terms:
            i = systems.index(key)
            estimate += coefficient * points[i]
            draws += coefficient * balanced[:, i]
        comparisons.append({"comparison": name, "estimate": estimate, "nominal_ci_low": np.quantile(draws, .025),
                            "nominal_ci_high": np.quantile(draws, .975), "draws": 20000, "seed": 13,
                            "role": "descriptive_fewshot_reference" if name.startswith("fewshot_") else "registered_policy_comparison",
                            "evidence_role": manifest["evidence_role"]})
    # Recheck all observed input bytes after analysis, before publishing tables.
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("Source changed during analysis")
    target.mkdir(parents=True)
    for name, frame in (("fold_metrics", table), ("aggregate_metrics", pd.DataFrame(aggregate)), ("paired_contrasts", pd.DataFrame(comparisons))):
        frame.to_csv(target / f"{name}.csv", index=False)
    np.savez_compressed(target / "review_confusion_and_bootstrap.npz", counts=tensor, balanced=balanced, pooled=pooled,
                        systems=np.asarray(systems), review_uids=np.asarray(order), folds=np.asarray(FOLDS))
    write_json(target / "input_manifest.json", manifest)
    audit = {"status": "PASS", "source_fold_units": 48, "source_condition_grids": 96, "derived_composition_grids": 48,
             "reported_fold_condition_systems": len(rows), "failure_count": 0, "test_contract_count": 0,
             "source_hashes": hashes, "analysis_code_sha256": sha256_file(Path(__file__)),
             "review_identity_sha256": canonical_sha256(order), "evidence_role": manifest["evidence_role"],
             "selection_retuned": False, "same_policy_components": True, "N_D_seen_scores_identical": True,
             "interval_scope": "nominal conditional validation review-cluster uncertainty, not seed/taxonomy/selection-adjusted inference"}
    write_json(target / "audit.json", audit)
    write_json(target / "output_hashes.json", {p.name: sha256_file(p) for p in sorted(target.iterdir()) if p.is_file()})
    return {"status": "PASS", "output": str(target), "folds": 12, "test_contract_count": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("template", "inventory", "audit-pilot", "analyse"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fewshot-root", type=Path)
    parser.add_argument("--filtered-qlora-root", type=Path)
    parser.add_argument("--masked-qlora-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "template":
        if args.manifest.exists():
            raise FileExistsError("Refusing to overwrite input manifest")
        if any(v is None for v in (args.fewshot_root, args.filtered_qlora_root, args.masked_qlora_root, args.data_dir)):
            parser.error("Template mode requires all three source roots and --data-dir")
        value = template(args.fewshot_root, args.filtered_qlora_root, args.masked_qlora_root, args.data_dir)
        validate_manifest(value)
        write_json(args.manifest, value)
        result = {"status": "TEMPLATE_WRITTEN", "manifest": str(args.manifest)}
    else:
        manifest = read_json(args.manifest)
        if args.mode == "analyse" and args.output is None:
            parser.error("Analysis requires --output")
        result = (inventory(manifest) if args.mode == "inventory" else
                  audit_pilot(manifest) if args.mode == "audit-pilot" else analyse(manifest, args.output))
        if args.output and args.mode in ("inventory", "audit-pilot"):
            if args.output.exists():
                raise FileExistsError("Use a new inventory output path")
            write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] != "PENDING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
