"""Complete-case, synchronised review-bootstrap analysis of the locked study."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from msc_project.experiments.taxonomy_training_policy_sensitivity import (
    STUDY_ID, IDENTITY, canonical_sha256, verify_receipt, sha256_file, write_json, receipt,
)
from msc_project.experiments.taxonomy_two_stage import capped_two_sentiment_prediction_mask
from msc_project.evaluation.resampling import (
    f1_from_counts as f1,
    review_cluster_bootstrap as bootstrap,
)
from campaign_taxonomy_training_policy_sensitivity import backup_job


def audit_score_grid(root, condition, fold, selection, recorded_counts):
    """Reconstruct saved review counts from sealed scores, not from headline F1."""
    from msc_project.experiments.taxonomy_protocol import registered_folds
    definition = next(f for f in registered_folds("L2") if f.fold_id == fold)
    scores = pd.read_csv(root / f"{condition}_scores.csv.gz", float_precision="round_trip")
    scores = scores.sort_values(IDENTITY).reset_index(drop=True)
    if (len(scores) != 1057 * 12 * 3 or scores.duplicated(IDENTITY).any()
            or scores.row_uid.nunique() != 1057 or not scores.row_uid.str.startswith("validation:").all()
            or set(scores.candidate_aspect) != set(definition.evaluation_aspects)
            or set(scores.candidate_sentiment) != {"negative", "neutral", "positive"}
            or not np.isfinite(scores[["aspect_score", "sentiment_score"]].to_numpy()).all()):
        raise ValueError("Sealed score grid coverage/finite-value failure")
    predicted = capped_two_sentiment_prediction_mask(scores,
        aspect_threshold=selection["aspect_threshold"],
        second_sentiment_threshold=selection["second_sentiment_threshold"]).to_numpy(bool)
    truth = scores.target.to_numpy(bool)
    evidence = scores[IDENTITY].copy()
    for name, mask in {"tp": truth & predicted, "fp": ~truth & predicted, "fn": truth & ~predicted}.items():
        evidence[name] = mask.astype(np.int64)
    evidence["partition"] = np.where(evidence.candidate_aspect.isin(definition.heldout_aspects), "heldout", "seen")
    regenerated = evidence.groupby(["row_uid", "partition"])[["tp", "fp", "fn"]].sum().sort_index()
    recorded = recorded_counts.set_index(["row_uid", "partition"])[["tp", "fp", "fn"]].sort_index()
    if not regenerated.equals(recorded):
        raise ValueError("Sealed scores do not reproduce the recorded review-level confusion counts")
    seen = scores.loc[scores.candidate_aspect.isin(definition.seen_aspects),
                      IDENTITY + ["aspect_score", "sentiment_score"]]
    return canonical_sha256(seen.to_dict("records"))


def main():
    config = json.loads((ROOT / f"configs/experiments/{STUDY_ID}.json").read_text())
    output = ROOT / config["output_root"]
    public = ROOT / config["report_root"]
    analysis = output / "analysis"
    if (analysis / "receipt.json").exists():
        verify_receipt(analysis)
        print("ANALYSIS_ALREADY_VERIFIED")
        return
    policies = config["policies"]
    expected = [(method, policy, fold) for method in ("tfidf", "distilbert", "fewshot", "mixed_component_diagnostic")
                for policy in policies for fold in config["folds"]]
    expected += [("tfidf", f"size_matched_label_masked_s{seed}", fold)
                 for seed in config["tfidf"]["subset_seeds"] for fold in config["folds"]]
    missing = ["/".join(job) for job in expected if not (output.joinpath(*job) / "receipt.json").exists()]
    if missing:
        raise RuntimeError(f"Complete analysis requires all 132 fold receipts. Missing {len(missing)}: {missing[:3]}")
    rows, counts_by_system, source_hashes = [], {}, {}
    review_order = None
    for index, (method, policy, fold) in enumerate(expected):
        root = output / method / policy / fold
        payload = verify_receipt(root)
        if (root / "FAILED.json").exists() or any(payload["contract"].get(k) != v for k, v in
            {"study_id": STUDY_ID, "config_sha256": canonical_sha256(config), "method": method,
             "policy": policy, "fold_id": fold}.items()):
            raise ValueError("Result receipt identity/failure conflict")
        source_hashes[str(root / "receipt.json")] = sha256_file(root / "receipt.json")
        selection = json.loads((root / "selection.json").read_text())
        seen_identity = None
        for condition in ("D", "N"):
            result = json.loads((root / f"{condition}_metrics.json").read_text())
            partitions = result["L2_E"]["partitions"]
            if result["L2_E"]["maximum_sentiments_per_selected_aspect"] > 2:
                raise ValueError("Third sentiment in result")
            row = {"method": method, "policy": policy, "fold_id": fold, "condition": condition,
                   "aspect_threshold": selection["aspect_threshold"],
                   "runner_up_threshold": selection["second_sentiment_threshold"],
                   "heldout_presence_f1": result["L2_S"]["aspect_presence"]["f1"],
                   "heldout_presence_ap": result["L2_S"]["aspect_presence"]["average_precision"],
                   "oracle_stage2_f1": result["L2_S"]["oracle_aspect_gated_sentiment"]["capped_two_pair_metrics"]["pair_micro_f1"],
                   **result["decoder_usage"]}
            for partition, metrics in partitions.items():
                for metric in ("pair_micro_f1", "pair_micro_precision", "pair_micro_recall"):
                    row[f"{partition}_{metric}"] = metrics[metric]
            if method == "distilbert":
                selected = json.loads((root / "model_selection.json").read_text())
                row["selected_learning_rate"] = selected["learning_rate"]
            rows.append(row)
            counts = pd.read_csv(root / f"{condition}_row_counts.csv")
            seen_digest = audit_score_grid(root, condition, fold, selection, counts)
            if seen_identity is not None and seen_identity != seen_digest:
                raise ValueError("N/D changed representation-invariant seen candidates")
            seen_identity = seen_digest
            counts = counts.loc[counts.partition.eq("heldout")].sort_values("row_uid")
            if len(counts) != 1057 or counts.row_uid.duplicated().any():
                raise ValueError("Review-cluster evidence coverage mismatch")
            order = counts.row_uid.tolist()
            if review_order is not None and review_order != order:
                raise ValueError("Review identities differ across folds/systems")
            review_order = order
            matrix = counts[["tp", "fp", "fn"]].to_numpy(np.int64)
            if not np.isclose(f1(matrix.sum(axis=0)), row["heldout_pair_micro_f1"], atol=1e-12):
                raise ValueError("Row-count F1 reconstruction mismatch")
            counts_by_system.setdefault((method, policy, condition), {})[fold] = matrix
        print(f"AUDIT_RESULTS {index+1}/{len(expected)} {method}/{policy}/{fold}", flush=True)
    frame = pd.DataFrame(rows)
    systems = sorted(counts_by_system)
    tensor = np.stack([np.stack([counts_by_system[k][fold] for fold in config["folds"]]) for k in systems])
    balanced_samples, pooled_samples = bootstrap(tensor)
    point = f1(tensor.sum(axis=2)).mean(axis=1)
    pooled = f1(tensor.sum(axis=(1, 2)))
    aggregate = []
    for index, (method, policy, condition) in enumerate(systems):
        subset = frame.loc[frame.method.eq(method) & frame.policy.eq(policy) & frame.condition.eq(condition)]
        aggregate.append({"method": method, "policy": policy, "condition": condition, "folds": 12,
                          "heldout_pair_f1": point[index], "pooled_heldout_pair_f1": pooled[index],
                          "nominal_ci_low": np.quantile(balanced_samples[:, index], .025),
                          "nominal_ci_high": np.quantile(balanced_samples[:, index], .975),
                          "overall_pair_f1": subset.overall_pair_micro_f1.mean(),
                          "seen_pair_f1": subset.seen_pair_micro_f1.mean(),
                          "heldout_presence_f1": subset.heldout_presence_f1.mean(),
                          "heldout_presence_ap": subset.heldout_presence_ap.mean(),
                          "oracle_stage2_f1": subset.oracle_stage2_f1.mean(),
                          "two_sentiment_instances": int(subset.two_sentiment_instances.sum()),
                          "selected_review_aspects": int(subset.selected_review_aspects.sum())})
    comparisons = []

    def contrast(name, terms):
        estimate = 0.0
        samples = np.zeros(20000)
        for system, coefficient in terms:
            index = systems.index(system)
            estimate += coefficient * point[index]
            samples += coefficient * balanced_samples[:, index]
        comparisons.append({"comparison": name, "estimate": estimate,
                            "nominal_ci_low": np.quantile(samples, .025),
                            "nominal_ci_high": np.quantile(samples, .975),
                            "draws": 20000, "seed": 13, "evidence_role": "posthoc_validation_sensitivity"})

    for method in ("tfidf", "distilbert", "fewshot", "mixed_component_diagnostic"):
        for condition in ("D", "N"):
            contrast(f"{method}_{condition}_masked_minus_filtered",
                     [((method, policies[1], condition), 1), ((method, policies[0], condition), -1)])
        for policy in policies:
            contrast(f"{method}_{policy}_D_minus_N", [((method, policy, "D"), 1), ((method, policy, "N"), -1)])
        contrast(f"{method}_change_in_D_minus_N", [((method, policies[1], "D"), 1), ((method, policies[1], "N"), -1),
                                                   ((method, policies[0], "D"), -1), ((method, policies[0], "N"), 1)])
    for seed in config["tfidf"]["subset_seeds"]:
        for condition in ("D", "N"):
            subset = ("tfidf", f"size_matched_label_masked_s{seed}", condition)
            contrast(f"tfidf_{condition}_size_s{seed}_minus_filtered", [(subset, 1), (("tfidf", policies[0], condition), -1)])
            contrast(f"tfidf_{condition}_full_masked_minus_size_s{seed}", [(("tfidf", policies[1], condition), 1), (subset, -1)])
    analysis.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    for name, table in (("fold_metrics", frame), ("aggregate_metrics", pd.DataFrame(aggregate)),
                        ("paired_contrasts", pd.DataFrame(comparisons))):
        table.to_csv(analysis / f"{name}.csv", index=False)
        table.to_csv(public / f"{name}.csv", index=False)
    np.savez_compressed(analysis / "bootstrap_samples.npz", balanced=balanced_samples, pooled=pooled_samples,
                        systems=np.array(systems))
    write_json(analysis / "audit.json", {"status": "pass", "fold_receipts": len(expected), "condition_results": len(rows),
        "systems_conditions": len(systems), "folds": 12, "review_clusters": 1057, "failure_count": 0,
        "test_contract_count": 0, "third_sentiment_violations": 0, "f1_reconstruction": "pass",
        "score_to_review_confusion_reconstruction": "pass", "seen_ND_score_invariance": "pass",
        "synchronised_review_identity_sha256": hashlib_sha(review_order), "source_receipts": source_hashes})
    main_rows = [r for r in aggregate if r["policy"] in policies and r["condition"] == "D"]
    lines = ["# Training-policy sensitivity results", "", "Post-hoc validation-only evidence. Official test was not reopened.", "",
             "| Method | Training policy | Held-out pair F1 | Overall pair F1 |", "|---|---|---:|---:|"]
    lines += [f"| {r['method']} | {r['policy']} | {r['heldout_pair_f1']:.4f} | {r['overall_pair_f1']:.4f} |" for r in main_rows]
    lines += ["", "All twelve folds, both N/D conditions and all three TF-IDF size-control seeds are reported in the attached aggregate tables.",
              "", "The mixed component diagnostic retains the old review-filtered QLoRA sentiment decoder. It is not a fully label-masked-trained system.",
              "", "Intervals are nominal conditional review-cluster intervals for a post-hoc sensitivity study. They do not cover training-seed or taxonomy-population uncertainty.",
              "", "DistilBERT uses paired local replays with matched head initialisation. Frozen few-shot uses exact-source canonical replay. Neither replay overwrites historical formal results.",
              "", "No new prompts, thresholds, models or subset seeds were selected using held-out outcomes. No QLoRA retraining was performed."]
    (analysis / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / f"docs/experiments/{STUDY_ID}_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt(analysis, {"study_id": STUDY_ID, "config_sha256": canonical_sha256(config),
                       "analysis_code_sha256": sha256_file(Path(__file__))})
    backup_job(analysis, ROOT.parent / "cloud_backups" / STUDY_ID / "analysis")
    print("COMPLETE_ALL_SENSITIVITY_ANALYSIS", flush=True)


def hashlib_sha(value):
    import hashlib
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


if __name__ == "__main__":
    main()
