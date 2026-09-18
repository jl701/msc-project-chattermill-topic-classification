"""Combine complete audited validation evidence without reopening official test."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_taxonomy_training_policy_extension as run
from analyse_taxonomy_training_policy_sensitivity import audit_score_grid
from msc_project.evaluation.resampling import (
    review_cluster_bootstrap as bootstrap,
    f1_from_counts as f1,
)


def main():
    config, _, validation, folds, _ = run.load_extension()
    output = ROOT / config["output_root"]
    parent = ROOT / "outputs/experimental" / run.PARENT_ID
    target = output / "analysis"
    if target.exists():
        raise RuntimeError("Existing combined analysis requires inspection")
    run.verify_receipt(parent / "analysis")
    parent_audit = json.loads((parent / "analysis/audit.json").read_text())
    if (parent_audit["status"] != "pass" or parent_audit["fold_receipts"] != 132
            or parent_audit["failure_count"] or parent_audit["test_contract_count"]):
        raise ValueError("Parent analysis incomplete")
    # Verify every source sealed by the completed parent analysis.
    source_hashes = dict(parent_audit["source_receipts"])
    for path, digest in source_hashes.items():
        if run.sha256_file(Path(path)) != digest:
            raise ValueError("Parent source receipt changed")
        run.verify_receipt(Path(path).parent)
    sources = []
    frame = pd.read_csv(parent / "analysis/fold_metrics.csv")
    for method, policy in frame[["method", "policy"]].drop_duplicates().itertuples(index=False, name=None):
        for fold in folds:
            sources.append((method, policy, fold.fold_id, parent / method / policy / fold.fold_id, "parent"))
    for method in config["invariance"]["methods"]:
        for policy in config["policies"]:
            for fold in folds:
                sources.append((method, policy, fold.fold_id, output / "invariance" / method / policy / fold.fold_id, "invariant"))
    for policy in config["policies"]:
        for fold in folds:
            root = output / "dcwt" / policy / fold.fold_id
            run.verify_receipt(root)
            for family in run.legacy.GENERATOR_FAMILIES:
                sources.append(("dcwt_" + family, policy, fold.fold_id, root / "generators" / family, "dcwt"))
    if len(sources) != 276:
        raise ValueError("Combined scope must contain 276 fold-policy-method units")
    rows, counts_by_system = [], {}
    review_order = sorted(validation.row_uid)
    for method, policy, fold, root, role in sources:
        payload = run.verify_receipt(root)
        if payload["contract"].get("method") != method or payload["contract"].get("policy") != policy or payload["contract"].get("fold_id") != fold:
            raise ValueError("Receipt scope mismatch")
        if role != "parent" and payload["contract"].get("config_sha256") != run.canonical_sha256(config):
            raise ValueError("Extension config changed")
        source_hashes[str(root / "receipt.json")] = run.sha256_file(root / "receipt.json")
        selection = json.loads((root / "selection.json").read_text())
        seen_hash = None
        for condition in ("D", "N"):
            result = json.loads((root / f"{condition}_metrics.json").read_text())
            counts = pd.read_csv(root / f"{condition}_row_counts.csv")
            if role == "dcwt":
                digest = audit_score_grid(root, condition, fold, selection, counts)
                if seen_hash is not None and seen_hash != digest:
                    raise ValueError("DCWT N/D shared seen scores differ")
                seen_hash = digest
            counts = counts.loc[counts.partition.eq("heldout")].sort_values("row_uid")
            if counts.row_uid.tolist() != review_order or counts.row_uid.duplicated().any():
                raise ValueError("Review evidence scope mismatch")
            values = counts[["tp", "fp", "fn"]].to_numpy(np.int64)
            parts = result["L2_E"]["partitions"]
            if (values < 0).any() or not np.isclose(f1(values.sum(axis=0)), parts["heldout"]["pair_micro_f1"], atol=1e-12):
                raise ValueError("Invalid confusion evidence")
            if result["L2_E"]["maximum_sentiments_per_selected_aspect"] > 2:
                raise ValueError("Third sentiment violation")
            counts_by_system.setdefault((method, policy, condition), {})[fold] = values
            rows.append({"method": method, "policy": policy, "fold_id": fold, "condition": condition,
                "evidence_role": "structural_invariance_not_independent_replication" if role == "invariant" else "paired_local_policy_replay",
                "heldout_pair_f1": parts["heldout"]["pair_micro_f1"],
                "overall_pair_f1": parts["overall"]["pair_micro_f1"],
                "seen_pair_f1": parts["seen"]["pair_micro_f1"],
                "presence_f1": result["L2_S"]["aspect_presence"]["f1"],
                "presence_ap": result["L2_S"]["aspect_presence"]["average_precision"],
                "oracle_stage2_f1": result["L2_S"]["oracle_aspect_gated_sentiment"]["capped_two_pair_metrics"]["pair_micro_f1"],
                "aspect_threshold": selection["aspect_threshold"],
                "runner_up_threshold": selection["second_sentiment_threshold"], **result["decoder_usage"]})
    table = pd.DataFrame(rows)
    systems = sorted(counts_by_system)
    tensor = np.stack([np.stack([counts_by_system[key][f.fold_id] for f in folds]) for key in systems])
    balanced, pooled = bootstrap(tensor, draws=20000, seed=13)
    points = f1(tensor.sum(axis=2)).mean(axis=1)
    pooled_points = f1(tensor.sum(axis=(1, 2)))
    aggregate = []
    for i, (method, policy, condition) in enumerate(systems):
        subset = table.loc[table.method.eq(method) & table.policy.eq(policy) & table.condition.eq(condition)]
        aggregate.append({"method": method, "policy": policy, "condition": condition, "folds": 12,
            "heldout_pair_f1": points[i], "pooled_heldout_pair_f1": pooled_points[i],
            "nominal_ci_low": np.quantile(balanced[:, i], .025), "nominal_ci_high": np.quantile(balanced[:, i], .975),
            **{name: subset[name].mean() for name in ("overall_pair_f1", "seen_pair_f1", "presence_f1", "presence_ap", "oracle_stage2_f1")},
            "two_sentiment_instances": int(subset.two_sentiment_instances.sum()),
            "selected_review_aspects": int(subset.selected_review_aspects.sum())})
    comparisons = []
    for method in sorted(table.method.unique()):
        for condition in ("D", "N"):
            a = systems.index((method, "review_filtered", condition))
            b = systems.index((method, "label_masked_all_reviews", condition))
            if method in config["invariance"]["methods"] and not np.array_equal(tensor[a], tensor[b]):
                raise ValueError("Frozen training-policy invariant changed")
            delta = balanced[:, b] - balanced[:, a]
            comparisons.append({"method": method, "condition": condition, "contrast": "masked_minus_filtered",
                "estimate": points[b] - points[a], "nominal_ci_low": np.quantile(delta, .025),
                "nominal_ci_high": np.quantile(delta, .975), "evidence_role": "structural_invariance" if method in config["invariance"]["methods"] else "posthoc_validation_sensitivity"})
    target.mkdir(parents=True)
    public = ROOT / config["report_root"]
    public.mkdir(parents=True, exist_ok=True)
    for name, data in (("fold_metrics", table), ("aggregate_metrics", pd.DataFrame(aggregate)), ("paired_contrasts", pd.DataFrame(comparisons))):
        data.to_csv(target / f"{name}.csv", index=False)
        data.to_csv(public / f"{name}.csv", index=False)
    np.savez_compressed(target / "bootstrap_samples.npz", balanced=balanced, pooled=pooled, systems=np.asarray(systems))
    run.write_json(target / "audit.json", {"status": "pass", "fold_policy_method_units": len(sources),
        "condition_records": len(rows), "system_conditions": len(systems), "review_clusters": 1057,
        "failure_count": 0, "test_contract_count": 0, "source_receipts": source_hashes,
        "parent_audit_sha256": run.sha256_file(parent / "analysis/audit.json"), "frozen_zero_delta": "structural, not replication"})
    lines = ["# Extended training-policy sensitivity", "", "Validation-only, proposed after official test. No official-test outcomes were reopened.", "",
        "Six original non-QLoRA families are covered. E5 and zero-shot Qwen are invariant references, not independent reruns.",
        "The mixed-component diagnostic retains the original review-filtered QLoRA Stage 2. It is not a label-masked QLoRA result.", "",
        "All twelve folds and both N/D conditions are reported. Kernel-ridge DCWT is the registered primary generator. The three original controls are retained, not selected using held-out outcomes.",
        "Intervals are nominal conditional review-cluster intervals, not confirmatory tests or training-seed uncertainty.", "",
        "| Method | D masked-minus-filtered F1 | Nominal 95% interval |", "|---|---:|---|"]
    for r in comparisons:
        if r["condition"] == "D":
            lines.append(f"| {r['method']} | {r['estimate']:+.4f} | [{r['nominal_ci_low']:+.4f}, {r['nominal_ci_high']:+.4f}] |")
    (target / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / f"docs/experiments/{run.EXT_ID}_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run.receipt(target, run.contract(config, "combined_analysis", analysis_sha256=run.sha256_file(Path(__file__))))
    run.backup_job(target, ROOT.parent / "cloud_backups" / run.EXT_ID / "analysis")
    print("COMPLETE_COMBINED_ANALYSIS", flush=True)


if __name__ == "__main__":
    main()
