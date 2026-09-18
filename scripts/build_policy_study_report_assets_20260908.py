"""Generate report assets from completed, sealed validation-only analyses."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from msc_project.experiments.taxonomy_training_policy_sensitivity import verify_receipt

PARENT = ROOT / "outputs/experimental/taxonomy_training_policy_sensitivity_v1"
EXT = ROOT / "outputs/experimental/taxonomy_training_policy_extension_v1"
DEST = ROOT.parent / "Thesis/Training_Policy_Study_20260908"
NAMES = {"tfidf": "TF-IDF", "distilbert": "DistilBERT", "fewshot": "Qwen few-shot",
         "dcwt_kernel_ridge": "DCWT (kernel ridge)", "e5_base_v2": "Frozen E5",
         "frozen_qwen_candidate_pair": "Qwen zero-shot",
         "mixed_component_diagnostic": "Mixed-component diagnostic",
         "dcwt_nearest_description_weight": "DCWT nearest-description control",
         "dcwt_mean_seen_weight": "DCWT mean-weight control",
         "dcwt_cosine_barycentric_weight": "DCWT cosine-weight control"}
MAIN = ["tfidf", "dcwt_kernel_ridge", "distilbert", "fewshot", "e5_base_v2", "frozen_qwen_candidate_pair", "mixed_component_diagnostic"]


def main():
    complete = json.loads((PARENT / "postprocess_20260908/completion_audit.json").read_text())
    if complete["status"] != "pass" or complete["failure_count"] or complete["test_contract_count"]:
        raise RuntimeError("Completion gate not passed")
    verify_receipt(PARENT / "analysis")
    verify_receipt(EXT / "analysis")
    audit = json.loads((EXT / "analysis/audit.json").read_text())
    if (audit["fold_policy_method_units"], audit["condition_records"], audit["system_conditions"]) != (276, 552, 46):
        raise ValueError("Unexpected combined scope")
    agg = pd.read_csv(EXT / "analysis/aggregate_metrics.csv")
    contrasts = pd.read_csv(EXT / "analysis/paired_contrasts.csv")
    folds = pd.read_csv(EXT / "analysis/fold_metrics.csv")
    samples = np.load(EXT / "analysis/bootstrap_samples.npz", allow_pickle=False)
    systems = [tuple(row) for row in samples["systems"].tolist()]
    if samples["balanced"].shape != (20000, 46) or not np.isfinite(samples["balanced"]).all():
        raise ValueError("Bootstrap sample shape/finiteness mismatch")
    for record in contrasts.itertuples(index=False):
        ia = systems.index((record.method, "review_filtered", record.condition))
        ib = systems.index((record.method, "label_masked_all_reviews", record.condition))
        interval = np.quantile(samples["balanced"][:, ib]-samples["balanced"][:, ia], [.025, .975])
        if not np.allclose(interval, [record.nominal_ci_low, record.nominal_ci_high], rtol=0, atol=1e-12):
            raise ValueError("Saved paired interval cannot be reconstructed")
    DEST.mkdir(exist_ok=False)
    table = []
    diagnostics = []
    for method in MAIN + [k for k in NAMES if k not in MAIN]:
        for condition in ("D", "N"):
            selected = agg.loc[agg.method.eq(method) & agg.condition.eq(condition)].set_index("policy")
            a, b = selected.loc["review_filtered"], selected.loc["label_masked_all_reviews"]
            delta = contrasts.loc[contrasts.method.eq(method) & contrasts.condition.eq(condition)].iloc[0]
            if not np.isclose(b.heldout_pair_f1-a.heldout_pair_f1, delta.estimate, rtol=0, atol=1e-12):
                raise ValueError("Point difference mismatch")
            ff = folds.loc[folds.method.eq(method) & folds.condition.eq(condition)].pivot(index="fold_id", columns="policy", values="heldout_pair_f1")
            changes = ff.label_masked_all_reviews-ff.review_filtered
            table.append({"method": method, "name": NAMES[method], "condition": condition,
                          "filtered_f1": a.heldout_pair_f1, "masked_f1": b.heldout_pair_f1,
                          "delta": delta.estimate, "ci_low": delta.nominal_ci_low, "ci_high": delta.nominal_ci_high,
                          "improved_folds": int((changes > 1e-12).sum()), "tied_folds": int((changes.abs() <= 1e-12).sum()),
                          "worsened_folds": int((changes < -1e-12).sum())})
            for metric in ("overall_pair_f1", "seen_pair_f1", "presence_f1", "presence_ap", "oracle_stage2_f1"):
                diagnostics.append({"method": method, "condition": condition, "metric": metric,
                                    "filtered": a[metric], "masked": b[metric], "delta": b[metric]-a[metric]})
    headline = pd.DataFrame(table)
    headline.to_csv(DEST / "headline_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(DEST / "stage_diagnostics.csv", index=False)
    # Fixed order and scope, not a ranking chosen from these results.
    plotted = MAIN[:4] + ["mixed_component_diagnostic"]
    data = headline.loc[headline.condition.eq("D")].set_index("method").loc[plotted]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    for i, row in enumerate(data.itertuples()):
        color = "#167665" if row.ci_low > 0 else "#315e9e" if row.ci_high < 0 else "#697586"
        ax.plot([row.ci_low, row.ci_high], [i, i], color=color, lw=3, solid_capstyle="round")
        ax.scatter(row.delta, i, color=color, s=62, zorder=3)
        ax.annotate(f"{row.delta:+.4f}  [{row.ci_low:+.4f}, {row.ci_high:+.4f}]",
                    xy=(1.01, i), xycoords=("axes fraction", "data"), va="center", fontsize=10)
    ax.axvline(0, color="#222d3a", lw=1)
    ax.axhline(3.5, color="#cfd7df", linestyle="--", lw=1)
    ax.set_yticks(range(5), [NAMES[m] for m in plotted])
    ax.invert_yaxis()
    ax.set_ylim(4.6, -0.6)
    low, high = min(data.ci_low.min(), 0), max(data.ci_high.max(), 0)
    padding = max((high-low)*.12, .008)
    ax.set_xlim(low-padding, high+padding)
    ax.set_xlabel("Masking minus review filtering: held-out pair F1")
    ax.grid(axis="x", alpha=.18)
    fig.suptitle("Does keeping reviews improve unseen-aspect performance?", x=.025, ha="left", fontsize=17, fontweight="bold")
    fig.text(.025, .89, "L2-D | 12-fold aspect-balanced mean | paired 20,000-draw review bootstrap", fontsize=11, color="#566274")
    fig.text(.025, .035, "Nominal conditional 95% intervals, not confirmatory tests. Mixed diagnostic retains the old QLoRA decoder.\nFrozen E5 and zero-shot Qwen are unchanged by construction and are omitted from this effect plot.", fontsize=9, color="#566274")
    fig.subplots_adjust(left=.255, right=.65, top=.81, bottom=.18)
    for suffix in ("png", "svg"):
        fig.savefig(DEST / f"policy_effects_D.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)
    matrix = []
    for method in plotted:
        values = folds.loc[folds.method.eq(method) & folds.condition.eq("D")].pivot(index="fold_id", columns="policy", values="heldout_pair_f1").sort_index()
        matrix.append((values.label_masked_all_reviews-values.review_filtered).to_numpy())
    matrix = np.array(matrix)
    span = max(np.abs(matrix).max(), .01)
    fig, ax = plt.subplots(figsize=(12, 5))
    heat = ax.imshow(matrix, cmap="RdBu", vmin=-span, vmax=span, aspect="auto")
    ax.set_yticks(range(5), [NAMES[m] for m in plotted])
    ax.set_xticks(range(12), [f"a{i:02d}" for i in range(1, 13)])
    for i in range(5):
        for j in range(12):
            ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(matrix[i,j]) > span*.55 else "#182333")
    fig.colorbar(heat, ax=ax, label="Masked minus filtered F1", fraction=.025, pad=.02)
    ax.set_title("The average can hide different responses across held-out aspects", loc="left", fontsize=15, pad=17)
    fig.text(.02, .025, "L2-D validation only. Each column is one fixed held-out aspect. Positive = masking higher, negative = filtering higher.", fontsize=9)
    fig.subplots_adjust(left=.24, right=.93, top=.82, bottom=.16)
    for suffix in ("png", "svg"):
        fig.savefig(DEST / f"fold_heterogeneity_D.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)
    checks = {"status": "pass", "source_analysis_audit": audit["status"], "paired_intervals_reconstructed": len(contrasts),
              "bootstrap_draws": 20000, "headline_records": len(headline), "no_new_model_selection": True,
              "source_files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                               [EXT / "analysis/aggregate_metrics.csv", EXT / "analysis/paired_contrasts.csv", EXT / "analysis/fold_metrics.csv"]}}
    (DEST / "report_asset_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(headline.to_json(orient="records", indent=2))
    print("REPORT_ASSETS_READY", DEST)


if __name__ == "__main__":
    main()
