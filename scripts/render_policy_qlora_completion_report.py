"""Render complete audited policy results, or explicitly watermarked synthetic QA.

No model invocation, statistical re-selection, label loading or existing-thesis edits.
Output directories are immutable. A receipt is written only after source rechecks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

POLICIES = ("review_filtered", "label_masked_all_reviews")
METHODS = ("fewshot", "qlora", "composition")
LABELS = {"fewshot": "Few-shot Qwen", "qlora": "QLoRA", "composition": "Fixed composition"}
FILES = {"audit.json", "input_manifest.json", "aggregate_metrics.csv", "fold_metrics.csv",
         "paired_contrasts.csv", "review_confusion_and_bootstrap.npz"}
ROLE = "post_test_validation_only_training_policy_sensitivity"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate(root, synthetic=False):
    root = Path(root).resolve()
    hashes = read(root / "output_hashes.json")
    if set(hashes) != FILES:
        raise ValueError("Incomplete or unexpected analysis file inventory")
    if any(digest(root / name) != value for name, value in hashes.items()):
        raise ValueError("Analysis output hash mismatch")
    audit, manifest = read(root / "audit.json"), read(root / "input_manifest.json")
    required = {"status": "PASS", "source_fold_units": 48, "source_condition_grids": 96,
                "derived_composition_grids": 48, "reported_fold_condition_systems": 144,
                "failure_count": 0, "test_contract_count": 0, "selection_retuned": False,
                "same_policy_components": True, "N_D_seen_scores_identical": True, "evidence_role": ROLE}
    if any(audit.get(k) != value for k, value in required.items()):
        raise ValueError("Complete study integrity PASS required")
    if (manifest.get("include_official_test") is not False or manifest.get("test_contract_count") != 0
            or manifest.get("allowed_splits") != ["validation"]
            or bool(manifest.get("synthetic_fixture", False)) != synthetic):
        raise ValueError("Validation boundary or explicit synthetic mode mismatch")
    aggregate = pd.read_csv(root / "aggregate_metrics.csv")
    folds = pd.read_csv(root / "fold_metrics.csv")
    contrasts = pd.read_csv(root / "paired_contrasts.csv")
    keys = ["method", "policy", "condition"]
    expected = {(m, p, c) for m in METHODS for p in POLICIES for c in ("N", "D")}
    if (len(aggregate) != 12 or set(aggregate[keys].itertuples(index=False, name=None)) != expected
            or not aggregate.folds.eq(12).all()):
        raise ValueError("Twelve complete aggregate system-conditions required")
    expected_folds = {(*key, f"l2-a{i:02d}") for key in expected for i in range(1, 13)}
    if len(folds) != 144 or set(folds[keys + ["fold_id"]].itertuples(index=False, name=None)) != expected_folds:
        raise ValueError("All 144 distinct fold rows required")
    for frame in (aggregate, folds, contrasts):
        if not np.isfinite(frame.select_dtypes(include="number").to_numpy(float)).all():
            raise ValueError("Non-finite reporting value")
    required_contrasts = {f"{m}_{c}_masked_minus_filtered" for m in METHODS for c in ("N", "D")}
    required_contrasts |= {f"{p}_{c}_composition_minus_{m}" for p in POLICIES for c in ("N", "D") for m in METHODS[:2]}
    required_contrasts |= {f"{c}_change_in_composition_gain_over_{m}" for c in ("N", "D") for m in METHODS[:2]}
    if (len(contrasts) != 18 or set(contrasts.comparison) != required_contrasts
            or not contrasts.draws.eq(20000).all() or not contrasts.seed.eq(13).all()):
        raise ValueError("Duplicate contrast or changed resampling settings")
    names = [f"{m}_D_masked_minus_filtered" for m in METHODS]
    names += [f"{p}_D_composition_minus_{m}" for p in POLICIES for m in METHODS[:2]]
    if not set(names).issubset(contrasts.comparison):
        raise ValueError("Missing paired comparison required by figure")
    if (contrasts.nominal_ci_low > contrasts.nominal_ci_high).any():
        raise ValueError("Reversed confidence interval")
    return hashes, aggregate, contrasts.set_index("comparison"), names


def render(root, output, synthetic=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new immutable report output directory")
    hashes, aggregate, contrasts, names = validate(root, synthetic)
    # Numerical scope and file validation happen before a figure is created.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    core = aggregate.loc[aggregate.condition.eq("D")].set_index(["policy", "method"])
    core = core.loc[[(p, m) for p in POLICIES for m in METHODS]].reset_index()
    columns = ["policy", "method", "heldout_pair_f1", "heldout_presence_f1", "oracle_stage2_f1"]
    core = core[columns]
    output.mkdir(parents=True)
    core.to_csv(output / "training_policy_core.csv", index=False)
    title = "SYNTHETIC PREVIEW - NOT RESEARCH RESULTS" if synthetic else "Training-policy dependence on validation"
    lines = ["% " + title, r"\begin{table}[tbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{" + ("Synthetic layout preview, not research results." if synthetic else
             "Validation comparison across twelve held-out aspects with names and minimal definitions. Oracle sentiment supplies the correct aspects and is diagnostic only.") + "}",
             r"\label{tab:training-policy-core}", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Training policy & System & \shortstack{Held-out\\pair F1} & \shortstack{Presence\\F1} & \shortstack{Oracle\\sentiment F1} \\", r"\midrule"]
    for row in core.itertuples(index=False):
        policy = "Remove reviews" if row.policy == POLICIES[0] else "Retain reviews"
        lines.append(f"{policy} & {LABELS[row.method]} & {row.heldout_pair_f1:.4f} & {row.heldout_presence_f1:.4f} & {row.oracle_stage2_f1:.4f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "training_policy_core.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Stack panels so labels remain readable at a thesis text width, not only
    # when a wide dashboard image is opened full-screen.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, layout="constrained")
    panels = [(names[:3], [LABELS[m] for m in METHODS], "Retain minus remove"),
              (names[3:], [f"{'Remove' if p == POLICIES[0] else 'Retain'}: vs {LABELS[m]}" for p in POLICIES for m in METHODS[:2]],
               "Composition gain within each policy")]
    for axis, (selected, labels, heading) in zip(axes, panels):
        values = contrasts.loc[selected]
        for y, (_, row) in enumerate(values.iterrows()):
            axis.plot([row.nominal_ci_low, row.nominal_ci_high], [y, y], color="#28658a", lw=2.2)
            axis.scatter([row.estimate], [y], color="#163d57", s=37, zorder=3)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_ylim(len(labels) - .5, -.5)
        axis.axvline(0, color="#777777", lw=.8, ls="--")
        axis.set_title(heading, fontsize=11, pad=14)
        axis.grid(axis="x", color="#e7e7e7")
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[-1].set_xlabel("Held-out pair F1 difference")
    fig.suptitle(title, fontsize=10, weight="bold", color="#a32626" if synthetic else "#163d57")
    fig.supxlabel("Dots: paired differences. Lines: nominal 95% review-cluster bootstrap intervals.\nNames and minimal definitions (D), twelve folds.", fontsize=8)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(output / f"training_policy_core_effects.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for name in ("aggregate_metrics.csv", "fold_metrics.csv", "paired_contrasts.csv"):
        shutil.copyfile(root / name, output / ("appendix_" + name))
    if hashes != read(root / "output_hashes.json") or any(digest(root / name) != value for name, value in hashes.items()):
        raise ValueError("Analysis changed during rendering, report not released")
    receipt = {"status": "SYNTHETIC_QA_ONLY" if synthetic else "PASS_RENDERED_ASSETS",
               "synthetic_preview": synthetic, "evidence_role": ROLE, "source_directory": str(root),
               "source_hashes": hashes, "test_contract_count": 0, "thesis_files_modified": False,
               "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-preview", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render(args.analysis_root, args.output, args.synthetic_preview), indent=2))


if __name__ == "__main__":
    main()
