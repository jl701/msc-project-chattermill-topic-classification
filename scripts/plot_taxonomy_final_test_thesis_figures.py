"""Render publication figures from frozen, review-text-free thesis tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "docs" / "thesis_figure_data" / "taxonomy_final_test_v1"
DEFAULT_OUTPUT = ROOT / "thesis" / "figures" / "taxonomy_final_test_v1"

UCL_BLUE = "#002855"
UCL_CYAN = "#00AEEF"
UCL_PURPLE = "#6F2C91"
UCL_RED = "#D62728"
UCL_GREY = "#667085"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def _save(fig: plt.Figure, output_root: Path, stem: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_primary(data_root: Path, output_root: Path) -> None:
    summary = pd.read_csv(data_root / "official_l2_model_condition_summary.csv")
    confirm = pd.read_csv(data_root / "official_confirmatory_results.csv")
    order = [
        "qwen_candidate_pair_qlora",
        "frozen_qwen_few_shot",
        "frozen_qwen_few_shot_stage1__qlora_stage2",
    ]
    labels = ["QLoRA", "Frozen Qwen\nfew-shot", "Fixed stage-wise\ncomposition"]
    points = [
        float(
            summary.loc[
                (summary.method_id == method) & (summary.condition == "D"),
                "heldout_pair_f1_fold_mean",
            ].iloc[0]
        )
        for method in order
    ]
    colours = [UCL_PURPLE, UCL_CYAN, UCL_BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.35), gridspec_kw={"width_ratios": [1.05, 1]})
    axes[0].bar(np.arange(3), points, color=colours, width=0.62)
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylim(0.44, 0.525)
    axes[0].set_ylabel("Aspect-balanced held-out pair F1")
    axes[0].set_title("A. Registered Level 2 D systems")
    for index, value in enumerate(points):
        axes[0].text(index, value + 0.002, f"{value:.3f}", ha="center", va="bottom")

    y = np.arange(len(confirm))[::-1]
    diff = confirm["point_difference"].to_numpy()
    low = confirm["ci_lower"].to_numpy()
    high = confirm["ci_upper"].to_numpy()
    axes[1].axvline(0, color="black", linewidth=0.9)
    axes[1].errorbar(
        diff,
        y,
        xerr=np.vstack([diff - low, high - diff]),
        fmt="o",
        color=UCL_BLUE,
        ecolor=UCL_BLUE,
        capsize=3,
    )
    axes[1].set_yticks(y, ["H1: vs few-shot", "H2: vs QLoRA"])
    axes[1].set_xlabel("Composition minus comparator F1")
    axes[1].set_title("B. Synchronized 95% bootstrap intervals")
    axes[1].set_xlim(-0.006, 0.048)
    for yi, value, upper in zip(y, diff, high, strict=True):
        axes[1].text(upper + 0.001, yi, f"{value:+.3f}", va="center")
    fig.suptitle("Protocol-locked official-test confirmation", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, output_root, "official_primary_confirmation")


def plot_fold_effects(data_root: Path, output_root: Path) -> None:
    folds = pd.read_csv(data_root / "official_l2_confirmatory_fold_results.csv")
    context = pd.read_csv(data_root / "official_l2_aspect_context.csv")
    names = context.set_index("fold_id")["heldout_aspect"].to_dict()
    short = {
        "Account management: Account access": "Account access",
        "Company brand: Competitor": "Competitor",
        "Company brand: General satisfaction": "General satisfaction",
        "Company brand: Reviews": "Reviews",
        "Logistics rides: Speed": "Speed",
        "Online experience: App website": "App/website",
        "Purchase booking experience: Ease of use": "Ease of use",
        "Staff support: Attitude of staff": "Staff attitude",
        "Staff support: Email": "Email",
        "Staff support: Phone": "Phone",
        "Value: Discounts promotions": "Discounts/promotions",
        "Value: Price value for money": "Price/value",
    }
    labels = [short[names[fold]] for fold in folds.fold_id]
    y = np.arange(len(folds))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.2), sharey=True)
    for axis, column, title, colour in (
        (axes[0], "fixed_minus_few_shot", "A. Composition minus few-shot", UCL_CYAN),
        (axes[1], "fixed_minus_qlora", "B. Composition minus QLoRA", UCL_PURPLE),
    ):
        values = folds[column].to_numpy()
        axis.axvline(0, color="black", linewidth=0.8)
        axis.hlines(y, 0, values, color=colour, linewidth=1.5)
        axis.scatter(values, y, color=colour, s=25, zorder=3)
        axis.set_title(title)
        axis.set_xlabel("Fold-level held-out pair F1 difference")
        limit = max(abs(values.min()), abs(values.max())) * 1.14
        axis.set_xlim(-limit, limit)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Official-test effects vary materially by held-out aspect", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_root, "official_fold_effects")


def plot_stage_diagnostics(data_root: Path, output_root: Path) -> None:
    stage = pd.read_csv(data_root / "official_l2_stage_diagnostics.csv")
    stage = stage[stage.condition == "D"].copy()
    order = [
        "Fixed stage-wise composition",
        "Frozen Qwen few-shot",
        "QLoRA",
        "Frozen Qwen zero-shot",
        "TF-IDF",
        "Frozen E5",
        "DCWT",
        "DistilBERT",
    ]
    stage["method"] = pd.Categorical(stage["method"], order, ordered=True)
    stage = stage.sort_values("method")
    y = np.arange(len(stage))
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.scatter(
        stage.heldout_presence_f1_fold_mean,
        y - 0.12,
        label="Stage 1: thresholded presence F1",
        color=UCL_CYAN,
        s=34,
    )
    ax.scatter(
        stage.oracle_gated_stage2_pair_f1_fold_mean,
        y + 0.12,
        label="Stage 2: oracle-gated pair F1",
        color=UCL_PURPLE,
        marker="s",
        s=30,
    )
    ax.set_yticks(y, stage.method.astype(str))
    ax.invert_yaxis()
    ax.set_xlim(0.05, 0.94)
    ax.set_xlabel("Aspect-balanced fold mean")
    ax.set_title("Stage diagnostics separate gating from conditional sentiment", fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    _save(fig, output_root, "official_stage_diagnostics")


def plot_description_effects(data_root: Path, output_root: Path) -> None:
    effects = pd.read_csv(data_root / "official_l2_description_effects.csv")
    effects = effects.sort_values("heldout_D_minus_N")
    y = np.arange(len(effects))
    values = effects.heldout_D_minus_N.to_numpy()
    colours = [UCL_BLUE if value >= 0 else UCL_RED for value in values]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.axvline(0, color="black", linewidth=0.9)
    ax.hlines(y, 0, values, color=colours, linewidth=1.8)
    ax.scatter(values, y, color=colours, s=34, zorder=3)
    ax.set_yticks(y, effects.method)
    ax.set_xlabel("Held-out pair F1 difference (minimal definition D minus name N)")
    ax.set_title("Minimal definitions have model-dependent official-test effects", fontweight="bold")
    ax.set_xlim(min(-0.03, values.min() - 0.008), max(0.062, values.max() + 0.008))
    fig.tight_layout()
    _save(fig, output_root, "official_description_effects")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _style()
    plot_primary(args.data_dir, args.output_dir)
    plot_fold_effects(args.data_dir, args.output_dir)
    plot_stage_diagnostics(args.data_dir, args.output_dir)
    plot_description_effects(args.data_dir, args.output_dir)
    print(f"Wrote eight figure files to {args.output_dir}")


if __name__ == "__main__":
    main()
