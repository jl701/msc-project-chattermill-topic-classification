from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DOC = PROJECT_ROOT / "docs" / "thesis_result_tables.md"
FIGURE_DIR = PROJECT_ROOT / "docs" / "thesis_figure_data"


PROTOCOL_LADDER = [
    {
        "section": "closed-topic",
        "system": "Word+char TF-IDF + Linear SVM",
        "protocol": "provided FABSA split",
        "split": "test",
        "pair_samples_f1": 0.7090,
        "pair_micro_f1": 0.7042,
        "pair_macro_f1": 0.4207,
        "aspect_samples_f1": 0.7736,
        "notes": "Fixed taxonomy reference point.",
    },
    {
        "section": "closed-topic",
        "system": "DistilBERT",
        "protocol": "provided FABSA split",
        "split": "test",
        "pair_samples_f1": 0.7803,
        "pair_micro_f1": 0.7738,
        "pair_macro_f1": 0.5377,
        "aspect_samples_f1": 0.8185,
        "notes": "Strongest closed-topic encoder baseline.",
    },
    {
        "section": "held-out organisation",
        "system": "Word+char TF-IDF + Linear SVM",
        "protocol": "held-out organisation",
        "split": "test",
        "pair_samples_f1": 0.7026,
        "pair_micro_f1": 0.6920,
        "pair_macro_f1": 0.3626,
        "aspect_samples_f1": 0.7629,
        "notes": "Domain-shift baseline; taxonomy is not held out.",
    },
    {
        "section": "held-out organisation",
        "system": "DistilBERT",
        "protocol": "held-out organisation",
        "split": "test",
        "pair_samples_f1": 0.7575,
        "pair_micro_f1": 0.7600,
        "pair_macro_f1": 0.4035,
        "aspect_samples_f1": 0.7986,
        "notes": "Validation-selected held-out-organisation encoder.",
    },
    {
        "section": "fixed held-out aspect",
        "system": "Candidate-label lexical TF-IDF + global sentiment",
        "protocol": "fixed three held-out aspects, example-filtered",
        "split": "test",
        "pair_samples_f1": 0.4626,
        "pair_micro_f1": 0.4596,
        "pair_macro_f1": 0.3703,
        "aspect_samples_f1": 0.5231,
        "notes": "Lexical lower bound for controlled unseen-aspect split.",
    },
    {
        "section": "fixed held-out aspect",
        "system": "Candidate-aspect DistilBERT + DistilBERT sentiment",
        "protocol": "fixed three held-out aspects, example-filtered",
        "split": "test",
        "pair_samples_f1": 0.6071,
        "pair_micro_f1": 0.5917,
        "pair_macro_f1": 0.4890,
        "aspect_samples_f1": 0.6651,
        "notes": "Strongest local fixed held-out-aspect baseline.",
    },
    {
        "section": "fixed held-out aspect",
        "system": "Qwen3-4B indexed zero-shot",
        "protocol": "fixed three held-out aspects",
        "split": "test",
        "pair_samples_f1": 0.5374,
        "pair_micro_f1": 0.5300,
        "pair_macro_f1": 0.4374,
        "aspect_samples_f1": 0.6340,
        "notes": "Open-weight zero-shot fixed split; not fine-tuned.",
    },
    {
        "section": "fixed held-out aspect",
        "system": "Qwen3-4B QLoRA indexed SFT",
        "protocol": "fixed three held-out aspects, example-filtered",
        "split": "test",
        "pair_samples_f1": 0.5528,
        "pair_micro_f1": 0.5552,
        "pair_macro_f1": 0.4393,
        "aspect_samples_f1": 0.6192,
        "notes": "One-epoch fixed-split QLoRA; modest adaptation gain over Qwen zero-shot, below the strongest local baseline.",
    },
]


LOAO_ROBUSTNESS = [
    {
        "system": "Lexical TF-IDF + global sentiment",
        "scope": "all-row LOAO, micro-F1 threshold selection, example-filtered",
        "split": "test",
        "pair_samples_f1_mean": 0.0903,
        "pair_micro_f1_mean": 0.3780,
        "pair_macro_f1_mean": "",
        "precision_mean": 0.3778,
        "recall_mean": 0.4895,
        "fp_rows_per_100_mean": 17.4753,
        "notes": "Best documented lexical all-row LOAO detection diagnostic.",
    },
    {
        "system": "Candidate-aspect DistilBERT + DistilBERT sentiment",
        "scope": "all-row LOAO, pair-micro-F1 threshold selection, example-filtered",
        "split": "test",
        "pair_samples_f1_mean": 0.0550,
        "pair_micro_f1_mean": 0.3128,
        "pair_macro_f1_mean": 0.2285,
        "precision_mean": 0.3345,
        "recall_mean": 0.4315,
        "fp_rows_per_100_mean": 12.7022,
        "notes": "Strong fixed local model does not transfer uniformly to all-row LOAO.",
    },
    {
        "system": "Qwen3-4B indexed zero-shot",
        "scope": "all-row LOAO, one held-out aspect per fold",
        "split": "test",
        "pair_samples_f1_mean": 0.1212,
        "pair_micro_f1_mean": 0.3378,
        "pair_macro_f1_mean": 0.2412,
        "precision_mean": 0.2379,
        "recall_mean": 0.8182,
        "fp_rows_per_100_mean": 34.4150,
        "notes": "High recall but weak empty-gold absence calibration.",
    },
]


QWEN_POSITIVE_DIAGNOSTIC = [
    {
        "system": "Qwen3-4B indexed zero-shot",
        "scope": "positive-gold rows from the all-row LOAO predictions",
        "split": "validation",
        "pair_samples_f1_mean": 0.8129,
        "pair_micro_f1_mean": 0.8692,
        "precision_mean": 0.9481,
        "recall_mean": 0.8115,
        "pair_macro_f1_mean": 0.6024,
        "sentiment_accuracy_when_gold_aspect_predicted": 0.9451,
    },
    {
        "system": "Qwen3-4B indexed zero-shot",
        "scope": "positive-gold rows from the all-row LOAO predictions",
        "split": "test",
        "pair_samples_f1_mean": 0.8194,
        "pair_micro_f1_mean": 0.8659,
        "precision_mean": 0.9338,
        "recall_mean": 0.8182,
        "pair_macro_f1_mean": 0.6184,
        "sentiment_accuracy_when_gold_aspect_predicted": 0.9314,
    },
]


CASCADE_TRADEOFF = [
    {
        "system": "Local DistilBERT only",
        "escalator": "none",
        "protocol": "fixed held-out aspect",
        "split": "test",
        "pair_samples_f1": 0.6071,
        "pair_micro_f1": 0.5917,
        "pair_macro_f1": 0.4890,
        "aspect_samples_f1": 0.6651,
        "call_rate": 0.0,
        "test_cost_usd": 0.0,
        "mean_called_latency_seconds": "",
    },
    {
        "system": "Gemini 2.5 Flash-Lite full",
        "escalator": "Flash-Lite",
        "protocol": "fixed held-out aspect",
        "split": "test",
        "pair_samples_f1": 0.5516,
        "pair_micro_f1": 0.5872,
        "pair_macro_f1": 0.4876,
        "aspect_samples_f1": 0.6062,
        "call_rate": 1.0,
        "test_cost_usd": 0.0096,
        "mean_called_latency_seconds": 0.372,
    },
    {
        "system": "Gemini 2.5 Flash full",
        "escalator": "Flash",
        "protocol": "fixed held-out aspect",
        "split": "test",
        "pair_samples_f1": 0.6071,
        "pair_micro_f1": 0.6541,
        "pair_macro_f1": 0.5547,
        "aspect_samples_f1": 0.6747,
        "call_rate": 1.0,
        "test_cost_usd": 0.2996,
        "mean_called_latency_seconds": 2.372,
    },
    {
        "system": "Gemini 2.5 Pro full",
        "escalator": "Pro",
        "protocol": "fixed held-out aspect",
        "split": "test",
        "pair_samples_f1": 0.7141,
        "pair_micro_f1": 0.7425,
        "pair_macro_f1": 0.6287,
        "aspect_samples_f1": 0.7746,
        "call_rate": 1.0,
        "test_cost_usd": 1.7140,
        "mean_called_latency_seconds": 5.651,
    },
    {
        "system": "Local -> Gemini Flash-Lite cascade",
        "escalator": "Flash-Lite",
        "protocol": "fixed held-out aspect cascade",
        "split": "test",
        "pair_samples_f1": 0.6679,
        "pair_micro_f1": 0.6579,
        "pair_macro_f1": 0.5401,
        "aspect_samples_f1": 0.7259,
        "call_rate": 0.5089,
        "test_cost_usd": 0.0052,
        "mean_called_latency_seconds": 0.348,
    },
    {
        "system": "Local -> Gemini Flash cascade",
        "escalator": "Flash",
        "protocol": "fixed held-out aspect cascade",
        "split": "test",
        "pair_samples_f1": 0.7459,
        "pair_micro_f1": 0.7348,
        "pair_macro_f1": 0.6223,
        "aspect_samples_f1": 0.7993,
        "call_rate": 0.9004,
        "test_cost_usd": 0.2789,
        "mean_called_latency_seconds": 2.437,
    },
    {
        "system": "Local -> Gemini Pro cascade",
        "escalator": "Pro",
        "protocol": "fixed held-out aspect cascade",
        "split": "test",
        "pair_samples_f1": 0.8102,
        "pair_micro_f1": 0.7955,
        "pair_macro_f1": 0.6809,
        "aspect_samples_f1": 0.8493,
        "call_rate": 0.9004,
        "test_cost_usd": 1.5421,
        "mean_called_latency_seconds": 5.648,
    },
]


FIXED_VS_LOAO_DROP = [
    {
        "system": "Candidate-aspect DistilBERT + DistilBERT sentiment",
        "fixed_protocol": "fixed three held-out aspects",
        "loao_protocol": "full all-row LOAO",
        "fixed_pair_micro_f1": 0.5917,
        "loao_pair_micro_f1_mean": 0.3128,
        "pair_micro_f1_drop": 0.2789,
        "fixed_pair_samples_f1": 0.6071,
        "loao_pair_samples_f1_mean": 0.0550,
        "notes": "Fixed split overstates robustness across aspect rotations.",
    },
    {
        "system": "Qwen3-4B indexed zero-shot",
        "fixed_protocol": "fixed three held-out aspects",
        "loao_protocol": "full all-row LOAO",
        "fixed_pair_micro_f1": 0.5300,
        "loao_pair_micro_f1_mean": 0.3378,
        "pair_micro_f1_drop": 0.1922,
        "fixed_pair_samples_f1": 0.5374,
        "loao_pair_samples_f1_mean": 0.1212,
        "notes": "LOAO exposes absence-calibration weakness hidden by the easier fixed split.",
    },
]


EVIDENCE_MAP = [
    {
        "claim": "Closed-topic supervised classifiers are strong when the full taxonomy is observed.",
        "protocol": "closed-topic provided FABSA split",
        "evidence_source": "docs/closed_topic_baselines.md; docs/generalisation_baselines.md",
        "table": "protocol_ladder",
    },
    {
        "claim": "Cross-organisation shift differs from taxonomy shift.",
        "protocol": "held-out organisation",
        "evidence_source": "docs/generalisation_baselines.md",
        "table": "protocol_ladder",
    },
    {
        "claim": "The strongest local fixed held-out-aspect result is candidate-aspect DistilBERT plus DistilBERT sentiment.",
        "protocol": "fixed three held-out aspects",
        "evidence_source": "docs/generalisation_baselines.md; docs/non_llm_open_topic_baseline.md",
        "table": "protocol_ladder",
    },
    {
        "claim": "Full all-row LOAO is substantially harder than the fixed held-out-aspect split.",
        "protocol": "12-fold all-row LOAO",
        "evidence_source": "docs/loao_heldout_aspect.md; docs/qwen_loao_experiment_analysis.md",
        "table": "loao_robustness and fixed_vs_loao_drop",
    },
    {
        "claim": "Qwen zero-shot recognises present aspects but over-predicts on empty-gold rows.",
        "protocol": "Qwen all-row LOAO plus positive-gold diagnostic",
        "evidence_source": "docs/qwen_feasibility.md; docs/qwen_loao_experiment_analysis.md",
        "table": "qwen_positive_diagnostic",
    },
    {
        "claim": "Fixed-split Qwen LoRA gives modest adaptation evidence, but does not replace full LOAO.",
        "protocol": "fixed three held-out aspects",
        "evidence_source": "docs/qwen_feasibility.md; docs/experiment_log.md",
        "table": "protocol_ladder",
    },
    {
        "claim": "Gemini and local-to-Gemini cascades are fixed-split deployment evidence, not LOAO robustness evidence.",
        "protocol": "fixed held-out-aspect hosted and cascade evaluation",
        "evidence_source": "docs/local_gemini_cascade.md; docs/gemini_candidate_label_baseline.md",
        "table": "cascade_tradeoff",
    },
]


def fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = []
    headers = [column.replace("_", " ").title() for column in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_markdown() -> None:
    lines = [
        "# Thesis Result Tables And Figure Data",
        "",
        "Last updated: 2026-07-02",
        "",
        "This document freezes thesis-ready aggregate tables before the full fine-tuned Qwen LoRA LOAO run. It deliberately keeps fixed-split, all-row LOAO, positive-gold diagnostic, and cascade/deployment evidence separate. The tables contain only aggregate metrics and source pointers; no raw review text or row-level predictions are included.",
        "",
        "## Evidence Map",
        "",
        markdown_table(EVIDENCE_MAP, ["claim", "protocol", "evidence_source", "table"]),
        "",
        "## Protocol Ladder",
        "",
        markdown_table(
            PROTOCOL_LADDER,
            [
                "section",
                "system",
                "protocol",
                "split",
                "pair_samples_f1",
                "pair_micro_f1",
                "pair_macro_f1",
                "aspect_samples_f1",
                "notes",
            ],
        ),
        "",
        "## All-Row LOAO Robustness",
        "",
        markdown_table(
            LOAO_ROBUSTNESS,
            [
                "system",
                "scope",
                "split",
                "pair_samples_f1_mean",
                "pair_micro_f1_mean",
                "pair_macro_f1_mean",
                "precision_mean",
                "recall_mean",
                "fp_rows_per_100_mean",
                "notes",
            ],
        ),
        "",
        "## Positive-Gold Diagnostic",
        "",
        markdown_table(
            QWEN_POSITIVE_DIAGNOSTIC,
            [
                "system",
                "scope",
                "split",
                "pair_samples_f1_mean",
                "pair_micro_f1_mean",
                "precision_mean",
                "recall_mean",
                "pair_macro_f1_mean",
                "sentiment_accuracy_when_gold_aspect_predicted",
            ],
        ),
        "",
        "## Gemini And Cascade Deployment Evidence",
        "",
        markdown_table(
            CASCADE_TRADEOFF,
            [
                "system",
                "protocol",
                "split",
                "pair_samples_f1",
                "pair_micro_f1",
                "pair_macro_f1",
                "aspect_samples_f1",
                "call_rate",
                "test_cost_usd",
                "mean_called_latency_seconds",
            ],
        ),
        "",
        "## Fixed Split Versus LOAO Drop",
        "",
        markdown_table(
            FIXED_VS_LOAO_DROP,
            [
                "system",
                "fixed_protocol",
                "loao_protocol",
                "fixed_pair_micro_f1",
                "loao_pair_micro_f1_mean",
                "pair_micro_f1_drop",
                "fixed_pair_samples_f1",
                "loao_pair_samples_f1_mean",
                "notes",
            ],
        ),
        "",
        "## Figure-Ready CSV Files",
        "",
        "- `docs/thesis_figure_data/protocol_ladder.csv`",
        "- `docs/thesis_figure_data/loao_robustness.csv`",
        "- `docs/thesis_figure_data/qwen_positive_diagnostic.csv`",
        "- `docs/thesis_figure_data/cascade_tradeoff.csv`",
        "- `docs/thesis_figure_data/fixed_vs_loao_drop.csv`",
        "",
        "## Interpretation Boundaries",
        "",
        "- Closed-topic and held-out-organisation rows are fixed-taxonomy references, not open-topic robustness evidence.",
        "- Fixed held-out-aspect rows evaluate a controlled three-aspect split and should not be merged with the 12-fold all-row LOAO rows.",
        "- Positive-gold LOAO rows remove the candidate-absence decision and should be used only as a diagnostic for recognition and sentiment once the held-out aspect is present.",
        "- Gemini and cascade rows are fixed-split hosted/deployment evidence. They support a cost-latency-quality discussion but do not replace a full LOAO robustness run.",
    ]
    OUTPUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_csv(PROTOCOL_LADDER, FIGURE_DIR / "protocol_ladder.csv")
    write_csv(LOAO_ROBUSTNESS, FIGURE_DIR / "loao_robustness.csv")
    write_csv(QWEN_POSITIVE_DIAGNOSTIC, FIGURE_DIR / "qwen_positive_diagnostic.csv")
    write_csv(CASCADE_TRADEOFF, FIGURE_DIR / "cascade_tradeoff.csv")
    write_csv(FIXED_VS_LOAO_DROP, FIGURE_DIR / "fixed_vs_loao_drop.csv")
    write_markdown()
    print(f"Wrote {OUTPUT_DOC}")
    print(f"Wrote figure data to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
