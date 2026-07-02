# Thesis Result Tables And Figure Data

Last updated: 2026-07-02

This document freezes thesis-ready aggregate tables before the full fine-tuned Qwen LoRA LOAO run. It deliberately keeps fixed-split, all-row LOAO, positive-gold diagnostic, and cascade/deployment evidence separate. The tables contain only aggregate metrics and source pointers; no raw review text or row-level predictions are included.

## Evidence Map

| Claim | Protocol | Evidence Source | Table |
| --- | --- | --- | --- |
| Closed-topic supervised classifiers are strong when the full taxonomy is observed. | closed-topic provided FABSA split | docs/closed_topic_baselines.md; docs/generalisation_baselines.md | protocol_ladder |
| Cross-organisation shift differs from taxonomy shift. | held-out organisation | docs/generalisation_baselines.md | protocol_ladder |
| The strongest local fixed held-out-aspect result is candidate-aspect DistilBERT plus DistilBERT sentiment. | fixed three held-out aspects | docs/generalisation_baselines.md; docs/non_llm_open_topic_baseline.md | protocol_ladder |
| Full all-row LOAO is substantially harder than the fixed held-out-aspect split. | 12-fold all-row LOAO | docs/loao_heldout_aspect.md; docs/qwen_loao_experiment_analysis.md | loao_robustness and fixed_vs_loao_drop |
| Qwen zero-shot recognises present aspects but over-predicts on empty-gold rows. | Qwen all-row LOAO plus positive-gold diagnostic | docs/qwen_feasibility.md; docs/qwen_loao_experiment_analysis.md | qwen_positive_diagnostic |
| Fixed-split Qwen LoRA gives modest adaptation evidence, but does not replace full LOAO. | fixed three held-out aspects | docs/qwen_feasibility.md; docs/experiment_log.md | protocol_ladder |
| Gemini and local-to-Gemini cascades are fixed-split deployment evidence, not LOAO robustness evidence. | fixed held-out-aspect hosted and cascade evaluation | docs/local_gemini_cascade.md; docs/gemini_candidate_label_baseline.md | cascade_tradeoff |

## Protocol Ladder

| Section | System | Protocol | Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-topic | Word+char TF-IDF + Linear SVM | provided FABSA split | test | 0.7090 | 0.7042 | 0.4207 | 0.7736 | Fixed taxonomy reference point. |
| closed-topic | DistilBERT | provided FABSA split | test | 0.7803 | 0.7738 | 0.5377 | 0.8185 | Strongest closed-topic encoder baseline. |
| held-out organisation | Word+char TF-IDF + Linear SVM | held-out organisation | test | 0.7026 | 0.6920 | 0.3626 | 0.7629 | Domain-shift baseline; taxonomy is not held out. |
| held-out organisation | DistilBERT | held-out organisation | test | 0.7575 | 0.7600 | 0.4035 | 0.7986 | Validation-selected held-out-organisation encoder. |
| fixed held-out aspect | Candidate-label lexical TF-IDF + global sentiment | fixed three held-out aspects, example-filtered | test | 0.4626 | 0.4596 | 0.3703 | 0.5231 | Lexical lower bound for controlled unseen-aspect split. |
| fixed held-out aspect | Candidate-aspect DistilBERT + DistilBERT sentiment | fixed three held-out aspects, example-filtered | test | 0.6071 | 0.5917 | 0.4890 | 0.6651 | Strongest local fixed held-out-aspect baseline. |
| fixed held-out aspect | Qwen3-4B indexed zero-shot | fixed three held-out aspects | test | 0.5374 | 0.5300 | 0.4374 | 0.6340 | Open-weight zero-shot fixed split; not fine-tuned. |
| fixed held-out aspect | Qwen3-4B QLoRA indexed SFT | fixed three held-out aspects, example-filtered | test | 0.5528 | 0.5552 | 0.4393 | 0.6192 | One-epoch fixed-split QLoRA; modest adaptation gain over Qwen zero-shot, below the strongest local baseline. |

## All-Row LOAO Robustness

| System | Scope | Split | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Macro F1 Mean | Precision Mean | Recall Mean | Fp Rows Per 100 Mean | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lexical TF-IDF + global sentiment | all-row LOAO, micro-F1 threshold selection, example-filtered | test | 0.0903 | 0.3780 |  | 0.3778 | 0.4895 | 17.4753 | Best documented lexical all-row LOAO detection diagnostic. |
| Candidate-aspect DistilBERT + DistilBERT sentiment | all-row LOAO, pair-micro-F1 threshold selection, example-filtered | test | 0.0550 | 0.3128 | 0.2285 | 0.3345 | 0.4315 | 12.7022 | Strong fixed local model does not transfer uniformly to all-row LOAO. |
| Qwen3-4B indexed zero-shot | all-row LOAO, one held-out aspect per fold | test | 0.1212 | 0.3378 | 0.2412 | 0.2379 | 0.8182 | 34.4150 | High recall but weak empty-gold absence calibration. |

## Positive-Gold Diagnostic

| System | Scope | Split | Pair Samples F1 Mean | Pair Micro F1 Mean | Precision Mean | Recall Mean | Pair Macro F1 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B indexed zero-shot | positive-gold rows from the all-row LOAO predictions | validation | 0.8129 | 0.8692 | 0.9481 | 0.8115 | 0.6024 | 0.9451 |
| Qwen3-4B indexed zero-shot | positive-gold rows from the all-row LOAO predictions | test | 0.8194 | 0.8659 | 0.9338 | 0.8182 | 0.6184 | 0.9314 |

## Gemini And Cascade Deployment Evidence

| System | Protocol | Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Call Rate | Test Cost Usd | Mean Called Latency Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Local DistilBERT only | fixed held-out aspect | test | 0.6071 | 0.5917 | 0.4890 | 0.6651 | 0.0000 | 0.0000 |  |
| Gemini 2.5 Flash-Lite full | fixed held-out aspect | test | 0.5516 | 0.5872 | 0.4876 | 0.6062 | 1.0000 | 0.0096 | 0.3720 |
| Gemini 2.5 Flash full | fixed held-out aspect | test | 0.6071 | 0.6541 | 0.5547 | 0.6747 | 1.0000 | 0.2996 | 2.3720 |
| Gemini 2.5 Pro full | fixed held-out aspect | test | 0.7141 | 0.7425 | 0.6287 | 0.7746 | 1.0000 | 1.7140 | 5.6510 |
| Local -> Gemini Flash-Lite cascade | fixed held-out aspect cascade | test | 0.6679 | 0.6579 | 0.5401 | 0.7259 | 0.5089 | 0.0052 | 0.3480 |
| Local -> Gemini Flash cascade | fixed held-out aspect cascade | test | 0.7459 | 0.7348 | 0.6223 | 0.7993 | 0.9004 | 0.2789 | 2.4370 |
| Local -> Gemini Pro cascade | fixed held-out aspect cascade | test | 0.8102 | 0.7955 | 0.6809 | 0.8493 | 0.9004 | 1.5421 | 5.6480 |

## Fixed Split Versus LOAO Drop

| System | Fixed Protocol | Loao Protocol | Fixed Pair Micro F1 | Loao Pair Micro F1 Mean | Pair Micro F1 Drop | Fixed Pair Samples F1 | Loao Pair Samples F1 Mean | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate-aspect DistilBERT + DistilBERT sentiment | fixed three held-out aspects | full all-row LOAO | 0.5917 | 0.3128 | 0.2789 | 0.6071 | 0.0550 | Fixed split overstates robustness across aspect rotations. |
| Qwen3-4B indexed zero-shot | fixed three held-out aspects | full all-row LOAO | 0.5300 | 0.3378 | 0.1922 | 0.5374 | 0.1212 | LOAO exposes absence-calibration weakness hidden by the easier fixed split. |

## Figure-Ready CSV Files

- `docs/thesis_figure_data/protocol_ladder.csv`
- `docs/thesis_figure_data/loao_robustness.csv`
- `docs/thesis_figure_data/qwen_positive_diagnostic.csv`
- `docs/thesis_figure_data/cascade_tradeoff.csv`
- `docs/thesis_figure_data/fixed_vs_loao_drop.csv`

## Interpretation Boundaries

- Closed-topic and held-out-organisation rows are fixed-taxonomy references, not open-topic robustness evidence.
- Fixed held-out-aspect rows evaluate a controlled three-aspect split and should not be merged with the 12-fold all-row LOAO rows.
- Positive-gold LOAO rows remove the candidate-absence decision and should be used only as a diagnostic for recognition and sentiment once the held-out aspect is present.
- Gemini and cascade rows are fixed-split hosted/deployment evidence. They support a cost-latency-quality discussion but do not replace a full LOAO robustness run.
