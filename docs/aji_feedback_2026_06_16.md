# Aji Feedback: 2026-06-16

This note records the confirmed direction after sharing the first closed-topic baselines and Qwen feasibility results.

## Generalisation Axes

Both generalisation axes are important:

1. **Cross-organisation / domain-shift generalisation**
   - Train on some organisations.
   - Evaluate on held-out organisations.
   - This measures performance on new companies.

2. **Open-topic / new-aspect generalisation**
   - Train without some aspects/topics.
   - Evaluate with held-out aspects/topics provided as candidate labels at inference.
   - This is the more novel and impactful axis if one must be prioritised.

The contrast between these two settings is itself a core project result.

## Splits

The provided FABSA split should be kept as the closed-topic benchmark for comparability.

New splits should be constructed for the generalisation settings:

- A strict held-out-organisation split.
- A held-out-aspect split.

These split decisions should be documented carefully because they will matter for the dissertation writeup.

## Metrics

The expected Qwen score around `0.74-0.75 F1` should be treated as a rough internal reference, not a precise target.

The headline metric should be **sample-level F1**, because this is the internal Chattermill metric and is easier to discuss legibly. Micro F1 and macro F1 should still be reported alongside it.

Macro F1 remains important for the MSc thesis because it exposes long-tail aspect behaviour hidden by micro F1.

## Candidate Labels

Candidate labels should always be provided at inference time for open-topic experiments.

The model should select from canonical labels rather than inventing topic names. Free-form topic generation would make evaluation fuzzy and difficult to defend.

## Topic Descriptions

There are no formal topic descriptions at present. Aji may be able to find tagger guidance used for human ground-truth scoring. If available, that guidance could support candidate-label prompting and open-topic evaluation.

## Qwen Compute

Local QLoRA feasibility is useful, but the local laptop GPU is not the environment for real Qwen experiments.

Aji will sync across MSc projects and Diego may provide guidance on GPU resources.

