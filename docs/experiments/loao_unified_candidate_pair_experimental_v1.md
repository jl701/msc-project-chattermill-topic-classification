# Unified Candidate-Pair LOAO Experiment v1

## Status

Pre-registered on 17 July 2026 before executing any new model or inspecting any new validation/test result.

This is an isolated exploratory experiment requested after the dissertation design lock. It does not modify the frozen main benchmark, headline result tables, report narrative, or thesis. Admission requires explicit user review and approval after the results are complete.

The machine-readable authority is `configs/experiments/loao_unified_candidate_pair_experimental_v1.json`. The frozen taxonomy descriptions are in `configs/experiments/fabsa_aspect_descriptions_v1.json`.

## Why the first four-state idea was rejected before execution

A read-only audit found 106 review–aspect instances with more than one annotated sentiment: 73 train, 12 validation, and 21 test. A single `absent/negative/neutral/positive` target would therefore discard valid pair labels. No model was run under that invalid assumption.

The unified task is instead:

```text
(review, candidate aspect definition, candidate sentiment) -> applicable / not applicable
```

Each held-out candidate is scored independently for negative, neutral, and positive. Any subset may be emitted; emitting none means absent. This preserves the authoritative pair-set representation.

## Fixed comparison

Every model uses the same outer folds, `example_filtered` training construction, all official validation/test rows, singleton held-out candidate, projected gold pair set, empty-prediction option, 4,096-pair training budget, seed, threshold selection, and metric implementation.

Two data variants isolate the proposed package:

- Matched control: canonical name plus sentiment; four seeded random non-gold pair negatives per positive.
- Enhanced: frozen definition/cues/boundary plus sentiment definition; both wrong sentiments for the true aspect, one same-parent-or-semantic hard aspect with the same sentiment, and one random absent aspect with the same sentiment.

Both are deterministically reduced to 2,048 positive and 2,048 negative training pairs by aspect-balanced round-robin sampling. This deliberately leaves some local-model data unused so that TF–IDF, DistilBERT, and Qwen receive the same pair budget on the 8 GB machine.

The comparison estimates the effect of the complete unified package; it does not separately identify description, hard-negative, and balancing effects. If the package succeeds, those ablations would require a later registration.

## Selection and leakage boundary

The experiment retains the frozen primary benchmark's target-calibrated cold-start information regime so that local-model deltas can be compared with the historical TF–IDF and DistilBERT rows. For each fold, the target validation labels may select a threshold and, for DistilBERT, an epoch. Test labels never select a candidate text, negative policy, training budget, model configuration, checkpoint, or threshold.

This is not a strict zero-label calibration result. A later strict-transfer experiment would need protocol-matched controls and cannot reuse the historical local headline values as unconditional baselines.

## Execution gates

The three validation-only pilot folds are:

1. `Company brand: Competitor` — difficult semantic boundary;
2. `Company brand: General satisfaction` — high-prevalence broad label;
3. `Staff support: Email` — rare specific label.

For a local model to proceed to the twelve-fold test confirmation, enhanced must beat its matched control by at least `+0.02` mean validation pair micro-F1, win at least two of three folds, and retain at least half of control recall in every fold.

Frozen candidate-wise Qwen is evaluated before QLoRA. QLoRA proceeds only if that control is not more than `0.01` below historical JSON zero-shot mean on the same pilot and reduces false-positive rows by at least 15%. QLoRA must then add at least `+0.02`, win at least two folds, and retain at least half of frozen-control recall. Failure stops that branch without test evaluation.

Full confirmation success requires at least `+0.02` mean test pair micro-F1 over the protocol-matched control and wins on at least 8 of 12 aspects. Historical headline deltas are context, not a substitute for the matched comparison.

## Isolation and reporting

All manifests, predictions, logits, adapters, checkpoints, and summaries go under ignored `outputs/experimental/loao_unified_candidate_pair_v1/`. The final experimental note will report negative results as well as positive ones, all commands, runtime, package/hardware state, limitations, and gate decisions.

Until user approval, do not edit:

- `docs/thesis_result_tables.md`;
- `docs/dissertation_experiment_design_lock_2026_07_16.md`;
- any file under `thesis/`.
