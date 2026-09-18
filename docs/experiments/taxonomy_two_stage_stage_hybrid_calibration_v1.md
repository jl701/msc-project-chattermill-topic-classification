# Two-Stage Stage-Hybrid and QLoRA Calibration Study v1

Date frozen before inspecting either new study outcome: 2026-08-23

## Research questions

This validation-only study answers two deliberately narrow questions suggested by the completed component audit.

1. Does the stronger observed aspect detector and the stronger observed conditional sentiment decoder form a better fixed two-stage system when their already-frozen outputs are recombined?
2. Does one low-capacity, global calibration rule convert QLoRA's stronger relative aspect ranking into better thresholded decisions without learning aspect-specific parameters?

The study does not add another model family, retrain QLoRA, tune prompts, or open the official test partition.

## Evidence and safety boundary

- Inputs are the verified formal-v2 L2 score shards for Frozen Qwen few-shot and QLoRA only.
- Every input must cover the same 1,057 validation `row_uid` values, twelve candidates and three sentiments, with `split=validation` and `test_contract_count=0`.
- Every result, selection, score shard and manifest used by this study must match its locally received SHA-256 receipt.
- The existing QLoRA L2-D a07/a12 seen-candidate canonicalisation is reproduced exactly; held-out D scores remain bitwise unchanged.
- The official test partition, labels, predictions and artifacts remain prohibited.
- N is an unchanged transfer sensitivity: all model fits and thresholds are selected from D only.

## Direction 1: fixed stage-wise hybrid

The primary hybrid takes Stage 1 aspect presence from Frozen Qwen few-shot and Stage 2 sentiment probabilities from QLoRA. Each source retains its original fold-local selected threshold. Threshold alignment maps those two frozen boundaries to 0.5 without changing either source decision. The existing top-one-plus-thresholded-runner-up decoder then emits at most two sentiments.

The reverse combination, QLoRA Stage 1 plus Frozen Qwen few-shot Stage 2, is retained as one necessary control. Neither direction fits or selects a new parameter. Because both formal systems retain a complete `1057 x 12 x 3` probability grid, the comparison is exact and requires no training or new model inference.

## Direction 2: low-capacity QLoRA relative calibrator

The calibrator changes Stage 1 only. For each review-candidate aspect instance it receives exactly two features:

1. the clipped logit of the absolute QLoRA aspect probability;
2. that logit minus the median logit of the other eleven aspect probabilities in the same review.

One L2-regularised, class-balanced logistic regression with standardised features is used. It has no aspect identity, description, sentiment or outcome-derived routing feature. QLoRA Stage 2 scores and its selected runner-up threshold remain unchanged.

### Leakage-safe primary estimate

For each outer held-out-aspect fold, reviews are deterministically assigned to five blocks by SHA-256 of seed 13 and `row_uid`. For every evaluation block:

- fit the scaler and logistic regression on D rows from the other four review blocks;
- permit labels from the eleven seen aspects only;
- prohibit every held-out-aspect label;
- prohibit labels from every review in the evaluation block;
- choose one aspect threshold on those same training rows using the fixed 0.00--1.00 grid in steps of 0.01;
- rank thresholds by seen pair micro-F1, then seen presence F1, then lower aspect call rate, then higher threshold;
- evaluate the held review block for all candidates, reporting held-out and overall results.

The five held blocks are concatenated to form a fully out-of-review, held-out-aspect-safe prediction for all 1,057 reviews. This cross-fitted output is the primary calibrator estimate.

### Final validation fit

A second diagnostic fit uses all D validation reviews but still only the eleven seen aspects. It represents the rule that could be frozen for a later disjoint partition. Its validation score is explicitly descriptive, not the primary cross-fitted estimate. Every D-trained scaler, logistic model and threshold is applied unchanged to N.

## Metrics, uncertainty and stopping rule

The primary metric is the unweighted mean of twelve held-out-aspect pair micro-F1 values. Secondary outputs include overall pair F1, pooled sensitivity metrics, precision, recall, presence F1/AP, oracle-gated sentiment-set F1 and aspect call rate. The study uses 20,000 synchronised review-cluster bootstrap draws with seed 13 and reports paired differences against the existing formal systems.

The pre-registered designs above are the complete search space. A negative result is retained. No new feature, threshold grid, model class, per-aspect rule or condition-specific refit will be introduced after outcome inspection.

## Reproducibility outputs

The local evidence package will contain verified input hashes, fold/block coefficients and thresholds, row-level confusion sufficient statistics, aggregate tables, paired bootstrap intervals and an audit manifest. Only aggregate, non-sensitive tables and documentation are committed; raw scores, reviews, checkpoints and credentials remain local.

## Completion pointer

The registered study was completed without expanding either method family. The fixed Few-shot-Stage-1 plus QLoRA-Stage-2 hybrid passed the point and paired-bootstrap comparison; the low-capacity QLoRA calibrator produced only an inconclusive held-out gain. Full results and integrity evidence are recorded in `docs/experiments/taxonomy_two_stage_stage_hybrid_calibration_v1_results.md`.
