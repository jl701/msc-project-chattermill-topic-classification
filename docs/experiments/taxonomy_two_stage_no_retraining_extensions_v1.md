# Validation-only no-retraining extensions

## Status

Completed on 23 August 2026 after preregistration and in the frozen order
below. Neither extension passed its promotion rule; the fixed hybrid remains
the validation-selected system. Full results are in
`docs/experiments/taxonomy_two_stage_no_retraining_extensions_v1_results.md`.

The protocol was preregistered on 23 August 2026 before either extension was evaluated. The
formal cloud evidence remains pinned to commit
`aa84212976a652d62cfca31ed8bf0516a216c485`. Only official `train` and
`validation` are permitted; official `test` remains sealed and the required
test-contract count is zero.

## Frozen execution order

1. Evaluate one smooth Stage-1 fusion of the already completed Frozen-Qwen
   few-shot and QLoRA systems. This phase is CPU-only and reuses the verified
   score shards exactly.
2. Evaluate one retrieval-based Frozen-Qwen few-shot Stage 1. This phase runs
   new validation inference, but it does not update model weights. Its Stage 2
   is the already frozen QLoRA sentiment component.

No additional fusion family, retrieval representation, prompt, demonstration
count, or model is added after outcomes are observed.

## Direction 1: smooth Stage-1 fusion

For each source, the raw aspect probability is converted to evidence relative
to that source's already frozen threshold:

`logit(probability) - logit(source threshold)`.

The two values are combined as
`alpha * few-shot + (1 - alpha) * QLoRA`, then mapped through a sigmoid. Alpha
is searched from 0 to 1 in steps of 0.1 and the fused threshold from 0 to 1 in
steps of 0.01. Selection is five-way review-cross-fitted inside each L2 outer
fold, uses condition D only, excludes the held-out aspect, and never trains on
the evaluation review block. The selected D policy is transferred unchanged to
condition N. QLoRA Stage 2 and the capped-two sentiment decoder do not change.

## Direction 2: retrieval-based few-shot Stage 1

The existing locked four-example aspect-presence prompt and Frozen-Qwen model
are retained. For every evaluation review-candidate pair, the retriever selects
two positive and two negative demonstrations from the exact outer training
fold crossed only with seen aspects. The retriever is the pinned frozen
`intfloat/e5-base-v2` encoder. A pair is represented by the L2-normalised sum
of its review embedding and candidate-card embedding. Demonstrations are ranked
within a fixed top-256 cosine shortlist per answer class, followed by
deterministic identity tie-breaking. Review rows must be distinct within each
answer class; different aspects are preferred.

### Pre-inference feasibility amendment

The first smoke stopped before any Qwen prediction because one positive
top-256 shortlist contained no feasible second item when both review and aspect
diversity were hard constraints. Before observing any model outcome, the rule
was amended once: distinct review rows remain mandatory; distinct aspects are
preferred, with a repeated aspect allowed only if the fixed shortlist cannot
satisfy aspect diversity. The shortlist, encoder, prompt, example counts and all
selection boundaries remain unchanged.

Each condition uses only the candidate card it permits: name plus description
for D and name only for the held-out candidate in N. Threshold selection is
five-way review-cross-fitted on D seen aspects only; the D-selected threshold is
then applied unchanged to N. QLoRA Stage 2 remains frozen.

## Decision rule

The primary measure is the aspect-balanced mean L2-D held-out pair micro-F1.
Both extensions are compared with the existing fixed Frozen-Qwen-few-shot
Stage-1 plus QLoRA-Stage-2 hybrid using a synchronised 20,000-draw review-level
bootstrap. An extension is promoted only when its point estimate is higher and
the paired 95% interval for the gain has a lower bound above zero. Otherwise it
is retained as a negative or inconclusive validation-only result.

The complete machine-readable contract is
`configs/experiments/taxonomy_two_stage_no_retraining_extensions_v1.json`.
