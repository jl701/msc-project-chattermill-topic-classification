# Two-stage taxonomy generalisation: pre-cloud v2 preregistration

Registered: 17 August 2026, before any new v2 experiment execution.

## Decision being tested

The main pipeline is now a genuine two-stage decision:

1. score the presence of every supplied candidate aspect and retain every aspect above a seen-only threshold;
2. for each retained aspect, score negative, neutral and positive sentiment.

The primary sentiment decoder always emits the highest-scoring sentiment and may add the runner-up when it reaches a second threshold selected on seen-aspect validation evidence. It can emit one or two sentiments, never three. The aspect decoder has no top-k cap. The earlier 36 independent aspect-sentiment decisions remain a matched architecture control, not a substitute for this model.

## What is frozen before execution

- Official `test` remains sealed. Only `train` and `validation` may be present in local or cloud inputs for this phase.
- Held-out validation labels are evaluation-only. Both thresholds and every trainable hyperparameter are selected from seen-aspect validation evidence inside the exact outer training scope.
- L1 and L2 retain the registered single-aspect folds. L3 retains the cyclic dual holdouts and the NN, DN, ND, DD and RR conditions. L4 retains the three registered parent-group holdouts.
- In L3, one model/checkpoint, one deterministic demonstration set, and one threshold pair are shared by all five conditions. Only the representations of the two held-out aspects change. Seen prompts and scores must be hash-identical across conditions.
- Rich guidance is an aspect card only. It cannot contain or imply the candidate sentiment answer.
- Unknown-aspect anomaly detection without a supplied name or description is a separate deferred Study U1. It is not renamed as L5 and cannot silently enter this mainline.

## Frozen Qwen comparison

The no-adapter Qwen comparison contains two distinct arms.

- Zero-shot preserves the existing locked next-token prompts.
- Few-shot adds fixed cross-aspect demonstrations selected from the exact outer train fold by stable SHA-256 order: two positive and two negative examples for aspect presence, and one eligible single-sentiment example for each sentiment class. No held-out aspect, retrieval, validation outcome, or target result can choose a demonstration.

Few-shot uses a separate prompt schema, 1,024-token segment-aware prompt budget, prompt hashes and cache. A zero-shot cache is therefore never reused as a few-shot cache.

## Trainable models

DistilBERT and QLoRA require new genuine two-stage training. Historical independent-pair checkpoints and adapters remain valid controls only. QLoRA uses one shared adapter per outer scope with explicit `aspect_presence` and `sentiment` tasks, a fixed 4,096-example budget split equally between stages, and the pre-existing learning-rate values only as a new search range. No previous winning learning rate is transferred.

## Strict reuse rule

Reuse is rejected unless data hashes, row/pair identities, fold/filter, model/tokenizer revision, checkpoint hashes, representation resources, task stage, rendered prompt hashes, truncation, verbalizers, quantisation, score semantics, selection population/objective/ties and the test-access ledger all match.

The completed matched L2 TF-IDF and E5 two-stage artifacts are eligible for exact reuse. The completed L2 Frozen-Qwen zero-shot cache is eligible only after its file hash and prompt contract are rechecked. Old one-stage results are control-only; old DistilBERT checkpoints and QLoRA adapters cannot become two-stage models.

## Execution boundary

Local work must finish implementation, tests, real-model smoke tests, exact-reuse auditing, cheap TF-IDF/E5 validation, safe zero-shot cache use and a test-free cloud handoff. Expensive Frozen-Qwen few-shot, full DistilBERT, and QLoRA validation runs start online only after the local audit passes.

The machine-readable authority is `configs/experiments/taxonomy_two_stage_precloud_v2.json`.
