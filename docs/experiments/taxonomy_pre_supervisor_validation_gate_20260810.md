# Taxonomy Validation Gate Before Supervisor Confirmation

Date: 2026-08-10

Status: V6 audited complete; V7 authorised for local validation-only execution.

## Purpose

This gate separates work that is scientifically reusable before the
2026-08-13 supervisor meeting from work that depends on the unresolved Level 3
candidate-information decision. It does not change the frozen protocol and it
does not authorise official-test access.

The unresolved question is whether a Level 3 held-out candidate without a
description should retain its canonical aspect name, receive only an opaque
supplied identity, or be absent altogether. The current registered `N`
condition means canonical aspect name plus candidate sentiment with no
description. Removing the supplied candidate entirely would be a different
open-set discovery task rather than the registered pair-scoring task.

## Already executed and reusable

- TF-IDF, E5, DistilBERT and Frozen Qwen have completed all registered
  validation-only work with zero failed jobs.
- QLoRA V1--V6 have completed 240 of 398 through-validation jobs with zero
  failed jobs.
- QLoRA V6 covers cyclic scopes a09-a10 through a12-a01. Its checkpoint and
  score-shard hashes, finite-score checks, parameter selections, resume state
  and zero-test-contract condition passed the 2026-08-10 close-out audit.
- No completed QLoRA selection batch rendered or scored a held-out target
  candidate. Selection used training data and seen-aspect validation only.

## Authorised before supervisor confirmation

The following batches may run in the frozen dependency plan, one at a time,
with `--include-official-test` absent:

1. V7: Level 4 `l4-g01` and `l4-g02` nested tuning on seen-aspect validation.
   Exact stop target:
   `select-tuned-qwen_candidate_pair_qlora-heldout-a08-a09-a10`.
2. V8: seed-13 content-addressed checkpoint reuse, the one required L4
   seen-calibration validation score, and threshold transfers. These jobs do
   not evaluate held-out Level 3 representations.
3. V9: Level 1 seed-23 robustness training and seen-only validation.
4. V10: Level 1 seed-42 robustness training and seen-only validation.
5. Integrity audits, tests, execution-record updates, and safe GitHub sync
   after each completed batch.

These jobs remain reusable if the meeting changes only the held-out Level 3
candidate rendering while retaining the supplied-candidate pair-scoring task,
because held-out candidates are excluded from training and calibration.

## Not authorised before supervisor confirmation

- Any Level 3 target-condition comparison under `NN`, `DN`, `ND`, `DD`, a new
  opaque-identity condition, or a review-only discovery formulation.
- Any experiment that changes candidate identity, candidate rendering,
  output schema, folds, training filtering, prompt, feature definition,
  metric, threshold rule, hyperparameter grid, or seed plan.
- Any replacement of the current protocol hash. A supervisor-requested change
  must receive a new recorded protocol/version or an explicitly audited
  additive extension; completed artifacts must not be silently relabelled.

## Prohibited until the final freeze gate

- Do not pass `--include-official-test`.
- Do not open, upload, inspect or score `test.csv`.
- Do not tune from target/test labels, predictions or metrics.
- Do not select a favourable condition or seed after observing target/test
  performance.
- Do not proceed after an OOM, non-finite score, corrupted resume state,
  missing shard, repeated thermal throttling or abnormal prediction collapse.

## V7 registered launch

The dry-run selected 260 cumulative jobs, found 240 already complete, and left
exactly 20 pending V7 jobs. It reported `include_official_test=false`.

```powershell
python scripts/execute_taxonomy_plan.py `
  --plan outputs/experimental/taxonomy_hybrid_execution_plan_v2_20260724.json `
  --method qwen_candidate_pair_qlora `
  --stop-after-job-id select-tuned-qwen_candidate_pair_qlora-heldout-a08-a09-a10 `
  --max-workers 1
```

The batch is resumable and must stop at the exact target above. On the local
RTX 5050 Laptop GPU, the telemetry-based expectation is approximately 9--10
hours for six fits, six validation scorings, six threshold selections and two
parameter selections.
