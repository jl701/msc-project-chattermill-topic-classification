# Taxonomy Validation Gate Before Supervisor Confirmation

Date: 2026-08-10

Status: V8 audited complete; V9 authorised for local validation-only execution.

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
- QLoRA V1--V8 have completed 326 of 398 through-validation jobs with zero
  failed jobs.
- QLoRA V8 revalidated 26 selected seed-13 checkpoint contracts, completed the
  missing `l4-g03` seen-calibration score, and transferred all 39 registered
  thresholds. Its checkpoint, score-shard and threshold hashes, finite-score
  checks, resume state, non-collapse checks and zero-test-contract condition
  passed the 2026-08-12 close-out audit.
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

## V7 completion and V8 registered launch

V7 completed at the registered boundary with 260 cumulative completed jobs,
zero failures and no official-test contract. The integrity audit verified six
checkpoint manifests (60 files), 48 score-shard manifests (171,234 finite
validation scores), two parameter selections, complete eight-shard contracts,
all declared hashes, an empty running ledger and no resume residue. Both group
scopes selected the pre-registered `1e-5` candidate from seen-only validation.

The V8 dry-run selects 326 cumulative jobs, skips all 260 completed jobs and
leaves exactly 66 pending validation-only jobs. It must stop at:

`formal-qwen_candidate_pair_qlora-seed0013-l4-g03-select-threshold`

The pending work consists of 26 selected seed-13 checkpoint reuse validations,
one required `l4-g03` seen-calibration validation scoring job and 39 threshold
transfers. All 26 selected checkpoint contracts already exist. V8 must use the
following target-bounded command without `--include-official-test`:

```powershell
python scripts/execute_taxonomy_plan.py `
  --plan outputs/experimental/taxonomy_hybrid_execution_plan_v2_20260724.json `
  --method qwen_candidate_pair_qlora `
  --stop-after-job-id formal-qwen_candidate_pair_qlora-seed0013-l4-g03-select-threshold `
  --max-workers 1
```

V8 may validate/reuse registered training artifacts and perform seen-only
calibration. It must not score any held-out `NN`, `DN`, `ND`, `DD` or `RR`
condition and does not resolve the pending Level 3 candidate-information
decision.

## V8 completion and V9 registered launch

V8 completed at the exact registered boundary with 326 cumulative jobs, zero
failures and zero test contracts. The audit verified 26 resumed checkpoint
contracts (260 files), one complete eight-shard validation score contract
(31,710 finite scores), and 39 hash-valid threshold-transfer artifacts. It
found no duplicate pairs, prediction collapse, resume residue, hard error,
thermal slowdown or official-test access. The formal `l4-g03` seen-validation
threshold is `0.746127575636`.

The V9 dry-run selects 362 cumulative jobs, skips all 326 completed jobs and
leaves exactly 36 Level 1 seed-23 validation-only jobs. It must stop at:

`formal-qwen_candidate_pair_qlora-seed0023-l1-a12-select-threshold`

The pending work comprises 12 registered seed-23 fits, 12 seen-calibration
validation scorings and 12 threshold transfers. V9 must use the following
target-bounded command without `--include-official-test`:

```powershell
python scripts/execute_taxonomy_plan.py `
  --plan outputs/experimental/taxonomy_hybrid_execution_plan_v2_20260724.json `
  --method qwen_candidate_pair_qlora `
  --stop-after-job-id formal-qwen_candidate_pair_qlora-seed0023-l1-a12-select-threshold `
  --max-workers 1
```

V9 changes only the pre-registered robustness seed. It uses frozen Level 1
training folds and seen-only validation calibration, does not evaluate any
held-out Level 3 candidate representation, and does not resolve or depend on
the pending supervisor decision.

## V9 completion and V10 registered launch

V9 completed at its exact registered boundary with 362 cumulative jobs, zero
failures and zero test contracts. Its 19.64-hour run produced 12 hash-valid
seed-23 checkpoint contracts, 12 complete eight-shard seen-calibration score
contracts containing 418,572 finite validation scores, and 12 hash-valid
threshold-transfer artifacts. The close-out audit found no missing shards,
non-finite or out-of-range scores, duplicate pair identities, predictive
collapse, resume residue, hard-error terms, monitoring alert or thermal
slowdown. Only `train` and `validation` were opened.

The V10 dry-run selects 398 cumulative validation-only jobs, skips all 362
completed jobs and leaves exactly 36 Level 1 seed-42 jobs. It must stop at:

`formal-qwen_candidate_pair_qlora-seed0042-l1-a12-select-threshold`

The pending work comprises 12 registered seed-42 fits, 12 seen-calibration
validation scorings and 12 threshold transfers. The registered command is:

```powershell
python scripts/execute_taxonomy_plan.py `
  --plan outputs/experimental/taxonomy_hybrid_execution_plan_v2_20260724.json `
  --method qwen_candidate_pair_qlora `
  --stop-after-job-id formal-qwen_candidate_pair_qlora-seed0042-l1-a12-select-threshold `
  --max-workers 1
```

`--include-official-test` remains absent. V10 changes only the robustness seed
and does not inspect or score a held-out Level 3 representation.
