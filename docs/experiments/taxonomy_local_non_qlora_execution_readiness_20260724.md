# Local Non-QLoRA Execution Readiness - 24 July 2026

> This report remains authoritative for local-versus-cloud placement and cache
> design, but its description-pending gate, four-condition Level 3 protocol
> hash, and 152,316-input Frozen-Qwen estimate were superseded later on
> 24 July 2026. The revised `RR` protocol and 209,448-input maximum are recorded
> in `taxonomy_level3_rich_guidance_preregistration_20260724.md`.

## 1. Outcome

The approved compute split is now implemented:

| Method | Training | Scoring | Selection/analysis |
| --- | --- | --- | --- |
| Strict train-only TF-IDF | local CPU | local CPU | local CPU |
| E5-base-v2 | frozen marker/local CPU | local GPU | local CPU |
| DistilBERT cross-encoder | local GPU | local GPU | local CPU |
| Frozen candidate-pair Qwen | frozen marker/local CPU | local GPU | local CPU |
| Candidate-pair QLoRA | cloud GPU | cloud GPU | cloud control |

This changes execution placement only. It does not change folds, candidate
grids, model inputs, training pairs, thresholds, metrics, descriptions,
comparison families, or the scientific protocol hash.

The placement and cache rules are frozen in
`configs/experiments/taxonomy_execution_placement_v1.json`. The generated
hybrid dependency plan still contains 2,611 jobs, now tagged by executor:

| Executor | Jobs | Meaning |
| --- | ---: | --- |
| local CPU | 1,733 | cheap training, selection, aggregation and analysis |
| local GPU | 354 | E5, DistilBERT and Frozen-Qwen model work |
| cloud GPU | 294 | QLoRA training and scoring |
| cloud control | 230 | QLoRA-side threshold/analysis orchestration |

Job count is not compute share: many CPU jobs are tiny bookkeeping stages,
while one QLoRA GPU job may take much longer.

The locally generated hybrid plan is:

```text
outputs/experimental/taxonomy_hybrid_execution_plan_v1_20260724.json
SHA-256 1817e4bf107e3bb3b00ff1993701cc05f40aa44ba07fdfb94ab6d801dd7f5565
```

It retains scientific protocol SHA-256
`1dff49a855048891a7b3ecdb6d69304ba0bc387d3eba87c4b945f0df34bdb174`,
exactly matching the pre-placement plan.

## 2. Frozen-Qwen raw-score cache

### Why this is scientifically valid

Frozen Qwen is identical across folds:

- the base model ID and immutable revision are fixed;
- no fold-specific adapter or task-specific weight update exists;
- the system prompt, chat template, Y/N verbalizers, truncation rule,
  quantisation and probability mapping are locked; and
- the model receives only the review and one rendered candidate claim.

Therefore, whenever two folds produce the exact same model input under the
same immutable scoring contract, the raw value

```text
P(Y) = softmax(logit_Y, logit_N)[Y]
```

is reusable. Fold-specific thresholds and final predictions are not reusable
and are deliberately excluded from the cache.

QLoRA is not cacheable across training scopes because each fold/candidate
configuration may load a different trained adapter. QLoRA remains entirely on
the cloud path.

### Cache key and contract

Each input key is SHA-256 over:

```text
exact review text
+ exact rendered candidate claim
+ candidate aspect
+ candidate sentiment
+ representation variant
```

`row_uid` is deliberately absent because the frozen model cannot distinguish
two rows with exactly the same text and claim. Validation and test can never
share a cache: the official split is part of the cache contract and directory
path.

The cache contract additionally pins:

- scientific protocol SHA-256;
- Frozen-Qwen method-specification and complete method-registry SHA-256;
- model ID and immutable revision;
- complete scientific-parameter SHA-256;
- prompt/encoding/next-token-scoring SHA-256; and
- official split (`validation` or `test`).

An incompatible contract opens a different content-addressed database or fails
closed if incompatible metadata is found at the requested path.

### What is and is not stored

The SQLite database contains only:

```text
input_sha256 TEXT PRIMARY KEY
raw_present_probability REAL
```

It does not store review text, candidate text, gold targets, thresholds,
binary predictions, metrics, or test conclusions. Existing fold-specific
score artifacts and run contracts remain the scientific source of truth. The
cache can be deleted and recomputed without changing the experiment.

### Leakage and test-use gates

- Strict validation requires an explicit `is_heldout` column and rejects the
  whole operation if any held-out candidate is present.
- Validation and test databases are physically and cryptographically
  isolated.
- The formal runner loads and validates the frozen threshold-transfer artifact
  before any test-cache lookup or test model inference.
- Cache hits do not bypass score-artifact contracts, shard validation, the
  test-use ledger, or final metric code.
- SQLite metadata and integrity are checked on opening; malformed
  probabilities and conflicting concurrent writes fail closed.

### Lazy model loading

The Frozen-Qwen runtime is now lazy. A cache lookup happens before the real
4B model is needed. If a resumed or later fold is completely cached, the model
is never loaded. On the first miss it is loaded once and reused for every shard
in that job.

## 3. Correctness evidence

### Deterministic tests

The focused test suite covers:

- exact-input deduplication within a request;
- reuse across different fold run contracts;
- validation/test split isolation;
- rejection of held-out validation candidates;
- rejection of mismatched parameters and cache metadata;
- absence of raw text, targets, thresholds and predictions from SQLite;
- cache-hit score ordering after row shuffling;
- zero model calls on a complete hit;
- lazy model loading; and
- unchanged existing shard/condition/resume behaviour.

The first focused run passed:

```text
23 passed
```

The complete repository suite passed:

```text
292 passed
```

`compileall` also completed for `src`, `scripts` and `tests`, and
`git diff --check` reported no whitespace error.

### Real frozen-Qwen synthetic benchmark

Command:

```text
python scripts/benchmark_frozen_qwen_raw_cache.py \
  --pair-count 12 \
  --output outputs/experimental/frozen_qwen_raw_cache_benchmark_v1_20260724.json
```

This used one hard-coded synthetic review and did not open any official FABSA
split.

| Check | Result |
| --- | ---: |
| official data read | false |
| direct versus cold-cache max absolute difference | 0.0 |
| direct versus cold-cache bitwise equality | true |
| direct versus warm-cache max absolute difference | 0.0 |
| direct versus warm-cache bitwise equality | true |
| warm-cache model-scored inputs | 0 |
| model load | 14.286 s |
| direct uncached inference, 12 pairs | 1.997 s |
| cold cache fill, 12 pairs | 1.274 s |
| warm SQLite lookup, 12 pairs | 0.00117 s |
| acceptance result | pass |

The direct/cold timing difference is normal model warm-up noise; score
equivalence and elimination of warm-hit inference are the acceptance criteria.
The local benchmark JSON SHA-256 is:

```text
4ee7a8d0dc03d65aa26ea0e88690a7659a5977b6093aadcfccdc2aee41e66283
```

## 4. Revised Frozen-Qwen workload

Without cross-fold caching, the complete seed-13 Frozen-Qwen core contains:

```text
validation 887,880
+ test 1,771,092
= 2,658,972 raw pair inferences
```

The largest exact split-isolated cache working set is:

```text
validation: 1,057 reviews x 12 aspects x 3 sentiments
          = 38,052

test:       1,587 reviews x 12 aspects x 3 sentiments x 2 representations
          = 114,264

total     = 152,316 unique raw inputs
```

This is a maximum 94.27% reduction in Frozen-Qwen model inference. It is a
safe upper bound on required unique inputs: duplicate review text can reduce
the actual number further, but the plan does not assume that benefit.

At the earlier conservative local rate of approximately seven pairs/second,
152,316 inputs take about 6.0 GPU hours, compared with about 105.5 hours
without the cache. Model-loading and SQLite overhead remain, but lazy loading
removes model loads for complete-hit jobs.

Current planning range for the local non-QLoRA path is approximately:

| Component | Planning time |
| --- | ---: |
| strict TF-IDF, including finite tuning and CPU orchestration | under 1-2 h |
| E5 scoring | about 0.2 h |
| DistilBERT finite tuning/training/scoring | about 5.7 h |
| Frozen Qwen after cache | about 6-7 h |
| local analysis and contingency | about 1-2 h |
| total local non-QLoRA path | about 13-17 h |

These are planning estimates, not observed full-suite runtimes. They are
replaced with measured throughput after each first complete local method
scope. QLoRA runtime is unchanged and remains a separate cloud budget.

## 5. Current authority and remaining gates

Completed in this stage:

- [x] freeze local-versus-cloud execution placement;
- [x] keep placement outside the scientific protocol hash;
- [x] tag all 2,611 jobs with an explicit executor;
- [x] implement a content-addressed, split-isolated Frozen-Qwen raw-score
  cache;
- [x] prevent held-out validation candidates from entering the cache;
- [x] require threshold validation before test-cache access;
- [x] add lazy Frozen-Qwen loading;
- [x] add deterministic cache/failure-path tests;
- [x] pass a real-model synthetic equivalence benchmark without official
  data; and
- [x] recalculate Frozen-Qwen/local planning workload.

Still deliberately blocked:

- [ ] approve and freeze the exact twelve minimal descriptions;
- [x] approve and freeze the statistical/primary-comparison contract
  (approved by the user on 24 July 2026);
- [ ] run official seen-only nested validation selection for TF-IDF and
  DistilBERT;
- [ ] run official local TF-IDF, E5, DistilBERT and Frozen-Qwen stages;
- [ ] run the first measured cloud QLoRA tuning scope;
- [ ] freeze every method configuration and threshold rule;
- [ ] perform the single authorised formal test pass; and
- [ ] produce Level 1-4 tables, confidence intervals and thesis claims.

The user's approval of the compute split authorises this engineering and
synthetic verification stage. The user separately approved the complete
statistical protocol on 24 July 2026. The exact description texts and their
leakage-safe validation/freeze rule remain unresolved. Consequently, no new
official validation or test result was inspected in this stage.

## 6. Execution order after the remaining description decision

```text
freeze the leakage-safe description protocol and exact descriptions
    -> run local TF-IDF nested selection
    -> run local DistilBERT nested selection
    -> generate fixed E5 and Frozen-Qwen selections
    -> complete local validation/runtime gates
    -> run one cloud QLoRA tuning/runtime benchmark
    -> revise cloud time/cost from measured throughput
    -> finish all admitted validation selection
    -> freeze every model, threshold and comparison
    -> run one dependency-controlled formal test pass
    -> analyse paired intervals and write thesis evidence
```

Local validation work may proceed before the long QLoRA sweep. Formal test
scoring must not be used as an interim leaderboard: all methods and scientific
rules must be frozen before the one authorised test pass.
