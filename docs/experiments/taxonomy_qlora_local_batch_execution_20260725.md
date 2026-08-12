# QLoRA Local Batch Execution Record

Date: 2026-07-25

Status: launched after user approval; official test remains sealed.

## Purpose

The complete registered QLoRA workload will run on the local RTX 5050 Laptop
GPU in resumable one-to-two-day batches. This changes only the administrative
executor placement. It does not change the frozen scientific protocol, model,
revision, candidate rendering, training recipe, threshold-transfer rule,
metric, seed plan, or test guard.

The authoritative scientific route remains
`docs/dissertation_loao_mainline_lock_2026_07_23.md`.

## Frozen execution identities

- plan:
  `outputs/experimental/taxonomy_hybrid_execution_plan_v2_20260724.json`;
- plan SHA-256:
  `e2a06d17ed35db8ce0afd3c67c22122dd04ba299019d11c34c6f1504f84e9685`;
- scientific protocol SHA-256:
  `d7ccded514ac1cbccf337e496e039ac418698566be0c0ca0c21e18608cc40f85`;
- method: `qwen_candidate_pair_qlora`;
- model: `Qwen/Qwen3-4B-Instruct-2507`;
- model revision:
  `cdbee75f17c01a7cc42f958dc650907174af0554`;
- seed for nested tuning: `13`;
- learning-rate candidates: `2e-6`, `5e-6`, `1e-5`;
- epochs: `1`;
- batch size / gradient accumulation: `1 / 8`;
- 4-bit loading: enabled;
- LoRA rank / alpha / dropout: `4 / 8 / 0.05`;
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- validation score shards: `8`;
- guarded QLoRA through-validation jobs: `398`; and
- official-test opt-in: absent.

All outputs, score shards, adapters, checkpoints, execution state and raw
telemetry remain ignored under `outputs/`.

## Local pre-flight evidence

- [x] Worktree was clean and based on GitHub main before the batch-control
  change.
- [x] Added a stable `--stop-after-job-id` boundary that remains exact across
  resumed invocations.
- [x] Verified that the V1 prefix contains exactly 40 validation-only jobs,
  ends at `select-tuned-qwen_candidate_pair_qlora-heldout-a04`, and excludes
  a05.
- [x] Passed the complete test suite: `313 passed`.
- [x] Passed a real cached-model QLoRA smoke covering 4-bit training,
  checkpoint reload, all `NN/DN/ND/DD/RR` renderings, sharding and resume.
- [x] Verified that the smoke reported `official_data_read: false`.
- [x] Complete and audit the first measured a01 tuning scope.
- [x] Complete V1 through the a04 parameter-selection target.

The smoke output root is:

```text
outputs/experimental/local_qlora_smoke_v1_20260725
```

Its measured fit and scoring times were approximately 36.08 and 16.75 seconds.
These tiny synthetic timings are compatibility evidence, not projections for
the official validation workload and not model-quality evidence.

## V1 boundary and resume command

V1 comprises the twelve tuning candidates for held-out aspects a01--a04:
three registered learning rates per aspect, with seen-only validation scoring,
threshold transfer and one parameter-selection artifact per aspect.

```powershell
python scripts/execute_taxonomy_plan.py `
  --plan outputs/experimental/taxonomy_hybrid_execution_plan_v2_20260724.json `
  --method qwen_candidate_pair_qlora `
  --stop-after-job-id select-tuned-qwen_candidate_pair_qlora-heldout-a04 `
  --max-workers 1
```

The same command is safe after interruption. Completed jobs are read from the
atomic execution state and skipped; no a05 job can enter the selected prefix.

The first a01 scope is initially run through the registered telemetry gate so
that actual training time, validation throughput, peak sampled GPU memory and
artifact size are measured before the remaining V1 jobs continue.

## First measured a01 gate result

The a01 gate completed at `2026-07-25T22:55:43.706694+00:00`. This is
seen-validation model selection evidence, not an official-test result.

| Item | Observed value |
| --- | ---: |
| Completed / expected jobs | 10 / 10 |
| Failed jobs | 0 |
| Total measured wall time | 18,004.63 s (5.00 h) |
| Three training jobs | 8,154.08 s |
| Three validation-scoring jobs | 9,796.48 s |
| Training pairs | 12,288 |
| Validation score pairs | 104,643 |
| Scoring throughput | 10.6817 pairs/s |
| Peak sampled GPU memory | 7,534 MiB |
| Three checkpoint trees | 83,172,226 bytes |
| Score CSVs | 28,480,311 bytes |

The frozen selection rule chose candidate `qwen_candidate_pair_qlora-003`,
whose registered learning rate is `1e-5`:

- seen-validation pair micro-F1: `0.5546095840`;
- pair samples F1: `0.4688080972`;
- pair micro precision / recall: `0.4884318766 / 0.6415306697`;
- presence F1: `0.9083245522`;
- transferred threshold: `0.810467272997`; and
- presence false-positive rows per 100: `0.5676442763`.

The registered candidates ranked monotonically by seen-validation pair
micro-F1 in this scope: `2e-6` gave `0.4912812737`, `5e-6` gave
`0.5223470662`, and `1e-5` gave `0.5546095840`. The registered grid is not
expanded after observing this result.

The post-run integrity audit found:

- three checkpoint manifests and 24 validation score-shard manifests;
- zero checkpoint or score-CSV hash mismatches;
- exactly 104,643 persisted score rows;
- zero non-finite or out-of-range probabilities;
- one frozen scientific protocol hash across every score shard;
- only `validation`, `L2`, `l2-a01`, and `seen-calibration` contracts; and
- zero test contracts.

The runtime, memory, resume and validation-quality gate therefore passed. V1
may continue through the pre-registered a04 boundary without changing the
scientific protocol or opening test.

## V1 completion

V1 completed at `2026-07-26T13:34:20.025816+00:00`, exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a04`. The executor did not
enter a05.

The complete batch ran for approximately 19.65 wall-clock hours from the first
measured a01 job, including the a01 audit, documentation and transition before
a02. Its reusable artifacts are:

- 40 completed dependency-plan jobs and zero failures;
- 12 checkpoint manifests covering four scopes and three candidates per scope;
- 96 validation score-shard manifests;
- 418,572 persisted validation scores;
- 332,688,905 bytes across the 12 checkpoint trees; and
- 114,455,906 bytes across the score CSVs and manifests.

The frozen seen-validation selection rule chose the registered `1e-5`
candidate in all four scopes:

| Scope | Selected LR | Pair micro-F1 | Pair samples F1 | Precision | Presence FP rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| heldout-a01 | `1e-5` | 0.554610 | 0.468808 | 0.488432 | 0.567644 |
| heldout-a02 | `1e-5` | 0.559423 | 0.475442 | 0.500000 | 0.094607 |
| heldout-a03 | `1e-5` | 0.542201 | 0.395489 | 0.533555 | 5.581835 |
| heldout-a04 | `1e-5` | 0.560661 | 0.476238 | 0.506368 | 0.189215 |

These are tuning results on seen-aspect validation data. They are not held-out
aspect results and are not official-test evidence.

The V1 close-out audit recomputed every recorded file hash and inspected every
score:

- 120 checkpoint files, zero missing files or hash mismatches;
- 96 score CSVs, zero missing files or hash mismatches;
- 418,572 scores, zero non-finite or out-of-range values;
- one scientific protocol SHA-256 across the batch;
- exactly the `l2-a01` through `l2-a04` seen-calibration folds; and
- zero test contracts.

V1 therefore passed its batch gate. V2 may proceed through the immutable a08
target without changing the candidate grid or selection rule.

## V2 completion

V2 completed at `2026-07-27T09:14:56.957496+00:00`, exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a08`. The executor did not
enter a09. The measured interval from the start of the first a05 training job
to the final a08 parameter selection was approximately 19.43 wall-clock hours.
The twelve recorded model fits account for 29,963.21 seconds (8.32 hours);
the remaining interval comprises validation scoring, model reloads, threshold
selection and executor overhead.

The reusable V2 artifacts are:

- 40 newly completed dependency-plan jobs, bringing the cumulative state to
  80 completed jobs with zero failures;
- 12 checkpoint manifests covering a05--a08 and three registered candidates
  per scope;
- 96 validation score-shard manifests;
- 418,572 persisted validation scores;
- 332,688,902 bytes across the 12 checkpoint trees; and
- 114,288,796 bytes across the score CSVs and manifests.

The frozen seen-validation selection rule again chose the registered `1e-5`
candidate in all four scopes:

| Scope | Selected LR | Threshold | Pair micro-F1 | Pair samples F1 | Precision | Recall | Presence F1 | Presence FP rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout-a05 | `1e-5` | 0.721732 | 0.545054 | 0.472630 | 0.474829 | 0.639655 | 0.912752 | 1.040681 |
| heldout-a06 | `1e-5` | 0.780000 | 0.576791 | 0.450009 | 0.507709 | 0.667632 | 0.852958 | 10.217597 |
| heldout-a07 | `1e-5` | 0.780000 | 0.528597 | 0.424136 | 0.487507 | 0.577252 | 0.861050 | 4.824976 |
| heldout-a08 | `1e-5` | 0.770000 | 0.545370 | 0.473049 | 0.457534 | 0.674942 | 0.916109 | 2.649007 |

These values are seen-aspect validation selection evidence. They do not score
the held-out aspect and are not official-test results. Candidate 003 was
selected by the pre-registered rule; the candidate grid was not expanded.

The V2 close-out audit recomputed both the raw CSV hashes and the canonical
pair/score hashes for every score shard, as well as every recorded checkpoint
file hash. It found:

- 120 checkpoint files, zero missing files or hash mismatches;
- 96 score CSVs, zero missing files, CSV-hash mismatches, pair-identity
  mismatches, or score-hash mismatches;
- 418,572 scores, zero non-finite or out-of-range values;
- all 12 complete eight-shard score contracts;
- one scientific protocol SHA-256 across the batch;
- exactly the `l2-a05` through `l2-a08` seen-calibration validation folds;
- an empty running ledger, an empty failure ledger and an intact resumable
  state at the exact a08 boundary; and
- zero test contracts.

GPU monitoring throughout V2 showed no hardware or software thermal
slowdown. Validation scoring repeatedly used nearly all available memory but
completed without an out-of-memory failure or corrupted resume state. V2
therefore passed its batch gate. The dry-run audit for V3 selects exactly 120
cumulative validation-only jobs, skips the 80 completed jobs, begins at a09
and stops exactly at the a12 parameter selection.

## V3 completion

V3 completed at `2026-07-28T05:12:16.346423+00:00`, exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a12`. The executor did not
enter the cyclic-pair schedule. The measured interval from the start of the
first a09 training job to the final a12 parameter selection was approximately
19.64 wall-clock hours. The twelve recorded model fits account for 30,628.55
seconds (8.51 hours); the remaining interval comprises validation scoring,
model reloads, threshold selection and executor overhead.

The reusable V3 artifacts are:

- 40 newly completed dependency-plan jobs, bringing the cumulative state to
  120 completed jobs with zero failures;
- 12 checkpoint manifests covering a09--a12 and three registered candidates
  per scope;
- 96 validation score-shard manifests;
- 418,572 persisted validation scores;
- 332,688,902 bytes across the 12 checkpoint trees; and
- 114,833,013 bytes across the score CSVs and manifests.

The frozen seen-validation selection rule again chose the registered `1e-5`
candidate in all four scopes:

| Scope | Selected LR | Threshold | Pair micro-F1 | Pair samples F1 | Precision | Recall | Presence F1 | Presence FP rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout-a09 | `1e-5` | 0.710000 | 0.550091 | 0.477981 | 0.474283 | 0.654743 | 0.917563 | 0.189215 |
| heldout-a10 | `1e-5` | 0.810467 | 0.552906 | 0.468234 | 0.509149 | 0.604891 | 0.902388 | 0.000000 |
| heldout-a11 | `1e-5` | 0.715413 | 0.554415 | 0.490598 | 0.471661 | 0.672385 | 0.934343 | 0.283822 |
| heldout-a12 | `1e-5` | 0.746128 | 0.536775 | 0.467602 | 0.463805 | 0.636994 | 0.918033 | 0.473037 |

These values are seen-aspect validation selection evidence. They do not score
the held-out aspect and are not official-test results. Candidate 003 was
selected by the pre-registered rule in every scope; the candidate grid was not
expanded.

The V3 close-out audit used round-trip CSV parsing and recomputed every raw
CSV, canonical pair-identity, canonical score, run-contract and checkpoint
file hash. It found:

- 120 checkpoint files, zero missing files or hash mismatches;
- 96 score CSVs, zero missing files, CSV-hash mismatches, pair-identity
  mismatches, score-hash mismatches or contract-hash mismatches;
- 418,572 scores, zero non-finite or out-of-range values and no duplicate pair
  identities within a shard;
- all 12 complete eight-shard score contracts;
- one frozen scientific protocol SHA-256 across the batch;
- exactly the `l2-a09` through `l2-a12` seen-calibration validation folds;
- an empty running ledger, an empty failure ledger and an intact resumable
  state at the exact a12 boundary; and
- zero test contracts.

GPU monitoring throughout V3 showed no hardware or software thermal
slowdown. Training remained near 4.9 GiB device memory, while validation
scoring repeatedly used approximately 7.9 GiB and completed without an
out-of-memory failure, non-finite score, corrupted resume state or abnormal
prediction collapse. V3 therefore passed its batch gate. V4 may proceed only
through the immutable `heldout-a04-a05` cyclic-pair parameter-selection target,
with official test still sealed.

## V4 completion

V4 completed at `2026-07-29T05:23:11.438203+00:00`, exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a04-a05`. The executor did not
enter a05-a06. The measured interval from the start of the first a01-a02
training job to the final a04-a05 parameter selection was approximately 24.04
wall-clock hours. This interval includes a safe pause after an unrelated GPU
process caused abnormal scoring throughput. The twelve recorded model fits
account for 30,271.43 seconds (8.41 hours); the remaining interval comprises
the pause, validation scoring, model reloads, threshold selection and executor
overhead.

The reusable V4 artifacts are:

- 40 newly completed dependency-plan jobs, bringing the cumulative state to
  160 completed jobs with zero failures;
- 12 checkpoint manifests covering a01-a02 through a04-a05 and three
  registered candidates per scope;
- 96 validation score-shard manifests;
- 380,520 persisted validation scores;
- 332,688,949 bytes across the 12 checkpoint trees; and
- 105,739,547 bytes across the score CSVs and manifests.

The frozen seen-validation selection rule again chose the registered `1e-5`
candidate in all four scopes:

| Scope | Selected LR | Threshold | Pair micro-F1 | Pair samples F1 | Precision | Recall | Presence F1 | Presence FP rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout-a01-a02 | `1e-5` | 0.810467 | 0.568154 | 0.483160 | 0.490956 | 0.674157 | 0.911193 | 0.756859 |
| heldout-a02-a03 | `1e-5` | 0.857757 | 0.574080 | 0.422043 | 0.542913 | 0.609044 | 0.830995 | 6.527909 |
| heldout-a03-a04 | `1e-5` | 0.837609 | 0.566097 | 0.424365 | 0.518240 | 0.623693 | 0.848048 | 6.622517 |
| heldout-a04-a05 | `1e-5` | 0.800680 | 0.561675 | 0.479918 | 0.491068 | 0.655995 | 0.903394 | 1.135289 |

These values are seen-aspect validation selection evidence. They do not score
either held-out aspect and are not official-test results. Candidate 003 was
selected by the pre-registered rule in every scope; the candidate grid was not
expanded.

The V4 close-out audit used round-trip CSV parsing and recomputed every raw
CSV, canonical pair-identity, canonical score, run-contract and checkpoint
file hash. It found:

- 120 checkpoint files, zero missing files or hash mismatches;
- 96 score CSVs, zero missing files, CSV-hash mismatches, pair-identity
  mismatches, score-hash mismatches or contract-hash mismatches;
- 380,520 scores, zero non-finite or out-of-range values and no duplicate pair
  identities within a shard;
- all 12 complete eight-shard score contracts;
- one frozen scientific protocol SHA-256 across the batch;
- exactly the `l3-a01-a02` through `l3-a04-a05` seen-calibration validation
  folds;
- an empty running ledger, an empty failure ledger and an intact resumable
  state at the exact a04-a05 boundary; and
- zero test contracts.

The contention event produced no failure, out-of-memory condition or partial
score shard. The executor reconciled the stale running marker on restart and
resumed from the complete, hash-verified checkpoint. Subsequent artifact
growth and throughput were normal. GPU monitoring showed no hardware or
software thermal slowdown, and the completed batch contained no non-finite
score, corrupted resume state or abnormal prediction collapse.

V4 therefore passed its batch gate. The V5 dry-run selects exactly 200
cumulative validation-only jobs, skips the 160 completed jobs, begins at
a05-a06 and stops exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a08-a09`. V5 may proceed only
through that immutable boundary, with official test still sealed.

## V5 completion

V5 completed at `2026-07-30T07:13:40.630680+00:00`, exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a08-a09`. The executor did not
enter a09-a10. The measured interval from the start of the first a05-a06
training job to the final a08-a09 parameter selection was approximately 18.49
wall-clock hours. The twelve recorded model fits account for 29,738.57 seconds
(8.26 hours); the remaining interval comprises validation scoring, model
reloads, threshold selection and executor overhead.

The reusable V5 artifacts are:

- 40 newly completed dependency-plan jobs, bringing the cumulative state to
  200 completed jobs with zero failures;
- 12 checkpoint manifests covering a05-a06 through a08-a09 and three
  registered candidates per scope;
- 96 validation score-shard manifests;
- 380,520 persisted validation scores;
- 332,688,951 bytes across the 12 checkpoint trees; and
- 105,240,002 bytes across the score CSVs and manifests.

The frozen seen-validation selection rule again chose the registered `1e-5`
candidate in all four scopes:

| Scope | Selected LR | Threshold | Pair micro-F1 | Pair samples F1 | Precision | Recall | Presence F1 | Presence FP rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout-a05-a06 | `1e-5` | 0.861528 | 0.573691 | 0.439063 | 0.507617 | 0.659541 | 0.836735 | 11.731315 |
| heldout-a06-a07 | `1e-5` | 0.780000 | 0.574713 | 0.401687 | 0.490196 | 0.694444 | 0.805699 | 17.691580 |
| heldout-a07-a08 | `1e-5` | 0.740163 | 0.537228 | 0.460404 | 0.447306 | 0.672401 | 0.881029 | 8.514664 |
| heldout-a08-a09 | `1e-5` | 0.752001 | 0.545960 | 0.464155 | 0.482374 | 0.628854 | 0.896226 | 2.838221 |

These values are seen-aspect validation selection evidence. They do not score
either held-out aspect and are not official-test results. Candidate 003 was
selected by the pre-registered rule in every scope; the candidate grid was not
expanded. The observed validation variation, including the higher presence
false-positive rate in a06-a07, did not trigger any result-dependent protocol
change.

The V5 close-out audit used round-trip CSV parsing and recomputed every raw
CSV, canonical pair-identity, canonical score, run-contract and checkpoint
file hash. It found:

- 120 checkpoint files, zero missing files or hash mismatches;
- 96 score CSVs, zero missing files, CSV-hash mismatches, pair-identity
  mismatches, score-hash mismatches or contract-hash mismatches;
- 380,520 scores, zero non-finite or out-of-range values and no duplicate pair
  identities within a shard;
- all 12 complete eight-shard score contracts;
- one frozen scientific protocol SHA-256 across the batch;
- exactly the `l3-a05-a06` through `l3-a08-a09` seen-calibration validation
  folds;
- all 12 parameter-selection source summaries present with finite thresholds
  and metrics;
- an empty running ledger, an empty failure ledger and an intact resumable
  state at the exact a08-a09 boundary; and
- zero test contracts.

GPU monitoring throughout V5 showed no hardware or software thermal slowdown.
Training remained near 5.0 GiB device memory, while validation scoring used up
to approximately 7.9 GiB and completed without an out-of-memory failure,
non-finite score, corrupted resume state or abnormal prediction collapse.

V5 therefore passed its batch gate. The V6 dry-run selects exactly 240
cumulative validation-only jobs, skips the 200 completed jobs, begins at
a09-a10 and stops exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a12-a01`. V6 may proceed only
through that immutable boundary, with official test still sealed.

## V6 completion

V6 completed at `2026-07-31T07:32:19.053751+00:00`, exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a12-a01`. The executor did not
enter the Level 4 group schedule. The measured interval from the V6 launch at
`2026-07-30T13:01:53+00:00` to the final parameter selection was approximately
18.51 wall-clock hours. The twelve recorded model fits account for 29,980.11
seconds (8.33 hours); the remainder comprises validation scoring, model
reloads, threshold selection and executor overhead.

The reusable V6 artifacts are:

- 40 newly completed dependency-plan jobs, bringing the cumulative state to
  240 completed jobs with zero failures;
- 12 checkpoint manifests covering a09-a10 through a12-a01 and three
  registered candidates per scope;
- 96 validation score-shard manifests forming 12 complete eight-shard
  contracts;
- 380,520 persisted validation scores;
- 332,688,952 bytes across the 12 checkpoint trees; and
- 105,978,933 bytes across the score CSVs and manifests.

The frozen seen-validation rule selected registered candidate
`qwen_candidate_pair_qlora-003` (`learning_rate=1e-5`) in all four scopes:

| Scope | Threshold | Pair micro-F1 | Pair samples F1 | Precision | Recall | Presence F1 | Presence FP rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout-a09-a10 | 0.841815 | 0.548412 | 0.459149 | 0.511860 | 0.590586 | 0.890871 | 0.189215 |
| heldout-a10-a11 | 0.763472 | 0.565637 | 0.489378 | 0.497665 | 0.655115 | 0.923469 | 0.283822 |
| heldout-a11-a12 | 0.746128 | 0.558881 | 0.491137 | 0.475177 | 0.678380 | 0.924180 | 0.567644 |
| heldout-a12-a01 | 0.763472 | 0.544336 | 0.462372 | 0.471372 | 0.644027 | 0.912815 | 1.040681 |

These are seen-aspect validation-selection measurements. They do not score
either held-out aspect and are not official-test results. No result-dependent
candidate-grid expansion or protocol change occurred.

The 2026-08-10 close-out audit reconstructed all checkpoint and run contracts,
recomputed every checkpoint-file, raw CSV, canonical pair-identity and score
hash, and round-trip parsed every score. It found:

- 120 checkpoint files and 96 score CSVs with zero missing files or hash
  mismatches;
- zero contract-hash mismatches and all 12 score contracts complete;
- zero duplicate pair identities, non-finite values or out-of-range scores;
- all four parameter selections present with finite metrics and all twelve
  source summaries present;
- one frozen scientific protocol SHA-256 across the batch;
- an empty running ledger, an empty failure ledger and an intact resumable
  state at the exact a12-a01 boundary; and
- zero test contracts.

The V6 log contains no out-of-memory error, traceback, resume conflict or
non-finite-score report. V6 therefore passed its batch gate. The V7 dry-run
selects exactly 260 cumulative validation-only jobs, skips the 240 completed
jobs, starts at `l4-g01`, and stops at
`select-tuned-qwen_candidate_pair_qlora-heldout-a08-a09-a10` without enabling
official test.

## V7 completion

V7 completed at `2026-08-11T04:28:05.730758+00:00`, exactly at
`select-tuned-qwen_candidate_pair_qlora-heldout-a08-a09-a10`. The executor did
not enter the seed-13 formal-reuse schedule. The measured interval from launch
at `2026-08-10T19:39:34.931729+00:00` to the final parameter selection was
approximately 8.81 wall-clock hours. The six recorded model fits account for
15,181.25 seconds (4.22 hours); the remainder comprises validation scoring,
model reloads, threshold selection and executor overhead.

The reusable V7 artifacts are:

- 20 newly completed dependency-plan jobs, bringing the cumulative state to
  260 completed jobs with zero failures;
- six checkpoint manifests covering `l4-g01` and `l4-g02`, with three
  registered candidates per scope;
- 48 validation score-shard manifests forming six complete eight-shard
  contracts;
- 171,234 persisted validation scores;
- 166,344,499 bytes across the six checkpoint trees; and
- 47,125,681 bytes across the score CSVs and manifests.

The frozen seen-validation rule selected registered candidate
`qwen_candidate_pair_qlora-003` (`learning_rate=1e-5`) in both scopes:

| Scope | Threshold | Pair micro-F1 | Pair samples F1 | Precision | Recall | Presence F1 | Presence FP rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout-a02-a03-a04 | 0.872337 | 0.570160 | 0.413979 | 0.528481 | 0.618977 | 0.835273 | 6.622517 |
| heldout-a08-a09-a10 | 0.715413 | 0.552638 | 0.470073 | 0.474305 | 0.661964 | 0.905208 | 2.932829 |

These are seen-aspect validation-selection measurements. They do not score a
held-out group or constitute official-test results. The candidate grid was not
expanded and no result-dependent protocol change occurred.

The 2026-08-11 close-out audit reconstructed all six checkpoint contracts and
six run contracts, round-trip parsed every score CSV, and recomputed every
checkpoint-file, raw CSV, canonical pair-identity and canonical score hash. It
found:

- 60 checkpoint files and 48 score CSVs with zero missing files or hash
  mismatches;
- zero contract-hash mismatches and all six eight-shard score contracts
  complete;
- zero duplicate pair identities, non-finite values or out-of-range scores;
- zero collapsed score contracts, with 607--1,109 distinct scores per
  contract;
- both parameter selections present with finite metrics and all six source
  summaries present;
- one frozen scientific protocol SHA-256 across the batch;
- an empty running ledger, an empty failure ledger, no temporary resume
  residue and an intact resumable state at the exact V7 boundary; and
- zero test contracts.

The V7 log contains no out-of-memory error, traceback, resume conflict,
non-finite-score report or thermal-slowdown event. V7 therefore passed its
batch gate. The V8 dry-run selects exactly 326 cumulative validation-only
jobs, skips the 260 completed jobs and stops at
`formal-qwen_candidate_pair_qlora-seed0013-l4-g03-select-threshold` without
enabling official test. Its 66 pending jobs comprise 26 content-addressed
seed-13 checkpoint validations/reuses, one missing `l4-g03` seen-calibration
validation scoring job, and 39 threshold transfers. All 26 selected checkpoint
contracts are already present and hash-valid.

## V8 completion

V8 completed at `2026-08-11T16:58:52.714495+00:00`, exactly at
`formal-qwen_candidate_pair_qlora-seed0013-l4-g03-select-threshold`. The
executor did not enter the seed-23 robustness schedule. The measured interval
from launch at `2026-08-11T15:55:01.573610+00:00` to the final threshold
transfer was approximately 1.06 wall-clock hours.

The 66 V8 jobs reused and revalidated 26 selected seed-13 checkpoint
contracts, scored the one missing `l4-g03` seen-calibration contract in eight
shards, and wrote 39 formal threshold-transfer artifacts. Reuse avoided new
training while verifying 260 checkpoint files (720,826,024 bytes). The new
score contract contains 31,710 validation pairs across eight complete shards
(8,661,806 CSV bytes) and produced 790 distinct finite scores spanning
`0.0000021908` to `0.9996485710`. The formal Level 4 group-three threshold is
`0.746127575636`; all 39 thresholds remain seen-aspect validation selections,
not held-out or official-test measurements.

The 2026-08-12 close-out audit recomputed every checkpoint-file, score CSV,
pair-identity, score and threshold-artifact hash. It found zero failed or
running jobs, zero missing shards, zero non-finite or out-of-range scores,
zero duplicate pair identities, zero collapsed score or threshold conditions,
zero resume residue, zero hard-error terms, zero thermal-slowdown records and
zero test contracts. Only the `train` and `validation` splits were opened.
V8 therefore passed its batch gate.

The V9 dry-run selects 362 cumulative validation-only jobs, skips all 326
completed jobs and leaves exactly 36 pending jobs: 12 seed-23 Level 1 fits, 12
seen-calibration scorings and 12 threshold transfers. It stops at
`formal-qwen_candidate_pair_qlora-seed0023-l1-a12-select-threshold` without
enabling official test. Based on the earlier single-aspect batches, the local
runtime expectation is approximately 19--20 wall-clock hours.

## V9 completion and V10 registered launch

V9 completed at `2026-08-12T19:21:39.577110+00:00`, exactly at
`formal-qwen_candidate_pair_qlora-seed0023-l1-a12-select-threshold`. The
measured interval from launch at `2026-08-11T23:43:25.167572+00:00` was
70,694.41 seconds, or approximately 19.64 wall-clock hours. The executor did
not enter the seed-42 schedule and never enabled official test.

The close-out audit revalidated all 12 seed-23 checkpoint contracts and their
120 declared files (332,688,908 bytes). It also verified 12 complete
eight-shard seen-calibration score contracts: 96 score manifests and 418,572
finite validation scores, with every CSV, score, pair-identity and contract
hash matching. All 12 threshold-transfer artifacts passed their self-hash and
checkpoint/score-reference checks; their transferred thresholds span
`0.721732348204` to `0.865213662386`.

The audit found zero failures, missing shards, non-finite or out-of-range
scores, duplicate pair identities, collapsed score or threshold conditions,
resume residue, hard-error terms, monitoring alerts, thermal-slowdown records
and test contracts. Only the `train` and `validation` splits were opened. V9
therefore passed its batch gate.

The V10 dry-run selects 398 cumulative validation-only jobs, skips all 362
completed jobs and leaves exactly 36 pending jobs: 12 seed-42 Level 1 fits, 12
seen-calibration scorings and 12 threshold transfers. It must stop at:

`formal-qwen_candidate_pair_qlora-seed0042-l1-a12-select-threshold`

The target-bounded V10 command is:

```powershell
python scripts/execute_taxonomy_plan.py `
  --plan outputs/experimental/taxonomy_hybrid_execution_plan_v2_20260724.json `
  --method qwen_candidate_pair_qlora `
  --stop-after-job-id formal-qwen_candidate_pair_qlora-seed0042-l1-a12-select-threshold `
  --max-workers 1
```

The command deliberately omits `--include-official-test`. V10 changes only
the pre-registered robustness seed and remains independent of the unresolved
Level 3 held-out representation decision.

## Batch progression

- [x] V0: target-bounded executor, complete tests and real QLoRA smoke.
- [x] V1: single-aspect scopes a01--a04.
- [x] V2: single-aspect scopes a05--a08.
- [x] V3: single-aspect scopes a09--a12.
- [x] V4: cyclic pairs a01-a02 through a04-a05.
- [x] V5: cyclic pairs a05-a06 through a08-a09.
- [x] V6: cyclic pairs a09-a10 through a12-a01.
- [x] V7: Company-brand and Staff-support group scopes.
- [x] V8: seed-13 formal validation reuse and threshold transfers.
- [x] V9: Level 1 seed-23 robustness.
- [ ] V10: Level 1 seed-42 robustness.
- [ ] G: validation freeze, leakage audit and user review before test.

## Monitoring and stop rules

The run stops immediately on a non-zero job result. During execution, monitor:

- executor state, current job and failed-job ledger;
- GPU utilisation, memory, temperature and power;
- new adapter/checkpoint and score-shard growth;
- training and scoring wall time;
- threshold-selection completion; and
- parameter-selection summaries.

Pause rather than alter the protocol if there is an out-of-memory failure,
repeated thermal throttling, corrupted or incompatible resume state, missing
shards, non-finite scores, dependency deadlock, or abnormal predictive
collapse. Any scientific change requires a new validation-only decision and a
new recorded configuration.

After all validation batches, execution must stop for the registered freeze
and leakage audit. Official test work requires the separate
`--include-official-test` flag and is not authorised by this launch.
