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

## Batch progression

- [x] V0: target-bounded executor, complete tests and real QLoRA smoke.
- [x] V1: single-aspect scopes a01--a04.
- [ ] V2: single-aspect scopes a05--a08.
- [ ] V3: single-aspect scopes a09--a12.
- [ ] V4: cyclic pairs a01-a02 through a04-a05.
- [ ] V5: cyclic pairs a05-a06 through a08-a09.
- [ ] V6: cyclic pairs a09-a10 through a12-a01.
- [ ] V7: Company-brand and Staff-support group scopes.
- [ ] V8: seed-13 formal validation reuse and threshold transfers.
- [ ] V9: Level 1 seed-23 robustness.
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
