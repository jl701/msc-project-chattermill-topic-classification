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
- [ ] Complete and audit the first measured a01 tuning scope.
- [ ] Complete V1 through the a04 parameter-selection target.

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
