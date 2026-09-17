# Three-GPU formal validation readiness

> **Execution revoked on 20 August 2026.** This document remains valid as
> infrastructure and recovery evidence, but its 26-scope/297-result launch
> manifest was superseded by the post-supervisor experimental redesign. Do not
> run these commands. Build and audit a new versioned manifest from
> `docs/dissertation_taxonomy_post_supervisor_plan_2026_08_20.md`.

Date: 18 August 2026
Status: **preparation and local implementation complete; the long formal
validation campaign has not started**.

## Decision boundary

The authorised cloud work is validation-only. Only `train.csv` and
`validation.csv` may exist in the cloud data directory. `test.csv`,
`--include-official-test`, test scoring and target/test-guided parameter changes
are forbidden. The selected decoder emits one or two sentiments per selected
aspect and never three.

The previous RTX 3090 was terminated only after the local backup, immutable
artifact-unit manifests and critical SHA-256 values were checked. No old Pod is
currently intended to remain running while this readiness record is prepared.

## Exact three-worker partition

The frozen plan is
`configs/experiments/taxonomy_two_stage_three_gpu_parallel_v1.json`.

| Worker | Formal work | Dependency boundary |
|---|---|---|
| `worker-qlora-a` | 13 QLoRA training scopes | all three learning rates, scope selection and selected scoring stay together |
| `worker-qlora-b` | the other 13 QLoRA training scopes | all three learning rates, scope selection and selected scoring stay together |
| `worker-distil-frozen` | all 26 DistilBERT scopes, then all 26 Frozen-Qwen few-shot scopes | Distil checkpoints and Frozen prompt-cache chunks remain isolated by method |

The two QLoRA assignments have equal registered prompt-contract weight
(`456` each). The exact trainable union contains 260 jobs: 156 candidates, 52
scope selections and 52 selected-score jobs. Frozen-Qwen adds 26 immutable
scope executions. The expected result union is exactly 99 validation
fold-conditions for each of three methods, or 297 unique result files.

No worker shares mutable state, a checkpoint directory, a score path or a sync
receipt with another worker. Parallelisation changes wall time only; it does
not average, substitute or approximately reuse results.

## Frozen-Qwen few-shot formal contract

For each exact outer training scope, demonstrations are selected by stable
SHA-256 order from that scope's train fold only:

- aspect presence: two positive and two negative demonstrations;
- conditional sentiment: one negative, one neutral and one positive
  demonstration;
- no retrieval and no result-guided demonstration choice;
- threshold and runner-up threshold selection use seen-aspect validation only;
- held-out validation labels are evaluation-only.

Every probability query is keyed by the scope contract, demonstration hash,
mode, review and candidate. New probabilities are written in independently
sealed, content-addressed cache chunks. A partial/orphaned chunk, changed
demonstration set or different scope contract aborts rather than borrowing an
old score.

## Runtime stop system

Each long child process is guarded every 30 seconds. The executor terminates
the current job and writes failure state for:

- two software-thermal slowdown samples;
- any hardware thermal, hardware slowdown or hardware power-brake sample;
- GPU temperature at or above 88 C;
- two consecutive NVIDIA telemetry failures;
- less than 20 GiB output-volume headroom;
- a missing or older-than-900-second local replication heartbeat;
- OOM, CUDA error, non-finite values, prediction collapse, third sentiment,
  incompatible resume, corrupt artifact or official-test evidence in the log.

The local receiver copies only manifest-complete units, verifies bytes and
SHA-256 before publication, never removes the remote source, never overwrites
different local data, and writes a heartbeat back to its worker root after
every successful poll.

## Pre-launch host gate

Each physical Secure-Cloud RTX 4090 must independently pass, on the final
pushed commit:

```text
python scripts/audit_taxonomy_two_stage_runpod_startup.py \
  --expected-commit <FINAL_COMMIT> \
  --worker-id <WORKER_ID> \
  --data-dir /workspace/fabsa_data \
  --hf-home /workspace/hf-cache \
  --gate-root /workspace/taxonomy_two_stage_4090_gate_v1 \
  --output-root /workspace/taxonomy_two_stage_formal_v1/<WORKER_ID> \
  --observed-volume-used-gib <RUNPOD_UI_USED_GIB> \
  --output /workspace/taxonomy_two_stage_formal_v1/<WORKER_ID>/startup_audit.json
```

Admission requires the exact commit, a clean tracked worktree, the registered
worker manifest, one RTX 4090, pinned offline model snapshots, matching
train/validation hashes, absent `test.csv`, no stale temporary artifacts,
passed real-model/soak evidence and the registered storage headroom.

Before the three-host release, one final-commit host must additionally run one
bounded DistilBERT candidate, sync it locally, pass checkpoint/hash inspection,
and rerun the identical bounded command to prove exact resume without
retraining. This gate is infrastructure evidence, not a reported model result.

## Local receivers (start before formal execution)

Run one receiver per worker, using that Pod's SSH host and port:

```text
python scripts/sync_runpod_taxonomy_artifacts.py \
  --ssh-host root@<HOST> --ssh-port <PORT> --identity-file <KEY> \
  --remote-root /workspace/taxonomy_two_stage_formal_v1/<WORKER_ID> \
  --local-root C:/Msc_DSML/Msc_Project/cloud_backups/taxonomy_two_stage_formal_v1/<WORKER_ID> \
  --interval-seconds 300
```

The first successful cycle must create
`_sync/receiver_heartbeat.json` remotely before the corresponding executor is
launched.

## Formal launch commands (intentionally not executed by this readiness step)

QLoRA workers each run:

```text
python scripts/run_taxonomy_two_stage_formal_campaign.py \
  --worker-id <worker-qlora-a-or-b> \
  --parallel-plan configs/experiments/taxonomy_two_stage_three_gpu_parallel_v1.json \
  --output-root /workspace/taxonomy_two_stage_formal_v1/<WORKER_ID> \
  --data-dir /workspace/fabsa_data --hf-home /workspace/hf-cache \
  --local-files-only \
  --sync-heartbeat /workspace/taxonomy_two_stage_formal_v1/<WORKER_ID>/_sync/receiver_heartbeat.json
```

The third worker first runs the same campaign command with
`--worker-id worker-distil-frozen`, then runs:

```text
python scripts/run_taxonomy_two_stage_frozen_few_shot_campaign.py \
  --output-root /workspace/taxonomy_two_stage_formal_v1/worker-distil-frozen \
  --data-dir /workspace/fabsa_data --hf-home /workspace/hf-cache \
  --local-files-only \
  --sync-heartbeat /workspace/taxonomy_two_stage_formal_v1/worker-distil-frozen/_sync/receiver_heartbeat.json
```

## Final fail-closed merge

After all three receivers finish, no scores are averaged. The local output is
accepted only if this command passes:

```text
python scripts/audit_taxonomy_two_stage_parallel_outputs.py \
  --campaign-root C:/Msc_DSML/Msc_Project/cloud_backups/taxonomy_two_stage_formal_v1 \
  --receipt C:/Msc_DSML/Msc_Project/cloud_backups/taxonomy_two_stage_formal_v1/parallel_merge_audit.json
```

It verifies the exact job states, candidate contracts, checkpoint contents,
selection hashes, score-shard contracts/CSV hashes/finite probabilities,
immutable sync coverage, validation-only boundaries and all 297 unique result
payloads. A missing, duplicate, conflicting or unsafe unit makes the merge
fail.

## Reuse boundary

Only a completely published unit with an identical scientific and path-bound
contract is reusable. A different prompt, demonstration set, scope,
representation condition, learning rate, model revision, source contract or
output-root binding creates new work. Interrupted cache chunks/checkpoints or
unmanifested score files are not reusable. Earlier one-stage and precloud
results remain controls; they are not substituted into this formal two-stage
campaign.
