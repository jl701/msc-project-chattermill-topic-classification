# Formal two-stage cloud execution and recovery

Date: 17 August 2026

Status: **local implementation and recovery preflight complete; the long formal
campaign has not started**. The final RunPod startup gate and live local sync
receiver must both pass before release.

## What is now automatic

Every completed checkpoint, learning-rate candidate, parameter selection,
validation score shard and fold-condition result is published as an immutable
artifact unit. Publication happens only after the underlying file or checkpoint
directory is complete and every file has a SHA-256 hash.

`scripts/sync_runpod_taxonomy_artifacts.py` can poll the RunPod over OpenSSH and
SCP. It downloads a complete unit into a local staging directory, verifies the
declared byte count and SHA-256 for every file, and only then publishes the
files under the local output root. A receipt under `_sync/received` makes the
operation idempotent. It does not delete remote files and refuses to overwrite
a different local file. SSH credentials and private keys are never written to
the repository.

This means completed units are automatically copied while the receiver is
running. It does not mean RunPod pushes through a powered-off laptop: if the
computer, network or receiver stops, cloud files remain in place and the same
command catches up after reconnection.

## Exact interruption and reuse boundary

The following completed objects are independently reusable:

- a checkpoint whose `TrainingContract`, file set and all file hashes match;
- a validation score shard whose `RunContract`, pair identities, targets,
  score hashes and CSV hash match;
- a sealed learning-rate candidate or selected-parameter record whose payload
  hash matches; and
- a locally received unit whose receipt and destination hashes still match.

An interruption before the completion manifest leaves no reusable unit. The
temporary state is deliberately not guessed or repaired. An interruption after
the manifest is safe: resume validates the object and skips the completed work.

A scientific configuration change produces a new contract hash and a new
content-addressed path. Old artifacts remain available as historical controls,
but they are not treated as exact reuse. Administrative changes that do not
enter a scientific contract, such as restarting the sync receiver, do not
invalidate completed artifacts.

## Frozen formal job graph

The formal trainable graph covers DistilBERT and QLoRA:

| Item | Count |
|---|---:|
| Unique outer training scopes | 26 |
| Learning-rate candidates per method and scope | 3 |
| Candidate checkpoints across both methods | 156 |
| Parameter-selection jobs | 52 |
| Selected-checkpoint scoring jobs | 52 |
| Fold-condition outputs per selected method | 99 |
| Validation score shards per fold-condition | 8 |

The runner loads only official train and validation. Learning rate, aspect
threshold and runner-up sentiment threshold are selected using seen-aspect
validation evidence inside the exact outer scope. Held-out validation labels
are evaluation-only. The primary decoder emits one sentiment for each selected
aspect and adds the runner-up only when its separately selected threshold is
met; a third sentiment is forbidden.

## Local recovery drill

The formal preflight performed all of the following with synthetic but
schema-identical artifacts:

- atomic checkpoint publication and exact resume;
- atomic two-stage score-shard publication and exact resume;
- a simulated transfer interruption before publication;
- a successful retry without recomputation;
- a second idempotent sync pass;
- rejection of a changed remote byte; and
- rejection of partial, corrupt or contract-incompatible state.

The focused suite passed 22 tests and the repository-wide suite passed 398
tests. The registered train and validation hashes matched, the formal job graph
contained zero test contracts, and the existing Secure-Cloud RTX 4090 gate
still had zero failures and zero test contracts.

## Required startup sequence

Do not skip or reorder these gates.

1. Resume the qualified Secure-Cloud RTX 4090 Pod, or qualify a newly allocated
   host before accepting its results.
2. Check out the reviewed Git commit and upload only `train.csv` and
   `validation.csv` to `/workspace/data/fabsa`. Do not upload `test.csv`.
3. Run the final in-Pod audit:

   ```bash
   python scripts/audit_taxonomy_two_stage_runpod_startup.py \
     --expected-commit <reviewed-commit> \
     --data-dir /workspace/data/fabsa \
     --hf-home /workspace/hf-cache \
     --gate-root outputs/experimental/taxonomy_two_stage_4090_gate_v1 \
     --output-root /workspace/taxonomy_two_stage_formal_v1 \
     --observed-volume-used-gib <conservative-RunPod-UI-upper-bound> \
     --output /workspace/taxonomy_two_stage_formal_v1/startup_audit.json
   ```

   The storage gate deliberately does not trust `df` for a RunPod network
   volume because it can expose the shared backing filesystem rather than the
   purchased quota.  It combines the 100 GiB RunPod quota, the observed used
   space, a preregistered 30 GiB campaign-growth upper bound, and a required
   20 GiB final headroom.

   The startup audit also resolves the exact registered DistilBERT and Qwen
   revisions from the persistent Hugging Face cache with network access
   disabled. It loads both configs and tokenizers, checks every indexed model
   weight shard, opens each safetensors header and rejects missing or broken
   cache links. A cache that merely exists but is not complete cannot release
   the campaign.

4. On the local computer, start the verified receiver and leave it running:

   ```powershell
   python scripts/sync_runpod_taxonomy_artifacts.py `
     --ssh-host root@<runpod-host> `
     --ssh-port <runpod-port> `
     --identity-file <private-key-path> `
     --remote-root /workspace/taxonomy_two_stage_formal_v1 `
     --local-root C:\Msc_DSML\Msc_Project\cloud_backups\taxonomy_two_stage_formal_v1 `
     --interval-seconds 300
   ```

5. Re-run the local readiness audit with the downloaded startup audit and
   require Git synchronisation. `formal_release_ready` must be true.
6. Execute one bounded candidate job first, inspect its checkpoint, sync
   receipt, finite metrics and GPU telemetry, then release the remaining graph.
7. Use `scripts/run_taxonomy_two_stage_formal_campaign.py` for serial execution.
   It writes the current job, completed count, per-job logs and GPU snapshots,
   and stops immediately on a non-zero exit or fatal stop token.

   The orchestrator resolves the portable `python` token in the frozen job
   graph to its own `sys.executable`. This is required so every child job uses
   the same audited virtual environment as the orchestrator; relying on the
   ambient `PATH` is forbidden.

   Every non-dry execution must also provide
   `--hf-home /workspace/hf-cache --local-files-only`. The orchestrator passes
   `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE=1` and
   `TRANSFORMERS_OFFLINE=1` explicitly to every child. The resolved cache path
   is stored in campaign state and an exact resume rejects a different path.

## 2026-08-18: Failed interpreter-launch attempt and clean recovery boundary

Corrected source commit `e2fdefb214300edf633c409721c05fe5203064b8`
passed the replacement-host startup audit on a Secure RTX 4090, including the
quota-aware storage gate, exact train/validation hashes, CUDA 12.8, required
packages, absence of `test.csv`, zero failures and zero test contracts. The
first bounded DistilBERT job then stopped before importing the dataset or model:
the frozen job graph's portable `python` token resolved through the container's
ambient `PATH` to a base interpreter without pandas instead of the audited
virtual environment.

The fail-closed orchestrator recorded `failure_count=1`, produced no checkpoint
or model result, and did not start another job. That failed state and log remain
immutable under the original remote root and in the local failure-evidence
backup. They must not be rewritten to look successful. The recovery commit
binds child jobs to `sys.executable` and adds a regression test. Its live retry
uses a distinct `taxonomy_two_stage_formal_v1_r2` execution root. Only artifacts
from that clean root may satisfy the bounded-candidate and formal-release gate.

## Current go/no-go

The scientific graph, artifact contracts and replication layer remain locally
ready, but the formal campaign is **NO-GO** until the cache-binding recovery
commit passes the full local test suite and the clean `r3` root passes a fresh
startup audit, one bounded candidate, verified local receipt and exact-resume
check. No formal model result has yet passed these release gates.

## 2026-08-18: Failed persistent-cache binding attempt and r3 boundary

Interpreter-recovery commit `1c1326b635df833a638f3d3f10e330d3742b218d`
passed a new startup audit under the clean `taxonomy_two_stage_formal_v1_r2`
root. The bounded DistilBERT child then used the correct audited virtual
environment, confirming the interpreter repair, but stopped before loading any
dataset rows or starting GPU training. The new container had no `HF_HOME`
environment variable, so Transformers searched its empty ephemeral root cache
instead of the complete persistent `/workspace/hf-cache` mounted on the
network volume. Offline-only model resolution therefore raised
`LocalEntryNotFoundError`.

The fail-closed campaign recorded one failure, created no checkpoint or model
result and launched no later job. State, telemetry and the complete log are
retained unchanged both remotely and in the local failure-evidence backup.
The recovery makes the persistent cache an explicit CLI and state contract,
passes the cache environment to every child, and extends the startup audit to
prove both pinned model snapshots are complete and usable offline. The next
attempt must use a distinct `taxonomy_two_stage_formal_v1_r3` root and repeat
all startup, bounded-candidate, verified-sync and exact-resume gates. The
cache-binding focused safety suite passed 19 tests and the repository-wide
suite passed 406 tests before this recovery was submitted.
