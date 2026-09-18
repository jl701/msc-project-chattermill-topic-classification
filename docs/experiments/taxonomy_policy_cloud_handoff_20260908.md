# Approved operational recovery and Qwen-only cloud handoff

Registered 8 September 2026 before any new cloud inference. This supplements,
but does not rewrite, the immutable scientific sensitivity and extension
configs. The user authorised cloud GPU work and, after MCP authorisation,
asked to check the connection and proceed. No Git commit/push is permitted.

## Verified starting point

The hosted RunPod MCP returned successful Pod, volume and billing responses.
Historical billed Pod IDs match the project's known formal validation/test
workers. No existing Pods or network volumes were returned. Recent three-day
spend was zero in the authorised account scope.

At approximately 17:44 China time, local campaign states recorded CPU 60/60,
DistilBERT 16/24 and DCWT 24/24. Their processes, including the dependency
finishers, no longer existed. Last state updates were approximately 17:40.
This coincides with the requested Codex reload, but causation is not proven.
The unfinished DistilBERT review-filtered a09 fold has no completion receipt.

## Recovery, without scientific changes

- Verify the recorded code/config hashes, completed receipts and separate
  local backup copies before reuse. Do not use recorded counts as proof.
- Preserve old GPU/finisher campaign directories and incomplete a09 artifacts
  in a dated incident directory. Do not delete evidence or overwrite them.
- Restart the unchanged local GPU campaign only after dead-process and
  immutable-artifact checks. It skips verified complete folds and reruns the
  interrupted, unsealed a09 fold from the same initial seed/recipe. This is
  operational recovery, not a result-guided retry. Keep all 24 DistilBERT
  policy-fold fits on the same local GPU/environment.
- Do not restart the old local Qwen finisher. One separate cloud handoff owns
  every pending Qwen key. Completed CPU and DCWT jobs are not rerun.

## Cloud boundary

One Community RTX 4090 Pod, official pinned PyTorch image. Catalog price is
indicative only. Verify actual hourly charge after creation and stop if it
exceeds USD 0.80/hour. Setup/inference ceiling: six hours, with inspection of
the first chunks before accepting a runtime estimate. Only Stop after verified
local receipt/backup, never automatic Terminate/Delete of unbacked-up results.

Upload an allowlisted code/input bundle, not the repository directory. It
contains the frozen inference code, pending validation review/candidate inputs,
the train-selected demonstrations, scientific config and source receipt.
No official-test data, label vault, credentials, full training datasets,
thesis or historical result directories are uploaded. A SSH public key may
be sent to the intended Pod. Its private key remains local.

Reuse the existing inference function with its original file hash. Keep the
prepared computational contract: pinned Qwen3-4B-Instruct-2507 revision,
NF4 double quantisation with float16 compute, length 1024 and batch 6.
No prompt, seed, demonstration selector, threshold or decoder changes.
Record the cloud wrapper, environment and input manifest separately. Hardware
relocation is not a claim of bitwise equivalence with a hypothetical new
local pass. Previously verified exact source scores are not recomputed or
selected by their values.

Every published chunk is finite, normalised, duplicate-free and bound to the
original preparation hash. All 48,974 expected keys must match exactly before
evaluation. The cloud reads no evaluation gold labels and computes no F1.
Transfer sealed chunks and receipt to local storage, verify SHA-256, then
back up locally. A second directory on the same disk is not off-device backup.

The unchanged local evaluation and paired analysis run only after complete
inference integrity and DistilBERT completion. The extension analysis follows.
Historical official results and thesis stay unchanged. This remains post-hoc,
validation-only sensitivity, not new confirmation or masked-QLoRA training.

## Failure policy

Preserve and stop on OOM, nonfinite probabilities, missing/duplicate keys,
hash/resume conflicts, unexpected constant predictions, disk exhaustion or
repeated thermal/hardware slowdowns. No automatic scientific retry or retuning.
Pure connection failures do not justify launching another duplicate Pod.

## Execution checkpoint and blocked data transfer

- Recovery audit passed for 148 complete top-level artifact units (60 TF-IDF,
  16 DistilBERT, 24 DCWT and 48 frozen-reference units), including receipt
  identity and file SHA-256 in the original and separate local backup.
  The first audit attempt used the wrong folder depth for frozen references,
  returned zero paths and stopped. Correcting that audit-only glob produced
  the expected 48 units. No experiment result was edited or relaxed.
- Old GPU and finisher campaign directories plus the unsealed a09 partial
  outputs were moved into `operational_recovery_20260908/interrupted_attempt`.
  No files were deleted. The unchanged GPU campaign resumed as PID 37588,
  verified and skipped 16 completed jobs, then restarted the interrupted a09
  filtered fit. Local GPU activity was 98%, 73 C, with failure/test counts zero.
- The allowlisted package has 76 files, 3,342,001 bytes and SHA-256
  `6f7a3b1145b0b64ae90e13c4c2d401515afa1ad28e0b662778f559c1bcfad3ca`.
- MCP created Community Pod `d0i7im52yt6yu6` at 09:56:14 UTC. The reported
  compute rate was USD 0.34/hour. SSH verified RTX 4090 / 24,564 MiB,
  driver 580.126.09 and Python 3.12.3. The Pod belongs to the same authenticated
  account whose historical billing records match the project workers.
- The SCP tool action was rejected before execution by safety review because
  it would export review text and demonstration examples to that external
  destination without sufficiently explicit data-transfer consent. No package
  or setup script was uploaded. No alternate transfer channel was attempted.
- Stop Pod succeeded with status `EXITED`. Compute is stopped. The 60 GB
  persistent Pod disk remains, so this is not a claim of zero storage charges.
  No new model download, inference, training or test access occurred remotely.
- Await explicit consent for the identified data and destination before
  restarting the Pod and transferring. No old local Qwen finisher was restarted,
  so no duplicate pending-key inference can start. DistilBERT continues locally.
- The two targeted sensitivity/extension test files passed: 14 tests in
  7.77 seconds. The first invocation lacked this src-layout project's
  `PYTHONPATH=src` and failed during import, before executing tests. Repeating
  with the normal import path passed without a source or experiment change.

## Explicit upload consent and host-capacity block

The user subsequently replied "允许" to the request identifying the validation
review text, training-selected demonstrations, code bundle and Pod
`d0i7im52yt6yu6`. The same package was rehashed before any transfer and still
matched `6f7a3b1145b0b64ae90e13c4c2d401515afa1ad28e0b662778f559c1bcfad3ca`.

Two subsequent MCP start attempts returned HTTP 400: "There are not enough
free GPUs on the host machine to start this pod." This is a host-capacity
failure, not an OAuth or experiment failure. No SSH upload or cloud inference
occurred. The original destination remains stopped. The live RTX 4090 catalog
reports low stock elsewhere, but availability of a replacement is not yet
verified by provisioning. Obtain consent to use a replacement Pod in the same
project account before changing the specifically identified upload destination.

During these checks the unchanged local GPU campaign advanced to 17/24, with
`distilbert__label_masked_all_reviews__l2-a09` active, failure count zero and
test-contract count zero. No Git commit/push or scientific configuration change
was made.

## Replacement-destination authorisation

The user explicitly authorised a replacement Pod in the same team account and
upload of the same validation-only bundle, then set the objective to complete
cloud few-shot inference and authorised the remaining in-scope operations.
Deploy at most one active replacement RTX 4090 for this workload, with the
same image, compute contract and price ceiling. Keep the unavailable old Pod
stopped. This does not authorise official-test access, scientific tuning,
QLoRA retraining or GitHub publication.

The replacement created at 10:16:14 UTC is `fsy1xq9jlm0fmi`, named
`taxonomy-policy-qwen-validation-20260908-r2`. The authenticated create response
reports Community RTX 4090 x1 and USD 0.34/hour, with the same pinned image,
30 GB container disk and 60 GB `/workspace` persistent Pod disk. The original
Pod is not restarted. This record describes provisioning, not successful
model inference. SSH readiness and data-transfer verification follow.

### Working destination and verified upload

The r2 Community host exposed no direct SSH port after its container became
ready. Its SSH proxy rejected the non-interactive command with "Your SSH
client doesn't support PTY". No dataset was uploaded to r2. Stop returned
`EXITED`, before provisioning the replacement below.

Under the user's replacement/in-scope-operation approval, the active Pod is
now Secure RTX 4090 `y9yh6i4r9y5ph4`, named
`taxonomy-policy-qwen-validation-20260908-r3`, created at 10:20:06 UTC. Its
actual advertised compute rate is USD 0.74/hour (below the 0.80 ceiling).
The ephemeral SSH endpoint is intentionally omitted from the public record.
Direct SSH verified RTX 4090 / 24,564 MiB,
driver 570.169 and Python 3.12.3. Changing cloud class is operational only.
The scientific data, model revision, package and inference settings are fixed.

SCP successfully uploaded the same authorised package and setup script.
Remote SHA-256 matched the package digest above, and the setup script matched
`d9b6c7e37e57dcdb7502a513c7a1620565d2aa8e9dd7b58d4c40dc2351e9bb53`.
`setsid bash /workspace/setup_policy_qwen_cloud.sh` launched detached with
logs at `/workspace/policy_qwen_setup.log`. The previous Pods remain stopped.

The new local transport-only receiver is
`scripts/receive_policy_qwen_cloud.py`. It verifies exact expected prompt
keys, mode/demonstration identities, finite normalised probabilities and
payload hashes. Received chunks are copied to a separate local backup as
they arrive. Final import additionally requires the exact remote receipt,
all 48,974 keys, no running model process, verified provenance and matching
local/backup file SHA-256. It then writes `received/READY_TO_STOP.json`.
The receiver does not change billing itself: the authorised MCP Stop action
must follow that verified completion. Thirteen synthetic receiver tests pass.
These tests neither read a dataset nor run a model.

### First real inference checkpoint

The input allowlist and all 48,974 keys passed on the Pod. The pinned model
download completed, followed by guarded parent PID 123 and inference PID 310.
At 10:30 UTC, 2,112 prompts had completed in approximately 82 seconds of
inference, with failure/test counts zero. GPU utilisation was approximately
95%, temperature 79 C and all recorded thermal/hardware slowdown flags inactive.
This establishes actual scoring, not just a running container.

Local receiver PID 35856 was launched hidden and detached from the app's job
object. It runs against the verified direct SSH endpoint. Its operational
state and logs are in `cloud_handoff_20260908/received` and `receiver.log`.

### Transport-only timeout repair

The initial receiver successfully verified and backed up 10 chunks (960 keys),
then its next multi-source SCP command exceeded 120 seconds. OpenSSH's repeated
connections made that transfer inefficient. The receiver had already exited
when a stop was attempted (`NoSuchProcess`), so no experiment process was
stopped. `FAILED_scp_v1.json`, the original log, partial staging files and the
original receiver source snapshot are preserved in the handoff directory.

Receiver v2 transfers the same allowlisted files as one gzipped tar stream per
batch over authenticated SSH. It rejects extra, duplicate, non-regular and
unsafe archive members, then applies the same chunk and file hash checks.
Existing verified chunks are revalidated, not replaced. No remote input,
model code, prompt or science config was changed. At this repair checkpoint
the cloud had completed 11,424 prompts with failure/test counts zero, GPU
100%, temperature 80 C and no active slowdown flags. The cloud inference was
not restarted. Follow `receiver_batched_v2.log` for the replacement receiver.

Receiver v2 PID 37716 passed 15 synthetic tests, including archive scope and
symlink rejection. It caught up from 960 keys to 13,536 keys in its first
successful batch cycle. An independent incremental audit subsequently checked
156 chunks (14,890 keys), with payload/probability validation and matching
backup SHA-256. The separate final audit script
`scripts/audit_policy_qwen_cloud_completion.py` is prepared to verify the
remote inventory, three complete local score copies, all frozen source/input
hashes and provenance before Stop. It does not compute any performance metric.

### Empty provisioning-attempt cleanup boundary

The user's broad authorisation covers completion of this cloud workload and
its resource cleanup. Pods `d0i7im52yt6yu6` (host capacity unavailable) and
`fsy1xq9jlm0fmi` (no direct SSH) are confirmed `EXITED`. Neither ever received
the data bundle or ran this experiment. Only their empty provisioning disks
may be removed to prevent storage waste. Their IDs, failure reasons and
creation/stop facts remain recorded here. The active Pod `y9yh6i4r9y5ph4`
contains results and is explicitly excluded from this deletion scope.

Both attempted empty-Pod deletions were rejected by the safety reviewer before
execution: it requires explicit deletion consent identifying those Pod IDs or
stronger accessible proof of empty disks. No alternative deletion mechanism
was used. A subsequent list still shows both as `EXITED` and the inference
Pod as the only `RUNNING` resource. Their disks remain billable. This cleanup
block does not block the running inference or its verified-backup/Stop goal.
Ask for exact-ID deletion consent separately, rather than repeatedly retrying.

## Completed cloud-inference goal

- Frozen inference completed all **48,974/48,974** pending keys in
  **2,072.463 seconds (34.54 minutes)**, producing 514 score chunks.
- Failure count and test-contract count are zero. There was no model restart,
  QLoRA retraining, prompt/threshold search or official-test access. The only
  retry after inference started was the separately documented local transport
  repair. No performance metrics were computed for this cloud close-out.
- The receiver downloaded the complete 516-file score unit (514 chunks,
  final state and receipt), verified exact identities and hashes, imported it
  into the study's `qwen_inference` directory and backed it up locally.
- Independent close-out audit **PASS** rechecked all frozen package/source/input
  hashes, the exact 48,974 keys, three complete local score copies and model/
  execution provenance. Audit SHA-256:
  `8ff8a42b19467895ddc4d7ec82ec1958e0e7d9943579ec55e156df98a0d06789`.
- Score receipt SHA-256:
  `5099e422c6d6420123365e422aee76a50df475899e1baf68f754c2a42bf72874`.
  Provenance receipt SHA-256:
  `e266239c3b77540d821d391e752a5bd6074e4ffd4e0b0b868a88a3e72fb0802a`.
- MCP Stop for `y9yh6i4r9y5ph4` succeeded after the audit. A subsequent
  list at approximately **11:08 UTC / 19:08 China time** confirmed all three
  Pods `EXITED` with `runtime=null`. GPU compute billing is stopped.
  All three 60 GB persistent Pod disks remain. This is not zero total billing.
- The combined sensitivity/extension/receiver regression run passed **29 tests**.
  No Git commit/push was performed. The local DistilBERT campaign was still
  healthy at **21/24**, running masked a11, when this cloud goal closed.

### Deliverables and next boundary

- Study scores: `outputs/experimental/taxonomy_training_policy_sensitivity_v1/qwen_inference`.
- Separate local backup: `../cloud_backups/taxonomy_training_policy_sensitivity_v1/qwen_inference`.
- Close-out audit: `cloud_handoff_20260908/completion_audit.json` under the study output.
- Model/environment manifests and runtime evidence: `cloud_handoff_20260908/received/provenance`.
- Stop confirmation: `cloud_handoff_20260908/gpu_stop_receipt.json`.

The cloud inference goal is complete. Local validation evaluation and the full
paired comparison analysis are subsequent work, not claimed complete here.
The old interrupted finisher remains archived and was not relaunched. These
backup directories are local copies on the same computer, not off-device backup.

## Exact-ID authorised cleanup completed

On 8 September 2026, approximately 21:08 China time, the user explicitly
authorised permanent deletion of Pods `d0i7im52yt6yu6` and `fsy1xq9jlm0fmi`
and their disk data, while retaining `y9yh6i4r9y5ph4`. Fresh get calls matched
both old Pod IDs/names and showed EXITED before deletion. Both MCP delete
calls returned `success=true, status=204`. A subsequent unfiltered list
returned exactly one Pod, the retained `y9yh6i4r9y5ph4`, EXITED, runtime=null.

The two old 60 GB persistent Pod disks are permanently removed and cannot be
recovered. No local artifacts or result-holding Pod were deleted. The retained
60 GB result disk still incurs storage charges. No GPU compute is running.
This supersedes the earlier cleanup-blocked status without erasing its history.

At the same check, local DistilBERT had reached 23/24 completed fold-policy
units and was running `distilbert__label_masked_all_reviews__l2-a12` under
parent PID 37588 / child PID 19108. State failure/test counts remained zero.
At 21:07:51 the GPU was 98% utilised, using 4,076/8,151 MiB at 80 C and 64.95 W.
The latest three telemetry records reported thermal/hardware slowdown flags
inactive. The study still permits train/validation only. No official test,
new evaluation job, new training job or Git commit/push was launched by this
cleanup/status turn. Full local evaluation and paired analysis remain pending.
