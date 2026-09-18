# Training-policy sensitivity extension — 8 September 2026

**Current status: COMPLETE, 8 September 2026.** Final combined analysis and
backup checks passed through the local postprocessing wrapper described at
the end. Earlier browser/campaign statements below are historical, not current
execution status.

The user approved completing all original model families except retraining
QLoRA. The parent sensitivity study remains immutable and continues running.
This extension is registered before new DCWT outcomes and does not reopen
official test. No Git commit or push is permitted in this turn.

## Scope

1. Finish the parent TF-IDF, DistilBERT, few-shot and fixed-old-QLoRA-decoder
   diagnostic. Do not silently replace the existing local neural replays.
2. Run the original DCWT recipe under both training policies, all twelve
   folds, D and N. Preserve kernel ridge as the primary family and report its
   three existing transfer controls. No new hyperparameter values are added.
3. Audit frozen E5 and zero-shot Qwen as training-policy-invariant references.
   Their forward scores do not depend on task-specific training reviews.
   Candidate interfaces, full validation rows, seen selection targets and
   decoder remain identical. This is dependency-based exact reuse, not an
   independent repeated inference experiment.
4. Combine all completed records into a six-method sensitivity table plus
   the mixed-component diagnostic. QLoRA's original decoder remains explicitly
   review-filtered. No fully masked-trained QLoRA/hybrid result is claimed.

## DCWT implementation boundary

Use the original four SVD dimensions, three LR C values, seven ridge alphas
and five cosine temperatures. Select within each family using the original
eleven pseudo-unseen seen-aspect tasks and presence-F1/AP tie rules. This
model-specific selection differs from TF-IDF's pair-F1 selection and must
not be changed merely to unify the code. N inherits D's selected recipe and
thresholds. The definition encoder is frozen and its small descriptor cache
is computed once on CPU and shared exactly by both new local policy arms.

Preserve fitted TF-IDF vectorisers, SVD projectors, scalers, all seen-weight
candidates, selected family settings, train-row manifests, scores, review
confusion counts and SHA-256 receipts. Both policies are local replays, not
claims of bitwise reproduction of old GPU descriptor passes.

## Frozen-method invariance boundary

The 22 August audited local replay is the source authority. Its recorded
fold-result and row-evidence hashes must match. Validate all 1,057 row IDs,
held-out gold counts, candidate/selection input identity, thresholds and
recomputed held-out F1. Both policy views refer to the same immutable source
result. A zero delta is an architectural invariance, not an empirical claim
that all frozen models are insensitive to every form of protocol change.
Changing validation filtering or prompts would invalidate this reuse.

## Execution and cloud status

The user has permitted RunPod for GPU work. Browser control of the existing
RunPod tab still returned `Debugger unattached`, and a fresh same-site tab
timed out. No Pod was deployed, no data was uploaded, and no new cloud charge
was initiated. Local GPU work and the existing guarded Qwen fallback remain
active. Any future offload needs an explicit transport/receipt contract and
must prevent the local fallback from calculating the same prompts twice.

CPU work uses two threads and a two-hour per-fold-policy safety ceiling.
This is a ceiling, not a runtime forecast. Failures are preserved and not
automatically retried. All new artifacts are separate from the thesis and
historical validation/test outputs. A second local directory is a backup
against accidental overwrite, not an off-device backup.

## Checklist

- [x] Inspect live parent jobs without changing them.
- [x] Register the approved extension and preserve original recipes.
- [x] Attempt browser reconnection without starting billing.
- [x] Implement and test CPU descriptor/weight-transfer route.
- [x] Verify frozen E5 and zero-shot references, all twelve folds.
- [x] Pass first complete DCWT fold-policy gate and backup.
- [x] Complete all 24 DCWT fold-policy fits and four families.
- [x] Join parent and extension evidence, run paired bootstrap and audits.
- [x] Report results and outstanding cloud/operational limitations.

## Execution receipt — 8 September, 16:48 China time

- Six synthetic/preflight tests passed. The retained TF-IDF/SVD/scaler
  transformations reproduced the legacy implementation exactly on the
  fixture. Tampered artifacts and failed parent dependencies were rejected.
- CPU description preparation completed with 24 frozen inputs and a
  SHA-256-verified second local copy.
- E5 and zero-shot Qwen invariance audit passed for 48 fold-policy units,
  96 D/N condition records, with zero new inference and zero test contracts.
- Hidden CPU extension campaign launched as PID 28492 at 16:46. The first
  job was `dcwt__review_filtered__l2-a01`. It completed in 117.2 seconds,
  including all four generator families and D/N scoring. Its complete score
  reconstruction and backup gate passed before starting the masked a01 arm.
  This first duration is not an all-fold timing guarantee.
- Parent CPU/GPU/finish PIDs remain 15660/33568/37112. Their configuration and
  source files were not changed. Latest sampled GPU temperature was 74 C,
  utilisation 100%, with hardware/thermal slowdown inactive.
- A final browser retry on the newly opened RunPod tab again returned
  `Debugger unattached`. The tab is visible in the inventory but cannot be
  controlled. No new RunPod job, upload or charge was initiated here. Existing
  account-wide billing cannot be verified from this disconnected surface.

The queue verifies every DCWT fold's full score grid, review-level confusion
counts, N/D shared seen scores and local backup before advancing. After all
24 DCWT jobs it waits for the existing parent to finish, then invokes the
combined analysis. This analysis requires 276 fold-policy-method units and
552 condition records, including the existing size controls and DCWT controls.
It uses 20,000 synchronised review-cluster draws with seed 13. These totals
include controls, not 276 independently trained neural models.

Commands:

```text
python -m pytest tests/test_taxonomy_training_policy_extension.py -q
python scripts/run_taxonomy_training_policy_extension.py --phase prepare
python scripts/run_taxonomy_training_policy_extension.py --phase invariance
python scripts/campaign_taxonomy_training_policy_extension.py
```

The analysis command is dependency-triggered, not a scheduled automation.
Keep the machine awake for the local campaigns. No Git commit/push, thesis
edit, official-test read or QLoRA retraining was performed in this extension.

## Final complete-case analysis

All 24 DCWT fits and 48 invariant fold-policy units were verified. Following
completion of the parent, the original combined analysis ran successfully
through `scripts/finish_policy_analysis_20260908.py`. The old interrupted
extension campaign was not restarted. Its historical waiting state is not
the authority for these completed artifacts.

`outputs/experimental/taxonomy_training_policy_extension_v1/analysis/audit.json`
records PASS for 276 fold-policy-method units, 552 N/D records and 46 system-
conditions, with 1,057 shared review clusters, failure=0 and test-contract=0.
All registered parent, extension and analysis backups were fully rehashed.
The report builder independently reconstructed all 20 combined paired
intervals from the sealed 20,000-draw bootstrap arrays.

Primary DCWT D delta is -0.011419, nominal 95% interval
[-0.016504, -0.006328]. Importantly, 8/12 folds improve and 4 worsen, with
larger losses determining the negative mean. The nearest-description D
control is effectively inconclusive: -0.000364 [-0.004857, +0.003972].
The frozen references remain exactly invariant and are not independent reruns.

The primary report recommends retaining the completed, explicitly review-
filtered main protocol and reporting sensitivity, not claiming filtering is
universally preferable. Complete masked QLoRA remains untested. Report and
unsent supervisor reply: `../Thesis/Training_Policy_Study_20260908/`.
Combined analysis receipt SHA-256:
`0b0a1e19164c4f1843789ef87def82f5f539ab7a99f41255ca4e9d08ca64986b`.
No thesis edit, official-test access, Git commit/push or cloud restart occurred.
