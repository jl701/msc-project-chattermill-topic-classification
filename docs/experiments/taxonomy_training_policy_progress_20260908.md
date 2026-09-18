# Training-policy sensitivity: progress snapshot

**Historical snapshot, superseded later on 8 September 2026.** All registered
evaluation and analysis stages are now complete. The authoritative completion
records are `taxonomy_training_policy_sensitivity_v1.md` and
`taxonomy_training_policy_extension_v1.md` in this directory. The reader-facing
report is `../Thesis/Training_Policy_Study_20260908/research_report.html`, relative
to the repository root. The pending-work statements below are retained only as
the original progress record.

Observed: 8 September 2026, approximately 21:01 China time (UTC+8).

## Scope and reporting boundary

This is the post-hoc validation-only comparison of whole-review filtering and
correct held-out-label masking. It is not a replacement official-test result.
No official-test data were opened, no QLoRA model was retrained, and no Git
commit or push was performed in this status check.

Point estimates below use all twelve completed L2 folds per system/condition.
The endpoint is the unweighted mean of per-fold held-out pair micro-F1. D means
name plus the frozen minimal description, N means name only. The delta is
label-masked minus review-filtered. These are not pooled F1 scores.

For every reported fold, the metric JSON and held-out review-count CSV were
checked against their receipt SHA-256. Each count file contained 1,057 unique
validation row IDs. F1 was reconstructed from TP/FP/FN and matched the metric
JSON within 1e-12. Every aggregate contains twelve distinct folds. This status
check does not replace the pending full score-grid and bootstrap analysis.

## Completed paired point estimates

| System | D filtered | D masked | D delta | N filtered | N masked | N delta |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.277251 | 0.248185 | -0.029066 | 0.210762 | 0.205601 | -0.005161 |
| DCWT kernel ridge | 0.145314 | 0.133895 | -0.011419 | 0.144329 | 0.136322 | -0.008007 |
| DCWT nearest-description control | 0.153412 | 0.153048 | -0.000364 | 0.157193 | 0.151726 | -0.005467 |
| DCWT cosine-weight control | 0.142614 | 0.126098 | -0.016516 | 0.141358 | 0.126047 | -0.015311 |
| DCWT mean-weight control | 0.139573 | 0.117649 | -0.021923 | 0.138323 | 0.116700 | -0.021623 |
| Frozen E5 (structural invariance) | 0.222184 | 0.222184 | 0.000000 | 0.241473 | 0.241473 | 0.000000 |
| Frozen Qwen zero-shot (structural invariance) | 0.451298 | 0.451298 | 0.000000 | 0.435860 | 0.435860 | 0.000000 |

E5 and zero-shot Qwen do not change with this training-policy intervention
under the registered fixed evaluation/selection inputs. Their identical rows
are a structural-invariance audit, not independent re-training or replication.
The existing invariance audit reports 48 fold-policy receipts, 96 condition
records, zero new inference, zero failures and zero test contracts.

### TF-IDF count-matched controls

These use label masking but match the review-filtered training-row count,
with three registered subset seeds. Every row contains twelve folds.

| Subset seed | D held-out F1 | N held-out F1 |
|---|---:|---:|
| 13 | 0.252316 | 0.205705 |
| 29 | 0.254186 | 0.206985 |
| 47 | 0.253775 | 0.203881 |

All three D point estimates remain below the review-filtered 0.277251.
This does not establish statistical significance or isolate a universal
causal advantage of review filtering. Formal paired uncertainty is pending.

## Execution progress

- TF-IDF: 60/60 sealed fold-policy units, including 36 count-matched controls.
- DCWT: 24/24 sealed policy-fold fits, each containing four generator families.
- DistilBERT: 22/24 top-level fold receipts. Parent PID 37588 and child PID
  40772 were alive. Current job: review-filtered l2-a12 (the 23rd job).
  The current fit produced a checkpoint receipt at 20:59:12. Campaign state
  continued updating, with failure_count=0 and test_contract_count=0.
  GPU snapshot at 20:59:32: 96% utilisation, 2,814/8,151 MiB, 76 C,
  62.8 W, P4. No partial-fold mean is reported for DistilBERT.
- Frozen Qwen few-shot: 48,974/48,974 pending prompts scored on the cloud.
  Completion audit PASS, 514 chunks / 516 files, three verified local copies,
  zero inference failures and zero test contracts. Pure inference took
  2,072.463 seconds. Local policy-fold evaluation is not yet complete.
- Mixed-component diagnostic: still pending local evaluation. It may reuse
  the old review-filtered QLoRA Stage 2 only under that explicit diagnostic
  label. It is not a newly label-masked-trained QLoRA system.
- Parent 132-unit analysis and combined 276-unit analysis are not present yet.
  Final paired bootstrap intervals and the complete combined comparison table
  therefore remain pending.

The old interrupted local finisher is archived and was not restarted. This
turn did not launch inference, retraining, or a replacement cloud resource.
Next scientific work is unchanged local Qwen evaluation, completion of the
remaining DistilBERT units, then the registered full integrity/bootstrap
analyses. No additional cloud inference is currently required for those steps.

## Pod deletion request and actual outcome

The user confirmed permanent deletion of the unused old Pods. Read-only get
calls verified the two previously identified IDs and names, both EXITED:

- d0i7im52yt6yu6 — taxonomy-policy-qwen-validation-20260908
- fsy1xq9jlm0fmi — taxonomy-policy-qwen-validation-20260908-r2

Both delete calls were rejected before execution by the safety reviewer. It
requires the exact Pod IDs to be written in a trusted user confirmation or
stronger independently accessible proof of disposable disk contents. No
alternative API, browser route or indirect deletion was attempted.

A subsequent complete list returned exactly three Pods, all EXITED with
runtime=null, including those two old Pods and the preserved result-holding
y9yh6i4r9y5ph4. Therefore **nothing was deleted**. GPU compute is stopped, but
all three 60 GB persistent Pod disks remain and storage billing is not zero.
No current storage-price claim was inferred from the API's advertised GPU
cost field. The result-holding Pod is excluded from deletion.

## Evidence locations

- Parent outputs: outputs/experimental/taxonomy_training_policy_sensitivity_v1
- Extension outputs: outputs/experimental/taxonomy_training_policy_extension_v1
- Cloud completion audit: parent cloud_handoff_20260908/completion_audit.json
- Cloud Stop receipt: parent cloud_handoff_20260908/gpu_stop_receipt.json
- Current DistilBERT state: parent campaign/gpu/state.json
- Detailed cloud history: docs/experiments/taxonomy_policy_cloud_handoff_20260908.md

Parent config SHA-256 remains
`a7775e0d98de47969226d1be938889723bf9145e0268e691ccf531a1ee368ebc`.
The three local score copies are on the same computer, not off-device backup.

## Follow-up at approximately 21:08 China time

The user supplied explicit permanent-deletion consent containing both old
Pod IDs. Both deletes returned HTTP 204 success. A subsequent full Pod list
contains only `y9yh6i4r9y5ph4` (EXITED, runtime=null, 60 GB persistent disk).
Thus the two old Pods and their disks are now permanently deleted. This
supersedes the blocked-cleanup snapshot above. The retained disk remains
billable, but no GPU compute is running.

DistilBERT is now 23/24, running the final label-masked l2-a12 unit. Parent
37588 and child 19108 are alive, with failure_count=0/test_contract_count=0.
The 21:07:51 GPU snapshot is 98%, 4,076/8,151 MiB, 80 C, 64.95 W, P4.
The latest three thermal/hardware slowdown observations are inactive.

Finishing this unit enables complete-case analysis once the already-scored
Qwen few-shot and mixed-component units have been evaluated locally. Those
evaluation steps do not require new cloud inference and can run independently
of the remaining DistilBERT fit. This remains validation-only sensitivity work,
not authorisation for another official-test run. No new evaluation process
was started during this status/cleanup turn.
