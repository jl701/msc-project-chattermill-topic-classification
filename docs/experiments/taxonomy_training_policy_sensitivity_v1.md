# Training-policy sensitivity: registered execution checklist

**Current status: COMPLETE, 8 September 2026.** The historical execution notes
below remain unchanged. Final local evaluation, paired analysis and backup
audit are recorded in the close-out section at the end. The current combined
table is the extension analysis, not an old running campaign-state snapshot.

Registered 8 September 2026, after the project's official-test results were
already revealed. This is a new **validation-only, post-hoc sensitivity study**,
not a new confirmatory evaluation and not a change to the final tested system.
The user authorised this bounded study on 8 September. No supervisor message
will be sent by the executor.

## Questions and scope

Measure information lost by removing target-annotated reviews. Compare that
policy with retaining all reviews while excluding the held-out aspect from
supervised targets. Study all twelve L2 folds with TF-IDF and DistilBERT under
D and N. Add three fixed, review-count-matched TF-IDF subset controls. Audit
few-shot demonstration changes and execute compatible new inference. Finally,
where exact cached QLoRA grids are verified, diagnose a new few-shot gate with
the **old review-filtered QLoRA decoder**, without joint tuning.

No QLoRA retraining, L3/L4 expansion, rich-description search, router work,
official-test access, or new cloud billing is authorised by this plan.

## Design decisions fixed before new outcome computation

- The machine-readable config is the authority for scope and parameters.
- Masking retains held-out-only reviews. These may supply negative supervision
  for the eleven seen candidates at Stage 1. They supply no gold-present seen
  Stage-2 examples. This intentionally differs from the old legacy masking
  helper, which discarded rows with no remaining labels.
- All target labels are removed from every training supervision field. The
  held-out candidate is never instantiated as a negative target.
- TF-IDF uses its original two-stage implementation and all eligible examples.
  Both policies are replayed locally. The three size controls use seeds 13,
  29 and 47. All are reported and none is selected using outcomes.
- DistilBERT replays both policies on the same local GPU using the original
  three learning rates, three epochs and 2,048 examples per stage. Seed 13 is
  reset **before head initialisation**, not just before optimisation. Replays
  are not claimed bitwise equal to the historical cloud realisations.
- The same seen-only model/threshold selection rule is used in both arms.
  Threshold values may differ. N inherits its own arm's D selection.
- N and D share identical seen-candidate scores by construction.
- Few-shot uses the original deterministic selector, demonstration counts,
  prompts, revisions and max length. Exact content/order and prompt hashes
  decide reuse. No retrieval or demonstration-count search is allowed.
- The mixed composition uses the masked few-shot source's own selected gate
  and the original QLoRA runner-up threshold. It does not optimise a gate
  against hybrid outcomes and does not represent masked QLoRA training.
- Primary descriptive contrast is the 12-fold aspect-balanced held-out pair
  F1 difference. Paired bootstrap uses synchronised review draws, 20,000
  repetitions and seed 13. Intervals are conditional, exploratory evidence.
- Low held-out F1 alone is a scientific outcome, not a reason to restart or
  modify the design. Whole-grid empty/constant collapse, nonfinite values,
  integrity faults, OOM and repeated thermal/hardware events require a stop.

## Pre-flight and preservation

Base HEAD: `919c2293bb71bc6ec43e6666e94f7cb3433b0956`, current branch
`agent/final-thesis-rewrite`. `git fetch origin` succeeded. Existing dirty
thesis sources, reference audits, report notes and untracked review material
are user-owned and will not be reverted, staged or silently rewritten.

Local hardware: RTX 5050 Laptop GPU, 8 GB. Recorded ML environment:
Python 3.12.7, PyTorch 2.10.0+cu128,
Transformers 4.57.6, pandas 2.3.3, scikit-learn 1.8.0.

Raw rows, model weights, scores and detailed manifests remain under ignored
outputs. Only safe code, configs, tests, aggregate reports and audit summaries
are eligible for Git. Completed files receive SHA-256 receipts and a separate
local backup. A second directory on the same disk is not off-device backup.

## Checklist

- [x] Read execution discipline, inspect code and original recipe.
- [x] Check GPU and existing environment, fetch remote, account for dirty tree.
- [x] Register this bounded study before new outcomes.
- [x] Implement separate masking route and prove supervision exclusion.
- [x] Complete synthetic tests, model smoke and integrity gates.
- [x] Data-loss, training-manifest and few-shot demonstration audit (12 folds).
- [x] TF-IDF original/masked and three size controls, N/D (60 fold fits).
- [x] DistilBERT paired local policies (72 LR candidates, 24 selections, N/D).
- [x] Few-shot exact reuse assessment and required new inference.
- [x] Fixed old-QLoRA decoder component-substitution diagnostic.
- [x] Paired analysis, N/D sensitivity, all size seeds and stage diagnostics.
- [x] SHA-256 verification, backups, safety audit and complete report.
- [x] Relevant tests, safe-file review and final user report. Git commit/push withheld at the user's explicit request.

## Execution log

Detailed run commands, timings, deviations, failures and results will be
appended here. No result is available at registration.

### 8 September: pre-scale gates

The 23 initial synthetic tests passed. All twelve data and demonstration audits
completed with no held-out supervision or test access. Eleven of 24 masked
fold-stage demonstration sets changed. The train-only DistilBERT smoke passed
at length 256 and batch 32, with 2.23 GB peak CUDA allocation.

The first TF-IDF gate failed while serialising the threshold-selection
dataclass's pandas grid to JSON. This was a new writer integration error,
not a model failure or an outcome-guided change. No full evaluation metrics
were saved. The failed gate and model file are retained under the study's
`smoke/failed_serialisation_gate` directory. Add a DataFrame/numpy JSON test,
retain nonfinite rejection, and repeat the unchanged first-fold gate before
scaling. The study's scientific config is unchanged.

### Frozen-Qwen exact-source reuse execution detail (before cache outcomes)

The paired few-shot sensitivity will use canonical exact-source replay for
both policies. Cache keys bind the pinned model, unchanged shared-demonstration
prompt branch, full rendered prompt, demonstration content/order, length 1024,
batch 6 and NF4/double-quantisation/float16-compute contract. A candidate from
another outer fold is reusable only when all model-visible inputs are exactly
identical and the destination demonstration eligibility has independently
passed. Source priority is ascending original fold ID, D before N, fixed
without looking at numerical scores. No averaging or performance-based
source choice is permitted. This identifies the exact source realisation,
not a claim that different batching passes have identical numerics.

The canonical replay is a separate sensitivity baseline, not a replacement
for historical few-shot formal results. Both policy arms use the same cache,
so identical selected demonstrations imply identical predictions and selection
inputs. Unmatched inputs are scored once and shared only on exact key match.
The original QLoRA canonicalisation rule is inherited with explicit provenance.

### Local execution checkpoint, 8 September, approximately 07:20 UTC

The serialization fix passed its regression test and the unchanged first
TF-IDF gate completed. The first DistilBERT fold then completed all three
learning rates, selected on seen candidates, scored N/D and verified its
checkpoint backup in 609 seconds. Full campaigns now run independently:

```text
python scripts/campaign_taxonomy_training_policy_sensitivity.py --worker cpu
python scripts/campaign_taxonomy_training_policy_sensitivity.py --worker gpu
python scripts/prepare_taxonomy_policy_qwen_sources.py
python scripts/finish_taxonomy_training_policy_sensitivity.py
```

Use the existing Miniconda ML environment, repository working directory and
offline cached models. CPU worker has 60 fold fits. GPU worker has 24 fold
selections, each comparing three learning-rate candidates. Every completed
fold is verified and backed up before the worker advances. At this checkpoint
CPU was 14/60 and GPU 3/24, both with failure and test-contract counts zero.
Live state files, not these historical numbers, determine current progress.

The Qwen preparation is complete. It verified 840 selected source files and
identified 48,974 unique new prompts. All source grids, exact query mappings,
demonstrations and provenance have an immutable preparation receipt. The
inference duration remains an estimate until its first chunks run. Its
registered time cap is 24 local GPU hours, not an approval for cloud usage.

The finish script waits for exact GPU 24/24 completion and lock release before
starting frozen Qwen. It monitors temperature, thermal/hardware slowdowns,
disk headroom and code/config hashes. It then evaluates both few-shot policy
arms and both mixed-component diagnostic arms. Complete-case analysis waits
for CPU 60/60 and all 132 fold receipts. Failed tasks are not automatically
restarted. The script only terminates children it launched. Independent
healthy core workers keep their own guards.

The current test run passed 42 tests. An additional real-artifact check
reconstructed review-level TP/FP/FN from eight completed N/D score grids and
verified bitwise-identical seen-candidate inputs to the N/D analyses. Final
analysis repeats score reconstruction and receipt-identity validation across
every fold, rather than trusting headline metrics alone.

Interpretation remains fixed: review-count matching does not match every
aspect's label frequency, co-occurrence structure or Stage-2 example count.
DistilBERT keeps 2,048 examples per stage, so increased eligible reviews change
the sampled supervision rather than increasing that training budget. The
few-shot comparison tests the registered deterministic selector, not all
possible demonstration designs. No finding in this study substitutes for
training a label-masked QLoRA adapter.

Status: **running, not complete**. Original thesis text and historical
validation/official-test outputs have not been modified by this study.

### Final close-out, 8 September 2026

The local interruption and cloud transport were resolved as documented in
`taxonomy_policy_cloud_handoff_20260908.md`. The original interrupted finisher
was not restarted. DistilBERT completed 24/24 at 21:16:31 China time. The
48,974 missing exact Qwen prompts had already been completed on the authorised
cloud Pod, fully downloaded, hash checked and backed up before that Pod stopped.

The user then explicitly authorised the remaining local evaluation/analysis.
`scripts/finish_policy_analysis_20260908.py` invoked the original evaluation
command followed by the unchanged parent and extension analysis scripts.
The wrapper disabled GPU access, recorded frozen source/config hashes and
retained every command and stage log. No inference or training was launched.

All stages completed in 568.27 seconds including the final backup audit:

- Parent: 132 units, 264 N/D records, 22 system-conditions.
- Combined extension: 276 units, 552 records, 46 system-conditions.
- 20,000 synchronised review-cluster bootstrap draws, seed 13.
- Score-to-TP/FP/FN reconstruction, N/D seen-score identity and exact scopes PASS.
- 207 sealed directories fully verified in both source and local backup.
- Frozen source/config hashes unchanged, failure_count=0, test_contract_count=0.
- 29 existing sensitivity/extension/receiver tests passed.

In D, masked-minus-filtered held-out F1 was -0.029066 for TF-IDF,
-0.006155 for DistilBERT, -0.032195 for few-shot, and -0.025441 for the
mixed-component diagnostic. DistilBERT's nominal interval crosses zero.
The other three parent D intervals lie below zero. These are exploratory
conditional intervals, not confirmatory protocol-superiority claims.

Final evidence: `outputs/experimental/taxonomy_training_policy_sensitivity_v1/analysis`
and `postprocess_20260908/completion_audit.json` beneath that study root.
The reader-facing HTML/Markdown report, two figures, complete headline table
and unsent Griffin reply are at `../Thesis/Training_Policy_Study_20260908/`.
The report explicitly distinguishes masked QLoRA training (not performed),
the fixed-old-decoder diagnostic, structural frozen-method invariance and
paired local/exact-source replays. Original thesis and official-test results
were not changed. No new supervisor message or Git publication was made.
