# Taxonomy final held-out evaluation: Phase 6 readiness decision

Date: 24 August 2026

Protocol: `taxonomy_two_stage_final_test_v1`

Decision: **GO to request one-time release authorisation; NO-GO to execute before that authorisation**

## 1. What this decision means

The validation-selected scientific programme is closed. The repository is
ready to move from test-blind preparation to a separately authorised, locked
held-out evaluation. This document does not authorise that transition and does
not contain an official-test path, hash, row identity, label, or result.

Only a new explicit user instruction may create the one-time release manifest.
That manifest must bind the clean, pushed execution commit and the exact
official-test bytes and ordered row identities before the runner can proceed.

## 2. Scientific freeze

The dissertation mainline is taxonomy generalisation under an
example-filtered leave-one-aspect-out protocol. The central evidence is the
true two-stage separation of aspect presence and conditional sentiment. The
final validation-selected candidate is the fixed stage-wise composition:

- Stage 1: Frozen Qwen few-shot score and frozen seen-only threshold;
- Stage 2: QLoRA sentiment scores and frozen runner-up threshold;
- decoder: top one plus a thresholded runner-up, capped at two sentiments;
- no retraining, retuning, learned routing, or joint optimisation.

The candidate is explicitly an exploratory, validation-selected composition.
Its Level-2 D validation held-out pair F1 is `0.525931`; its nominal paired
gains are `+0.021048` over Frozen Qwen few-shot and `+0.046028` over QLoRA.
These are conditional validation estimates, not final confirmatory claims.

## 3. Locked held-out questions

The confirmatory family is intentionally small and hierarchical:

1. H1: fixed composition versus Frozen Qwen few-shot on Level-2 D;
2. H2: fixed composition versus QLoRA on Level-2 D, confirmatory only if H1
   passes; otherwise descriptive.

The primary endpoint is the aspect-balanced mean held-out pair micro-F1 across
the twelve fixed folds. The pooled held-out pair F1 is sensitivity only. The
bootstrap uses 20,000 synchronized review-cluster draws and seed 13.

The complete descriptive Level-2 N/D base roster and three Level-4 D stress
groups remain reportable regardless of outcome. Level 1, Level 3, rich R,
Router, calibration, fusion, retrieval, reverse composition, and legacy
protocols are excluded from the final run. A descriptive winner cannot replace
the frozen candidate after reveal.

## 4. Completed release-preparation evidence

| Gate | Evidence | Result |
|---|---|---|
| Preregistration | Machine-readable protocol plus human-readable registration | Passed |
| Artifact inventory | 159 entries, 459 files, 8,468,719,835 bytes | Zero hash mismatches |
| Job graph | 93 scoring + 12 composition + 1 seal + 1 analysis | Exact 107-node graph |
| Validation mirror | 12 folds, 1,057 review clusters | Exact H1/H2 reproduction |
| Synthetic dry-run | 105 score bundles and 210 independently verified backup receipts | Passed end to end |
| Reveal boundary | Score graph must be complete, immutable, hashed, backed up, and sealed first | Enforced |
| Official-test guard | Pre-release config and inert release template | `include_official_test=false`; `test_contract_count=0` |
| Regression suite | Whole repository | 503 passed |
| Static checks | New runner, audit modules, scripts, and tests | Passed |
| Thesis build | XeLaTeX/BibTeX multi-pass | 62 pages; no fatal error |
| Security scan | Frozen tracked preparation files | Zero detected secret patterns |

The machine-readable pre-release audit is
`docs/experiments/taxonomy_final_test_v1_pre_release_freeze_audit.json`.

## 5. Execution and failure boundary

The final run reuses the frozen train-only checkpoints/adapters,
demonstrations, prompts, resources, and validation-selected thresholds.
Train-plus-validation refitting is forbidden. Identical seen-candidate N/D
scores are shared from one canonical score object by construction.

All score bundles must be produced without labels, hashed, copied to immutable
backup storage, re-hashed, and sealed before the single reveal. Interim fold
metrics are prohibited. Data/hash/schema mismatch aborts before reveal. There
is no post-release batch fallback. OOM, non-finite scores, collapse, a third
sentiment, or resume conflict is a recorded protocol failure. Low performance
is a scientific result, never a rerun condition.

## 6. Resource plan

The locked inference-only plan uses up to three RTX 4090 Pods:

- Pod A: Frozen Qwen few-shot, all registered Level-2 and Level-4 scopes;
- Pod B: QLoRA plus DistilBERT registered scopes;
- Pod C: Frozen Qwen zero-shot, TF--IDF, E5, and DCWT.

Expected consumption is 16--18 GPU-hours, with a hard planning ceiling of
20 GPU-hours. At `$0.75/hour`, the expected cost is `$12.00--$13.50`, with a
`$15.00` ceiling. Estimated wall time is 8--10 hours. Pods are stopped only
after their exact job boundary and verified backup receipts are complete.

## 7. Remaining legal transition

Before execution, and only after explicit one-time authorisation:

1. verify the preparation commit is clean and equal to upstream;
2. bind that commit, the exact official-test file hash, ordered row-identity
   hash, expected 1,587 rows, backup identifier, timestamp, and no-fallback
   policy in a new release manifest;
3. run the manifest validation gate; and
4. execute the complete graph once, revealing metrics only after sealing.

Until those steps occur, the correct operational state is **ready but sealed**.
