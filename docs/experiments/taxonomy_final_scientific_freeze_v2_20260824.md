# Taxonomy final scientific freeze v2

Date: 24 August 2026

Status: **PASS for pre-release preparation; official test remains locked**

Protocol: `taxonomy_two_stage_final_test_v1`

## 1. Purpose and boundary

This freeze closes validation-based model development and binds the revised
two-stage thesis direction, final candidate, comparator family, descriptive
roster, inference resources, failure rules, and score/reveal workflow. It
supersedes the 22 August scientific freeze only for post-freeze candidate
selection and final-test preparation. The repaired formal-v2 validation
evidence remains unchanged.

This record is not an official-test release. At freeze time:

- `include_official_test=false`;
- `test_contract_count=0`;
- the release template is deliberately inert;
- no official-test path, bytes, row identities, labels, or outcomes have been
  used by the Phase 1--6 preparation; and
- an authorised release can be created only after this changeset is committed,
  pushed, clean, and explicitly approved once by the user.

## 2. Frozen scientific direction

The thesis is a supplied-candidate, example-filtered taxonomy-generalisation
study. Its primary experiment is Level 2 leave-one-aspect-out transfer under a
genuine two-stage factorisation: Stage 1 decides aspect presence and Stage 2
predicts sentiment conditional on that aspect. Candidate semantic interfaces,
stage-specific failures, operating points, and adaptation method are supporting
questions.

Level 1 is a closed-taxonomy development reference, Level 2 is the primary
taxonomy-shift setting, Level 3 is post-hoc appendix robustness only, and Level
4 is a separate three-group structured-shift stress test. Their F1 values are
not interpreted as one subtractable difficulty ladder.

## 3. Final candidate and hypotheses

The only promoted post-formal-freeze candidate is the **exploratory,
validation-selected fixed stage-wise composition**:

- Stage 1 score and threshold: Frozen Qwen few-shot;
- Stage 2 conditional sentiment scores and runner-up threshold: QLoRA;
- decoder: top-one plus thresholded runner-up, capped at two sentiments;
- new training, retuning, joint optimisation, calibration, or learned routing:
  none.

The Level-2 D confirmatory family is frozen hierarchically:

1. H1: fixed composition versus Frozen Qwen few-shot;
2. H2: fixed composition versus QLoRA, confirmatory only if H1 passes.

The primary endpoint is the unweighted mean of held-out pair micro-F1 over the
twelve fixed LOAO folds. The paired interval uses 20,000 synchronised review-
cluster bootstrap draws with seed 13. The pooled held-out pair F1 is sensitivity
only.

## 4. Validation evidence retained at freeze

The fixed composition's Level-2 D validation held-out pair F1 is `0.5259309432`.
Its nominal validation-selected paired differences are:

- versus Frozen Qwen few-shot: `+0.0210483193`, 95% interval
  `[0.0007745024, 0.0426563334]`;
- versus QLoRA: `+0.0460277131`, 95% interval
  `[0.0216113424, 0.0697525708]`.

These intervals are conditional on the selected candidate, one neural
realisation, the fixed twelve aspects, and the observed validation reviews.
They are not selection-adjusted confirmatory intervals. The final locked held-
out evaluation exists to test whether the ordering transfers to another review
partition.

The reverse composition, Global Router, relative calibrator, smooth fusion,
and E5 retrieval were inspected and not promoted. Rich-description R and DCWT
remain bounded negative findings. No further Router, MoE, calibrator, prompt,
retrieval, description-card, RL, or Level-3 search is authorised.

## 5. Final roster

- **Confirmatory L2-D:** fixed composition, Frozen Qwen few-shot, QLoRA.
- **Descriptive L2 N/D base roster:** TF--IDF, E5, DCWT, DistilBERT, Frozen
  Qwen zero-shot, Frozen Qwen few-shot, QLoRA; fixed composition is reported
  from its source grids.
- **Descriptive L4-D:** DistilBERT, Frozen Qwen few-shot, QLoRA, with all three
  parent groups reported separately.
- **Excluded from final test:** L1, L3, rich R, Router, calibrator, fusion,
  retrieval, reverse composition, and all legacy fixed-split/one-stage systems.

Every registered result is reported regardless of direction. A descriptive
winner cannot replace the validation-selected candidate after reveal.

## 6. Frozen reuse and canonicalisation

The final run reuses all train-only selected checkpoints/adapters, deterministic
few-shot demonstrations, validation-selected seen-only thresholds, prompt and
verbalizer contracts, model revisions, and the approved minimal-description
resource. Train-plus-validation refitting is forbidden.

For L2, D is scored over the complete 36-pair grid and N scores only the held-
out candidate. Every unchanged seen-candidate N score is copied from the same D
score object by construction. The final runner cannot reproduce the earlier
two-pass QLoRA invariant drift and cannot repair scores after outcomes.

## 7. Audited pre-release evidence

- machine preregistration SHA-256:
  `b622cae1ffec4e1af473dbf9f52cf5a7c4e79e15beb7e6c90b15d7663d8b11dc`;
- frozen artifact inventory: 159 entries, 459 files, 8,468,719,835 bytes,
  zero SHA-256 mismatches and zero test contracts;
- job graph: 93 base scoring jobs, 12 composition jobs, one seal, one analysis;
- validation mirror: 12 folds and 1,057 review clusters, with exact H1/H2 point
  and interval reproduction, zero test contracts;
- synthetic end-to-end dry-run: 105 immutable score bundles, 210 verified
  score/manifest backup receipts, one successful reveal, zero failures, zero
  test contracts;
- focused final-runner tests: 13 passed;
- whole-repository regression suite: 503 passed;
- thesis: XeLaTeX/BibTeX multi-pass build succeeded, 62 pages, no fatal LaTeX
  error; and
- the machine-readable pre-release audit binds the exact files and hashes used
  by this freeze.

## 8. Historical test-use disclosure

Earlier project phases accessed the released official test under superseded
fixed-split and one-stage protocols. The revised two-stage LOAO checkpoints,
thresholds, capped-two decoder, and fixed composition were selected without
revised-protocol official-test outcomes. A later run is therefore a locked
held-out evaluation of the revised protocol, not the project's first completely
blind or untouched test.

## 9. Failure and reveal policy

All 105 score bundles must be complete, finite, non-collapsed, immutable,
hashed, copied, and independently re-hashed before labels can be joined. No
interim fold outcome can be displayed. Metrics are revealed once.

Data/hash/schema mismatch aborts before metrics. There is no post-release batch
fallback. OOM, non-finite scores, prediction collapse, third-sentiment output,
or resume conflict is a protocol failure. A pure operational interruption may
be repeated only before reveal with byte-identical configuration and preserved
evidence. Low performance is a scientific outcome and never a rerun condition.

## 10. Next legal state transition

After this preparation commit is pushed and the repository is clean and equal
to its upstream, Phase 6 may report **ready to request one-time authorisation**.
That wording is deliberately narrower than permission to execute. Only a new,
explicit user instruction may create the release record, at which point the
record must bind the exact execution commit, official-test file and ordered row
identities, 1,587 rows, immutable backup identifier, timestamp, and no-fallback
policy.

## 11. Post-authorisation pre-scoring repair

After explicit user authorisation, an identity-only release record bound the
1,587 ordered official-test reviews to commit `3fd0263`. No score job, metric,
or outcome inspection occurred. Deployment review then found that the worker
materialisation loader would read `label_vault.csv` only to verify its hash.
Although labels were never model inputs, this was stricter-governance
non-compliance because a scoring Pod should not possess or open the vault.

That first release was marked superseded before scoring. The runner was
repaired so worker scoring, composition and sealing require only the strictly
unlabelled `row_uid,text` file plus its manifest and release record. Only the
single local analysis process can require and SHA-256 verify the private vault.
The repair changes no scientific method, model, threshold, prompt, fold,
metric, decoder or hypothesis. It must pass the full suite, be committed and
pushed, and receive a replacement commit-bound release before GPU start.
