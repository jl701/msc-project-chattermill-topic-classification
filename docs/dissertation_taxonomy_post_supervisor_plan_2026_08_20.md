# Post-supervisor dissertation experiment plan

Date: 20 August 2026; final-test preparation status updated 24 August 2026

Status: **formal validation and bounded extensions complete; final held-out evaluation frozen pre-release**

Current decisions are indexed by
`docs/taxonomy_current_authority_20260824.md`. The 22 August scientific-freeze
repair remains authoritative for the repaired formal-v2 validation evidence,
but it predates the validation-selected fixed stage-wise composition and cannot
authorise a final held-out run.

## 1. Outcome of the UCL supervisor meeting

The project remains a study of aspect-based sentiment classification under an
evolving taxonomy, using a genuine two-stage decision:

1. score the presence of every candidate aspect;
2. select every aspect above a seen-only threshold; and
3. assign one sentiment, or a second sentiment above a separately selected
   threshold, to every selected aspect.

The meeting changes the experimental emphasis:

- the clean single-unseen LOAO experiment becomes the primary place to test
  whether minimal and richer descriptions help;
- the former supplied-candidate Level 1 becomes a diagnostic view inside the
  single-unseen experiment rather than a separate training campaign;
- the ordinary all-aspects-seen Logistic Regression experiment becomes the new
  Level 1 closed-taxonomy reference;
- the full dual-unseen description crossover is no longer a core requirement;
- dual-unseen evaluation is retained only as a targeted, optional robustness
  study;
- parent-group holdout remains a secondary structured taxonomy-shift stress
  test; and
- the main Level 2 method roster must comprehensively include Frozen Qwen
  zero-shot, Frozen Qwen few-shot and task-specific QLoRA, alongside the
  classical and supervised controls.

The supervisor also proposed a high-risk but legitimate additional method:
predict the held-out aspect's logistic-regression parameters from the mapping
between eleven seen-aspect description embeddings and their learned classifier
weights. This is now pre-registered as a Level 2 method.

## 2. New level definitions

The memorable taxonomy-shift sequence is:

```text
L1: zero unseen aspects
L2: one unseen aspect
L3: two unseen aspects
L4: one whole parent group unseen
```

These labels describe taxonomy exposure. Raw F1 values are not automatically
comparable as a single monotonic curve because the tasks and held-out sets
differ.

### Level 1 — Closed-taxonomy supervised reference

All twelve aspects occur in training and evaluation. The required reference is
word-plus-character TF-IDF with One-vs-Rest Logistic Regression over the 36
aspect-sentiment pairs.

Level 1 answers whether ordinary in-distribution classification is learnable.
It is a development reference, not an unbiased estimate and not a numerical
measure of the generalisation loss in Level 2, because the output structure,
training filter and selection boundary differ. It receives a short methods
paragraph and a compact result table; it is not presented as a novel method.

### Level 2 — Single-unseen LOAO main experiment

Run twelve outer folds. Each fold removes one aspect from task-specific
training, trains on the other eleven, and evaluates all twelve candidate
aspects on every validation review.

The held-out candidate has three paired representations:

| ID | Held-out candidate information |
|---|---|
| `N` | canonical name only |
| `D` | canonical name plus the frozen minimal definition |
| `R` | globally selected positive-rich guidance, evaluated as a bounded local ablation |

Within every paired comparison, the same checkpoint or frozen model, rows,
candidate identities, deterministic few-shot examples and thresholds must be
used. Seen candidates always retain name plus minimal definition. `D - N` is
the cross-model primary contrast. `R - D` is a bounded TF-IDF/E5 ablation:
the pre-registered rich-interface gate did not pass, so `R` is not expanded to
Frozen-Qwen few-shot, DistilBERT or QLoRA.

Level 2 has two views, not two independently trained levels:

- **L2-S supplied/oracle diagnostic:** extract the held-out aspect-presence
  score, rank, average precision and F1, and evaluate sentiment with the gold
  aspect gate opened. This preserves the former Level 1 question.
- **L2-E end-to-end main result:** evaluate the full seen-plus-unseen predicted
  aspect-sentiment set after both stages and thresholds.

The complete Level 2 roster is:

1. strict train-only TF-IDF true two-stage baseline;
2. frozen E5-base-v2 true two-stage baseline;
3. description-to-classifier weight transfer;
4. genuine two-stage DistilBERT;
5. Frozen Qwen zero-shot;
6. Frozen Qwen few-shot; and
7. genuine two-stage QLoRA.

The old independent 36-pair architecture remains a matched control only.

### Level 3 — Targeted dual-unseen robustness

Level 3 no longer carries the primary description claim. If it is executed,
hold out six deterministic, disjoint pairs from the frozen canonical order:

```text
a01-a02, a03-a04, a05-a06,
a07-a08, a09-a10, a11-a12
```

This covers every aspect exactly once, includes three within-parent and three
cross-parent boundaries, and is selected without inspecting outcomes. The
primary condition is `DD`: both unseen aspects receive their names and minimal
definitions. The former twelve cyclic pairs and complete `NN/DN/ND/DD/RR`
crossover are not required for the revised thesis claim.

Level 3 remains optional until Level 2 is frozen. Compatible already-completed
validation rows may be reported after exact audit; missing trainable-model runs
are not automatically authorised. A separate opaque-unknown anomaly detector
still needs a precise candidate/output identity and sentiment contract and is
therefore deferred rather than silently treated as `NN`.

### Level 4 — Parent-group structured shift

Retain all three registered parent groups:

- `Company brand`: a02, a03, a04;
- `Staff support`: a08, a09, a10; and
- `Value`: a11, a12.

All held-out group candidates receive name plus minimal definition. Level 4
does not repeat the full Level 2 description sweep. It tests whether an entire
semantic sub-taxonomy can be absent from supervised training. Report every
group separately with review-cluster uncertainty and avoid strong inference
from only three group folds.

## 3. Additional Level 2 method: classifier-weight synthesis

The new method is **Description-to-Classifier Weight Transfer (DCWT)**. In each
outer Level 2 fold it learns eleven aspect-presence Logistic Regression models
in one shared train-only latent TF-IDF coordinate system, embeds their canonical
name-plus-description texts with frozen E5, and learns a regularised mapping
from description embeddings to classifier coefficient-plus-intercept vectors.
It then synthesises the missing twelfth classifier from the twelfth descriptor.

Because there are only eleven source tasks, unrestricted high-dimensional
regression and neural hypernetworks are prohibited. The registered primary
model is centred linear-kernel ridge in a low-dimensional SVD review space.
It must be compared with mean-weight, nearest-description and
cosine-barycentric weight transfer. Hyperparameters and threshold are selected
through leave-one-seen-aspect-out pseudo-unseen validation; the real held-out
aspect remains evaluation-only.

DCWT changes Stage 1 only. Stage 2 remains the exact shared
aspect-conditioned, capped-two sentiment component. Its detailed contract is
in `taxonomy_description_to_classifier_weight_transfer_v1.json` and the
corresponding experiment note.

## 4. Fixed methodological decisions

- Aspect selection is multilabel and uncapped; there is no forced top-k aspect
  fallback.
- Sentiment is aspect-conditioned. Emit top-1 and optionally a runner-up above
  the seen-only threshold; never emit all three sentiments.
- Thresholds are method- and outer-scope-specific and may differ between
  methods. They cannot use held-out aspect labels.
- The minimal descriptions and selected rich guidance remain frozen. They
  cannot be rewritten after inspecting outcomes; the negative `R` confirmation
  closes rather than restarts interface tuning.
- Frozen-Qwen few-shot demonstrations are deterministic, train-only,
  cross-aspect and shared across `N/D`; no retrieval or result-guided example
  choice is permitted.
- Official test remains sealed. Current work is train plus validation only.
- Any later official-test pass is single-use and requires a new test-blind
  freeze plus explicit user approval.

## 5. What is already complete

- The true two-stage architecture and capped-two sentiment policy are
  implemented and tested.
- Matched Level 2 validation shows the two-stage structure improves TF-IDF and
  E5 relative to independent 36-pair scoring.
- Strict TF-IDF and E5 now have complete twelve-fold selected-interface
  `D/R` confirmation in addition to their Level 2 `N/D` evidence. Frozen-Qwen
  zero-shot has exact `N/D` evidence; its older rich rendering is historical
  only because it is not the selected rich interface.
- The rich-interface study completed 72 train-only pseudo-folds and selected
  `R2_positive_concat` globally. Formal TF-IDF/E5 confirmation then showed
  lower presence AP, presence F1 and held-out pair F1 with more false
  positives, so expensive-model `R` expansion is closed.
- Previous TF-IDF/E5/Frozen-Qwen zero-shot Level 3 and Level 4 validation
  artifacts are complete under their recorded contracts.
- Genuine two-stage DistilBERT, Frozen-Qwen few-shot and QLoRA execution,
  memory and cloud recovery paths passed bounded gates.
- Immutable checkpoint, score-shard, cache, resume, telemetry and local backup
  mechanisms are implemented.
- The old formal three-GPU campaign did not start, so no formal long-run result
  is discarded by this redesign.
- DCWT is implemented, tested and complete across all twelve Level 2 folds;
  the learned kernel-ridge mapping did not beat its nearest-description-weight
  control and is retained as an informative negative result.
- The new Level 1 closed-taxonomy reference is complete.
- Formal cloud v2 was frozen at 15 scopes, 150 trainable work units and 81
  validation result payloads, and subsequently completed on 2026-08-21. The
  replicated campaign states, checkpoints, selections, score shards and result
  payloads passed the completion audit with zero failures and zero test
  contracts. Scientific-analysis freeze remains separate and is governed by
  the 2026-08-22 repair checklist.

## 6. Exact reuse policy

Reuse is exact or rejected. No result is reusable merely because it appears
close to the new task. The audit must match data and row hashes, fold/filter,
model revision, checkpoint, representation, prompt, few-shot examples,
threshold population/objective, decoder, score semantics and test ledger.

Likely audit candidates include matched Level 2 TF-IDF/E5 scores, compatible
Frozen-Qwen zero-shot cache rows, the six selected old Level 3 `DD` pairs and
all three old Level 4 `D` folds. Old independent-pair DistilBERT checkpoints and
QLoRA adapters remain controls only. Any changed `N/D` prompt or score contract
produces new work; previous non-selected `R` renderings cannot substitute for
the selected local ablation.

## 7. Revised execution order

### Phase 0 — Governance and inventory

- [x] Record the new Level 1–4 definitions.
- [x] Register DCWT as a Level 2 method.
- [x] Revoke the old 26-scope/297-result launch plan.
- [x] Build a machine-readable exact-reuse versus missing-work matrix.
- [x] Update the main thesis terminology only after preserving the user's
  current uncommitted thesis edits.

### Phase 1 — Cheap local work

- [x] Run or exact-audit the validation-only Level 1 TF-IDF + Logistic
  Regression reference.
- [x] Implement DCWT with synthetic recovery, leakage, finite-score and
  deterministic tests.
- [x] Run one real DCWT outer-fold smoke, then all twelve validation folds.
- [x] Complete or exact-reuse TF-IDF, E5 and Frozen-Qwen zero-shot Level 2
  `N/D` results.
- [x] Produce L2-S oracle diagnostics and L2-E end-to-end metrics from the same
  score contracts.
- [x] Run the bounded train-only rich-interface search, freeze one global `R`,
  complete the TF-IDF/E5 validation confirmation and close cloud expansion.
- [x] Recompute the reduced workload, runtime, storage and cost forecast.

### Phase 2 — Revised cloud preparation

- [x] Generate a new cloud plan for mandatory Level 2 and Level 4 scopes only.
- [x] Repartition the final jobs across available GPUs without shared mutable
  state.
- [x] Re-run final-commit startup, thermal, storage, resume and local-sync gates.
- [x] Verify that no unneeded Pod or volume is actively billing before launch.

The mandatory Level 2 plus Level 4 union has 15 unique training scopes rather
than the old 26. If the trainable scheduler remains unchanged, this is roughly
150 rather than 260 trainable work units. The rich gate removes `R` only from
formal scoring, not from the shared checkpoint-selection graph; therefore the
work-unit count remains 150 while the exact result payload count falls to 81.

The regenerated formal-v2 manifest now fixes the exact counts at 150 trainable
jobs and 81 result payloads. Applying the qualified Secure RTX 4090 rates to
the reduced graph gives a 25%-margin estimate of 24.60 GPU-hours, $18.45 at
$0.75/h, and about 9.57 hours ideal three-worker wall time. The machine-readable
plan, worker assignments, forecast and offline union audit are complete; the
fresh live-host gates remain intentionally pending.

### Phase 3 — Formal validation-only cloud work

- [x] Run Frozen-Qwen few-shot on the revised mandatory scopes.
- [x] Run genuine two-stage DistilBERT on the revised mandatory scopes.
- [x] Run genuine two-stage QLoRA with the registered learning-rate selection.
- [x] Continuously replicate immutable completed artifacts locally.
- [x] Stop on any OOM, non-finite value, collapse, third sentiment, hash/resume
  conflict, thermal slowdown, official-test evidence or replication failure.

### Phase 4 — Validation analysis and optional Level 3 decision

- [x] Audit the complete mandatory artifact union.
- [x] Report Level 2 paired `D-N` effects across the full formal roster and the
  completed bounded TF-IDF/E5 `R-D` ablation with identical review-level
  bootstrap draws and per-aspect deltas.
- [x] Report Level 4 groups individually with appropriately limited inference.
- [x] Retain the historical Level 3 evidence only as appendix-level post-hoc
  targeted robustness, report selected and alternate cyclic matchings, reject
  exact primary-result reuse, and add no expensive missing-model runs.

### Phase 5 — Thesis integration and final freeze

- [x] Update Methods, Results plan, limitations, diagrams and experiment tables
  to the new naming.
- [x] Explain Level 1 as a conventional reference, Level 2 as the contribution,
  Level 3 as optional robustness and Level 4 as structured shift.
- [x] Record negative DCWT or description results rather than hiding them.
- [x] Run tests, artifact/hash audits, secret scan and backup verification.
- [x] Issue a new test-blind scientific freeze.
- [x] Keep official test blocked until explicit approval for one final pass.

### Phase 6 — Post-freeze bounded component studies

- [x] Run the single registered global Router study; retain it as a negative
  validation-only result after its cross-fitted estimate failed the success
  rule.
- [x] Recombine the already-frozen formal score grids into the fixed Frozen
  Qwen few-shot Stage-1 plus QLoRA Stage-2 hybrid and its reverse component
  control.
- [x] Evaluate one global two-feature QLoRA relative calibrator with
  held-out-aspect-safe, review-cross-fitted fitting and D-only threshold
  selection.
- [x] Promote the fixed stage hybrid on validation evidence: L2-D held-out
  pair F1 `0.525931`, paired gain `+0.021048` over Frozen Qwen few-shot with
  95% interval `[+0.000775,+0.042656]`.
- [x] Retain the relative calibrator as an inconclusive ablation: held-out
  gain `+0.002481` over QLoRA with interval
  `[-0.004619,+0.009603]`.
- [x] Verify hashes, cross-fit separation, finite scores, capped-two decoding,
  zero failures and zero test contracts. Keep the official test sealed.
- [x] Evaluate one leakage-safe smooth fusion of Frozen-Qwen few-shot and
  QLoRA Stage-1 evidence. Retain it as a non-promoted ablation: D held-out
  pair F1 `0.513711`, gain `-0.012220` versus the fixed hybrid, 95% interval
  `[-0.032407,+0.008033]`.
- [x] Evaluate one preregistered E5-retrieved Frozen-Qwen few-shot Stage 1
  without model updates. Retain it as a non-promoted ablation: D held-out
  pair F1 `0.517509`, gain `-0.008422`, 95% interval
  `[-0.034933,+0.017751]`.
- [x] Complete all 24 retrieval condition units (304,416 scores), audit 961
  critical inputs and keep failures, non-finite scores, retrieval-contract
  violations, prediction collapses and test contracts at zero.

The validation-selected L2-D candidate for any later separately authorised
final evaluation is now the fixed Frozen Qwen few-shot Stage-1 plus QLoRA
Stage-2 hybrid. This decision does not itself authorise or open the official
test partition.

### Phase 7 — Final held-out pre-release preparation

- [x] Align the thesis research questions and Methods with the final
  taxonomy-generalisation, stage-decomposition, and fixed-composition claims.
- [x] Replace stale source-of-truth pointers and publish one current-authority
  index.
- [x] Freeze a machine-readable final-test preregistration, exact roster,
  hierarchical hypotheses, metrics, no-retraining rule, failure policy, and
  historical-test disclosure.
- [x] Inventory and SHA-256 bind every reusable checkpoint, selection,
  demonstration contract, validation-era score resource, and locked code or
  candidate resource needed by the final runner.
- [x] Implement a manifest-driven execution path that fails closed unless a
  separate one-time release manifest binds the final commit and official-test
  bytes.
- [x] Complete synthetic and validation-mirror dry-runs with
  `include_official_test=false` and `test_contract_count=0`.
- [x] Freeze immutable score, seal, backup, and one-time analysis boundaries;
  prohibit interim fold outcomes.
- [x] Run focused and regression tests, compile the thesis, scan for secrets and
  accidental test contracts, audit hashes, and push safe tracked changes.
- [x] Issue a final Phase 6 GO/NO-GO and exact execution/cost plan. A GO means
  ready to request one-time authorisation; it does not itself release the test.

## 8. Revoked launch boundary

`taxonomy_two_stage_three_gpu_parallel_v1.json` and its 26-scope launch
commands are historical readiness evidence only. They must not be run after
this plan. A new versioned manifest must be generated from the revised
mandatory Level 2 and Level 4 matrix and must pass the complete safety gate on
the final commit.

## 9. Global stop conditions

Stop rather than improvise if any run shows official-test access, target-guided
selection, missing/duplicate scopes, incompatible resume state, checkpoint or
score hash conflict, non-finite loss/probability, prediction collapse, a third
sentiment, OOM, repeated thermal/hardware slowdown, failed local replication or
an active unneeded cloud resource causing unbounded billing.
