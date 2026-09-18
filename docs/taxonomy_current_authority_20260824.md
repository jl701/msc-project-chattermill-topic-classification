# Taxonomy project current authority

> **Navigation update, 16 September 2026:** this document remains the authority
> for the original locked evaluation. The [current evidence index](README.md)
> also links the [completed training-policy study](results/training_policy_completed/README.md).
> That later validation-only extension does not reopen the official result.

## Writing-direction addendum — 10 September 2026

For subsequent thesis drafting and revision, follow the user-approved
[narrative decision](thesis_narrative_decision_20260910.md): locate the
aspect-detection/sentiment bottleneck, evaluate complementary fixed stage
composition, then examine dependence on training-data construction. The
post-original-test, validation-only training-policy study is a separate
extension, not a revision of the original official-test selection or claims.
This addendum changes writing priorities, not frozen experimental results or
test permissions. The dated sections below retain the original mainline's
authority and chronology.

Created: 24 August 2026; last updated: 25 August 2026

Status: **final held-out evaluation complete; registered claims resolved;
post-test model development closed**

This document is the current source-of-truth index for the taxonomy-
generalisation dissertation mainline. It supersedes earlier roadmap, launch,
mainline-lock, and scientific-freeze documents only where their decisions
conflict with the boundaries below. Historical documents and results remain
part of the provenance record.

## 1. Scientific mainline

The dissertation studies a supplied-candidate, example-filtered taxonomy shift:
one or more aspects are absent from task-specific training, but the candidate
taxonomy is supplied at inference. The primary question is how a genuine
two-stage system transfers to unseen aspects and how failure divides between:

1. Stage 1 aspect-presence gating;
2. Stage 2 aspect-conditioned sentiment;
3. the supplied candidate semantic interface; and
4. model adaptation and operating-point selection.

The thesis is not an unrestricted aspect-discovery study and does not treat
Level 1, Level 2, and Level 4 as points on one subtractable difficulty curve.

## 2. Evidential roles

- **Level 1:** closed-taxonomy development reference; no final official-test
  run is required.
- **Level 2:** twelve leave-one-aspect-out folds; primary revised-protocol
  experiment. `D` is the primary minimal-description condition and `N` is an
  unchanged-transfer descriptive condition.
- **Level 3:** historical dual-unseen post-hoc appendix robustness only; no new
  trainable runs and no official-test run.
- **Level 4:** three non-exchangeable parent-group shifts; descriptive `D`-only
  stress test, reported group by group.
- **Rich `R`:** one bounded TF--IDF/E5 validation study; closed negative result,
  excluded from final held-out evaluation.

## 3. Admitted validation evidence

The seven registered Level 2 base systems are strict train-only TF--IDF, frozen
E5, description-to-classifier weight transfer, DistilBERT, Frozen Qwen
zero-shot, Frozen Qwen few-shot, and QLoRA. Formal-v2 trainable/few-shot
artifacts originate from execution commit
`aa84212976a652d62cfca31ed8bf0516a216c485`.

The matched one-stage versus two-stage TF--IDF and E5 comparisons support the
architecture contribution. Description effects are model-dependent rather than
universal. Stage decomposition supports complementary operating-point strengths:
Frozen-Qwen few-shot has the stronger thresholded held-out gate, while QLoRA
has the stronger oracle-gated conditional sentiment decoder.

## 4. Validation-selected final candidate

The only promoted post-freeze candidate is the **validation-selected fixed
stage-wise composition**:

- Stage 1 score and threshold: Frozen Qwen few-shot;
- Stage 2 conditional sentiment scores and runner-up threshold: QLoRA;
- decoder: frozen top-one plus thresholded runner-up, capped at two sentiments;
- retraining, retuning, joint optimisation, learned routing: none.

Its Level 2 `D` validation held-out pair F1 is `0.525931`. Its nominal paired
gain is `+0.021048` over Frozen Qwen few-shot and `+0.046028` over QLoRA. These
are exploratory, validation-selected estimates, not selection-adjusted
confirmatory effects.

The reverse composition, global Router, relative calibrator, smooth fusion,
and E5-retrieved few-shot variants were inspected and were not promoted. They
remain validation appendix evidence and are excluded from the final roster.

## 5. Final held-out evaluation authority and outcome

The machine-readable authority is
`configs/experiments/taxonomy_final_test_v1.json`. It freezes:

- confirmatory Level 2 `D` systems: fixed composition, Frozen Qwen few-shot,
  and QLoRA;
- hierarchical H1 (composition versus few-shot), then H2 (composition versus
  QLoRA only if H1 passes);
- primary endpoint: aspect-balanced mean held-out pair micro-F1 across the
  twelve fixed folds;
- 20,000 synchronized review-cluster bootstrap draws, seed 13;
- no train-plus-validation refit, no threshold/prompt/model selection on test,
  no interim outcome inspection, and no post-test candidate promotion;
- full descriptive Level 2 `N/D` base roster and descriptive Level 4 `D`
  roster; and
- explicit exclusions for Level 1, Level 3, rich `R`, Router, calibration,
  fusion, retrieval, reverse composition, and legacy protocols.

The pre-release configuration remains immutable with
`include_official_test=false` and `test_contract_count=0`. A separate
authorised release bound the scoring execution to commit
`6a0b42df370d40c3b93eff3867fb8909af9539b8`, the exact test bytes and 1,587
ordered review identities. Worker materialisation was strictly label-free.

All 93 base scores and 12 fixed compositions completed. The 105-bundle graph
was sealed and 210 score/manifest files were independently copied before the
local private label vault was opened exactly once. The registered Level-2 `D`
primary result is:

- fixed stage-wise composition: `0.507538`;
- Frozen Qwen few-shot: `0.495398`;
- QLoRA: `0.486437`.

H1 composition-minus-few-shot is `+0.012140`, synchronized 95% interval
`[+0.001407,+0.022439]`, so H1 passed. The H1 gate opened H2;
composition-minus-QLoRA is `+0.021101`, interval
`[+0.000184,+0.041118]`, so H2 also passed. `failure_count=0`, the outcomes
were revealed once, and post-test tuning remains forbidden. The authoritative
completion record is
`docs/experiments/taxonomy_final_test_v1_results_20260825.md`.

## 6. Historical test-use disclosure

Earlier phases accessed the released official test under superseded fixed-split
and one-stage protocols. The revised two-stage LOAO models, thresholds,
capped-two decoder, and fixed composition were selected without
revised-protocol test outcomes. Any later run is therefore described as a
locked held-out evaluation of the revised protocol, not as the project's first
completely blind test.

## 7. Closed post-test work boundary

The user explicitly authorised the locked held-out evaluation and the later
GPU start. Operational failures under superseded commits were archived before
outcomes were revealed; none was mixed into the final graph. The successful
release used the repaired label-free contract and the single execution commit
above. All three scoring Pods were stopped after exact local receipt and hash
verification; their current GPU billing rate is zero.

The live work is now limited to faithful reporting, thesis integration,
predeclared error analysis and reproducibility packaging. No new model,
prompt, description, retriever, Router, calibrator, threshold, metric or
training seed may be selected in response to the final result. A low fold or a
descriptive method winning a secondary quantity remains a scientific outcome,
not a reason to rerun or promote another candidate.

## 8. Superseded or supporting records

- `docs/dissertation_loao_mainline_lock_2026_07_23.md`: historical mainline
  decisions, superseded where inconsistent with this document.
- `docs/dissertation_taxonomy_post_supervisor_plan_2026_08_20.md`: completed
  supervisor-redesign and validation-development record.
- `docs/experiments/taxonomy_post_supervisor_scientific_freeze_repair_checklist_20260822.md`:
  valid for the repaired formal-v2 validation evidence, but predates the fixed
  composition and cannot authorise its final evaluation.
- `docs/experiments/taxonomy_two_stage_stage_hybrid_calibration_v1_results.md`:
  fixed-composition and calibration results.
- `docs/experiments/taxonomy_two_stage_no_retraining_extensions_v1_results.md`:
  closed fusion and retrieval extensions.
- the author-only chronological experiment ledger (not distributed publicly).
- `docs/experiment_reproducibility_register.md`: parameter and artifact index.
