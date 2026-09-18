# Taxonomy Two-Stage Final Test v1 — Frozen Pre-Release Preregistration

**Status:** frozen pre-release; official test locked

**Protocol:** `taxonomy_two_stage_final_test_v1`

**Machine-readable authority:** `configs/experiments/taxonomy_final_test_v1.json`

**Formal validation artifact origin:** `aa84212976a652d62cfca31ed8bf0516a216c485`

**Test contracts at freeze:** `0`

This document freezes the scientific question, comparison family, reuse
boundary, analysis policy, and operational release gate before any revised-
protocol official-test access. It does not authorise execution and deliberately
contains no official-test path, bytes, hash, labels, or outcomes.

## 1. Scientific estimand

The final evaluation asks whether a supplied-candidate, truly two-stage ABSA
system generalises to aspects that were strictly absent from task-specific
training under the project's example-filtered taxonomy-shift protocol.

For each L2 fold, every training review containing the held-out aspect is
removed. The resulting shift therefore combines label novelty, fold-dependent
training-set reduction, removal of co-occurring seen-aspect evidence, and a new
review partition. It is not claimed to isolate a pure label-space causal effect,
and it is not open-world aspect generation because the candidate taxonomy is
always supplied.

The primary scope is the held-out aspect's three aspect–sentiment pairs over the
complete 12-aspect × 3-sentiment candidate grid. L1, L2, L3, and L4 are distinct
scenarios, not a directly subtractable difficulty ladder.

## 2. Frozen experimental roster

### 2.1 Confirmatory L2-D systems

Only the following three systems enter the confirmatory family:

1. **Validation-selected fixed stage-wise composition** — Frozen Qwen few-shot
   Stage 1 aspect-presence scores and thresholds, followed by QLoRA Stage 2
   conditional sentiment scores and its frozen runner-up threshold.
2. **Frozen Qwen few-shot** — strongest validation source system.
3. **Qwen candidate-pair QLoRA** — the other source system.

The composition is fixed, not learned. It performs no retraining, retuning,
joint optimisation, routing, mixture-of-experts gating, or test-time selection.
It was proposed after examining validation-only stage diagnostics and is
therefore explicitly an **exploratory validation-selected candidate** until the
locked held-out evaluation.

### 2.2 Descriptive L2 systems

For both D and N conditions, the full descriptive base roster is:

- strict train-only TF-IDF;
- Frozen E5 base v2;
- description-to-classifier weight transfer (the frozen kernel-ridge primary);
- DistilBERT review–candidate cross-encoder;
- Frozen Qwen zero-shot;
- Frozen Qwen few-shot;
- Qwen candidate-pair QLoRA;
- the fixed stage-wise composition derived from the two frozen source grids.

D is primary. N is descriptive and cannot replace D after outcomes are known.
Seen-candidate prompts shared by N and D must use exactly one canonical score
object by construction; they must not be scored independently and repaired
afterwards.

### 2.3 Descriptive L4 systems

The three registered parent-group shifts are evaluated in D only for:

- DistilBERT review–candidate cross-encoder;
- Frozen Qwen few-shot;
- Qwen candidate-pair QLoRA.

Groups are reported separately. No population-level inference over parent
groups is authorised.

### 2.4 Exclusions

No official-test execution is authorised for L1, L3, rich-description R,
Global Router, relative calibrator, smooth fusion, retrieval few-shot, reverse
composition, legacy fixed-split systems, legacy one-stage systems, or any new
post-freeze candidate. A descriptive system cannot be promoted after test
results are revealed.

## 3. Frozen reuse boundary

The final run reuses the exact train-only model realisations selected during
formal validation:

- all 15 DistilBERT selected checkpoints;
- all 15 QLoRA selected adapters;
- all 15 deterministic Frozen Qwen few-shot demonstration selections;
- all registered per-fold Stage 1 and capped-two runner-up thresholds;
- the approved `fabsa_aspect_descriptions_minimal_v2` resource;
- the frozen prompt/verbalizer contracts and model revisions.

Training plus validation refitting is forbidden. Doing so would create a new
model realisation, demonstration pool, and threshold relationship rather than
test the validation-selected system. Test labels may never select checkpoints,
demonstrations, thresholds, prompts, batch sizes, models, metrics, or folds.

The public, review-text-free inventory and its hashes are recorded in
`docs/experiments/taxonomy_final_test_v1_artifact_inventory.json`.

## 4. Decoder

For each Stage 1-selected aspect, Stage 2 always emits the highest-scoring
sentiment. It emits the runner-up only when that score exceeds the frozen
fold-specific threshold. No third sentiment is permitted. No top-k aspect
constraint is applied; aspect presence is thresholded independently using the
frozen Stage 1 threshold.

## 5. Endpoints and hypotheses

### 5.1 Primary endpoint

`L2_D_aspect_balanced_mean_heldout_pair_micro_f1`: compute held-out pair
micro-F1 within each of the 12 fixed LOAO folds, then take the unweighted mean
over folds.

### 5.2 Sensitivity endpoint

`L2_D_pooled_heldout_pair_micro_f1`: pool held-out predictions and labels over
all 12 folds before computing pair micro-F1. It cannot replace the primary
endpoint.

### 5.3 Hierarchical confirmatory family

- **H1:** fixed composition has higher primary-endpoint performance than Frozen
  Qwen few-shot.
- **H2:** fixed composition has higher primary-endpoint performance than QLoRA,
  and is confirmatory only if H1 passes.

For each comparison, use a synchronised paired review-cluster bootstrap with
20,000 draws and seed 13. The same sampled review identities are used for both
systems and synchronised across folds. Superiority requires the paired
difference's percentile 95% interval lower bound to be greater than zero. If H1
does not pass, H2 is descriptive only. No validation-development-wide
post-hoc multiplicity correction is claimed.

### 5.4 Secondary and diagnostic reporting

Secondary outcomes include held-out precision/recall, overall and seen pair
micro-F1, pair macro-F1, exact-set match, prediction-set size, aspect call rate,
and all per-fold results/supports. Presence AP/F1, oracle-gated Stage 2 F1,
Stage 1 FP/FN, collapse checks, and L4 group results are diagnostic only.
Oracle-gated metrics are never presented as end-to-end performance.

Bootstrap intervals condition on the fixed trained realisations and fixed 12
aspects; they do not cover neural optimisation variance or a random taxonomy
population.

## 6. One-time execution and reveal protocol

The preregistration alone cannot execute the official test. A later one-time
release record must bind:

- this preregistration's SHA-256;
- the clean, remotely synced execution commit;
- the official test file and ordered row-UID SHA-256 values and 1,587 rows;
- the canonical relative filename;
- an immutable backup-root identifier;
- no post-release batch fallback;
- a unique authorisation ID and timestamp;
- a new explicit user authorisation.

Before labels can be used for metrics, all 105 score/composition bundles must be
complete, finite, non-collapsed, immutable, hashed, copied to the registered
backup, re-hashed there, and sealed as one score graph. Workers receive only
unlabelled review identities/text and cannot access the separate label vault.
All predeclared results are then calculated and revealed once. Fold-level or
aggregate outcomes must not be inspected while scoring is in progress.

## 7. Failure policy

- Data, schema, row-identity, artifact, prompt, checkpoint, or hash mismatch:
  abort before metrics and preserve evidence.
- Pure operational interruption: an exact rerun is allowed only if no outcome
  was revealed, the failed attempt remains logged, and every byte of scientific
  configuration is unchanged.
- OOM, non-finite score, prediction collapse, third-sentiment output, or resume
  conflict: protocol/integrity failure; no performance-guided change.
- Batch-size change after release: forbidden; no fallback is registered.
- Low performance, a confidence interval crossing zero, or failed hypothesis:
  scientific outcome; never a rerun condition.
- A descriptive-method failure may leave primary reporting valid only when the
  shared infrastructure and all confirmatory score grids are demonstrably
  unaffected; the descriptive failure must still be reported.

## 8. Historical test-use disclosure

Earlier project phases accessed the released official test under superseded
fixed-split and one-stage protocols. The revised two-stage LOAO models,
thresholds, capped-two decoder, and fixed composition were selected without
revised-protocol official-test outcomes. The final run is therefore described
as a **locked held-out evaluation of the revised protocol**, not as the
project's first completely blind or untouched test.

## 9. Permitted conclusions

The final run may assess whether the validation-selected composition and the
main L2 ordering transfer to a new review partition under this exact protocol.
Even a positive result cannot establish universal superiority, general benefit
from descriptions, open-world discovery, training-seed stability, or broad
hierarchical-taxonomy generalisation. A negative result leaves the thesis's
validation-supported architecture, diagnostic, and bounded-negative findings
intact and must not trigger further model search.

## 10. Current lock

At this freeze, `include_official_test=false`, `test_contract_count=0`, and the
release template is deliberately inert. The next legitimate state transition
is not automatic: it requires a complete preflight PASS followed by a new,
explicit, one-time authorisation from the user.
