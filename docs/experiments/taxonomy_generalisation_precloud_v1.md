# Taxonomy-Generalisation Pre-Cloud Implementation v1

## Status

This implementation plan was registered on 23 July 2026 after the user approved
the Levels 1-4 taxonomy-generalisation mainline, retained DistilBERT, deferred
Level 5 and router tuning, and required paired confidence intervals.

No new official validation or test result may be generated in this phase.
Development may use synthetic fixtures and bounded local model smoke tests.

The machine-readable authority is
`configs/experiments/taxonomy_generalisation_precloud_v1.json`. The dissertation
roadmap remains `docs/dissertation_loao_mainline_lock_2026_07_23.md`.

## Objective

Complete every safe task needed before renting cloud GPU capacity:

1. create and validate the minimal-description resource;
2. implement Levels 1-4 folds and exhaustive candidate grids;
3. implement strict seen-aspect calibration;
4. implement paired review-cluster confidence intervals;
5. expose common method manifests and score artifacts;
6. add leakage, cache, resume and shard guards;
7. complete synthetic and bounded local smoke tests; and
8. produce a cloud-readiness pack for user approval.

## Frozen boundaries

- The task remains aspect-conditioned pair-set prediction.
- Level 3 trains on ten aspects and evaluates all twelve candidates.
- The difficulty curve is `L1-D -> L2-D -> L3-DD -> L4-D`.
- DistilBERT remains the supervised contextual baseline.
- Router tuning, DeBERTa sweeps and Level 5 are outside this phase.
- Target-aspect validation labels cannot select any threshold, checkpoint,
  prompt, parser, description or policy.
- Test labels are not read during implementation or smoke testing.
- Parameter optimisation begins only after the permitted validation protocol is
  frozen and uses its registered budget and stopping rule.

## Evidence and stopping rule

This phase ends at the Cloud Readiness Gate. Completion requires:

- exact configuration and description hashes;
- all unit, integration, leakage and synthetic end-to-end tests passing;
- Qwen/QLoRA bounded local smoke evidence, including resume and shard merge;
- exact command templates and output locations;
- measured one-fold resource requirements where local hardware permits;
- a remaining cloud runtime/cost estimate; and
- a detailed report separating completed facts, unexecuted formal experiments,
  limitations and user decisions still required.

The description resource remains `pending_user_approval`. Formal
validation/test execution is blocked until the user approves its exact twelve
definitions and the final statistical contract.

## Implementation progress

### Stage 1 - resource and protocol core (complete)

Completed on 23 July 2026 without reading new official validation or test
results:

- created the definition-only `minimal_v2` resource and locked its content hash;
- preregistered all Level 1-4 fold schedules, pair sampling, strict threshold
  transfer, seeds, budgets, statistics and stopping boundary;
- implemented the one-candidate, all-candidate, cyclic dual-holdout and
  parent-group fold objects;
- implemented example-filtered training, full official-row evaluation,
  exhaustive aspect-sentiment grids and deterministic training pairs;
- implemented Level 3 `NN/DN/ND/DD` rendering with identical pair identities;
- implemented seen/unseen metric partitions and their harmonic mean;
- implemented paired review-cluster bootstrap, paired unit intervals, exact or
  Monte Carlo sign-flip tests, and Holm adjustment; and
- passed the complete repository test suite: 222 tests.

Reflection after this stage:

- candidate scope, gold-label scope and training-label scope are separate
  explicit fields, preventing Level 1 and Level 2 from being conflated;
- all seen labels use the minimal definition, while only held-out label
  representations change in registered description interventions;
- same-review multi-sentiment gold labels cannot be generated as negative
  training pairs;
- strict threshold calibration filters to seen aspects before candidate
  thresholds or metrics are computed; and
- formal data execution remains closed because description and statistical
  approval gates are intentionally still pending.

### Stage 2 - method and execution contracts (complete)

Completed on 23 July 2026 without reading new official validation or test
results:

- froze an ordered five-method registry with pinned E5, DistilBERT and Qwen
  revisions, starting recipes, finite seen-only tuning grids and stopping rules;
- exposed one probability-scoring runtime boundary for strict TF-IDF, E5,
  DistilBERT, Frozen Qwen and QLoRA;
- moved the reusable QLoRA training loop into the source package and added
  deterministic epoch callbacks;
- implemented validation-only preparation so tuning jobs need not load the
  official test split;
- implemented content-addressed run contracts, review-cluster sharding, atomic
  score writes, exact shard merge, cache validation and fail-closed resume;
- removed review and candidate text from score artifacts while retaining local
  identity and gold-alignment evidence;
- added a local formal-test ledger that allows exact-contract resume but rejects
  an incompatible second use of the same test endpoint;
- implemented per-aspect, per-sentiment and Level 3 crossover diagnostics; and
- passed the complete repository test suite: 248 tests.

Reflection after this stage:

- identical interfaces do not imply identical model internals: E5 and Frozen
  Qwen remain frozen, whereas TF-IDF, DistilBERT and QLoRA learn from the same
  registered pair manifest;
- strict TF-IDF may fit review and seen-candidate text from the training
  manifest, but its vocabulary and IDF cannot see held-out candidates,
  validation text or test text;
- Frozen Qwen and QLoRA are locked to the same base-model revision, prompt,
  candidate rendering, verbalizers and probability mapping;
- all conditions in a crossover must share training and evaluation identity
  hashes; and
- no method smoke has yet been counted as complete: real cached-model smoke,
  checkpoint recovery and the cloud runner remain the next stage.

### Stage 3 - cloud runner, inference and real-model smoke (complete)

Completed on 23 July 2026 without reading new official validation or test
results:

- implemented guarded `train -> score-validation -> select-threshold ->
  score-test -> analyse-test` phases;
- made training open only the official train split, validation stages open
  train plus validation, and test stages open train plus test;
- corrected strict Level 1 calibration so held-out validation targets are
  never scored and thresholds transfer from the matched eleven-seen-aspect
  L2-D grid;
- reused that exact calibration artifact for Level 2 threshold selection;
- made every trainable method select hyperparameters independently inside each
  of the 26 unique outer training scopes, using seen-candidate validation
  grids only and never pooling evidence across outer folds;
- added fixed-recipe selection artifacts for E5 and Frozen Qwen without a
  fictitious tuning sweep;
- reduced DistilBERT and QLoRA optimisation to finite three-learning-rate
  grids with historically grounded fixed epoch counts;
- added one-load/all-shards execution while preserving independent resumable
  shards;
- deduplicated identical rendered claims across Level 3 conditions, reducing
  three-review smoke inference from 432 logical calls to 126 unique calls;
- identified and reused the common L3 `(A11,A12)` / L4 `Value` training scope,
  leaving 26 unique core training scopes;
- integrated per-condition review-cluster confidence intervals, nonlinear
  harmonic-F1 resampling, paired matched comparisons, explicit
  different-task cross-level degradation and Holm-adjusted sign-flip families;
- added three-seed Level 1 QLoRA sensitivity aggregation without conflating
  seed and review-sampling uncertainty;
- added scientific-protocol hashes to training and score contracts while
  excluding administrative approval status;
- generated a 2,611-job dependency-aware cloud plan with 102 actual QLoRA
  training runs and exact score-volume estimates;
- completed bounded real-model synthetic smoke tests for all five methods,
  including checkpoint reload, two-shard merge and resume;
- completed an independent second-process QLoRA resume with zero retraining and
  zero rescoring; and
- passed the complete repository test suite: 285 tests.

Reflection after this stage:

- the strict Level 1 curve was previously under-specified because a
  target-only evaluation grid contains no seen candidates for calibration;
  the matched L2-D calibration branch resolves this without using target
  validation labels;
- a three-fold global pilot would still be non-strict because an aspect held
  out in one fold is seen in another; nested selection per outer training
  scope removes that indirect route;
- condition caching, cross-level calibration reuse, shared training scopes and
  frozen-method marker checkpoints reduce compute without changing any
  scientific comparison;
- matched model/description contrasts and cross-level difficulty changes now
  have different, explicit inference contracts;
- synthetic smoke metrics remain prohibited from thesis result tables; and
- formal execution remains blocked until the exact descriptions and
  statistical contract are user-approved and frozen.

The detailed readiness hand-off is
`docs/experiments/taxonomy_generalisation_precloud_readiness_20260723.md`.
