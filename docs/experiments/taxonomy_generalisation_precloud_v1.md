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
