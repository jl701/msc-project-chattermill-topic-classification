# Dissertation Taxonomy-Generalisation Mainline Lock - 23 July 2026

## Status and authority

This is the sole operational source of truth for the dissertation's experiment
mainline, active results, remaining work, and stopping decisions.

The user approved this revised direction on 23 July 2026 after discussing the
UCL supervisor's proposed train/test designs. It supersedes every older
roadmap, modelling pivot, launch queue, and experiment-priority note. Historical
experiment reports remain valid within their recorded scopes, but they do not
define what should be run next.

No new experiment was run to create this lock.

The pre-cloud implementation subsequently completed on 23 July 2026. Its
authoritative readiness evidence is
`docs/experiments/taxonomy_generalisation_precloud_readiness_20260723.md`.
This does not change the experiment order or authorise official
validation/test execution.

On 24 July 2026 the user approved the compute split: TF-IDF, E5, DistilBERT and
Frozen Qwen run locally, while every QLoRA model stage runs in the cloud.
The implementation and current workload evidence are in
`docs/experiments/taxonomy_local_non_qlora_execution_readiness_20260724.md`.
This later report supersedes the 23 July report only for execution placement,
Frozen-Qwen caching and workload estimates. Scientific descriptions and
statistical gates were initially pending separately. The user approved and
froze the complete statistical protocol on 24 July 2026.

Before formal local execution, the thesis route and literature were realigned
on 24 July 2026. The evidence map and terminology audit are recorded in
`docs/experiments/taxonomy_literature_route_alignment_20260724.md`. This review
confirmed the Level 1--4 route and five-method roster, corrected Level 1 versus
label-partially-unseen terminology, and repositioned earlier Gemini/router work
as historical or secondary evidence. No official validation or test experiment
was run during the literature stage.

Later on 24 July 2026 the user approved the exact minimal definitions and one
secondary Level 3 `DD` versus `RR` rich-taxonomy-guidance comparison. The
pre-registration and revised protocol hashes are recorded in
`docs/experiments/taxonomy_level3_rich_guidance_preregistration_20260724.md`.
This secondary comparison does not alter the primary `NN/DN/ND/DD` crossover
or the `L1-D -> L2-D -> L3-DD -> L4-D` difficulty curve.

On 25 July 2026 all four local methods completed their guarded official
train/validation work with the official test still sealed. Strict TF-IDF,
E5-base-v2, DistilBERT and Frozen Qwen completed `1,262/1,262`, `93/93`,
`326/326` and `93/93` registered jobs respectively, all with zero failures.
Their authoritative completion records are:

- `docs/experiments/taxonomy_tfidf_validation_completion_20260725.md`;
- `docs/experiments/taxonomy_e5_validation_completion_20260724.md`;
- `docs/experiments/taxonomy_distilbert_validation_completion_20260725.md`;
  and
- `docs/experiments/taxonomy_frozen_qwen_validation_completion_20260725.md`.

The next incomplete gate is the first measured cloud QLoRA tuning/runtime
scope. No formal test pass or cross-level result claim is authorised before
QLoRA selection and every method/threshold rule are frozen.

## Working title and central question

Working title:

> From Supplied Candidates to Evolving Taxonomies: Generalising Fine-grained
> Aspect-Based Sentiment Models to Unseen Aspects

Central question:

> As aspect-sentiment classification moves from one supplied unseen aspect to
> mixed seen/unseen candidate sets, multiple unseen aspects, and unseen
> parent-group shifts, how quickly do different model families degrade, and
> when do human-authored label descriptions and QLoRA reduce that degradation?

The contribution is therefore not another flat model leaderboard. It is a
controlled difficulty ladder for taxonomy generalisation, together with matched
tests of label-side semantic information and task-specific adaptation.

The narrative remains valid whether an intervention improves or harms F1.
Negative results must narrow the claim rather than be hidden or replaced by a
more favourable post-hoc protocol.

## Fixed task contract

Every active method must:

1. receive a supplied candidate aspect or candidate set rather than discover
   unrestricted topics;
2. predict sentiment conditional on each candidate aspect;
3. permit different sentiments for different aspects in the same review;
4. retain the pair-set output so that multiple aspect-sentiment pairs and empty
   predictions are representable;
5. evaluate every official row in the declared validation/test split;
6. use corrected positive-class presence F1 that excludes true negatives;
7. keep all vocabularies, IDF weights, encoders, adapters, thresholds, prompts,
   parsers, and policies inside the declared information regime; and
8. use test labels once, only after the complete configuration is frozen.

The common prediction unit for new stress-test work is:

```text
(review, candidate aspect representation, candidate sentiment)
                              |
                              v
                    applicable / not applicable
```

Predictions for all permitted candidates are unioned into the review-level
aspect-sentiment pair set.

## Label-description governance

There is no official FABSA annotation guidance. The project will use a
human-authored, label-side semantic resource under the following rules.

### Representation levels

- **Name-only:** the canonical hierarchical aspect name is always visible.
- **Name + minimal definition:** the canonical name plus one short neutral
  definition is visible.
- **Rich guidance:** definition, lexical cues, or decision boundaries. This is
  frozen secondary Level 3 evidence, not the core description treatment.
- **Opaque ID only:** an identifier such as `A12` without name or definition.
  This is permitted only as a negative-control diagnostic.

"No description" never means hiding the canonical aspect name in a core
comparison.

### Leakage-prevention rules

The minimal descriptions:

1. may use only the twelve canonical names, their parent-child hierarchy, and
   general language knowledge;
2. must not use review text, corpus keywords, label frequencies, validation/test
   labels, predictions, confusion matrices, or error analyses;
3. must not contain example reviews copied or paraphrased from FABSA;
4. must be reviewed and frozen before any new validation/test result is
   inspected;
5. must be versioned with provenance, an exact content hash, and an immutable
   canonical order;
6. must not be rewritten after results are known; and
7. must be presented in the dissertation appendix so the supplied supervision is
   auditable.

The exact `configs/experiments/fabsa_aspect_descriptions_minimal_v2.json`
resource was approved and frozen on 24 July 2026 without changing its twelve
definitions. Cues and boundaries must not silently enter the minimal treatment.
The separate
`configs/experiments/fabsa_aspect_rich_taxonomy_guidance_v1.json` resource uses
one uniform template: the exact minimal definition, three to five
non-corpus-derived aliases, one inclusion boundary, and one contrastive
boundary. It is permitted only in the registered Level 3 `RR` secondary
condition.

The validation rule was pre-registered on 24 July 2026 before formal local
execution. Description text may receive a blind semantic-format check for
neutrality, label-name consistency, and prohibited information, or a
pseudo-unseen check using seen-aspect labels only. Target-aspect validation or
test performance must never select, rank, or rewrite a definition. The
usefulness of a frozen definition is evaluated only afterwards through the
matched name-only versus minimal-definition contrast; a weak result is an
experimental result, not permission to edit the resource.

Descriptions are legitimate label-side semantic supervision, not target
labelled examples. Thesis wording must call the corresponding condition
**description-assisted zero-shot generalisation**, not zero-information
classification.

For the new stress-test suite, every seen aspect is represented by its canonical
name plus the same frozen minimal definition whenever a method consumes a label
representation during task-specific training. Frozen similarity methods that do
not learn from labelled pairs still use the same label representation at
scoring time. Description availability is manipulated only for held-out aspects
at inference. This prevents a hidden train/test representation mismatch from
being confused with the intended unseen-label intervention.

## Approved difficulty ladder

### Level 1 - Supplied-candidate single-unseen LOAO

For each of the twelve folds:

- train without one target aspect using `example_filtered`;
- test only whether that supplied target aspect is present;
- predict its candidate-specific sentiment when present;
- retain all official validation/test rows and allow an empty output.

This is the completed entry benchmark. The current headline table is
target-calibrated; a strict zero-label calibration view remains required for
the cross-level difficulty curve. The curve uses the description-complete
`L1-D` endpoint: the supplied held-out candidate is shown by canonical name plus
the frozen minimal definition. A matched name-only condition remains an
intervention control rather than the curve endpoint.

### Level 2 - Generalized single-unseen LOAO

For each fold:

- train on eleven aspects and hold out the twelfth;
- test all twelve candidate aspects jointly on every review;
- score seen and unseen aspect-sentiment pairs separately and together.

This tests whether the unseen aspect is suppressed by familiar labels and
whether adding the unseen label destabilises seen-label predictions.
The curve uses `L2-D`: every seen label has its frozen minimal definition during
training, and the held-out label is also supplied with its minimal definition
at inference.

### Level 3 - Dual-unseen asymmetric-description generalisation

This supervisor-proposed experiment is a core dissertation contribution, not
an appendix diagnostic.

For each registered dual-holdout fold:

- train on ten aspects using name + minimal definition;
- hold out two aspects from all task-specific training evidence;
- test all twelve candidate aspects jointly on every official evaluation row;
- treat the ten training aspects as seen and the held-out pair as unseen; and
- allow zero, one, or multiple aspect-sentiment pairs, including neither,
  either, or both unseen aspects.

Run the same trained model under five test representations:

| Condition | Unseen aspect A | Unseen aspect B |
| --- | --- | --- |
| `NN` | name only | name only |
| `DN` | name + definition | name only |
| `ND` | name only | name + definition |
| `DD` | name + definition | name + definition |
| `RR` | rich taxonomy guidance | rich taxonomy guidance |

`DN` and `ND` are a mandatory crossover. A one-direction comparison is
confounded by the two aspects' intrinsic difficulty.

The ten seen candidates always retain their frozen minimal definitions in all
five conditions. Only the two unseen candidates change representation. The
cross-level difficulty curve uses the complete-description `L3-DD` endpoint;
`NN`, `DN`, and `ND` isolate the core description intervention. `RR` is
secondary evidence about whether richer fixed taxonomy guidance improves on
`DD`; it is never substituted into the difficulty curve.

The initial fold schedule is the twelve cyclic pairs in frozen canonical order:

```text
(A1, A2), (A2, A3), ..., (A11, A12), (A12, A1)
```

This schedule is deterministic, gives every aspect two dual-unseen
appearances, includes both related and cross-parent pairs, and avoids selecting
pairs after seeing results. It must be confirmed during pre-registration before
execution; any replacement schedule requires a written rationale based only on
taxonomy structure and compute, not results.

### Level 4 - Multi-aspect parent-group holdout

Hold out all children of each multi-child parent:

- `Company brand` (three aspects);
- `Staff support` (three aspects); and
- `Value` (two aspects).

Test all twelve candidates jointly. This measures generalisation when several
semantically related labels and an entire supervised sub-taxonomy are unseen.
The curve uses `L4-D`: all seen labels have definitions during training and all
held-out group labels receive their frozen minimal definitions at inference.

### Deferred Level 5 - Compound organisation and taxonomy shift

This experiment is not part of the current dissertation execution scope. It is
retained only as a clearly labelled future extension:

- train on permitted organisations and seen aspects;
- select configurations on the permitted validation organisation without
  target-label calibration;
- test on unseen organisations with unseen aspects.

No Level 5 code sweep or GPU run should be launched unless the user explicitly
reopens the scope after Levels 1-4 are complete.

## Core description intervention

The main description study is the Level 3 `NN/DN/ND/DD` crossover. It directly
tests incomplete label documentation when two unseen labels compete. The
separate `RR` condition is a secondary rich-guidance extension.

All five conditions must share:

- identical train/validation/test rows;
- identical positive and negative candidate-pair identities;
- identical training budget and seeds;
- identical sentiment representation;
- identical hyperparameters and checkpoint-selection rule;
- identical strict calibration rule; and
- identical metrics and aggregation.

Only the two unseen aspects' test-time definition availability may change
between `NN`, `DN`, `ND`, `DD`, and `RR`. Negative examples must be sampled
once and reused; representation availability must not alter the training-pair
set. `RR` must reuse the exact `DD` model and threshold and must be reported as
rich taxonomy guidance, not as another minimal-description condition.

The analysis must include:

- F1 for described and undescribed unseen aspects;
- within-aspect described-minus-name-only deltas;
- both-present recall;
- performance when only the undescribed aspect is gold;
- false positives when neither is gold;
- whether predictions are biased toward the described candidate; and
- per-aspect conditional sentiment diagnostics.

The secondary `RR - DD` family must report paired review-cluster intervals,
per-method effect sizes, and Holm-adjusted sign-flip evidence within the five
methods. It cannot be used to hide or replace the core `DD` result.

## Method roster

The core methods for new protocols are:

| Method | Thesis role |
| --- | --- |
| Strict train-only TF-IDF | Classical lexical baseline |
| E5-base-v2 | Frozen semantic representation baseline |
| DistilBERT review-candidate cross-encoder | Supervised contextual baseline |
| Frozen candidate-pair Qwen | Matched no-adapter LLM control |
| Candidate-pair QLoRA | Main task-adaptation method |

Count BoW remains a Level 1 sanity control. MiniLM remains repository/appendix
evidence. The failed strict TF-IDF-to-Qwen router, Gemini branches, legacy
global-sentiment methods, legacy TF-IDF, shallow sentiment, and unrelated model
sweeps do not enter the new stress-test mainline.

Methods share the outer protocol and prediction interface. Their internal
architectures need not be identical. Intervention claims are permitted only
where the compared conditions are matched within a method or explicitly share a
registered component block.

## Information regimes and measurement

### Strict core regime

The difficulty curve and new stress tests use strict zero-label calibration:

- no target-aspect validation label may select a threshold, prompt, parser,
  checkpoint, policy, description, or hyperparameter;
- validation grids are constructed from seen-aspect candidates only, so
  held-out candidate claims are never rendered or scored on validation;
- every trainable method selects hyperparameters independently within each
  unique outer training scope; validation evidence is never pooled across
  outer folds, because a target aspect in one fold is seen in another;
- target thresholds must be transferred from seen-aspect evidence according to
  a pre-registered rule; and
- the target test set is evaluated once.

Existing target-calibrated Level 1 results remain a separately labelled,
less-restrictive reference.

### Metrics

Level 1 retains the unweighted mean of fold-level pair micro-F1 as its primary
metric.

Levels 2-4 must report:

- overall pair micro-F1 and pair macro-F1;
- seen-aspect and unseen-aspect pair F1 separately;
- a pre-registered harmonic mean of seen and unseen performance where both
  partitions exist;
- corrected presence precision, recall, and F1;
- exact-match rate;
- false-positive and false-negative rows per 100;
- per-aspect and per-sentiment results;
- conditional sentiment accuracy or macro-F1 with denominator and coverage;
- prediction-set size and described-candidate selection bias for Level 3; and
- runtime, GPU memory, latency, schema validity, and cost where applicable.

Raw F1 values from different levels are not the same benchmark. The main
cross-level analysis is each method's degradation from one level to the next and
the amount recovered by description or adaptation.

### Statistical uncertainty and paired inference

Confidence intervals are a required part of the final thesis evidence, not an
optional post-hoc decoration. The exact implementation was approved and frozen
by the user on 24 July 2026 before any new official validation/test result was
inspected. The contract is:

- report point estimates and 95% confidence intervals for the primary metrics;
- compare matched systems with a paired interval for the metric difference,
  rather than inferring improvement from two separate model intervals;
- use `row_uid` as the review-level bootstrap cluster, retaining every
  candidate aspect, candidate sentiment, gold label, and paired model
  prediction for the sampled review;
- recompute F1 and every nonlinear metric inside each bootstrap replicate;
- use identical bootstrap draws for both sides of a matched comparison;
- supplement review-level intervals with paired per-aspect deltas,
  wins/ties/losses, and an exact sign-flip test where the registered number of
  units permits it;
- keep review-sampling, aspect/fold, and training-seed uncertainty explicitly
  separate; and
- describe an interval excluding zero as evidence of a stable paired
  difference under the registered resampling regime, never as absolute proof.

The primary proposed paired contrasts are:

1. candidate-pair QLoRA minus Frozen candidate-pair Qwen;
2. minimal-description minus name-only within a matched model, especially the
   Level 3 `NN/DN/ND/DD` crossover; and
3. registered within-method degradation across the description-complete
   difficulty endpoints.

Level 4 has only three parent-group folds. It must therefore report
review-cluster intervals within every group and show all group results; it must
not claim strong cross-group inference from three fold values. Levels 2-4 use
seed 13 initially, so their intervals do not include training-seed variability.
The final matched Level 1 QLoRA analysis may additionally use seeds 13, 23, and
42 to report seed sensitivity.

## Completed Level 1 evidence

Only these rows are active in the existing target-calibrated, twelve-fold,
all-row Level 1 table:

| Method | Test mean pair micro-F1 | Role |
| --- | ---: | --- |
| Count BoW | 0.314412 | Sanity control |
| Strict train-only character TF-IDF | 0.385640 | Main classical baseline |
| MiniLM-L6-v2 | 0.388883 | Appendix/repository |
| E5-base-v2 | 0.401301 | Main frozen embedding baseline |
| DistilBERT review-candidate cross-encoder | 0.315848 | Contextual baseline |
| Frozen candidate-pair Qwen | 0.337804 | Matched QLoRA control |
| Candidate-pair QLoRA | 0.483158 | Current strongest Level 1 method |

Every row uses aspect-conditioned sentiment. The QLoRA result uses a bundled
enhanced package: description, hard negatives, balancing, candidate-pair
training, adapter training, and target validation threshold selection. It does
not isolate a minimal-description effect and is not yet strict zero-label
evidence.

The active generated table remains `docs/thesis_result_tables.md`. Retired
global-sentiment and legacy-router values remain prohibited.

## Approved execution checklist

A box may be checked only when the code/configuration, result, documentation,
relevant tests, and safe GitHub sync for that item are complete. Partial pilots
must be marked as pilots, not completion.

### A. Governance and cleanup

- [x] Approve the taxonomy-generalisation difficulty ladder as the thesis
  mainline.
- [x] Make this file the sole operational roadmap and TODO list.
- [x] Retain aspect-conditioned sentiment and pair-set prediction as fixed task
  requirements.
- [x] Elevate the dual-unseen asymmetric-description crossover to a core
  experiment.
- [x] Mark earlier roadmap, hybrid-pivot, and QLoRA-launch documents as
  superseded for future-work decisions.
- [x] Preserve completed experiment reports as historical provenance rather
  than deleting them.
- [x] Fix Level 3 as ten-aspect training followed by joint all-twelve-candidate
  evaluation, with two candidates marked unseen.
- [x] Give every seen label the frozen minimal definition in the new stress-test
  suite.
- [x] Fix the main difficulty curve as `L1-D -> L2-D -> L3-DD -> L4-D`.
- [x] Limit seeds 23 and 42 to the matched Level 1 QLoRA endpoints; initially
  use seed 13 for Levels 2-4.
- [x] Defer Level 5 outside the current dissertation execution scope.
- [x] Retain DistilBERT as the supervised contextual baseline rather than
  replacing it with a new encoder sweep.
- [x] Require confidence intervals and paired uncertainty for final primary
  claims.
- [x] Realign the thesis introduction, literature review, method route, and
  references with the approved Level 1--4 taxonomy-generalisation mainline
  before formal local execution.

### B. Freeze the description resource

- [x] Audit all twelve existing definitions using only canonical names and
  taxonomy hierarchy.
- [x] Create the minimal-description v2 file without cues, examples, or
  decision-boundary text.
- [x] Record authorship, allowed sources, forbidden sources, canonical order,
  and SHA-256.
- [x] Record the final freeze time after user approval without changing the
  reviewed text.
- [x] Add tests for exact label coverage, non-empty definitions, stable order,
  and manifest hash.
- [x] Pre-register a leakage-safe description validation rule. Target-aspect
  validation/test F1 must not select or rewrite target descriptions; permitted
  evidence may include blind semantic review and pseudo-unseen experiments
  using seen-aspect labels only.
- [x] Freeze the exact twelve descriptions under that rule before any new
  official validation/test run.
- [x] Pre-register one secondary Level 3 `DD` versus `RR` rich-taxonomy-guidance
  comparison without expanding to a full three-by-three crossover.
- [x] Freeze twelve uniform rich label cards with 3--5 non-corpus aliases, one
  inclusion boundary, one contrastive boundary, provenance, and SHA-256.

### C. Build the common stress-test infrastructure

- [x] Pre-register protocol IDs, split construction, candidate scope, pair
  sampling, seeds, budgets, selection rules, outputs, and stop rules.
- [x] Implement one candidate-pair builder that supports one, two, parent-group,
  and all-candidate evaluation.
- [x] Implement generalized seen/unseen metric partitions and harmonic-mean
  reporting.
- [x] Implement paired review-cluster bootstrap that resamples `row_uid` and
  retains all candidate pairs for each sampled review.
- [x] Extend the existing paired-aspect bootstrap/sign-flip utilities with
  exact alignment guards and protocol-specific aggregation.
- [x] Implement Level 3 `NN/DN/ND/DD/RR` rendering with identical row and pair
  identities and one shared training manifest across conditions.
- [x] Add leakage checks for rows, organisations, supervision labels,
  vocabularies, description hashes, target calibration, and test reuse.
- [x] Add focused unit and smoke tests before model execution.
- [x] Add deterministic statistical tests covering cluster preservation,
  paired resampling, nonlinear metric recomputation, degenerate intervals, and
  mismatched prediction failures.
- [x] Complete real cached-model synthetic smoke tests for all five methods,
  including checkpoint reload, sharding and resume, without reading official
  validation/test data.
- [x] Generate a dependency-aware cloud plan with exact training scopes, pair
  counts, commands and formal approval gates.
- [x] Replace cross-fold pilot tuning with nested per-training-scope selection
  and verify that validation never renders held-out candidates.
- [x] Freeze the local TF-IDF/E5/DistilBERT/Frozen-Qwen versus cloud QLoRA
  execution placement without changing the scientific protocol hash.
- [x] Implement and verify a split-isolated Frozen-Qwen raw-score cache that
  stores no text, labels, thresholds, predictions or metrics.
- [x] Add lazy Frozen-Qwen loading so a complete cache hit never loads the
  4B model.
- [x] Tag the dependency plan by executor and recalculate the revised
  Frozen-Qwen maximum from 2,773,236 uncached pairs to at most 209,448 unique
  inputs after adding `RR`.
- [x] Pass a bounded real-model cache-equivalence benchmark using synthetic
  reviews only.
- [x] Freeze a cloud dependency lock and implement a resumable first-scope
  QLoRA telemetry gate that records wall time, sampled peak GPU memory,
  scoring throughput and artifact size without enabling official test.

### D. Complete strict Level 1

- [x] Register the cross-aspect zero-label threshold-transfer rule.
- [ ] Recalculate strict zero-label Level 1 results from saved compatible scores
  where retraining is unnecessary.
- [ ] Produce matched name-only Frozen Qwen and name-only QLoRA evidence where
  existing artifacts are not compatible.
- [ ] Separate the current enhanced QLoRA package from the minimal description
  and name-only conditions.
- [ ] Freeze the strict Level 1 table used by the difficulty curve.

### E. Run Level 2 generalized single-unseen LOAO

- [x] Complete and integrity-audit strict TF-IDF nested train/validation
  selection for all 26 shared outer training scopes.
- [x] Complete and integrity-audit the fixed E5 train/validation threshold
  transfers.
- [x] Complete and integrity-audit DistilBERT nested train/validation
  selection across all 26 outer training scopes.
- [x] Complete the Frozen Qwen validation/runtime gate and verify the exact
  split-isolated raw-score cache on all official validation scopes.
- [ ] Complete nested seen-only parameter selection for every unique training
  scope.
- [ ] Run strict TF-IDF, E5, and DistilBERT on all twelve folds.
- [ ] Run Frozen Qwen and QLoRA validation/runtime gates under the same outer
  protocol.
- [ ] Pass the registered quality/runtime gate before full Qwen/QLoRA scoring.
- [ ] Complete all admitted twelve-fold evaluations.
- [ ] Report overall, seen, unseen, harmonic-mean, exact-match, presence, and
  sentiment results.

### F. Run Level 3 dual-unseen asymmetric-description crossover

- [x] Freeze the cyclic twelve-pair schedule before results.
- [x] Build `example_filtered` ten-aspect training folds with no held-out
  supervision.
- [x] Expand every evaluation review across all twelve candidates, preserving
  seen/unseen membership for the ten-plus-two split.
- [x] Verify that `NN`, `DN`, `ND`, `DD`, and `RR` reuse the exact same trained
  model, rows, pair identities, thresholds, and metrics.
- [ ] Complete nested seen-only selection and validation/runtime gates for all
  admitted core methods.
- [ ] Run all admitted models on the full registered pair schedule.
- [ ] Report within-aspect description effects, described-candidate bias,
  neither/either/both-present cases, and sentiment diagnostics.
- [ ] Report the secondary `RR - DD` rich-guidance family separately from the
  primary crossover.
- [ ] Freeze the Level 3 table and crossover figure.

### G. Run Level 4 parent-group holdout

- [x] Pre-register Company brand, Staff support, and Value group folds.
- [ ] Audit remaining training support and validation/test target support.
- [ ] Run cheap/core local models first.
- [ ] Admit Qwen/QLoRA only after the registered validation and runtime gate.
- [ ] Compare single-unseen, dual-unseen, and group-unseen degradation.

### H. Deferred Level 5

- [x] Record compound organisation-plus-taxonomy shift as future work rather
  than part of the current execution queue.
- [ ] Reopen Level 5 only through a new user-approved scope decision after
  Levels 1-4 are complete.

### I. Robustness, statistics, and final thesis evidence

- [x] Obtain user approval of the exact confidence-interval and primary-contrast
  plan before any new test result is inspected.
- [x] Pre-register bootstrap unit, interval method, bootstrap count and seed,
  primary contrasts, sign-flip rule, multiplicity handling, and permitted
  wording.
- [ ] Use seed 13 for nested per-training-scope configuration selection.
- [ ] Use seed 13 for the initial complete Level 2, Level 3, and Level 4 runs.
- [ ] Add seeds 23 and 42 only for the final matched Level 1 QLoRA endpoints.
- [ ] Aggregate the matched Level 1 seed-by-aspect results and report paired
  uncertainty.
- [ ] Report review-cluster 95% intervals for primary model metrics and paired
  95% intervals for every registered primary delta.
- [ ] Report paired aspect deltas, wins/ties/losses, and sign-flip evidence
  without treating the twelve aspects as independent review observations.
- [ ] Label every interval by the uncertainty it includes: review sampling,
  aspect/fold variation, and, where available, training-seed variation.
- [ ] Report Level 4 review-cluster intervals per parent group and avoid a
  strong cross-group significance claim from only three group folds.
- [ ] Produce the cross-level difficulty curve.
- [ ] Produce the Level 3 description crossover table/figure.
- [ ] Produce one compact Level 3 `RR` versus `DD` secondary table.
- [ ] Produce one matched Frozen-Qwen-to-QLoRA adaptation table.
- [ ] Freeze the final permitted claims, limitations, source paths, and table
  registry.
- [ ] Update Methods, Results, Discussion, abstract, captions, and appendices.
- [ ] Compile and visually inspect the complete thesis PDF.
- [ ] Run final tests, citation checks, metric checks, retired-result search,
  secret scan, and reproducibility audit.
- [x] Commit and push every completed safe pre-cloud stage to GitHub main.

## Execution order and gates

The required order is:

```text
minimal and rich resource freeze
    -> statistical protocol freeze
    -> infrastructure, leakage, and statistical tests
    -> local end-to-end smoke tests
    -> local non-QLoRA nested selection and validation gates
    -> one-scope cloud QLoRA runtime/cost benchmark
    -> complete all admitted validation selection
    -> freeze every method, threshold and comparison
    -> one controlled Level 1-4 formal test pass
    -> seeds, statistics, and thesis freeze
```

Run cheap deterministic/local methods before expensive QLoRA inference. Frozen
Qwen is local and may reuse only exact raw `P(Y)` values under the registered
split-isolated cache contract; QLoRA adapters are never shared across training
scopes. New pipelines require validation smoke tests before full sweeps. No
long cloud GPU run may begin until the code, tests, resumability, commands,
manifests, expected runtime, and cost have passed a user-reviewed
cloud-readiness gate. A failed or uninformative result is documented and
closed; test-guided redesign is not permitted.

Level 3 is mandatory core evidence. Level 4 is the planned stress test but
remains subject to registered data-support, runtime, and thesis-value gates.
Level 5 is deferred. A gate may stop an expensive model within a level; it may
not remove an unfavourable completed local result.

## Stop rules

- Do not use target labels to write or revise descriptions.
- Do not hide aspect names in a core "no description" condition.
- Do not run a one-direction described-versus-undescribed pair without the
  `DN/ND` crossover.
- Do not alter negative-pair identities across description conditions.
- Do not tune thresholds, prompts, parsers, folds, or descriptions on test.
- Do not construct or revise rich aliases or boundaries from target reviews,
  corpus frequencies, predictions, error analysis, or target F1.
- Do not replace the primary Level 3 crossover or difficulty endpoint with
  `RR`, even if `RR` performs better.
- Do not share raw-score caches between validation and test, cache QLoRA across
  adapters, or store labels, thresholds, predictions or metrics in the
  Frozen-Qwen cache.
- Do not access the test cache before the frozen threshold-transfer artifact
  and complete scientific configuration have passed their gates.
- Do not call different difficulty levels an identical protocol.
- Do not add another encoder, router, Gemini branch, anomaly detector, or model
  sweep unless a named mainline research question cannot be answered otherwise.
- Do not restore global-document sentiment, legacy TF-IDF, shallow sentiment,
  or retired router results.
- If strict calibration, harder protocols, or extra seeds weaken QLoRA, narrow
  the claim rather than selecting a favourable post-hoc subset.
