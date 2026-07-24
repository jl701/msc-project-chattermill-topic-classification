# Taxonomy-Generalisation Pre-Cloud Readiness Report — 23 July 2026

> Historical readiness snapshot. The scientific protocol remains current, but
> the execution placement, Frozen-Qwen workload and local/cloud time estimates
> in Sections 11-12 were superseded on 24 July 2026 by
> `docs/experiments/taxonomy_local_non_qlora_execution_readiness_20260724.md`.
> TF-IDF, E5, DistilBERT and cached Frozen Qwen are now local; QLoRA is cloud.

## Executive conclusion

The pre-cloud implementation for the approved dissertation mainline is
complete and locally verified. The repository now contains one guarded
pipeline for:

```text
L1-D -> L2-D -> L3-DD -> L4-D
```

with the five registered methods:

1. strict train-only TF-IDF;
2. E5-base-v2;
3. DistilBERT review-candidate cross-encoder;
4. Frozen candidate-pair Qwen; and
5. candidate-pair QLoRA.

The code implements the complete Level 3 `NN/DN/ND/DD` crossover, strict
seen-aspect calibration, immutable parameter and threshold selection,
content-addressed checkpoints and scores, exact resume, review-cluster
confidence intervals, paired comparison families and a dependency-aware cloud
execution plan.

No new official validation or test inference was run. All real-model smoke
tests used code-generated synthetic reviews and explicitly record
`official_data_read=false`. Formal execution remains fail-closed until the
description resource and statistical plan are approved and marked
`approved_and_frozen`.

The complete repository test suite passes:

```text
285 passed
```

## 1. The exact scientific question

The experiment is not a flat leaderboard. It asks:

> As an aspect-conditioned sentiment system moves from one supplied unseen
> aspect to mixed seen/unseen candidate sets, two unseen aspects with
> asymmetric descriptions, and an unseen parent-group, how do different model
> families degrade, and how much is recovered by minimal label descriptions
> and QLoRA task adaptation?

Every model receives claims of the form:

```text
(review, candidate aspect representation, candidate sentiment)
                              |
                              v
                    applicable / not applicable
```

For one aspect there are three candidate claims, one for each sentiment. This
is not a separate `present/not-present` classifier followed by a three-way
classifier. The three binary claims jointly represent both decisions:

- no positive claim means that the aspect is absent;
- exactly one positive claim means that the aspect is present with that
  sentiment; and
- multiple positive claims are possible raw errors and remain visible to the
  pair-set metric rather than being silently repaired.

The output for a review is the union of all accepted aspect-sentiment pairs.
Consequently, the same review may contain several aspects with different
sentiments, or no predicted aspect at all.

## 2. The four difficulty levels

| Level | Training holdout | Evaluation candidates | Core condition | Scientific role |
| --- | --- | --- | --- | --- |
| L1 | one aspect | supplied held-out aspect only | `D` | easiest supplied-candidate unseen-aspect task |
| L2 | one aspect | all 12 aspects | `D` | unseen aspect competes with 11 familiar aspects |
| L3 | two cyclic aspects | all 12 aspects | `NN/DN/ND/DD` | dual-unseen description crossover |
| L4 | all children of one parent | all 12 aspects | `D` | unseen sub-taxonomy/parent-group stress test |

L1 and L2 with the same held-out aspect have identical training evidence and
therefore share a checkpoint. The `Value` L4 fold holds out exactly `A11+A12`,
which is also one registered L3 dual-holdout scope, so that checkpoint is
shared as well. There are therefore 26, not 27, unique seed-13 training scopes.

Raw F1 across levels is not called an identical benchmark: the candidate set
and task difficulty change. The thesis reports both endpoint scores and
harder-minus-easier degradation, with an explicit label that the two task
grids differ.

## 3. The Level 3 supervisor experiment

Every L3 fold trains once on the same ten seen aspects. The ten seen aspects
always have their minimal definitions. Only the two held-out aspect
representations change at inference:

| Condition | Held-out A | Held-out B |
| --- | --- | --- |
| `NN` | name only | name only |
| `DN` | name + definition | name only |
| `ND` | name only | name + definition |
| `DD` | name + definition | name + definition |

The four conditions share the model checkpoint, review rows, gold pairs,
candidate-aspect-sentiment identities, threshold, seed and metric code. The
runner verifies those identities before scoring.

The final analysis includes:

- the confirmatory `DD - NN` comparison for every method;
- `DN - NN` and `ND - NN`;
- `DD - ND` and `DD - DN`;
- both-present recall;
- false positives when neither held-out aspect is present;
- recall when only the name-only aspect is gold;
- selection bias towards the described candidate; and
- held-out per-aspect and per-sentiment diagnostics.

The directional contrasts are kept because a one-direction experiment would
confound description availability with the intrinsic difficulty of A versus B.

## 4. Minimal descriptions and leakage control

The proposed resource is
`configs/experiments/fabsa_aspect_descriptions_minimal_v2.json`, content hash:

```text
fc93cf27efdb64ad335f39f4a0dbdbd3dad13b1aae5de0010280931867af7d4c
```

It contains definitions only—no corpus-derived cues, review examples,
confusion-driven boundaries or validation/test observations.

| Aspect | Proposed frozen minimal definition |
| --- | --- |
| Account access | Accessing or regaining access to a customer account, including signing in, passwords, registration, verification, locked accounts, and account credentials. |
| Competitor | Another company, provider, brand, or alternative is named, compared, preferred, considered, or used as a substitute for the focal company. |
| General satisfaction | An overall judgement of the company, service, or complete customer experience rather than a judgement limited to one operational feature. |
| Reviews | Customer reviews, ratings, testimonials, reputation feedback, or the act of reading, leaving, requesting, or responding to a review. |
| Speed | How quickly or slowly a journey, ride, delivery, collection, or logistics service happens, including delays, waiting time, and punctuality. |
| App/website | Using the mobile app, website, webpage, online portal, or digital interface, including navigation, loading, crashes, bugs, and online functionality. |
| Ease of use | How easy, difficult, clear, or convenient it is to purchase, order, reserve, book, check out, or complete the customer transaction workflow. |
| Attitude of staff | The behaviour, manner, helpfulness, friendliness, rudeness, professionalism, knowledge, or empathy of employees and support staff. |
| Email | Customer contact or support through email, including sending messages, replies, response time, missing responses, and the quality of an email exchange. |
| Phone | Customer contact or support by telephone, including calls, call centres, being on hold, callbacks, answering, and telephone conversations. |
| Discounts/promotions | Discounts, promotions, promotional offers, coupons, vouchers, sales, special deals, loyalty offers, or advertised price reductions. |
| Price/value | The ordinary price, cost, charge, fee, affordability, expensiveness, cheapness, or whether the service is worth what the customer paid. |

The file records authorship, allowed and forbidden sources, canonical order and
content hash. Its status remains `pending_user_approval`; approval must add the
freeze status/time without changing the twelve texts. If any text changes, the
content hash changes and all incompatible checkpoints/scores fail validation.

## 5. Strict calibration, especially Level 1

Target-aspect validation labels may not select a threshold, checkpoint,
description, prompt, parser or hyperparameter.

Every level constructs a separate seen-only calibration grid. Held-out
candidate claims are not merely filtered after inference: they are never
rendered or scored on validation. One outer-fold threshold is selected from
the pooled seen-aspect calibration pairs.

L1 required a special correction. Its evaluation grid contains only the
supplied held-out aspect, so it cannot itself provide seen-aspect calibration.
The implemented strict route is:

1. train the L1/L2-shared checkpoint without the held-out aspect;
2. construct the corresponding L2-D calibration grid using the eleven seen
   candidates only;
3. score that seen-only grid;
4. select the threshold from the eleven seen aspects only;
5. transfer that threshold unchanged to the L1 name-only and description test
   conditions; and
6. score only the held-out candidate on the L1 test task.

The L1 runner never scores held-out validation targets. Its seen-calibration
score artifact is exactly reusable by L2 threshold selection, avoiding a
second model load.

## 6. Model implementations and finite optimisation

### Strict train-only TF-IDF

The vectorizer fits only the permitted training-pair manifest. It may see
training review text and the ten/eleven seen candidate texts, but it cannot see
held-out candidate text, validation text or test text.

The classifier is balanced logistic regression over six lexical interaction
features. The finite grid is:

```text
C in {0.1, 0.3, 1, 3, 10}
x
feature block in {all six, char cosine+cues, word cosine+cues}
= 15 candidates
```

### E5-base-v2

This is a frozen bi-encoder using `passage:` for reviews, `query:` for
candidates, masked-mean pooling and L2-normalised similarity. The exact model
revision is pinned. Only the shared outer-fold threshold is selected; there is
no encoder or prompt sweep.

### DistilBERT cross-encoder

The review and one candidate claim are jointly encoded by
`distilbert-base-uncased` with a binary sequence-classification head. The
historically used three-epoch recipe is fixed. The permitted optimisation is:

```text
learning rate in {2e-5, 3e-5, 5e-5}
= 3 candidates
```

Fixing three epochs avoids independently retraining redundant one-, two-,
three- and four-epoch trajectories while still optimising the most influential
registered parameter.

### Frozen candidate-pair Qwen

`Qwen/Qwen3-4B-Instruct-2507` is pinned to revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. It receives the same candidate
claim as QLoRA and answers one token, `Y` or `N`. The score is:

```text
P(Y) = exp(logit_Y) / (exp(logit_Y) + exp(logit_N))
```

The model is loaded in 4-bit NF4. Prompt, verbalizers, parser, quantisation and
revision are locked.

### Candidate-pair QLoRA

QLoRA uses the same base revision, input format, verbalizers and probability
mapping as Frozen Qwen. The difference is a task-trained low-rank adapter on
`q_proj`, `k_proj`, `v_proj` and `o_proj`, with rank 4, alpha 8 and dropout
0.05.

The historically successful one-epoch recipe is fixed. The finite seen-only
grid is:

```text
learning rate in {2e-6, 5e-6, 1e-5}
= 3 candidates
```

TF-IDF, DistilBERT and QLoRA are tuned independently inside each of the 26
unique outer training scopes. Their three or fifteen candidates are compared
only on that scope's seen-aspect validation grid. Validation evidence is never
pooled across outer folds: an aspect held out in one fold is seen in another,
so cross-fold pooling would leak information about the target taxonomy item
into global parameter selection.

Selection uses seen-validation pair micro-F1 followed by the predeclared
tie-breakers. The grid cannot be expanded after seeing results. E5 and Frozen
Qwen generate an immutable global fixed-recipe selection artifact without
pretending that a hyperparameter sweep occurred.

## 7. Training pairs and negative construction

Task-trained methods receive the same deterministic pair manifest, capped at
4,096 pairs per training scope:

- every permitted gold aspect-sentiment pair;
- both wrong sentiments for the same gold aspect;
- one absent same-parent aspect with the same sentiment when available; and
- one deterministic random absent aspect with the same sentiment.

Pairs are deduplicated by:

```text
(row_uid, candidate_aspect, candidate_sentiment)
```

A valid gold sentiment for a multi-label review cannot be generated as a
negative. Held-out aspects are absent both from original retained training
rows and from constructed candidate pairs.

## 8. Reproducibility and execution safety

The formal runner is split into:

```text
train
-> score-validation
-> select-threshold
-> score-test
-> analyse-test
```

Important safeguards are:

- `train` opens only the official train file;
- validation stages open train + validation, never test;
- test stages open train + test only after selection is frozen;
- model revisions, method registry, description hash, scientific protocol
  hash, training manifest, evaluation rows, pair identities, parameters, seed
  and shard count enter content-addressed contracts;
- administrative approval-state changes do not alter the scientific protocol
  hash, but any scientific change does;
- score files exclude raw review and candidate text;
- writes are atomic and existing incompatible artifacts are rejected;
- row-based sharding keeps all claims for one `row_uid` together;
- `--all-shards` loads the model once and processes eight independently
  resumable shards;
- Frozen E5/Qwen train stages write a registry reload marker without loading or
  fitting the model;
- a local ledger permits exact test resume but rejects a second incompatible
  contract for the same endpoint; and
- seed 13/23/42 threshold and summary paths cannot collide.

## 9. Statistical contract

Every final fold-condition reports point estimates plus review-cluster
percentile 95% intervals. The bootstrap:

- resamples `row_uid`;
- keeps every candidate, sentiment and fold observation for a sampled review;
- uses 20,000 replicates and seed 13;
- recomputes nonlinear F1 inside every replicate; and
- recomputes the seen/unseen harmonic F1 from the two resampled F1 values.

The three primary families are:

1. QLoRA minus matched Frozen Qwen at `L1-D`, `L2-D`, `L3-DD`, `L4-D`;
2. `L3-DD` minus `L3-NN` within each method, with the four directional
   crossover contrasts as registered supporting evidence; and
3. within-method degradation along `L1-D -> L2-D -> L3-DD -> L4-D`.

Families 1 and 2 have identical observation keys and gold sets. They receive
paired review-cluster difference intervals, paired fold/aspect
wins/ties/losses, sign-flip tests and Holm adjustment within the declared
family.

Family 3 compares different task grids. It uses common review resamples but is
explicitly labelled `different_task_grids_shared_review_resampling`; it does
not receive a misleading matched-fold significance claim.

Level 1 QLoRA additionally runs seeds 23 and 42. Each seed gets a conditional
review-sampling interval; three-seed spread is reported descriptively rather
than claimed as a precise population interval over random initialisations.

## 10. Local real-model smoke evidence

The final v6 smoke used one synthetic L3 fold, four conditions, 47 synthetic
training pairs and three synthetic evaluation reviews. Each condition has 108
logical pair outputs.

Across conditions, the naive total is 432 model calls. The union contains only
126 distinct rendered claims:

```text
10 seen aspects x 3 sentiments
+ 2 held-out aspects x 2 representation variants x 3 sentiments
= 42 claims/review
```

For three reviews this is 126 calls, a 70.8% reduction from 432.

| Method | Fit/load seconds | Score seconds | Unique claims | Result |
| --- | ---: | ---: | ---: | --- |
| Strict TF-IDF | 0.035 | 2.429 | 126 | pass |
| E5-base-v2 | 0.920 | 3.124 | 126 | pass |
| DistilBERT | 1.799 | 2.663 | 126 | pass |
| Frozen Qwen | 10.460 | 16.227 | 126 | pass |
| QLoRA | 35.157 | 17.428 | 126 | pass |

Every method passed checkpoint reload, two-shard merge and within-process
resume. A second QLoRA process reported:

```text
fit_seconds = 0
new_pairs_scored_this_run = 0
scientific hashes/output = identical
```

Synthetic F1 values are intentionally not reported as results. The smoke only
validates execution behaviour.

## 11. Exact cloud workload

The generated plan contains 2,611 dependency-aware jobs. This number includes
cheap selection and CPU analysis stages; it is not the number of GPU training
runs.

| Component | Count |
| --- | ---: |
| tuning train jobs | 546 |
| tuning validation-score jobs | 546 |
| tuning threshold jobs | 546 |
| per-scope/global parameter-selection jobs | 80 |
| formal train/marker jobs | 154 |
| formal validation-score jobs | 81 |
| formal threshold jobs | 219 |
| formal test-score jobs | 219 |
| formal test-analysis jobs | 219 |
| final primary-comparison job | 1 |

QLoRA has 102 actual task-specific training executions:

```text
26 independent training scopes x 3 registered learning rates
+ 24 extra Level 1 scopes for seeds 23 and 42
= 102
```

The selected candidate already has a checkpoint for every seed-13 training
scope, so the 26 selected seed-13 checkpoints are reused rather than
retrained. Their validation scores are also reused wherever the tuning fold
and formal fold have the same scientific contract. L1 and L2 share each
single-aspect training scope and seen-calibration grid. The Level 3
`(Discounts/promotions, Price/value)` and Level 4 `Value` folds share a
training scope but not the same fold contract, so Level 4 receives its own
formal validation score.

The exact unique model-score estimates are:

| Scope | Pair scores |
| --- | ---: |
| one method, complete seed-13 core | 2,658,972 |
| QLoRA extra Level 1 seeds | 1,065,672 |
| QLoRA tuning | 2,568,510 |
| selected QLoRA tuning scores reused by formal core | −856,170 |
| complete QLoRA actually scored | 5,436,984 |
| Frozen Qwen + complete QLoRA actually scored | 8,095,956 |
| all methods, tuning and extra seeds actually scored | 29,771,592 |

The 3.52 MB generated plan is local at
`outputs/experimental/taxonomy_cloud_execution_plan_v1_20260723.json`, with
SHA-256:

```text
b9e6444beb7f3d4f4fbc7e4effcb0bcb0bb78f5a4c528742b1ec95698829d358
```

It is generated rather than committed because it contains 2,611 mechanical
command records; the generator and its tests are tracked.

## 12. Runtime and AutoDL budget

Historical full QLoRA evidence gives approximately 10.63 local GPU hours for
12 training folds. The new Qwen workload is much larger. At the observed local
rate of roughly seven
candidate claims/second:

- Frozen + QLoRA scoring alone is about 321 local GPU hours;
- 102 QLoRA training runs extrapolate to about 90 local GPU hours; and
- the Qwen portion is therefore roughly 412 local GPU hours before overhead.

This is why the complete suite should not run on the local laptop.

AutoDL currently lists RTX 4090 24 GB at ¥1.98/hour, RTX 5090 32 GB at
¥2.88/hour, A800 80 GB at ¥4.98/hour and H800 80 GB at ¥8.88/hour. AutoDL also
states that billing follows instance on-time rather than GPU utilisation and
stops when the instance is shut down:

- <https://www.autodl.com/>
- <https://www.autodl.com/docs/price/>

The recommended first choice is one RTX 4090 because the 4-bit 4B model already
fits the local 8 GB device, and 24 GB permits the registered batch sizes. A
larger-memory data-centre GPU should be chosen only if a measured one-fold
benchmark is sufficiently faster to offset its higher hourly price.

Planning range, not a guarantee:

| Assumption | Total wall time | 4090 compute cost |
| --- | ---: | ---: |
| 5x local Qwen speed plus cheaper-model/analysis overhead | about 100 h | about ¥198 |
| 3x local Qwen speed plus cheaper-model/analysis overhead | about 175 h | about ¥347 |
| 25% retry/setup reserve | about 125–219 h billed | about ¥248–434 |

A practical budget reserve is ¥400–500. This must be replaced by a measured
projection after the first registered QLoRA tuning fold. The first cloud gate
records:

- actual peak VRAM;
- QLoRA pairs/second and training time;
- Frozen Qwen pairs/second;
- checkpoint upload/storage size;
- projected total hours and cost; and
- whether the 4090 remains the cost-efficient choice.

For price comparison, a 5090 must be more than `2.88 / 1.98 = 1.45x` faster
than a 4090 to reduce compute cost. An A800 must be more than `4.98 / 1.98 =
2.52x` faster. Peak TFLOPS alone is not accepted as proof; the decision uses
the measured project workload.

## 13. What is complete and what is deliberately not complete

Complete:

- Levels 1–4 fold/candidate/data contracts;
- all five model runtimes;
- finite tuning and stopping rules;
- strict seen-only calibration that never renders held-out validation
  candidates;
- nested per-training-scope hyperparameter selection with no cross-fold
  validation pooling;
- immutable checkpoints, scores, selection and test-use governance;
- eight-shard single-load execution;
- L3 condition-score deduplication;
- point, interval, paired and cross-protocol statistics;
- Level 1 three-seed aggregation;
- cloud command/dependency generation;
- five-method real-model synthetic smoke;
- exact cross-process QLoRA resume; and
- 285 passing tests.

Not complete and not claimed:

- the twelve descriptions are not yet user-approved/frozen;
- the statistical contract is not yet user-approved/frozen;
- no new official validation tuning has run;
- no new official test result exists;
- no strict final L1 result table exists;
- no Level 2–4 result table exists;
- no claim that QLoRA remains best under the harder strict protocols exists;
- cloud time/cost has not yet been measured on the rented GPU; and
- thesis Results/Discussion must not be updated with synthetic smoke values.

## 14. Required approval before cloud execution

The user should approve or revise exactly two scientific gates:

1. the twelve minimal descriptions above, after which the resource status and
   freeze time can be recorded without changing the texts; and
2. the statistical contract: 20,000 row-cluster percentile bootstrap
   replicates, paired model/description differences, sign-flip evidence, Holm
   within the two matched primary families, shared-review but explicitly
   non-identical cross-level degradation, and three-seed Level 1 QLoRA
   sensitivity.

After those gates are frozen, execution should begin with the registered
selection stage and one QLoRA cloud benchmark. It must not jump directly to
full test scoring.
