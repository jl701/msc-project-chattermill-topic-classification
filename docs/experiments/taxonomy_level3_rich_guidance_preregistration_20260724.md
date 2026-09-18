# Level 3 Rich Taxonomy Guidance Pre-registration

Date: 24 July 2026

Status: user-approved and frozen before any new official taxonomy-generalisation
validation or test result was inspected

## 1. Decision

The user approved two description decisions:

1. freeze the exact twelve `minimal_v2` definitions as the core `D`
   representation; and
2. add one secondary Level 3 `DD` versus `RR` comparison using fixed rich
   taxonomy guidance.

The rich condition does not replace the primary `NN/DN/ND/DD` crossover and
does not enter the cross-level difficulty curve. The primary curve remains:

```text
L1-D -> L2-D -> L3-DD -> L4-D
```

## 2. Thesis-route alignment audit

The change is aligned with the approved thesis question for the following
reasons:

- the central object remains supplied-candidate aspect-conditioned sentiment
  under increasingly difficult taxonomy shift;
- the five-method roster, outer folds, candidate sets, training-pair budget,
  strict seen-only calibration, metrics, and statistical protocol are
  unchanged;
- `NN/DN/ND/DD` remains the core incomplete-documentation experiment;
- `RR` asks one narrower secondary question: whether richer fixed taxonomy
  guidance recovers dual-unseen performance beyond minimal definitions; and
- no label-description training, synthetic target-aspect training examples,
  target-corpus keyword mining, or target-result-guided rewriting is added.

The project therefore remains a controlled difficulty-and-intervention study,
not a description-generation thesis or an unrestricted model sweep.

## 3. Frozen resources

### Minimal `D`

Path:
`configs/experiments/fabsa_aspect_descriptions_minimal_v2.json`

Content SHA-256:

```text
fc93cf27efdb64ad335f39f4a0dbdbd3dad13b1aae5de0010280931867af7d4c
```

The user approved the exact existing text. No definition changed during the
freeze.

### Rich `R`

Path:
`configs/experiments/fabsa_aspect_rich_taxonomy_guidance_v1.json`

Content SHA-256:

```text
289ba3238cb6772f9bfda98eb73ac8a1108ba4b2bb3d23eecf72826881f415b9
```

Every aspect uses the same ordered template:

```text
canonical hierarchical name
+ the exact frozen minimal definition
+ 3-5 aliases
+ one inclusion boundary
+ one contrastive boundary
+ the separately supplied candidate sentiment
```

Aliases name a topic, entity, process, or channel rather than expressing praise
or criticism. The rich cards use only canonical names, hierarchy, and general
language knowledge. FABSA reviews, corpus terms or frequencies, gold labels,
predictions, confusion matrices, error analysis, and target F1 were prohibited.

### Bound bundle

The formal runner binds the two independently hashed resources into:

```text
fabsa_description_bundle_v1
fcf546d227ad2ac52ccfa9682fc3685d2e396069ed911016f0bc12bf205f1367
```

The loader fails closed if canonical order differs, if a rich card changes the
minimal definition, if any card lacks 3-5 unique aliases or either boundary, or
if either component hash changes.

## 4. Exact Level 3 contract

For every cyclic dual-holdout fold:

- ten seen aspects form the task-specific training scope;
- the training manifest uses minimal definitions and is built once;
- the two held-out aspects never enter the training rows or candidate pairs;
- strict validation renders and scores seen aspects only;
- one seen-only threshold is transferred to all five Level 3 conditions; and
- test uses the same reviews, aspect-sentiment pair identities, checkpoint,
  threshold, seed, and metrics in every condition.

Representations are:

| Condition | Held-out A | Held-out B | Role |
| --- | --- | --- | --- |
| `NN` | name only | name only | Core |
| `DN` | minimal | name only | Core |
| `ND` | name only | minimal | Core |
| `DD` | minimal | minimal | Core and difficulty endpoint |
| `RR` | rich guidance | rich guidance | Secondary only |

The ten seen candidates retain minimal definitions in every condition. Only
the two held-out candidate texts change. The implementation must verify common
training-manifest and evaluation-pair hashes across all five conditions.

## 5. Registered analysis

Primary families remain unchanged:

1. matched QLoRA minus Frozen Qwen at `L1-D`, `L2-D`, `L3-DD`, and `L4-D`;
2. minimal-definition effects within the Level 3 `NN/DN/ND/DD` crossover; and
3. within-method degradation along the four-level difficulty curve.

The new secondary family is:

```text
L3 RR - L3 DD
```

It is computed separately for each of the five methods using paired
review-cluster resampling. Holm adjustment is applied within the five-method
secondary family. Results must be called **rich taxonomy guidance**, not
minimal description, and cannot replace an unfavourable `DD` result.

## 6. Compute and cache consequence

`RR` adds no training scope or threshold-selection job. It adds one new
held-out representation per aspect at Level 3.

The registered per-method core scoring estimate changes from 2,658,972 to
2,773,236 logical unique pair scores. For Frozen Qwen, the maximum
split-isolated cache working set changes from 152,316 to:

```text
validation 38,052
+ test    171,396
= total   209,448 unique raw inputs
```

At the previous conservative seven-pair/second planning rate, the complete
Frozen-Qwen cache fill is approximately 8.3 hours before overhead, rather than
approximately 6.0 hours. Exact throughput will replace this estimate after the
first formal local scope.

## 7. Protocol identity

The scientific protocol SHA-256 after adding the secondary `RR` condition is:

```text
d7ccded514ac1cbccf337e496e039ac418698566be0c0ca0c21e18608cc40f85
```

The earlier `1dff49...` plan and cache contracts are scientifically obsolete
for new execution. They remain historical readiness evidence only and must not
be reused by the formal runner.

## 8. Execution gate

Before official validation selection:

- the focused resource/protocol/pipeline tests must pass;
- the full repository suite must pass;
- synthetic smoke must cover `RR` for every local method;
- the regenerated dependency plan and hashes must be recorded; and
- the implementation stage must be committed and pushed.

Official test scoring remains blocked until all admitted local and cloud method
configurations and transferred thresholds are frozen.

## 9. Pre-flight outcome

The implementation gate passed on 2026-07-24:

- focused taxonomy suite: 52 passed;
- full repository suite: 297 passed;
- all four local real-model synthetic smokes covered
  `NN/DN/ND/DD/RR`, validated resumable shards, and recorded
  `official_data_read: false`; and
- dependency plan v2 contains 2,611 jobs and has SHA-256
  `e2a06d17ed35db8ce0afd3c67c22122dd04ba299019d11c34c6f1504f84e9685`.

The detailed evidence is recorded in
`docs/experiments/taxonomy_rich_guidance_preflight_evidence_20260724.md`.
Official validation may begin after the implementation commit is pushed.
Official test remains sealed.
