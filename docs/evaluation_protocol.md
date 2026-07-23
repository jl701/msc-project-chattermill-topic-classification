# FABSA Evaluation Protocol

This note records the FABSA evaluation setup. The current experiment hierarchy,
admission rules, and completion checklist are defined only in
`docs/dissertation_loao_mainline_lock_2026_07_23.md`. The older
`docs/dissertation_experiment_design_lock_2026_07_16.md` is historical
provenance. The completed singleton all-row LOAO protocol below is now Level 1
of the approved taxonomy-generalisation difficulty ladder rather than the sole
future benchmark.

## 1. Closed-Topic Benchmark

The provided FABSA split is kept as the closed-topic benchmark for comparability:

| Split | Rows | Role |
| --- | ---: | --- |
| train | 7,930 | Train on the full fixed taxonomy |
| validation | 1,057 | Tune thresholds / select runs |
| test | 1,587 | Final closed-topic comparison |

All three provided splits contain the same 12 aspects. This makes the split useful for standard fixed-taxonomy modelling, but not for unseen-topic evaluation.

## 2. Held-Out Organisation Split

This split evaluates cross-company/domain-shift generalisation.

Current selected split:

| Split | Organisations | Rows | Supervision aspects | Pair labels |
| --- | --- | ---: | ---: | ---: |
| train | all except validation/test orgs | 7,020 | 12 | 36 |
| validation | `600` | 1,533 | 12 | 32 |
| test | `369`, `727` | 2,021 | 12 | 32 |

The selected organisations are large enough to preserve broad aspect coverage while keeping train/validation/test organisation sets disjoint.

Leakage checks:

- Row overlap between train/validation/test: `0`.
- Organisation overlap between train/validation/test: `0`.
- Label overlap is expected in this protocol because the taxonomy is fixed; the shift is in organisation/domain, not in labels.

## 3. Held-Out Aspect Split

This split evaluates open-topic/new-aspect generalisation under a candidate-label setup.

Current held-out aspects:

- `Account management: Account access`
- `Company brand: Competitor`
- `Value: Discounts promotions`

These aspects were selected because they cover different business areas and have enough validation/test examples for an initial benchmark.

Evaluation uses candidate labels at inference. The model should select from canonical labels rather than generate free-form topic names.

### Strategy A: Label-Masked Training

Rows from the official training split are kept if they still contain at least one seen aspect after held-out labels are removed.

| Split | Rows | Supervision aspects | Pair labels |
| --- | ---: | ---: | ---: |
| train | 7,632 | 9 | 27 |
| validation | 212 | 3 | 8 |
| test | 281 | 3 | 8 |

This setting tests whether a model can learn from texts that may contain held-out-topic language, while never receiving held-out-topic supervision.

### Strategy B: Example-Filtered Training

Any official training row containing a held-out aspect is removed entirely.

| Split | Rows | Supervision aspects | Pair labels |
| --- | ---: | ---: | ---: |
| train | 6,495 | 9 | 26 |
| validation | 212 | 3 | 8 |
| test | 281 | 3 | 8 |

This is stricter because the training texts should not contain held-out-topic examples.

Leakage checks for both strategies:

- Row overlap between train/validation/test: `0`.
- Train-vs-validation held-out supervision aspect overlap: `0`.
- Train-vs-test held-out supervision aspect overlap: `0`.
- Organisation overlap is expected because this protocol isolates the label axis, not the company axis.

## 4. Frozen LOAO Open-Topic Protocol

Protocol version: `loao_open_topic_all_row_v1`

Status: frozen as the completed Level 1 supplied-candidate benchmark. It remains
the entry point for the dissertation, but it no longer defines the entire
experimental mainline.

This is the canonical Level 1 protocol for the supplied single-unseen-candidate
claim. Fixed held-out-aspect experiments remain useful capability and
deployment evidence, but they must not be presented as full twelve-fold
robustness evidence. Levels 2-5 below intentionally change the candidate scope
or shift axes and therefore use distinct protocol IDs.

### Fold Construction

For each of the 12 FABSA aspects:

1. Hold out exactly one aspect.
2. For supervised or fine-tuned models, train without held-out-aspect supervision.
3. Use `example_filtered` as the primary clean training strategy for future headline LOAO experiments: remove any training row containing the held-out aspect.
4. Keep `label_masked` only as an incomplete-label-noise ablation unless a specific experiment justifies making it primary.
5. Evaluation uses the official validation and test rows without row filtering.
6. Gold labels are filtered to the current held-out aspect only.
7. Candidate labels at inference contain exactly the current held-out aspect.
8. Empty predictions are allowed and are required for rows that do not mention the held-out aspect.

This all-row design is stricter than evaluating only rows that contain the held-out aspect. It tests whether a model can decide that a new candidate topic is absent as well as present.

### Prediction Target

The evaluated labels are aspect+sentiment pairs:

```text
<held-out aspect> | positive
<held-out aspect> | neutral
<held-out aspect> | negative
```

For LLM systems, the frozen prompt family is indexed candidate-label structured output. The model should return candidate IDs rather than copied aspect names, and the parser should map valid IDs back to canonical labels before scoring. Invalid JSON, invalid candidate IDs, invalid sentiments, duplicate items, and conflicting sentiments must be counted in diagnostics. Generated prediction files may contain review text and must remain under ignored `outputs/`.

### Main Metrics

For the all-row LOAO protocol, the primary robustness comparison metric is:

- mean test `pair_micro_f1` across the 12 held-out aspects.

The primary metric is pair micro F1 rather than pair samples F1 because all-row LOAO contains many empty-gold rows. Sample-level F1 does not reward true-negative empty predictions and can hide false-positive behaviour on empty-gold rows. Pair samples F1 must still be reported for continuity with the rest of the project, but it should not be the sole LOAO selection or interpretation metric.

Report these metrics for validation and test:

- binary candidate-presence precision, recall, and F1
- presence PR-AUC only when the method exposes a genuine continuous presence score
- per-aspect `pair_samples_f1`
- per-aspect `pair_micro_f1`
- per-aspect `pair_micro_precision`
- per-aspect `pair_micro_recall`
- per-aspect `pair_macro_f1`
- false-positive rows per 100 reviews
- false-positive labels per 100 reviews
- false-negative rows per 100 reviews
- exact-match rate
- aspect samples/micro/macro F1
- oracle-presence sentiment accuracy or macro F1 where a modular sentiment component can be run on every gold-present row
- detected-present sentiment accuracy or macro F1, accompanied by its denominator, coverage of gold-present rows, and candidate recall

The candidate-presence metrics collapse sentiment before scoring. They are not interchangeable with strict pair precision and recall, for which a correct aspect with the wrong sentiment remains an error. A discrete Qwen JSON decision has no PR-AUC unless an independently specified continuous presence score is available.

Candidate-presence F1 is the positive-class binary score `2TP / (2TP + FP + FN)`. True-negative absent rows must not enter this F1. In particular, do not use scikit-learn binary `average="micro"` on the one-column singleton-aspect indicator because it collapses to an accuracy-like score that includes true negatives. The corrected implementation and frozen-prediction audit are recorded in `docs/experiments/loao_presence_metric_correction_20260717.md`.

For LLM runs, also report:

- valid JSON rate
- schema-valid or parser-valid rate, when applicable
- invalid candidate-ID count
- invalid sentiment count
- duplicate prediction count
- conflicting-sentiment count
- seconds per example
- hardware or API endpoint family
- token usage and approximate cost, when relevant

### Aggregation

The main LOAO table should report the unweighted mean, standard deviation, minimum, and maximum across the 12 held-out aspects. Per-aspect tables should also be kept because aspect difficulty is part of the result.

Pooled micro F1 across all folds may be reported as an optional diagnostic, but it must not replace the unweighted aspect-level spread because pooled scores are dominated by more frequent aspects.

### Selection Rules

Validation data may be used for model, threshold, prompt, and hyperparameter selection. Test labels must only be used after the selected configuration is fixed.

For thresholded local models:

- select thresholds within each fold on validation data
- use validation `pair_micro_f1` as the preferred all-row LOAO selection metric
- report the selected threshold per aspect

This is the target-calibrated cold-start regime because each target aspect may contribute validation gold to its own threshold. The strict zero-label supporting experiment must instead select every threshold, prompt, parser policy, hyperparameter, and router rule using only the other 11 aspects, then apply the frozen pipeline to the target test fold. The existing leave-one-aspect router-policy diagnostic is partial because its underlying local threshold remains target-calibrated.

For zero-shot LLM runs:

- fix the prompt variant before full test evaluation
- do not select prompt variants or parsing policies using test labels
- use validation only for smoke checks, prompt selection, and failure diagnosis

For fine-tuned Qwen or other trained LLMs:

- select the final training configuration by validation aggregate LOAO performance
- evaluate the selected configuration once on test
- record all training data filters, seeds, LoRA/QLoRA parameters, epochs, batch sizes, gradient accumulation, sequence length, and checkpoint-selection rules

### Diagnostic Variants

Positive-row LOAO is allowed only as a diagnostic:

- validation/test rows contain the held-out aspect
- the candidate set contains only the held-out aspect
- at least one prediction may be forced

This view mostly measures sentiment assignment once aspect presence is assumed. It must not be used as the main open-topic robustness result.

Sampled hosted-LLM LOAO is allowed when API cost makes full LOAO unreasonable, but it must be labelled as a sampled diagnostic. It cannot replace the full all-row local/selected-model LOAO result.

### Comparability Rules

- Do not directly compare fixed three-aspect scores with all-row LOAO scores as if they were the same benchmark.
- Use fixed held-out-aspect scores for controlled candidate-label capability and deployment/cascade analysis.
- Use all-row LOAO scores for the open-topic robustness claim.
- Use positive-row LOAO only for sentiment diagnostics.
- Use hosted Gemini fixed/cascade results as hosted-LLM reference or upper-bound evidence unless a clearly labelled Gemini LOAO diagnostic is run.
- Any future change to row scope, candidate-set size, aggregation, or primary metric must be named as a new protocol version rather than silently replacing `loao_open_topic_all_row_v1`.

## 5. Approved Taxonomy-Generalisation Stress-Test Suite

The stress-test suite is governed by
`docs/dissertation_loao_mainline_lock_2026_07_23.md`. The summary below fixes
the protocol boundaries that future implementation must preserve.

### Level 2: Generalized Single-Unseen LOAO

For every target aspect:

- train on the other eleven aspects using `example_filtered`;
- evaluate all twelve candidate aspects on every official validation/test row;
- keep the target aspect marked as unseen and the other eleven as seen;
- allow a review to receive zero, one, or multiple aspect-sentiment pairs; and
- report overall, seen, unseen, and harmonic-mean performance.

The candidate-set expansion makes this a new protocol. Its raw F1 must not be
placed beside Level 1 as though only the model changed.

### Level 3: Dual-Unseen Asymmetric Descriptions

For each pre-registered pair of held-out aspects:

- remove every training row containing either target aspect;
- train on the remaining ten aspects;
- evaluate both unseen aspects simultaneously on all official rows; and
- render the unseen pair as `NN`, `DN`, `ND`, and `DD`, where `N` means
  canonical name only and `D` means canonical name plus the frozen minimal
  definition.

All four conditions must reuse the same model, rows, candidate-pair identities,
thresholds, and metrics. `DN` and `ND` form the required crossover.

### Level 4: Parent-Group Holdout

Hold out all children of Company brand, Staff support, or Value in turn, then
test all twelve candidates jointly. The parent groups are separate folds and
must not be pooled into a synthetic training set.

### Level 5: Compound Organisation And Taxonomy Shift

Construct the aspect holdout inside the organisation-disjoint split. Training,
validation, and test organisations must be disjoint, and no target-aspect label
may calibrate the target pipeline. Report target support and per-aspect
uncertainty because some aspect-sentiment labels are rare.

### Strict Information Regime

The new difficulty curve uses strict zero-label selection. A target-aspect
validation label may not select descriptions, thresholds, prompts, parsers,
checkpoints, policies, or hyperparameters. Existing target-calibrated Level 1
results remain separately labelled reference evidence.

### Cross-Level Interpretation

Levels deliberately differ in candidate count and distribution shift.
Cross-level analysis must therefore focus on within-method degradation and on
the amount recovered by a matched intervention. It must not treat raw scores
from different levels as one interchangeable leaderboard.

## Reproduction

Analyse split candidates:

```powershell
python .\scripts\analyse_split_candidates.py
```

Build split manifests:

```powershell
python .\scripts\build_fabsa_splits.py
```

Generated manifests are written under:

```text
outputs/splits/
```

The output directory is ignored by Git. The committed code and this note document how to reproduce the split decisions.
