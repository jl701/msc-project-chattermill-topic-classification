# Matched one-stage versus two-stage taxonomy validation v1

Date: 17 August 2026

Status: completed and audited; the protocol below was preregistered before
execution

Result: both fast methods improve under the factorised architecture. See
`docs/experiments/taxonomy_matched_one_vs_two_stage_validation_v1_results.md`.

## Question

Does splitting taxonomy prediction into aspect presence followed by conditional
sentiment improve held-out aspect-sentiment prediction compared with scoring all
36 aspect-sentiment pairs independently?

This is a sequential validation study motivated by the supervisor discussion.
The capped-two sentiment rule was frozen before this run and is the primary
two-stage decoder. A top-one argmax decoder is retained only as a diagnostic.

## Matched contract

- Methods: strict train-only TF-IDF and frozen E5-base-v2.
- Data: official train and validation only. Official test remains sealed.
- Folds: the same twelve registered Level 2 single-held-out-aspect folds.
- Training filter: remove every training review containing the held-out aspect.
- Validation: both architectures receive exactly the same complete validation
  rows and the same 12 candidate aspects.
- Representation: canonical aspect name plus approved minimal description for
  every seen and held-out aspect in both architectures.
- Selection: each architecture selects its own necessary threshold only from
  seen-aspect validation candidates, with the same end-to-end pair micro-F1
  objective and deterministic tie-breaks.
- Evaluation: held-out-aspect validation candidates are scored and evaluated
  after thresholds are frozen. Their labels never enter selection.

The training split is identical, while the supervised examples necessarily
encode the architecture being tested. The one-stage TF-IDF model retains the
original deterministic 4,096-pair training budget and negative sampling, but
all candidate strings are rerendered as name plus description before fitting.
The two-stage TF-IDF model trains an aspect-presence classifier on review-aspect
examples and a conditional sentiment classifier on gold review-aspect examples.
E5 remains frozen and receives no task-specific fitting.

## Systems compared

`one_stage_independent_pairs` scores all 36 aspect-sentiment candidates and
emits every pair at or above its seen-validation pair threshold.

`two_stage_capped_two` first thresholds 12 aspect-presence scores. For each
selected aspect it always emits the highest-scoring sentiment and emits the
runner-up only when that score reaches a second threshold selected on seen
validation. Aspect output remains uncapped; sentiment output is capped at two.

`two_stage_argmax` uses the same two-stage aspect scores and aspect threshold,
but emits exactly one sentiment per selected aspect. It is a diagnostic that
separates the effect of task factorisation from the capped-two decoder.

## Reporting and safety

The primary result is the unweighted mean held-out pair micro-F1 across folds
for one-stage independent pairs versus two-stage capped-two. Precision, recall,
aspect F1, fold wins/ties/losses, output cardinality, and two-sentiment recall
are secondary evidence. All score and identity artifacts require finite values
and recorded SHA-256 hashes. Any official-test access, held-out-guided selection,
representation mismatch, row mismatch, resume conflict, or incomplete artifact
stops the run.
