# Dataset-faithful capped-two sentiment validation v2

Date: 17 August 2026

Status: completed and audited; the frozen protocol below was preregistered
before execution

Result: the capped-two decoder preserved the dataset structure at a small mean
F1 cost. See
`docs/experiments/taxonomy_capped_two_sentiment_validation_v2_results.md`.

## Motivation and provenance

This is a transparent sequential follow-up to the completed
`taxonomy_multi_sentiment_decoder_validation_v1`. That first ablation jointly
selected aspect and sentiment thresholds and allowed up to three sentiments.
It showed that calibration transfer could change aspect predictions and that
unconditional multi-sentiment output was harmful. Its results remain immutable.

The follow-up isolates the narrower structural question: once the parent Study
C aspect decisions are frozen, should a selected aspect receive one sentiment
or, when supported by its runner-up score, two?

## Dataset-derived output structure

The official train and validation splits contain reviews with many aspects.
Train reviews contain up to seven aspects and validation reviews up to five;
17.62% and 17.31%, respectively, contain at least three aspects. Aspect output
therefore remains uncapped and genuinely multi-label.

Within one review-aspect instance, however, both splits contain at most two
sentiments. There are 73 two-sentiment instances in train and 12 in validation;
all other labelled review-aspect instances contain one sentiment. The decoder
therefore permits at most two sentiments and never emits a third.

No official test data or labels were opened to derive these counts.

## Fixed experiment

- Methods: strict train-only TF-IDF and frozen E5-base-v2.
- Data: official train and validation only; official test remains sealed.
- Folds: all twelve registered Level 2 single-held-out-aspect folds.
- Representation: canonical aspect name plus approved minimal description.
- Training rows containing the held-out aspect are removed.
- The parent Study C aspect threshold is selected on seen-aspect validation and
  then used unchanged by both sentiment decoders.
- Held-out-aspect validation labels are evaluation-only.

For every aspect selected by the frozen parent threshold, the control emits the
highest-scoring sentiment. The capped-two decoder also emits that top sentiment
and adds the runner-up only when the runner-up score reaches a second-sentiment
threshold selected on seen-aspect validation. Candidate thresholds are the
exact strict candidates induced by selected seen runner-up scores, plus a value
above the maximum that exactly reproduces argmax.

Selection maximises end-to-end seen-validation pair micro-F1. Ties prefer pair
samples F1, precision, fewer doubled aspect instances, and the higher threshold.
The aspect threshold cannot change during this search.

## Reporting boundary

Primary reporting is the unweighted mean held-out pair micro-F1 over twelve
folds, with precision, recall, fold wins, output cardinality, and two-sentiment
gold-label recall. Because this design follows inspection of v1 validation
results, it is reported as sequential development evidence, not an independent
confirmation. No further rule will be selected from its held-out results.

Raw scores and review-level outputs remain under the ignored output root. The
frozen config, implementation, tests, aggregate result, and audit may be
tracked.
