# Taxonomy multi-sentiment decoder validation v1

Date: 17 August 2026

Status: completed and audited; the frozen protocol below was preregistered
before execution

Result: both automatic multi-sentiment decoders finished below their matched
argmax controls. See
`docs/experiments/taxonomy_multi_sentiment_decoder_validation_v1_results.md`.

## Question

The completed true two-stage Study C emits exactly one sentiment for every
selected aspect. The official labels contain 72 train rows and 12 validation
rows in which one aspect has more than one gold sentiment. This ablation asks
whether supporting those cases improves held-out pair F1, or whether additional
false-positive sentiments reduce precision enough to make F1 worse.

## Fixed scope

- Methods: strict train-only TF-IDF and frozen E5-base-v2 only.
- Data: official train and validation only; official test remains sealed.
- Folds: all twelve registered Level 2 single-held-out-aspect folds.
- Candidate representation: canonical name plus approved minimal description.
- Training rows containing the held-out aspect are removed.
- Thresholds use seen-aspect validation labels only.
- Held-out-aspect validation labels are evaluation-only.

The existing Study C argmax results are immutable. This is a separate decoder
ablation and cannot retroactively select or alter the parent result.

## Decoders

The control selects an aspect by its validation-selected aspect threshold and
emits exactly the highest-scoring sentiment.

The primary multi-sentiment decoder jointly selects an aspect threshold and a
sentiment threshold on seen-aspect validation. For every selected aspect it
emits all sentiments meeting the sentiment threshold. If none meets it, the
argmax sentiment is emitted, so every selected aspect still has at least one
sentiment. The sentiment-threshold grid contains 65 fixed empirical quantiles
plus a value above the maximum. The last candidate exactly reproduces argmax,
so the multi-capable validation family nests the control.

A fixed top-two decoder is a secondary stress test. It always emits the two
highest-scoring sentiments for every selected aspect and reselects only the
aspect threshold on seen validation. It is included to measure the precision
cost of unconditional multi-sentiment output, not as the recommended model.

Threshold selection maximises end-to-end pair micro-F1. Ties prefer pair
samples F1, precision, fewer false-positive rows, fewer sentiments per selected
aspect, and then higher thresholds.

## Reporting

The primary comparison is the unweighted mean held-out pair micro-F1 across
twelve folds. Precision, recall, fold wins, emitted sentiment cardinality, and
recall on multi-gold aspect instances are also reported. A multi-sentiment
result may be worse on held-out data even though its seen-validation search
nests argmax, because the selected threshold must transfer to an unseen aspect.

Raw grids and scores remain under the ignored output root. The frozen config,
implementation, tests, and aggregate result may be tracked.
