# Taxonomy multi-sentiment decoder validation v1: results

Date: 17 August 2026

Status: complete and audited

## Question and fixed boundary

The completed true two-stage Study C emits one argmax sentiment for each
selected aspect. This validation-only ablation tested whether a decoder that
can automatically emit one, two, or three sentiments improves held-out pair
F1. It used the existing strict train-only TF-IDF and frozen E5-base-v2
pipelines across all twelve Level 2 held-out-aspect folds.

The official test remained sealed. Aspect and sentiment thresholds were
selected only from seen-aspect validation labels; held-out-aspect validation
labels were used only for evaluation. The existing Study C result was not
modified.

## Compared decoders

- **Argmax control:** select an aspect with its seen-validation aspect
  threshold, then emit exactly its highest-scoring sentiment.
- **Automatic multi-threshold:** jointly select aspect and sentiment thresholds
  on seen validation. Emit every sentiment above the sentiment threshold, with
  an argmax fallback when none passes. This permits one, two, or three outputs.
- **Forced top-two stress test:** always emit the two highest-scoring
  sentiments for each selected aspect. This is a deliberately unconditional
  negative control, not a proposed final decoder.

## Held-out results

Values are unweighted means over twelve held-out-aspect folds.

| Method | Decoder | Pair P | Pair R | Pair F1 | Delta vs argmax | Wins / ties / losses | Mean sentiments / selected aspect | Multi-output rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | Argmax | 0.2744 | 0.3706 | **0.2780** | — | — | 1.0000 | 0.00% |
| TF-IDF | Automatic threshold | 0.2749 | 0.3631 | 0.2704 | **−0.0076** | 3 / 7 / 2 | 1.0020 | 0.20% |
| TF-IDF | Forced top-two | 0.2089 | 0.4446 | 0.2451 | **−0.0329** | 3 / 0 / 9 | 2.0000 | 100.00% |
| E5-base-v2 | Argmax | 0.2273 | 0.3702 | **0.2246** | — | — | 1.0000 | 0.00% |
| E5-base-v2 | Automatic threshold | 0.2225 | 0.3780 | 0.2224 | **−0.0021** | 0 / 10 / 2 | 1.0306 | 2.11% |
| E5-base-v2 | Forced top-two | 0.1616 | 0.3242 | 0.1732 | **−0.0513** | 0 / 0 / 12 | 2.0000 | 100.00% |

Both automatic decoders therefore finish below their matched argmax controls.
Allowing multiple outputs is technically possible, but the additional recall
does not offset the precision and threshold-transfer cost in this dataset.

## Rare multi-sentiment labels

The twelve held-out folds contain 24 gold sentiment labels belonging to 12
review-aspect instances with more than one gold sentiment. Recovery was:

| Method | Argmax | Automatic threshold | Forced top-two |
|---|---:|---:|---:|
| TF-IDF | 9 / 24 (37.5%) | 9 / 24 (37.5%) | 16 / 24 (66.7%) |
| E5-base-v2 | 6 / 24 (25.0%) | 8 / 24 (33.3%) | 9 / 24 (37.5%) |

The forced top-two decoder demonstrates the trade-off directly: it can recover
more secondary gold sentiments, but it adds a second false label to the much
larger set of single-sentiment aspect instances. Its total F1 is consequently
substantially worse.

## Calibration diagnostics

The TF-IDF automatic decoder emitted multiple held-out sentiments in only
about 0.20% of selected aspect instances. Its largest loss occurred on
`l2-a08` (0.3299 to 0.2275), even though its held-out outputs on that fold had
one sentiment per selected aspect. The joint seen-validation search had chosen
a different aspect threshold, which transferred poorly to the unseen aspect.
This loss is therefore a full-structure calibration effect, not evidence that
many extra sentiment labels were emitted on `l2-a08`.

E5 emitted multiple sentiments primarily on `l2-a05` and `l2-a07`; these were
also its only two losing folds. It recovered two additional multi-gold labels
overall, but the extra predictions reduced mean precision enough to leave F1
below argmax.

## Decision

Retain one-sentiment argmax as the current default. A multi-capable output
structure should not be adopted merely because the annotation format permits
rare multiple sentiments. The present evidence shows no end-to-end F1 benefit
for either fast model.

If a later thesis claim needs to isolate only the sentiment-cardinality choice,
the clean supplementary test is to freeze the parent Study C aspect threshold
and select only a second-sentiment threshold on seen validation. That would
remove the joint aspect-threshold calibration effect observed for TF-IDF. It
should be preregistered as a separate follow-up rather than selected from these
held-out results.

That sequential follow-up was subsequently frozen and completed as
`taxonomy_capped_two_sentiment_validation_v2`; its result is recorded in
`docs/experiments/taxonomy_capped_two_sentiment_validation_v2_results.md`.

## Audit

- Methods: 2; completed folds: 24 / 24; failed folds: 0.
- Decoder rows: 72 / 72.
- Parent Study C argmax precision, recall, and F1 reproduced to `1e-12` in
  every fold.
- Official-test artifacts/accesses: 0.
- Non-finite values: 0.
- Audited artifacts with SHA-256 records: 28.
- Machine-readable audit:
  `outputs/experimental/taxonomy_multi_sentiment_decoder_validation_v1/audit.json`.

Raw fold outputs remain under the ignored experimental output root. The frozen
configuration, implementation, tests, audit script, and this aggregate report
are safe to track.
