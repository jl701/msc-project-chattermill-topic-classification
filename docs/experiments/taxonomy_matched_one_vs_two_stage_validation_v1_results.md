# Matched one-stage versus two-stage taxonomy validation v1: results

Date: 17 August 2026

Status: complete and audited

## Result

Under a genuinely matched Level 2 validation contract, splitting aspect
presence from conditional sentiment improves held-out aspect-sentiment F1 for
both fast methods. The gain is not caused by different validation reviews,
candidate descriptions, or access to held-out labels during threshold
selection.

Values are unweighted means over the same twelve single-held-out-aspect folds.
The primary two-stage result uses the frozen capped-two sentiment decoder.

| Method | Architecture / decoder | Pair P | Pair R | Pair F1 | Aspect F1 | F1 delta vs one-stage | Wins / ties / losses vs one-stage |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | One-stage independent pairs | 0.1678 | 0.5111 | 0.2196 | 0.3084 | — | — |
| TF-IDF | Two-stage capped-two | 0.2721 | 0.3727 | **0.2773** | **0.4110** | **+0.0577** | **11 / 0 / 1** |
| TF-IDF | Two-stage argmax diagnostic | 0.2744 | 0.3706 | 0.2780 | 0.4110 | +0.0584 | 11 / 0 / 1 |
| E5-base-v2 | One-stage independent pairs | 0.1684 | 0.3718 | 0.1860 | 0.2726 | — | — |
| E5-base-v2 | Two-stage capped-two | 0.2192 | 0.3780 | **0.2222** | **0.2906** | **+0.0362** | **9 / 0 / 3** |
| E5-base-v2 | Two-stage argmax diagnostic | 0.2273 | 0.3702 | 0.2246 | 0.2906 | +0.0386 | 10 / 0 / 2 |

The factorised architecture raises precision substantially. TF-IDF trades some
of the one-stage system's very high recall for much better precision, producing
the largest F1 gain. E5 improves precision without materially reducing recall.
These are paired validation-fold improvements; no claim of statistical
significance is made from the descriptive means and wins alone.

## What caused the improvement

The two-stage argmax diagnostic is only 0.0008 above capped-two for TF-IDF and
0.0024 above it for E5. By contrast, both two-stage decoders are 0.036–0.058
above their matched one-stage controls. The main gain therefore comes from
factorising the decision:

1. estimate whether each of the 12 aspects is present;
2. threshold the aspect-presence scores; and
3. decide sentiment only for selected aspects.

This supports the supervisor's reasoning that aspect detection and sentiment
classification benefit from separate decision spaces. It does not establish
that capped-two itself improves F1; capped-two is retained because it represents
the annotation structure while imposing only a small F1 cost.

## Sentiment structure

| Method | Architecture / decoder | Mean sentiments per selected aspect | Two-sentiment gold labels recovered |
|---|---|---:|---:|
| TF-IDF | One-stage independent pairs | 1.4723 | 13 / 24 (54.2%) |
| TF-IDF | Two-stage capped-two | 1.0162 | 9 / 24 (37.5%) |
| TF-IDF | Two-stage argmax | 1.0000 | 9 / 24 (37.5%) |
| E5-base-v2 | One-stage independent pairs | 1.7122 | 15 / 24 (62.5%) |
| E5-base-v2 | Two-stage capped-two | 1.0514 | 8 / 24 (33.3%) |
| E5-base-v2 | Two-stage argmax | 1.0000 | 6 / 24 (25.0%) |

The original independent-pair control can emit all three sentiments for one
aspect and did so in both methods. This is preserved in the control because it
is part of the original 36-independent-decision formulation. The adopted
two-stage system remains dataset-faithful: aspect output is uncapped, while
each selected aspect receives one or two sentiments and never three.

## Exact matched conditions

- Same twelve registered Level 2 folds.
- Same 1,057 official validation rows in every fold and architecture.
- Same 36 aspect-sentiment evaluation identities per review.
- Same canonical aspect name plus approved minimal description.
- Same training split after removing any review containing the held-out aspect.
- Architecture-specific thresholds selected only on seen-aspect validation
  candidates with the same end-to-end pair-F1 objective and deterministic
  tie-breaks.
- Held-out aspect labels used only after all thresholds were frozen.
- TF-IDF uses the registered starting recipe in both architectures. The
  one-stage branch retains its original deterministic 4,096-pair budget and
  negative sampling; the two-stage branch uses aspect-presence examples and
  gold-aspect conditional-sentiment examples, as required by the architecture.
- E5 is frozen and receives no task-specific fitting.

## Decision

For future two-stage experiments, retain:

1. uncapped multi-label aspect selection;
2. aspect presence followed by conditional sentiment;
3. capped-two sentiment decoding as the primary dataset-faithful rule;
4. argmax only as a constrained diagnostic; and
5. the original 36-pair system only as a matched architecture control.

This experiment is sequential validation evidence and does not authorise an
official-test run. It also does not modify the separately deferred unknown-
aspect anomaly-detection study.

## Safety and audit

- Methods: 2; folds: 24 / 24; decoder rows: 72 / 72.
- Failed folds: 0; resume conflicts: 0; non-finite values: 0.
- Official-test artifacts and test-contract count: 0.
- Matched validation-row, pair-identity, representation, and score hashes are
  recorded for every fold.
- Maximum sentiments per aspect: one-stage control 3, argmax 1, capped-two 2.
- Audited artifacts with SHA-256 records: 28.
- Protocol SHA-256:
  `9788502d6bdce86a389c1584b67b18a5be85762279d0b203c78cc531b034387c`.
- Machine-readable audit:
  `outputs/experimental/taxonomy_matched_one_vs_two_stage_validation_v1/audit.json`.

Raw fold results remain under the ignored experimental output root. The frozen
configuration, runner, audit, focused tests, aggregate table, and this report
are safe to track.
