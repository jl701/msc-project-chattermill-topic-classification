# DistilBERT Cross-Encoder Validation and Selection Completion

Date: 2026-07-25

Status: complete for official train/validation; official test remains sealed.

## Frozen search and selection rule

- Method: `distilbert_review_candidate_cross_encoder`
- Thesis role: supervised contextual baseline
- Model: `distilbert-base-uncased`
- Model revision:
  `12040accade4e8a0f71eabdb258fecc2e7e948be`
- Task format: binary sequence classification of one review and one rendered
  aspect-sentiment candidate
- Search scope: three pre-registered learning-rate candidates inside each of
  26 distinct outer training scopes
- Learning-rate grid: `2e-5`, `3e-5`, `5e-5`
- Fixed recipe:
  - maximum length: `256`;
  - training batch size: `32`;
  - evaluation batch size: `96`;
  - weight decay: `0.01`;
  - epochs: `3`;
  - warmup ratio: `0.1`;
  - AMP: enabled; and
  - selected checkpoint epoch: `3`.
- Primary selection metric: seen-validation pair micro F1
- Tie-breaks: pair samples F1, pair micro precision, then lower learning rate
- Adaptive grid expansion: forbidden and not used

All parameter selection was nested inside the relevant outer training scope.
Held-out candidate claims were not rendered or scored during validation
selection, and validation evidence was not pooled across outer folds.

## Selected-recipe distribution

All 26 training scopes selected one of the three registered candidates:

| Learning rate | Training scopes |
| --- | ---: |
| `2e-5` | 0 |
| `3e-5` | 5 |
| `5e-5` | 21 |

The dominant `5e-5` recipe was not imposed globally: `3e-5` independently won
five outer-scope comparisons. The result therefore reflects the registered
nested selection rule rather than post-hoc global parameter choice.

SHA-256 over the newline-joined sorted content hashes of the 26 parameter
selection files:
`14bc6be0bc6f57748ad5e35b37fdbef3a69b2e72a47662fed2afaaf75f617524`

## Execution and integrity audit

The guarded through-validation plan completed `326/326` DistilBERT jobs with
zero failures and no running jobs:

| Artifact | Count |
| --- | ---: |
| Candidate training checkpoint contracts | 78 |
| Candidate tuning summaries | 78 |
| Frozen outer-scope parameter selections | 26 |
| Unique validation score contracts | 79 |
| Score CSV shards | 632 |
| Matching score manifests | 632 |
| Threshold-transfer artifacts | 39 |
| Final seen-validation summaries | 39 |

The 78 candidate checkpoints are the complete `26 x 3` registered grid. The
79 score contracts comprise the candidate validation scores plus one
additional formal Level 4 calibration scope required by cross-level
training-scope reuse. Every score contract has exactly eight shards.

The integrity audit found:

- all 78 checkpoint contracts have valid self-hashes;
- every checkpoint model file exists and matches its recorded SHA-256;
- all 79 score contracts have valid self-hashes and complete shard indices
  `0` through `7`;
- every one of the 632 CSV files matches its manifest SHA-256;
- all 39 threshold artifacts have valid self-hashes, reference existing
  checkpoint and validation-score contracts, and keep calibration and held-out
  aspects disjoint;
- one scientific-protocol hash and one method-spec hash throughout;
- one bound description-resource hash throughout:
  `fcf546d227ad2ac52ccfa9682fc3685d2e396069ed911016f0bc12bf205f1367`;
- all scored artifacts declare `validation`, never `test`;
- no stale temporary checkpoint directory remains; and
- no official-test manifest or test-use artifact was created.

Total executor wall time, including the short recoverable process interruption,
was `06:07:30.58`.

Execution-state SHA-256:
`c3f7b388758baa3efb19f03235fc23b477f6b2c2aa47daae67670b16d1d4e317`

SHA-256 over the newline-joined sorted 39 embedded threshold-artifact
identities:
`b231621355f71ee2a529c0511e2b67336fee2d1eedd0e0eb9ae50566fcabf06b`

## Seen-validation calibration diagnostics

These diagnostics describe only seen-aspect evidence used to select the
learning rate and transferred threshold. They are not official test
performance, do not measure unseen-label generalisation, and cannot answer the
description or difficulty research questions.

| Level | Folds | Mean threshold | Mean seen pair micro F1 | Min-max | Mean presence F1 | Mean conditional sentiment accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | 12 | 0.827264 | 0.432605 | 0.375639-0.473222 | 0.902840 | 0.943656 |
| L2 | 12 | 0.827264 | 0.432605 | 0.375639-0.473222 | 0.902840 | 0.943656 |
| L3 | 12 | 0.857799 | 0.447860 | 0.409589-0.494463 | 0.869250 | 0.953087 |
| L4 | 3 | 0.864370 | 0.459951 | 0.422171-0.487506 | 0.882195 | 0.953607 |

L1 and L2 are exactly identical by registered calibration reuse because they
share the same held-out aspect and outer training scope. Their candidate
interfaces differ only during the still-sealed formal evaluation. Cross-level
validation means must not be interpreted as a difficulty curve because the
seen-aspect calibration subsets differ.

