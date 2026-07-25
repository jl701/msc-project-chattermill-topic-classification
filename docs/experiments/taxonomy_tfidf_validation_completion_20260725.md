# Strict Train-Only TF-IDF Validation and Threshold-Transfer Completion

Date: 2026-07-25

Status: complete for official train/validation; official test remains sealed.

## Frozen search and selection rule

- Method: `strict_train_only_tfidf`
- Thesis role: classical lexical candidate-pair baseline
- Search scope: 15 pre-registered candidates inside each of 26 distinct outer
  training scopes
- Grid:
  - logistic-regression `C`: `0.1`, `0.3`, `1.0`, `3.0`, `10.0`;
  - feature ablation: `all_six`, `char_cosine_and_cues`,
    `word_cosine_and_cues`.
- Primary selection metric: seen-validation pair micro F1
- Tie-breaks: pair samples F1, pair micro precision, then lower model
  complexity
- Adaptive grid expansion: forbidden and not used

The vectorizers were fitted only on the official training-fold review text and
seen-candidate text present in that training manifest. Held-out candidate text,
validation review text, test review text, target predictions, and target
performance did not enter fitting or candidate selection.

## Selected-recipe distribution

All 26 training scopes selected one of the registered candidates:

| Selected component | Training scopes |
| --- | ---: |
| `word_cosine_and_cues` | 18 |
| `char_cosine_and_cues` | 4 |
| `all_six` | 4 |
| `C = 0.1` | 15 |
| `C = 0.3` | 4 |
| `C = 1.0` | 2 |
| `C = 3.0` | 2 |
| `C = 10.0` | 3 |

Nine of the fifteen registered candidate identities were selected at least
once. This is expected under nested per-training-scope selection and is not a
post-hoc global recipe choice.

SHA-256 over the newline-joined sorted content hashes of the 26 parameter
selection files:
`5f8f0c03912bf3159565d7786beb27d3c4905d29c3572a4440e05cb6a4d72ed9`

## Execution and integrity audit

The guarded through-validation plan completed `1,262/1,262` TF-IDF jobs with
zero failures and no running jobs:

| Artifact | Count |
| --- | ---: |
| Candidate training checkpoint contracts | 390 |
| Candidate tuning summaries | 390 |
| Frozen outer-scope parameter selections | 26 |
| Unique validation score contracts | 391 |
| Score CSV shards | 3,128 |
| Matching score manifests | 3,128 |
| Threshold-transfer artifacts | 39 |

The 390 candidate checkpoints are the complete `26 x 15` registered grid. The
391 score contracts comprise the grid's cached validation scores plus one
additional formal Level 4 calibration scope required by cross-level
training-scope reuse. Every score contract has exactly eight shards.

The integrity audit found:

- zero missing checkpoint models and zero checkpoint model-hash mismatches;
- zero missing score CSVs and zero score CSV-hash mismatches across
  approximately 3.27 GiB of saved scores;
- zero missing or malformed shard index sets;
- one scientific-protocol hash and one method-spec hash throughout;
- one bound description-resource hash throughout:
  `fcf546d227ad2ac52ccfa9682fc3685d2e396069ed911016f0bc12bf205f1367`;
- all score manifests declare `validation`, never `test`;
- all 39 threshold artifacts have valid content hashes and reference existing
  validation score contracts;
- no official-test artifact, analysis, or test-use ledger entry exists.

A single uncommitted temporary checkpoint directory left by the earlier
Windows state-lock interruption contained no manifest and was not referenced
by any selection or execution state. It was removed after the completed
artifacts were verified; 390 manifest/model pairs remain.

Execution-state SHA-256:
`61657c61a536dfc182651181ce4db7c00e9fa9fb1f650e0ad4135be6e6f5d861`

SHA-256 over the newline-joined sorted 39 embedded threshold-artifact
identities:
`dd20c6f4fef73074f8d16ce6739b65566d7e1f8bc7dc5d9af68dacf925e19157`

## Seen-validation calibration diagnostics

These values describe only the seen-aspect data used to choose the nested
recipe and transferred threshold. They are not test performance, do not
measure unseen-label generalisation, and cannot answer `D` versus `R`.

| Level | Folds | Mean threshold | Mean seen pair micro F1 | Min-max seen pair micro F1 |
| --- | ---: | ---: | ---: | ---: |
| L1 | 12 | 0.618025 | 0.279524 | 0.241588-0.294500 |
| L2 | 12 | 0.618025 | 0.279524 | 0.241588-0.294500 |
| L3 | 12 | 0.595175 | 0.276459 | 0.209011-0.301173 |
| L4 | 3 | 0.612225 | 0.289055 | 0.283759-0.292673 |

L1 and L2 deliberately reuse the same nested training/seen-calibration scope
because they hold out the same single aspect; their candidate interfaces differ
only at sealed evaluation. The cross-level means must not be interpreted as a
difficulty curve because their seen-aspect calibration subsets differ. The
registered unseen comparisons require the single sealed test pass after every
local and cloud method has frozen its recipe and threshold.
