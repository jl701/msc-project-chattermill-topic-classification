# Taxonomy Validation Freeze Gate G

Date: 2026-08-17

Status: technical pre-freeze passed; scientific freeze awaits the recorded
Level 3 candidate-representation decision. Official test remains sealed.

## Purpose and boundary

Gate G closes the train/validation evidence chain before any single-pass
official-test execution. This review is deliberately test-blind: it does not
read `test.csv`, create a test run contract, pass `--include-official-test`, or
inspect target/test predictions. The exact QLoRA boundary remains
`formal-qwen_candidate_pair_qlora-seed0042-l1-a12-select-threshold`.

## Completed execution states

| Method | Completed | Failed | Running | Official test included |
|---|---:|---:|---:|---|
| Strict train-only TF-IDF | 1,262 | 0 | 0 | No |
| E5-base-v2 | 93 | 0 | 0 | No |
| DistilBERT pair cross-encoder | 326 | 0 | 0 | No |
| Frozen Qwen candidate pair | 93 | 0 | 0 | No |
| QLoRA Qwen candidate pair | 398 | 0 | 0 | No |

The QLoRA V10 close-out independently revalidated 12 seed-42 checkpoint
contracts, 120 checkpoint files, 96 score shards, 418,572 finite scores and 12
threshold transfers. It found no missing/hash-mismatched artifact, duplicate
pair, predictive collapse, non-finite value, resume residue, alert, thermal
slowdown or test contract.

## Cross-method validation inventory

| Method | Checkpoints | Declared checkpoint files | Validation score manifests | Validation score rows | Thresholds | Parameter selections | Validation summaries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict train-only TF-IDF | 390 | 390 | 3,128 | 12,874,260 | 39 | 26 | 39 |
| E5-base-v2 | 26 | 26 | 216 | 887,880 | 39 | 1 global fixed recipe | 39 |
| DistilBERT pair cross-encoder | 78 | 468 | 632 | 2,600,220 | 39 | 26 | 39 |
| Frozen Qwen candidate pair | 26 | 26 | 216 | 887,880 | 39 | 1 global fixed recipe | 39 |
| QLoRA Qwen candidate pair | 102 | 1,020 | 824 | 3,437,364 | 63 | 26 | 63 |

Every one of the 5,016 score manifests in this inventory declares
`split=validation`; the corresponding test-score-manifest count is zero.
There are also zero formal test summaries and zero test-use ledger claims.

## Freeze anchors

The tracked code baseline at the start of Gate G is Git commit `84527cd`.
The following SHA-256 values identify the scientific configuration:

| Artifact | SHA-256 |
|---|---|
| `taxonomy_generalisation_precloud_v1.json` | `c63e550b25a82001f2d85c67a7088de3680cccad7d6c42cabb5577b4f4f66ef3` |
| `taxonomy_method_registry_v1.json` | `f59d71d1d1e006ddb042e70bf855da4fa5e1529f9d28d9c98e71bdf75696c9c7` |
| Minimal descriptions v2 | `8f72951f436bd6af606dd7d07ec249f406ce6b7d6ae690817978da5430c92106` |
| Rich guidance v1 | `649ad01b32e931dcaba5339bebb50bfdd270928900cfa38d1bbccbeaec7d3f31` |
| Hybrid execution plan v2 | `e2a06d17ed35db8ce0afd3c67c22122dd04ba299019d11c34c6f1504f84e9685` |

A path-and-content hash over 6,161 local validation evidence files is
`2729ea3c45e55510fb6b4ebbc61cc3217d6b62e1b6b1d32e98c8cd8dab2680fb`.
The set contains checkpoint manifests, score manifests, all 219 threshold
transfers, all 80 parameter-selection artifacts, formal validation summaries
and the five through-validation state files. Checkpoint manifests transitively
declare model-file hashes; score manifests transitively declare raw-CSV,
pair-identity, score and run-contract hashes. Raw review text, predictions and
model weights remain local-only and are not committed.

## Technical leakage and integrity result

The technical pre-freeze passes:

- minimal descriptions, rich guidance and statistics are all
  `approved_and_frozen`;
- all five state ledgers are complete with no running or failed job;
- all 219 threshold artifacts pass their self-hash and reference an existing
  checkpoint and validation score contract;
- every threshold calibration-aspect set is non-empty and disjoint from its
  held-out-aspect set;
- there are no test score manifests, test summaries or test-use claims; and
- V10's full file/hash/finite-score audit passes with no resume residue.

This audit checks the frozen contracts and validation artifacts without
loading official test data. Synthetic and contract-level regression tests are
the permitted way to exercise test guards before the one real test pass.
The focused Gate G suite passed 73 tests covering resources, protocol folds,
formal gates, leakage audit, method contracts, checkpoints, score shards,
threshold transfers, execution plans and primary-comparison guards. A real
configuration-level gate probe confirmed both description and statistical
resources as `approved_and_frozen`. The final target-bounded dry-run selected
398 validation-only QLoRA jobs, found all 398 already complete and reported
`include_official_test=false`.

## Scientific freeze blocker

No post-meeting decision is recorded for the Level 3 held-out representation.
The current registered `N` condition means canonical hierarchical aspect name
plus candidate sentiment, without a definition. Earlier supervisor discussion
considered a different asymmetric task in which one held-out aspect receives
name plus description while the other receives neither name nor description.
That alternative changes candidate identity/rendering and is not silently
equivalent to `DN`, `ND` or `NN`.

Consequently, Gate G does **not** authorise official-test opening yet. The
completed training, seen-only calibration, checkpoints and thresholds remain
reusable because held-out Level 3 candidate claims were excluded from those
steps. Before test, one of the following must be written into a versioned
protocol:

1. retain the registered supplied-candidate crossover (`N` = canonical name,
   `D` = canonical name plus minimal definition); or
2. add/replace it with an explicitly specified opaque or absent-identity Level
   3 condition, including how the unnamed candidate is supplied, how output
   identity is represented and how TF-IDF/E5/DistilBERT/Qwen receive the same
   information.

If the second route is chosen, implementation and synthetic/validation-only
contract checks must complete before this freeze is reissued. Neither route
may be chosen after observing official-test performance.

## Next authorised action

Run the Gate G regression suite and preserve this technical freeze. Then obtain
and record the Level 3 decision. Only a subsequent explicit test-opening
approval may use `--include-official-test`; this document is not that approval.
