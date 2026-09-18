# E5 Validation and Threshold-Transfer Completion

Date: 2026-07-24

Status: complete for official train/validation; official test remains sealed.

## Frozen recipe

- Method: `e5_base_v2`
- Selection scope: `registered_fixed_recipe`
- Parameter SHA-256:
  `584218f2d0af9bb5703377aba010106d43dc4e12a2442fdc3113d5ff62096319`
- Parameter-selection file SHA-256:
  `a297d47cd2fdfb5a8f089d2c68f86f9b472ecc9847384eacf1f714578a876ada`
- No encoder, pooling, prefix, or model sweep was performed.

## Execution audit

The guarded through-validation plan completed `93/93` E5 jobs with zero
failures and no running jobs:

| Artifact | Count |
| --- | ---: |
| Frozen task-scope checkpoint contracts | 26 |
| Validation score scopes | 27 |
| Score CSV shards | 216 |
| Matching score manifests | 216 |
| Threshold-transfer artifacts | 39 |

The 27 score scopes comprise 12 L1, 12 L3, and 3 L4 calibration grids. L2
reuses its matched L1 seen-validation scores and receives a separate
`D`-condition threshold-transfer contract. Every Level 3 artifact lists
`NN/DN/ND/DD/RR` with one shared seen-only threshold.

Execution-state SHA-256:
`da27a4c9e0ba01514dd6dd9ee5d01cf1b05928f1183a5c99262bd53bc4ac251a`

SHA-256 over the sorted 39 embedded threshold-artifact identities:
`2e8eab229c581c4d474011a8710a26c7376cf05d1cfe64a2bb886bc1b44f159b`

No official-test job, score, analysis, or test-use ledger entry was created.

## Seen-validation calibration diagnostics

These diagnostics describe only the data used to select transferred
thresholds. They are not test performance, do not measure unseen
generalisation, and cannot answer any `D` versus `R` question.

| Level | Folds | Mean threshold | Mean seen pair micro F1 | Min–max seen pair micro F1 |
| --- | ---: | ---: | ---: | ---: |
| L1 | 12 | 0.880360 | 0.287862 | 0.229508–0.312791 |
| L2 | 12 | 0.880360 | 0.287862 | 0.229508–0.312791 |
| L3 | 12 | 0.880744 | 0.284625 | 0.179376–0.320768 |
| L4 | 3 | 0.880275 | 0.313321 | 0.300662–0.327515 |

L1 and L2 are identical by registered calibration reuse. The level means must
not be read as a difficulty curve: the seen-aspect subsets and folds differ.
Only the single sealed test pass can estimate the registered unseen
comparisons.
