# Rich-description validation confirmation v1

Date: 20 August 2026  
Status: **complete; cloud expansion gate closed**

## Decision

The globally selected `R2_positive_concat` interface did not outperform the
minimal-description `D` control on formal validation. Both cheap confirmation
methods showed lower held-out pair F1, lower aspect-presence average precision,
lower aspect-presence F1 and more false-positive rows.

Therefore:

- retain one `R` result as a controlled Level 2 ablation in the main thesis;
- put the multi-interface design search in the appendix;
- use `N/D` only for Frozen-Qwen few-shot, DistilBERT and QLoRA cloud runs; and
- do not redesign or retune `R` from validation outcomes.

## Twelve-fold macro means

| Method | Condition | Presence AP | Presence F1 | Precision | Recall | Held-out pair F1 | Overall pair F1 | FP rows/100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| strict TF-IDF | D | 0.501334 | 0.411028 | 0.397694 | 0.592337 | 0.277251 | 0.334630 | 7.962788 |
| strict TF-IDF | R | 0.492230 | 0.370519 | 0.325422 | 0.633249 | 0.247899 | 0.331960 | 12.582781 |
| frozen E5 | D | 0.352326 | 0.290552 | 0.286952 | 0.466483 | 0.222184 | 0.326267 | 11.628824 |
| frozen E5 | R | 0.340878 | 0.276189 | 0.270139 | 0.475847 | 0.211957 | 0.323716 | 13.568275 |

## Paired R minus D effects

| Method | Presence AP | Presence F1 | Precision | Recall | Held-out pair F1 | Overall pair F1 | FP rows/100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict TF-IDF | -0.009105 | -0.040509 | -0.072272 | +0.040911 | -0.029352 | -0.002669 | +4.619994 |
| frozen E5 | -0.011448 | -0.014363 | -0.016814 | +0.009364 | -0.010227 | -0.002551 | +1.939451 |

The pattern is interpretable: the added aliases and inclusion boundary make
the detector more permissive, slightly increasing recall, but the precision
loss and false-positive increase are larger. On TF-IDF, precision was lower in
all twelve folds; on E5 it was lower in ten of twelve folds. Overall pair F1
was lower in ten TF-IDF folds and all twelve E5 folds.

## Integrity audit

- development: 72/72 pseudo-folds and 576/576 interface records;
- confirmation: 12/12 TF-IDF folds and 12/12 E5 folds;
- failure, non-finite, resume-conflict and test-contract counts: all zero;
- official test: sealed and never loaded;
- the recomputed `D` results exactly match the earlier completed Level 2 `D`
  control for all four audited headline metrics.

Evidence hashes:

- development summary:
  `c1801fb962a2cd868a0b1183343892749bb71c3e993c5730f6815623ffd23d6f`;
- TF-IDF confirmation:
  `07e8590ed2d6629918961f427e657023197c1db62a77331f9fa5450027776dba`;
- E5 confirmation:
  `0d9ccd5b2714981cffa5ab0293490d4dd9de51a6efdd2ce6d4592e79bad57b0c`;
- evidence audit:
  `921e9a1578f5f88c5966601780261a3f3fea466fbd25359d60aef8663f09c8c1`.

## Claim boundary

This result does not prove that rich descriptions are intrinsically harmful.
It shows that, under the frozen FABSA resource, selected positive-rich
interface, two-stage decoder and two confirmation representations, richer
candidate text did not improve end-to-end validation. That bounded negative
claim is sufficient to stop costly model-family expansion without turning the
thesis into an open-ended prompt-engineering study.
