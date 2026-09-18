# Taxonomy final held-out evaluation v1: completion and results

Date completed: 25 August 2026

Protocol: `taxonomy_two_stage_final_test_v1`

Scoring execution commit:
`6a0b42df370d40c3b93eff3867fb8909af9539b8`

Status: **complete; H1 passed; gate-kept H2 passed; post-test model
development closed**

## Executive conclusion

The locked revised-protocol held-out evaluation completed successfully. All 93
registered base score jobs finished before any outcome was revealed. The 12
pre-registered fixed stage-wise compositions were then generated without
training or threshold selection. The complete 105-bundle graph passed its
integrity barrier and 210 score/manifest files were copied to an independent
local backup before the private labels were joined exactly once.

The validation-selected fixed stage-wise composition achieved an
aspect-balanced Level-2 `D` held-out pair micro-F1 of **0.507538**. It exceeded
Frozen Qwen few-shot by **+0.012140**, with a synchronized review-cluster
bootstrap 95% interval of **[+0.001407, +0.022439]**. H1 therefore passed.
Under the frozen hierarchical rule, H2 was then confirmatory: the composition
exceeded QLoRA by **+0.021101**, interval **[+0.000184, +0.041118]**. Both
registered superiority claims passed without retraining, retuning or changing
the endpoint.

This is a locked held-out assessment of the revised two-stage protocol, not a
claim that the project had never previously encountered the released test
partition under superseded protocols.

## 1. Execution and reveal integrity

| Check | Result |
| --- | ---: |
| Registered base score jobs | 93/93 |
| Fixed-composition jobs | 12/12 |
| Sealed score bundles | 105/105 |
| Independently copied score/manifest files | 210/210 |
| Registered metric rows | 603/603 |
| Diagnostic rows | 201/201 |
| Score manifest/hash failures | 0 |
| Non-finite metric or diagnostic values | 0 |
| Third-sentiment violations | 0 |
| Registered bundle-level prediction collapses | 0 |
| Outcome reveals | exactly 1 |
| Post-test tuning permitted | no |

The original worker receipts also covered 93 score bundles plus three
`READY_TO_STOP` units. Before consolidation, 189 received files were rechecked
against their recorded byte sizes and SHA-256 values; all matched. The three
RunPod workers were stopped after local receipt verification, so the retained
GPU billing rate is zero.

The single-reveal manifest SHA-256 is
`e861af7229c53a41a47fe5623553774a83d9f5173031bb507019dd49110698f5`.
The compact post-reveal audit is
`docs/experiments/taxonomy_final_test_v1_post_reveal_audit_20260825.json`.

## 2. Confirmatory Level-2 `D` result

The primary endpoint is the unweighted mean of the twelve fold-specific
held-out pair micro-F1 values. The sensitivity endpoint pools held-out pair
counts across folds. The interval resamples the same 1,587 review identities
across methods and folds using 20,000 draws and seed 13.

| System | Primary aspect-balanced F1 | Pooled held-out F1 | Overall pair F1 |
| --- | ---: | ---: | ---: |
| **Fixed stage-wise composition** | **0.507538** | **0.582352** | 0.582950 |
| Frozen Qwen few-shot | 0.495398 | 0.563566 | 0.539356 |
| QLoRA | 0.486437 | 0.521724 | **0.588388** |

| Hypothesis | Registered comparison | Difference | 95% interval | Decision |
| --- | --- | ---: | --- | --- |
| H1 | composition minus few-shot | **+0.012140** | **[+0.001407, +0.022439]** | superiority passed |
| H2 | composition minus QLoRA | **+0.021101** | **[+0.000184, +0.041118]** | H1 gate passed; superiority passed |

The composition therefore improves the unseen-aspect endpoint while retaining
most of QLoRA's strong complete-grid result. QLoRA remains slightly higher on
overall F1, which is not the pre-registered primary endpoint.

## 3. Complete Level-2 descriptive roster

All rows below are the same twelve-fold aspect-balanced means. `D` supplies the
frozen minimal definition to the held-out candidate; `N` supplies its canonical
name only. No `N`-specific threshold, model or composition was selected.

| Method | D held-out F1 | N held-out F1 | D-N | D overall F1 |
| --- | ---: | ---: | ---: | ---: |
| Fixed stage-wise composition | **0.507538** | **0.514898** | -0.007361 | 0.582950 |
| Frozen Qwen few-shot | 0.495398 | 0.514050 | -0.018653 | 0.539356 |
| QLoRA | 0.486437 | 0.434821 | **+0.051616** | **0.588388** |
| Frozen Qwen zero-shot | 0.467884 | 0.447550 | +0.020334 | 0.513759 |
| TF-IDF | 0.283620 | 0.232464 | +0.051156 | 0.338368 |
| Frozen E5 | 0.227157 | 0.240587 | -0.013429 | 0.333108 |
| DCWT | 0.142485 | 0.142733 | -0.000247 | 0.182338 |
| DistilBERT | 0.114802 | 0.102608 | +0.012194 | 0.477007 |

The description comparison is descriptive, not part of the confirmatory
family. It reinforces the validation conclusion that descriptions are a
model-dependent interface intervention: the final held-out effect is positive
for QLoRA and TF-IDF, smaller for zero-shot Qwen and DistilBERT, approximately
zero for DCWT, and negative for E5, few-shot and the composition.

## 4. Stage decomposition

| Method, Level-2 `D` | Presence AP | Thresholded presence F1 | Oracle-gated Stage-2 pair F1 | End-to-end held-out pair F1 |
| --- | ---: | ---: | ---: | ---: |
| Fixed stage-wise composition | 0.481819 | **0.559198** | **0.897727** | **0.507538** |
| Frozen Qwen few-shot | 0.481819 | **0.559198** | 0.780809 | 0.495398 |
| QLoRA | **0.570746** | 0.516239 | **0.897727** | 0.486437 |
| Frozen Qwen zero-shot | 0.399966 | 0.510158 | 0.857123 | 0.467884 |
| TF-IDF | 0.469687 | 0.418853 | 0.665728 | 0.283620 |
| Frozen E5 | 0.346619 | 0.304334 | 0.741075 | 0.227157 |
| DCWT | 0.148159 | 0.209580 | 0.669956 | 0.142485 |
| DistilBERT | 0.232568 | 0.137959 | 0.875928 | 0.114802 |

The mechanism observed on validation replicated. The composition exactly
inherits few-shot's thresholded Stage-1 gate and QLoRA's conditional sentiment
decoder. QLoRA still has higher presence AP, showing that ranking quality and
the frozen operating point are distinct. DistilBERT still has a strong oracle
Stage 2 but a weak gate, so its principal failure is unseen-aspect detection.

## 5. Fold heterogeneity

The composition is not uniformly best on every held-out aspect:

- versus few-shot: 7 folds improved, 2 tied and 3 worsened;
- versus QLoRA: 7 folds improved and 5 worsened;
- the largest composition gain over QLoRA occurs on `a06` (+0.329053), while
  the largest loss occurs on `a05` (-0.427978);
- the largest gain over few-shot occurs on `a02` (+0.140264), while the largest
  loss occurs on `a04` (-0.054201).

This heterogeneity is why the thesis should report the full fold table and the
synchronized review-cluster interval, rather than claiming universal
per-aspect dominance.

## 6. Validation-to-test transfer

| System | Validation primary F1 | Final held-out primary F1 | Test-validation |
| --- | ---: | ---: | ---: |
| Fixed stage-wise composition | 0.525931 | **0.507538** | -0.018393 |
| Frozen Qwen few-shot | 0.504883 | 0.495398 | -0.009485 |
| QLoRA | 0.479903 | 0.486437 | +0.006534 |

The validation ordering transferred unchanged. The composition's margin over
few-shot narrowed from +0.021048 to +0.012140 and its margin over QLoRA narrowed
from +0.046028 to +0.021101, but both remained positive under the locked
confirmatory analysis.

## 7. Level-4 structured shift

Level 4 remains descriptive because it contains only three non-exchangeable
parent groups.

| Method | Company brand | Staff support | Value | Three-group mean |
| --- | ---: | ---: | ---: | ---: |
| DistilBERT | 0.081395 | 0.210762 | 0.253145 | 0.181767 |
| **Frozen Qwen few-shot** | **0.504405** | **0.597496** | **0.599018** | **0.566973** |
| QLoRA | 0.454140 | 0.590837 | 0.580729 | 0.541902 |

Few-shot is higher on held-out pair F1 in all three groups, matching the
validation ordering. QLoRA is higher on overall full-grid pair F1 in all three
groups. These are structured stress-test observations, not population-level
claims about arbitrary taxonomy hierarchies.

## 8. Recorded local failure mode

One of the 603 aggregate metric records has an empty prediction set: Frozen
Qwen zero-shot, Level-2 `N`, fold `a02`, held-out partition. It misses all 121
gold held-out pairs in that partition. The corresponding complete score bundle
is not globally empty: it makes 3,569 predictions over the full grid and has
overall F1 0.515280. This is therefore retained as a genuine partition-local
generalisation failure, not treated as an infrastructure error and not rerun.
It does not involve a confirmatory system.

## 9. Thesis-safe interpretation

The result supports the following claims:

1. A fixed, validation-selected separation of the stronger thresholded aspect
   gate and the stronger conditional sentiment decoder transfers to a new
   review partition and passes both registered comparisons.
2. The scientific value is stage specialization, not a learned Router, MoE or
   jointly optimized ensemble.
3. True two-stage diagnostics remain useful because they identify whether an
   end-to-end failure arises at aspect detection or conditional sentiment.
4. Minimal descriptions do not have a universal effect; their direction and
   magnitude depend on the model and operating point.
5. The composition is best on the primary unseen-aspect endpoint, while QLoRA
   remains best on the complete-grid overall endpoint.

The result does not support universal superiority across aspects, datasets or
taxonomies; causal attribution of QLoRA versus prompting; a monotonic
Level-1-to-Level-4 difficulty ladder; or open-world aspect discovery.

## 10. Reproducible evidence

Tracked, review-text-free tables are under
`docs/thesis_figure_data/taxonomy_final_test_v1/`:

- `official_confirmatory_results.csv`;
- `official_l2_model_condition_summary.csv`;
- `official_l2_stage_diagnostics.csv`;
- `official_l2_confirmatory_fold_results.csv`;
- `official_l2_description_effects.csv`;
- `official_l4_group_results.csv`;
- `official_validation_test_comparison.csv`.

The reproducible post-reveal converter is
`scripts/analyse_taxonomy_final_test_results.py`. Heavy sealed scores, the
private label vault and row-level prediction sets remain ignored locally. The
complete independent backup is under
`C:/Msc_DSML/Msc_Project/cloud_backups/taxonomy_final_test_v1_6a0b42d_final_graph`.

No result in this report authorizes post-test threshold, prompt, model,
description, retriever, Router, calibrator or metric selection.
