# Taxonomy two-stage scientific-freeze results

Date: 2026-08-22

Status: **PASS — validation-only scientific freeze**

Formal execution boundary: `aa84212976a652d62cfca31ed8bf0516a216c485`

Official test: **sealed and unopened** (`include_official_test=false`, `test_contract_count=0`).

## What is primary

The primary Level 2 estimand is the unweighted mean of the 12 held-out-aspect pair micro-F1 values. Uncertainty uses 20,000 synchronized `row_uid` review-cluster bootstrap draws (seed 13), recomputing F1 inside each draw. The pooled held-out score is a sensitivity estimand. Intervals condition on the observed trained realization of each neural system and the fixed 12-aspect taxonomy.

## Primary Level 2 minimal-description comparison

| System | N F1 | D F1 | D−N | Review-bootstrap 95% CI | Pooled N | Pooled D | Pooled D−N | Better/tie/worse folds | Fold-sign sensitivity p | Holm-adjusted sensitivity p |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.2108 | 0.2773 | +0.0665 | [0.0455, 0.0879] | 0.2940 | 0.3394 | +0.0454 | 9/0/3 | 0.0938 | 0.6562 |
| Frozen E5 | 0.2415 | 0.2222 | -0.0193 | [-0.0328, -0.0060] | 0.3210 | 0.3038 | -0.0171 | 6/0/6 | 0.3203 | 1.0000 |
| Description-to-weight transfer | 0.1443 | 0.1453 | +0.0010 | [-0.0033, 0.0052] | 0.1578 | 0.1586 | +0.0008 | 10/0/2 | 0.8457 | 1.0000 |
| DistilBERT | 0.1057 | 0.1054 | -0.0003 | [-0.0149, 0.0146] | 0.1107 | 0.1002 | -0.0105 | 6/0/6 | 0.9863 | 1.0000 |
| Frozen Qwen zero-shot | 0.4359 | 0.4513 | +0.0154 | [-0.0095, 0.0415] | 0.4805 | 0.4994 | +0.0189 | 6/0/6 | 0.7095 | 1.0000 |
| Frozen Qwen few-shot | 0.5019 | 0.5049 | +0.0030 | [-0.0231, 0.0308] | 0.5592 | 0.5646 | +0.0054 | 7/0/5 | 0.9194 | 1.0000 |
| QLoRA Qwen | 0.4420 | 0.4799 | +0.0379 | [0.0151, 0.0633] | 0.4669 | 0.5192 | +0.0523 | 8/0/4 | 0.3843 | 1.0000 |

The fold-sign calculation is an enumerated sensitivity diagnostic, not a design-exact randomization test. Holm adjustment applies to that explicitly labelled seven-method sensitivity family.

## Complete Level 2 evaluation views

| System | Card | Held-out pair F1 | Overall pair F1 | Seen pair F1 | Held-out presence AP | Oracle-gated sentiment-set F1 |
|---|---|---|---|---|---|---|
| Description-to-weight transfer | D | 0.1453 | 0.2152 | 0.2196 | 0.2201 | 0.6367 |
| Description-to-weight transfer | N | 0.1443 | 0.2149 | 0.2196 | 0.2127 | 0.6180 |
| DistilBERT | D | 0.1054 | 0.4999 | 0.5228 | 0.2381 | 0.8690 |
| DistilBERT | N | 0.1057 | 0.4969 | 0.5228 | 0.2099 | 0.8761 |
| Frozen E5 | D | 0.2222 | 0.3263 | 0.3271 | 0.3523 | 0.7647 |
| Frozen E5 | N | 0.2415 | 0.3281 | 0.3271 | 0.3453 | 0.7546 |
| Frozen Qwen zero-shot | D | 0.4513 | 0.4994 | 0.4991 | 0.3900 | 0.8678 |
| Frozen Qwen zero-shot | N | 0.4359 | 0.4981 | 0.4991 | 0.4438 | 0.8610 |
| Frozen Qwen few-shot | D | 0.5049 | 0.5462 | 0.5432 | 0.5059 | 0.7657 |
| Frozen Qwen few-shot | N | 0.5019 | 0.5456 | 0.5432 | 0.5817 | 0.8299 |
| QLoRA Qwen | D | 0.4799 | 0.5868 | 0.5931 | 0.5834 | 0.8854 |
| QLoRA Qwen | N | 0.4420 | 0.5842 | 0.5931 | 0.5339 | 0.8878 |
| TF-IDF | D | 0.2773 | 0.3346 | 0.3336 | 0.5013 | 0.6368 |
| TF-IDF | N | 0.2108 | 0.3317 | 0.3336 | 0.4148 | 0.6188 |

Held-out presence F1 is omitted from this display because it duplicates held-out aspect F1 by construction. Oracle-gated sentiment-set F1 is diagnostic: it supplies gold aspect gates and therefore is not end-to-end performance.
The companion `l2_fold_pair_error_diagnostics.csv` preserves every fold's TP, FP, FN, precision, recall, F1, error-row rates and prediction-set size for appendix analysis.

## Level 1 closed-taxonomy development reference

| System | Validation reviews | Pair micro-F1 | Pair macro-F1 | Aspect micro-F1 | Selected C | Threshold | Boundary | Interpretation |
|---|---|---|---|---|---|---|---|---|
| word+character TF-IDF OvR logistic regression | 1057 | 0.7051 | 0.4650 | 0.7595 | 4.00 | 0.5445 | validation_all_36_labels | closed-taxonomy development reference; not an unbiased generalisation-loss estimate |

## Training filtering and held-out support

| Fold | Held-out aspect | Original train | Filtered train | Removed train | Validation reviews | Positive review support |
|---|---|---|---|---|---|---|
| l2-a01 | Account management: Account access | 7930 | 7457 | 473 | 1057 | 81 |
| l2-a02 | Company brand: Competitor | 7930 | 7314 | 616 | 1057 | 86 |
| l2-a03 | Company brand: General satisfaction | 7930 | 5026 | 2904 | 1057 | 401 |
| l2-a04 | Company brand: Reviews | 7930 | 7753 | 177 | 1057 | 22 |
| l2-a05 | Logistics rides: Speed | 7930 | 6928 | 1002 | 1057 | 118 |
| l2-a06 | Online experience: App website | 7930 | 4330 | 3600 | 1057 | 468 |
| l2-a07 | Purchase booking experience: Ease of use | 7930 | 5566 | 2364 | 1057 | 335 |
| l2-a08 | Staff support: Attitude of staff | 7930 | 6859 | 1071 | 1057 | 125 |
| l2-a09 | Staff support: Email | 7930 | 7810 | 120 | 1057 | 13 |
| l2-a10 | Staff support: Phone | 7930 | 7749 | 181 | 1057 | 18 |
| l2-a11 | Value: Discounts promotions | 7930 | 7528 | 402 | 1057 | 51 |
| l2-a12 | Value: Price value for money | 7930 | 6915 | 1015 | 1057 | 128 |

## Selected rich-description confirmation

| System | Selected card | Folds | D held-out pair F1 | R held-out pair F1 | Held-out pair F1 R−D | R better | Tie | R worse | Presence AP R−D | Presence F1 R−D | Precision R−D | Recall R−D | FP rows/100 R−D | Overall pair F1 R−D | D exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen E5 | R2_positive_concat | 12 | 0.2222 | 0.2120 | -0.0102 | 4 | 0 | 8 | -0.0114 | -0.0144 | -0.0168 | +0.0094 | +1.94 | -0.0026 | True |
| TF-IDF | R2_positive_concat | 12 | 0.2773 | 0.2479 | -0.0294 | 2 | 0 | 10 | -0.0091 | -0.0405 | -0.0723 | +0.0409 | +4.62 | -0.0027 | True |

This is a bounded secondary confirmation after validation-only interface development, not a new primary endpoint.

## Architecture and decoder controls

| Study | System | Variant | Pair F1 | Reference | Delta |
|---|---|---|---|---|---|
| matched one-stage versus two-stage | TF-IDF | one_stage_independent_pairs | 0.2196 | one_stage_independent_pairs | +0.0000 |
| matched one-stage versus two-stage | TF-IDF | two_stage_argmax | 0.2780 | one_stage_independent_pairs | +0.0584 |
| matched one-stage versus two-stage | TF-IDF | two_stage_capped_two | 0.2773 | one_stage_independent_pairs | +0.0577 |
| capped-two versus top-one | TF-IDF | argmax | 0.2780 | argmax | +0.0000 |
| capped-two versus top-one | TF-IDF | capped_two_threshold | 0.2773 | argmax | -0.0008 |
| matched one-stage versus two-stage | Frozen E5 | one_stage_independent_pairs | 0.1860 | one_stage_independent_pairs | +0.0000 |
| matched one-stage versus two-stage | Frozen E5 | two_stage_argmax | 0.2246 | one_stage_independent_pairs | +0.0386 |
| matched one-stage versus two-stage | Frozen E5 | two_stage_capped_two | 0.2222 | one_stage_independent_pairs | +0.0362 |
| capped-two versus top-one | Frozen E5 | argmax | 0.2246 | argmax | +0.0000 |
| capped-two versus top-one | Frozen E5 | capped_two_threshold | 0.2222 | argmax | -0.0024 |

## Description-to-weight transfer generator controls

| Generator | Folds | N held-out F1 | D held-out F1 | D−N | D presence AP | D presence F1 | D FP rows/100 | Evidence role |
|---|---|---|---|---|---|---|---|---|
| cosine_barycentric_weight | 12 | 0.1414 | 0.1426 | +0.0013 | 0.2172 | 0.2251 | 70.42 | appendix generator diagnostic; no inferential p-value |
| kernel_ridge | 12 | 0.1443 | 0.1453 | +0.0010 | 0.2201 | 0.2368 | 58.02 | appendix generator diagnostic; no inferential p-value |
| mean_seen_weight | 12 | 0.1383 | 0.1396 | +0.0012 | 0.2080 | 0.2150 | 81.82 | appendix generator diagnostic; no inferential p-value |
| nearest_description_weight | 12 | 0.1572 | 0.1534 | -0.0038 | 0.2109 | 0.2453 | 59.44 | appendix generator diagnostic; no inferential p-value |

## Level 3 targeted robustness

| evidence_role | method_id | matching_set | condition | targeted_pair_count | heldout_pair_micro_f1_mean | heldout_pair_micro_f1_sd | heldout_presence_f1_mean | heldout_pair_gold_label_support | inferential_p_value_reported | outcome_blind_pair_selection |
|---|---|---|---|---|---|---|---|---|---|---|
| appendix_post_hoc_targeted_robustness | e5_base_v2 | alternate_cyclic_matching | DD | 6 | 0.2265 | 0.1296 | 0.3770 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | alternate_cyclic_matching | DN | 6 | 0.2408 | 0.1396 | 0.3920 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | alternate_cyclic_matching | ND | 6 | 0.2296 | 0.1419 | 0.3967 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | alternate_cyclic_matching | NN | 6 | 0.2403 | 0.1503 | 0.4016 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | alternate_cyclic_matching | RR | 6 | 0.2280 | 0.1386 | 0.3776 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | selected_canonical_matching | DD | 6 | 0.2544 | 0.1442 | 0.4229 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | selected_canonical_matching | DN | 6 | 0.2649 | 0.1635 | 0.4542 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | selected_canonical_matching | ND | 6 | 0.2666 | 0.1614 | 0.4178 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | selected_canonical_matching | NN | 6 | 0.2791 | 0.1721 | 0.4478 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | e5_base_v2 | selected_canonical_matching | RR | 6 | 0.2554 | 0.1366 | 0.4133 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | alternate_cyclic_matching | DD | 6 | 0.4846 | 0.1221 | 0.5969 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | alternate_cyclic_matching | DN | 6 | 0.4650 | 0.1396 | 0.5691 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | alternate_cyclic_matching | ND | 6 | 0.5165 | 0.1144 | 0.6295 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | alternate_cyclic_matching | NN | 6 | 0.4970 | 0.1334 | 0.5841 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | alternate_cyclic_matching | RR | 6 | 0.4343 | 0.1272 | 0.5575 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | selected_canonical_matching | DD | 6 | 0.4657 | 0.1086 | 0.5791 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | selected_canonical_matching | DN | 6 | 0.4801 | 0.0895 | 0.5755 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | selected_canonical_matching | ND | 6 | 0.4500 | 0.1237 | 0.5621 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | selected_canonical_matching | NN | 6 | 0.4689 | 0.1055 | 0.5523 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | frozen_qwen_candidate_pair | selected_canonical_matching | RR | 6 | 0.4275 | 0.1367 | 0.5500 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | alternate_cyclic_matching | DD | 6 | 0.2755 | 0.1008 | 0.5128 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | alternate_cyclic_matching | DN | 6 | 0.2216 | 0.1263 | 0.3989 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | alternate_cyclic_matching | ND | 6 | 0.2586 | 0.1270 | 0.5076 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | alternate_cyclic_matching | NN | 6 | 0.1876 | 0.1168 | 0.3429 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | alternate_cyclic_matching | RR | 6 | 0.2228 | 0.0861 | 0.4596 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | selected_canonical_matching | DD | 6 | 0.2743 | 0.1435 | 0.4864 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | selected_canonical_matching | DN | 6 | 0.2689 | 0.1407 | 0.4751 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | selected_canonical_matching | ND | 6 | 0.2473 | 0.1400 | 0.4410 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | selected_canonical_matching | NN | 6 | 0.2397 | 0.1328 | 0.4187 | 1858 | False | False |
| appendix_post_hoc_targeted_robustness | strict_train_only_tfidf | selected_canonical_matching | RR | 6 | 0.2414 | 0.1436 | 0.4641 | 1858 | False | False |

Level 3 is post-hoc and appendix-only. All 12 historical dual-unseen folds are audited so that the selected canonical matching is shown beside the alternate cyclic matching. Artifact payloads do not contain the complete row identities and held-out score hashes required for outcome-blind exact reuse, so no inferential p-values are attached.

## Level 4 parent-group stress tests

| System | Parent-group fold | Held-out group | Reviews | Positive support | Held-out pair F1 | 95% CI low | 95% CI high |
|---|---|---|---|---|---|---|---|
| DistilBERT | l4-g01 | Company brand | 1057 | 472 | 0.0693 | 0.0388 | 0.1016 |
| DistilBERT | l4-g02 | Staff support | 1057 | 137 | 0.2131 | 0.1419 | 0.2839 |
| DistilBERT | l4-g03 | Value | 1057 | 173 | 0.2463 | 0.2074 | 0.2843 |
| Frozen Qwen few-shot | l4-g01 | Company brand | 1057 | 472 | 0.5123 | 0.4776 | 0.5465 |
| Frozen Qwen few-shot | l4-g02 | Staff support | 1057 | 137 | 0.5805 | 0.5237 | 0.6333 |
| Frozen Qwen few-shot | l4-g03 | Value | 1057 | 173 | 0.5916 | 0.5337 | 0.6450 |
| QLoRA Qwen | l4-g01 | Company brand | 1057 | 472 | 0.4243 | 0.3921 | 0.4562 |
| QLoRA Qwen | l4-g02 | Staff support | 1057 | 137 | 0.5423 | 0.4899 | 0.5928 |
| QLoRA Qwen | l4-g03 | Value | 1057 | 173 | 0.5789 | 0.5358 | 0.6200 |

Parent groups are non-exchangeable stress cases and are reported separately; no cross-group significance claim is made.

## Registered negative findings

| Finding | System | Estimate | Evidence role |
|---|---|---|---|
| Minimal description did not improve the aspect-balanced held-out pair F1 | Frozen E5 | -0.0193 | primary L2 matched contrast |
| Minimal description did not improve the aspect-balanced held-out pair F1 | DistilBERT | -0.0003 | primary L2 matched contrast |
| Selected rich card did not improve over the minimal description | Frozen E5 | -0.0102 | secondary confirmation |
| Selected rich card did not improve over the minimal description | TF-IDF | -0.0294 | secondary confirmation |
| Dataset-faithful capped-two decoder had a small F1 cost | TF-IDF | -0.0008 | sequential validation decoder control |
| Dataset-faithful capped-two decoder had a small F1 cost | Frozen E5 | -0.0024 | sequential validation decoder control |
| Kernel-ridge description-to-weight transfer did not outperform its strongest simple generator control | Description-to-weight transfer | -0.0081 | appendix generator diagnostic versus nearest_description_weight |

## Claim boundaries

- `D−N` is matched within method and fold, but for trainable methods it also includes train–inference representation match.
- Cross-method contrasts compare complete systems; QLoRA versus frozen Qwen is not a causal LoRA-only estimate.
- Frozen few-shot versus zero-shot changes demonstrations, length/truncation and system-specific operating points.
- L1 is a conventional development reference, not an unbiased taxonomy-generalisation-loss estimator.
- Single-seed neural results remain a disclosed limitation.
- No result in this record uses or opens the official test labels.

## Freeze provenance

Manifest schema: `taxonomy_scientific_freeze_manifest_v1`

- Audited inputs: 35
- Audited outputs: 26
- Audit files: 10
