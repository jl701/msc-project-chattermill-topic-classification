# Training-policy sensitivity results

Post-hoc validation-only evidence. Official test was not reopened.

| Method | Training policy | Held-out pair F1 | Overall pair F1 |
|---|---|---:|---:|
| distilbert | label_masked_all_reviews | 0.1061 | 0.4782 |
| distilbert | review_filtered | 0.1122 | 0.4967 |
| fewshot | label_masked_all_reviews | 0.4730 | 0.5306 |
| fewshot | review_filtered | 0.5052 | 0.5463 |
| mixed_component_diagnostic | label_masked_all_reviews | 0.5005 | 0.5729 |
| mixed_component_diagnostic | review_filtered | 0.5259 | 0.5841 |
| tfidf | label_masked_all_reviews | 0.2482 | 0.3318 |
| tfidf | review_filtered | 0.2773 | 0.3346 |

All twelve folds, both N/D conditions and all three TF-IDF size-control seeds are reported in the attached aggregate tables.

The mixed component diagnostic retains the old review-filtered QLoRA sentiment decoder. It is not a fully label-masked-trained system.

Intervals are nominal conditional review-cluster intervals for a post-hoc sensitivity study. They do not cover training-seed or taxonomy-population uncertainty.

DistilBERT uses paired local replays with matched head initialisation. Frozen few-shot uses exact-source canonical replay. Neither replay overwrites historical formal results.

No new prompts, thresholds, models or subset seeds were selected using held-out outcomes. No QLoRA retraining was performed.
