# Strongest Local Non-LLM Open-Topic Baseline

Metric correction, 2026-07-17: historical one-aspect `aspect_micro_f1` values in the LOAO tables below included true-negative absent rows and are not positive-class candidate-presence F1. Corrected presence metrics are in `docs/experiments/loao_presence_metric_correction_20260717.md`; pair metrics and model selection are unchanged.

Last updated: 2026-07-01

This note records the current strongest local non-LLM baseline for the fixed held-out-aspect open-topic protocol. It is intended to be reusable in the dissertation methods/results sections.

## Claim

The strongest current local non-LLM open-topic baseline is:

```text
Candidate-aspect DistilBERT selector
+ DistilBERT aspect-conditioned sentiment classifier
```

On the fixed three-aspect held-out-aspect test set, the best validation-selected run reaches:

| Metric | Value |
| --- | ---: |
| Pair samples F1 | 0.6071 |
| Pair micro F1 | 0.5917 |
| Pair macro F1 | 0.4890 |
| Aspect samples F1 | 0.6651 |
| Sentiment accuracy when gold aspect is predicted | 0.9100 |

This is the best fixed held-out-aspect non-LLM result currently available in the repository. It is stronger than the earlier candidate-aspect DistilBERT selector with global TF-IDF sentiment (`0.5816` pair samples F1) and the Qwen indexed zero-shot baseline (`0.5374` pair samples F1). It should not be interpreted as a full leave-one-aspect-out robustness result; the completed LOAO check below shows that fixed-split strength does not transfer uniformly across all held-out aspects.

## Evaluation Setting

The evaluation is the fixed held-out-aspect protocol described in `docs/evaluation_protocol.md`.

Held-out aspects:

- `Account management: Account access`
- `Company brand: Competitor`
- `Value: Discounts promotions`

Training strategies:

- `label_masked`: training rows are retained but held-out aspect labels are removed from supervision.
- `example_filtered`: any training row containing a held-out aspect is removed.

The main reported result uses `example_filtered`, because it avoids censored-label noise and performed best for the strong candidate-aspect pipeline. Validation and test evaluation use only held-out aspect+sentiment labels, with the held-out candidate aspects provided at inference time.

Headline metric:

- Pair samples F1 over aspect+sentiment pair labels.

Supporting metrics:

- Pair micro F1.
- Pair macro F1.
- Aspect-only samples F1.
- Sentiment accuracy when the gold aspect is predicted.

## Method

The pipeline has two local supervised components.

### 1. Candidate-Aspect Selector

The selector is a DistilBERT cross-encoder. For each review and each candidate aspect, it scores whether the aspect is relevant:

```text
(review text, candidate aspect) -> relevance score
```

The selector is trained only on seen aspects. At evaluation time, it receives the held-out candidate aspect names and selects from those canonical candidates. It does not generate topic names.

Best selector configuration:

| Parameter | Value |
| --- | --- |
| Model | `distilbert-base-uncased` |
| Strategy | `example_filtered` |
| Learning rate | `3e-5` |
| Epochs | `3` |
| Best epoch | `1` |
| Batch size | `32` |
| Eval batch size | `96` |
| Negative samples per positive | `3` |
| Threshold | `0.37` |
| Selection metric | validation pair samples F1 |

### 2. Aspect-Conditioned Sentiment Classifier

The sentiment model is a second DistilBERT classifier trained on aspect-specific sentiment examples:

```text
(review text, candidate aspect) -> negative | neutral | positive
```

This replaces the earlier global sentiment approximation, where one document-level polarity was applied to every selected aspect.

Best sentiment configuration:

| Parameter | Value |
| --- | --- |
| Model | `distilbert-base-uncased` |
| Learning rate | `2e-5` |
| Epochs | `3` |
| Batch size | `16` |
| Eval batch size | `64` |
| Class weighting | balanced |
| Selection metric | validation accuracy |
| Max length | `256` |

The sentiment model is trained before the selector in the combined pipeline and is used to create sentiment predictions for selected candidate aspects.

## Main Results

### Controlled Lexical Selector Ablation

The lexical selector is intentionally weak, but it isolates the sentiment component because the aspect-selection output is unchanged.

| Sentiment Mode | Strategy | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: |
| Global TF-IDF sentiment | `label_masked` | 0.4698 | 0.4667 | 0.3703 | 0.8867 |
| Lightweight TF-IDF aspect-conditioned sentiment | `label_masked` | 0.4520 | 0.4491 | 0.3425 | 0.8533 |
| DistilBERT aspect-conditioned sentiment | `label_masked` | 0.4840 | 0.4807 | 0.4063 | 0.9133 |
| Global TF-IDF sentiment | `example_filtered` | 0.4626 | 0.4596 | 0.3703 | 0.8851 |
| Lightweight TF-IDF aspect-conditioned sentiment | `example_filtered` | 0.4389 | 0.4351 | 0.3422 | 0.8378 |
| DistilBERT aspect-conditioned sentiment | `example_filtered` | 0.4804 | 0.4772 | 0.3985 | 0.9189 |

Interpretation: the earlier weak TF-IDF aspect-conditioned sentiment result was limited by the sentiment model, not by the per-aspect formulation. DistilBERT aspect-conditioned sentiment improves the controlled setup over both global sentiment and shallow aspect-conditioned sentiment.

### Strong Candidate-Aspect Pipeline

| Model | Strategy | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Candidate-aspect DistilBERT + global TF-IDF sentiment | `label_masked` | 0.5595 | 0.5462 | 0.4267 | 0.6419 | not recorded in summary table |
| Candidate-aspect DistilBERT + global TF-IDF sentiment | `example_filtered` | 0.5816 | 0.5646 | 0.4538 | 0.6835 | not recorded in summary table |
| Candidate-aspect DistilBERT + DistilBERT aspect-conditioned sentiment | `label_masked` | 0.5412 | 0.5343 | 0.4713 | 0.6088 | 0.8947 |
| Candidate-aspect DistilBERT + DistilBERT aspect-conditioned sentiment | `example_filtered` | **0.6071** | **0.5917** | **0.4890** | 0.6651 | 0.9100 |

Interpretation: the stronger sentiment model improves the cleaner `example_filtered` candidate-aspect pipeline, but not the noisier `label_masked` setup. This is consistent with the incomplete-label-noise interpretation of label-masked training.

## Tuning Attempts

The following fixed-split checks did not beat the best run:

| Variant | Pair Samples F1 | Note |
| --- | ---: | --- |
| `example_filtered`, selector LR `2e-5`, 3 epochs | 0.6001 | Strong but below best |
| `example_filtered`, selector LR `3e-5`, 3 epochs | **0.6071** | Best current fixed-split result |
| `example_filtered`, selector LR `4e-5`, 3 epochs | 0.5614 | Degraded |
| `example_filtered`, selector LR `2e-5`, 5 epochs | 0.5325 | Degraded |
| `example_filtered`, selector LR `2e-5`, top-2 cap | 0.5949 | Slightly worse |
| `label_masked`, selector LR `3e-5`, 3 epochs | 0.4981 | Degraded |

This suggests the current fixed held-out-aspect non-LLM baseline is close enough to pause local tuning and move to report framing or the next LLM phase.

## LOAO Robustness Check

The strongest fixed-split local non-LLM branch was also run through the all-row leave-one-aspect-out held-out-aspect protocol. Each of the 12 FABSA aspects is held out in turn; validation/test keep all official rows; gold labels are filtered to the current held-out aspect only; and the candidate set contains only that held-out aspect. This is the main robustness diagnostic requested after Aji's 2026-06-21 feedback.

Preferred full LOAO run:

```text
Candidate-aspect DistilBERT selector
+ DistilBERT aspect-conditioned sentiment classifier
+ example_filtered training
+ all-row LOAO
+ validation pair micro F1 threshold selection
```

Mean test spread across 12 held-out aspects:

| Variant | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | Pair Macro F1 Mean | FP Rows / 100 Mean | FN Rows / 100 Mean | Aspect Micro F1 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selector LR `3e-5` | 0.0550 | 0.3128 | 0.3345 | 0.4315 | 0.2285 | 12.7022 | 8.6746 | 0.7862 | 0.9319 |
| Selector LR `2e-5` | 0.0577 | 0.2941 | 0.3021 | 0.3921 | 0.2250 | 13.2798 | 8.3176 | 0.7840 | 0.9301 |

The LR `3e-5` run remains the best DistilBERT LOAO setting by pair micro F1. However, it does not beat the previously recorded lexical micro-F1-selected all-row LOAO lower bound (`0.3780` mean pair micro F1 for `example_filtered` with global sentiment). This is not a sentiment failure: sentiment accuracy when the gold aspect is predicted is high, around `0.93`. The bottleneck is unseen-aspect relevance detection and threshold calibration when the held-out aspect rotates through rare, ambiguous, and less lexically transparent labels.

This means the local non-LLM story should be framed carefully:

- fixed three-aspect result: strong local candidate-label baseline, useful as the local system to compare with Qwen/Gemini
- LOAO result: robustness warning showing that the fixed split is not enough for a broad open-topic generalisation claim
- dissertation implication: the next modelling value lies in LLM-assisted candidate-label reasoning, candidate-label descriptions, qualitative error analysis, and later Qwen fine-tuning/evaluation, not more DistilBERT selector tuning

## Reproduction

Run the strongest current fixed held-out-aspect non-LLM baseline:

```powershell
python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered
```

Run the controlled lexical selector sentiment ablation:

```powershell
python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --output-dir .\outputs\baselines\generalisation_transformer_sentiment_lr2e-5_ep3_balanced_accuracy
```

Validation:

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

## Dissertation Use

This baseline should be described as a strong local non-LLM candidate-label baseline for open-topic generalisation. It is useful because it separates the open-topic problem into two interpretable parts:

1. Select unseen candidate aspects using label-aware text matching.
2. Assign aspect-specific sentiment only after candidate aspects are selected.

The main limitation is that the headline result is still a fixed three-aspect held-out evaluation. The completed all-row LOAO run shows that the same DistilBERT branch is not uniformly robust across all held-out aspects. Use the fixed result as the strongest local non-LLM baseline and the LOAO result as the robustness caveat.
