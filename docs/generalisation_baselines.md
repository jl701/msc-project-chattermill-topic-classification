# FABSA Generalisation Baselines

This note records the first baseline results for the new FABSA generalisation protocols.

Headline metric: **pair samples F1**. Pair micro F1 and pair macro F1 are reported alongside it.

## Summary

| Setting | Model | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Closed-topic provided split | Word+char TF-IDF + Linear SVM | 0.7090 | 0.7042 | 0.4207 | 0.7736 |
| Closed-topic provided split | DistilBERT | 0.7803 | 0.7738 | 0.5377 | 0.8185 |
| Held-out organisation | Word+char TF-IDF + Linear SVM | 0.7026 | 0.6920 | 0.3626 | 0.7629 |
| Held-out organisation | DistilBERT | 0.7575 | 0.7600 | 0.4035 | 0.7986 |
| Held-out aspect, label-masked | Candidate-label lexical TF-IDF + global sentiment | 0.4698 | 0.4667 | 0.3703 | 0.5302 |
| Held-out aspect, example-filtered | Candidate-label lexical TF-IDF + global sentiment | 0.4626 | 0.4596 | 0.3703 | 0.5231 |
| Held-out aspect, label-masked | Candidate-label lexical TF-IDF + aspect-conditioned sentiment | 0.4520 | 0.4491 | 0.3425 | 0.5302 |
| Held-out aspect, example-filtered | Candidate-label lexical TF-IDF + aspect-conditioned sentiment | 0.4389 | 0.4351 | 0.3422 | 0.5231 |
| Held-out aspect, label-masked | Candidate-aspect cross-encoder + global sentiment | 0.5595 | 0.5462 | 0.4267 | 0.6419 |
| Held-out aspect, example-filtered | Candidate-aspect cross-encoder + global sentiment | 0.5816 | 0.5646 | 0.4538 | 0.6835 |
| Held-out aspect | Qwen3-4B-Instruct indexed zero-shot | 0.5374 | 0.5300 | 0.4374 | 0.6340 |

## Held-Out Organisation

Protocol:

- Validation organisation: `600`
- Test organisations: `369`, `727`
- Train rows: 7,020
- Test rows: 2,021

Best validation-selected refined model:

```text
model: Linear SVM
features: word+character TF-IDF
word ngrams: 1-3
max features: 60,000 total
C: 0.2
class weight: balanced
decision threshold: -0.16
```

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.6414 | 0.6251 | 0.3141 | 0.6956 |
| test | 0.7026 | 0.6920 | 0.3626 | 0.7629 |

The refined search only slightly improved the default sweep, so the traditional cross-organisation baseline appears close to a plateau.

A BERT-style encoder baseline was then run on the same held-out-organisation protocol. This reuses the DistilBERT multi-label classifier from the closed-topic setting, but trains on the held-out-organisation training split and tunes thresholds on validation organisation `600`.

Validation-selected DistilBERT configuration:

```text
model: distilbert-base-uncased
max length: 256
batch size: 16
positive-class weighting: sqrt
learning rate: 6e-5
epochs run: 10
best epoch: 8
threshold: 0.44
```

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.7317 | 0.7157 | 0.3924 | 0.7623 |
| test | 0.7575 | 0.7600 | 0.4035 | 0.7986 |

Neighbouring learning rates were checked:

| Learning Rate | Best Validation Pair Samples F1 | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 |
| ---: | ---: | ---: | ---: | ---: |
| 4e-5 | 0.7170 | 0.7570 | 0.7541 | 0.4013 |
| 5e-5 | 0.7283 | 0.7565 | 0.7567 | 0.4046 |
| 6e-5 | 0.7317 | 0.7575 | 0.7600 | 0.4035 |
| 7e-5 | 0.7269 | 0.7594 | 0.7629 | 0.4106 |

The final reported model is selected by validation pair samples F1, so the `6e-5` run is used even though the `7e-5` run has a slightly higher test score.

## Held-Out Aspect

Protocol:

- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Evaluation label scope: held-out labels only.
- Candidate labels are provided at inference.

The first non-LLM baseline is intentionally simple:

1. Select candidate aspects using character TF-IDF similarity between the feedback text and canonical aspect names.
2. Predict one global sentiment from the feedback text using a TF-IDF Logistic Regression sentiment classifier trained on seen labels.
3. Combine selected aspect(s) with the predicted sentiment.

This is a lower-bound baseline. It can use candidate label names, but it cannot really reason over unseen topic definitions.

| Strategy | Train Rows | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Label-masked | 7,632 | 0.4698 | 0.4667 | 0.3703 | 0.5302 |
| Example-filtered | 6,495 | 0.4626 | 0.4596 | 0.3703 | 0.5231 |

The two strategies are close. Label-masked training is slightly stronger, which is expected because it keeps more training data.

Aji later flagged that the global sentiment step is not ideal because FABSA sentiment is aspect-level, not document-level. An aspect-conditioned sentiment variant was therefore added:

1. Select candidate aspects using the same lexical aspect selector.
2. Predict sentiment separately for each selected `(feedback text, candidate aspect)` pair.
3. Combine each selected aspect with its own predicted sentiment.

This is methodologically cleaner for mixed-sentiment reviews, but the current lightweight TF-IDF Logistic Regression implementation did not improve the fixed three-aspect lexical lower bound:

| Sentiment Mode | Strategy | Train Rows | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Global | Label-masked | 7,632 | 0.4698 | 0.4667 | 0.3703 | 0.8867 |
| Global | Example-filtered | 6,495 | 0.4626 | 0.4596 | 0.3703 | 0.8851 |
| Aspect-conditioned | Label-masked | 7,632 | 0.4520 | 0.4491 | 0.3425 | 0.8533 |
| Aspect-conditioned | Example-filtered | 6,495 | 0.4389 | 0.4351 | 0.3422 | 0.8378 |

The aspect-conditioned result should not be read as evidence against per-aspect sentiment modelling. It only shows that this simple bag-of-words/character TF-IDF sentiment classifier is weaker than the global sentiment prior on the current fixed held-out-aspect lexical setup. It remains the right direction for stronger models and for mixed-sentiment correctness.

A stronger label-aware non-LLM baseline was added after the lexical lower bound. It trains a DistilBERT cross-encoder to score `(feedback text, candidate aspect)` relevance using only seen-aspect supervision, then combines selected candidate aspects with a global TF-IDF Logistic Regression sentiment classifier. Evaluation remains held-out-only: validation/test targets include only labels for the held-out aspects.

The model consumes canonical candidate aspects at inference and does not generate topic names.

Best validation-selected configuration:

```text
aspect selector: distilbert-base-uncased cross-encoder
sentiment model: word+character TF-IDF Logistic Regression
learning rate: 2e-5
batch size: 32
negative samples per positive aspect: 3
epochs run: 3
best epoch: 2
```

| Strategy | Train Rows | Best Epoch | Threshold | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Label-masked | 7,632 | 2 | 0.13 | 0.5595 | 0.5462 | 0.4267 | 0.6419 |
| Example-filtered | 6,495 | 2 | 0.06 | 0.5816 | 0.5646 | 0.4538 | 0.6835 |

Small follow-up checks did not improve the validation-selected result:

- Pair-label cross-encoder over full aspect+sentiment candidate labels underperformed the lexical baseline.
- A top-1 aspect selection constraint reduced F1.
- A higher learning rate (`3e-5`) reduced validation/test performance for label-masked training.
- Heavier negative sampling (`5` negatives per positive) reduced performance for example-filtered training.

After Aji's 2026-06-21 feedback, a leave-one-aspect-out robustness evaluation was added. The full completed LOAO run is currently the lexical lower bound, with an all-row view that includes negative rows and a positive-row diagnostic that mirrors the older row scope. See `docs/loao_heldout_aspect.md` for the detailed protocol, spread tables, and interpretation.

All-row lexical LOAO test spread across 12 held-out aspects. With sample-F1 threshold selection, the headline sample-F1 is slightly higher but precision is very low because sample-F1 does not penalise false positives on empty-gold rows beyond assigning the same zero row score as a true-negative empty prediction.

| Selection | Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | FP Rows / 100 Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sample F1 | Label-masked | 0.1299 | 0.2304 | 0.1458 | 0.8857 | 77.5362 |
| Sample F1 | Example-filtered | 0.1288 | 0.2345 | 0.1495 | 0.8777 | 77.4207 |
| Micro F1, global sentiment | Label-masked | 0.0926 | 0.3635 | 0.3912 | 0.4883 | 18.5728 |
| Micro F1, global sentiment | Example-filtered | 0.0903 | 0.3780 | 0.3778 | 0.4895 | 17.4753 |
| Micro F1, aspect-conditioned sentiment | Label-masked | 0.0883 | 0.3511 | 0.3811 | 0.4700 | 18.5728 |
| Micro F1, aspect-conditioned sentiment | Example-filtered | 0.0842 | 0.3576 | 0.3627 | 0.4541 | 16.7034 |

Positive-row lexical LOAO is much higher, but it should be treated as a sentiment diagnostic rather than a true aspect-selection evaluation because each fold has only one candidate aspect and every evaluated row contains it.

## Qwen Held-Out Aspect Smoke Test

A zero-shot Qwen3-4B-Instruct smoke test was added for the held-out-aspect protocol. The key prompt change is to use indexed candidate labels:

```text
Candidate aspects:
A1. Account management: Account access
A2. Company brand: Competitor
A3. Value: Discounts promotions

Return a JSON array of objects with keys "aspect_id" and "sentiment".
```

This avoids brittle free-form copying of canonical aspect strings while still preventing the model from inventing topic names.

Prompt comparison on 50 sampled validation rows selected the plain `indexed` prompt over conservative and descriptive variants. Full-split indexed zero-shot results:

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Seconds / Example |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.5753 | 0.5762 | 0.4514 | 0.6382 | 1.0000 | 1.77 |
| test | 0.5374 | 0.5300 | 0.4374 | 0.6340 | 1.0000 | 1.74 |

This is a zero-shot prompt baseline, not a fine-tuned Qwen result. It is competitive enough to justify full Qwen fine-tuning once stronger GPU resources are available.

## Interpretation

The closed-topic DistilBERT result remains the strongest current benchmark on the provided split.

The held-out organisation traditional result is close to the closed-topic traditional baseline on pair samples F1, but pair macro F1 drops. The held-out-organisation DistilBERT run gives a clear improvement over the traditional model, but its macro F1 remains lower than closed-topic DistilBERT. This suggests that domain shift is still hurting long-tail labels even when overall performance is strong.

The held-out aspect results are much lower than the closed-topic and held-out-organisation results, as expected. A fixed-output supervised classifier is not a meaningful model for unseen labels. The candidate-aspect cross-encoder is a stronger first label-aware baseline and improves substantially over the lexical lower bound. Qwen indexed zero-shot is competitive but does not yet beat the best held-out-aspect test score. The global sentiment component has now been ablated against a lightweight aspect-conditioned sentiment model; the next non-LLM improvement should use a stronger aspect-conditioned sentiment classifier or move to joint aspect+sentiment pair scoring. Full Qwen candidate-label fine-tuning should still wait until GPU resources are clearer.

See `docs/heldout_aspect_error_analysis.md` for row-level error analysis.

## Reproduction

Run the default generalisation baselines:

```powershell
python .\scripts\run_generalisation_baselines.py
```

The current default held-out-aspect lexical sentiment mode is aspect-conditioned. To reproduce the original global-sentiment lexical lower bound, pass:

```powershell
python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --sentiment-mode global --output-dir .\outputs\baselines\generalisation_global_sentiment_rerun
```

Run the refined held-out organisation SVM grid:

```powershell
python .\scripts\run_generalisation_baselines.py --protocol heldout-org --refined-cross-org --output-dir .\outputs\baselines\generalisation_refined
```

Run the held-out-organisation DistilBERT baseline:

```powershell
python .\scripts\run_transformer_baseline.py --protocol heldout-org --epochs 10 --batch-size 16 --learning-rate 6e-5 --pos-weight sqrt --output-dir .\outputs\baselines\transformer_heldout_org\distilbert_lr6e-5_sqrt
```

Run the candidate-aspect cross-encoder held-out-aspect baseline:

```powershell
python .\scripts\run_aspect_label_aware_baseline.py --strategy both --sentiment-mode global --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_lr2e-5_ep3_neg3
```

Run the Qwen indexed zero-shot held-out-aspect smoke test:

```powershell
python .\scripts\run_qwen_heldout_aspect_smoke.py --split validation --limit 10000 --load-in-4bit --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_smoke\validation_indexed_full
python .\scripts\run_qwen_heldout_aspect_smoke.py --split test --limit 10000 --load-in-4bit --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_smoke\test_indexed_full
```

Prepare GPU-ready Qwen held-out-aspect SFT data:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy both --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_sft_indexed
```

Outputs are written under:

```text
outputs/baselines/
```

The output directory is ignored by Git.
