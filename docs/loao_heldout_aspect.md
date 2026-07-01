# Leave-One-Aspect-Out Held-Out Aspect Evaluation

Last updated: 2026-07-01

This note records the leave-one-aspect-out (LOAO) held-out-aspect experiment added after Aji's 2026-06-21 feedback.

The frozen cross-model protocol for future Qwen/Gemini/DistilBERT robustness work is `loao_open_topic_all_row_v1` in `docs/evaluation_protocol.md`. This document records the existing LOAO results and should be read as evidence under that frozen protocol.

## Motivation

Aji's main concern was that holding out a fixed set of three aspects can make the open-topic number fragile. If the selected held-out aspects are unusually common, rare, easy, or hard, the headline result may reflect the specific aspect choice rather than the general difficulty of unseen-aspect generalisation.

The LOAO experiment rotates each of the 12 FABSA aspects as the held-out aspect and reports the spread across folds.

## Evaluation Design

Two LOAO evaluation views are recorded.

### Main LOAO: All Evaluation Rows

This is the main robustness view.

- Hold out one aspect at a time.
- Train supervision excludes that aspect.
- Validation and test keep all official rows.
- Gold labels are filtered to the held-out aspect only.
- Rows without the held-out aspect therefore have empty gold labels.
- Candidate labels at inference contain only the current held-out aspect.
- Empty predictions are allowed.

This avoids the degenerate single-candidate case where the model is evaluated only on rows that are already known to contain the held-out aspect.

Important metric note: pair samples F1 does not reward true-negative empty/empty rows. It also does not distinguish a true-negative empty prediction from a false-positive prediction on an empty-gold row, because both receive row-level F1 `0`. For the all-row view, pair micro F1, precision, recall, and false-positive diagnostics should therefore be read alongside pair samples F1.

Two threshold-selection settings are useful:

- `pair_samples_f1` selection: preserves Aji's headline metric as the validation selection objective, but can choose over-predicting thresholds in all-row LOAO.
- `pair_micro_f1` selection: better for all-row detection diagnostics because false positives on empty-gold rows reduce micro precision and micro F1.

### Diagnostic LOAO: Positive Rows Only

This reproduces the older held-out-aspect row scope more closely.

- Validation and test include only rows containing the held-out aspect.
- Candidate labels contain only the current held-out aspect.
- At least one prediction per row is forced.

This view is useful as a sentiment-coupling diagnostic, but it is not a real aspect-selection test. Aspect F1 is trivially `1.0000` because every row contains the only candidate aspect.

## Baseline

The completed full LOAO lexical baselines are:

- `candidate_label_lexical_tfidf_with_global_sentiment`
- `candidate_label_lexical_tfidf_with_aspect_conditioned_sentiment`
- aspect relevance: character TF-IDF similarity between review text and the canonical aspect label
- global sentiment: one TF-IDF Logistic Regression sentiment prediction per review
- aspect-conditioned sentiment: one TF-IDF Logistic Regression sentiment prediction per `(review, candidate aspect)` pair
- strategies: `label_masked` and `example_filtered`

The DistilBERT candidate-aspect cross-encoder path is now complete for the strongest local non-LLM branch:

- aspect relevance: DistilBERT cross-encoder over `(review text, candidate aspect)`
- sentiment: DistilBERT aspect-conditioned classifier over `(review text, candidate aspect)`
- strategy: `example_filtered`
- evaluation: all rows, one held-out aspect per fold, validation threshold selected by pair micro F1

Two full 12-fold runs were completed on the RTX 5050 Laptop GPU:

- selector LR `3e-5`, 3 epochs, 3 negatives per positive
- selector LR `2e-5`, 3 epochs, 3 negatives per positive

The LR `3e-5` run is the preferred DistilBERT LOAO result by mean pair micro F1. This is different from the fixed three-aspect result: the fixed split reaches `0.6071` pair samples F1, but full LOAO exposes much larger aspect-to-aspect variation and a harder unseen-aspect detection problem.

## Main All-Row Results

Output directory for sample-F1-selected all-row LOAO:

```text
outputs/baselines/loao_heldout_aspect_lexical_all_rows/
```

Output directory for micro-F1-selected all-row LOAO:

```text
outputs/baselines/loao_heldout_aspect_lexical_all_rows_micro_selection/
```

Output directory for micro-F1-selected all-row LOAO with aspect-conditioned sentiment:

```text
outputs/baselines/loao_heldout_aspect_lexical_aspect_conditioned_micro_selection/
```

Output directories for micro-F1-selected all-row LOAO with the DistilBERT candidate-aspect selector and DistilBERT aspect-conditioned sentiment:

```text
outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630/
outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701/
```

Sample-F1-selected test spread across 12 held-out aspects:

| Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | FP Rows / 100 Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `label_masked` | 0.1299 | 0.2304 | 0.1458 | 0.8857 | 77.5362 |
| `example_filtered` | 0.1288 | 0.2345 | 0.1495 | 0.8777 | 77.4207 |

Micro-F1-selected test spread across 12 held-out aspects:

| Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | FP Rows / 100 Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `label_masked` | 0.0926 | 0.3635 | 0.3912 | 0.4883 | 18.5728 |
| `example_filtered` | 0.0903 | 0.3780 | 0.3778 | 0.4895 | 17.4753 |

Micro-F1 selection is the preferred all-row detection diagnostic. It trades lower sample-F1 and recall for much lower false-positive rates and substantially higher pair micro F1.

Aspect-conditioned sentiment was then tested with the same lexical aspect selector and the preferred micro-F1 threshold selection:

| Sentiment Mode | Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | FP Rows / 100 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global | `label_masked` | 0.0926 | 0.3635 | 0.3912 | 0.4883 | 18.5728 | 0.8867 |
| Global | `example_filtered` | 0.0903 | 0.3780 | 0.3778 | 0.4895 | 17.4753 | 0.8783 |
| Aspect-conditioned | `label_masked` | 0.0883 | 0.3511 | 0.3811 | 0.4700 | 18.5728 | 0.8592 |
| Aspect-conditioned | `example_filtered` | 0.0842 | 0.3576 | 0.3627 | 0.4541 | 16.7034 | 0.8390 |

This confirms that the lightweight aspect-conditioned sentiment classifier is not yet a performance improvement over the older global sentiment prior. It is still methodologically useful because it removes the document-level sentiment assumption that Aji queried, but the classifier itself needs to become stronger before this route is worth using as the headline non-LLM result.

The full DistilBERT candidate-aspect selector plus DistilBERT aspect-conditioned sentiment LOAO run was then completed for `example_filtered`. Unlike the fixed three-aspect setting, this all-row LOAO result is not stronger than the lexical micro-F1-selected lower bound:

| Model | Selector LR | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | Pair Macro F1 Mean | FP Rows / 100 Mean | FN Rows / 100 Mean | Aspect Micro F1 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical selector + global sentiment | n/a | 0.0903 | 0.3780 | 0.3778 | 0.4895 | n/a | 17.4753 | n/a | n/a | 0.8783 |
| Lexical selector + lightweight aspect-conditioned sentiment | n/a | 0.0842 | 0.3576 | 0.3627 | 0.4541 | n/a | 16.7034 | n/a | n/a | 0.8390 |
| DistilBERT selector + DistilBERT aspect-conditioned sentiment | `3e-5` | 0.0550 | 0.3128 | 0.3345 | 0.4315 | 0.2285 | 12.7022 | 8.6746 | 0.7862 | 0.9319 |
| DistilBERT selector + DistilBERT aspect-conditioned sentiment | `2e-5` | 0.0577 | 0.2941 | 0.3021 | 0.3921 | 0.2250 | 13.2798 | 8.3176 | 0.7840 | 0.9301 |

The LR `3e-5` DistilBERT run is therefore the strongest DistilBERT LOAO setting tried, but it is not the strongest all-row LOAO result. The sentiment component itself is strong, with mean sentiment accuracy above `0.93` when the gold aspect is predicted. The failure mode is mainly unseen-aspect detection and threshold calibration under taxonomy shift, not aspect-conditioned sentiment assignment.

Per-aspect test results for the preferred LR `3e-5` DistilBERT LOAO run:

| Held-Out Aspect | Pair Samples F1 | Pair Micro F1 | Pair Precision | Pair Recall | Threshold | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Staff support: Phone` | 0.0126 | 0.5128 | 0.5128 | 0.5128 | 0.36 | 3 |
| `Purchase booking experience: Ease of use` | 0.2327 | 0.4593 | 0.3345 | 0.7327 | 0.16 | 2 |
| `Staff support: Attitude of staff` | 0.0744 | 0.4395 | 0.3512 | 0.5871 | 0.15 | 1 |
| `Value: Price value for money` | 0.0750 | 0.3760 | 0.2787 | 0.5777 | 0.26 | 3 |
| `Logistics rides: Speed` | 0.0309 | 0.3643 | 0.6125 | 0.2593 | 0.31 | 1 |
| `Online experience: App website` | 0.0855 | 0.2918 | 0.6990 | 0.1844 | 0.05 | 3 |
| `Staff support: Email` | 0.0057 | 0.2903 | 0.2250 | 0.4091 | 0.82 | 1 |
| `Account management: Account access` | 0.0202 | 0.2807 | 0.2148 | 0.4051 | 0.12 | 3 |
| `Company brand: General satisfaction` | 0.0624 | 0.2491 | 0.4605 | 0.1707 | 0.05 | 2 |
| `Value: Discounts promotions` | 0.0239 | 0.2099 | 0.1392 | 0.4270 | 0.20 | 1 |
| `Company brand: Competitor` | 0.0214 | 0.1785 | 0.1308 | 0.2810 | 0.26 | 1 |
| `Company brand: Reviews` | 0.0151 | 0.1011 | 0.0549 | 0.6316 | 0.10 | 1 |

Sample-F1-selected per-aspect test results for `label_masked`:

| Held-Out Aspect | Positive Test Rows | Pair Samples F1 | Pair Precision | Pair Recall | FP Rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Staff support: Email` | 22 | 0.0126 | 0.2000 | 0.9091 | 4.9149 |
| `Company brand: Reviews` | 38 | 0.0208 | 0.0208 | 0.8684 | 97.6055 |
| `Staff support: Phone` | 39 | 0.0227 | 0.0227 | 0.9231 | 97.5425 |
| `Account management: Account access` | 79 | 0.0416 | 0.0416 | 0.8354 | 95.0221 |
| `Value: Discounts promotions` | 89 | 0.0473 | 0.0473 | 0.8427 | 94.3919 |
| `Company brand: Competitor` | 121 | 0.0693 | 0.0693 | 0.9091 | 92.3756 |
| `Logistics rides: Speed` | 189 | 0.1109 | 0.1109 | 0.9312 | 88.0907 |
| `Staff support: Attitude of staff` | 201 | 0.1128 | 0.1128 | 0.8905 | 87.3346 |
| `Value: Price value for money` | 206 | 0.1134 | 0.1134 | 0.8738 | 87.0195 |
| `Purchase booking experience: Ease of use` | 503 | 0.2863 | 0.2867 | 0.9010 | 68.3050 |
| `Company brand: General satisfaction` | 580 | 0.3277 | 0.3277 | 0.8966 | 63.4531 |
| `Online experience: App website` | 724 | 0.3930 | 0.3970 | 0.8479 | 54.3793 |

Per-aspect test results for `example_filtered` were very similar. The full table is saved locally in:

```text
outputs/baselines/loao_heldout_aspect_lexical_all_rows/test_results.csv
```

## Qwen Zero-Shot All-Row LOAO

The local open-weight LLM zero-shot robustness baseline was completed after the non-LLM LOAO runs. It uses the same indexed candidate-label formulation as the fixed held-out-aspect Qwen run, but rotates all 12 FABSA aspects.

Protocol:

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Loading: 4-bit bitsandbytes NF4 double quantisation.
- Prompt: `indexed`, one candidate aspect per fold, JSON array output using `aspect_id`.
- Evaluation: all official validation/test rows, gold labels filtered to the held-out aspect, empty predictions allowed.
- Output directories:
  - `outputs/llm/qwen_loao_heldout_aspect_all_rows_validation_20260701/`
  - `outputs/llm/qwen_loao_heldout_aspect_all_rows_test_20260701/`

Commands:

```powershell
python .\scripts\run_qwen_loao_heldout_aspect.py --split validation --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701
python .\scripts\run_qwen_loao_heldout_aspect.py --split test --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701
python .\scripts\analyse_qwen_loao_predictions.py --validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701
```

Aggregate all-row spread:

| Split | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | Pair Macro F1 Mean | FP Rows / 100 Mean | Valid JSON | Schema Valid | Seconds / Example |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.1194 | 0.3293 | 0.2310 | 0.8115 | 0.2340 | 34.7446 | 1.0000 | 0.9961 | 0.9756 |
| test | 0.1212 | 0.3378 | 0.2379 | 0.8182 | 0.2412 | 34.4150 | 1.0000 | 0.9955 | 1.1184 |

Test per-aspect results, sorted by pair micro F1:

| Held-Out Aspect | Pair Samples F1 | Pair Micro F1 | Precision | Recall | Pair Macro F1 | FP Rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Company brand: Reviews` | 0.0176 | 0.0513 | 0.0266 | 0.7368 | 0.0426 | 64.2722 |
| `Staff support: Email` | 0.0132 | 0.1144 | 0.0609 | 0.9545 | 0.0805 | 20.3529 |
| `Account management: Account access` | 0.0378 | 0.1527 | 0.0849 | 0.7595 | 0.1910 | 40.1386 |
| `Value: Discounts promotions` | 0.0473 | 0.1913 | 0.1079 | 0.8427 | 0.1733 | 38.5003 |
| `Staff support: Phone` | 0.0227 | 0.2188 | 0.1241 | 0.9231 | 0.1509 | 15.8790 |
| `Company brand: Competitor` | 0.0340 | 0.2298 | 0.1547 | 0.4463 | 0.1566 | 18.3995 |
| `Value: Price value for money` | 0.1204 | 0.2679 | 0.1566 | 0.9272 | 0.1837 | 64.0832 |
| `Company brand: General satisfaction` | 0.3056 | 0.5446 | 0.4038 | 0.8362 | 0.3317 | 44.1714 |
| `Purchase booking experience: Ease of use` | 0.2932 | 0.5469 | 0.3887 | 0.9228 | 0.3439 | 45.0536 |
| `Logistics rides: Speed` | 0.1040 | 0.5660 | 0.4188 | 0.8730 | 0.3684 | 14.3037 |
| `Staff support: Attitude of staff` | 0.1059 | 0.5685 | 0.4308 | 0.8358 | 0.3788 | 13.6736 |
| `Online experience: App website` | 0.3527 | 0.6011 | 0.4969 | 0.7604 | 0.4927 | 34.1525 |

The same prediction files were filtered to positive-gold rows as a diagnostic, without new model calls. On the test split, positive-gold rows reach `0.8194` mean pair samples F1, `0.8659` mean pair micro F1, `0.9338` mean precision, and `0.9314` mean sentiment accuracy when the gold aspect is predicted. This confirms that Qwen zero-shot can recognise and label present held-out aspects well, but over-predicts the candidate aspect on empty-gold rows.

Compared with the preferred DistilBERT LOAO row, Qwen has higher mean test pair samples F1 (`0.1212` vs `0.0550`) and slightly higher mean pair micro F1 (`0.3378` vs `0.3128`), but substantially lower precision and many more empty-gold false-positive rows (`34.4150` vs `12.7022` per 100 reviews). Qwen is therefore a stronger recall-oriented semantic matcher, while DistilBERT is more conservative.

The full cross-system interpretation and thesis-use framing are recorded in `docs/qwen_loao_experiment_analysis.md`.

## Positive-Row Diagnostic Results

Output directory:

```text
outputs/baselines/loao_heldout_aspect_lexical_positive_rows/
```

Test spread across 12 held-out aspects:

| Strategy | Pair Samples F1 Mean | Pair Samples F1 Std | Min | Max | Pair Micro F1 Mean | Pair Macro F1 Mean | Aspect Micro F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `label_masked` | 0.8870 | 0.0294 | 0.8354 | 0.9312 | 0.8868 | 0.6316 | 1.0000 |
| `example_filtered` | 0.8790 | 0.0254 | 0.8315 | 0.9231 | 0.8788 | 0.6272 | 1.0000 |

The same diagnostic with aspect-conditioned sentiment:

| Sentiment Mode | Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Macro F1 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: |
| Global | `label_masked` | 0.8870 | 0.8868 | 0.6316 | 0.8857 |
| Global | `example_filtered` | 0.8790 | 0.8788 | 0.6272 | 0.8777 |
| Aspect-conditioned | `label_masked` | 0.8647 | 0.8646 | 0.5924 | 0.8636 |
| Aspect-conditioned | `example_filtered` | 0.8442 | 0.8440 | 0.5777 | 0.8430 |

These high scores should not be interpreted as solving open-topic selection. Because each fold has one candidate aspect and only positive rows, aspect selection is trivial. The result mainly shows how often the sentiment classifier assigns the right polarity once the held-out aspect is already known to be present. Under this lightweight TF-IDF setup, aspect-conditioned sentiment is cleaner conceptually but weaker empirically than the global sentiment classifier.

## Interpretation

The all-row LOAO result confirms Aji's concern: held-out-aspect performance swings strongly depending on which aspect is held out. On the lexical lower bound, sample-F1-selected test pair samples F1 ranges from about `0.0126` to about `0.3930` under `label_masked`.

The added diagnostics reveal an important threshold-selection issue. If validation threshold selection optimises sample-F1 in all-row LOAO, the model often over-predicts because false positives on empty-gold rows do not lower sample-F1 any more than true-negative empty predictions. Under `label_masked`, sample-F1 selection gives high recall (`0.8857`) but very low precision (`0.1458`) and about `77.5` false-positive rows per 100 reviews. Selecting thresholds by validation pair micro F1 gives a more balanced detector: precision `0.3912`, recall `0.4883`, pair micro F1 `0.3635`, and about `18.6` false-positive rows per 100 reviews.

The current lexical candidate selector is much better on common or lexically obvious aspects such as `Online experience: App website`, `Company brand: General satisfaction`, and `Purchase booking experience: Ease of use`. It is weak on rare or less lexically transparent aspects such as `Company brand: Reviews`, `Staff support: Phone`, and several lower-frequency labels.

The gap between the all-row view and the positive-row diagnostic is the important methodological point:

- positive-row LOAO mostly measures sentiment after the aspect is assumed present
- all-row LOAO exposes the actual unseen-aspect candidate selection problem

The `label_masked` and `example_filtered` lexical results are very close. This does not remove Aji's concern about false-negative noise in `label_masked`; it only means this simple lexical lower bound is not very sensitive to that training-data difference.

The aspect-conditioned sentiment ablation clarifies the sentiment issue but does not improve the lexical LOAO baseline when implemented with a shallow TF-IDF classifier. The stronger DistilBERT aspect-conditioned sentiment model fixes much of that sentiment weakness, but the full DistilBERT LOAO result still does not improve the all-row robustness headline. This is useful negative evidence: stronger local encoders can perform very well on a fixed held-out-aspect split, yet still fail to generalise uniformly when every aspect becomes the unseen topic in turn.

The main bottleneck is now aspect relevance under taxonomy shift. The DistilBERT LOAO run has high mean aspect micro F1 because most rows are true negatives for a one-aspect fold, but pair micro F1 and per-aspect spread show that the model is unstable for rare, ambiguous, or less lexically transparent aspects. This supports presenting the local DistilBERT pipeline as the strongest fixed-split non-LLM baseline, while using LOAO as the robustness diagnostic that motivates LLM-assisted candidate-label reasoning and selective escalation.

## Reproduction

Run the main all-row lexical LOAO:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode global --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows
```

Run the preferred all-row detection diagnostic with micro-F1 threshold selection:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode global --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows_micro_selection
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_micro_selection
```

Run the positive-row diagnostic:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode global --eval-row-scope containing_heldout --ensure-one --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_positive_rows
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --eval-row-scope containing_heldout --ensure-one --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_positive_rows
```

Run a tiny cross-encoder smoke test:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy label_masked --heldout-aspect "Staff support: Email" --epochs 1 --batch-size 16 --eval-batch-size 32 --learning-rate 2e-5 --negatives-per-positive 1 --train-limit 100 --eval-limit 100 --output-dir .\outputs\baselines\loao_cross_encoder_tiny_smoke
```

Run the preferred full DistilBERT LOAO result:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_YYYYMMDD
```

Run the LR `2e-5` tuning check:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_YYYYMMDD
```

## Next Step

The strongest local non-LLM fixed-split baseline and the strongest DistilBERT LOAO robustness check are now both complete. More DistilBERT LOAO hyperparameter tuning is unlikely to change the dissertation story. The useful next options are:

- test candidate-aspect descriptions as label-representation support
- write a qualitative error taxonomy for the fixed held-out-aspect and LOAO failures
- use the hosted Gemini indexed candidate-label and local-to-Gemini cascade results as the next LLM-centred evidence block
- run Qwen fine-tuning/evaluation once stronger GPU access is available
