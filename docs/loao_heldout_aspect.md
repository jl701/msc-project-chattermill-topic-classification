# Leave-One-Aspect-Out Held-Out Aspect Evaluation

This note records the leave-one-aspect-out (LOAO) held-out-aspect experiment added after Aji's 2026-06-21 feedback.

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

The completed full LOAO baseline is:

- `candidate_label_lexical_tfidf_with_global_sentiment`
- aspect relevance: character TF-IDF similarity between review text and the canonical aspect label
- sentiment: one global TF-IDF Logistic Regression sentiment prediction per review
- strategies: `label_masked` and `example_filtered`

The DistilBERT candidate-aspect cross-encoder path was implemented and smoke-tested, but full LOAO was not completed on the current local machine. A full single-fold smoke attempt did not finish within 30 minutes on the GTX 1660 Ti Max-Q 6 GB GPU. A tiny 100-train/100-eval/1-epoch smoke test completed, so the code path works, but full cross-encoder LOAO should run later on stronger compute or with a deliberately lighter model.

## Main All-Row Results

Output directory for sample-F1-selected all-row LOAO:

```text
outputs/baselines/loao_heldout_aspect_lexical_all_rows/
```

Output directory for micro-F1-selected all-row LOAO:

```text
outputs/baselines/loao_heldout_aspect_lexical_all_rows_micro_selection/
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

These high scores should not be interpreted as solving open-topic selection. Because each fold has one candidate aspect and only positive rows, aspect selection is trivial. The result mainly shows that the global sentiment classifier is often able to assign the majority sentiment correctly once the held-out aspect is already known to be present.

## Interpretation

The all-row LOAO result confirms Aji's concern: held-out-aspect performance swings strongly depending on which aspect is held out. On the lexical lower bound, sample-F1-selected test pair samples F1 ranges from about `0.0126` to about `0.3930` under `label_masked`.

The added diagnostics reveal an important threshold-selection issue. If validation threshold selection optimises sample-F1 in all-row LOAO, the model often over-predicts because false positives on empty-gold rows do not lower sample-F1 any more than true-negative empty predictions. Under `label_masked`, sample-F1 selection gives high recall (`0.8857`) but very low precision (`0.1458`) and about `77.5` false-positive rows per 100 reviews. Selecting thresholds by validation pair micro F1 gives a more balanced detector: precision `0.3912`, recall `0.4883`, pair micro F1 `0.3635`, and about `18.6` false-positive rows per 100 reviews.

The current lexical candidate selector is much better on common or lexically obvious aspects such as `Online experience: App website`, `Company brand: General satisfaction`, and `Purchase booking experience: Ease of use`. It is weak on rare or less lexically transparent aspects such as `Company brand: Reviews`, `Staff support: Phone`, and several lower-frequency labels.

The gap between the all-row view and the positive-row diagnostic is the important methodological point:

- positive-row LOAO mostly measures sentiment after the aspect is assumed present
- all-row LOAO exposes the actual unseen-aspect candidate selection problem

The `label_masked` and `example_filtered` lexical results are very close. This does not remove Aji's concern about false-negative noise in `label_masked`; it only means this simple lexical lower bound is not very sensitive to that training-data difference.

## Reproduction

Run the main all-row lexical LOAO:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows
```

Run the preferred all-row detection diagnostic with micro-F1 threshold selection:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows_micro_selection
```

Run the positive-row diagnostic:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --eval-row-scope containing_heldout --ensure-one --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_positive_rows
```

Run a tiny cross-encoder smoke test:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy label_masked --heldout-aspect "Staff support: Email" --epochs 1 --batch-size 16 --eval-batch-size 32 --learning-rate 2e-5 --negatives-per-positive 1 --train-limit 100 --eval-limit 100 --output-dir .\outputs\baselines\loao_cross_encoder_tiny_smoke
```

## Next Step

The next methodological step is not more lexical tuning. The next useful improvement is to replace the global sentiment component with either:

- aspect-conditioned sentiment: `(review, candidate aspect) -> sentiment`
- joint pair scoring: `(review, candidate aspect + sentiment) -> applicable / not applicable`

This should be done after the LOAO protocol is stable.
