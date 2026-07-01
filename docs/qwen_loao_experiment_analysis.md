# Qwen LOAO Zero-Shot Robustness Analysis

Date: 2026-07-01

This note records the dissertation-oriented analysis of the completed local open-weight Qwen full leave-one-aspect-out (LOAO) zero-shot baseline. It compares Qwen against the strongest local DistilBERT LOAO result and separates true all-row open-topic robustness from positive-gold-row sentiment diagnostics and fixed held-out-aspect LLM evidence.

## Evidence Sources

No new model inference was run for this analysis pass. The analysis reuses completed local outputs:

- Qwen all-row validation LOAO: `outputs/llm/qwen_loao_heldout_aspect_all_rows_validation_20260701/`
- Qwen all-row test LOAO: `outputs/llm/qwen_loao_heldout_aspect_all_rows_test_20260701/`
- Qwen positive-gold-row diagnostic: `outputs/analysis/qwen_loao_positive_diagnostic_20260701/`
- DistilBERT LOAO, preferred LR `3e-5`: `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630/`
- DistilBERT LOAO, LR `2e-5` check: `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701/`

The derived comparison tables are written to ignored local output:

```text
outputs/analysis/qwen_loao_full_interpretation_20260701/
```

Reproduction command for the derived comparison:

```powershell
python .\scripts\analyse_qwen_loao_comparison.py --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --qwen-positive-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701 --distilbert-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630 --distilbert-lr2e5-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701 --output-dir .\outputs\analysis\qwen_loao_full_interpretation_20260701
```

Original Qwen generation commands:

```powershell
python .\scripts\run_qwen_loao_heldout_aspect.py --split validation --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701
python .\scripts\run_qwen_loao_heldout_aspect.py --split test --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701
python .\scripts\analyse_qwen_loao_predictions.py --validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701
```

## Protocol

Qwen configuration:

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Loading: local Transformers causal LM with default 4-bit bitsandbytes NF4 double quantisation.
- Prompt: indexed candidate-label prompt, JSON array output, one `aspect_id` per predicted label.
- Candidate set: exactly one held-out FABSA aspect per LOAO fold.
- Decoding: deterministic generation, `max_input_tokens=1024`, `max_new_tokens=192`.
- Hardware: local NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM.

LOAO evaluation configuration:

- Folds: all 12 FABSA aspects.
- Split scope: official validation and test rows.
- Main row scope: all rows in each split.
- Gold label scope: labels filtered to the held-out aspect only.
- Empty predictions: allowed and necessary, because most rows do not contain the held-out aspect.
- Aggregation: mean, standard deviation, min, max, and spread across held-out aspects.

This all-row LOAO protocol is the robustness evidence. A positive-gold-row diagnostic is also reported, but it should not be treated as open-topic robustness because it removes empty-gold rows and therefore largely removes the candidate absence decision.

## Headline Comparison

Test split, mean across the 12 held-out aspects:

| System | Split | Pair Samples Mean | Pair Micro Mean | Precision Mean | Recall Mean | Pair Macro Mean | FP Rows/100 Mean | FN Rows/100 Mean | Valid JSON | Schema Valid | Seconds/Example |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B zero-shot LOAO all-row | test | 0.1212 | 0.3378 | 0.2379 | 0.8182 | 0.2412 | 34.4150 | 1.9114 | 1.0000 | 0.9955 | 1.1184 |
| Qwen3-4B positive-gold diagnostic | test | 0.8194 | 0.8659 | 0.9338 | 0.8182 | 0.6184 | 0.0000 | 12.1174 | n/a | n/a | n/a |
| DistilBERT CE + DistilBERT sentiment LOAO LR3e-5 | test | 0.0550 | 0.3128 | 0.3345 | 0.4315 | 0.2285 | 12.7022 | 8.6746 | n/a | n/a | n/a |
| DistilBERT CE + DistilBERT sentiment LOAO LR2e-5 | test | 0.0577 | 0.2941 | 0.3021 | 0.3921 | 0.2250 | 13.2798 | 8.3176 | n/a | n/a | n/a |
| Documented lexical TF-IDF + global sentiment LOAO | test | 0.0903 | 0.3780 | 0.3778 | 0.4895 | n/a | 17.4753 | n/a | n/a | n/a | n/a |

Main reading:

- Qwen is the stronger open-weight zero-shot semantic matcher than the preferred DistilBERT LOAO row by mean pair samples F1 (`0.1212` vs `0.0550`) and mean pair micro F1 (`0.3378` vs `0.3128`).
- The gain is small on pair micro F1, only `+0.0250`, and it comes with a clear calibration cost.
- Qwen recall is much higher (`0.8182` vs `0.4315`), but precision is lower (`0.2379` vs `0.3345`).
- Qwen has many more empty-gold false-positive rows (`34.4150` vs `12.7022` per 100 reviews).
- DistilBERT is therefore more conservative, while Qwen is recall-oriented and over-selects the single candidate aspect.
- The documented lexical global-sentiment lower bound still has the highest mean pair micro F1 among these all-row LOAO rows (`0.3780`), which is useful negative evidence: stronger semantic models are not automatically better under all-row absence calibration.

## Per-Aspect Qwen Versus DistilBERT

Test split, sorted by Qwen minus DistilBERT pair micro F1:

| Held-Out Aspect | Qwen Micro | DistilBERT Micro | Delta | Qwen Precision | DistilBERT Precision | Qwen Recall | DistilBERT Recall | Qwen FP/100 | DistilBERT FP/100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Online experience: App website` | 0.6011 | 0.2918 | 0.3093 | 0.4969 | 0.6990 | 0.7604 | 0.1844 | 34.1525 | 3.0876 |
| `Company brand: General satisfaction` | 0.5446 | 0.2491 | 0.2956 | 0.4038 | 0.4605 | 0.8362 | 0.1707 | 44.1714 | 6.8053 |
| `Logistics rides: Speed` | 0.5660 | 0.3643 | 0.2017 | 0.4188 | 0.6125 | 0.8730 | 0.2593 | 14.3037 | 1.7643 |
| `Staff support: Attitude of staff` | 0.5685 | 0.4395 | 0.1290 | 0.4308 | 0.3512 | 0.8358 | 0.5871 | 13.6736 | 13.2325 |
| `Purchase booking experience: Ease of use` | 0.5469 | 0.4593 | 0.0876 | 0.3887 | 0.3345 | 0.9228 | 0.7327 | 45.0536 | 44.2344 |
| `Company brand: Competitor` | 0.2298 | 0.1785 | 0.0513 | 0.1547 | 0.1308 | 0.4463 | 0.2810 | 18.3995 | 14.0517 |
| `Value: Discounts promotions` | 0.1913 | 0.2099 | -0.0186 | 0.1079 | 0.1392 | 0.8427 | 0.4270 | 38.5003 | 14.6188 |
| `Company brand: Reviews` | 0.0513 | 0.1011 | -0.0498 | 0.0266 | 0.0549 | 0.7368 | 0.6316 | 64.2722 | 25.9609 |
| `Value: Price value for money` | 0.2679 | 0.3760 | -0.1081 | 0.1566 | 0.2787 | 0.9272 | 0.5777 | 64.0832 | 18.3365 |
| `Account management: Account access` | 0.1527 | 0.2807 | -0.1280 | 0.0849 | 0.2148 | 0.7595 | 0.4051 | 40.1386 | 7.3094 |
| `Staff support: Email` | 0.1144 | 0.2903 | -0.1759 | 0.0609 | 0.2250 | 0.9545 | 0.4091 | 20.3529 | 1.9534 |
| `Staff support: Phone` | 0.2188 | 0.5128 | -0.2940 | 0.1241 | 0.5128 | 0.9231 | 0.5128 | 15.8790 | 1.0712 |

The per-aspect comparison is not a simple Qwen win. Qwen beats DistilBERT on 6 aspects and loses on 6 aspects by pair micro F1.

Qwen's largest advantages are on broader or semantically transparent aspects:

- `Online experience: App website`: +0.3093 pair micro F1.
- `Company brand: General satisfaction`: +0.2956.
- `Logistics rides: Speed`: +0.2017.
- `Staff support: Attitude of staff`: +0.1290.

DistilBERT's largest advantages are on narrower support-channel, account, and value aspects:

- `Staff support: Phone`: Qwen -0.2940.
- `Staff support: Email`: Qwen -0.1759.
- `Account management: Account access`: Qwen -0.1280.
- `Value: Price value for money`: Qwen -0.1081.

This split is important for the dissertation. Qwen's instruction-following prior appears useful for broad semantic matching, but its zero-shot prompt is not calibrated enough to decide that a candidate aspect is absent. DistilBERT is less semantically adventurous and therefore often misses positives, but it avoids many empty-gold false positives.

## Positive-Gold Diagnostic

The positive-gold diagnostic filters the same Qwen predictions to rows where the held-out aspect is present. It does not make new model calls.

| Held-Out Aspect | All-Row Micro | Positive-Gold Micro | Gap | All-Row Precision | Positive-Gold Precision | All-Row FP/100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Staff support: Email` | 0.1144 | 0.9545 | 0.8401 | 0.0609 | 0.9545 | 20.3529 |
| `Company brand: Reviews` | 0.0513 | 0.7778 | 0.7265 | 0.0266 | 0.8235 | 64.2722 |
| `Staff support: Phone` | 0.2188 | 0.9351 | 0.7162 | 0.1241 | 0.9474 | 15.8790 |
| `Value: Discounts promotions` | 0.1913 | 0.8671 | 0.6757 | 0.1079 | 0.8929 | 38.5003 |
| `Value: Price value for money` | 0.2679 | 0.9340 | 0.6661 | 0.1566 | 0.9409 | 64.0832 |
| `Account management: Account access` | 0.1527 | 0.8054 | 0.6527 | 0.0849 | 0.8571 | 40.1386 |
| `Purchase booking experience: Ease of use` | 0.5469 | 0.9424 | 0.3954 | 0.3887 | 0.9628 | 45.0536 |
| `Company brand: Competitor` | 0.2298 | 0.6067 | 0.3770 | 0.1547 | 0.9474 | 18.3995 |
| `Logistics rides: Speed` | 0.5660 | 0.9270 | 0.3609 | 0.4188 | 0.9880 | 14.3037 |
| `Company brand: General satisfaction` | 0.5446 | 0.8981 | 0.3535 | 0.4038 | 0.9700 | 44.1714 |
| `Staff support: Attitude of staff` | 0.5685 | 0.8984 | 0.3299 | 0.4308 | 0.9711 | 13.6736 |
| `Online experience: App website` | 0.6011 | 0.8445 | 0.2435 | 0.4969 | 0.9496 | 34.1525 |

This diagnostic explains why the all-row result is low. Qwen's positive-gold test mean is strong:

- pair samples F1 mean: `0.8194`
- pair micro F1 mean: `0.8659`
- pair precision mean: `0.9338`
- pair recall mean: `0.8182`
- sentiment accuracy when the gold aspect is predicted: `0.9314`

The all-row failure is therefore not mainly a JSON, label mapping, or sentiment problem. It is a candidate absence problem: the model often predicts the single candidate aspect for rows where that aspect is not in the gold labels.

## Metric Interpretation

Pair samples F1 is retained for continuity with the rest of the project, but it is not sufficient for all-row LOAO. In a one-candidate fold, many rows are empty-gold rows. A true-negative empty prediction and a false-positive prediction on an empty-gold row can both contribute weakly at the row level, so pair samples F1 alone can hide over-prediction.

Pair micro F1 is the preferred all-row robustness comparison because it exposes the precision-recall trade-off at the label level. Pair macro F1 remains useful because it shows whether rare sentiment labels are handled, but it is unstable under sparse one-aspect folds.

Aspect micro F1 must be interpreted cautiously in one-candidate all-row LOAO. It can look high for conservative models because true-negative empty rows dominate the fold. For this reason, the central robustness claims should use pair micro F1, precision, recall, false-positive rows per 100 reviews, and per-aspect spread rather than aspect micro F1 alone.

The positive-gold diagnostic answers a different question: if the aspect is actually present, can the system recognise and sentiment-label it? For Qwen, the answer is mostly yes. The full all-row task adds the harder question: can it abstain when the candidate aspect is absent? For Qwen zero-shot, the answer is not yet reliable enough.

## Fixed Split And Gemini Boundaries

The fixed held-out-aspect results are still useful, but they should not be merged with full LOAO evidence.

| Evidence Type | System | Test Pair Samples F1 | Test Pair Micro F1 | Interpretation |
| --- | --- | ---: | ---: | --- |
| Fixed held-out aspect | Candidate-aspect DistilBERT + DistilBERT sentiment | 0.6071 | 0.5917 | Strongest local non-LLM fixed-split baseline. |
| Fixed held-out aspect | Qwen3-4B indexed zero-shot | 0.5374 | 0.5300 | Competitive prompt baseline on the easier fixed split. |
| Fixed held-out aspect | Gemini 2.5 Flash indexed JSON-schema | 0.6071 | 0.6541 | Hosted fixed-split baseline; not LOAO robustness evidence. |
| Fixed held-out aspect cascade | Local -> Gemini 2.5 Pro | 0.8102 | 0.7955 | Strongest fixed-split selective-deployment result; not LOAO robustness evidence. |
| Full all-row LOAO | Qwen3-4B indexed zero-shot | 0.1212 | 0.3378 | Open-weight zero-shot robustness baseline before Qwen fine-tuning. |
| Full all-row LOAO | DistilBERT CE + DistilBERT sentiment | 0.0550 | 0.3128 | Strong fixed local model does not transfer uniformly to full LOAO. |

The drop from fixed Qwen (`0.5300` pair micro F1) to full all-row LOAO (`0.3378`) is expected because the fixed split is narrower and easier. The full LOAO result is the better evidence for open-topic taxonomy-shift robustness.

Gemini fixed/cascade results should be discussed as hosted fixed-split and selective-deployment evidence. They can motivate future hosted LOAO diagnostics, but they do not currently prove full LOAO robustness.

## Article Contribution

This result contributes to the final dissertation in several concrete ways:

1. It establishes a local open-weight LLM zero-shot LOAO baseline before any Qwen fine-tuning. This gives the later Qwen SFT/QLoRA phase a clean pre-training comparator.
2. It shows that fixed held-out-aspect success overstates open-topic robustness. Qwen and DistilBERT both look much stronger on fixed held-out-aspect evidence than under full all-row LOAO.
3. It separates semantic recognition from absence calibration. Qwen is strong on positive rows but weak on empty-gold abstention.
4. It clarifies the DistilBERT/Qwen trade-off. DistilBERT is conservative and misses positives; Qwen is recall-oriented and over-predicts.
5. It supports the dissertation claim that taxonomy-shift FABSA is not just sentiment classification. The central bottleneck is candidate aspect relevance under missing-topic conditions.
6. It justifies keeping multiple metrics. Pair samples F1, pair micro F1, pair macro F1, precision/recall, false-positive rows, and positive-row diagnostics each reveal different failure modes.
7. It provides evidence for an absence-aware fine-tuning or calibration objective for Qwen, rather than treating zero-shot prompting as sufficient.
8. It gives a careful boundary for hosted LLM claims: Gemini fixed/cascade results are strong deployment evidence, but they are not LOAO robustness evidence unless a separate LOAO run or sampled diagnostic is performed.

Suggested dissertation wording:

```text
The local Qwen zero-shot LOAO baseline improved mean pair micro F1 over the preferred DistilBERT LOAO row (0.338 vs 0.313), but the improvement came from much higher recall rather than better calibration. Qwen reached 0.866 mean pair micro F1 on positive-gold rows, yet only 0.338 in the all-row setting because it frequently predicted the held-out candidate aspect for reviews where that aspect was absent. This indicates that open-weight LLM prompting provides useful semantic matching under taxonomy shift, but robust open-topic FABSA also requires absence-aware calibration or fine-tuning.
```

## Limitations

- The Qwen run uses a single zero-shot prompt variant, `indexed`, selected from earlier fixed held-out-aspect prompt checks. It is not a prompt-sweep over all possible absence-calibrated wording.
- Each LOAO fold supplies exactly one candidate aspect. This is useful for isolating held-out-aspect relevance, but a production open-topic system may need to score many candidate aspects at once.
- The lexical lower-bound row is included from the documented project record, not regenerated in this local analysis directory. Its raw output directory is not present in the current workspace snapshot.
- Qwen uses local 4-bit quantised inference. Quantisation is necessary for the 8 GB laptop GPU, but it may slightly differ from full-precision or larger-GPU inference.
- Runtime and latency are local hardware measurements and should not be generalised to hosted or server GPU deployment.
- Positive-gold diagnostics should not be reported as open-topic robustness; they are sentiment/relevance-on-present-aspect diagnostics only.

## Next Step

The next Qwen experiment should not simply rerun the same zero-shot LOAO. The useful next step is Qwen fine-tuning or calibration with the same indexed candidate-label representation and an explicit all-row LOAO evaluation after fixed-split checks.

Recommended priorities:

1. Keep this result as the zero-shot open-weight LOAO baseline.
2. Plan Qwen SFT/QLoRA with an absence-aware objective or evaluation threshold, because empty-gold false positives are the main zero-shot weakness.
3. Preserve all-row LOAO as the final robustness check; use positive-gold rows only as a diagnostic.
4. Consider candidate-aspect descriptions only if they directly target absence calibration or rare/narrow aspect disambiguation.
5. Do not run full Gemini LOAO by default; if needed, use a sampled LOAO diagnostic with clear cost and scope boundaries.
