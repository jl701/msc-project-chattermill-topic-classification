# Gemini Fixed Held-Out-Aspect Error Analysis

Last updated: 2026-07-01

This note analyses the completed Gemini 2.5 Flash fixed held-out-aspect run. The evaluated task remains the fixed three-aspect candidate-label protocol, not leave-one-aspect-out robustness.

No review text is quoted below. Qualitative examples use only row IDs plus gold/predicted labels because the local prediction JSONL files under `outputs/` contain review text and are ignored by Git.

## Source Outputs

Analysed local ignored outputs:

- Gemini Flash: `outputs/llm/gemini_candidate_label_20260701_0145_fixed_full/`
- Gemini Pro subset: `outputs/llm/gemini_candidate_label_20260701_025042_pro_test50/`
- Local non-LLM comparator: `outputs/baselines/aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered/example_filtered/best_test_predictions.jsonl`
- Qwen comparator: `outputs/qwen_heldout_aspect_smoke/test_indexed_full/predictions_indexed.jsonl`

Reproduction command:

```powershell
python .\scripts\analyse_gemini_heldout_aspect_errors.py --output-dir .\outputs\analysis\gemini_error_analysis
python .\scripts\analyse_gemini_heldout_aspect_errors.py --pro-predictions .\outputs\llm\gemini_candidate_label_20260701_025042_pro_test50\predictions_test_indexed.jsonl --output-dir .\outputs\analysis\gemini_error_analysis_with_pro
```

The generated analysis summary is intentionally left under ignored `outputs/analysis/gemini_error_analysis/summary.json`.

## Headline Comparison

| Model | Pair Samples F1 | Pair Micro P | Pair Micro R | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Exact Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT selector + DistilBERT aspect-conditioned sentiment | 0.6071 | 0.5333 | 0.6644 | 0.5917 | 0.4890 | 0.6651 | 144 |
| Qwen3-4B indexed zero-shot | 0.5374 | 0.4870 | 0.5813 | 0.5300 | 0.4374 | 0.6340 | 125 |
| Gemini 2.5 Flash indexed JSON-schema | 0.6071 | 0.6475 | 0.6609 | 0.6541 | 0.5547 | 0.6747 | 138 |

Gemini and the local DistilBERT pipeline have the same pair samples F1 on the test split, but the equality is not evidence that they make the same errors. Gemini is much more precise at pair level, while the local pipeline has slightly higher recall and more exact rows. Their row-level gains cancel under samples F1, but Gemini's better precision improves pair micro and macro F1.

## Prediction Cardinality

| Model | Gold Labels / Row | Pred Labels / Row | Empty Prediction Rows | More-Than-Gold Rows | Fewer-Than-Gold Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT selector + DistilBERT aspect-conditioned sentiment | 1.0285 | 1.2811 | 0 | 50 | 4 |
| Qwen3-4B indexed zero-shot | 1.0285 | 1.2278 | 15 | 56 | 21 |
| Gemini 2.5 Flash indexed JSON-schema | 1.0285 | 1.0498 | 42 | 50 | 46 |

This explains the metric split. The local DistilBERT pipeline is recall-oriented because it enforces at least one prediction per row and predicts more labels overall. Gemini is more selective: it predicts almost the same number of labels as the gold set overall, but it also returns empty predictions on 42 rows. That selectivity lifts precision and macro F1, while missed rows keep samples F1 from moving above the local headline result.

## Error Categories

Rows can belong to more than one category.

| Model | Exact Rows | Missed-All Rows | Aspect Miss Rows | Aspect Over-Predict Rows | Sentiment Error Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT selector + DistilBERT aspect-conditioned sentiment | 144 | 0 | 78 | 119 | 18 |
| Qwen3-4B indexed zero-shot | 125 | 15 | 86 | 117 | 35 |
| Gemini 2.5 Flash indexed JSON-schema | 138 | 42 | 78 | 80 | 20 |

Gemini has the same number of aspect-miss rows as the local DistilBERT comparator, but far fewer aspect over-predictions. Its main new weakness is missed-all rows, mostly where the gold label is `Company brand: Competitor`.

## Gemini Aspect-Level Behaviour

| Aspect | Support | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Account management: Account access | 79 | 123 | 0.6260 | 0.9747 | 0.7624 |
| Company brand: Competitor | 121 | 64 | 0.8906 | 0.4711 | 0.6162 |
| Value: Discounts promotions | 89 | 108 | 0.7130 | 0.8652 | 0.7817 |

Gemini is very recall-heavy for `Account management: Account access` and `Value: Discounts promotions`, but highly conservative for `Company brand: Competitor`. This conservatism is the single clearest aspect-level failure: competitor labels have high precision but low recall.

## Gemini Sentiment-Level Behaviour

| Sentiment | Support | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| negative | 113 | 140 | 0.6714 | 0.8319 | 0.7431 |
| neutral | 34 | 19 | 0.9474 | 0.5294 | 0.6792 |
| positive | 142 | 136 | 0.5809 | 0.5563 | 0.5683 |

Negative labels are the strongest sentiment band. Neutral precision is high but recall remains weak because neutral labels are rare and often subtle. Positive labels are the broadest source of pair-level confusion, especially when competitor praise is missed or when promotional/value language is over-selected.

## Pair-Level Notes

Important Gemini pair patterns:

- `Company brand: Competitor | positive`: support 93, precision 0.9143, recall 0.3441, F1 0.5000.
- `Account management: Account access | positive`: support 6, precision 0.1765, recall 1.0000, F1 0.3000.
- `Value: Discounts promotions | positive`: support 43, precision 0.6119, recall 0.9535, F1 0.7455.
- `Value: Discounts promotions | neutral`: support 9, precision 1.0000, recall 0.3333, F1 0.5000.

The main dissertation-relevant story is therefore not simply "Gemini is better". Gemini is better calibrated than Qwen zero-shot and has stronger pair micro/macro F1 than the local fixed-split baseline, but it misses many positive competitor mentions and still struggles with rare neutral sentiment.

## Safe Qualitative Examples

These examples intentionally omit review text.

| Pattern | Row ID | Gold | Gemini Prediction |
| --- | --- | --- | --- |
| Extra aspect on an otherwise correct competitor row | `test:301982009` | `Company brand: Competitor | positive` | `Account management: Account access | positive`; `Company brand: Competitor | positive` |
| Competitor positive missed as account access | `test:301972578` | `Company brand: Competitor | positive` | `Account management: Account access | positive` |
| Neutral discounts shifted to positive | `test:327098247` | `Value: Discounts promotions | neutral` | `Value: Discounts promotions | positive` |
| Competitor positive missed entirely | `test:301984923` | `Company brand: Competitor | positive` | empty prediction |
| Discounts negative correctly recovered where Qwen missed sentiment | `test:301988596` | `Value: Discounts promotions | negative` | `Value: Discounts promotions | negative` |

## Gemini Pro Small-Subset Result

After the user explicitly lifted the earlier environment-variable-only constraint, a 50-row Gemini Pro subset was run. The key was used only as a process-local environment variable and was not written to the repository, documentation, summary JSON, or prediction files.

Command:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-pro --split test --limit 50 --sample --seed 13 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_20260701_025042_pro_test50
```

For a fair comparison, Flash was evaluated on the same deterministic 50-row test subset by filtering the existing full Flash prediction file.

| Model / Subset | Rows | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Schema Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini Flash, test sample seed 13 | 50 | 0.6360 | 0.6731 | 0.4627 | 0.7560 | 1.0000 | 1.0000 |
| Gemini Pro, test sample seed 13 | 50 | 0.6933 | 0.7379 | 0.5250 | 0.7933 | 1.0000 | 1.0000 |

Row-level exactness on the same 50 rows:

| Comparison | Rows |
| --- | ---: |
| Both exact | 24 |
| Pro only exact | 6 |
| Flash only exact | 3 |
| Neither exact | 17 |

Pro therefore improves the same-subset row-level samples F1 by `+0.0573`, pair micro F1 by `+0.0648`, pair macro F1 by `+0.0623`, and aspect samples F1 by `+0.0373`.

Token, latency, and cost diagnostics:

| Model / Subset | Input Tokens | Output Tokens, Including Thinking | Reasoning Tokens | Total Tokens | Mean Latency | Approx Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini Flash, same 50 rows | 9,728 | 18,164 | 16,274 | 27,892 | 2.1751 s | $0.0483 |
| Gemini Pro, same 50 rows | 9,728 | 25,507 | 23,625 | 35,235 | 4.9026 s | $0.2672 |

Approximate cost uses public Gemini API Standard rates from the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing), checked on 2026-07-01. Flash uses `$0.30 / 1M input tokens` and `$2.50 / 1M output tokens`; Pro uses `$1.25 / 1M input tokens` and `$10.00 / 1M output tokens` for prompts up to 200k tokens. The endpoint's output-token count includes thinking tokens, so reasoning is counted through output-token billing and reported separately.

Decision: Pro is directionally stronger on this small fixed-subset comparison, but it is also about `2.25x` slower and `5.53x` more expensive than Flash on the same rows. This is enough to document Pro as a small upper hosted-LLM check. It is not enough by itself to justify full Pro validation/test or Pro LOAO without an explicit dissertation-value reason.

## Current Interpretation

Gemini Flash is a credible hosted fixed held-out-aspect baseline because it matches the strongest local non-LLM pair samples F1 and improves pair micro/macro F1 with perfect test schema validity. The error analysis qualifies that result: the gain is mostly better precision and fewer over-predicted aspects, not uniformly better aspect recall. A later full fixed-split Pro run confirmed that Pro is the strongest hosted model on this fixed protocol; see `docs/gemini_candidate_label_baseline.md` and `docs/experiment_log.md` for the full Pro and Flash-Lite Pareto table. The fixed three-aspect result should remain separate from the LOAO robustness claim.
