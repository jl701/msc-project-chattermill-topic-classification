# Local-To-Gemini Uncertainty Cascade

Last updated: 2026-07-01

This note records the fixed held-out-aspect cascade experiment that routes uncertain local predictions to hosted Gemini. The aim is not only to maximise F1, but to test a practical deployment pattern: use a local model for routine rows and call a hosted LLM only when the local prediction looks unreliable.

## Protocol

Evaluation uses the same fixed held-out-aspect setup as the Qwen and Gemini candidate-label baselines:

- held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- evaluation label scope: held-out aspect+sentiment pair labels only
- evaluation row scope: rows containing at least one held-out aspect
- headline metric: pair samples F1
- supporting metrics: pair micro F1, pair macro F1, aspect samples F1

The local model is the strongest current non-LLM fixed held-out-aspect baseline:

```text
Candidate-aspect DistilBERT selector
+ DistilBERT aspect-conditioned sentiment
strategy: example_filtered
selector learning rate: 3e-5
selector epochs: 3
negative samples per positive: 3
sentiment learning rate: 2e-5
sentiment epochs: 3
```

The Gemini escalators are full validation/test fixed-split hosted runs using the same indexed JSON-schema prompt:

- `vertex_ai/gemini-2.5-flash-lite`
- `vertex_ai/gemini-2.5-flash`
- `vertex_ai/gemini-2.5-pro`

All runs use `response_format=json_schema`, `temperature=0`, and `max_tokens=2048`.

## Uncertainty Policy Search

The historical local prediction files do not store calibrated aspect probabilities or margins, so the cascade uses validation-derived reliability features computed from local validation predictions only. The test labels are not used for selecting policies.

Local uncertainty features include:

- number of predicted labels
- multi-prediction indicator
- sentiment indicators
- lowest/mean validation precision and F1 among the predicted pair labels
- lowest/mean validation precision and F1 among the predicted aspects
- lowest/mean validation precision and F1 among the predicted sentiments
- trigger features for predicted pair labels, aspects, and sentiments

For each Gemini escalator, the sweep evaluated `10,578` validation-selected candidate policies:

- no escalation and full escalation controls
- ranked escalation from 0% to 100% in 1 percentage point increments
- exact validation-observed feature thresholds
- trigger-any and trigger-all policies over one or two predicted labels/aspects/sentiments
- weighted rank policies over pair/sentiment/aspect reliability features
- combination modes:
  - `replace`
  - `gemini_nonempty_else_local`
  - `union`
  - `intersection`
  - `agreement_or_gemini`
  - `agreement_or_local`

The selected headline policy is always the best validation policy by pair samples F1, then pair micro F1, then pair macro F1, with lower call rate as a final tie-breaker. Budget rows are selected by taking the best validation policy whose Gemini call rate is below the stated budget.

## Commands

Full hosted Gemini runs:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-flash-lite --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40 --output-dir .\outputs\llm\gemini_candidate_label_20260701_034545_flash_lite_fixed_full

python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-pro --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 1.25 --output-cost-per-1m 10.00 --output-dir .\outputs\llm\gemini_candidate_label_20260701_031040_pro_fixed_full
```

The Flash full run had already been completed:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full
```

Cascade sweeps:

```powershell
python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_034545_flash_lite_fixed_full --input-cost-per-1m 0.10 --output-cost-per-1m 0.40 --output-dir .\outputs\analysis\local_gemini_cascade_flash_lite_grid1 --rank-rate-step 1

python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full --input-cost-per-1m 0.30 --output-cost-per-1m 2.50 --output-dir .\outputs\analysis\local_gemini_cascade_flash_grid1 --rank-rate-step 1

python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_031040_pro_fixed_full --input-cost-per-1m 1.25 --output-cost-per-1m 10.00 --output-dir .\outputs\analysis\local_gemini_cascade_pro_grid1 --rank-rate-step 1
```

Pro cascade deep-dive:

```powershell
python .\scripts\analyse_local_gemini_cascade.py
```

Generated prediction and analysis outputs remain ignored under `outputs/` because they can contain review text.

## Full Hosted Baselines

Approximate costs use public Gemini API Standard rates from the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing), checked on 2026-07-01. Output tokens include thinking tokens. The early Flash run did not store cost rates, so its cost below was recomputed from the stored token counts.

| Model | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 | Valid JSON | Schema Valid | Mean Latency | Test Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT pipeline | 0.6071 | 0.5917 | 0.4890 | 0.6651 | n/a | n/a | local | n/a |
| Gemini 2.5 Flash-Lite full | 0.5516 | 0.5872 | 0.4876 | 0.6062 | 1.0000 | 0.9964 | 0.372 s | $0.0096 |
| Gemini 2.5 Flash full | 0.6071 | 0.6541 | 0.5547 | 0.6747 | 1.0000 | 1.0000 | 2.372 s | $0.2996 |
| Gemini 2.5 Pro full | 0.7141 | 0.7425 | 0.6287 | 0.7746 | 1.0000 | 1.0000 | 5.651 s | $1.7140 |

Full Pro removes the earlier 50-row subset caveat. It is clearly stronger than Flash on the fixed split, but also slower and more expensive.

## Validation-Selected Cascades

| Escalator | Validation-Selected Policy | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 | Gemini Calls | Call Rate | Test Gemini Cost | Mean Called Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None, local only | n/a | 0.6071 | 0.5917 | 0.4890 | 0.6651 | 0 | 0.0000 | $0.0000 | n/a |
| Flash-Lite | weighted pair+sentiment reliability, 51%, Gemini non-empty else local | 0.6679 | 0.6579 | 0.5401 | 0.7259 | 143 | 0.5089 | $0.0052 | 0.348 s |
| Flash | low minimum pair precision, 90%, Gemini non-empty else local | 0.7459 | 0.7348 | 0.6223 | 0.7993 | 253 | 0.9004 | $0.2789 | 2.437 s |
| Pro | low minimum pair precision, 90%, Gemini non-empty else local | 0.8102 | 0.7955 | 0.6809 | 0.8493 | 253 | 0.9004 | $1.5421 | 5.648 s |

The main dissertation result is that selective escalation dominates either model family alone on the fixed split. The best local-only and Flash-only headline test score is `0.6071`, while Flash cascade reaches `0.7459` and Pro cascade reaches `0.8102`.

## Why The Pro Cascade Beats Pure Pro

The Pro cascade does not beat pure Pro because the local model is globally stronger. Pure Pro is still stronger than the local model on average: Pro reaches `0.7141` pair samples F1 versus local `0.6071`. The cascade wins because their errors are complementary and because the selected combination mode is `gemini_nonempty_else_local`.

The validation-selected Pro policy escalates `253 / 281` test rows and keeps the local prediction for `28` rows. Pro is trusted when it returns a non-empty structured answer. If Pro returns no labels, or if the row was not selected for escalation, the cascade preserves the local prediction.

| System | Pair Samples F1 | Pair Micro F1 | TP | FP | FN | Empty Prediction Rows | Exact Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local only | 0.6071 | 0.5917 | 192 | 168 | 97 | 0 | 144 |
| Pure Pro | 0.7141 | 0.7425 | 222 | 87 | 67 | 28 | 167 |
| Local -> Pro cascade | 0.8102 | 0.7955 | 249 | 88 | 40 | 0 | 194 |

The largest pure-Pro failure mode is abstention. Pure Pro has `28` empty prediction rows on a test set where every evaluated row contains at least one held-out-aspect label. The cascade recovers all `28` of those rows with local predictions. On the Pro-empty rows, Pro has mean row sample F1 `0.0000`, while the local fallback and cascade both have `0.6905`.

Row-level comparison confirms that the cascade gain is broad rather than a single-label artefact:

- Cascade is better than pure Pro on `29` rows, equal on `250`, and worse on only `2`.
- All `29` cascade-over-Pro gains come from preserving or falling back to the local prediction.
- `20` of the `29` gains are directly from Pro-empty rows.
- The cascade reduces pure-Pro pair false negatives from `67` to `40`, while keeping false positives almost unchanged (`87` to `88`).
- Exact-match rows increase from `167` for pure Pro to `194` for the cascade.

The main label-level gain is recall on `Company brand: Competitor`. At aspect level, cascade improves this aspect F1 from `0.7407` to `0.8487`, with `21` fewer false negatives and only `1` additional false positive. At pair level, the largest improvements are:

| Pair Label | Pro F1 | Cascade F1 | Cascade - Pro F1 | FP Delta | FN Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Account management: Account access | neutral` | 0.7179 | 0.8696 | +0.1516 | +1 | -6 |
| `Value: Discounts promotions | neutral` | 0.6154 | 0.7500 | +0.1346 | +1 | -2 |
| `Company brand: Competitor | positive` | 0.6933 | 0.8263 | +0.1330 | 0 | -17 |
| `Company brand: Competitor | negative` | 0.7619 | 0.8358 | +0.0739 | 0 | -4 |

There is a small cost: `Account management: Account access` aspect F1 drops from `0.8432` under pure Pro to `0.8168` under the cascade because the cascade adds `6` false positives on that aspect. This is outweighed by the recall gain on competitor labels and by recovering Pro-empty rows.

For the dissertation, the defensible interpretation is that the cascade is an error-complementarity result. Pro supplies stronger semantic generalisation for most uncertain local rows, while the local model acts as a deterministic safety net for hosted abstention and a small subset of rows where its fixed-label classifier is more reliable. The analysis uses test labels only after the validation-selected policy has been fixed, so it explains the result but does not select the headline policy.

## Budgeted Test Diagnostics

The table below reports test metrics for the best validation policy at selected call-rate budgets. These rows are diagnostics for cost/performance trade-offs. They should not replace the validation-selected headline policy.

| Escalator | Budget | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 | Calls | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Lite | 10% | 0.6225 | 0.6060 | 0.4885 | 0.6770 | 28 | $0.0010 |
| Flash-Lite | 20% | 0.6511 | 0.6365 | 0.5155 | 0.7038 | 48 | $0.0019 |
| Flash-Lite | 50% | 0.6684 | 0.6547 | 0.5371 | 0.7300 | 118 | $0.0044 |
| Flash-Lite | 80% | 0.6679 | 0.6579 | 0.5401 | 0.7259 | 143 | $0.0052 |
| Flash | 10% | 0.6439 | 0.6271 | 0.5232 | 0.7001 | 28 | $0.0329 |
| Flash | 20% | 0.6636 | 0.6507 | 0.5399 | 0.7216 | 56 | $0.0760 |
| Flash | 50% | 0.7163 | 0.7038 | 0.5910 | 0.7689 | 135 | $0.1632 |
| Flash | 80% | 0.7495 | 0.7384 | 0.6175 | 0.8064 | 222 | $0.2438 |
| Flash | 90% | 0.7530 | 0.7416 | 0.6254 | 0.8100 | 247 | $0.2701 |
| Pro | 10% | 0.6427 | 0.6273 | 0.5228 | 0.7007 | 28 | $0.1738 |
| Pro | 20% | 0.6795 | 0.6646 | 0.5517 | 0.7304 | 51 | $0.3108 |
| Pro | 50% | 0.7515 | 0.7368 | 0.6314 | 0.7923 | 140 | $0.8390 |
| Pro | 80% | 0.8149 | 0.8013 | 0.6760 | 0.8517 | 222 | $1.3461 |
| Pro | 90% | 0.8114 | 0.7981 | 0.6821 | 0.8517 | 244 | $1.4956 |

The 80% Pro diagnostic has the highest observed test pair samples F1 (`0.8149`), but it is reported as a budget row rather than the headline result. The headline result remains the validation-selected best policy, which does not inspect test labels.

## Score/Margin Uncertainty Check

Completed on 2026-07-02 as a pre-Qwen-LoRA-full-LOAO methodological check.

The original strongest local prediction files did not store candidate-aspect probabilities or margins, and the local output directory did not contain a saved candidate-aspect selector checkpoint. To make a fair score/margin check possible without new Gemini calls, the local fixed held-out-aspect baseline was rerun with the same documented configuration and the prediction writer was extended to export aggregate score features:

- per-candidate aspect score;
- selected threshold;
- top score;
- second score;
- top-minus-second score margin;
- distance to threshold;
- count above threshold;
- selected-score minimum and mean.

The rerun reproduced the documented strongest local fixed held-out-aspect result:

| Metric | Value |
| --- | ---: |
| Validation pair samples F1 | 0.6226 |
| Validation pair micro F1 | 0.6049 |
| Validation-selected threshold | 0.37 |
| Test pair samples F1 | 0.6071 |
| Test pair micro F1 | 0.5917 |
| Test pair macro F1 | 0.4890 |

The cascade sweep was then rerun against the existing cached Gemini prediction files. No Gemini API calls were made. The expanded feature set increased the candidate policy count from `10,578` to `20,598`, but validation selection did not choose a score/margin policy as the headline.

| Escalator | Selected Policy | Test Pair Samples F1 | Test Pair Micro F1 | Call Rate | Test Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| Flash-Lite | weighted pair+sentiment reliability, 51%, Gemini non-empty else local | 0.6679 | 0.6579 | 0.5089 | $0.0052 |
| Flash | low minimum pair precision, 90%, Gemini non-empty else local | 0.7459 | 0.7348 | 0.9004 | $0.2789 |
| Pro | low minimum pair precision, 90%, Gemini non-empty else local | 0.8102 | 0.7955 | 0.9004 | $1.5421 |

This is a negative methodological result rather than a new headline improvement. The model-score features are now available for future local/cascade runs, but on the current fixed held-out-aspect data they do not improve validation-selected F1 or reduce the high Flash/Pro call rate compared with the validation-reliability proxy.

Commands:

```powershell
python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702

python .\scripts\run_local_gemini_cascade.py --local-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702\example_filtered --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_034545_flash_lite_fixed_full --input-cost-per-1m 0.10 --output-cost-per-1m 0.40 --output-dir .\outputs\analysis\local_gemini_cascade_flash_lite_score_margin_20260702 --rank-rate-step 1

python .\scripts\run_local_gemini_cascade.py --local-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702\example_filtered --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full --input-cost-per-1m 0.30 --output-cost-per-1m 2.50 --output-dir .\outputs\analysis\local_gemini_cascade_flash_score_margin_20260702 --rank-rate-step 1

python .\scripts\run_local_gemini_cascade.py --local-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702\example_filtered --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_031040_pro_fixed_full --input-cost-per-1m 1.25 --output-cost-per-1m 10.00 --output-dir .\outputs\analysis\local_gemini_cascade_pro_score_margin_20260702 --rank-rate-step 1
```

## Interpretation

The cascade is the strongest fixed held-out-aspect system result so far and is valuable for the dissertation because it links model quality to deployment constraints:

- Flash-Lite is useful as an ultra-cheap hosted fallback, but it is not strong enough as a full replacement for the local model.
- Flash is the practical hosted escalator: it gives a large improvement over local-only at substantially lower cost and latency than Pro.
- Pro is the upper hosted-quality escalator: it gives the best fixed-split F1, but its latency and cost make it more suitable for high-value or highly uncertain rows.
- The common winning pattern is `gemini_nonempty_else_local`, meaning Gemini is trusted when it produces a non-empty structured answer, while the local prediction protects against hosted abstentions.

This result should still be framed carefully. It is a fixed three-aspect held-out result, not LOAO robustness evidence. The score/margin check above confirms that the original validation-reliability proxy remains the defensible headline uncertainty signal for the current cascade because adding local selector score features did not improve validation-selected cascade performance.

## Current Recommendation

For the dissertation, report three hosted/local points:

1. Local-only baseline: cheapest and private, but limited fixed-split F1.
2. Flash cascade: practical accuracy/cost/latency trade-off.
3. Pro cascade: upper hosted-quality bound for selective escalation.

Do not run Gemini LOAO by default. The next useful experiments are candidate-aspect description support and qualitative error taxonomy, while LOAO should remain focused on the selected local branch unless a clear budget and dissertation-value argument is made.
