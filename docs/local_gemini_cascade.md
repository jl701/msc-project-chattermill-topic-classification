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

## Interpretation

The cascade is the strongest fixed held-out-aspect system result so far and is valuable for the dissertation because it links model quality to deployment constraints:

- Flash-Lite is useful as an ultra-cheap hosted fallback, but it is not strong enough as a full replacement for the local model.
- Flash is the practical hosted escalator: it gives a large improvement over local-only at substantially lower cost and latency than Pro.
- Pro is the upper hosted-quality escalator: it gives the best fixed-split F1, but its latency and cost make it more suitable for high-value or highly uncertain rows.
- The common winning pattern is `gemini_nonempty_else_local`, meaning Gemini is trusted when it produces a non-empty structured answer, while the local prediction protects against hosted abstentions.

This result should still be framed carefully. It is a fixed three-aspect held-out result, not LOAO robustness evidence. The uncertainty signal is also a validation-reliability proxy rather than a calibrated probability margin, because the archived local prediction files did not store aspect scores. A future local rerun could export aspect probabilities and add score-margin uncertainty features, but the current experiment is already a valid selective-deployment result because policy selection is based only on validation behaviour and then evaluated on held-out test rows.

## Current Recommendation

For the dissertation, report three hosted/local points:

1. Local-only baseline: cheapest and private, but limited fixed-split F1.
2. Flash cascade: practical accuracy/cost/latency trade-off.
3. Pro cascade: upper hosted-quality bound for selective escalation.

Do not run Gemini LOAO by default. The next useful experiments are candidate-aspect description support and qualitative error taxonomy, while LOAO should remain focused on the selected local branch unless a clear budget and dissertation-value argument is made.
