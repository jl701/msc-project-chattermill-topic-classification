# Qualitative Error Taxonomy

Last updated: 2026-07-02

This note records Task 5: a manually verified, Gemini-assisted qualitative error taxonomy across local DistilBERT, Qwen zero-shot, Gemini hosted baselines, aspect-description variants, and local-to-Gemini cascades. It is part of the pre-Qwen-full-LOAO completion roadmap in `docs/thesis_completion_roadmap.md`.

## Pre-Registered Configuration

### Objective

Convert the completed quantitative experiments into a thesis-facing explanation of why the systems fail differently. The aim is not to create a new leaderboard result, but to identify recurring error mechanisms that motivate later Qwen fine-tuning, calibration, and selective-deployment work.

### Source Artifacts

Tracked docs and local ignored outputs used as inputs:

- Local fixed held-out-aspect baseline:
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered/example_filtered/best_test_predictions.jsonl`
- Qwen fixed held-out-aspect baseline:
  - `outputs/qwen_heldout_aspect_smoke/test_indexed_full/predictions_indexed.jsonl`
- Gemini fixed held-out-aspect baselines:
  - Flash-Lite: `outputs/llm/gemini_candidate_label_20260701_034545_flash_lite_fixed_full/predictions_test_indexed.jsonl`
  - Flash: `outputs/llm/gemini_candidate_label_20260701_0145_fixed_full/predictions_test_indexed.jsonl`
  - Pro: `outputs/llm/gemini_candidate_label_20260701_031040_pro_fixed_full/predictions_test_indexed.jsonl`
- Gemini aspect-description variants:
  - Flash-Lite label descriptions: `outputs/llm/gemini_candidate_label_20260701_desc_flash_lite_test_full/predictions_test_indexed_generated_descriptions.jsonl`
  - Flash-Lite decision-boundary descriptions: `outputs/llm/gemini_candidate_label_20260701_desc_boundary_flash_lite_test_full/predictions_test_indexed_generated_descriptions.jsonl`
  - Flash label descriptions: `outputs/llm/gemini_candidate_label_20260701_desc_flash_test_full/predictions_test_indexed_generated_descriptions.jsonl`
- Local-to-Gemini cascade:
  - `outputs/analysis/local_gemini_cascade_pro_grid1/selected_test_predictions.jsonl`
  - `outputs/analysis/local_gemini_cascade_pro_deep_dive/summary.json`
- Qwen LOAO comparison:
  - `outputs/analysis/qwen_loao_full_interpretation_20260701/summary.json`
  - `outputs/analysis/qwen_loao_full_interpretation_20260701/qwen_vs_distilbert_per_aspect_test.csv`
  - `outputs/analysis/qwen_loao_full_interpretation_20260701/qwen_all_row_vs_positive_gold_test.csv`

### Local Analysis Parameters

- Split: fixed held-out-aspect `test` for row-level taxonomy packets.
- LOAO evidence: existing 12-aspect all-row and positive-gold summaries only; no new LOAO model calls.
- Automatic row-level tags:
  - `competitor_positive_miss`
  - `account_access_overprediction`
  - `discounts_value_boundary`
  - `neutral_under_recall`
  - `empty_abstention`
  - `local_overprediction`
  - `qwen_overprediction`
  - `pro_empty_cascade_recovery`
  - `description_precision_shift`
  - `description_recall_loss`
- Local output directory: `outputs/analysis/qualitative_error_taxonomy_20260702/`.
- Raw review text may appear only in ignored local outputs used for manual/Gemini-assisted analysis. Tracked documentation must omit raw review text.

### Gemini-Assisted Step

Gemini may be used to draft candidate taxonomy categories from a compact local packet of selected examples. The assistant output is treated as a draft only. Final categories in this document must be manually reviewed and phrased by the researcher.

Planned model:

- `vertex_ai/gemini-2.5-pro`

Prompt constraints:

- Use row IDs, gold/predicted labels, automatic tags, and short review text snippets only in local ignored prompts/outputs.
- Do not commit prompts, raw snippets, or Gemini raw taxonomy drafts if they contain review text.
- Do not use Gemini as an automatic evaluator or source of final truth.

### Stopping Rule

Run enough analyses to cover:

1. local versus Gemini fixed-split differences;
2. pure Pro versus Pro cascade complementarity;
3. Flash-Lite/Flash description trade-offs;
4. Qwen LOAO calibration and aspect-boundary failures;
5. implications for Qwen fine-tuning.

Stop when additional categories are redundant or unsupported by available evidence.

## Status

Completed on 2026-07-02. This is the preferred non-major task before Qwen LoRA full LOAO because it uses existing outputs and directly strengthens the thesis discussion chapter.

## Implementation And Outputs

Implemented:

- `scripts/analyse_qualitative_error_taxonomy.py`
- `tests/test_qualitative_error_taxonomy.py`

Local ignored outputs:

- `outputs/analysis/qualitative_error_taxonomy_20260702/summary.json`
- `outputs/analysis/qualitative_error_taxonomy_20260702/category_counts.csv`
- `outputs/analysis/qualitative_error_taxonomy_20260702/fixed_split_cases_no_text.jsonl`
- `outputs/analysis/qualitative_error_taxonomy_20260702/fixed_split_cases_with_text.jsonl`
- `outputs/analysis/qualitative_error_taxonomy_20260702/gemini_taxonomy_prompt.md`
- `outputs/analysis/qualitative_error_taxonomy_20260702/gemini_taxonomy_draft.json`

The `with_text` file, Gemini prompt, and Gemini draft are local-only because they may include review text or snippets. This tracked document intentionally omits raw review text.

The Gemini-assisted draft used `vertex_ai/gemini-2.5-pro` with `response_format=json_schema`, `temperature=0`, and `max_tokens=7000`. The saved successful draft call reported:

| Prompt Tokens | Completion Tokens | Reasoning Tokens | Total Tokens | Approx Public-Rate Cost |
| ---: | ---: | ---: | ---: | ---: |
| 19,494 | 6,362 | 4,336 | 25,856 | about $0.088 |

One earlier draft attempt returned non-parseable JSON; the script was updated to preserve raw responses and parse errors before the successful second call.

## Local Model Error Summary

The taxonomy script aligned `281` fixed held-out-aspect test rows across local, Qwen, Gemini, description variants, and the Pro cascade. The mean row sample F1 values match the existing full-test headline metrics.

| System | Mean Row F1 | Exact Rows | Empty Rows | Aspect Miss Rows | Aspect Over-Pred Rows | Sentiment Error Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT | 0.6071 | 144 | 0 | 78 | 119 | 18 |
| Qwen zero-shot | 0.5374 | 125 | 15 | 86 | 117 | 35 |
| Gemini Flash-Lite | 0.5516 | 138 | 37 | 107 | 88 | 17 |
| Gemini Flash | 0.6071 | 138 | 42 | 78 | 80 | 20 |
| Gemini Pro | 0.7141 | 167 | 28 | 49 | 67 | 18 |
| Flash-Lite + label descriptions | 0.5925 | 152 | 31 | 92 | 75 | 20 |
| Flash-Lite + boundary descriptions | 0.5724 | 149 | 45 | 103 | 64 | 17 |
| Flash + label descriptions | 0.5893 | 150 | 61 | 88 | 48 | 24 |
| Local -> Pro cascade | 0.8102 | 194 | 0 | 28 | 74 | 12 |

## Heuristic Signals

The following counts are automatic row-level signals. They are prevalence indicators, not manually labelled gold categories. A row can have multiple tags.

| Heuristic Signal | Rows | Main Interpretation |
| --- | ---: | --- |
| `discounts_value_boundary` | 155 | Broad semantic boundary bleed around value, promotions, competitor value, and related concepts. |
| `account_access_overprediction` | 112 | Account/access label triggered by broader account, card, control, or feature language. |
| `empty_abstention` | 84 | Hosted models or prompt variants return no label on positive-gold rows. |
| `competitor_positive_miss` | 82 | Positive competitor/comparison evidence is missed or mapped to adjacent aspects. |
| `description_precision_shift` | 66 | Description prompt removes false positives or shifts toward precision. |
| `description_row_gain` | 57 | Description prompt improves the row-level prediction. |
| `qwen_overprediction` | 56 | Qwen predicts extra labels on fixed-split rows. |
| `local_overprediction` | 50 | Local DistilBERT branch predicts extra labels on fixed-split rows. |
| `description_recall_loss` | 45 | Description prompt increases false negatives or empty predictions. |
| `neutral_under_recall` | 25 | Neutral sentiment is missed or shifted to positive/negative. |
| `pro_empty_cascade_recovery` | 20 | Pro abstains but the cascade recovers a non-empty useful prediction. |

## Final Taxonomy

The final taxonomy is manually consolidated from the automatic signals, existing quantitative summaries, and the Gemini-assisted draft. It should be treated as a thesis-facing analytical framework, not as an additional supervised label set.

### 1. Semantic Boundary Bleed

Definition:

- The model selects a semantically adjacent aspect because the review contains plausible but non-decisive lexical cues.

Main signals:

- `account_access_overprediction`: 112 rows.
- `discounts_value_boundary`: 155 rows.
- Recurrent examples include competitor praise or value language being mapped to account access or promotions.

Models affected:

- All systems, though stronger Gemini models reduce some over-prediction relative to local/Qwen.

Thesis use:

- This is the core open-topic candidate-label difficulty: the model must understand not only the review, but the precise boundary of the candidate aspect.

Qwen fine-tuning implication:

- Use hard negatives and contrastive instructions. Train with near-miss examples where account features are not account access, general value is not discounts/promotions, and broad company praise is not necessarily a competitor comparison.

### 2. Competitor-Positive Recall Bottleneck

Definition:

- The system fails to recognise positive comparison or competitor evidence, especially when the comparison is implicit.

Main signals:

- `competitor_positive_miss`: 82 rows.
- Involved models by automatic count: Flash-Lite 69, Flash 56, Qwen 50, Pro 37, local 21, Pro cascade 20.

Models affected:

- Gemini Flash/Flash-Lite and Qwen are especially affected; Pro improves but still has misses.

Thesis use:

- Explains why Gemini can have high precision but still lose samples F1: competitor evidence is often subtle and is easy to abstain from.

Qwen fine-tuning implication:

- Add focused positive competitor examples, including implicit comparisons, "best app/bank/provider" claims, and alternative-provider references. Pair them with non-competitor brand satisfaction hard negatives.

### 3. Generative Over-Prediction Or Fail-Noisy Behaviour

Definition:

- The model emits extra labels rather than abstaining, improving recall but harming precision.

Main signals:

- `qwen_overprediction`: 56 fixed-split rows.
- `local_overprediction`: 50 fixed-split rows.
- Qwen LOAO all-row summary: mean precision `0.2379`, mean recall `0.8182`, and false-positive rows per 100 `34.4150`.

Models affected:

- Qwen zero-shot most strongly; local DistilBERT also shows a milder version.

Thesis use:

- Separates Qwen's open-weight zero-shot failure from Gemini's hosted LLM failure. Qwen tends to fail noisy; Gemini often fails silent.

Qwen fine-tuning implication:

- Prioritise abstention/no-label calibration, hard negatives, and threshold-like instruction tuning. Fine-tuning should reduce false positives without destroying Qwen's useful broad semantic recall.

### 4. Cautious Abstention Or Fail-Silent Behaviour

Definition:

- The model predicts an empty label set for a row with a held-out gold label.

Main signals:

- `empty_abstention`: 84 rows where at least one model abstains.
- Empty rows by system: Flash 42, Pro 28, Flash-Lite 37, Flash + descriptions 61, Flash-Lite boundary descriptions 45.
- `pro_empty_cascade_recovery`: 20 rows.

Models affected:

- Gemini family and description prompts, especially stricter/boundary wording.

Thesis use:

- Explains why pure Pro is not the best deployed system despite being the strongest pure hosted baseline. Silent misses are operationally dangerous because they hide relevant feedback.

Qwen fine-tuning implication:

- Fine-tuning should not simply make Qwen conservative. The goal is calibrated abstention: reduce false positives while avoiding Gemini-style empty misses on subtle positives.

### 5. Sentiment Polarity Under-Recall

Definition:

- The aspect is partly understood, but the sentiment polarity is missed, especially for neutral labels.

Main signals:

- `neutral_under_recall`: 25 fixed-split rows.
- Existing Gemini analysis shows neutral labels have high precision but weak recall.
- Qwen fixed split has 35 sentiment error rows, more than local, Flash, or Pro.

Models affected:

- Qwen and Gemini variants; neutral is the most fragile polarity.

Thesis use:

- Separates aspect relevance from sentiment prediction. Some failures are not about topic detection but about polarity boundaries.

Qwen fine-tuning implication:

- Include balanced sentiment examples per aspect, with special attention to neutral and weakly evaluative language. Keep aspect detection and sentiment calibration jointly supervised.

### 6. Prompt-Induced Precision-Recall Shift

Definition:

- Adding generated aspect descriptions changes the model's operating point, sometimes improving precision but sometimes increasing false negatives or empty predictions.

Main signals:

- `description_precision_shift`: 66 rows.
- `description_row_gain`: 57 rows.
- `description_recall_loss`: 45 rows.
- Flash-Lite label-only descriptions improve test pair samples F1 from `0.5516` to `0.5925`.
- Flash label-only descriptions reduce test pair samples F1 from `0.6071` to `0.5893`, while slightly improving pair micro/macro F1.

Models affected:

- Gemini Flash-Lite and Flash description variants.

Thesis use:

- Supports a nuanced prompt-engineering claim. Explicit label semantics help cheap models, but richer descriptions are not universally beneficial and can make stronger models too cautious.

Qwen fine-tuning implication:

- Fine-tuning should internalise stable label semantics rather than relying on fragile prompt wording. Description-like definitions can be used in training data, but they should be paired with hard negatives and tested for recall loss.

### 7. Cascade Complementarity

Definition:

- Local and hosted models make different errors, so a selective system can outperform either pure component by combining their strengths.

Main signals:

- Local -> Pro cascade mean row F1/test pair samples F1: `0.8102`.
- Pure Pro test pair samples F1: `0.7141`.
- Cascade exact rows: 194 versus Pro 167.
- Pro empty rows: 28; cascade recovers useful non-empty predictions on 20 automatically tagged Pro-empty rows.
- Cascade has only 28 aspect-miss rows, compared with Pro 49 and local 78.

Models affected:

- Local DistilBERT and Gemini Pro cascade.

Thesis use:

- This is the strongest fixed-split system result and the clearest deployment argument. The cascade works because it mitigates Pro abstention and local over-prediction through complementarity, not because one component is universally better.

Qwen fine-tuning implication:

- A future fine-tuned Qwen could be evaluated both as a standalone local/open-weight model and as an escalator or fallback component. The taxonomy suggests it should be calibrated enough to participate in a cascade, not merely optimised for recall.

## Implications Before Qwen Fine-Tuning

Task 5 supports doing Qwen fine-tuning next, but it also clarifies what that fine-tuning should target.

Priority training objectives:

1. **Abstention calibration**: reduce Qwen's false positives on empty/near-miss cases without losing broad semantic recall.
2. **Hard-negative aspect boundaries**: explicitly contrast account access versus account features, discounts/promotions versus price/value, and competitor comparison versus general brand satisfaction.
3. **Competitor-positive recall**: add implicit comparison and alternative-provider examples.
4. **Neutral sentiment coverage**: balance neutral examples and ambiguous weak-polarity language.
5. **Stable label semantics**: include label definitions or description-style prompts during training only if they are validated against recall loss.
6. **Cascade readiness**: export confidence or uncertainty signals from future Qwen runs so it can be used in selective deployment comparisons.

## Limitations

- The taxonomy uses heuristic signals and manually reviewed aggregates, not a newly annotated gold error dataset.
- Fixed-split row-level analysis covers only three held-out aspects.
- LOAO evidence is summary/per-aspect evidence rather than full row-level qualitative inspection in this document.
- Gemini-assisted drafting was used to suggest category wording, but final categories are manually consolidated.
- Raw review text remains local-only and is not quoted here.

## Validation

Final validation after the Gemini-assisted documentation update:

- Unit tests: 75 tests OK with `python -m unittest discover -s tests`.
- Compile check: passed with `python -m compileall -q src scripts tests`.
- Diff whitespace check: passed with `git diff --check`.
- Secret scan: no `sk-...` or Google API-key pattern matches in tracked project/docs/scripts/test/thesis paths.
