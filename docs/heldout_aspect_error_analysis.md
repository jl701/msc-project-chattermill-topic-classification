# Held-Out Aspect Error Analysis

This note records the first row-level error analysis for the held-out-aspect protocol.

Evaluation scope remains **held-out labels only**. Candidate labels are provided at inference, and models are not allowed to invent topic names.

## Models Analysed

The analysed prediction files are local outputs under `outputs/`, which is ignored by Git.

| Model | Strategy / Prompt | Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Candidate-aspect DistilBERT cross-encoder + global sentiment | label-masked | test | 0.5745 | 0.5605 | 0.4326 | 0.6610 |
| Candidate-aspect DistilBERT cross-encoder + global sentiment | example-filtered | test | 0.5610 | 0.5544 | 0.4477 | 0.6476 |
| Qwen3-4B-Instruct zero-shot | indexed prompt | validation | 0.5753 | 0.5762 | 0.4514 | 0.6382 |
| Qwen3-4B-Instruct zero-shot | indexed prompt | test | 0.5374 | 0.5300 | 0.4374 | 0.6340 |

The DistilBERT rows are from a repeated prediction-export run and differ slightly from the best observed aggregate run in `docs/generalisation_baselines.md`. The difference is small enough to treat these as the same performance band for analysis.

## Qwen Prompt Finding

The first Qwen held-out-aspect prompt used canonical aspect strings directly. It often returned parent names such as `Account management` instead of the exact candidate label `Account management: Account access`; these outputs were correctly filtered as invalid candidate choices.

An indexed candidate-label prompt fixed this format issue:

```text
Candidate aspects:
A1. Account management: Account access
A2. Company brand: Competitor
A3. Value: Discounts promotions

Return JSON objects with keys "aspect_id" and "sentiment".
```

This keeps the model constrained to canonical candidate labels while avoiding brittle string copying.

Prompt comparison on 50 sampled validation rows:

| Prompt Variant | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Valid JSON | Predicted Labels / Row |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 0.2000 | 0.2857 | 0.0556 | 1.0000 | 0.40 |
| conservative | 0.2000 | 0.2500 | 0.0444 | 1.0000 | 0.60 |
| indexed | 0.6000 | 0.6018 | 0.4719 | 1.0000 | 1.24 |
| indexed_conservative | 0.4933 | 0.5714 | 0.3552 | 1.0000 | 0.80 |
| indexed_descriptive | 0.5367 | 0.5405 | 0.3824 | 1.0000 | 1.20 |
| indexed_conservative_descriptive | 0.5000 | 0.5361 | 0.3155 | 1.0000 | 0.92 |

The plain `indexed` prompt is the current best prompt. The conservative variants under-predict; the descriptive variants did not help with the simple label-derived keyword descriptions.

## Error Type Summary

| Model | Exact Rows | Missed-All Rows | Aspect Miss Rows | Aspect Over-Predict Rows | Sentiment Error Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| DistilBERT cross-encoder, label-masked | 113 | 0 | 63 | 149 | 28 |
| DistilBERT cross-encoder, example-filtered | 120 | 0 | 72 | 145 | 29 |
| Qwen indexed zero-shot | 125 | 15 | 86 | 117 | 35 |

The DistilBERT cross-encoder almost never predicts no label, but over-predicts candidate aspects frequently. Qwen has more exact rows and fewer over-predictions, but also misses all held-out labels on 15 test rows.

## Aspect-Level Behaviour

Aspect-level F1 on the test split:

| Model | Account Access F1 | Competitor F1 | Discounts Promotions F1 |
| --- | ---: | ---: | ---: |
| DistilBERT cross-encoder, label-masked | 0.6540 | 0.6822 | 0.5897 |
| DistilBERT cross-encoder, example-filtered | 0.6625 | 0.6523 | 0.6207 |
| Qwen indexed zero-shot | 0.7024 | 0.6238 | 0.6018 |

Qwen is strongest on `Account management: Account access`, but weaker on `Company brand: Competitor`, especially for positive competitor mentions.

## Pair-Level Observations

Important pair-level patterns:

- `Company brand: Competitor | positive` has high support but low Qwen recall: `0.3011`.
- `Account management: Account access | negative` is strong for Qwen: F1 `0.7500`.
- `Value: Discounts promotions | positive` has high Qwen recall but low precision, indicating over-selection of promotional value when offers, rewards, or points are mentioned.
- Neutral labels remain difficult for all methods because they are rare and semantically subtle.

## Interpretation

The open-topic result is not simply a weak-model issue. The main failure modes are:

1. Candidate selection calibration: models often select too many held-out aspects.
2. Canonical output control: LLMs need ID-based candidate labels to avoid near-miss topic strings.
3. Sentiment coupling: aspect selection and sentiment prediction interact, especially for positive discounts and competitor mentions.
4. Rare neutral labels: neutral sentiment has low support and unstable F1.

The Qwen indexed zero-shot baseline is already competitive with the DistilBERT label-aware baseline on validation without fine-tuning, but it does not beat the best held-out-aspect test result yet. This supports using Qwen fine-tuning next, but the fine-tuning should preserve indexed candidate-label prompting rather than free-form topic generation.

## GPU-Ready Next Step

The prepared held-out-aspect Qwen SFT files are written locally under:

```text
outputs/qwen_heldout_aspect_sft_indexed/
```

They contain:

- `label_masked/train.jsonl`, `validation.jsonl`, `test.jsonl`
- `example_filtered/train.jsonl`, `validation.jsonl`, `test.jsonl`
- metadata files recording candidate aspects and prompt format

The training split uses seen-aspect supervision only. Validation and test use held-out candidate aspects and held-out labels only. The prompt uses `aspect_id` outputs, which should reduce invalid canonical label errors during full Qwen fine-tuning.

