# Qwen3-4B Feasibility Notes

This note records the first local feasibility checks for Qwen3-4B on the closed-topic FABSA aspect+sentiment task.

## Environment

Local GPU:

```text
NVIDIA GeForce RTX 5050 Laptop GPU
8 GB VRAM
```

The standard PyTorch install was CPU-only. CUDA training required:

```powershell
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0+cu128
```

Additional LLM dependencies:

```powershell
python -m pip install -r requirements-llm.txt
```

The local pilot used:

```text
model: Qwen/Qwen3-4B-Instruct-2507
loading: 4-bit bitsandbytes NF4
training: LoRA over attention and MLP projection layers
LoRA rank: 8
LoRA alpha: 16
trainable parameters: 16,515,072
trainable percentage: about 0.41%
```

## Prompt Format

The model is prompted as an aspect-sentiment extractor:

```text
Review:
...

Candidate aspects:
- Account management: Account access
- Company brand: Competitor
...

Return a JSON array of objects with keys "aspect" and "sentiment".
```

The model must output canonical FABSA aspects only. This keeps evaluation string-exact and avoids synonym mapping problems.

## Data Scope

These are feasibility results, not final full-validation or test results.

The current local pilot evaluates only the first 100 validation rows because generation is slow on the 8 GB laptop GPU.

## Results On Validation First 100 Rows

| Run | Train Examples | Epochs | Best Epoch | Pair Micro F1 | Pair Macro F1 | Aspect Micro F1 | Valid JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot Qwen3-4B-Instruct | 0 | 0 | - | 0.5407 | 0.3076 | 0.5539 | 0.99 |
| LoRA, lr 2e-4 | 500 | 1 | 1 | 0.6476 | 0.3429 | 0.6815 | 1.00 |
| LoRA, lr 2e-4 | 500 | 2 | 2 | 0.7193 | 0.3853 | 0.7449 | 1.00 |
| LoRA, lr 2e-4 | 500 | 3 | 3 | 0.7283 | 0.4011 | 0.7616 | 1.00 |
| LoRA, lr 2e-4 | 1000 | 2 | 2 | 0.7405 | 0.3719 | 0.7661 | 1.00 |
| LoRA, lr 2e-4 | 1000 | 3 | 2 | 0.7586 | 0.4047 | 0.8012 | 1.00 |
| LoRA, lr 1e-4 | 1000 | 3 | 3 | 0.7616 | 0.3998 | 0.7930 | 1.00 |

## Interpretation

Local QLoRA is feasible on the 8 GB Windows laptop GPU. The model loads in 4-bit, trains LoRA adapters, and produces valid canonical JSON reliably.

Fine-tuning clearly improves over zero-shot Qwen on the 100-row validation slice:

```text
zero-shot pair micro F1: 0.541
best pilot pair micro F1: 0.762
```

However, the local setup is slow. The best 1000-example, 3-epoch run took about 58 minutes and still evaluated only 100 validation examples. Full training and full validation/test evaluation should ideally run on a stronger Linux GPU environment.

The Qwen pilot is not yet directly comparable with the DistilBERT closed-topic baseline, because DistilBERT was evaluated on the full validation/test splits while Qwen was evaluated on a 100-row validation slice.

## Recommended Next Step

Use this local setup for smoke tests and prompt/data-format checks. For serious Qwen experiments:

- Run on a stronger GPU.
- Train on the full FABSA training split.
- Evaluate on full validation and test splits.
- Save adapter checkpoints for the best validation epoch.
- Consider using the same candidate-label prompt design later for open-topic experiments.

## Methodological Grounding

The open-topic Qwen direction is not treated as free-form topic generation. It is a candidate-label prediction problem: the model receives the feedback text plus a list of allowed canonical candidate aspects, then returns selected aspect/sentiment pairs.

This design is motivated by several related research lines:

- Instruction-tuned ABSA, such as InstructABSA, which reformulates ABSA subtasks as instruction-following generation tasks.
- Few-shot instruction tuning for ABSA, which supports fine-tuning sequence-to-sequence or instruction models with task prompts rather than relying only on fixed classifier heads.
- Unified generative ABSA, which predicts structured aspect/sentiment outputs as generated sequences.
- Entailment-style zero-shot text classification, where candidate labels are supplied as natural-language hypotheses or label descriptions instead of being fixed output-head classes.
- Structured-output prompting and constrained generation, which support parseable model outputs and reduce label-format errors.

For this project, the indexed candidate-label format is a practical adaptation of those ideas:

```text
Candidate aspects:
A1. Account management: Account access
A2. Company brand: Competitor
A3. Value: Discounts promotions

Return a JSON array of objects with keys "aspect_id" and "sentiment".
```

The model output is then mapped from `aspect_id` back to the canonical FABSA label. This avoids near-miss strings such as `Account management`, while still preventing the model from inventing topic names outside the candidate set.

Useful references:

- Scaria et al., 2024, `InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis`.
- Varia et al., 2023, `Instruction Tuning for Few-Shot Aspect-Based Sentiment Analysis`.
- Zhang et al., 2021, `A Unified Generative Framework for Aspect-Based Sentiment Analysis`.
- Yin et al., 2019, `Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach`.

## Held-Out Aspect Candidate-Label Smoke Test

A held-out-aspect Qwen zero-shot smoke test was added after the first label-aware non-LLM baselines.

The important prompt change is to use indexed candidate labels:

```text
Candidate aspects:
A1. Account management: Account access
A2. Company brand: Competitor
A3. Value: Discounts promotions

Return a JSON array of objects with keys "aspect_id" and "sentiment".
```

This avoids near-miss canonical strings such as `Account management`, while still preventing free-form topic generation.

Full held-out-aspect indexed zero-shot results:

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Seconds / Example |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.5753 | 0.5762 | 0.4514 | 0.6382 | 1.0000 | 1.77 |
| test | 0.5374 | 0.5300 | 0.4374 | 0.6340 | 1.0000 | 1.74 |

These are zero-shot prompt results, not fine-tuned Qwen results. The local GPU is sufficient for prompt smoke tests, but full Qwen fine-tuning should still wait for stronger GPU access.

GPU-ready held-out-aspect SFT JSONL files can be generated with:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy both --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_sft_indexed
```

The training split uses seen-aspect supervision only. Validation and test use held-out candidate aspects and held-out labels only.

## Full LOAO Zero-Shot Baseline

A full local open-weight Qwen LOAO zero-shot run was completed on 2026-07-01 before Qwen fine-tuning. This is not a fine-tuned result. It is a robustness baseline for the indexed candidate-label prompt under taxonomy shift.

Protocol:

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Loading: local Transformers causal LM, default 4-bit bitsandbytes NF4 double quantisation.
- Prompt: indexed candidate label, top-level JSON array output, `aspect_id` mapped back to the canonical FABSA aspect.
- Decoding: deterministic generation, `max_input_tokens=1024`, `max_new_tokens=192`.
- LOAO folds: all 12 FABSA aspects, one held-out aspect supplied as the only candidate label in each fold.
- Main evaluation scope: all official validation/test rows, with gold labels filtered to the held-out aspect and empty predictions allowed.
- Split-builder metadata: `strategy=label_masked`; this does not affect zero-shot evaluation rows under all-row LOAO.
- Hardware: local RTX 5050 Laptop GPU, 8 GB VRAM.

Commands:

```powershell
python .\scripts\run_qwen_loao_heldout_aspect.py --split validation --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701
python .\scripts\run_qwen_loao_heldout_aspect.py --split test --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701
python .\scripts\analyse_qwen_loao_predictions.py --validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701
```

Local ignored outputs:

- `outputs/llm/qwen_loao_heldout_aspect_all_rows_validation_20260701/`
- `outputs/llm/qwen_loao_heldout_aspect_all_rows_test_20260701/`
- derived positive-row diagnostic: `outputs/analysis/qwen_loao_positive_diagnostic_20260701/`

Aggregate all-row LOAO spread:

| Split | Rows / Fold | Pair Samples F1 Mean | Pair Samples F1 Min-Max | Pair Micro F1 Mean | Pair Micro F1 Min-Max | Pair Macro F1 Mean | Pair Precision Mean | Pair Recall Mean | FP Rows / 100 Mean | Valid JSON | Schema Valid | Seconds / Example |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 1,057 | 0.1194 | 0.0104-0.3500 | 0.3293 | 0.0449-0.5916 | 0.2340 | 0.2310 | 0.8115 | 34.7446 | 1.0000 | 0.9961 | 0.9756 |
| test | 1,587 | 0.1212 | 0.0132-0.3527 | 0.3378 | 0.0513-0.6011 | 0.2412 | 0.2379 | 0.8182 | 34.4150 | 1.0000 | 0.9955 | 1.1184 |

Test per-aspect pair micro F1 ranked from hardest to easiest:

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

Positive-gold-row diagnostic from the same predictions:

| Split | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | Pair Macro F1 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.8129 | 0.8692 | 0.9481 | 0.8115 | 0.6024 | 0.9451 |
| test | 0.8194 | 0.8659 | 0.9338 | 0.8182 | 0.6184 | 0.9314 |

Interpretation:

- Qwen's structured output is stable locally: valid JSON is 1.0000 on both full all-row splits.
- The only schema issue observed was recoverable `aspect_id` formatting such as `A1. Account management: Account access`; these rows are mapped back to `A1` but still counted as schema-invalid.
- Full all-row LOAO exposes a high-recall, low-precision behaviour. On the test split, mean recall is 0.8182 but mean precision is only 0.2379, with 34.4150 empty-gold false-positive rows per 100 reviews.
- Positive-gold rows are much stronger than all-row rows. This indicates that the main zero-shot failure mode is deciding when the single candidate aspect is absent, not producing parseable JSON or assigning sentiment once the aspect is truly present.
- The hardest test aspects are `Company brand: Reviews`, `Staff support: Email`, `Account management: Account access`, and `Value: Discounts promotions` by pair micro F1. `Company brand: Reviews` and `Value: Price value for money` are especially over-predicted on empty-gold rows.
- This supports the need for Qwen fine-tuning or calibration before treating local open-weight LLMs as robust open-topic classifiers. The zero-shot prompt is semantically useful, but it is not calibrated for all-row absence detection under taxonomy shift.

Comparison boundaries:

- The earlier fixed held-out-aspect Qwen result (`0.5374` test pair samples F1, `0.5300` pair micro F1) used the fixed three held-out aspects and evaluated rows containing those aspects. It should not be compared directly with all-row LOAO, but the large drop confirms that the fixed split was easier.
- The strongest DistilBERT LOAO result has lower mean test pair samples F1 (`0.0550`) and slightly lower mean pair micro F1 (`0.3128`) than Qwen LOAO, but it has much higher precision and fewer false-positive rows. Qwen is therefore better at recall-oriented semantic matching, while DistilBERT is more conservative.
- Gemini fixed/cascade results remain fixed held-out-aspect or selective-deployment evidence, not LOAO robustness evidence.

The full dissertation-oriented comparison, including the per-aspect Qwen-vs-DistilBERT LOAO table, the positive-gold diagnostic gap, metric interpretation, and article contribution framing, is recorded in `docs/qwen_loao_experiment_analysis.md`.

Next step:

- Use this as the local open-weight zero-shot LOAO baseline before Qwen SFT/QLoRA.
- For Qwen fine-tuning, keep the indexed candidate-label format and explicitly evaluate all-row LOAO after any fixed-split improvement.
- Consider calibration or an absence-aware prompt/objective, because empty-gold false positives are the main all-row bottleneck.

## Gemini Comparison Result

A Gemini hosted candidate-label runner was added and evaluated on 2026-07-01 to provide a closed hosted-LLM comparison under the same indexed held-out-aspect protocol:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full
```

It uses candidate IDs rather than copied aspect strings, normalises predictions into the same pair-label metrics, and records JSON/schema validity, latency, token usage, and reasoning/thinking tokens when the endpoint reports them.

Final fixed held-out-aspect test result:

| Model | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Valid JSON |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct indexed zero-shot | 0.5374 | 0.5300 | 0.4374 | 1.0000 |
| Gemini 2.5 Flash indexed JSON-schema | 0.6071 | 0.6541 | 0.5547 | 1.0000 |

Gemini is therefore the stronger zero-shot hosted/generative candidate-label baseline on this fixed three-aspect evaluation. This does not replace the need for full Qwen fine-tuning or LOAO robustness checks. See `docs/gemini_candidate_label_baseline.md` for the validation sweep, max-token truncation finding, schema diagnostics, latency, and token-cost accounting.
