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

## Gemini Comparison Path

A Gemini hosted candidate-label runner was added on 2026-07-01 to provide a closed hosted-LLM comparison under the same indexed held-out-aspect protocol:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 5 --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_YYYYMMDD_HHMMSS
```

It uses candidate IDs rather than copied aspect strings, normalises predictions into the same pair-label metrics, and records JSON/schema validity, latency, token usage, and reasoning/thinking tokens when the endpoint reports them.

The implementation has only been dry-run locally so far because no compatible API credentials were available in the environment. See `docs/gemini_candidate_label_baseline.md` before running hosted sweeps or comparing Gemini against Qwen.
