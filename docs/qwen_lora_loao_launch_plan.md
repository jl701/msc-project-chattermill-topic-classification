# Qwen LoRA Held-Out-Aspect And LOAO Launch Plan

Last updated: 2026-07-02

This note records the completed fixed held-out-aspect Qwen LoRA run, the later single-fold all-row validation-gated pilot, and the launch plan for a possible full 12-fold Qwen LoRA leave-one-aspect-out (LOAO) run. The fixed run and single-fold pilot are complete; the full 12-fold Qwen LoRA LOAO run has not been started.

## Current Readiness

Completed prerequisites:

- `scripts/run_qwen_lora_heldout_aspect.py` reads indexed held-out-aspect SFT JSONL.
- The runner supports `Qwen/Qwen3-4B-Instruct-2507`, 4-bit QLoRA, configurable LoRA rank/alpha/dropout/target modules, train/validation/test splits, canonical candidate-ID parsing, existing evaluation metrics, manifest logging, adapter saving, adapter-weight resume, prediction resume, and complete-prediction skip.
- `scripts/prepare_qwen_heldout_aspect_sft_data.py` can prepare either the default fixed three-aspect split or a one-aspect fold with `--heldout-aspect`.
- Focused tests cover JSONL loading, prompt-answer loss masking, manifest fields, adapter skip/resume logic, prediction resume checks, and indexed prediction normalisation.
- Tiny local smoke completed on the RTX 5050 Laptop GPU:
  - train rows: 4;
  - validation rows: 1;
  - test rows: 1;
  - LoRA step: 1;
  - adapter saved under ignored `outputs/`;
  - validation/test JSON and schema validity: `1.0000`;
  - metrics and manifest written.
- One fixed held-out-aspect QLoRA configuration completed on the RTX 5050 Laptop GPU:
  - train rows used: `6,485` after skipping `10` fully truncated-answer rows;
  - validation rows: `212`;
  - test rows: `281`;
  - test pair samples F1: `0.5528`;
  - test pair micro F1: `0.5552`;
  - test valid JSON/schema-valid rates: `1.0000 / 0.9964`.
- One single-fold all-row Qwen LoRA validation-gated pilot completed on `Company brand: Competitor`:
  - tested grouped SFT, conservative prompting, singleton neg1, singleton neg0.25, and singleton neg0.10;
  - best validation branch: singleton neg0.10;
  - validation pair micro F1: `0.1900`;
  - validation precision/recall: `0.1667 / 0.2209`;
  - valid JSON/schema-valid rates: `1.0000 / 1.0000`;
  - did not beat same-fold Qwen zero-shot `0.2397` or local DistilBERT `0.2490`.

## Completed Fixed Held-Out-Aspect Configuration

The fixed held-out-aspect Qwen LoRA run was completed after explicit approval for a long local GPU run. It is useful adaptation evidence, but it is still fixed-split evidence and must not be treated as full LOAO robustness evidence.

Input data:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_sft_indexed
```

Successful validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Successful test command after validation review:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702 --eval-split test --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --skip-training-if-adapter-exists --resume-predictions --skip-existing-predictions
```

Observed fixed-run result:

| Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Schema Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.5991 | 0.5977 | 0.4335 | 0.6635 | 1.0000 | 1.0000 |
| test | 0.5528 | 0.5552 | 0.4393 | 0.6192 | 1.0000 | 0.9964 |

Fixed-run limitations:

- The fixed run used only the fixed three held-out aspects, not all-row LOAO folds.
- It does not evaluate empty-gold absence calibration.
- It was a single one-epoch local configuration, not a full Qwen hyperparameter sweep.
- The full LOAO templates below remain necessary before claiming fine-tuned Qwen robustness across aspect rotations.

## Single-Fold All-Row Validation Gate

The first full all-row Qwen LoRA fold was run on `Company brand: Competitor` before launching a 12-fold sweep.

Outcome:

| Variant | Validation Pair Micro F1 | Precision | Recall | FP Rows / 100 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| grouped indexed SFT | 0.1681 partial | 0.0935 | 0.8333 | 88.7850 | Early-stopped for over-prediction |
| conservative prompt-only reuse | 0.1707 partial | 0.0946 | 0.8750 | 75.0000 | Early-stopped for over-prediction |
| singleton neg1 | 0.0625 | 0.3000 | 0.0349 | 0.1892 | Too conservative |
| singleton neg0.25 | 0.1803 | 0.3056 | 0.1279 | 1.6083 | Below baseline |
| singleton neg0.10 | 0.1900 | 0.1667 | 0.2209 | 7.2848 | Best branch, still below baseline |

Same-fold validation baselines:

- Qwen zero-shot: pair micro F1 `0.2397`;
- local DistilBERT: pair micro F1 `0.2490`.

Launch implication:

- Do not launch the full 12-fold Qwen LoRA LOAO with the current grouped or singleton SFT recipe.
- The templates below remain useful, but only after a revised absence-calibration objective passes a single-fold validation gate.

## Full 12-Fold LOAO Fold List

Use deterministic fold IDs:

| Fold | Held-Out Aspect | Fold ID |
| ---: | --- | --- |
| 01 | Account management: Account access | `01_account_management_account_access` |
| 02 | Company brand: Competitor | `02_company_brand_competitor` |
| 03 | Company brand: General satisfaction | `03_company_brand_general_satisfaction` |
| 04 | Company brand: Reviews | `04_company_brand_reviews` |
| 05 | Logistics rides: Speed | `05_logistics_rides_speed` |
| 06 | Online experience: App website | `06_online_experience_app_website` |
| 07 | Purchase booking experience: Ease of use | `07_purchase_booking_experience_ease_of_use` |
| 08 | Staff support: Attitude of staff | `08_staff_support_attitude_of_staff` |
| 09 | Staff support: Email | `09_staff_support_email` |
| 10 | Staff support: Phone | `10_staff_support_phone` |
| 11 | Value: Discounts promotions | `11_value_discounts_promotions` |
| 12 | Value: Price value for money | `12_value_price_value_for_money` |

## Full LOAO Data Preparation Template

Prepare one fold directory per aspect. Use `--eval-row-scope all` for the frozen all-row LOAO protocol. Example for fold 09:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --heldout-aspect "Staff support: Email" --eval-row-scope all --output-dir .\outputs\qwen_lora_loao_sft_YYYYMMDD\09_staff_support_email
```

Output pattern:

```text
outputs/qwen_lora_loao_sft_YYYYMMDD/<fold_id>/example_filtered/
  metadata.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

## Full LOAO Validation Command Template

Run validation first for every fold. Do not run test until validation summaries and manifests have been checked.

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_YYYYMMDD\<fold_id> --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_example_filtered_r8_lr1e-5_ep1_YYYYMMDD\<fold_id> --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

## Full LOAO Test Command Template

After validation review, run test using the same output directory and saved adapter:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_YYYYMMDD\<fold_id> --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_example_filtered_r8_lr1e-5_ep1_YYYYMMDD\<fold_id> --eval-split test --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --gradient-checkpointing --save-adapter --skip-training-if-adapter-exists --resume-predictions --skip-existing-predictions
```

## Resume And Recovery Plan

Prediction recovery:

- Re-run the same command with `--resume-predictions --skip-existing-predictions`.
- Existing prediction JSONL files are accepted only when their row-index/id prefix matches the requested split.
- Complete prediction files are skipped and metrics are recomputed from the stored rows.
- Each runner invocation writes both latest `manifest.json` and a run-specific manifest under `<output_dir>/manifests/`, so validation and test commands can be audited separately even when they share an adapter directory.

Adapter recovery:

- The normal final adapter path is `<output_dir>/adapter_final/`.
- Test commands should use `--skip-training-if-adapter-exists`; the runner auto-loads the existing `adapter_final` when no explicit `--resume-from-adapter` is supplied.
- If `--save-epoch-adapters` was used and the final adapter is missing, resume manually with `--resume-from-adapter <output_dir>\checkpoints\adapter_epoch_XX`.
- Optimiser and scheduler state are not restored. Treat interrupted training resumes as adapter-weight continuation, not exact bitwise continuation.

Fold-level restart rule:

- If training fails before any adapter checkpoint is saved, delete or archive that fold output directory under ignored `outputs/` and rerun the fold.
- If generation fails after adapter saving, keep the adapter and rerun only the missing validation/test predictions.

## Resource And Storage Assumptions

Current local confirmed environment:

- GPU: NVIDIA GeForce RTX 5050 Laptop GPU.
- VRAM: about 8 GB.
- PyTorch: `2.10.0+cu128`.
- Transformers: `4.57.6`.
- PEFT: `0.19.1`.
- bitsandbytes: `0.49.2`.
- accelerate: `1.14.0`.

Recommended full-LOAO target:

- Prefer a 16 GB or larger CUDA GPU for full 12-fold training/evaluation.
- Keep the local 8 GB GPU for smoke tests, short fixed-split diagnostics, or recovery checks.
- Reserve at least 30 GB free space for Hugging Face cache, 12 adapters, optional epoch adapters, prediction JSONL, manifests, summaries, and logs.

Runtime expectation:

- The successful tiny local smoke took about 27 seconds for 4 train rows, 1 validation row, and 1 test row.
- Zero-shot all-row Qwen LOAO previously required roughly 1 second per example for generation on the local laptop.
- Fine-tuned generation and 12-fold training are expected to exceed two hours on the local laptop and may take many hours. Do not start full LOAO locally without an explicit GPU window.

Data-transfer rules:

- Transfer only tracked code, public FABSA data if needed, and generated SFT JSONL derived from public FABSA.
- Do not transfer credentials, API keys, private endpoints, raw internal Chattermill data, review-text packets from private data, checkpoints, adapters, or raw prediction files into Git.
- Keep all model artifacts and raw predictions under ignored `outputs/`, `models/`, or `checkpoints/`.
- Summarise only aggregate metrics and plot-ready data into tracked docs.

## Pre-Launch Checks

Immediately before any full LOAO launch, rerun:

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git diff --check
rg -n "sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|Bearer [A-Za-z0-9._-]{20,}" . --glob '!outputs/**' --glob '!data/**' --glob '!models/**' --glob '!checkpoints/**' --glob '!.git/**'
git status --short --branch
```

Launch gate:

- Full 12-fold Qwen LoRA LOAO can start only after a revised absence-calibration objective passes a single-fold validation gate and the target GPU environment, available storage, package versions, and data-transfer rules are confirmed for the actual machine that will run it.
- The current grouped/singleton SFT recipe failed the validation gate; the current document records assumptions and command templates, not a completed or currently approved full-LOAO launch.
