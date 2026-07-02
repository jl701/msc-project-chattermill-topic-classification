# Qwen LoRA Held-Out-Aspect And LOAO Launch Plan

Last updated: 2026-07-02

This note defines the launch plan for the optional fixed held-out-aspect Qwen LoRA run and the later full 12-fold Qwen LoRA leave-one-aspect-out (LOAO) run. It is a plan only: no full fixed held-out-aspect Qwen LoRA run and no full 12-fold Qwen LoRA LOAO run have been started.

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

## Optional Fixed Held-Out-Aspect Configuration

The fixed held-out-aspect Qwen LoRA run is not required before writing the full LOAO plan, but it is useful adaptation evidence if GPU time allows. On the local 8 GB laptop GPU it is expected to take more than two hours once full validation/test generation is included, so it should not be started without explicit confirmation.

Prepare the fixed three-aspect SFT data:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_sft_indexed
```

Recommended validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_YYYYMMDD --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Recommended test command after validation review:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_YYYYMMDD --eval-split test --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --gradient-checkpointing --save-adapter --skip-training-if-adapter-exists --resume-predictions --skip-existing-predictions
```

Fixed-run stopping rule:

- If validation JSON/schema validity drops materially below the zero-shot fixed result, inspect raw local predictions before running test.
- If validation over-predicts badly on empty-gold rows, do not treat the fixed run as sufficient evidence for LOAO readiness.
- Do not choose test-facing hyperparameters after inspecting test labels.

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

Prepare one fold directory per aspect. Example for fold 09:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --heldout-aspect "Staff support: Email" --output-dir .\outputs\qwen_lora_loao_sft_YYYYMMDD\09_staff_support_email
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

- Full 12-fold Qwen LoRA LOAO can start only after the target GPU environment, available storage, package versions, and data-transfer rules are confirmed for the actual machine that will run it.
- This confirmation is still open; the current document records assumptions and command templates, not a completed full-LOAO launch.
