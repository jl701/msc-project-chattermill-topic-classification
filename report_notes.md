# Report Notes

Last updated: 2026-07-02

This file is a compact evidence ledger for dissertation/report drafting. Detailed experiment records remain in `docs/`.

## 2026-07-01 - Gemini-Generated Aspect Descriptions

Task:

- Test whether Gemini-generated descriptions of the three fixed held-out FABSA aspects improve hosted candidate-label classification.

Protocol:

- Fixed held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Split strategy: `example_filtered`
- Evaluation label scope: held-out labels only
- Prompt base: indexed candidate IDs with `response_format=json_schema`
- Max tokens: `2048`
- Temperature: `0`
- Descriptions generated from canonical aspect names only; no validation/test review text used.

Tracked evidence:

- Implementation: `src/msc_project/llm/candidate_label.py`, `scripts/run_gemini_heldout_aspect.py`
- Tests: `tests/test_llm_candidate_label.py`
- Configs:
  - `configs/gemini_aspect_descriptions_fixed_heldout.json`
  - `configs/gemini_aspect_descriptions_decision_boundary_heldout.json`
- Detailed write-up: `docs/gemini_aspect_descriptions.md`
- Chronological log: `docs/experiment_log.md`

Observed results:

| Model / Prompt | Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Flash-Lite indexed | test | 0.5516 | 0.5872 | 0.4876 | 0.6062 |
| Flash-Lite label-only descriptions | test | 0.5925 | 0.6263 | 0.5691 | 0.6625 |
| Flash-Lite decision-boundary descriptions | test | 0.5724 | 0.6199 | 0.5456 | 0.6340 |
| Flash indexed | test | 0.6071 | 0.6541 | 0.5547 | 0.6747 |
| Flash label-only descriptions | test | 0.5893 | 0.6555 | 0.5655 | 0.6676 |
| Pro indexed | validation 50-row sample | 0.8000 | 0.8113 | 0.5985 | 0.8400 |
| Pro label-only descriptions | validation 50-row sample | 0.7467 | 0.7921 | 0.6128 | 0.8067 |

Interpretation:

- Label-only descriptions materially improve the cheap Flash-Lite hosted baseline.
- Decision-boundary descriptions reduce false positives but can increase empty predictions and false negatives.
- Flash descriptions improve validation but do not improve test pair samples F1.
- Pro descriptions do not justify a full run based on the validation diagnostic.
- Dissertation framing: this is a label-semantics ablation and precision-recall trade-off, not a universal prompt improvement.

Open follow-up:

- Sampled Gemini LOAO diagnostic if a lightweight robustness signal is needed.
- Cascade uncertainty improvement using local selector score/margin features.
- Qualitative error taxonomy with manual review.

## 2026-07-02 - Qualitative Error Taxonomy Pre-Registration

Task:

- Build a Gemini-assisted, manually verified qualitative error taxonomy before moving to Qwen fine-tuning.

Planned configuration:

- Use existing local ignored prediction outputs; do not rerun fixed-split or LOAO models.
- Fixed-split row-level sources: local DistilBERT, Qwen zero-shot, Gemini Flash-Lite/Flash/Pro, aspect-description variants, and Pro cascade.
- LOAO sources: existing Qwen-vs-DistilBERT comparison summaries and per-aspect CSVs.
- Output directory: `outputs/analysis/qualitative_error_taxonomy_20260702/`.
- Tracked documentation: `docs/qualitative_error_taxonomy.md`.
- Gemini Pro may assist taxonomy drafting from local ignored packets, but final categories must be manually reviewed.
- Raw review text must remain local-only and must not be committed.

Observed result:

- Implemented `scripts/analyse_qualitative_error_taxonomy.py`.
- Local output directory: `outputs/analysis/qualitative_error_taxonomy_20260702/`.
- Gemini Pro draft call succeeded with 19,494 prompt tokens, 6,362 completion tokens, 4,336 reasoning tokens, and 25,856 total tokens.
- Final tracked write-up: `docs/qualitative_error_taxonomy.md`.

Final taxonomy:

1. Semantic boundary bleed.
2. Competitor-positive recall bottleneck.
3. Generative over-prediction / fail-noisy behaviour.
4. Cautious abstention / fail-silent behaviour.
5. Sentiment polarity under-recall.
6. Prompt-induced precision-recall shift.
7. Cascade complementarity.

Qwen fine-tuning targets:

- abstention/no-label calibration;
- hard-negative aspect boundaries;
- competitor-positive recall;
- neutral sentiment coverage;
- stable label semantics;
- cascade-ready uncertainty signals.

## 2026-07-02 - Thesis Completion Roadmap

Decision:

- Keep LOAO as the dissertation's open-topic robustness spine.
- Treat full fine-tuned Qwen LoRA LOAO as the only major compute-bound unfinished experiment.
- Complete all non-major thesis work before GPU access is resolved.

Immediate non-major work:

- Finish qualitative error taxonomy from existing outputs.
- Build thesis-ready result tables and figure data.
- Try cascade score/margin uncertainty without new Gemini calls.
- Prepare and smoke-test the Qwen LoRA SFT/evaluation runner.
- Refresh the LaTeX thesis skeleton around the completed Gemini/Qwen/LOAO evidence.

Tracked roadmap:

- `docs/thesis_completion_roadmap.md`

## 2026-07-02 - Gemini-Assisted Qualitative Error Taxonomy

Command:

```powershell
python .\scripts\analyse_qualitative_error_taxonomy.py --output-dir .\outputs\analysis\qualitative_error_taxonomy_20260702

python .\scripts\analyse_qualitative_error_taxonomy.py --output-dir .\outputs\analysis\qualitative_error_taxonomy_20260702 --max-examples-per-category 10 --max-gemini-examples-per-category 2 --snippet-chars 220 --gemini-draft --gemini-model vertex_ai/gemini-2.5-pro --gemini-max-tokens 7000 --request-timeout 240
```

Observed Gemini-assisted packet:

- Rows aligned across fixed held-out-aspect test predictions: 281.
- Gemini Pro draft: successful, used only as an assistant for candidate taxonomy wording.
- Gemini usage: 19,494 prompt tokens, 6,362 completion tokens, including 4,336 reasoning tokens; 25,856 total tokens.
- Output directory: `outputs/analysis/qualitative_error_taxonomy_20260702/`.
- Tracked summary: `docs/qualitative_error_taxonomy.md`.

Key mechanism counts:

| Mechanism | Rows |
| --- | ---: |
| discounts/value boundary | 155 |
| account-access overprediction | 112 |
| empty abstention | 84 |
| competitor positive miss | 82 |
| description precision shift | 66 |
| Qwen overprediction | 56 |
| local overprediction | 50 |
| description recall loss | 45 |
| neutral under-recall | 25 |
| Pro-empty cascade recovery | 20 |

Interpretation:

- Qualitative evidence supports the quantitative story: local and Qwen over-predict, hosted Gemini is more conservative and can abstain, descriptions move precision/recall, and the Pro cascade works by recovering hosted abstentions.
- The final taxonomy remains manually consolidated: semantic boundary bleed, competitor-positive recall bottleneck, generative over-prediction, cautious abstention, sentiment polarity under-recall, prompt-induced precision-recall shift, and cascade complementarity.

## 2026-07-02 - Pre-Qwen LoRA Full LOAO Checklist Audit

Task:

- Re-check completed work and record the remaining work before full fine-tuned Qwen LoRA LOAO as a tickable checklist.

Tracked checklist:

- `docs/thesis_completion_roadmap.md`, section `Pre-Qwen LoRA Full LOAO Checklist`.

Completed before the checklist:

- Closed-topic, held-out organisation, fixed held-out-aspect, lexical/DistilBERT LOAO, Qwen zero-shot LOAO, Gemini Pareto, local-to-Gemini cascade, Gemini descriptions, qualitative taxonomy, and thesis skeleton refresh.

Remaining next work before full Qwen LoRA LOAO:

1. Thesis-ready result tables and figure data.
2. Cascade score/margin uncertainty improvement.
3. Final Qwen held-out-aspect LoRA SFT runner with manifest logging.
4. Resume/skip behaviour and focused tests for the runner.
5. Tiny local Qwen LoRA held-out-aspect smoke test.
6. Optional fixed held-out-aspect Qwen LoRA configuration before full LOAO.
7. Full 12-fold LOAO command templates and GPU environment confirmation.

## 2026-07-02 - Thesis-Ready Result Tables And Figure Data

Command:

```powershell
python .\scripts\build_thesis_result_tables.py
```

Inputs:

- Existing tracked documentation and already documented aggregate metrics.
- No raw prediction files, review text, model checkpoints, API credentials, or private outputs were committed.

Outputs:

- `docs/thesis_result_tables.md`
- `docs/thesis_figure_data/protocol_ladder.csv`
- `docs/thesis_figure_data/loao_robustness.csv`
- `docs/thesis_figure_data/qwen_positive_diagnostic.csv`
- `docs/thesis_figure_data/cascade_tradeoff.csv`
- `docs/thesis_figure_data/fixed_vs_loao_drop.csv`

Interpretation:

- The result table pack keeps fixed-split evidence, all-row LOAO robustness, Qwen positive-gold diagnostics, and Gemini/cascade deployment evidence separate.
- This closes the first pre-Qwen-LoRA-full-LOAO checklist item.

Limitations:

- The tables freeze the currently documented headline metrics; they should be regenerated after any future Qwen LoRA fixed-split or LOAO result.

Next step:

- Try cascade score/margin uncertainty without rerunning Gemini.

## 2026-07-02 - Planned Cascade Score/Margin Check

Objective:

- Test whether local DistilBERT selector score/margin features improve local-to-Gemini cascade selection without rerunning Gemini.

Pre-run audit:

- Existing strongest local prediction files do not contain candidate aspect scores or margins.
- The strongest local output directory contains prediction JSONL, per-label CSV, summary JSON, and a sentiment summary, but no saved candidate-aspect selector checkpoint.
- CUDA is available locally: PyTorch `2.10.0+cu128`, NVIDIA GeForce RTX 5050 Laptop GPU.

Planned command:

```powershell
python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702
```

Expected output:

- Local ignored prediction JSONL with aggregate `score_features` for validation and test rows.
- Local ignored summary for checking whether the rerun reproduces the documented fixed held-out-aspect baseline closely enough for cascade analysis.

Risk and stopping rule:

- If the local rerun fails or materially diverges from the documented baseline, do not treat score/margin cascade results as a replacement headline.
- Existing Gemini predictions will be reused; no Gemini API calls are planned.

Observed result:

- Local score-export rerun completed and reproduced the documented strongest local fixed held-out-aspect result:
  - validation pair samples F1 `0.6226`;
  - validation pair micro F1 `0.6049`;
  - selected threshold `0.37`;
  - test pair samples F1 `0.6071`;
  - test pair micro F1 `0.5917`.
- Score/margin-aware cascade sweeps reused existing Gemini predictions only.
- Candidate policies increased from `10,578` to `20,598`.
- Validation-selected results did not improve over the existing reliability-proxy cascade:
  - Flash-Lite: test pair samples F1 `0.6679`, call rate `0.5089`;
  - Flash: test pair samples F1 `0.7459`, call rate `0.9004`;
  - Pro: test pair samples F1 `0.8102`, call rate `0.9004`.

Interpretation:

- This is a negative methodological check: local score/margin features are now exportable and included in the policy search, but they did not replace or improve the validation-reliability proxy for the current fixed-split cascade.

Next step:

- Build the final Qwen held-out-aspect LoRA SFT runner with manifest and resume/skip behaviour.

## 2026-07-02 - Final Qwen Held-Out-Aspect LoRA Runner Readiness

Commands:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --limit 24 --output-dir .\outputs\qwen_heldout_aspect_sft_tiny_20260702

python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_tiny_20260702 --strategy example_filtered --output-dir .\outputs\qwen_lora_heldout_aspect_dry_run_20260702 --train-limit 8 --validation-limit 4 --test-limit 4 --epochs 1 --max-train-steps 1 --batch-size 1 --grad-accumulation-steps 1 --learning-rate 1e-4 --max-length 512 --max-input-tokens 512 --max-new-tokens 64 --dry-run

python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --heldout-aspect "Staff support: Email" --limit 3 --output-dir .\outputs\qwen_loao_sft_single_fold_check_20260702

python -m unittest tests.test_qwen_lora_heldout_runner tests.test_qwen_loao_runner tests.test_qwen_format
```

Inputs:

- Indexed held-out-aspect SFT JSONL generated from tracked split/preparation code.
- No raw outputs or model artifacts were moved into Git.

Outputs:

- `scripts/run_qwen_lora_heldout_aspect.py`
- `scripts/prepare_qwen_heldout_aspect_sft_data.py`
- `tests/test_qwen_lora_heldout_runner.py`
- Local ignored dry-run/JSONL outputs under `outputs/`.

Observed result:

- Runner dry-run completed and wrote `manifest.json` and `summary.json`.
- Manifest records command, cwd, git commit, Qwen model name, 4-bit loading, LoRA parameters, split protocol, row scope, seed, hardware, output directory, and package versions.
- Tests passed for JSONL loading, prompt-answer loss masking, manifest fields, prediction resume/skip logic, adapter skip decision logic, and indexed prediction normalisation.
- One single-aspect SFT generation check succeeded, so later LOAO fold directories can be prepared with `--heldout-aspect`.

Interpretation:

- The runner/manifest/resume/test readiness checklist items are complete.
- The actual tiny Qwen load/train/generate smoke test is still pending and should not be described as complete until it runs.

Limitations:

- Training resume reloads adapter weights only; optimiser and scheduler state are not restored.
- Dry-run does not verify bitsandbytes model loading, adapter saving, generation, JSON parsing from generated text, or metrics from actual Qwen predictions.

Next step:

- Run the tiny local held-out-aspect Qwen LoRA smoke test.

## 2026-07-02 - Tiny Local Qwen LoRA Held-Out-Aspect Smoke Test

Command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_tiny_20260702 --strategy example_filtered --output-dir .\outputs\qwen_lora_heldout_aspect_tiny_smoke_evalmode_20260702 --train-limit 4 --validation-limit 1 --test-limit 1 --epochs 1 --max-train-steps 1 --batch-size 1 --grad-accumulation-steps 1 --learning-rate 1e-6 --max-length 512 --max-input-tokens 512 --max-new-tokens 96 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions
```

Inputs:

- `outputs/qwen_heldout_aspect_sft_tiny_20260702/example_filtered/`
- 4 train rows, 1 validation row, 1 test row.

Outputs:

- Ignored local output: `outputs/qwen_lora_heldout_aspect_tiny_smoke_evalmode_20260702/`
- Adapter saved under `adapter_final/`.
- Validation/test prediction JSONL, manifest, and summary written.

Observed result:

- Model loading passed for `Qwen/Qwen3-4B-Instruct-2507` with 4-bit QLoRA.
- One LoRA training step completed: `global_step=1`, train loss `1.2265`.
- Adapter saving passed.
- Validation/test generation passed.
- Validation/test valid JSON rate: `1.0000`.
- Validation/test schema valid rate: `1.0000`.
- Candidate ID mapping passed; the validation raw output used `A1` and was normalised to `Account management: Account access | neutral`.
- Metrics were written to `summary.json`.

Interpretation:

- The final Qwen LoRA held-out-aspect runner passes a real local smoke test.
- These metrics are not thesis performance evidence because the evaluation has only one row per split.

Limitations:

- The canonical smoke used `--no-gradient-checkpointing` and a very low LR (`1e-6`) for local stability.
- Earlier tiny attempts exposed invalid repeated-token generation when inference stayed in the training/checkpointing path; the runner now explicitly prepares the model for inference before generation.

Next step:

- Do not run a long fixed Qwen LoRA configuration without a resource/time decision. Prepare the fixed-split command and full LOAO launch plan first.

## 2026-07-02 - Qwen LoRA Full LOAO Launch Plan

Output:

- `docs/qwen_lora_loao_launch_plan.md`

Observed result:

- Optional fixed held-out-aspect Qwen LoRA validation/test command templates are documented.
- The fixed run was not launched because it is expected to exceed two hours on the local 8 GB laptop GPU once full validation/test generation is included.
- Full 12-fold Qwen LoRA LOAO command templates are documented for all 12 aspect folds.
- The plan defines fold IDs, SFT data-preparation directories, output directory pattern, validation-before-test order, adapter naming, prediction recovery, adapter recovery, resource/storage assumptions, data-transfer rules, and pre-launch validation checks.
- Runner recovery semantics were tightened so `--skip-training-if-adapter-exists` auto-loads `adapter_final`.

Interpretation:

- The launch plan checklist item is complete.
- The optional fixed full run, target GPU confirmation, and launch-immediate validation gate remain open.

Limitations:

- Runtime/storage estimates are based on smoke-test and previous zero-shot generation timings, not a completed full fine-tuned fold.
- Full 12-fold Qwen LoRA LOAO has not been run.

Next step:

- Run final repo validation and safety checks, then commit safe tracked changes only.

## 2026-07-02 - Planned Fixed Held-Out-Aspect Qwen LoRA Run

Objective:

- Run the optional fixed held-out-aspect Qwen LoRA configuration after user approval for a long local GPU run.
- Treat it as fixed-split adaptation evidence, not full LOAO robustness evidence.

Pre-flight:

- Git worktree was clean before planning.
- Branch state: `main...origin/main [ahead 2]`.
- Current commit: `0e2cd3f llm: prepare qwen lora heldout runner`.
- CUDA available on NVIDIA GeForce RTX 5050 Laptop GPU, about 8 GB VRAM.

Inputs:

- `outputs/qwen_heldout_aspect_sft_indexed/example_filtered/`
- Fixed held-out aspects:
  - `Account management: Account access`;
  - `Company brand: Competitor`;
  - `Value: Discounts promotions`.
- Row counts:
  - train `6,495`;
  - validation `212`;
  - test `281`.

Planned validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Planned test command, only after validation sanity checks:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_20260702 --eval-split test --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --skip-training-if-adapter-exists --resume-predictions --skip-existing-predictions
```

Planned output:

- Local ignored output directory: `outputs/llm/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_20260702/`
- Local ignored logs under `outputs/logs/`.
- Only aggregate metrics and documentation updates should be committed.

Risk and stopping rule:

- Expected runtime exceeds two hours on the local laptop GPU.
- If no-checkpointing OOMs, retry with `--gradient-checkpointing` and record the fallback.
- Do not run test if validation JSON/schema validity collapses or the adapter/prediction outputs are incomplete.
- Do not tune hyperparameters from test labels.

Observed first attempt:

- The `lr=1e-5` validation run started and reached training step 250.
- Loss became non-finite:
  - step 150 mean loss `0.3105`;
  - step 200 mean loss `NaN`;
  - step 250 mean loss `NaN`.
- The process was stopped before validation generation to avoid producing a likely invalid adapter.
- Runner was updated to fail fast on non-finite training loss.
- Tokenizer audit found the root cause: with `max_length=512`, `10` of `6,495` training rows had the assistant answer fully truncated, leaving no supervised labels.
- `ChatSftDataset` now skips fully truncated-answer rows and logs the skipped count.

Tokenizer audit:

| Max Length | Used Rows | Skipped Fully Truncated Answer Rows |
| ---: | ---: | ---: |
| 512 | 6,485 | 10 |
| 768 | 6,493 | 2 |
| 1024 | 6,495 | 0 |

Skip-fix rerun:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Validation outcome:

- The corrected run skipped `10 / 6,495` training rows whose assistant answers were fully truncated at `max_length=512`.
- Used training rows: `6,485`.
- Optimiser steps: `811`.
- Final train loss: `0.1147`.
- Validation rows: `212`.
- Validation pair samples F1: `0.5991`.
- Validation pair micro F1: `0.5977`.
- Validation pair macro F1: `0.4335`.
- Validation valid JSON/schema-valid rates: `1.0000 / 1.0000`.
- Full validation run time: `6,321.4` seconds.

Test command after validation sanity:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702 --eval-split test --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --skip-training-if-adapter-exists --resume-predictions --skip-existing-predictions
```

Test outcome:

- The test command reused `adapter_final` and skipped training.
- Test rows: `281`.
- Test pair samples F1: `0.5528`.
- Test pair micro F1: `0.5552`.
- Test pair macro F1: `0.4393`.
- Test aspect samples F1: `0.6192`.
- Test sentiment accuracy when the gold aspect was predicted: `0.8944`.
- Test valid JSON/schema-valid rates: `1.0000 / 0.9964`.
- One test output had a schema conflict from conflicting sentiment; parsing still produced aggregate metrics.
- Test generation/evaluation runtime: `794.5` seconds.

Report interpretation:

- Fixed-split Qwen LoRA now provides real adaptation evidence, not only a tiny smoke test.
- The gain over Qwen indexed zero-shot is positive but small:
  - pair samples F1: `0.5374` to `0.5528`;
  - pair micro F1: `0.5300` to `0.5552`;
  - pair macro F1: `0.4374` to `0.4393`.
- This does not change the headline fixed-split ranking. The strongest local non-LLM result remains the candidate-aspect DistilBERT plus DistilBERT sentiment baseline, and Gemini/cascade results remain stronger on the fixed deployment comparison.
- The contribution is methodological and evidential:
  - the final QLoRA runner handles full fixed-split training and validation/test generation on the local GPU;
  - adapter reuse supports validation-first then test evaluation;
  - JSON reliability stays high after fine-tuning;
  - the modest fixed-split gain justifies, but does not replace, the planned full LOAO adaptation experiment.

Limitations:

- This is not all-row LOAO and does not evaluate empty-gold absence calibration.
- It is a single one-epoch local configuration, not a tuned Qwen sweep.
- Full 12-fold fine-tuned Qwen LoRA LOAO remains the major pending compute-bound experiment.
