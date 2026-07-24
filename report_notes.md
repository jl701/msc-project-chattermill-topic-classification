# Report Notes

Last updated: 2026-07-24

This file is a compact evidence ledger for dissertation/report drafting. Detailed experiment records remain in `docs/`.

## 2026-07-24 - Description Freeze and Level 3 Rich-Guidance Extension

Decision:

- The user approved and froze the exact twelve `minimal_v2` definitions.
- Add one secondary Level 3 `DD` versus `RR` comparison.
- `R` means the canonical name, exact minimal definition, 3--5
  non-corpus-derived aliases, one inclusion boundary, and one contrastive
  boundary.
- Do not add the complete nine-condition name/minimal/rich factorial.
- Keep `NN/DN/ND/DD` as the primary crossover and `L3-DD` as the difficulty
  endpoint.

Scientific control:

- `RR` changes only the two unseen candidate texts.
- The ten seen labels, training manifest, model, threshold, rows, pair
  identities, seed, budget, and metrics remain matched to `DD`.
- No FABSA review text, corpus statistics, labels, predictions, error analysis,
  or target performance entered either resource.
- Gao-style label-description training remains outside the active protocol.

Frozen hashes:

- minimal: `fc93cf27efdb64ad335f39f4a0dbdbd3dad13b1aae5de0010280931867af7d4c`;
- rich: `289ba3238cb6772f9bfda98eb73ac8a1108ba4b2bb3d23eecf72826881f415b9`;
- bound resource: `fcf546d227ad2ac52ccfa9682fc3685d2e396069ed911016f0bc12bf205f1367`;
- revised scientific protocol:
  `d7ccded514ac1cbccf337e496e039ac418698566be0c0ca0c21e18608cc40f85`.

Tracked pre-registration:

- `docs/experiments/taxonomy_level3_rich_guidance_preregistration_20260724.md`

Execution state:

- Focused resource/protocol/pipeline/governance tests: 52 passed.
- Full repository suite: 297 passed.
- Strict TF-IDF, E5, DistilBERT, and Frozen Qwen real-model synthetic smokes
  all passed `NN/DN/ND/DD/RR`, shard-resume validation, and the
  `official_data_read: false` guard.
- Execution plan v2 contains 2,611 dependency-ordered jobs; SHA-256:
  `e2a06d17ed35db8ce0afd3c67c22122dd04ba299019d11c34c6f1504f84e9685`.
- No new official validation or test model result had been inspected at this
  pre-flight checkpoint. Official test remains sealed.

## 2026-07-24 - Exact Validation Threshold-Sweep Optimisation

- The first formal TF-IDF validation scope exposed a quadratic threshold-sweep
  implementation.
- It was replaced by an exact descending sufficient-statistic sweep; the
  registered thresholds, metrics, ranking, and tie-breaks are unchanged.
- Brute-force equivalence is covered by a dedicated test; full suite:
  303 passed.
- First official scope benchmark: 34,881 pairs and 32,965 thresholds in
  approximately 0.30 seconds.
- Official test remained sealed.
- Detailed record:
  `docs/experiments/taxonomy_threshold_sweep_optimization_20260724.md`.

## 2026-07-24 - E5 Official Validation Complete

- Guarded through-validation plan: 93/93 jobs, zero failures.
- Fixed parameter SHA-256:
  `584218f2d0af9bb5703377aba010106d43dc4e12a2442fdc3113d5ff62096319`.
- Artifacts: 26 checkpoint contracts, 27 validation score scopes,
  216 score shards with 216 matching manifests, and 39 threshold transfers.
- Level 3 threshold contracts contain `NN/DN/ND/DD/RR` with one shared
  seen-only threshold.
- No official-test artifact or ledger use was created.
- Validation-only completion record:
  `docs/experiments/taxonomy_e5_validation_completion_20260724.md`.

## 2026-07-24 - Taxonomy-Generalisation Literature and Thesis Alignment

Task:

- Re-review the literature after the dissertation route changed from a
  fixed-split/LLM/router story to a Level 1--4 taxonomy-generalisation
  difficulty ladder.
- Update the local thesis before any formal local model execution.

Evidence process:

- Used original papers and official ACL, CVPR, OpenReview, arXiv, and Qwen
  sources.
- Covered candidate-conditioned ABSA, generalised zero-shot text
  classification, structured zero-shot multi-label learning, label
  descriptions and hierarchy, E5/DistilBERT/Qwen/QLoRA, and paired statistical
  evaluation.
- No official validation or test model run was performed.

Main decisions:

- Level 1 is a supplied-candidate singleton unseen-label diagnostic, not a
  label-fully-unseen task.
- Level 2 is the closest label-partially-unseen/generalised zero-shot analogue.
- Levels 3 and 4 are progressively harder multi-unseen and sub-taxonomy
  extensions.
- The exact `NN/DN/ND/DD` dual-unseen crossover is a project synthesis supported
  by adjacent description and generalised zero-shot literature, not a copied
  standard benchmark.
- Frozen Qwen versus QLoRA is interpreted as an unseen-label preservation versus
  task-adaptation comparison, motivated directly by recent multi-label
  fine-tuning evidence.
- Earlier closed-taxonomy, organisation, Gemini, and routing work remains
  historical or secondary evidence and no longer defines the thesis spine.

Tracked outputs:

- `docs/experiments/taxonomy_literature_route_alignment_20260724.md`
- `thesis/chapters/01_introduction.tex`
- `thesis/chapters/02_literature_review.tex`
- `thesis/main.tex`
- `thesis/references.bib`
- `docs/dissertation_loao_mainline_lock_2026_07_23.md`

Validation:

- 37 unique citation keys resolved against 52 unique bibliography entries;
  no key was missing or duplicated.
- The full XeLaTeX/BibTeX build completed and produced a 30-page thesis PDF
  without undefined citations, references, or LaTeX errors.
- `git diff --check` passed.
- The complete repository test suite passed: 292 tests.
- Formal local model execution was deliberately not started.

## 2026-07-23 - Approved Taxonomy-Generalisation Mainline

Decision:

- Replace the single-benchmark completion route with an increasing-difficulty
  taxonomy-generalisation mainline.
- Retain supplied-candidate LOAO as Level 1.
- Add generalized single-unseen prediction, dual-unseen asymmetric-description
  crossover, parent-group holdout, and compound organisation plus taxonomy
  shift.
- Treat the supervisor-proposed `NN/DN/ND/DD` dual-unseen crossover as a core
  dissertation experiment.
- Use human-authored minimal aspect definitions only after provenance, leakage
  rules, exact content, and SHA-256 are frozen.
- Keep canonical aspect names visible in every core condition.
- Use strict zero-label calibration for the cross-level difficulty curve.
- Keep aspect-conditioned sentiment and pair-set output for every active method.

Authority and live checklist:

- `docs/dissertation_loao_mainline_lock_2026_07_23.md`

Execution status:

- Mainline approved and documented.
- No new stress-test experiment has yet been run.
- The next unchecked block is description-resource audit and freeze.

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

## 2026-07-02 - Qwen-Local Hybrid Direction Pivot

Decision:

- The project should not continue by default with full 12-fold Qwen JSON-SFT LOAO under the current recipe.
- The fixed held-out-aspect Qwen LoRA result was only a modest improvement over Qwen zero-shot and stayed below the strongest local DistilBERT fixed baseline.
- The single-fold all-row Qwen LoRA pilot failed its validation gate on `Company brand: Competitor`.
- The first local-to-Qwen LOAO cascade diagnostic gave positive signal, so the next modelling direction is:

```text
DistilBERT = cheap calibrated gate
Qwen = semantic judge for uncertain or unfamiliar unseen-aspect cases
```

Evidence:

| System | Test Mean Pair Micro F1 | Qwen Call Rate |
| --- | ---: | ---: |
| Local DistilBERT only | 0.3128 | 0.0000 |
| Qwen zero-shot only | 0.3378 | 1.0000 |
| Local-to-Qwen global agreement gate | 0.3470 | 0.1868 |
| Optimistic per-aspect validation-selected mixed policy | 0.4131 | 0.3645 |

Completed registered follow-up:

- The strongest local DistilBERT LOAO branch was rerun with candidate score, threshold-distance, and sentiment-confidence features.
- Existing Qwen zero-shot LOAO predictions were reused; no new Qwen calls were made.
- Global validation-selected score-distance routing improved test mean pair micro F1 to `0.3800` with Qwen call rate `22.4%`.
- The per-aspect validation-selected mixed diagnostic reached `0.4186` with Qwen call rate `34.8%`.
- Sentiment-margin-only routing was weak (`0.3204` best test mean pair micro F1), so the useful uncertainty signal is the local aspect selector's distance to threshold.

Tracked source-of-truth note:

- `docs/qwen_local_hybrid_direction.md`

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

## 2026-07-02 - Planned Single-Fold All-Row Qwen LoRA LOAO Pilot

Question:

- Before spending time on all 12 Qwen LoRA LOAO folds, test one real all-row fold.
- The key issue is absence calibration, not fixed-split positive-row recognition.

Chosen fold:

- `Company brand: Competitor`

Why this fold:

- It is a known difficult semantic boundary for Qwen.
- It has enough positive rows to interpret:
  - validation positives: `86`;
  - test positives: `121`.
- It also has many empty-gold rows, so it can test over-prediction:
  - validation empty-gold rows: `971`;
  - test empty-gold rows: `1,466`.

Existing all-row validation baselines for this fold:

| System | Pair Samples F1 | Pair Micro F1 | Precision | Recall | FP Rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen zero-shot | 0.0360 | 0.2397 | 0.1645 | 0.4419 | 17.7862 |
| DistilBERT local | 0.0293 | 0.2490 | 0.1902 | 0.3605 | 12.2990 |

Planned data command:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --heldout-aspect "Company brand: Competitor" --eval-row-scope all --output-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow
```

Planned validation-only training/evaluation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_single_fold_company_brand_competitor_r8_lr1e-5_ep1_allrow_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Decision rule:

- Run validation first only.
- Run test only if validation pair micro F1 improves materially over Qwen zero-shot or shows a clear precision/false-positive improvement without recall collapse.
- If validation does not improve, do not spend more GPU time on test. Record the negative result and consider a separately pre-registered absence-aware one-candidate SFT format.

Expected contribution:

- Positive result: justifies expanding Qwen LoRA LOAO beyond one fold.
- Negative result: protects the dissertation from wasting time on a likely weak full fine-tuning sweep and motivates absence-aware SFT or calibration as future work.

Primary run early-stop note:

- Training completed successfully:
  - used rows: `7,302 / 7,314`;
  - optimiser steps: `913`;
  - train loss: `0.1178`;
  - training runtime: `7,483.0` seconds.
- Validation was stopped after `107 / 1,057` rows because the partial result showed severe over-prediction:
  - pair micro F1: `0.1681`;
  - precision: `0.0935`;
  - recall: `0.8333`;
  - FP rows / 100: `88.7850`;
  - predicted labels per example: `1.0000`;
  - valid JSON/schema-valid: `1.0000 / 1.0000`.
- This is not a completed validation result; it is a negative early diagnostic.
- Interpretation: the standard indexed prompt/data setup trained, but it was not calibrated for all-row absence. It predicted the single candidate aspect for every prefix row.

Next optimisation before retraining:

- Reuse the saved adapter and rerun validation with `indexed_conservative` prompts.
- This checks whether explicit `[]` absence guidance can reduce over-prediction without spending another full training run.
- If conservative prompting still fails, the next worthwhile optimisation is a separately pre-registered absence-aware one-candidate SFT data format, not another ordinary full-fold rerun.

Conservative prompt-only early-stop note:

- Reusing the adapter with `indexed_conservative` prompts reduced but did not solve over-prediction.
- Partial validation after `88 / 1,057` rows:
  - pair micro F1: `0.1707`;
  - precision: `0.0946`;
  - recall: `0.8750`;
  - FP rows / 100: `75.0000`;
  - predicted labels per example: `0.8409`;
  - valid JSON/schema-valid: `1.0000 / 1.0000`.
- This remains worse than Qwen zero-shot validation pair micro F1 `0.2397` and FP rows / 100 `17.7862`.

Next optimisation:

- Use absence-aware singleton SFT.
- Training data:
  - one candidate aspect per training example;
  - positive singleton examples for seen-aspect labels;
  - sampled negative singleton examples with `[]`;
  - prompt variant `indexed_conservative`;
  - train rows `24,368`, split almost 1:1 between non-empty and empty outputs.
- Training budget is capped at `913` optimiser steps to match the standard-indexed primary run.
- If this does not materially improve all-row validation, stop this single-fold optimisation and record that ordinary Qwen LoRA/SFT is insufficient without a more specialised calibration objective.

Singleton neg1 completed validation result:

- Validation rows: `1,057`.
- Positive-gold rows: `86`.
- Pair samples F1: `0.0028`.
- Pair micro F1: `0.0625`.
- Pair precision: `0.3000`.
- Pair recall: `0.0349`.
- FP rows / 100: `0.1892`.
- FN rows / 100: `7.3794`.
- Predicted labels per example: `0.0095`.
- Valid JSON/schema-valid: `1.0000 / 1.0000`.
- Training runtime: `5,863.1` seconds; validation runtime: `436.8` seconds.

Interpretation:

- This is a completed negative validation result.
- The one-candidate 1:1 positive/negative SFT branch solved the over-prediction failure but became too conservative.
- The evidence now brackets the calibration issue:
  - grouped SFT over-predicts almost every all-row validation row;
  - singleton neg1 under-predicts almost every held-out positive row.
- Before stopping the single-fold optimisation, run one mid-ratio singleton branch (`--singleton-negative-ratio 0.25`) with the same `913` optimiser-step budget.
- If this mid-ratio branch still fails to beat Qwen zero-shot validation pair micro F1 (`0.2397`) or has an unusable FP/FN trade-off, do not run test and do not start full 12-fold Qwen LoRA LOAO from the current SFT recipe.

Planned mid-ratio data command:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed_conservative --heldout-aspect "Company brand: Competitor" --eval-row-scope all --train-candidate-mode singleton --singleton-negative-ratio 0.25 --seed 13 --output-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg025_indexed_conservative
```

Planned mid-ratio validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg025_indexed_conservative --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_single_fold_company_brand_competitor_singleton_neg025_r8_lr1e-5_steps913_allrow_20260702 --eval-split validation --epochs 1 --max-train-steps 913 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Singleton neg0.25 completed validation result:

- Validation rows: `1,057`.
- Positive-gold rows: `86`.
- Pair samples F1: `0.0104`.
- Pair micro F1: `0.1803`.
- Pair precision: `0.3056`.
- Pair recall: `0.1279`.
- Pair label TP / FP / FN: `11 / 25 / 75`.
- Predicted labels: `36`.
- FP rows / 100: `1.6083`.
- FN rows / 100: `6.3387`.
- Valid JSON/schema-valid: `1.0000 / 1.0000`.
- Training runtime: `5,871.9` seconds; validation runtime: `510.8` seconds.

Interpretation:

- This is the best Qwen LoRA all-row single-fold branch so far, but still below same-fold Qwen zero-shot (`0.2397`) and local DistilBERT (`0.2490`) validation pair micro F1.
- It proves that absence-aware SFT can reduce false positives, but the current recipe still misses too many positives.
- Do not run neg0.25 test.
- Run one final recall-shift branch with `--singleton-negative-ratio 0.10`, using the same fold, prompt, seed, and `913` optimiser-step budget.
- If neg0.10 remains below the same-fold baselines, stop this single-fold optimisation and frame the result as evidence that full Qwen LoRA LOAO is not justified under the current SFT/calibration recipe.

Planned final recall-shift data command:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed_conservative --heldout-aspect "Company brand: Competitor" --eval-row-scope all --train-candidate-mode singleton --singleton-negative-ratio 0.10 --seed 13 --output-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg010_indexed_conservative
```

Planned final recall-shift validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg010_indexed_conservative --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_single_fold_company_brand_competitor_singleton_neg010_r8_lr1e-5_steps913_allrow_20260702 --eval-split validation --epochs 1 --max-train-steps 913 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

## 2026-07-02 - Planned Local-to-Qwen LOAO Cascade Diagnostic

Question:

- Test the model-division idea inspired by the fixed-split local-to-Gemini cascade:
  - DistilBERT provides a cheap calibrated local gate;
  - Qwen provides semantic judgement for unseen-topic cases.
- First run a no-new-model-call diagnostic using existing full LOAO predictions.

Inputs:

- Local DistilBERT LOAO:
  `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630/`
- Qwen zero-shot LOAO validation:
  `outputs/llm/qwen_loao_heldout_aspect_all_rows_validation_20260701/`
- Qwen zero-shot LOAO test:
  `outputs/llm/qwen_loao_heldout_aspect_all_rows_test_20260701/`

Planned command:

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --output-dir .\outputs\analysis\local_qwen_loao_cascade_20260702
```

Interpretation rule:

- If simple local/Qwen agreement or validation-selected policies improve over local-only and Qwen-only LOAO, the cascade direction has empirical signal and should be upgraded to a true score/margin uncertainty gate.
- If they do not improve, the result is still useful: Qwen's over-prediction may not be recoverable by a simple local gate without row-level confidence or a better Qwen candidate-wise judge.
- This diagnostic does not call Qwen and does not replace a future candidate-wise calibrated Qwen experiment.

Observed outcome:

- Command:

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --output-dir .\outputs\analysis\local_qwen_loao_cascade_20260702
```

- Global validation-selected policy: `aspect_agreement_qwen_sentiment`.
- Test mean pair micro F1:
  - local DistilBERT only: `0.3128`;
  - Qwen only: `0.3378`;
  - global local-to-Qwen agreement gate: `0.3470`;
  - optimistic per-aspect validation-selected mixed policy: `0.4131`.
- Global agreement gate test diagnostics:
  - precision `0.4116`;
  - recall `0.4147`;
  - FP rows / 100 `8.8112`;
  - FN rows / 100 `9.0265`;
  - Qwen call rate `0.1868`.
- Optimistic per-aspect mixed-policy diagnostics:
  - precision `0.3461`;
  - recall `0.5779`;
  - FP rows / 100 `16.1101`;
  - FN rows / 100 `3.8332`;
  - Qwen call rate `0.3645`.

Interpretation:

- The second method has a clear signal.
- A simple global confirmation gate already improves mean LOAO pair micro F1 over both local-only and Qwen-only while using Qwen on fewer than one fifth of rows.
- The per-aspect selection result suggests that some held-out aspects should be Qwen-led, while others benefit from local/Qwen agreement.
- This mirrors the earlier local-to-Gemini finding: the strongest design is not full LLM replacement but selective model division.
- The result is not yet the final method because the old local LOAO prediction files lack score/margin features. The next stronger version should use local selector score, distance to threshold, and margin to route uncertain rows to Qwen.

Singleton neg0.10 completed validation result:

- Validation rows: `1,057`.
- Positive-gold rows: `86`.
- Pair samples F1: `0.0180`.
- Pair micro F1: `0.1900`.
- Pair precision: `0.1667`.
- Pair recall: `0.2209`.
- Pair label TP / FP / FN: `19 / 95 / 67`.
- Predicted labels: `114`.
- FP rows / 100: `7.2848`.
- FN rows / 100: `4.6358`.
- Valid JSON/schema-valid: `1.0000 / 1.0000`.
- Training runtime: `6,407.7` seconds; validation runtime: `736.8` seconds.

Final decision:

- Neg0.10 is the best singleton-ratio Qwen LoRA all-row single-fold branch, but it remains below same-fold Qwen zero-shot (`0.2397`) and local DistilBERT (`0.2490`) validation pair micro F1.
- Do not run test for this branch.
- Do not launch full 12-fold Qwen LoRA LOAO with this SFT recipe.
- The thesis contribution is a validation-gated negative result: the runner and local QLoRA path work, but simple absence-aware singleton SFT does not yet justify a full fine-tuned LOAO sweep on the selected hard fold.

## 2026-07-03 - Local-to-Qwen Score-Distance Routing Result

Question:

- Can Qwen be used more effectively as a selective semantic judge for local DistilBERT uncertainty than as a full replacement or current JSON-SFT model?

Commands:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702

python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_score_margin_cascade_20260702
```

Observed outcome:

- Local rerun-only test mean pair micro F1: `0.3158`.
- Qwen-only test mean pair micro F1: `0.3378`.
- Previous global agreement gate: `0.3470`, Qwen call rate `18.7%`.
- Global validation-selected score-distance gate, `score_abs_replace_le_0.05`: `0.3800`, precision `0.3323`, recall `0.5426`, Qwen call rate `22.4%`.
- Per-aspect validation-selected mixed diagnostic: `0.4186`, precision `0.3567`, recall `0.5549`, Qwen call rate `34.8%`.
- Best sentiment-margin-only policy: `0.3204`, Qwen call rate `1.1%`.

Thesis interpretation:

- This is now the strongest evidence for the revised Qwen role: DistilBERT should act as the cheap calibrated gate and Qwen should judge locally uncertain unseen-aspect cases.
- The gain comes from aspect-selector score-distance uncertainty, not sentiment-margin uncertainty.
- This global score-distance gate was the thesis-safe selected result at this stage, and is now superseded by the later asymmetric global router; the per-aspect policy remains a useful upper-bound diagnostic.
- Full 12-fold Qwen JSON-SFT LOAO remains deferred because the current fine-tuning recipe failed its validation gate. Stronger GPU access alone is not enough to restart that path.

## 2026-07-03 - User-Prioritised Local-to-Qwen Queue Status

The local-to-Qwen work should keep this stable numbering:

1. Asymmetric score-distance router.
   - Completed on 2026-07-03.
   - Selected global policy: `score_asym_rescue_le_0.05_confirm_le_0.30`.
   - Reused existing local/Qwen LOAO predictions only.

2. Cost-quality / F1-call-rate Pareto curve.
   - Completed on 2026-07-03 and paired with item 1.
   - Public aggregate CSV: `docs/thesis_figure_data/qwen_local_qwen_loao_pareto.csv`.
   - Reports F1, precision, recall, false-positive rows, false-negative rows, and Qwen call rate across the validation policy grid and selected test comparisons.

3. Lightweight defer router.
   - Promising but lower priority.
   - Train only a small, regularised router over non-text features; high overfitting risk.

4. Fair per-aspect routing.
   - Diagnostic only for now.
   - The current per-aspect result is strong but optimistic.
   - Do not make it the main method because a real new topic will not usually have enough topic-specific validation data to choose its own policy.

5. Candidate-wise Qwen semantic judge.
   - Deferred.
   - Formulation: `review + one candidate aspect -> absent / positive / negative / neutral`.
   - This is a separate Qwen-inference route, not a continuation of items 1 and 2.

6. Aspect descriptions and boundary examples.
   - Deferred with item 5.
   - Test only as a small controlled ablation, not another broad prompt sweep.

Current thesis decision:

- Items 1 and 2 are enough for the main thesis-facing method if completed cleanly.
- The headline method should be one unified global router shared across all held-out aspects.
- Per-aspect routing remains an upper-bound diagnostic.
- Items 5 and 6 should not be run unless a supervisor specifically requests another Qwen inference method.

Deferred beyond the current queue:

- Qwen logits or absent-threshold calibration.
- DistilBERT shortlist plus Qwen judging for large candidate sets.
- Gatekeeper-style confidence tuning.
- Full Qwen JSON-SFT LOAO under the current grouped/singleton recipe.

## 2026-07-03 - Local-to-Qwen Asymmetric Router And Pareto Result

Command:

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_asymmetric_score_router_20260703 --public-pareto-csv .\docs\thesis_figure_data\qwen_local_qwen_loao_pareto.csv --public-selected-csv .\docs\thesis_figure_data\qwen_local_qwen_loao_selected.csv
```

Inputs:

- Existing local DistilBERT LOAO score export.
- Existing Qwen zero-shot LOAO validation/test predictions.
- No new Qwen calls.
- No DistilBERT retraining.

Observed result:

| System | Test Mean Pair Micro F1 | Pair Samples F1 | Precision | Recall | FP Rows / 100 | FN Rows / 100 | Qwen Call Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local-only rerun | 0.3158 | 0.0527 | 0.3449 | 0.3965 | 10.4442 | 8.9582 | 0.0000 |
| Qwen-only | 0.3378 | 0.1212 | 0.2379 | 0.8182 | 34.4150 | 1.9114 | 1.0000 |
| Agreement gate on score-export rerun | 0.3467 | 0.0513 | 0.4200 | 0.3815 | 7.1519 | 9.3153 | 0.1614 |
| `score_abs_replace_le_0.05` | 0.3800 | 0.0998 | 0.3323 | 0.5426 | 16.6667 | 4.0485 | 0.2239 |
| `score_asym_rescue_le_0.05_confirm_le_0.30` | 0.3900 | 0.0988 | 0.3530 | 0.5300 | 15.2752 | 4.2166 | 0.2905 |

Per-aspect diagnostic:

- Expanded-grid per-aspect validation-selected mixed policy:
  - pair micro F1 `0.4178`;
  - pair samples F1 `0.1015`;
  - precision `0.3511`;
  - recall `0.5650`;
  - Qwen call rate `0.3442`.

Interpretation:

- Items 1 and 2 are complete.
- The main method is the unified global router `score_asym_rescue_le_0.05_confirm_le_0.30`.
- It beats the previous global score-distance gate on the primary LOAO detection metric (`0.3900` versus `0.3800` pair micro F1).
- The Pareto table shows that validation-selected selective Qwen use dominates always-Qwen for pair micro F1 at much lower call rates.
- Qwen-only still has higher pair samples F1, but its precision and false-positive rows are much worse; this should be framed as a precision/recall/cost trade-off, not as a universal metric win.
- Per-aspect routing remains an upper-bound diagnostic, not the deployable method.

Tracked public outputs:

- `docs/thesis_figure_data/qwen_local_qwen_loao_pareto.csv`
- `docs/thesis_figure_data/qwen_local_qwen_loao_selected.csv`
