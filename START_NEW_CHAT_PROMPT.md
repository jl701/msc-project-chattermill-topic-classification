# Prompt For Continuing In A New Chat

Use this prompt when opening a fresh Codex/ChatGPT conversation for the project.

```text
I am continuing my UCL MSc project with Chattermill.

Please communicate with me in Chinese, but keep all code, comments, docstrings, README content, and project documentation in English. The code will be reviewed by Aji, so keep implementation concise, readable, and easy to inspect.

Local workspace path options:
- Current/new machine: D:\Msc_Project
- Previous/alternate machine: C:\Msc_DSML\Msc_Project

Path fallback rule:
- Use `D:\Msc_Project` first on the current machine.
- If a `D:\Msc_Project` path does not exist, replace only the root with `C:\Msc_DSML\Msc_Project` and retry.

GitHub working folder options:
- D:\Msc_Project\msc-project-chattermill-topic-classification
- C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification

Private GitHub repo:
https://github.com/jl701/msc-project-chattermill-topic-classification

Local FABSA export options:
- D:\Msc_Project\Project_Preparation\Public_Datasets\FABSA
- C:\Msc_DSML\Msc_Project\Project_Preparation\Public_Datasets\FABSA

Project title:
Open-vocabulary Topic Classification with LLMs

Before doing new work, please inspect these files. If a `D:\Msc_Project` path is unavailable, use the path fallback rule above:

1. D:\Msc_Project\msc-project-chattermill-topic-classification\PROJECT_OVERVIEW.md
2. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\aji_feedback_2026_06_16.md
3. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\evaluation_protocol.md
4. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\closed_topic_baselines.md
5. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\generalisation_baselines.md
6. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\qwen_feasibility.md
7. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\handoff_notes_2026_06_16.md
8. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\heldout_aspect_error_analysis.md
9. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\aji_updates_2026_06_21.md
10. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\experiment_log.md
11. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\loao_heldout_aspect.md
12. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\non_llm_open_topic_baseline.md
13. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\next_stage_and_literature_review_plan.md
14. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\literature_review_scoping_2026_06_29.md
15. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\aji_updates_2026_06_29.md
16. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\dissertation_internal_spec.md
17. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\literature_review_matrix.md
18. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\gemini_candidate_label_baseline.md
19. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\gemini_error_analysis.md
20. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\local_gemini_cascade.md
21. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\tasks_1_to_3_thesis_prep.md
22. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\llm_next_experiment_directions.md
23. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\experiment_reproducibility_register.md
24. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\thesis_completion_roadmap.md
25. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\qualitative_error_taxonomy.md
26. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\thesis_result_tables.md
27. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\qwen_lora_loao_launch_plan.md
28. D:\Msc_Project\msc-project-chattermill-topic-classification\docs\qwen_local_hybrid_direction.md

Immediate instruction for the new chat:

Before implementing anything, perform a strict audit of the current progress and the proposed next direction. Do not merely continue from the previous plan. Read the docs listed above and then check the actual code paths. Challenge the project as if reviewing an MSc methods section:

1. Verify whether the current split protocols match the research question and Aji's feedback.
2. Verify whether the metrics are being used and interpreted correctly, especially pair samples F1, pair micro F1, pair macro F1, empty-gold rows in all-row LOAO, and positive-row LOAO as a sentiment-only diagnostic.
3. Verify whether the current reported results are comparable or not comparable across fixed three-aspect held-out evaluation, all-row LOAO, positive-row LOAO, closed-topic, and held-out organisation.
4. Check whether the latest conclusion is logically sound: DistilBERT aspect-conditioned sentiment improves the controlled lexical sentiment ablation and the example-filtered strong fixed held-out-aspect baseline, but not label-masked training.
5. Challenge the next proposed direction after the completed Gemini/Qwen zero-shot and Qwen LoRA pilot phase. Thesis-ready tables, cascade score/margin uncertainty, Qwen SFT runner readiness, tiny Qwen LoRA smoke testing, the fixed held-out-aspect Qwen LoRA validation/test run, the full LOAO launch plan, and a single-fold all-row Qwen LoRA validation-gated pilot are now complete. The single-fold pilot did not beat the same-fold zero-shot/local validation baselines, so full 12-fold Qwen LoRA LOAO should not be launched with the current SFT recipe.
6. Read `docs/qwen_local_hybrid_direction.md` before proposing new Qwen work. The current pivot is local-to-Qwen hybrid routing: DistilBERT is the cheap calibrated gate and Qwen is the semantic judge for uncertain or unfamiliar unseen-aspect cases. Do not drift back to full Qwen JSON-SFT LOAO unless a revised objective passes a validation gate.
7. Only after this audit, propose a concrete next plan. If the plan still looks sound, proceed with implementation.

Current confirmed project context:

- The main dataset is FABSA, a public Chattermill-related customer feedback dataset.
- The task is multi-label aspect+sentiment classification.
- The prediction unit is the full aspect+sentiment pair, e.g. `Online experience: App website | negative`.
- If the aspect is correct but sentiment is wrong, the pair-level prediction is incorrect.
- Headline metric should be sample-level/pair samples F1.
- Pair micro F1 and pair macro F1 should also always be reported.
- Macro F1 is important because it exposes long-tail aspect behaviour.
- The expected Qwen score around 0.74-0.75 F1 is only a rough internal reference, not a precise target.

Aji's confirmed direction:

- Both cross-organisation/domain-shift and open-topic/new-aspect generalisation matter.
- The contrast between these two axes is itself a core result.
- If one has to be prioritised, open-topic/new-aspect is likely more novel and impactful.
- The provided FABSA train/validation/test split should be kept as the closed-topic benchmark.
- New splits have been built for held-out organisation and held-out aspect evaluation.
- Open-topic should use candidate labels at inference.
- The model should select from canonical labels and should not freely invent topic names.
- Qwen zero-shot full all-row LOAO has now been completed locally.
- The final held-out-aspect Qwen LoRA runner, manifest logging, resume/skip behaviour, focused tests, tiny local QLoRA smoke test, fixed held-out-aspect QLoRA validation/test run, full 12-fold launch plan, and a single-fold all-row Qwen LoRA validation-gated pilot are complete.
- The single-fold pilot used `Company brand: Competitor`; the best singleton branch was neg0.10 with validation pair micro F1 `0.1900`, below same-fold Qwen zero-shot `0.2397` and local DistilBERT `0.2490`.
- Full 12-fold Qwen LoRA LOAO should wait for a revised absence-calibration objective, not merely a stronger GPU/storage window.
- The completed immediate Qwen direction is local-to-Qwen asymmetric score-distance routing, documented in `docs/qwen_local_hybrid_direction.md`: the unified global validation-selected rule `score_asym_rescue_le_0.05_confirm_le_0.30` reaches test mean pair micro F1 `0.3900` with Qwen call rate `29.1%`, improving over the earlier `score_abs_replace_le_0.05` gate at `0.3800`; the expanded-grid per-aspect validation-selected mixed diagnostic reaches `0.4178` with Qwen call rate `34.4%`.
- The user-confirmed Qwen hybrid priorities 1 and 2 are complete: asymmetric score-distance routing and the F1/call-rate Pareto curve. These two are sufficient for the thesis-facing contribution unless a supervisor requests more evidence. Use one unified global router as the headline method. Per-aspect routing is diagnostic only, and candidate-wise Qwen semantic judging plus aspect descriptions are deferred.
- The current completion roadmap is `docs/thesis_completion_roadmap.md`. It explicitly separates work that can be completed before GPU access from the full fine-tuned Qwen LoRA LOAO run.

Latest Aji update from 2026-06-21:

- Keep the overall held-out aspect protocol; do not redesign it from scratch.
- Before heavy Qwen fine-tuning, rotate held-out aspects. This has now been done for zero-shot Qwen across the 12 FABSA aspects; use it as the open-weight LLM LOAO robustness baseline before fine-tuning.
- Treat label-masked vs example-filtered as an ablation about incomplete-label noise. Label-masked can keep text containing a held-out aspect while removing that aspect from supervision, creating false-negative or censored-label noise. Example-filtered removes these rows, giving cleaner but smaller training data.
- The earlier lexical and candidate-aspect cross-encoder baselines used global sentiment: one document-level polarity was applied to all selected aspects. This is a limitation for FABSA because sentiment is per-aspect. A lightweight aspect-conditioned sentiment pipeline was cleaner methodologically but slightly weaker than the global sentiment baseline. A stronger DistilBERT aspect-conditioned sentiment pipeline has now been implemented and evaluated; it improves the controlled lexical sentiment ablation and the example-filtered strong fixed held-out-aspect baseline, but not label-masked training.
- Full 12-fold Qwen LoRA LOAO remains parked because the current SFT recipe failed the single-fold validation gate; stronger GPU access alone is not a sufficient launch condition. The successful Qwen follow-up so far is score-distance hybrid routing, not direct JSON-SFT replacement.
- Aji provided access to Chattermill's Gemini Vertex AI endpoint through an OpenAI-compatible API. Do not store the key in the repo. Use it for hosted LLM baselines after the LOAO robustness work is started.
- The GitHub branch issue has been fixed: remote `main` now points to the full setup commit, and local `main` tracks `origin/main`.

Current implemented split protocols:

Closed-topic:
- Use the provided FABSA split.
- Train 7,930 rows, validation 1,057 rows, test 1,587 rows.
- All splits contain the same 12 aspects.

Held-out organisation:
- Validation organisation: 600
- Test organisations: 369 and 727
- Train rows: 7,020
- Validation rows: 1,533
- Test rows: 2,021
- Organisation overlap across train/validation/test is zero.

Held-out aspect:
- Held-out aspects:
  - Account management: Account access
  - Company brand: Competitor
  - Value: Discounts promotions
- Strategy A: label-masked training.
- Strategy B: example-filtered training.
- Evaluation uses held-out labels only.
- Candidate labels are provided at inference.
- Train-vs-validation/test supervision aspect overlap is zero.

Current best local results:

Closed-topic traditional:
- Word+char TF-IDF + Linear SVM.
- Test pair samples F1: 0.7090
- Test pair micro F1: 0.7042
- Test pair macro F1: 0.4207

Closed-topic BERT-style:
- DistilBERT, square-root positive-class weighting.
- Best observed test pair samples F1: 0.7803
- Test pair micro F1: 0.7738
- Test pair macro F1: 0.5377

Held-out organisation traditional:
- Refined word+char TF-IDF + Linear SVM.
- Test pair samples F1: 0.7026
- Test pair micro F1: 0.6920
- Test pair macro F1: 0.3626

Held-out organisation BERT-style:
- DistilBERT, square-root positive-class weighting.
- Validation-selected learning rate: 6e-5
- Best epoch: 8
- Threshold: 0.44
- Test pair samples F1: 0.7575
- Test pair micro F1: 0.7600
- Test pair macro F1: 0.4035

Held-out aspect lexical lower-bound:
- Candidate-label lexical TF-IDF + global sentiment classifier.
- Label-masked test pair samples F1: 0.4698
- Example-filtered test pair samples F1: 0.4626
- A lightweight TF-IDF aspect-conditioned sentiment version has also been implemented:
  - Label-masked test pair samples F1: 0.4520
  - Example-filtered test pair samples F1: 0.4389
  - Interpretation: this fixes the global-sentiment assumption but the shallow TF-IDF sentiment classifier is empirically weaker than the global prior.
- A DistilBERT aspect-conditioned sentiment version has now been implemented:
  - Label-masked test pair samples F1: 0.4840
  - Label-masked test pair micro F1: 0.4807
  - Label-masked test pair macro F1: 0.4063
  - Example-filtered test pair samples F1: 0.4804
  - Example-filtered test pair micro F1: 0.4772
  - Example-filtered test pair macro F1: 0.3985
  - Interpretation: stronger aspect-conditioned sentiment improves the controlled lexical sentiment ablation, but the lexical aspect selector remains the bottleneck.

Held-out aspect label-aware baseline:
- Candidate-aspect DistilBERT cross-encoder + global TF-IDF Logistic Regression sentiment classifier.
- Evaluation uses held-out labels only.
- Label-masked test pair samples F1: 0.5595
- Label-masked test pair micro F1: 0.5462
- Label-masked test pair macro F1: 0.4267
- Example-filtered test pair samples F1: 0.5816
- Example-filtered test pair micro F1: 0.5646
- Example-filtered test pair macro F1: 0.4538
- Candidate-aspect DistilBERT cross-encoder + DistilBERT aspect-conditioned sentiment:
  - Label-masked, selector LR 2e-5:
    - test pair samples F1: 0.5412
    - test pair micro F1: 0.5343
    - test pair macro F1: 0.4713
  - Example-filtered, selector LR 2e-5:
    - test pair samples F1: 0.6001
    - test pair micro F1: 0.5859
    - test pair macro F1: 0.4942
  - Example-filtered, selector LR 3e-5, best current non-LLM fixed held-out-aspect run:
    - test pair samples F1: 0.6071
    - test pair micro F1: 0.5917
    - test pair macro F1: 0.4890
    - test aspect samples F1: 0.6651
    - sentiment accuracy when the gold aspect is predicted: 0.9100
  - Interpretation: DistilBERT aspect-conditioned sentiment improves the cleaner example-filtered strong baseline over global sentiment, but label-masked remains weaker and should be treated as incomplete-label-noise evidence rather than the main result.

Held-out aspect Qwen zero-shot:
- Model: Qwen/Qwen3-4B-Instruct-2507
- Prompt: indexed candidate labels with `aspect_id` outputs.
- Full validation pair samples F1: 0.5753
- Full validation pair micro F1: 0.5762
- Full validation pair macro F1: 0.4514
- Full test pair samples F1: 0.5374
- Full test pair micro F1: 0.5300
- Full test pair macro F1: 0.4374
- Valid JSON rate: 1.0000
- Plain canonical string prompts often produced near-miss aspect names such as `Account management`; indexed prompts fixed this.
- This is the fixed three-aspect held-out evaluation, not LOAO robustness evidence.

Leave-one-aspect-out held-out aspect:
- Added after Aji's 2026-06-21 feedback.
- Main all-row lexical LOAO keeps all official validation/test rows and filters gold labels to one held-out aspect, allowing empty predictions.
- This avoids the trivial one-candidate positive-only setup and exposes false positives.
- Sample-F1-selected all-row LOAO test spread across 12 aspects:
  - label-masked pair samples F1 mean 0.1299, pair micro F1 mean 0.2304, precision 0.1458, recall 0.8857, false-positive rows per 100 reviews 77.5362.
  - example-filtered pair samples F1 mean 0.1288, pair micro F1 mean 0.2345, precision 0.1495, recall 0.8777, false-positive rows per 100 reviews 77.4207.
- Micro-F1-selected all-row LOAO is the preferred detection diagnostic:
  - label-masked pair samples F1 mean 0.0926, pair micro F1 mean 0.3635, precision 0.3912, recall 0.4883, false-positive rows per 100 reviews 18.5728.
  - example-filtered pair samples F1 mean 0.0903, pair micro F1 mean 0.3780, precision 0.3778, recall 0.4895, false-positive rows per 100 reviews 17.4753.
- Micro-F1-selected all-row LOAO with lightweight aspect-conditioned sentiment:
  - label-masked pair samples F1 mean 0.0883, pair micro F1 mean 0.3511, precision 0.3811, recall 0.4700, false-positive rows per 100 reviews 18.5728.
  - example-filtered pair samples F1 mean 0.0842, pair micro F1 mean 0.3576, precision 0.3627, recall 0.4541, false-positive rows per 100 reviews 16.7034.
- The headline metric remains pair samples F1, but all-row threshold selection should not rely on sample-F1 alone because it does not reward true-negative empty rows and can hide false positives.
- Positive-row LOAO is much higher, about 0.88 test pair samples F1, but aspect F1 is trivially 1.0000 and it should be treated as a sentiment diagnostic.
- Positive-row LOAO with lightweight aspect-conditioned sentiment is about 0.86 label-masked / 0.84 example-filtered test pair samples F1, slightly below global sentiment.
- Full DistilBERT cross-encoder LOAO with DistilBERT aspect-conditioned sentiment is now complete for `example_filtered`, all-row evaluation, and pair-micro-F1 threshold selection.
  - selector LR `3e-5`: pair samples F1 mean 0.0550, pair micro F1 mean 0.3128, precision 0.3345, recall 0.4315, pair macro F1 mean 0.2285, aspect micro F1 mean 0.7862, sentiment accuracy when gold aspect predicted 0.9319.
  - selector LR `2e-5`: pair samples F1 mean 0.0577, pair micro F1 mean 0.2941, precision 0.3021, recall 0.3921, pair macro F1 mean 0.2250, aspect micro F1 mean 0.7840, sentiment accuracy when gold aspect predicted 0.9301.
  - The LR `3e-5` run is the preferred DistilBERT LOAO setting by pair micro F1, but it does not beat the documented lexical global-sentiment micro-selected LOAO lower bound.
  - Interpretation: the fixed three-aspect local DistilBERT result is strong, but full LOAO exposes weak unseen-aspect relevance detection and threshold calibration. Sentiment is not the main bottleneck.
- Full Qwen zero-shot LOAO is now complete:
  - model: `Qwen/Qwen3-4B-Instruct-2507`
  - loading: local 4-bit bitsandbytes NF4
  - prompt: indexed candidate labels, one held-out aspect per fold, JSON array with `aspect_id`
  - evaluation: all official validation/test rows, gold labels filtered to the held-out aspect, empty predictions allowed
  - validation mean pair samples F1: 0.1194
  - validation mean pair micro F1: 0.3293
  - validation mean pair macro F1: 0.2340
  - validation valid JSON / schema valid: 1.0000 / 0.9961
  - test mean pair samples F1: 0.1212
  - test mean pair micro F1: 0.3378
  - test mean pair macro F1: 0.2412
  - test mean precision / recall: 0.2379 / 0.8182
  - test false-positive rows per 100 reviews: 34.4150
  - test valid JSON / schema valid: 1.0000 / 0.9955
  - hardest test aspect by pair micro F1: `Company brand: Reviews` at 0.0513, with 64.2722 false-positive rows per 100 reviews
  - strongest test aspect by pair micro F1: `Online experience: App website` at 0.6011
  - positive-gold-row diagnostic from the same predictions is much stronger: test mean pair samples F1 0.8194 and pair micro F1 0.8659
  - detailed comparison with the preferred DistilBERT LOAO row is recorded in `docs/qwen_loao_experiment_analysis.md`
  - Qwen and DistilBERT split the 12 aspects 6/6 by pair micro F1; Qwen wins broad semantic aspects but loses on several support-channel, account, and value aspects because of empty-gold over-prediction
  - interpretation: Qwen zero-shot has strong semantic matching when the held-out aspect is present, but it over-predicts on empty-gold rows and needs fine-tuning/calibration for all-row open-topic robustness.
- See `docs/loao_heldout_aspect.md`.

Held-out aspect error analysis:
- DistilBERT cross-encoder mainly over-predicts candidate aspects.
- Qwen indexed zero-shot has more exact rows but also some missed-all rows.
- Qwen is strongest on `Account management: Account access`, weaker on `Company brand: Competitor`, especially positive competitor mentions.
- See `docs/heldout_aspect_error_analysis.md`.

Qwen feasibility:
- Model: Qwen/Qwen3-4B-Instruct-2507
- Previous local pilot was on an RTX 5050 Laptop GPU, 8 GB VRAM.
- Current local machine has NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM.
- 4-bit QLoRA with rank-8 LoRA runs locally.
- Results are only on the first 100 validation rows, not full validation/test.
- Zero-shot first-100 validation pair micro F1: 0.541
- Best local LoRA pilot first-100 validation pair micro F1: 0.762
- Local zero-shot full all-row Qwen LOAO is now complete; use local Qwen for prompt/zero-shot baselines and smoke tests, but still use stronger GPU access for full fine-tuning.
- Final held-out-aspect Qwen LoRA runner is implemented as `scripts/run_qwen_lora_heldout_aspect.py`.
- The runner consumes indexed held-out-aspect SFT JSONL, supports 4-bit QLoRA, configurable LoRA parameters, train/validation/test splits, manifest logging, adapter resume, prediction resume/skip, canonical `aspect_id` parsing, and existing evaluation metrics.
- Tiny local held-out-aspect QLoRA smoke test completed:
  - command output: `outputs/qwen_lora_heldout_aspect_tiny_smoke_evalmode_20260702/` (ignored)
  - 4 train rows, 1 validation row, 1 test row, one train step
  - validation/test valid JSON and schema-valid rates: 1.0000
  - adapter saved under ignored `adapter_final/`
- Fixed held-out-aspect QLoRA validation/test run completed:
  - output: `outputs/llm/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702/` (ignored)
  - strategy: `example_filtered`
  - LoRA: rank 8, alpha 16, dropout 0.05
  - LR: `1e-5`, 1 epoch, 4-bit QLoRA
  - test pair samples F1: 0.5528
  - test pair micro F1: 0.5552
  - test pair macro F1: 0.4393
  - test valid JSON/schema-valid rates: 1.0000 / 0.9964
  - interpretation: modest fixed-split adaptation gain over Qwen zero-shot, still below the strongest local fixed baseline, and not LOAO robustness evidence
- Full 12-fold Qwen LoRA LOAO launch templates are in `docs/qwen_lora_loao_launch_plan.md`.

Gemini / Vertex AI access:
- Aji provided an OpenAI-compatible endpoint:
  - `OPENAI_BASE_URL=<private-openai-compatible-endpoint>/v1`
  - `OPENAI_API_KEY` should be set locally only; never commit it.
- Recommended models:
  - `vertex_ai/gemini-2.5-flash` as the default fast/cheap baseline.
  - `vertex_ai/gemini-2.5-pro` as the most capable baseline.
  - `vertex_ai/gemini-2.5-flash-lite` as the cheapest/lowest-latency baseline.
- The `vertex_ai/` prefix is required.
- Models are served in `europe-west4`; unsupported-region models can return 404.
- Initial budget is $100 per key; exhausted budget returns 429.
- Use enough `max_tokens` because Gemini 2.5 may spend tokens thinking first.
- The old 2026-06-22 preference to postpone Gemini has been superseded by the 2026-07-01 request to implement the hosted Gemini candidate-label baseline.
- Gemini runner status:
  - implemented: `scripts/run_gemini_heldout_aspect.py`
  - shared parser/schema utilities: `src/msc_project/llm/candidate_label.py`
  - tests: `tests/test_llm_candidate_label.py`
  - documentation: `docs/gemini_candidate_label_baseline.md`
  - real API evaluation: completed for fixed held-out-aspect validation/test
  - final fixed configuration: `vertex_ai/gemini-2.5-flash`, `indexed`, `response_format=json_schema`, `max_tokens=2048`
- Gemini fixed held-out-aspect result:
  - validation pair samples F1: 0.6146
  - validation pair micro F1: 0.6446
  - validation pair macro F1: 0.4830
  - validation valid JSON / schema-valid: 1.0000 / 0.9953
  - test pair samples F1: 0.6071
  - test pair micro F1: 0.6541
  - test pair macro F1: 0.5547
  - test aspect samples F1: 0.6747
  - test valid JSON / schema-valid: 1.0000 / 1.0000
  - test mean latency: 2.3721 seconds/example
  - test tokens: 56,338 input, 113,067 output/completion, 102,574 reasoning, 169,405 total
- Gemini fixed-split error analysis:
  - documented in `docs/gemini_error_analysis.md`
  - Gemini and the strongest local DistilBERT pipeline both score 0.6071 test pair samples F1, but Gemini has higher pair precision (0.6475 vs 0.5333) and fewer aspect over-prediction rows (80 vs 119)
  - Gemini predicts fewer labels per row (1.0498 vs 1.2811) and has more empty predictions (42 vs 0)
  - the main Gemini recall weakness is `Company brand: Competitor`, especially positive competitor mentions
- Gemini Pro status:
  - completed a small 50-row fixed test subset after the user explicitly authorised API use beyond the earlier environment-variable-only restriction
  - command: test split, `--limit 50 --sample --seed 13`, indexed JSON-schema, `max_tokens=2048`
  - Pro subset result: pair samples F1 0.6933, pair micro F1 0.7379, pair macro F1 0.5250, aspect samples F1 0.7933
  - Flash on the same subset: pair samples F1 0.6360, pair micro F1 0.6731, pair macro F1 0.4627, aspect samples F1 0.7560
  - Pro same-subset latency/cost trade-off: about 2.25x slower and 5.53x more expensive than Flash
  - full fixed-split Pro validation/test is now complete:
    - validation pair samples F1 0.7270
    - test pair samples F1 0.7141
    - test pair micro F1 0.7425
    - test pair macro F1 0.6287
    - test mean latency 5.6510 seconds/example
    - validation + test approximate cost $2.9781
  - do not run full Gemini Pro LOAO unless an explicit dissertation-value justification is given first
- Gemini Flash-Lite status:
  - full fixed-split Flash-Lite validation/test is complete:
    - validation pair samples F1 0.5876
    - test pair samples F1 0.5516
    - test pair micro F1 0.5872
    - test pair macro F1 0.4876
    - test mean latency 0.3719 seconds/example
    - validation + test approximate cost $0.0171
  - Flash-Lite is the cheapest/fastest hosted point; Pro is strongest; Flash is the balanced hosted baseline
- Local-to-Gemini cascade status:
  - implemented in `scripts/run_local_gemini_cascade.py`
  - shared helpers in `src/msc_project/evaluation/cascade.py`
  - tests in `tests/test_cascade_evaluation.py`
  - documentation in `docs/local_gemini_cascade.md`
  - policy selection uses validation-only local reliability features, not test labels
  - each escalator sweep used 10,578 candidate policies with 1 percentage point ranked escalation steps
  - local -> Flash-Lite cascade:
    - test pair samples F1 0.6679
    - test pair micro F1 0.6579
    - test pair macro F1 0.5401
    - call rate 50.9%
    - test Gemini cost about $0.0052
  - local -> Flash cascade:
    - test pair samples F1 0.7459
    - test pair micro F1 0.7348
    - test pair macro F1 0.6223
    - call rate 90.0%
    - test Gemini cost about $0.2789
  - local -> Pro cascade:
    - test pair samples F1 0.8102
    - test pair micro F1 0.7955
    - test pair macro F1 0.6809
    - call rate 90.0%
    - test Gemini cost about $1.5421
  - Pro cascade deep-dive:
    - implemented in `scripts/analyse_local_gemini_cascade.py`
    - documented in `docs/local_gemini_cascade.md`
    - pure Pro test pair samples F1 is 0.7141, so local -> Pro is better than pure Pro by +0.0961
    - reason: the winning `gemini_nonempty_else_local` policy keeps Pro on most escalated rows but falls back to local for Pro-empty/non-escalated rows
    - pure Pro has 28 empty prediction rows; cascade recovers all 28 with local predictions
    - cascade versus pure Pro: 29 better rows, 250 equal rows, 2 worse rows
    - pair false negatives drop from 67 to 40 while false positives remain nearly flat, 87 to 88
    - exact-match rows rise from 167 for pure Pro to 194 for the cascade
  - this is the strongest fixed-split system result so far, but it is still fixed three-aspect evidence rather than LOAO robustness evidence
- Gemini/local dissertation-value roadmap:
  - fixed-split hosted Pareto baselines are now complete: Gemini Flash-Lite, Flash, and Pro
  - local-to-Gemini uncertainty cascade is now complete
  - Gemini-generated aspect descriptions are now complete
  - label-only descriptions improve Flash-Lite test pair samples F1 from 0.5516 to 0.5925, pair micro F1 from 0.5872 to 0.6263, pair macro F1 from 0.4876 to 0.5691, and aspect samples F1 from 0.6062 to 0.6625
  - Flash descriptions improve validation but reduce test pair samples F1, while slightly improving test micro/macro F1
  - Pro 50-row validation diagnostic worsens primary pair samples F1, so full Pro descriptions are not justified
  - details are in `docs/gemini_aspect_descriptions.md`
  - Tasks 1-3 are consolidated for thesis/task-4 handoff in `docs/tasks_1_to_3_thesis_prep.md`
  - the next LLM roadmap is recorded in `docs/llm_next_experiment_directions.md`
  - the active completion roadmap is recorded in `docs/thesis_completion_roadmap.md`
  - Gemini-assisted qualitative error taxonomy is complete in `docs/qualitative_error_taxonomy.md`
  - the taxonomy used a small Gemini Pro draft only as an assistant; final categories are manually consolidated, and raw snippets/prompts/drafts remain ignored under `outputs/`
  - completed pre-Qwen-LoRA-full-LOAO work:
    - thesis-ready result tables and figure data
    - cascade score/margin uncertainty using existing Gemini predictions
    - Qwen LoRA SFT runner readiness and smoke test
    - fixed held-out-aspect Qwen LoRA validation/test run
    - full Qwen LoRA all-row LOAO launch plan
    - single-fold all-row Qwen LoRA validation-gated pilot on `Company brand: Competitor`
  - remaining launch-gate work:
    - revised absence-calibration/training objective for Qwen all-row LOAO
    - target GPU, storage, package versions, and data-transfer rules confirmation after the revised objective passes validation
    - full Qwen LoRA all-row LOAO only after the revised recipe passes a validation gate
  - full Gemini LOAO is not the default because estimated all-row LOAO cost/latency is high relative to the expected dissertation value and the current robustness spine already uses local DistilBERT LOAO plus Qwen zero-shot LOAO
- Important Gemini finding: `max_tokens=512` caused truncated JSON because Gemini spent most completion tokens thinking first. Use `max_tokens=2048` for this prompt unless a later sweep proves a cheaper reliable setting.
- The runner defaults to `response_format=json_schema`, JSON object wrapper `{"labels": [...]}`, and indexed candidate IDs; it can fall back to plain JSON if the endpoint rejects response format.

Latest user decision and completed local baseline phase from 2026-06-27:

- The requested non-LLM phase has been completed up to **candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment**.
- The result should be treated sceptically:
  - DistilBERT aspect-conditioned sentiment clearly improves the controlled lexical sentiment ablation.
  - It improves the strong `example_filtered` candidate-aspect pipeline from 0.5816 to 0.6071 test pair samples F1.
  - It does not improve `label_masked`; the best comparable label-masked run is 0.5412 versus the older global-sentiment result of 0.5595.
  - This supports using `example_filtered` as the cleaner fixed held-out-aspect result and retaining `label_masked` as an incomplete-label-noise ablation.
- The strongest current non-LLM fixed held-out-aspect result is:
  - candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment
  - strategy: `example_filtered`
  - sentiment LR: `2e-5`, 3 epochs, balanced class weights, validation accuracy selection
  - selector LR: `3e-5`, 3 epochs, 3 negatives per positive, validation pair samples F1 selection
  - test pair samples F1: 0.6071
  - test pair micro F1: 0.5917
  - test pair macro F1: 0.4890
  - test aspect samples F1: 0.6651
- Small tuning attempts have already been checked:
  - selector LR `4e-5` was worse
  - 5 selector epochs was worse
  - top-2 prediction cap was slightly worse
  - label-masked LR `3e-5` was worse
- Gemini or Qwen full fine-tuning/evaluation should be treated as the next major phase, not part of the completed local non-LLM baseline phase.
- Gemini Flash fixed held-out-aspect is now completed and should be compared against the current local non-LLM result:
  - Gemini Flash matches test pair samples F1 at 0.6071 and improves pair micro/macro F1.
  - This is not LOAO robustness evidence and should not be over-claimed.
- Detailed write-up:
  - `docs/non_llm_open_topic_baseline.md`
  - `docs/qwen_loao_experiment_analysis.md`
  - `docs/next_stage_and_literature_review_plan.md`
  - `docs/gemini_candidate_label_baseline.md`
  - `docs/gemini_error_analysis.md`
  - `docs/local_gemini_cascade.md`

Latest Aji/literature-review framing update from 2026-06-29:

- Aji confirmed that the leave-one-aspect-out setup is the part that matters most for the open-topic claim.
- For Gemini, use `response_format` / JSON mode if the endpoint supports it, rather than only measuring valid JSON rate.
- Gemini cost comparisons must include reasoning/thinking tokens, not just visible output tokens.
- The dissertation should be framed with an hourglass shape:
  - broad opening: customer feedback analytics, review mining, ABSA, and multi-label topic/sentiment classification
  - narrow waist: candidate-label open-topic aspect+sentiment classification on FABSA, evaluated through held-out aspects and cross-organisation shift
  - broad ending: practical open-vocabulary feedback analytics systems combining local encoders, open LLMs, and hosted LLMs under accuracy/cost/latency/governance constraints
- First local literature scoping is recorded in:
  - `docs/literature_review_scoping_2026_06_29.md`
  - `docs/aji_updates_2026_06_29.md`
- Stage 1 and Stage 2 pre-experiment documents are recorded in:
  - `docs/dissertation_internal_spec.md`
  - `docs/literature_review_matrix.md`

Thesis LaTeX workflow:

- Dissertation prose should now be edited directly in LaTeX, not as parallel Markdown drafts.
- Main entry point: `thesis/main.tex`.
- Literature review draft: `thesis/chapters/02_literature_review.tex`.
- Bibliography: `thesis/references.bib`.
- Pro-model literature-review feedback and revision checklist: `thesis/notes/pro_literature_review_feedback_2026_06_30.tex`.
- Workflow rule: `docs/thesis_latex_workflow.md`.
- If asked to revise the dissertation, literature review, citations, chapter structure, or thesis wording, edit the LaTeX source first and only use `docs/` for supporting logs/plans.

Important repository hygiene:

- Do not commit internal Chattermill data, credentials, API keys, model checkpoints, or confidential outputs.
- Do not commit the `outputs/` directory.
- Public FABSA can be inspected locally, but do not commit full copied data files unless explicitly approved.
- Current project files may still be uncommitted locally, so check `git status` before making changes.

Experiment logging rule:

- After every meaningful experiment, implementation change, or evaluation run, update `docs/experiment_log.md`.
- Record the code/protocol change, dataset split, model or baseline, exact command, output location, headline/supporting metrics, interpretation, limitations, and next step.
- Use `docs/experiment_reproducibility_register.md` as the consolidated cross-experiment reproducibility index. It audits older runs, records canonical commands/parameters, and lists remaining provenance caveats.
- Future experiments should ideally save exact command, git commit, parsed CLI args, hardware/API metadata, and package versions alongside `summary.json`.
- Every future experiment must be closed by documenting the exact parameters/configuration, updating the relevant project docs, running appropriate validation checks, and committing/pushing safe tracked changes to GitHub.
- Do not leave results only in local `outputs/`; summarise the result in tracked documentation before stopping.
- Commit and push only clean code, documentation, and non-sensitive configs. Never commit or push local generated outputs, raw data, credentials, API keys, checkpoints, model weights, raw prediction files, or confidential material.
- If GitHub push is blocked by network/auth/conflict issues, record the blocker in `docs/experiment_log.md` and push as soon as the blocker is resolved.

Useful commands:

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests
python .\scripts\analyse_split_candidates.py
python .\scripts\build_fabsa_splits.py
python .\scripts\run_generalisation_baselines.py
python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --sentiment-mode global --output-dir .\outputs\baselines\generalisation_global_sentiment_rerun
python .\scripts\run_generalisation_baselines.py --protocol heldout-org --refined-cross-org --output-dir .\outputs\baselines\generalisation_refined
python .\scripts\run_transformer_baseline.py --protocol heldout-org --epochs 10 --batch-size 16 --learning-rate 6e-5 --pos-weight sqrt
python .\scripts\run_aspect_label_aware_baseline.py --strategy both --sentiment-mode global --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3
python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --output-dir .\outputs\baselines\generalisation_transformer_sentiment_lr2e-5_ep3_balanced_accuracy
python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode global --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode global --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows_micro_selection
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_micro_selection
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --eval-row-scope containing_heldout --ensure-one --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_positive_rows
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_YYYYMMDD
python .\scripts\run_qwen_heldout_aspect_smoke.py --split validation --limit 10000 --load-in-4bit --prompt-variant indexed
python .\scripts\run_qwen_heldout_aspect_smoke.py --split test --limit 10000 --load-in-4bit --prompt-variant indexed
python .\scripts\run_qwen_loao_heldout_aspect.py --split validation --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_YYYYMMDD
python .\scripts\run_qwen_loao_heldout_aspect.py --split test --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_YYYYMMDD
python .\scripts\analyse_qwen_loao_predictions.py --validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_YYYYMMDD --test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_YYYYMMDD --output-dir .\outputs\analysis\qwen_loao_positive_diagnostic_YYYYMMDD
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy both --prompt-variant indexed
python .\scripts\run_gemini_heldout_aspect.py --dry-run --split validation --limit 2 --response-format json_schema --output-dir .\outputs\llm\gemini_candidate_label_dry_run_check
python .\scripts\run_gemini_heldout_aspect.py --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_YYYYMMDD_HHMMSS
python .\scripts\analyse_gemini_heldout_aspect_errors.py --output-dir .\outputs\analysis\gemini_error_analysis
python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-pro --split test --limit 50 --sample --seed 13 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_YYYYMMDD_HHMM_pro_test50
python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full --input-cost-per-1m 0.30 --output-cost-per-1m 2.50 --output-dir .\outputs\analysis\local_gemini_cascade_flash_grid1 --rank-rate-step 1
```

Recommended next steps:

1. First inspect the repo and confirm the current state with `git status`.
2. Perform the strict audit described near the top of this prompt before implementing the next task.
3. LOAO lexical evaluation, lightweight aspect-conditioned sentiment, and the full DistilBERT selector + DistilBERT aspect-conditioned sentiment LOAO robustness check have been implemented and documented.
4. Qwen zero-shot full all-row LOAO is now implemented, run, and documented as the local open-weight LLM robustness baseline before fine-tuning.
5. Keep using the LOAO all-row view for robustness checks and the positive-row view only as a sentiment diagnostic.
6. Treat candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment as the current strongest non-LLM fixed held-out-aspect baseline.
7. Treat the completed DistilBERT LOAO result as a robustness caveat, not as a new headline model improvement.
8. Treat Gemini Flash as the completed hosted fixed held-out-aspect baseline, but do not run full Gemini LOAO unless the cost/benefit is explicitly justified.
9. The local-to-Gemini cascade is complete; follow `docs/llm_next_experiment_directions.md` with the new Qwen LOAO result in mind.
10. Gemini-generated aspect descriptions are complete; do not rerun the same Task 4 API work unless a new variant or thesis question is explicitly requested.
11. Use `docs/thesis_completion_roadmap.md` and `docs/qwen_local_hybrid_direction.md` as the current task order. Thesis-ready tables/figure data, cascade score/margin uncertainty, Qwen LoRA runner readiness, tiny smoke testing, the fixed held-out-aspect Qwen LoRA validation/test run, the full Qwen LoRA LOAO launch plan, the single-fold all-row Qwen LoRA validation-gated pilot, local-to-Qwen score-distance routing, asymmetric global routing, and F1/call-rate Pareto reporting are complete. Treat the unified global router as the main thesis method; keep per-aspect routing diagnostic only; defer lightweight learned routing, candidate-wise Qwen judging, and descriptions unless a supervisor requests more evidence.
12. Treat sampled Gemini LOAO as optional fallback or supervisor-requested work, not the default next experiment.
13. If I ask to publish changes, commit/push only clean code and documentation, without committing outputs, data, credentials, checkpoints, or generated artifacts.

Please start by summarising what you find in the current docs and repo state, then perform the strict audit, then propose the next concrete plan before implementing.
```
