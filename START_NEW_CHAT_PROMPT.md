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

Immediate instruction for the new chat:

Before implementing anything, perform a strict audit of the current progress and the proposed next direction. Do not merely continue from the previous plan. Read the docs listed above and then check the actual code paths. Challenge the project as if reviewing an MSc methods section:

1. Verify whether the current split protocols match the research question and Aji's feedback.
2. Verify whether the metrics are being used and interpreted correctly, especially pair samples F1, pair micro F1, pair macro F1, empty-gold rows in all-row LOAO, and positive-row LOAO as a sentiment-only diagnostic.
3. Verify whether the current reported results are comparable or not comparable across fixed three-aspect held-out evaluation, all-row LOAO, positive-row LOAO, closed-topic, and held-out organisation.
4. Check whether the latest conclusion is logically sound: DistilBERT aspect-conditioned sentiment improves the controlled lexical sentiment ablation and the example-filtered strong fixed held-out-aspect baseline, but not label-masked training.
5. Challenge the next proposed direction after the completed non-LLM baseline phase. Decide whether the next work should be a hosted Gemini candidate-label baseline, Qwen fine-tuning/evaluation, or a small robustness diagnostic before moving to LLMs.
6. Only after this audit, propose a concrete next plan. If the plan still looks sound, proceed with implementation.

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
- Qwen full experiments should wait for better GPU access; local QLoRA is only for smoke tests.

Latest Aji update from 2026-06-21:

- Keep the overall held-out aspect protocol; do not redesign it from scratch.
- Before heavy Qwen fine-tuning, rotate held-out aspects. The next robustness experiment should be leave-one-aspect-out across the 12 FABSA aspects and report the spread.
- Treat label-masked vs example-filtered as an ablation about incomplete-label noise. Label-masked can keep text containing a held-out aspect while removing that aspect from supervision, creating false-negative or censored-label noise. Example-filtered removes these rows, giving cleaner but smaller training data.
- The earlier lexical and candidate-aspect cross-encoder baselines used global sentiment: one document-level polarity was applied to all selected aspects. This is a limitation for FABSA because sentiment is per-aspect. A lightweight aspect-conditioned sentiment pipeline was cleaner methodologically but slightly weaker than the global sentiment baseline. A stronger DistilBERT aspect-conditioned sentiment pipeline has now been implemented and evaluated; it improves the controlled lexical sentiment ablation and the example-filtered strong fixed held-out-aspect baseline, but not label-masked training.
- Full Qwen fine-tuning remains parked until stronger GPU access is clearer.
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
- Full DistilBERT cross-encoder LOAO was not completed locally; a single full fold did not finish within 30 minutes on the 6 GB GPU. A tiny cross-encoder smoke test passed.
- See `docs/loao_heldout_aspect.md`.

Held-out aspect error analysis:
- DistilBERT cross-encoder mainly over-predicts candidate aspects.
- Qwen indexed zero-shot has more exact rows but also some missed-all rows.
- Qwen is strongest on `Account management: Account access`, weaker on `Company brand: Competitor`, especially positive competitor mentions.
- See `docs/heldout_aspect_error_analysis.md`.

Qwen feasibility:
- Model: Qwen/Qwen3-4B-Instruct-2507
- Previous local pilot was on an RTX 5050 Laptop GPU, 8 GB VRAM.
- Current new machine has NVIDIA GeForce GTX 1660 Ti with Max-Q Design, 6 GB VRAM.
- 4-bit QLoRA with rank-8 LoRA runs locally.
- Results are only on the first 100 validation rows, not full validation/test.
- Zero-shot first-100 validation pair micro F1: 0.541
- Best local LoRA pilot first-100 validation pair micro F1: 0.762
- Use local Qwen only for smoke tests until proper GPU access is available.

Gemini / Vertex AI access:
- Aji provided an OpenAI-compatible endpoint:
  - `OPENAI_BASE_URL=https://llm-api.datascience.chattermill.xyz/v1`
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
  - this is the strongest fixed-split system result so far, but it is still fixed three-aspect evidence rather than LOAO robustness evidence
- Gemini/local dissertation-value roadmap:
  - fixed-split hosted Pareto baselines are now complete: Gemini Flash-Lite, Flash, and Pro
  - local-to-Gemini uncertainty cascade is now complete
  - next test Gemini-generated candidate-aspect descriptions as label-representation support, without validation/test leakage
  - then use Gemini Pro for qualitative error-taxonomy assistance, with manual review and no automatic metric claims
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
- Commit only concise documentation summaries, not local generated outputs or credentials.

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
python .\scripts\run_qwen_heldout_aspect_smoke.py --split validation --limit 10000 --load-in-4bit --prompt-variant indexed
python .\scripts\run_qwen_heldout_aspect_smoke.py --split test --limit 10000 --load-in-4bit --prompt-variant indexed
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
3. LOAO lexical evaluation, lightweight aspect-conditioned sentiment, and DistilBERT aspect-conditioned sentiment have been implemented and documented.
4. Keep using the LOAO all-row view for robustness checks and the positive-row view only as a sentiment diagnostic.
5. Treat candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment as the current strongest non-LLM fixed held-out-aspect baseline.
6. Treat Gemini Flash as the completed hosted fixed held-out-aspect baseline, but do not run full Gemini LOAO unless the cost/benefit is explicitly justified.
7. The local-to-Gemini cascade is complete; useful next options are aspect descriptions and qualitative error taxonomy, then Qwen fine-tuning/evaluation on stronger GPU access if available.
8. If I ask to publish changes, commit/push only clean code and documentation, without committing outputs, data, credentials, checkpoints, or generated artifacts.

Please start by summarising what you find in the current docs and repo state, then perform the strict audit, then propose the next concrete plan before implementing.
```
