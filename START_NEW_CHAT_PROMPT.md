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
- The earlier lexical and candidate-aspect cross-encoder baselines used global sentiment: one document-level polarity was applied to all selected aspects. This is a limitation for FABSA because sentiment is per-aspect. A lightweight aspect-conditioned sentiment pipeline has now been implemented and evaluated; it is cleaner methodologically but slightly weaker than the global sentiment baseline, so stronger aspect-conditioned sentiment or joint aspect+sentiment pair scoring remains the next non-LLM modelling improvement.
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
- A lightweight aspect-conditioned sentiment version has also been implemented:
  - Label-masked test pair samples F1: 0.4520
  - Example-filtered test pair samples F1: 0.4389
  - Interpretation: this fixes the global-sentiment assumption but the shallow TF-IDF sentiment classifier is empirically weaker than the global prior.

Held-out aspect label-aware baseline:
- Candidate-aspect DistilBERT cross-encoder + global TF-IDF Logistic Regression sentiment classifier.
- Evaluation uses held-out labels only.
- Label-masked test pair samples F1: 0.5595
- Label-masked test pair micro F1: 0.5462
- Label-masked test pair macro F1: 0.4267
- Example-filtered test pair samples F1: 0.5816
- Example-filtered test pair micro F1: 0.5646
- Example-filtered test pair macro F1: 0.4538

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
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode global --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode global --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows_micro_selection
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_micro_selection
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --eval-row-scope containing_heldout --ensure-one --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_positive_rows
python .\scripts\run_qwen_heldout_aspect_smoke.py --split validation --limit 10000 --load-in-4bit --prompt-variant indexed
python .\scripts\run_qwen_heldout_aspect_smoke.py --split test --limit 10000 --load-in-4bit --prompt-variant indexed
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy both --prompt-variant indexed
```

Recommended next steps:

1. First inspect the repo and confirm the current state with `git status`.
2. LOAO lexical evaluation and lightweight aspect-conditioned sentiment have been implemented and documented.
3. Keep using the LOAO all-row view for robustness checks and the positive-row view only as a sentiment diagnostic.
4. The next non-LLM modelling option is a stronger aspect-conditioned sentiment classifier, such as DistilBERT for `(review, candidate aspect) -> sentiment`, or the later joint aspect+sentiment pair scorer.
5. Add a Gemini OpenAI-compatible hosted LLM baseline using the indexed candidate-label output format before spending local GPU time on full Qwen fine-tuning.
6. Keep full Qwen fine-tuning parked until stronger GPU access is available.
7. If I ask to publish changes, commit/push only clean code and documentation, without committing outputs, data, credentials, checkpoints, or generated artifacts.

Please start by summarising what you find in the current docs and repo state, then propose the next concrete plan before implementing.
```
