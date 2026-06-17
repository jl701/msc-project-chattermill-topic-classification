# Prompt For Continuing In A New Chat

Use this prompt when opening a fresh Codex/ChatGPT conversation for the project.

```text
I am continuing my UCL MSc project with Chattermill.

Please communicate with me in Chinese, but keep all code, comments, docstrings, README content, and project documentation in English. The code will be reviewed by Aji, so keep implementation concise, readable, and easy to inspect.

Local workspace:
C:\Msc_DSML\Msc_Project

GitHub working folder:
C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification

Private GitHub repo:
https://github.com/jl701/msc-project-chattermill-topic-classification

Local FABSA export:
C:\Msc_DSML\Msc_Project\Project_Preparation\Public_Datasets\FABSA

Project title:
Open-vocabulary Topic Classification with LLMs

Before doing new work, please inspect these files:

1. C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\PROJECT_OVERVIEW.md
2. C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\docs\aji_feedback_2026_06_16.md
3. C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\docs\evaluation_protocol.md
4. C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\docs\closed_topic_baselines.md
5. C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\docs\generalisation_baselines.md
6. C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\docs\qwen_feasibility.md
7. C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\docs\handoff_notes_2026_06_16.md
8. C:\Msc_DSML\Msc_Project\msc-project-chattermill-topic-classification\docs\heldout_aspect_error_analysis.md

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

Held-out aspect error analysis:
- DistilBERT cross-encoder mainly over-predicts candidate aspects.
- Qwen indexed zero-shot has more exact rows but also some missed-all rows.
- Qwen is strongest on `Account management: Account access`, weaker on `Company brand: Competitor`, especially positive competitor mentions.
- See `docs/heldout_aspect_error_analysis.md`.

Qwen feasibility:
- Model: Qwen/Qwen3-4B-Instruct-2507
- Local RTX 5050 Laptop GPU, 8 GB VRAM.
- 4-bit QLoRA with rank-8 LoRA runs locally.
- Results are only on the first 100 validation rows, not full validation/test.
- Zero-shot first-100 validation pair micro F1: 0.541
- Best local LoRA pilot first-100 validation pair micro F1: 0.762
- Use local Qwen only for smoke tests until proper GPU access is available.

Important repository hygiene:

- Do not commit internal Chattermill data, credentials, API keys, model checkpoints, or confidential outputs.
- Do not commit the `outputs/` directory.
- Public FABSA can be inspected locally, but do not commit full copied data files unless explicitly approved.
- Current project files may still be uncommitted locally, so check `git status` before making changes.

Useful commands:

```powershell
python -m unittest discover -s tests
python .\scripts\analyse_split_candidates.py
python .\scripts\build_fabsa_splits.py
python .\scripts\run_generalisation_baselines.py
python .\scripts\run_generalisation_baselines.py --protocol heldout-org --refined-cross-org --output-dir .\outputs\baselines\generalisation_refined
python .\scripts\run_transformer_baseline.py --protocol heldout-org --epochs 10 --batch-size 16 --learning-rate 6e-5 --pos-weight sqrt
python .\scripts\run_aspect_label_aware_baseline.py --strategy both --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3
python .\scripts\run_qwen_heldout_aspect_smoke.py --split validation --limit 10000 --load-in-4bit --prompt-variant indexed
python .\scripts\run_qwen_heldout_aspect_smoke.py --split test --limit 10000 --load-in-4bit --prompt-variant indexed
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy both --prompt-variant indexed
```

Recommended next steps:

1. First inspect the repo and confirm the current state with `git status`.
2. If I ask to publish the setup, help me commit/push the clean code and documentation, without committing outputs or data.
3. If I ask to continue experiments, the preferred next step is full Qwen candidate-label fine-tuning on a stronger GPU using the indexed held-out-aspect SFT/evaluation files.
4. If strong GPU access is still unavailable, add another non-LLM label-aware held-out-aspect baseline, such as a bi-encoder or NLI-style candidate-label model.
5. Keep local Qwen runs to smoke tests and data-format checks unless I explicitly ask for a slow local pilot.

Please start by summarising what you find in the current docs and repo state, then propose the next concrete plan before implementing.
```
