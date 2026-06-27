# MSc Project: Open-Vocabulary Topic Classification

Private project workspace for the UCL MSc project with Chattermill.

Working title:

**Open-vocabulary Topic Classification with LLMs**

The main project context, current understanding, datasets, modelling plan, evaluation plan, and open questions are collected in:

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- [docs/github_upload_scope.md](docs/github_upload_scope.md)
- [docs/non_llm_open_topic_baseline.md](docs/non_llm_open_topic_baseline.md)
- [docs/next_stage_and_literature_review_plan.md](docs/next_stage_and_literature_review_plan.md)

The current first phase is a closed-topic FABSA baseline:

- Load the public FABSA train/validation/test splits from the local export.
- Predict multi-label aspect+sentiment pairs.
- Report sample-level F1, micro F1, and macro F1.
- Keep the provided split as the closed-topic benchmark.
- Add held-out organisation and held-out aspect protocols for generalisation.

## Local Data

The expected local FABSA export is outside this repository:

```text
D:\Msc_Project\Project_Preparation\Public_Datasets\FABSA
C:\Msc_DSML\Msc_Project\Project_Preparation\Public_Datasets\FABSA
```

You can override this with:

```powershell
$env:FABSA_DATA_DIR="C:\path\to\FABSA"
```

## Commands

Explore the exported FABSA splits:

```powershell
python .\scripts\explore_fabsa.py
```

Run the first TF-IDF + Logistic Regression baseline:

```powershell
python .\scripts\run_tfidf_logreg.py --eval-split all
```

Run the current traditional baseline sweep:

```powershell
python .\scripts\run_classical_baselines.py
```

Run a BERT-style closed-topic baseline:

```powershell
python .\scripts\run_transformer_baseline.py
```

Run a BERT-style held-out-organisation baseline:

```powershell
python .\scripts\run_transformer_baseline.py --protocol heldout-org --epochs 10 --batch-size 16 --learning-rate 6e-5 --pos-weight sqrt
```

Analyse candidate held-out organisation/aspect splits:

```powershell
python .\scripts\analyse_split_candidates.py
```

Build reproducible split manifests:

```powershell
python .\scripts\build_fabsa_splits.py
```

Run the first generalisation baselines:

```powershell
python .\scripts\run_generalisation_baselines.py
python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --sentiment-mode global
```

Run the refined held-out organisation SVM grid:

```powershell
python .\scripts\run_generalisation_baselines.py --protocol heldout-org --refined-cross-org --output-dir .\outputs\baselines\generalisation_refined
```

Run the candidate-aspect cross-encoder held-out-aspect baseline:

```powershell
python .\scripts\run_aspect_label_aware_baseline.py --strategy both --sentiment-mode global --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3
```

Run the strongest current non-LLM fixed held-out-aspect baseline:

```powershell
python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3
```

Run leave-one-aspect-out held-out-aspect lexical evaluation:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode global
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --selection-metric pair_micro_f1
```

Run a Qwen held-out-aspect indexed zero-shot smoke test:

```powershell
python .\scripts\run_qwen_heldout_aspect_smoke.py --split validation --limit 10000 --load-in-4bit --prompt-variant indexed
```

Prepare Qwen held-out-aspect SFT/evaluation JSONL files:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy both --prompt-variant indexed
```

Prepare Qwen SFT-style chat data:

```powershell
python .\scripts\prepare_qwen_sft_data.py
```

Run a small Qwen zero-shot smoke test:

```powershell
python .\scripts\run_qwen_zero_shot.py --load-in-4bit --limit 20
```

Run a tiny Qwen LoRA/QLoRA pilot:

```powershell
python .\scripts\run_qwen_lora_pilot.py --train-limit 500 --eval-limit 100 --epochs 1
```

On this Windows machine, CUDA training required the CUDA-enabled PyTorch wheel:

```powershell
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0+cu128
```

Run unit tests:

```powershell
python -m unittest discover -s tests
```

This repository is in setup stage. Do not commit internal Chattermill data, credentials, model checkpoints, or confidential outputs unless explicitly approved.
