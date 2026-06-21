# Experiment Log

This document is the running experiment log for the MSc project. It should be updated after every meaningful experiment, implementation change, or evaluation run before handing work back to the user.

## Logging Rule

Every completed experiment must add an entry to this file. The entry should be detailed enough that another Codex/ChatGPT session, the user, or Aji can reconstruct:

- what changed in the code or protocol
- why the experiment was run
- exactly which data split, labels, and model variant were used
- which command or script was run
- where local outputs were written
- which headline and supporting metrics were obtained
- what the result means
- what should be done next

Do not commit local `outputs/`, checkpoints, credentials, API keys, or private data. Summarise results in this document and keep generated artifacts local unless explicitly approved.

## Required Entry Template

````markdown
## YYYY-MM-DD: Short Experiment Name

### Purpose
- Why this experiment was needed.

### Code Or Protocol Changes
- Files changed.
- Behaviour changed.
- Any compatibility or reproducibility notes.

### Setup
- Dataset and split.
- Labels evaluated.
- Model or baseline.
- Important hyperparameters.
- Hardware or API endpoint, if relevant.

### Commands
```powershell
# exact command(s)
```

### Outputs
- Local output directory or files.
- Any files intentionally committed.
- Any generated files intentionally left uncommitted.

### Results
| Metric | Value |
| --- | ---: |
| Headline metric | 0.0000 |

### Interpretation
- What the result shows.
- Known limitations or failure modes.

### Next Step
- The next concrete action.
````

## 2026-06-21: Project Handoff And Aji Update Consolidation

### Purpose

- Preserve the latest Aji feedback and Gemini/Vertex AI access context for future project sessions.
- Fix the new-machine handoff so both possible workspace roots are recorded.
- Add this experiment logging rule before starting the leave-one-aspect-out work.

### Code Or Protocol Changes

- Added `docs/aji_updates_2026_06_21.md`.
- Updated `START_NEW_CHAT_PROMPT.md` with:
  - dual workspace path roots
  - latest Aji feedback
  - Gemini endpoint notes without the real API key
  - next-step ordering
- Added `docs/experiment_log.md`.

### Setup

- Current/new workspace: `D:\Msc_Project`
- Previous/alternate workspace: `C:\Msc_DSML\Msc_Project`
- Repository: `jl701/msc-project-chattermill-topic-classification`
- Branch: `main`

### Commands

```powershell
git status -sb
rg -n "sk-[A-Za-z0-9_-]+|OPENAI_API_KEY|llm-api\.datascience" . --glob '!outputs/**' --glob '!.venv/**' --glob '!data/**' --glob '!checkpoints/**'
```

### Outputs

- Documentation-only changes.
- No experiment outputs, checkpoints, data files, or credentials were added.
- The real Gemini API key is not stored in the repository.

### Results

| Check | Result |
| --- | --- |
| GitHub branch state | local `main` tracks `origin/main` |
| OneDrive sync for handoff docs | SHA256 hashes matched |
| Real API key committed | No |

### Interpretation

The project now has a durable handoff path for future sessions and a fixed rule for recording experiment details after each run.

### Next Step

Implement and run leave-one-aspect-out held-out-aspect evaluation across all 12 FABSA aspects, preserving both `label_masked` and `example_filtered`.

## 2026-06-21: Leave-One-Aspect-Out Lexical Held-Out Aspect Evaluation

### Purpose

- Address Aji's feedback that a single fixed set of three held-out aspects makes the open-topic result fragile.
- Rotate each FABSA aspect as the held-out aspect and report the spread across all 12 aspects.
- Avoid a trivial one-candidate positive-only LOAO setup by adding an all-row evaluation view with negative rows.

### Code Or Protocol Changes

- Updated `src/msc_project/data/splits.py`:
  - added `eval_row_scope="containing_heldout" | "all"` to `build_heldout_aspect_split`
  - kept the old default behaviour as `containing_heldout`
  - added the all-row mode for LOAO negative rows
- Updated `scripts/run_generalisation_baselines.py`:
  - added `ensure_one` control for lexical candidate predictions
- Updated `scripts/run_aspect_label_aware_baseline.py`:
  - added empty-prediction support and `eval_row_scope` passthrough
- Added `scripts/run_loao_heldout_aspect.py`:
  - runs LOAO across selected or all aspects
  - supports lexical and cross-encoder baselines
  - writes per-fold results and spread tables
- Updated `src/msc_project/evaluation/metrics.py`:
  - implemented manual sample-F1 calculation so single-class LOAO folds work
- Added tests for all-row held-out aspect splits and single-class sample F1.

### Setup

- Dataset: FABSA public train/validation/test export.
- Aspects: all 12 FABSA aspects.
- Strategies:
  - `label_masked`
  - `example_filtered`
- Completed full baseline:
  - candidate-label lexical TF-IDF + global sentiment Logistic Regression
- Main evaluation row scope:
  - all validation/test rows
  - held-out aspect labels only
  - empty predictions allowed
- Diagnostic row scope:
  - only rows containing the held-out aspect
  - at least one prediction forced
- Current machine:
  - NVIDIA GeForce GTX 1660 Ti with Max-Q Design, 6 GB VRAM

### Commands

```powershell
python -m unittest discover -s tests
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --eval-row-scope containing_heldout --ensure-one --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_positive_rows
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy label_masked --heldout-aspect "Staff support: Email" --epochs 1 --batch-size 16 --eval-batch-size 32 --learning-rate 2e-5 --negatives-per-positive 1 --train-limit 100 --eval-limit 100 --output-dir .\outputs\baselines\loao_cross_encoder_tiny_smoke
```

An attempted full DistilBERT cross-encoder single-fold smoke did not finish within 30 minutes on the local 6 GB GPU, so full cross-encoder LOAO was not run as a formal result.

### Outputs

- Main all-row lexical LOAO:
  - `outputs/baselines/loao_heldout_aspect_lexical_all_rows/`
- Positive-row diagnostic lexical LOAO:
  - `outputs/baselines/loao_heldout_aspect_lexical_positive_rows/`
- Cross-encoder tiny smoke:
  - `outputs/baselines/loao_cross_encoder_tiny_smoke/`
- Detailed committed summary:
  - `docs/loao_heldout_aspect.md`

Generated output files remain ignored by Git.

### Results

Main all-row lexical LOAO, test split:

| Strategy | Pair Samples F1 Mean | Pair Samples F1 Std | Min | Max | Pair Micro F1 Mean | Pair Macro F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `label_masked` | 0.1299 | 0.1256 | 0.0126 | 0.3930 | 0.2304 | 0.1730 |
| `example_filtered` | 0.1288 | 0.1260 | 0.0126 | 0.3974 | 0.2345 | 0.1790 |

Positive-row diagnostic lexical LOAO, test split:

| Strategy | Pair Samples F1 Mean | Pair Samples F1 Std | Min | Max | Aspect Micro F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `label_masked` | 0.8870 | 0.0294 | 0.8354 | 0.9312 | 1.0000 |
| `example_filtered` | 0.8790 | 0.0254 | 0.8315 | 0.9231 | 1.0000 |

Validation:

- `python -m unittest discover -s tests`
- `32 tests OK`

### Interpretation

- The all-row LOAO result confirms Aji's concern: performance varies strongly by held-out aspect.
- Positive-row LOAO is much easier and mainly measures global sentiment after the aspect is assumed present.
- The lexical model is not enough for robust unseen-aspect selection, especially for rare or less lexically transparent aspects.
- `label_masked` and `example_filtered` are very close for the lexical lower bound.
- The global sentiment component remains a limitation and should be improved next.

### Next Step

Improve the held-out-aspect baseline with aspect-conditioned sentiment or joint aspect+sentiment pair scoring. Do not start full Qwen fine-tuning yet.
