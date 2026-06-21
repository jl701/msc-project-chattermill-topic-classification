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

## 2026-06-21: LOAO Diagnostics And Threshold Selection Check

### Purpose

- Clarify how all-row LOAO sample-F1 should be interpreted.
- Add diagnostics that expose precision, recall, false positives, false negatives, empty-row behaviour, and sentiment correctness when the gold aspect is predicted.
- Check whether selecting thresholds by validation pair samples F1 is appropriate for all-row LOAO.

### Code Or Protocol Changes

- Updated `src/msc_project/evaluation/metrics.py`:
  - added pair/aspect micro precision and micro recall
  - added TP/FP/FN label counts
  - added empty-gold, empty-prediction, false-positive-row, and false-negative-row diagnostics
  - added false-positive rows/labels per 100 reviews
  - added exact-match rate
  - added `sentiment_accuracy_when_gold_aspect_predicted`
- Updated `scripts/run_loao_heldout_aspect.py`:
  - includes the new diagnostics in per-fold and spread tables
  - supports `--selection-metric`
- Updated `scripts/run_generalisation_baselines.py` and `scripts/run_aspect_label_aware_baseline.py`:
  - supports configurable validation threshold/model selection metric
- Added `tests/test_selection_metrics.py`.

### Setup

- Dataset: FABSA public train/validation/test export.
- Protocol: all-row LOAO with 12 held-out aspect folds and both training strategies.
- Baseline: candidate-label lexical TF-IDF + global sentiment Logistic Regression.
- Compared validation selection metrics:
  - `pair_samples_f1`
  - `pair_micro_f1`

### Commands

```powershell
python -m unittest discover -s tests
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_all_rows_micro_selection
python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --eval-row-scope containing_heldout --ensure-one --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_positive_rows
```

### Outputs

- Sample-F1-selected all-row LOAO:
  - `outputs/baselines/loao_heldout_aspect_lexical_all_rows/`
- Micro-F1-selected all-row LOAO:
  - `outputs/baselines/loao_heldout_aspect_lexical_all_rows_micro_selection/`
- Positive-row diagnostic:
  - `outputs/baselines/loao_heldout_aspect_lexical_positive_rows/`
- Updated committed docs:
  - `docs/loao_heldout_aspect.md`
  - `docs/generalisation_baselines.md`

Generated output files remain ignored by Git.

### Results

All-row LOAO, test split:

| Selection | Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | FP Rows / 100 Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sample F1 | `label_masked` | 0.1299 | 0.2304 | 0.1458 | 0.8857 | 77.5362 |
| Sample F1 | `example_filtered` | 0.1288 | 0.2345 | 0.1495 | 0.8777 | 77.4207 |
| Micro F1 | `label_masked` | 0.0926 | 0.3635 | 0.3912 | 0.4883 | 18.5728 |
| Micro F1 | `example_filtered` | 0.0903 | 0.3780 | 0.3778 | 0.4895 | 17.4753 |

Positive-row diagnostic, test split:

| Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean |
| --- | ---: | ---: | ---: | ---: |
| `label_masked` | 0.8870 | 0.8868 | 0.8879 | 0.8857 |
| `example_filtered` | 0.8790 | 0.8788 | 0.8799 | 0.8777 |

Validation:

- `python -m unittest discover -s tests`
- `37 tests OK`

### Interpretation

- Pair samples F1 remains the headline metric, following Aji's guidance.
- For all-row LOAO, pair samples F1 alone is not suitable for threshold selection because empty-gold true negatives and empty-gold false positives both contribute row F1 `0`.
- Sample-F1 selection therefore over-predicts: recall is high, but precision is very low and false-positive rows are excessive.
- Micro-F1 selection is the better all-row detection diagnostic: it lowers sample-F1 but gives a much more meaningful precision/recall trade-off.
- The positive-row diagnostic remains useful for sentiment interpretation, but it does not test aspect detection.

### Next Step

Use all-row LOAO with micro-F1 threshold selection as the main detection diagnostic, while still reporting pair samples F1 as the headline metric. The next modelling improvement remains aspect-conditioned sentiment or joint aspect+sentiment pair scoring.

## 2026-06-21: Aspect-Conditioned Sentiment Pipeline

### Purpose

- Address Aji's concern that the earlier held-out-aspect baselines used one global document-level sentiment and applied it to every selected aspect.
- Add a cleaner per-aspect sentiment option: `(review text, candidate aspect) -> sentiment`.
- Keep the old global sentiment path available as an ablation and reproducibility baseline.

### Code Or Protocol Changes

- Updated `src/msc_project/baselines/candidate_label.py`:
  - added `aspect_conditioned` and `global` sentiment modes
  - added `AspectConditionedSentimentModel`
  - added `train_aspect_conditioned_sentiment_model`
  - added reusable sentiment lookup and pair-construction helpers
  - changed `CandidateLexicalBaseline` default sentiment mode to `aspect_conditioned`
- Updated `scripts/run_generalisation_baselines.py`:
  - added `--sentiment-mode`
  - threaded sentiment mode into the held-out-aspect lexical baseline
- Updated `scripts/run_aspect_label_aware_baseline.py`:
  - added `--sentiment-mode`
  - switched candidate-aspect cross-encoder pair construction to use per-candidate sentiment lookup
- Updated `scripts/run_loao_heldout_aspect.py`:
  - added `--sentiment-mode`
  - included `sentiment_mode` in LOAO result tables and aggregation
- Added `tests/test_candidate_label_sentiment.py`.

### Setup

- Dataset: FABSA public train/validation/test export.
- Main changed model: candidate-label lexical TF-IDF aspect selector + aspect-conditioned TF-IDF Logistic Regression sentiment classifier.
- The global sentiment classifier was rerun to confirm that the code change preserved the original lower-bound results.
- Full cross-encoder LOAO was not run; only a tiny cross-encoder smoke test was run because full local DistilBERT LOAO is too slow on the GTX 1660 Ti Max-Q 6 GB GPU.

### Commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode aspect_conditioned --output-dir .\outputs\baselines\generalisation_aspect_conditioned_sentiment
.\.venv\Scripts\python.exe .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode global --output-dir .\outputs\baselines\generalisation_global_sentiment_rerun
.\.venv\Scripts\python.exe .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_micro_selection
.\.venv\Scripts\python.exe .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --eval-row-scope containing_heldout --ensure-one --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_positive_rows
.\.venv\Scripts\python.exe .\scripts\run_aspect_label_aware_baseline.py --strategy label_masked --sentiment-mode aspect_conditioned --train-limit 100 --eval-limit 100 --epochs 1 --batch-size 8 --eval-batch-size 16 --output-dir .\outputs\baselines\aspect_label_aware_aspect_conditioned_smoke
```

### Outputs

- Fixed three-aspect aspect-conditioned lexical results:
  - `outputs/baselines/generalisation_aspect_conditioned_sentiment/`
- Fixed three-aspect global sentiment rerun:
  - `outputs/baselines/generalisation_global_sentiment_rerun/`
- All-row LOAO with aspect-conditioned sentiment and micro-F1 threshold selection:
  - `outputs/baselines/loao_heldout_aspect_lexical_aspect_conditioned_micro_selection/`
- Positive-row LOAO sentiment diagnostic with aspect-conditioned sentiment:
  - `outputs/baselines/loao_heldout_aspect_lexical_aspect_conditioned_positive_rows/`
- Cross-encoder aspect-conditioned smoke test:
  - `outputs/baselines/aspect_label_aware_aspect_conditioned_smoke/`

Generated output files remain ignored by Git.

### Results

Fixed three-aspect held-out aspect lexical baseline, test split:

| Sentiment Mode | Strategy | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: |
| Global | `label_masked` | 0.4698 | 0.4667 | 0.3703 | 0.8867 |
| Global | `example_filtered` | 0.4626 | 0.4596 | 0.3703 | 0.8851 |
| Aspect-conditioned | `label_masked` | 0.4520 | 0.4491 | 0.3425 | 0.8533 |
| Aspect-conditioned | `example_filtered` | 0.4389 | 0.4351 | 0.3422 | 0.8378 |

All-row LOAO lexical baseline with micro-F1 threshold selection, test split:

| Sentiment Mode | Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | FP Rows / 100 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global | `label_masked` | 0.0926 | 0.3635 | 0.3912 | 0.4883 | 18.5728 | 0.8867 |
| Global | `example_filtered` | 0.0903 | 0.3780 | 0.3778 | 0.4895 | 17.4753 | 0.8783 |
| Aspect-conditioned | `label_masked` | 0.0883 | 0.3511 | 0.3811 | 0.4700 | 18.5728 | 0.8592 |
| Aspect-conditioned | `example_filtered` | 0.0842 | 0.3576 | 0.3627 | 0.4541 | 16.7034 | 0.8390 |

Positive-row LOAO sentiment diagnostic, test split:

| Sentiment Mode | Strategy | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Macro F1 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: |
| Global | `label_masked` | 0.8870 | 0.8868 | 0.6316 | 0.8857 |
| Global | `example_filtered` | 0.8790 | 0.8788 | 0.6272 | 0.8777 |
| Aspect-conditioned | `label_masked` | 0.8647 | 0.8646 | 0.5924 | 0.8636 |
| Aspect-conditioned | `example_filtered` | 0.8442 | 0.8440 | 0.5777 | 0.8430 |

Validation:

- First test attempt accidentally used system Python and failed because system Python lacked numpy/pandas.
- Correct project environment command:
  - `.\.venv\Scripts\python.exe -m unittest discover -s tests`
  - `43 tests OK`

### Interpretation

- The aspect-conditioned pipeline fixes the modelling assumption: sentiment is now predicted per selected candidate aspect rather than once per document.
- The current lightweight TF-IDF aspect-conditioned sentiment classifier did not improve the lexical results. It is slightly weaker than the global sentiment classifier in the fixed three-aspect split and in LOAO.
- This does not mean per-aspect sentiment is the wrong direction. It means the current shallow sentiment classifier is not strong enough to benefit from the cleaner formulation.
- In all-row LOAO, the main bottleneck is still unseen-aspect detection. Aspect-conditioned sentiment changes pair labels after aspects have been selected; it does not fix weak aspect selection.
- In positive-row LOAO, where aspect selection is trivial, the global sentiment prior remains slightly stronger than the lightweight aspect-conditioned model.
- The old global sentiment baseline should remain in the report as a useful empirical reference, while the aspect-conditioned version should be discussed as a methodologically cleaner but currently weaker ablation.

### Next Step

- Do not spend time on more lexical sentiment tuning unless a very small ablation is needed for the write-up.
- The next useful non-LLM step is a stronger aspect-conditioned sentiment model, such as a DistilBERT cross-encoder for `(review, candidate aspect) -> sentiment`, or the deferred joint aspect+sentiment pair scorer.
- A hosted Gemini indexed candidate-label baseline is now attractive because it can naturally output per-aspect sentiment without local GPU fine-tuning.

## 2026-06-22: Planning Note For Next Work

### Decision

The immediate next task should be **DistilBERT aspect-conditioned sentiment**, not Gemini.

### Rationale

- The user wants to exhaust free/local resources before spending hosted Gemini credits.
- DistilBERT aspect-conditioned sentiment directly addresses Aji's question about the global sentiment component.
- It is a controlled non-LLM ablation: keep the aspect selector fixed, replace only the sentiment module.
- It is less complex than the later joint aspect+sentiment pair scorer.
- Gemini remains useful, but should be a later hosted LLM baseline rather than the next step.

### Proposed Audit Before Implementation

Before implementing, the next assistant should strictly review:

- whether the current split protocols match Aji's feedback and the research question
- whether pair samples F1, pair micro F1, pair macro F1, all-row LOAO, and positive-row LOAO are being interpreted correctly
- whether fixed three-aspect held-out results, all-row LOAO, positive-row LOAO, closed-topic, and held-out organisation results are being compared only where appropriate
- whether the current conclusion is sound: lightweight TF-IDF aspect-conditioned sentiment is methodologically cleaner but empirically weaker than global sentiment
- whether DistilBERT aspect-conditioned sentiment is truly the best next step before Gemini

### Proposed Experiment Order

1. Fixed three-aspect held-out split:
   - lexical aspect selector + DistilBERT aspect-conditioned sentiment
   - compare against lexical + global sentiment and lexical + TF-IDF aspect-conditioned sentiment
2. Positive-row LOAO:
   - use as a pure sentiment diagnostic because the aspect is already known to be present
3. All-row LOAO:
   - use as the full detection diagnostic after the sentiment module works
4. If useful and compute allows:
   - candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment
5. Keep joint aspect+sentiment pair scoring and Gemini for later.
