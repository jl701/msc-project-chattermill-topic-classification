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

## 2026-06-27: DistilBERT Aspect-Conditioned Sentiment Baseline

### Purpose

- Revisit the global-sentiment limitation flagged by Aji: earlier held-out-aspect baselines predicted one document-level polarity and reused it for every selected aspect.
- Test whether a stronger local/free sentiment model can improve the methodologically cleaner `(feedback text, candidate aspect) -> sentiment` formulation.
- Stop at the strongest non-LLM baseline requested for this phase: candidate-aspect DistilBERT selector plus DistilBERT aspect-conditioned sentiment.
- Keep the result sceptical: measure both controlled lexical-selector changes and full pipeline changes, and record tuning attempts that did not help.

### Code Or Protocol Changes

- Added `src/msc_project/baselines/transformer_sentiment.py`:
  - trains a DistilBERT three-way sentiment classifier over `(review text, candidate aspect)` examples
  - supports validation selection by accuracy, macro F1, or micro F1
  - supports class weighting and AMP
  - exposes `predict_pairs(texts, aspects)` for pipeline integration
- Updated `src/msc_project/baselines/candidate_label.py`:
  - added `transformer_aspect_conditioned` sentiment mode
  - allowed an externally fitted sentiment model to be passed into the candidate-label pipeline
- Updated `scripts/run_generalisation_baselines.py`:
  - added CLI controls for transformer aspect-conditioned sentiment
  - trains DistilBERT sentiment before evaluating the lexical held-out-aspect selector
- Updated `scripts/run_aspect_label_aware_baseline.py`:
  - added `--sentiment-mode transformer_aspect_conditioned`
  - trains DistilBERT aspect-conditioned sentiment before training/evaluating the candidate-aspect DistilBERT selector
  - frees the sentiment model before selector training to reduce GPU memory pressure
- Updated `scripts/run_loao_heldout_aspect.py`:
  - intentionally kept LOAO to the fast sentiment modes for now because full transformer-sentiment LOAO is outside this phase and would be slow locally
- Added `tests/test_transformer_sentiment.py`.

### Setup

- Dataset: FABSA public train/validation/test export.
- Protocol: fixed three-aspect held-out aspect evaluation, containing-heldout row scope, held-out labels only.
- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Strategies:
  - `label_masked`
  - `example_filtered`
- Sentiment model:
  - `distilbert-base-uncased`
  - input: `Review: ... Aspect: ...`
  - classes: `negative`, `neutral`, `positive`
  - 3 epochs unless otherwise noted
  - learning rate `2e-5`
  - batch size 16, eval batch size 64
  - balanced class weights
  - validation accuracy selection for the final runs
- Aspect selector for the strongest baseline:
  - `distilbert-base-uncased` candidate-aspect cross-encoder
  - batch size 32, eval batch size 96
  - 3 negatives per positive
  - validation pair samples F1 selection
- Hardware:
  - NVIDIA GeForce RTX 5050 Laptop GPU
  - CUDA-enabled PyTorch was available.

### Commands

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests

python .\scripts\run_aspect_label_aware_baseline.py --strategy label_masked --sentiment-mode transformer_aspect_conditioned --train-limit 80 --eval-limit 40 --sentiment-epochs 1 --sentiment-batch-size 8 --sentiment-eval-batch-size 16 --epochs 1 --batch-size 8 --eval-batch-size 16 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_smoke

python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric macro_f1 --output-dir .\outputs\baselines\generalisation_transformer_sentiment_lr2e-5_ep3_balanced_macro

python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --output-dir .\outputs\baselines\generalisation_transformer_sentiment_lr2e-5_ep3_balanced_accuracy

python .\scripts\run_aspect_label_aware_baseline.py --strategy both --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr2e-5_ep3_neg3

python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 5 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr2e-5_ep5_neg3_example_filtered

python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3 --max-predictions-per-row 2 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr2e-5_ep3_neg3_top2_example_filtered

python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered

python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 4e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr4e-5_ep3_neg3_example_filtered

python .\scripts\run_aspect_label_aware_baseline.py --strategy label_masked --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_label_masked
```

### Outputs

- Smoke test:
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_smoke/`
- Controlled lexical selector runs:
  - `outputs/baselines/generalisation_transformer_sentiment_lr2e-5_ep3_balanced_macro/`
  - `outputs/baselines/generalisation_transformer_sentiment_lr2e-5_ep3_balanced_accuracy/`
- Candidate-aspect DistilBERT selector plus DistilBERT sentiment runs:
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_lr2e-5_ep3_neg3/`
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_lr2e-5_ep5_neg3_example_filtered/`
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_lr2e-5_ep3_neg3_top2_example_filtered/`
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered/`
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_lr4e-5_ep3_neg3_example_filtered/`
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_label_masked/`
- Generated output files remain ignored by Git.

### Results

Controlled lexical selector, test split:

| Sentiment Mode | Strategy | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Sentiment Accuracy When Gold Aspect Predicted |
| --- | --- | ---: | ---: | ---: | ---: |
| Global TF-IDF sentiment | `label_masked` | 0.4698 | 0.4667 | 0.3703 | 0.8867 |
| Lightweight TF-IDF aspect-conditioned sentiment | `label_masked` | 0.4520 | 0.4491 | 0.3425 | 0.8533 |
| DistilBERT aspect-conditioned sentiment | `label_masked` | 0.4840 | 0.4807 | 0.4063 | 0.9133 |
| Global TF-IDF sentiment | `example_filtered` | 0.4626 | 0.4596 | 0.3703 | 0.8851 |
| Lightweight TF-IDF aspect-conditioned sentiment | `example_filtered` | 0.4389 | 0.4351 | 0.3422 | 0.8378 |
| DistilBERT aspect-conditioned sentiment | `example_filtered` | 0.4804 | 0.4772 | 0.3985 | 0.9189 |

Candidate-aspect DistilBERT selector plus DistilBERT aspect-conditioned sentiment, test split:

| Strategy / Variant | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 | Sentiment Accuracy When Gold Aspect Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| `label_masked`, selector LR `2e-5`, 3 epochs | 0.5412 | 0.5343 | 0.4713 | 0.6088 | 0.8947 |
| `label_masked`, selector LR `3e-5`, 3 epochs | 0.4981 | 0.4925 | 0.4531 | 0.5598 | 0.8919 |
| `example_filtered`, selector LR `2e-5`, 3 epochs | 0.6001 | 0.5859 | 0.4942 | 0.6600 | 0.9095 |
| `example_filtered`, selector LR `2e-5`, 5 epochs | 0.5325 | 0.5298 | 0.4573 | 0.5942 | 0.8990 |
| `example_filtered`, selector LR `2e-5`, top-2 cap | 0.5949 | 0.5823 | 0.4942 | 0.6548 | 0.9095 |
| `example_filtered`, selector LR `3e-5`, 3 epochs | **0.6071** | **0.5917** | 0.4890 | **0.6651** | 0.9100 |
| `example_filtered`, selector LR `4e-5`, 3 epochs | 0.5614 | 0.5478 | 0.4610 | 0.6260 | 0.8972 |

Best current non-LLM fixed held-out-aspect result:

| Model | Strategy | Test Pair Samples F1 | Test Pair Micro F1 | Test Pair Macro F1 | Test Aspect Samples F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Candidate-aspect DistilBERT selector + global TF-IDF sentiment | `example_filtered` | 0.5816 | 0.5646 | 0.4538 | 0.6835 |
| Candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment | `example_filtered` | **0.6071** | **0.5917** | **0.4890** | 0.6651 |
| Qwen3-4B-Instruct indexed zero-shot | fixed held-out aspects | 0.5374 | 0.5300 | 0.4374 | 0.6340 |

Validation:

- `python -m unittest discover -s tests`
- `48 tests OK`
- `python -m compileall -q src scripts tests`
- compile check passed.

### Interpretation

- The user's memory is partly confirmed. DistilBERT aspect-conditioned sentiment improves over global sentiment in the controlled lexical-selector setup, and it clearly improves over the earlier lightweight TF-IDF aspect-conditioned sentiment classifier.
- The improvement is not universal once the stronger aspect selector is included. The `example_filtered` candidate-aspect DistilBERT selector benefits from DistilBERT aspect-conditioned sentiment, improving test pair samples F1 from `0.5816` to `0.6071`.
- The `label_masked` candidate-aspect setup gets worse than the previous global-sentiment result (`0.5412` versus `0.5595` at selector LR `2e-5`, and worse again at LR `3e-5`). This supports treating label-masked training as noisier because it can retain rows with censored held-out supervision.
- The best run has slightly lower aspect-only samples F1 than the previous global-sentiment candidate-aspect example-filtered baseline (`0.6651` versus `0.6835`), but higher pair-level F1 because sentiment is more accurate when the relevant aspect is selected.
- Additional tuning did not find a better local fixed-split result:
  - 5 selector epochs overfit or selected poorly on test.
  - a top-2 prediction cap did not help.
  - selector LR `4e-5` degraded.
  - label-masked LR `3e-5` degraded.
- This result should be reported as the strongest current fixed three-aspect non-LLM baseline, not as a full LOAO robustness result.

### Next Step

- Document this as the end of the current non-LLM fixed held-out-aspect phase.
- Do not spend more time on small fixed-split local tuning unless a report-specific ablation is needed.
- The next major modelling stage is Gemini or Qwen candidate-label fine-tuning/evaluation, but it should start as a separate phase after this baseline is written up and reviewed.

## 2026-06-27: Non-LLM Open-Topic Baseline Write-Up And Pause Checkpoint

### Purpose

- Convert the strongest local non-LLM open-topic result into a reusable methods/results note.
- Record a clean pause point before starting the next major Gemini or Qwen modelling phase.
- Preserve the literature-review and dissertation-framework direction so the project can temporarily shift from experiments to writing and reading.

### Code Or Protocol Changes

- No modelling protocol changed in this entry.
- Added `docs/non_llm_open_topic_baseline.md`.
- Added `docs/next_stage_and_literature_review_plan.md`.
- Updated project entry points so future sessions can find these notes:
  - `README.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`
  - `docs/experiment_log.md`

### Setup

- Baseline being documented:
  - candidate-aspect DistilBERT selector
  - DistilBERT aspect-conditioned sentiment classifier
  - fixed three-aspect held-out-aspect protocol
  - `example_filtered` training
- Best test result:
  - pair samples F1: `0.6071`
  - pair micro F1: `0.5917`
  - pair macro F1: `0.4890`

### Commands

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git status --short --branch
```

### Outputs

- New documentation:
  - `docs/non_llm_open_topic_baseline.md`
  - `docs/next_stage_and_literature_review_plan.md`
- Generated experiment outputs remain local under `outputs/` and ignored by Git.
- No data, checkpoints, credentials, or raw confidential materials were added.

### Results

| Check | Result |
| --- | --- |
| Strongest local non-LLM fixed held-out-aspect baseline documented | Yes |
| Next Gemini/Qwen phase recorded as paused | Yes |
| Literature review themes and dissertation structure recorded | Yes |
| Unit tests | 48 tests OK |
| Compile check | Passed |

### Interpretation

- The local non-LLM open-topic baseline phase is ready to be treated as a completed dissertation result block.
- The next useful work is not more small fixed-split DistilBERT tuning; it is either literature-review/report framing or a separate LLM phase.
- The literature review should now organise the project around ABSA, multi-label classification, domain generalisation, open-topic/candidate-label classification, and instruction-following LLMs.

### Next Step

- Pause major modelling.
- Work on the dissertation literature review, research framing, and result-table structure.
- When modelling resumes, decide explicitly between a hosted Gemini candidate-label baseline and Qwen candidate-label fine-tuning/evaluation.

## 2026-06-29: Aji Guidance And Literature Review Scoping

### Purpose

- Record the latest Aji guidance from the Slack screenshot supplied by the user.
- Start the literature review phase by scanning the locally downloaded papers.
- Convert the user's hourglass dissertation idea into a concrete broad-to-narrow-to-broad project framing.

### Code Or Protocol Changes

- No code or modelling protocol changed in this entry.
- Added `docs/aji_updates_2026_06_29.md`.
- Added `docs/literature_review_scoping_2026_06_29.md`.
- Updated:
  - `docs/next_stage_and_literature_review_plan.md`
  - `README.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`

### Setup

- Local literature folders scanned:
  - `C:\Msc_DSML\Msc_Project\Project_Preparation\Background_Paper`
  - `C:\Msc_DSML\Msc_Project\Project_Preparation\Recent_Paper`
- Main local papers identified:
  - Hu and Liu, 2004, *Mining and Summarizing Customer Reviews*
  - Tang et al., 2016, *Aspect Level Sentiment Classification with Deep Memory Network*
  - Kontonatsios et al., 2023, *FABSA: An aspect-based sentiment analysis dataset of user reviews*
  - Ding et al., 2022, *Towards Open-Domain Topic Classification* (local 2023 arXiv copy)
  - Yu et al., 2023, *Open, Closed, or Small Language Models for Text Classification?*
  - Ventirozos et al., *Exploring Zero-Shot ACSA with Unified Meaning Representation in Chain-of-Thought Prompting*
  - Lim et al., 2026, *Parameter-Efficient Adaptation of Qwen2.5 for Aspect-Based Sentiment Analysis Using Low-Rank Adaptation and Parameter-Efficient Fine-Tuning*

### Commands

```powershell
git status --short --branch
Get-ChildItem -LiteralPath 'C:\Msc_DSML\Msc_Project' -Recurse -File -Include *.pdf,*.bib,*.ris,*.md,*.txt
```

### Outputs

- `docs/aji_updates_2026_06_29.md`
- `docs/literature_review_scoping_2026_06_29.md`
- Updated handoff and project overview documents.

### Results

| Item | Result |
| --- | --- |
| Aji's latest LOAO/Gemini guidance recorded | Yes |
| Local downloaded literature mapped into themes | Yes |
| Hourglass dissertation framing drafted | Yes |
| Narrow topic proposed | Candidate-label open-topic aspect+sentiment classification for customer feedback |

### Interpretation

- The dissertation should start broadly from customer feedback analytics, review mining, ABSA, and multi-label classification.
- It should narrow to candidate-label open-topic aspect+sentiment classification, where held-out aspects are the central technical claim and held-out organisation is a supporting deployment-shift axis.
- It can broaden again to practical open-vocabulary feedback analytics systems, comparing local encoders, open LLMs, and hosted LLMs under accuracy, latency, cost, privacy, and governance constraints.
- Aji's latest guidance strengthens the role of LOAO in the open-topic claim and adds two concrete Gemini evaluation requirements: use JSON mode if available and include reasoning tokens in cost accounting.

### Next Step

- Use `docs/literature_review_scoping_2026_06_29.md` as the starting point for a full literature review matrix.
- Fill the missing literature gaps: multi-label evaluation, entailment-style zero-shot classification, domain generalisation, instruction-tuned ABSA, and practical LLM cost/latency evaluation.
- Ask a stronger reasoning model to critique and refine the hourglass framing before drafting the literature review prose.

## 2026-06-29: Eight-Stage Future Work Roadmap

### Purpose

- Record the agreed future-work order after reviewing the Pro-model dissertation-topic feedback.
- Keep the project oriented toward Qwen/Gemini and structured LLM comparisons while preserving a robust taxonomy-shift dissertation spine.

### Code Or Protocol Changes

- No code or modelling protocol changed.
- Updated `docs/next_stage_and_literature_review_plan.md` with an eight-stage roadmap.

### Results

The recorded stages are:

1. Freeze the dissertation spine and internal experiment spec.
2. Build the literature review matrix.
3. Complete LOAO for the strongest local baseline.
4. Run a controlled Gemini pilot.
5. Choose the main modelling branch.
6. Strengthen Qwen under the chosen branch.
7. Implement joint aspect-sentiment pair scoring if needed.
8. Write and synthesise the dissertation.

### Interpretation

- Qwen/Gemini remain a major intended direction, but the dissertation should not depend on them succeeding before the evaluation spine is stable.
- Stage 1 and Stage 2 are the immediate pre-experiment tasks.
- LOAO remains the first experimental priority after the literature/framework checkpoint.

### Next Step

- Complete Stage 1 and Stage 2 before resuming major modelling.

## 2026-06-29: Stage 1 And Stage 2 Completion

### Purpose

- Complete the immediate pre-experiment tasks before resuming LOAO, Gemini, Qwen, or joint pair-scoring work.
- Freeze the dissertation spine and experiment rules.
- Build the first usable literature review matrix.

### Code Or Protocol Changes

- No code changed.
- Added `docs/dissertation_internal_spec.md`.
- Added `docs/literature_review_matrix.md`.
- Updated:
  - `README.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`
  - `docs/next_stage_and_literature_review_plan.md`
  - `docs/experiment_log.md`

### Results

Stage 1 output:

- `docs/dissertation_internal_spec.md`
- Freezes the working spine:
  - `structured candidate-label aspect-sentiment modelling under taxonomy shift in customer feedback`
- Defines:
  - task variants
  - candidate-label input format
  - indexed JSON output schema
  - invalid-output handling
  - metrics
  - primary split/protocol roles
  - local two-stage baseline, structured LLMs, and joint pair scoring

Stage 2 output:

- `docs/literature_review_matrix.md`
- Maps local papers into:
  - customer review mining
  - ABSA
  - multi-label classification and evaluation
  - domain/organisation generalisation
  - open-topic taxonomy shift
  - candidate-label and zero-shot classification
  - structured-output LLMs and deployment trade-offs
- Records literature gaps and search strings.

### Interpretation

- The project now has a documented bridge from completed experiments to dissertation framing.
- The immediate next experimental priority remains LOAO for the strongest local baseline.
- Gemini/Qwen should be resumed only under the frozen candidate-label JSON/evaluation spec.

### Next Step

- Begin Stage 3: complete LOAO for the strongest local two-stage baseline, or pause experiments and expand `docs/literature_review_matrix.md` into a full related-work outline.
