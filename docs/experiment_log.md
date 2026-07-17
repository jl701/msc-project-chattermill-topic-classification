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

An experiment is not considered closed until the parameters/configuration and results have been recorded in project documentation and the safe tracked changes have been committed and pushed to GitHub. The minimum close-out checklist is:

1. Record the exact command, data split, row/label scope, model, prompt or feature pipeline, hyperparameters, random seed, hardware/API endpoint family, output directory, validation-selection rule, headline metrics, supporting metrics, and caveats.
2. Update `docs/experiment_reproducibility_register.md` when the experiment creates a new family of runs, a new canonical command, or a new headline result.
3. Run the relevant validation checks for the changed code/docs.
4. Check Git status and stage only safe files.
5. Commit and push code, documentation, and non-sensitive configs to GitHub so the experiment record is not lost.
6. If GitHub push is temporarily impossible, record the blocker and push as soon as the blocker is resolved.

Do not commit local `outputs/`, checkpoints, credentials, API keys, private data, or raw prediction files that may contain review text. Summarise results in this document and keep generated artifacts local unless explicitly approved.

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

An attempted full DistilBERT cross-encoder single-fold smoke did not finish within 30 minutes on the local 6 GB GPU, so full cross-encoder LOAO was not run as a formal result at this point. This status is superseded by the 2026-07-01 full DistilBERT LOAO run on the RTX 5050 Laptop GPU.

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
- Full cross-encoder LOAO was not run at this point; only a tiny cross-encoder smoke test was run because full local DistilBERT LOAO was too slow on the GTX 1660 Ti Max-Q 6 GB GPU. This status is superseded by the 2026-07-01 full DistilBERT LOAO run on the RTX 5050 Laptop GPU.

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

## 2026-07-01: Gemini Hosted Candidate-Label Runner Implementation

### Purpose

- Implement the hosted Gemini candidate-label baseline requested after the local non-LLM baseline phase.
- Reuse the Qwen indexed held-out-aspect protocol while adding Aji's required hosted-LLM diagnostics: JSON/schema validity, parse failures, invalid labels, latency, token usage, reasoning/thinking tokens, and optional cost estimates.
- Stop cleanly without fabricating results because no compatible API credentials were available in the environment.

### Code Or Protocol Changes

- Added `src/msc_project/llm/candidate_label.py`:
  - indexed candidate-label prompt construction
  - JSON schema for structured output
  - parser supporting Qwen-style top-level arrays and JSON-mode `{"labels": [...]}` wrappers
  - diagnostics for schema validity, invalid candidate IDs, invalid sentiments, duplicates, conflicts, and aspect-name fallbacks
  - OpenAI/Gemini-style token usage extraction and optional cost estimation
- Added `scripts/run_gemini_heldout_aspect.py`:
  - OpenAI-compatible Chat Completions HTTP client using the Python standard library
  - default model `vertex_ai/gemini-2.5-flash`
  - default `response_format=json_schema`
  - `--response-format-fallback` support for endpoints that reject JSON mode/schema
  - independent output directory pattern under `outputs/llm/gemini_candidate_label_YYYYMMDD_HHMMSS`
  - `--dry-run` mode for request construction without API calls
- Added `tests/test_llm_candidate_label.py`.
- Added `docs/gemini_candidate_label_baseline.md`.
- Updated:
  - `README.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`
  - `docs/generalisation_baselines.md`
  - `docs/qwen_feasibility.md`
  - `docs/experiment_log.md`

### Setup

- Dataset: FABSA public export.
- Protocol: fixed held-out-aspect candidate-label evaluation.
- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Default runner metadata strategy: `example_filtered`.
- Default split for smoke tests: validation.
- Default prompt variant: `indexed`.
- Default response format: `json_schema`.
- Environment audit:
  - `OPENAI_BASE_URL`: missing
  - `OPENAI_API_KEY`: missing
  - `GEMINI_API_KEY`: missing
  - `GOOGLE_API_KEY`: missing
  - `GOOGLE_APPLICATION_CREDENTIALS`: missing

### Commands

```powershell
git status --short --branch
git fetch origin
git status --short --branch
git log --oneline --decorate -8

python -m unittest tests.test_llm_candidate_label
python -m unittest discover -s tests
python -m compileall -q src scripts tests
python .\scripts\run_gemini_heldout_aspect.py --dry-run --split validation --limit 2 --response-format json_schema --output-dir .\outputs\llm\gemini_candidate_label_dry_run_check
```

### Outputs

- Code and documentation files listed above are intended for commit.
- Dry-run local outputs:
  - `outputs/llm/gemini_candidate_label_dry_run_check/summary.json`
  - `outputs/llm/gemini_candidate_label_dry_run_check/requests_validation_indexed.jsonl`
- Generated `outputs/` files remain ignored by Git and must not be committed because request JSONL files contain review text.

### Results

| Check | Result |
| --- | --- |
| Git state before work | clean `main`, aligned with `origin/main` |
| Parser/schema tests | 11 tests OK |
| Full unit test suite | 59 tests OK |
| Compile check | passed |
| Gemini dry-run request construction | passed |
| Real hosted Gemini smoke test | blocked by missing credentials |
| Gemini F1 metrics | not available |

### Interpretation

- The Gemini baseline is now implemented as a reproducible runner with the correct candidate-label output protocol and hosted-LLM diagnostics.
- JSON schema mode uses an object wrapper because JSON-mode endpoints commonly require a top-level object; the parser still accepts the Qwen top-level array for plain JSON compatibility.
- The implementation should be treated as ready for hosted smoke testing, not as a completed empirical result.
- No predictive claim about Gemini should be made until a real API run produces validation/test metrics.

### Next Step

- Once `OPENAI_BASE_URL` and `OPENAI_API_KEY` are available, run:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 5 --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_YYYYMMDD_HHMMSS
```

- If the smoke test passes, run a sampled validation sweep over at least `indexed`, `indexed_conservative`, and `indexed_descriptive`, then run full validation/test for the selected configuration.

## 2026-07-01: Gemini Flash Fixed Held-Out-Aspect Evaluation

### Purpose

- Run the hosted Gemini candidate-label baseline after the user supplied Aji's endpoint details and API key.
- Evaluate Gemini Flash under the same indexed candidate-label held-out-aspect protocol used for Qwen zero-shot.
- Check Aji's two requirements in practice:
  - use `response_format` / JSON mode if the endpoint supports it;
  - count reasoning/thinking tokens in cost diagnostics.

### Code Or Protocol Changes

- No code changed during the hosted run.
- Documentation was updated after the run:
  - `docs/gemini_candidate_label_baseline.md`
  - `docs/generalisation_baselines.md`
  - `docs/qwen_feasibility.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`
  - `README.md`
  - `docs/experiment_log.md`

### Setup

- Dataset: FABSA public export.
- Protocol: fixed held-out-aspect candidate-label evaluation.
- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Strategy metadata: `example_filtered`.
- Model: `vertex_ai/gemini-2.5-flash`.
- Endpoint: Chattermill Vertex AI OpenAI-compatible endpoint.
- Prompt: `indexed`.
- Response format: `json_schema`.
- Final max tokens: `2048`.
- Temperature: `0`.
- API key handling:
  - used only as a process-local environment variable;
  - not written to any repository file;
  - not committed.

### Commands

```powershell
python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 5 --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_20260701_0110_smoke

python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 50 --sample --prompt-variant indexed --prompt-variant indexed_conservative --prompt-variant indexed_descriptive --response-format json_schema --response-format-fallback --output-dir .\outputs\llm\gemini_candidate_label_20260701_0115_validation_sweep

python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 50 --sample --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 1024 --output-dir .\outputs\llm\gemini_candidate_label_20260701_0125_validation_indexed_max1024

python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 5 --sample --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 1024 --thinking-budget 0 --output-dir .\outputs\llm\gemini_candidate_label_20260701_0135_thinking0_smoke

python .\scripts\run_gemini_heldout_aspect.py --split validation --limit 50 --sample --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_20260701_0140_validation_indexed_max2048

python .\scripts\run_gemini_heldout_aspect.py --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full
```

### Outputs

- Local ignored output directories:
  - `outputs/llm/gemini_candidate_label_20260701_0110_smoke/`
  - `outputs/llm/gemini_candidate_label_20260701_0115_validation_sweep/`
  - `outputs/llm/gemini_candidate_label_20260701_0125_validation_indexed_max1024/`
  - `outputs/llm/gemini_candidate_label_20260701_0135_thinking0_smoke/`
  - `outputs/llm/gemini_candidate_label_20260701_0140_validation_indexed_max2048/`
  - `outputs/llm/gemini_candidate_label_20260701_0145_fixed_full/`
- Generated prediction/request files remain ignored by Git because they contain review text.

### Results

Small validation sweep on 50 sampled validation rows:

| Configuration | Max Tokens | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Valid JSON | Schema Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `indexed` | 512 | 0.5867 | 0.7033 | 0.5353 | 0.8400 | 0.8400 |
| `indexed_conservative` | 512 | 0.4933 | 0.6207 | 0.4215 | 0.8200 | 0.8000 |
| `indexed_descriptive` | 512 | 0.5067 | 0.6429 | 0.4575 | 0.7800 | 0.7800 |
| `indexed` | 1024 | 0.6733 | 0.7475 | 0.5453 | 0.9800 | 0.9800 |
| `indexed` | 2048 | 0.6733 | 0.7327 | 0.5287 | 1.0000 | 1.0000 |

Full fixed held-out-aspect evaluation:

| Split | Examples | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Schema Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 212 | 0.6146 | 0.6446 | 0.4830 | 0.7129 | 1.0000 | 0.9953 |
| test | 281 | 0.6071 | 0.6541 | 0.5547 | 0.6747 | 1.0000 | 1.0000 |

Token and approximate public-rate cost diagnostics:

| Split | Input Tokens | Output Tokens, Including Thinking | Reasoning Tokens | Total Tokens | Mean Latency | Approx Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 41,960 | 89,052 | 80,780 | 131,012 | 2.5074 s | $0.2352 |
| test | 56,338 | 113,067 | 102,574 | 169,405 | 2.3721 s | $0.2996 |

Approximate cost uses public Gemini 2.5 Flash Standard rates checked on 2026-07-01:

- input: `$0.30 / 1M tokens`
- output: `$2.50 / 1M tokens`

The endpoint reports `input_tokens + output_tokens = total_tokens`, and `reasoning_tokens` is a subset of output/completion tokens. Cost therefore bills output tokens once while still reporting reasoning tokens separately.

### Interpretation

- The endpoint honours `response_format=json_schema`; no plain-JSON fallback was needed.
- `max_tokens=512` is too low for this task because Gemini 2.5 Flash often spends nearly all completion tokens thinking first and returns truncated JSON.
- `max_tokens=2048` fixed schema reliability on the 50-row sweep and produced perfect test JSON/schema validity.
- The attempted `thinking_budget=0` smoke test did not materially suppress reasoning tokens, so it was not used as the final optimisation.
- Gemini 2.5 Flash is now stronger than Qwen indexed zero-shot on the fixed held-out-aspect test split.
- Gemini matches the strongest local non-LLM fixed held-out-aspect headline pair samples F1 (`0.6071`) and improves pair micro/macro F1, but the comparison has hosted cost, latency, and governance trade-offs.
- This is still a fixed three-aspect result, not LOAO robustness evidence.

### Next Step

- Do not run full Gemini LOAO by default.
- Useful next checks are:
  - row-level Gemini error analysis against Qwen and the local non-LLM baseline;
  - a small Gemini Pro subset if the dissertation needs an upper hosted-LLM comparison;
  - Qwen fine-tuning/evaluation on stronger GPU access;
  - LOAO robustness only if the cost/benefit is explicitly justified.

## 2026-07-01: Gemini Flash Fixed-Split Error Analysis And Pro Subset Preparation

### Purpose

- Analyse the completed Gemini Flash fixed held-out-aspect predictions beyond aggregate F1.
- Explain why Gemini matches the strongest local non-LLM pair samples F1 while improving pair micro/macro F1.
- Prepare the requested Gemini Pro small-subset comparison without running a full Pro or LOAO job.

### Code Or Protocol Changes

- Added `scripts/analyse_gemini_heldout_aspect_errors.py`.
- Added `docs/gemini_error_analysis.md`.
- Updated project summary docs with the Gemini error profile and Pro-subset status.
- No API credentials were written to code, docs, outputs, or Git.

### Setup

- Dataset: FABSA fixed held-out-aspect test split.
- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Evaluation scope: held-out aspect+sentiment pair labels only.
- Gemini Flash source output: `outputs/llm/gemini_candidate_label_20260701_0145_fixed_full/`.
- Local comparator: candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment, example-filtered, selector LR `3e-5`.
- Qwen comparator: Qwen3-4B indexed zero-shot test output.
- Pro subset status: not run because the active shell was missing `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `GOOGLE_API_KEY`. Under the current credential rule, the API key must be read only from shell environment variables.

### Commands

```powershell
python .\scripts\analyse_gemini_heldout_aspect_errors.py --output-dir .\outputs\analysis\gemini_error_analysis
```

Prepared but not run until endpoint variables are available:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-pro --split test --limit 50 --sample --seed 13 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_YYYYMMDD_HHMM_pro_test50
```

### Outputs

- Local ignored analysis output: `outputs/analysis/gemini_error_analysis/summary.json`.
- Committed documentation summary: `docs/gemini_error_analysis.md`.
- Generated outputs remain ignored by Git because prediction and per-row files may contain review text.

### Results

Aggregate fixed test comparison:

| Model | Pair Samples F1 | Pair Micro Precision | Pair Micro Recall | Pair Micro F1 | Pair Macro F1 | Exact Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT selector + DistilBERT sentiment | 0.6071 | 0.5333 | 0.6644 | 0.5917 | 0.4890 | 144 |
| Qwen3-4B indexed zero-shot | 0.5374 | 0.4870 | 0.5813 | 0.5300 | 0.4374 | 125 |
| Gemini 2.5 Flash indexed JSON-schema | 0.6071 | 0.6475 | 0.6609 | 0.6541 | 0.5547 | 138 |

Prediction-cardinality diagnostics:

| Model | Pred Labels / Row | Empty Prediction Rows | Aspect Over-Predict Rows | Aspect Miss Rows | Sentiment Error Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT selector + DistilBERT sentiment | 1.2811 | 0 | 119 | 78 | 18 |
| Qwen3-4B indexed zero-shot | 1.2278 | 15 | 117 | 86 | 35 |
| Gemini 2.5 Flash indexed JSON-schema | 1.0498 | 42 | 80 | 78 | 20 |

Gemini aspect-level behaviour:

| Aspect | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Account management: Account access | 0.6260 | 0.9747 | 0.7624 |
| Company brand: Competitor | 0.8906 | 0.4711 | 0.6162 |
| Value: Discounts promotions | 0.7130 | 0.8652 | 0.7817 |

Flash baseline on the prepared Pro 50-row test subset (`--limit 50 --sample --seed 13`):

| Model / Subset | Rows | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Schema Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini Flash, same Pro subset | 50 | 0.6360 | 0.6731 | 0.4627 | 0.7560 | 1.0000 | 1.0000 |

Same-subset Flash token/latency diagnostics:

| Input Tokens | Output Tokens, Including Thinking | Reasoning Tokens | Total Tokens | Mean Latency | Approx Flash Cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9,728 | 18,164 | 16,274 | 27,892 | 2.1751 s | $0.0483 |

Using public Gemini API Standard rates from the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing), checked on 2026-07-01, a same-token Gemini 2.5 Pro planning estimate for this 50-row subset is about `$0.1938`. Actual Pro cost may differ because Pro may spend different reasoning/output tokens.

### Interpretation

- Gemini Flash ties the local DistilBERT headline samples F1 through a different error profile, not because the two models make the same predictions.
- Gemini is more selective and precise: fewer predicted labels per row and fewer aspect over-predictions.
- The local DistilBERT pipeline has slightly more exact rows and no empty predictions, but it over-predicts candidate aspects much more often.
- Gemini's main aspect-level weakness is recall for `Company brand: Competitor`, especially positive competitor mentions.
- The 50-row Pro subset is justified as a small upper-bound check once credentials are present, but full Pro validation/test or Pro LOAO is not justified until the subset shows a clear gain over Flash.

### Next Step

- Set `OPENAI_BASE_URL` and an API-key variable in the shell, then run only the prepared 50-row Gemini Pro subset.
- Do not run full Gemini Pro or Gemini LOAO unless the small subset provides a strong cost/latency/value reason.

## 2026-07-01: Gemini Pro 50-Row Fixed-Subset Comparison

### Purpose

- Run the requested small Gemini Pro comparison after the user explicitly authorised API use beyond the earlier environment-variable-only restriction.
- Compare Pro against Flash on the same deterministic fixed held-out-aspect test subset.
- Decide whether full Pro validation/test or Pro LOAO is justified.

### Code Or Protocol Changes

- Extended `scripts/analyse_gemini_heldout_aspect_errors.py` with optional `--pro-predictions` support.
- Updated Gemini documentation with the completed Pro subset result.
- No API key was committed, written into project documentation, or written into model output JSON.

### Setup

- Dataset: FABSA fixed held-out-aspect test split.
- Row selection: `--limit 50 --sample --seed 13`.
- Labels: held-out aspect+sentiment pair labels only.
- Model: `vertex_ai/gemini-2.5-pro`.
- Prompt: indexed candidate labels.
- Response format: `json_schema`.
- Max tokens: `2048`.
- Temperature: `0`.

### Commands

```powershell
python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-pro --split test --limit 50 --sample --seed 13 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_20260701_025042_pro_test50

python .\scripts\analyse_gemini_heldout_aspect_errors.py --pro-predictions .\outputs\llm\gemini_candidate_label_20260701_025042_pro_test50\predictions_test_indexed.jsonl --output-dir .\outputs\analysis\gemini_error_analysis_with_pro
```

### Outputs

- Local ignored Pro output: `outputs/llm/gemini_candidate_label_20260701_025042_pro_test50/`.
- Local ignored analysis output: `outputs/analysis/gemini_error_analysis_with_pro/summary.json`.
- Documentation updated in:
  - `docs/gemini_error_analysis.md`
  - `docs/gemini_candidate_label_baseline.md`
  - `docs/generalisation_baselines.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`

### Results

Same 50-row test subset comparison:

| Model | Pair Samples F1 | Pair Micro P | Pair Micro R | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Schema Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini Flash | 0.6360 | 0.6604 | 0.6863 | 0.6731 | 0.4627 | 0.7560 | 1.0000 | 1.0000 |
| Gemini Pro | 0.6933 | 0.7308 | 0.7451 | 0.7379 | 0.5250 | 0.7933 | 1.0000 | 1.0000 |

Row-level exactness:

| Comparison | Rows |
| --- | ---: |
| Both exact | 24 |
| Pro only exact | 6 |
| Flash only exact | 3 |
| Neither exact | 17 |

Latency and token/cost diagnostics:

| Model | Mean Latency | Input Tokens | Output Tokens, Including Thinking | Reasoning Tokens | Total Tokens | Approx Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini Flash | 2.1751 s | 9,728 | 18,164 | 16,274 | 27,892 | $0.0483 |
| Gemini Pro | 4.9026 s | 9,728 | 25,507 | 23,625 | 35,235 | $0.2672 |

Approximate cost uses public Gemini API Standard rates from the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing), checked on 2026-07-01. Output tokens include thinking tokens.

### Interpretation

- Pro is directionally stronger than Flash on this small fixed-subset check: `+0.0573` pair samples F1, `+0.0648` pair micro F1, `+0.0623` pair macro F1.
- Pro also keeps perfect JSON/schema validity on this subset.
- The improvement is bought with about `2.25x` higher mean latency and `5.53x` higher approximate cost.
- This is useful as an upper hosted-LLM diagnostic, but it is not a full benchmark and should not be mixed with full-split Flash results without the subset qualifier.

### Next Step

- Do not run full Gemini Pro validation/test or Pro LOAO by default.
- Full Pro is only worth considering if the dissertation needs a stronger hosted-LLM upper-bound comparison and the added cost/latency can be justified explicitly.

## 2026-07-01: Gemini Follow-Up Experiment Rationale

### Purpose

- Record the dissertation-driven rationale for the next Gemini/local experiments before running additional hosted jobs.
- Separate low-cost fixed-split hosted baselines from expensive LOAO expansion.
- Preserve the insight that the most valuable future Gemini work is system and methods evidence, not only a higher F1 table.

### Code Or Protocol Changes

- Updated project planning documentation only.
- No model code changed in this rationale step.

### Setup

- Existing evidence:
  - full fixed-split Gemini Flash validation/test is complete;
  - 50-row Gemini Pro subset improves over Flash but is slower and more expensive;
  - local DistilBERT remains a strong non-hosted fixed-split baseline;
  - full Gemini Pro LOAO is still expensive and not justified by default.

### Results

The selected near-term sequence is:

1. Run full fixed-split Gemini Pro validation/test as the stronger hosted upper-bound baseline.
2. Run full fixed-split Gemini Flash-Lite validation/test as the cheapest hosted baseline.
3. Build a local-to-Gemini uncertainty cascade after the hosted Pareto table is complete.
4. Test Gemini-generated candidate-aspect descriptions as a label-representation experiment without validation/test leakage.
5. Use Gemini Pro as a qualitative error-taxonomy aid with manual review, not as an automatic evaluator.

### Interpretation

- Full Pro fixed validation/test is now justified because it is expected to cost only a few dollars and removes the subset caveat.
- Flash-Lite is useful because it gives the cheapest hosted point in an accuracy-cost-latency Pareto comparison.
- The cascade experiment is likely the most industrially relevant follow-up because it tests whether local models can handle most rows while Gemini handles uncertain cases.
- Candidate descriptions are methodologically valuable because they probe how new candidate labels should be represented.
- Qualitative error taxonomy helps the dissertation discussion but should not be treated as metric evidence.

### Next Step

- Run full fixed-split Gemini Pro validation/test.
- Run full fixed-split Gemini Flash-Lite validation/test.
- Do not run full Gemini Pro LOAO in this phase.

## 2026-07-01: Gemini Pro And Flash-Lite Full Fixed-Split Evaluation

### Purpose

- Complete the fixed-split hosted Pareto comparison after the 50-row Pro subset showed a clear directional gain over Flash.
- Add a full Pro upper-bound hosted baseline.
- Add a full Flash-Lite cheapest/lowest-latency hosted baseline.
- Avoid full Gemini LOAO in this phase.

### Code Or Protocol Changes

- No runner changes were required.
- Updated project documentation with the completed Pro and Flash-Lite results.
- API credentials were used only for the hosted calls and were not committed or written into tracked files.

### Setup

- Dataset: FABSA fixed held-out-aspect validation and test splits.
- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Evaluation scope: held-out aspect+sentiment pair labels only.
- Prompt: indexed candidate labels.
- Response format: `json_schema`.
- Max tokens: `2048`.
- Temperature: `0`.
- Models:
  - `vertex_ai/gemini-2.5-pro`
  - `vertex_ai/gemini-2.5-flash-lite`

### Commands

```powershell
python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-pro --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 1.25 --output-cost-per-1m 10.00 --output-dir .\outputs\llm\gemini_candidate_label_20260701_031040_pro_fixed_full

python .\scripts\run_gemini_heldout_aspect.py --model vertex_ai/gemini-2.5-flash-lite --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40 --output-dir .\outputs\llm\gemini_candidate_label_20260701_034545_flash_lite_fixed_full
```

### Outputs

- Local ignored Pro output: `outputs/llm/gemini_candidate_label_20260701_031040_pro_fixed_full/`.
- Local ignored Flash-Lite output: `outputs/llm/gemini_candidate_label_20260701_034545_flash_lite_fixed_full/`.
- Documentation updated in:
  - `docs/gemini_candidate_label_baseline.md`
  - `docs/generalisation_baselines.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`

### Results

Fixed held-out-aspect validation/test:

| Model | Split | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Valid JSON | Schema Valid | Mean Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini Flash-Lite | validation | 0.5876 | 0.5973 | 0.4477 | 0.6783 | 1.0000 | 1.0000 | 0.3605 s |
| Gemini Flash-Lite | test | 0.5516 | 0.5872 | 0.4876 | 0.6062 | 1.0000 | 0.9964 | 0.3719 s |
| Gemini Flash | validation | 0.6146 | 0.6446 | 0.4830 | 0.7129 | 1.0000 | 0.9953 | 2.5074 s |
| Gemini Flash | test | 0.6071 | 0.6541 | 0.5547 | 0.6747 | 1.0000 | 1.0000 | 2.3721 s |
| Gemini Pro | validation | 0.7270 | 0.7377 | 0.6092 | 0.7954 | 1.0000 | 0.9953 | 5.3514 s |
| Gemini Pro | test | 0.7141 | 0.7425 | 0.6287 | 0.7746 | 1.0000 | 1.0000 | 5.6510 s |

Token and approximate public-rate cost diagnostics:

| Model | Split | Input Tokens | Output Tokens, Including Thinking | Reasoning Tokens | Total Tokens | Approx Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gemini Flash-Lite | validation | 41,960 | 8,206 | not reported | 50,166 | $0.0075 |
| Gemini Flash-Lite | test | 56,338 | 9,852 | not reported | 66,190 | $0.0096 |
| Gemini Flash | validation | 41,960 | 89,052 | 80,780 | 131,012 | $0.2352 |
| Gemini Flash | test | 56,338 | 113,067 | 102,574 | 169,405 | $0.2996 |
| Gemini Pro | validation | 41,960 | 121,157 | 112,365 | 163,117 | $1.2640 |
| Gemini Pro | test | 56,338 | 164,362 | 153,349 | 220,700 | $1.7140 |

Approximate validation+test costs:

| Model | Approx Cost |
| --- | ---: |
| Gemini Flash-Lite | $0.0171 |
| Gemini Flash | $0.5348 |
| Gemini Pro | $2.9781 |

### Interpretation

- The full Pro run confirms the 50-row subset direction. Pro is clearly the strongest fixed-split hosted model, with `0.7141` test pair samples F1 versus Flash's `0.6071`.
- Flash-Lite is much cheaper and faster than Flash and Pro, but its test F1 is below the strongest local non-LLM baseline and below Flash.
- Flash remains the balanced hosted baseline: it matches the strongest local fixed-split headline result while improving micro/macro F1.
- The hosted Pareto table is now complete for the fixed three-aspect protocol.
- These results are still not LOAO robustness evidence.

### Next Step

- Do not run full Gemini Pro LOAO by default.
- Next dissertation-value experiments should be tackled separately:
  - local-to-Gemini uncertainty cascade;
  - Gemini-generated candidate-label descriptions;
  - Gemini-assisted qualitative error taxonomy with manual review.

### Validation

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

| Check | Result |
| --- | --- |
| Unit tests | 63 tests OK |
| Compile check | passed |

## 2026-07-01 - Gemini-Generated Aspect Descriptions

### Purpose

Task 4 tested whether Gemini-generated natural-language descriptions for the three fixed held-out candidate aspects improve hosted candidate-label classification. Descriptions were generated from canonical aspect names only, without validation or test review text.

### Code And Configs

- Added generated-description prompt variants:
  - `indexed_generated_descriptions`
  - `indexed_conservative_generated_descriptions`
- Added `--aspect-descriptions-json` to `scripts/run_gemini_heldout_aspect.py`.
- Added non-sensitive tracked configs:
  - `configs/gemini_aspect_descriptions_fixed_heldout.json`
  - `configs/gemini_aspect_descriptions_decision_boundary_heldout.json`
- Detailed write-up: `docs/gemini_aspect_descriptions.md`.

### Commands

Representative commands:

```powershell
python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_generated_descriptions_dry_run_check --split validation --limit 2 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --dry-run

python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_flash_lite_validation_full --model vertex_ai/gemini-2.5-flash-lite --split validation --limit 10000 --prompt-variant indexed_generated_descriptions --prompt-variant indexed_conservative_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40

python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_boundary_flash_lite_validation_full --model vertex_ai/gemini-2.5-flash-lite --split validation --limit 10000 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_decision_boundary_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.10 --output-cost-per-1m 0.40

python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_flash_validation_full --model vertex_ai/gemini-2.5-flash --split validation --limit 10000 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 0.30 --output-cost-per-1m 2.50

python .\scripts\run_gemini_heldout_aspect.py --output-dir .\outputs\llm\gemini_candidate_label_20260701_desc_pro_val50 --model vertex_ai/gemini-2.5-pro --split validation --limit 50 --sample --seed 13 --prompt-variant indexed_generated_descriptions --aspect-descriptions-json .\configs\gemini_aspect_descriptions_fixed_heldout.json --response-format json_schema --response-format-fallback --max-tokens 2048 --input-cost-per-1m 1.25 --output-cost-per-1m 10.00
```

### Outputs

Local ignored outputs:

- `outputs/llm/gemini_candidate_label_20260701_desc_flash_lite_val50/`
- `outputs/llm/gemini_candidate_label_20260701_desc_flash_val50/`
- `outputs/llm/gemini_candidate_label_20260701_desc_flash_lite_validation_full/`
- `outputs/llm/gemini_candidate_label_20260701_desc_flash_lite_test_full/`
- `outputs/llm/gemini_candidate_label_20260701_desc_boundary_flash_lite_validation_full/`
- `outputs/llm/gemini_candidate_label_20260701_desc_boundary_flash_lite_test_full/`
- `outputs/llm/gemini_candidate_label_20260701_desc_flash_validation_full/`
- `outputs/llm/gemini_candidate_label_20260701_desc_flash_test_full/`
- `outputs/llm/gemini_candidate_label_20260701_desc_pro_val50/`
- `outputs/analysis/gemini_description_ablation_summary.json`

Generated outputs remain ignored because prediction and request files can contain review text.

### Results

Flash-Lite full test:

| Prompt | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Indexed baseline | 0.5516 | 0.5872 | 0.4876 | 0.6062 | $0.0096 |
| Label-only descriptions | 0.5925 | 0.6263 | 0.5691 | 0.6625 | $0.0118 |
| Decision-boundary descriptions | 0.5724 | 0.6199 | 0.5456 | 0.6340 | $0.0119 |

Flash full test:

| Prompt | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Indexed baseline | 0.6071 | 0.6541 | 0.5547 | 0.6747 | $0.2996 |
| Label-only descriptions | 0.5893 | 0.6555 | 0.5655 | 0.6676 | $0.3390 |

Pro 50-row validation diagnostic:

| Prompt | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 |
| --- | ---: | ---: | ---: | ---: |
| Indexed baseline | 0.8000 | 0.8113 | 0.5985 | 0.8400 |
| Label-only descriptions | 0.7467 | 0.7921 | 0.6128 | 0.8067 |

### Interpretation

- Description prompts are useful for Flash-Lite, where label-only descriptions substantially improve test F1 with a small cost increase.
- Decision-boundary wording reduces false positives but can increase empty predictions and false negatives.
- Flash descriptions improve validation but reduce test pair samples F1, while slightly improving test micro/macro F1.
- Pro did not show a positive primary-metric signal on the validation diagnostic, so full Pro descriptions were not run.
- The dissertation should frame this as a label-semantics ablation and precision-recall trade-off, not as a universal prompt improvement.

### Validation

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
rg -n "sk-[A-Za-z0-9_\-]{12,}" PROJECT_OVERVIEW.md START_NEW_CHAT_PROMPT.md report_notes.md configs docs scripts src tests
git diff --check
```

| Check | Result |
| --- | --- |
| Unit tests | 73 tests OK |
| Compile check | passed |
| Secret scan | no matches |
| Diff whitespace check | passed |

## 2026-07-02 - Gemini-Assisted Qualitative Error Taxonomy

### Purpose

Task 5 converted the completed local/Qwen/Gemini/cascade experiments into a manually consolidated qualitative error taxonomy. The goal was to strengthen the thesis discussion and define the main targets for later Qwen fine-tuning, not to produce another leaderboard result.

### Pre-Registered Configuration

The configuration was recorded before implementation in `docs/qualitative_error_taxonomy.md` and `report_notes.md`.

Key choices:

- reuse existing fixed-split and LOAO outputs;
- avoid new model evaluation runs;
- generate local row-level taxonomy packets with and without review text;
- keep raw review text and Gemini prompts/drafts local-only;
- use Gemini Pro only as an assistant for candidate category drafting;
- manually consolidate the final taxonomy.

### Commands

```powershell
python .\scripts\analyse_qualitative_error_taxonomy.py --output-dir .\outputs\analysis\qualitative_error_taxonomy_20260702 --max-examples-per-category 10 --max-gemini-examples-per-category 3 --snippet-chars 320

python .\scripts\analyse_qualitative_error_taxonomy.py --output-dir .\outputs\analysis\qualitative_error_taxonomy_20260702 --max-examples-per-category 10 --max-gemini-examples-per-category 2 --snippet-chars 220 --gemini-draft --gemini-model vertex_ai/gemini-2.5-pro --gemini-max-tokens 7000 --request-timeout 240
```

### Outputs

Local ignored outputs:

- `outputs/analysis/qualitative_error_taxonomy_20260702/summary.json`
- `outputs/analysis/qualitative_error_taxonomy_20260702/category_counts.csv`
- `outputs/analysis/qualitative_error_taxonomy_20260702/fixed_split_cases_no_text.jsonl`
- `outputs/analysis/qualitative_error_taxonomy_20260702/fixed_split_cases_with_text.jsonl`
- `outputs/analysis/qualitative_error_taxonomy_20260702/gemini_taxonomy_prompt.md`
- `outputs/analysis/qualitative_error_taxonomy_20260702/gemini_taxonomy_draft.json`

Tracked write-up:

- `docs/qualitative_error_taxonomy.md`

### Results

Aligned fixed-split row summary:

| System | Mean Row F1 | Exact Rows | Empty Rows | Aspect Miss Rows | Aspect Over-Pred Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT | 0.6071 | 144 | 0 | 78 | 119 |
| Qwen zero-shot | 0.5374 | 125 | 15 | 86 | 117 |
| Gemini Flash | 0.6071 | 138 | 42 | 78 | 80 |
| Gemini Pro | 0.7141 | 167 | 28 | 49 | 67 |
| Local -> Pro cascade | 0.8102 | 194 | 0 | 28 | 74 |

Final manually consolidated taxonomy:

1. Semantic boundary bleed.
2. Competitor-positive recall bottleneck.
3. Generative over-prediction / fail-noisy behaviour.
4. Cautious abstention / fail-silent behaviour.
5. Sentiment polarity under-recall.
6. Prompt-induced precision-recall shift.
7. Cascade complementarity.

Gemini-assisted draft:

- Model: `vertex_ai/gemini-2.5-pro`
- Prompt tokens: 19,494
- Completion tokens: 6,362
- Reasoning tokens: 4,336
- Total tokens: 25,856
- Approx public-rate cost: about `$0.088`

### Interpretation

This taxonomy gives the next Qwen fine-tuning phase concrete targets: abstention/no-label calibration, hard-negative aspect boundaries, competitor-positive recall, neutral sentiment coverage, stable label semantics, and cascade-ready uncertainty signals.

### Validation

| Check | Result |
| --- | --- |
| Unit tests | 75 tests OK: `python -m unittest discover -s tests` |
| Compile check | passed: `python -m compileall -q src scripts tests` |
| Diff whitespace check | passed: `git diff --check` |
| Secret scan | no `sk-...` or Google API-key pattern matches in tracked project/docs/scripts/test/thesis paths |

## 2026-07-01: Qwen LOAO Full Interpretation Analysis

### Purpose

- Convert the completed Qwen full all-row LOAO run into a dissertation-ready evidence block.
- Compare Qwen zero-shot LOAO against the preferred local DistilBERT LOAO result, not only against fixed held-out-aspect baselines.
- Separate all-row open-topic robustness from positive-gold-row sentiment diagnostics and fixed held-out-aspect Gemini/cascade evidence.

### Code Or Protocol Changes

- Added `scripts/analyse_qwen_loao_comparison.py`.
- Added `tests/test_qwen_loao_comparison.py`.
- The script reads existing Qwen LOAO summaries, Qwen positive-gold diagnostic summaries, and DistilBERT LOAO CSV summaries, then writes derived comparison CSV/JSON under ignored `outputs/analysis/`.
- No new model inference was run.
- No raw prediction file, review text, data file, checkpoint, credential, or model weight is committed.

### Setup

- Qwen model: `Qwen/Qwen3-4B-Instruct-2507`.
- Qwen loading: local 4-bit bitsandbytes NF4 double quantisation.
- Prompt: indexed candidate-label JSON array using `aspect_id`.
- LOAO protocol: all 12 FABSA aspects, one held-out candidate aspect per fold.
- Main evaluation: all official validation/test rows, gold labels filtered to the held-out aspect, empty predictions allowed.
- Local comparator: candidate-aspect DistilBERT cross-encoder + DistilBERT aspect-conditioned sentiment, `example_filtered`, validation pair-micro-F1 threshold selection, selector LR `3e-5`.

### Command

```powershell
python .\scripts\analyse_qwen_loao_comparison.py --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --qwen-positive-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701 --distilbert-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630 --distilbert-lr2e5-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701 --output-dir .\outputs\analysis\qwen_loao_full_interpretation_20260701
```

### Outputs

- Local ignored derived output directory: `outputs/analysis/qwen_loao_full_interpretation_20260701/`.
- Files written:
  - `loao_system_comparison.csv`
  - `qwen_vs_distilbert_per_aspect_test.csv`
  - `qwen_all_row_vs_positive_gold_test.csv`
  - `summary.json`
- Committed documentation:
  - `docs/qwen_loao_experiment_analysis.md`
  - updated Qwen, LOAO, generalisation, project overview, handoff, and reproducibility docs.

### Results

Test all-row LOAO mean across 12 held-out aspects:

| System | Pair Samples F1 | Pair Micro F1 | Precision | Recall | Pair Macro F1 | FP Rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B indexed zero-shot | 0.1212 | 0.3378 | 0.2379 | 0.8182 | 0.2412 | 34.4150 |
| DistilBERT cross-encoder + DistilBERT sentiment, LR `3e-5` | 0.0550 | 0.3128 | 0.3345 | 0.4315 | 0.2285 | 12.7022 |

Qwen positive-gold-row diagnostic from the same prediction files:

| Split | Pair Samples F1 Mean | Pair Micro F1 Mean | Precision Mean | Recall Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 0.8194 | 0.8659 | 0.9338 | 0.8182 | 0.9314 |

Per-aspect comparison:

- Qwen beats the preferred DistilBERT LOAO row on 6 of 12 aspects by pair micro F1.
- DistilBERT beats Qwen on 6 of 12 aspects.
- Qwen's largest gains are on `Online experience: App website`, `Company brand: General satisfaction`, `Logistics rides: Speed`, and `Staff support: Attitude of staff`.
- Qwen's largest losses are on `Staff support: Phone`, `Staff support: Email`, `Account management: Account access`, and `Value: Price value for money`.

Runtime and output reliability:

- Qwen validation+test generation time from recorded summaries: about 9.35 hours.
- Weighted Qwen validation+test throughput: about 1.0613 seconds/example.
- Qwen test valid JSON rate: 1.0000.
- Qwen test schema-valid rate: 0.9955, with recoverable prefixed aspect IDs as the main schema issue.

### Interpretation

- Qwen zero-shot is a useful local open-weight LOAO baseline before fine-tuning, but it is not a calibrated open-topic system.
- The Qwen advantage over DistilBERT is recall-driven. It recognises many more true held-out positives, but predicts the candidate aspect too often on empty-gold rows.
- The positive-gold diagnostic shows that Qwen's semantic recognition and sentiment assignment are strong when the held-out aspect is present.
- The all-row result shows that the hard problem is absence calibration under taxonomy shift.
- Fixed held-out-aspect Qwen/Gemini/cascade results remain conceptually separate. They are not LOAO robustness evidence.

### Next Step

- Treat `docs/qwen_loao_experiment_analysis.md` as the main thesis-facing Qwen LOAO analysis note.
- Use this as the zero-shot open-weight comparator before Qwen fine-tuning or calibration.
- If Qwen fine-tuning proceeds, keep the indexed candidate-label prompt format and evaluate with the same all-row LOAO protocol.

### Validation

```powershell
python .\scripts\analyse_qwen_loao_comparison.py --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --qwen-positive-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701 --distilbert-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630 --distilbert-lr2e5-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701 --output-dir .\outputs\analysis\qwen_loao_full_interpretation_20260701
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git diff --check
rg -n "sk-[A-Za-z0-9_-]{12,}|AIza[0-9A-Za-z_-]{20,}|Bearer [A-Za-z0-9._-]{20,}" . --glob '!outputs/**' --glob '!data/**' --glob '!models/**' --glob '!checkpoints/**' --glob '!artifacts/**' --glob '!runs/**' --glob '!.git/**'
git ls-files outputs data models checkpoints artifacts runs
```

| Check | Result |
| --- | --- |
| Comparison script | passed |
| Unit tests | 71 tests OK |
| Compile check | passed |
| Diff whitespace check | passed, with CRLF conversion warnings only |
| Secret scan over tracked code/docs paths | no real API key found |
| Tracked generated-output/data directories | none found |

## 2026-07-01: Qwen Zero-Shot Full All-Row LOAO Baseline

### Purpose

- Complete the missing local open-weight LLM zero-shot LOAO robustness baseline before Qwen fine-tuning.
- Use the actual project model, `Qwen/Qwen3-4B-Instruct-2507`, without fine-tuning.
- Evaluate whether indexed candidate-label prompting remains robust when every FABSA aspect is rotated into the unseen candidate position.
- Keep this evidence separate from fixed three-aspect Qwen/Gemini results and from Gemini cascade deployment evidence.

### Code Or Protocol Changes

- Added `src/msc_project/llm/qwen_local.py` with shared local Qwen chat-template, 4-bit loading, and deterministic generation helpers.
- Added `scripts/run_qwen_loao_heldout_aspect.py`.
  - Defaults to indexed candidate labels and 4-bit loading.
  - Runs all 12 FABSA aspects unless `--heldout-aspect` is supplied.
  - Uses all-row validation/test evaluation by default.
  - Writes request/prediction JSONL under ignored `outputs/`.
  - Supports `--resume` for complete or partial prediction JSONL files.
  - Writes per-aspect CSV/JSON summaries and aggregate spread tables.
- Reused `msc_project.llm.candidate_label` parsing/diagnostics and `evaluate_pair_and_aspect`.
- Updated the central candidate-label parser to recover prefixed IDs such as `A1. Account management: Account access`, while still counting those rows as schema-invalid.
- Added tests for the Qwen LOAO runner helpers and prefixed-ID parsing.

### Setup

- Dataset: FABSA official validation/test splits.
- Protocol: leave-one-aspect-out over all 12 FABSA aspects.
- Main evaluation scope: all official validation/test rows for each fold.
- Gold labels: filtered to the held-out aspect only.
- Candidate set: exactly the current held-out aspect.
- Empty predictions: allowed.
- Prompt: indexed candidate-label JSON array with `aspect_id`.
- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Loading: local Transformers causal LM, 4-bit bitsandbytes NF4 double quantisation.
- Decoding: deterministic generation, `max_input_tokens=1024`, `max_new_tokens=192`.
- Hardware: RTX 5050 Laptop GPU, 8 GB VRAM.

### Commands

Smoke tests:

```powershell
python .\scripts\run_qwen_loao_heldout_aspect.py --split both --heldout-aspect "Account management: Account access" --limit 30 --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_smoke_account_access_20260701
python .\scripts\run_qwen_loao_heldout_aspect.py --split validation --heldout-aspect "Account management: Account access" --limit 12 --prompt-variant indexed --load-in-4bit --output-dir .\outputs\llm\qwen_loao_smoke_parser_check_20260701
```

Full LOAO:

```powershell
python .\scripts\run_qwen_loao_heldout_aspect.py --split validation --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701
python .\scripts\run_qwen_loao_heldout_aspect.py --split test --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701
```

Derived positive-gold diagnostic, computed from the same prediction files without new model calls:

```powershell
python .\scripts\analyse_qwen_loao_predictions.py --validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701
```

### Outputs

- Local ignored outputs:
  - `outputs/llm/qwen_loao_smoke_account_access_20260701/`
  - `outputs/llm/qwen_loao_smoke_parser_check_20260701/`
  - `outputs/llm/qwen_loao_heldout_aspect_all_rows_validation_20260701/`
  - `outputs/llm/qwen_loao_heldout_aspect_all_rows_test_20260701/`
  - `outputs/analysis/qwen_loao_positive_diagnostic_20260701/`
- Raw predictions contain review text and remain uncommitted under ignored `outputs/`.
- Committed files are limited to code, tests, and aggregate documentation.

### Results

Smoke checks:

- Model loaded in 4-bit on the RTX 5050 Laptop GPU.
- Valid JSON was stable.
- Candidate IDs mapped back to canonical aspects.
- A recoverable schema issue was found and fixed: Qwen sometimes emitted `aspect_id` values like `A1. Account management: Account access`; these now map to `A1` but still count as schema-invalid.
- Smoke speed was about `0.7-1.1` seconds/example for the sampled all-row one-candidate setting.

Full all-row LOAO aggregate spread:

| Split | Rows / Fold | Wall Runtime | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | Pair Macro F1 Mean | FP Rows / 100 Mean | Valid JSON | Schema Valid | Seconds / Example |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 1,057 | 3.45 h | 0.1194 | 0.3293 | 0.2310 | 0.8115 | 0.2340 | 34.7446 | 1.0000 | 0.9961 | 0.9756 |
| test | 1,587 | 5.93 h | 0.1212 | 0.3378 | 0.2379 | 0.8182 | 0.2412 | 34.4150 | 1.0000 | 0.9955 | 1.1184 |

Test pair micro F1 ranged from `0.0513` for `Company brand: Reviews` to `0.6011` for `Online experience: App website`.

Hardest test aspects by pair micro F1:

| Held-Out Aspect | Pair Samples F1 | Pair Micro F1 | Precision | Recall | FP Rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Company brand: Reviews` | 0.0176 | 0.0513 | 0.0266 | 0.7368 | 64.2722 |
| `Staff support: Email` | 0.0132 | 0.1144 | 0.0609 | 0.9545 | 20.3529 |
| `Account management: Account access` | 0.0378 | 0.1527 | 0.0849 | 0.7595 | 40.1386 |
| `Value: Discounts promotions` | 0.0473 | 0.1913 | 0.1079 | 0.8427 | 38.5003 |

Strongest test aspects by pair micro F1:

| Held-Out Aspect | Pair Samples F1 | Pair Micro F1 | Precision | Recall | FP Rows / 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Logistics rides: Speed` | 0.1040 | 0.5660 | 0.4188 | 0.8730 | 14.3037 |
| `Staff support: Attitude of staff` | 0.1059 | 0.5685 | 0.4308 | 0.8358 | 13.6736 |
| `Online experience: App website` | 0.3527 | 0.6011 | 0.4969 | 0.7604 | 34.1525 |

Positive-gold-row diagnostic from the same predictions:

| Split | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | Pair Macro F1 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.8129 | 0.8692 | 0.9481 | 0.8115 | 0.6024 | 0.9451 |
| test | 0.8194 | 0.8659 | 0.9338 | 0.8182 | 0.6184 | 0.9314 |

### Interpretation

- Qwen zero-shot structured output is reliable: valid JSON is `1.0000` on both full all-row splits.
- Full all-row LOAO exposes high recall but weak absence calibration. On test, Qwen's mean recall is `0.8182`, but mean precision is only `0.2379`, with `34.4150` false-positive rows per 100 reviews.
- The positive-gold diagnostic is strong, so the main failure mode is not sentiment assignment or canonical ID mapping once the held-out aspect is present. The bottleneck is deciding that a candidate aspect is absent from empty-gold rows.
- Compared with the strongest DistilBERT LOAO run, Qwen has higher mean pair samples F1 (`0.1212` vs `0.0550`) and slightly higher mean pair micro F1 (`0.3378` vs `0.3128`), but with much lower precision and many more false-positive rows. Qwen is a better recall-oriented semantic matcher; DistilBERT is more conservative.
- Compared with the fixed held-out-aspect Qwen result (`0.5374` test pair samples F1, `0.5300` pair micro F1), full all-row LOAO is far harder and should be treated as the robustness evidence.
- Gemini fixed and cascade results remain conceptually separate. They are not LOAO robustness evidence.
- This result supports Qwen fine-tuning or calibration as the next open-weight LLM step. Zero-shot Qwen is useful but not deployable as a calibrated open-topic all-row detector.

### Limitations

- The run used one candidate aspect per LOAO fold, matching the frozen LOAO protocol. It does not test multi-candidate full-taxonomy prompting.
- The split-builder `strategy` is recorded as `label_masked`, but no training is performed and all-row evaluation rows are unchanged by that setting.
- The positive-row diagnostic is not a robustness result; it only explains behaviour after filtering to rows where the held-out aspect is present.
- Aspect-only LOAO metrics should be treated cautiously in one-candidate all-row settings because true-negative-heavy rows can make them look more optimistic than pair metrics.

### Next Step

- Use this as the local open-weight zero-shot LOAO baseline before Qwen fine-tuning.
- Keep the indexed candidate-label format for Qwen SFT/QLoRA.
- Prioritise absence calibration and all-row LOAO evaluation after any Qwen fine-tuning improvement.

### Validation

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git diff --check
rg -n "sk-[A-Za-z0-9_-]{12,}|AIza[0-9A-Za-z_-]{20,}|OPENAI_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|OPENAI_BASE_URL|Bearer [A-Za-z0-9._-]{20,}" . --glob '!outputs/**' --glob '!data/**' --glob '!models/**' --glob '!checkpoints/**' --glob '!artifacts/**' --glob '!runs/**' --glob '!.git/**'
```

| Check | Result |
| --- | --- |
| Unit tests | 68 tests OK |
| Compile check | passed |
| Diff whitespace check | passed, with CRLF conversion warnings only |
| Sensitive information scan | no real API key found; only existing placeholder endpoint/env-var documentation and script env-var names matched |

## 2026-07-01: Experiment Reproducibility Audit

### Purpose

- Check whether experiments from the start of the project have enough recorded parameters for future reproduction.
- Consolidate commands, model parameters, data splits, output paths, and provenance caveats into one index.
- Supplement older experiments that pre-date the stricter 2026-06-21 experiment-log template.

### Code Or Protocol Changes

- Added `docs/experiment_reproducibility_register.md`.
- Updated documentation entry points:
  - `README.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`

### Setup

- Scope: all main project experiment families, including data/split analysis, closed-topic baselines, held-out organisation, held-out aspect, LOAO, Qwen feasibility, Gemini hosted baselines, cascade experiments, and error analyses.
- Sources inspected:
  - committed documentation under `docs/`
  - script CLI arguments
  - local aggregate `summary.json` files under ignored `outputs/`
- Raw prediction outputs and review text were not copied into documentation.

### Commands

```powershell
rg --files docs scripts src tests thesis
rg -n "Command|Commands|Reproduction|python .\\scripts|learning rate|epochs|batch|threshold|output-dir|Result|Validation|LOAO|Gemini|Qwen|DistilBERT|SVM|TF-IDF" docs PROJECT_OVERVIEW.md README.md START_NEW_CHAT_PROMPT.md
Get-ChildItem -Path outputs -Recurse -File -Include summary.json,full_summary.json,*.csv
python .\scripts\run_classical_baselines.py --help
python .\scripts\run_transformer_baseline.py --help
python .\scripts\run_generalisation_baselines.py --help
python .\scripts\run_aspect_label_aware_baseline.py --help
python .\scripts\run_loao_heldout_aspect.py --help
python .\scripts\run_qwen_heldout_aspect_smoke.py --help
python .\scripts\run_gemini_heldout_aspect.py --help
```

### Outputs

- New committed documentation:
  - `docs/experiment_reproducibility_register.md`
- No generated outputs, data files, credentials, checkpoints, or prediction files were committed.

### Results

| Finding | Status |
| --- | --- |
| Post-2026-06-21 experiment-log entries | Strong, command-level record exists |
| Gemini/cascade experiments | Strong, with commands, token/cost diagnostics, and output paths |
| Strongest local non-LLM and LOAO experiments | Strong after latest documentation updates |
| Closed-topic DistilBERT tuning | Recoverable from summaries; now centralised in register |
| Qwen LoRA feasibility pilots | Recoverable from summaries; now centralised in register |
| Earliest exploratory TF-IDF/Qwen smoke runs | Adequate for non-headline exploratory status, but not all historical commands were logged verbatim |

### Interpretation

The dissertation-relevant results are reproducible enough for thesis use after this audit. The main weakness was not missing outputs, but lack of a single cross-experiment map for older runs. The new register fills that gap and records remaining caveats honestly.

### Next Step

For future experiments, add exact command, git commit, parsed CLI arguments, hardware/API metadata, and package versions to each run summary where feasible.

## 2026-07-01: Frozen LOAO Open-Topic Protocol

### Purpose

- Freeze the canonical leave-one-aspect-out protocol before the Qwen zero-shot LOAO and later Qwen fine-tuning branch.
- Prevent fixed three-aspect, all-row LOAO, positive-row LOAO, and hosted cascade results from being mixed as if they were the same benchmark.
- Define the metric hierarchy for open-topic robustness.

### Code Or Protocol Changes

- Updated `docs/evaluation_protocol.md` with protocol version `loao_open_topic_all_row_v1`.
- Updated `docs/loao_heldout_aspect.md` to point future LOAO work to the frozen protocol.

### Frozen Decisions

- Main open-topic robustness view: all-row LOAO over all 12 FABSA aspects.
- Evaluation rows: all official validation/test rows per held-out aspect.
- Evaluation labels: gold labels filtered to the current held-out aspect.
- Candidate set: exactly the current held-out aspect.
- Empty predictions: allowed.
- Primary robustness comparison metric: mean test `pair_micro_f1` across held-out aspects.
- Pair samples F1: still reported for continuity, but not used alone for all-row LOAO selection or interpretation.
- Primary future training strategy: `example_filtered`; `label_masked` remains an incomplete-label-noise ablation.
- Positive-row LOAO: sentiment diagnostic only, not open-topic robustness evidence.
- Gemini fixed/cascade results: hosted reference and selective-deployment evidence, not LOAO evidence unless a labelled LOAO diagnostic is run.

### Next Step

Use this frozen protocol to interpret the Qwen zero-shot LOAO run recorded below. Later Qwen LoRA/QLoRA fine-tuning should use the same protocol unless a new version is explicitly named.

## 2026-07-01: LLM Next Experiment Roadmap

### Purpose

- Record the next LLM-centred experiment direction after completing the strongest local DistilBERT LOAO robustness run and the fixed-split Gemini Pareto/cascade experiments.
- Avoid treating full Gemini LOAO as the default next step because the estimated all-row hosted cost and latency are high relative to the added dissertation value.
- Preserve a concrete ordered roadmap for future work and new-chat handoff.

### Decision

The next experiment roadmap is documented in `docs/llm_next_experiment_directions.md`.

Recommended order:

1. Candidate-aspect descriptions with Gemini Flash.
2. Sampled Gemini LOAO diagnostic, not full Gemini LOAO.
3. Cascade uncertainty improvement using local score/margin export and existing Gemini predictions.
4. Qualitative error taxonomy with manual review.
5. Qwen fine-tuning/evaluation once stronger GPU access is available.

### Rationale

- Candidate descriptions test whether richer label semantics improve candidate-label prediction.
- A sampled Gemini LOAO diagnostic gives a limited robustness signal without paying for full all-row hosted LOAO.
- Cascade uncertainty improvement strengthens the selective-deployment method without new Gemini calls.
- Qualitative error taxonomy turns the fixed-split and LOAO results into dissertation discussion evidence.
- Qwen fine-tuning remains the next major open/local LLM stage once compute is available.

### Documentation Updated

- `docs/llm_next_experiment_directions.md`
- `docs/gemini_candidate_label_baseline.md`
- `docs/tasks_1_to_3_thesis_prep.md`
- `PROJECT_OVERVIEW.md`
- `START_NEW_CHAT_PROMPT.md`
- `README.md`
| Secret scan over tracked code/docs paths | no real API key found |

## 2026-07-01: Strongest DistilBERT LOAO Robustness Run

### Purpose

- Complete the previously missing strongest local non-LLM LOAO check.
- Test whether the fixed three-aspect result for `candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment` remains strong when every FABSA aspect is rotated into the unseen position.
- Use the all-row LOAO view as the main robustness diagnostic, not the easier positive-row sentiment diagnostic.

### Setup

- Protocol: leave-one-aspect-out held-out-aspect evaluation over all 12 FABSA aspects.
- Evaluation scope: all official validation/test rows, with gold labels filtered to the current held-out aspect.
- Candidate set: only the current held-out aspect.
- Strategy: `example_filtered`.
- Aspect selector: DistilBERT cross-encoder over `(review text, candidate aspect)`.
- Sentiment model: DistilBERT aspect-conditioned classifier over `(review text, candidate aspect)`.
- Threshold selection: validation pair micro F1.
- Hardware used for the full run: RTX 5050 Laptop GPU.

### Commands

Tiny smoke test:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --heldout-aspect "Staff support: Email" --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 1 --sentiment-learning-rate 2e-5 --sentiment-batch-size 8 --sentiment-eval-batch-size 16 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 1 --batch-size 8 --eval-batch-size 16 --learning-rate 3e-5 --negatives-per-positive 1 --train-limit 80 --eval-limit 40 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_tiny_smoke_20260630
```

Preferred full LOAO run:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630
```

Selector LR tuning check:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701
```

### Outputs

- Local ignored outputs:
  - `outputs/baselines/loao_cross_encoder_transformer_sentiment_tiny_smoke_20260630/`
  - `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630/`
  - `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701/`
- The output directories contain aggregate CSV/JSON files and per-fold prediction diagnostics under `outputs/`, so they remain ignored and are not committed.
- No model checkpoints were saved.

### Results

Mean test spread across 12 held-out aspects:

| Variant | Pair Samples F1 Mean | Pair Micro F1 Mean | Pair Precision Mean | Pair Recall Mean | Pair Macro F1 Mean | FP Rows / 100 Mean | FN Rows / 100 Mean | Aspect Micro F1 Mean | Sentiment Accuracy When Gold Aspect Predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selector LR `3e-5` | 0.0550 | 0.3128 | 0.3345 | 0.4315 | 0.2285 | 12.7022 | 8.6746 | 0.7862 | 0.9319 |
| Selector LR `2e-5` | 0.0577 | 0.2941 | 0.3021 | 0.3921 | 0.2250 | 13.2798 | 8.3176 | 0.7840 | 0.9301 |

The LR `3e-5` setting is the preferred DistilBERT LOAO result because it has higher mean pair micro F1, precision, recall, pair macro F1, and sentiment accuracy. The LR `2e-5` run has slightly higher pair samples F1, but all-row LOAO pair samples F1 is not the preferred threshold-selection or diagnostic metric because it handles empty-gold true negatives poorly.

Per-aspect LR `3e-5` pair micro F1 ranged from `0.1011` for `Company brand: Reviews` to `0.5128` for `Staff support: Phone`. This confirms large aspect-to-aspect variation.

### Interpretation

- The fixed three-aspect non-LLM headline remains `0.6071` pair samples F1 for candidate-aspect DistilBERT plus DistilBERT aspect-conditioned sentiment.
- The full all-row LOAO robustness check does not reproduce that strength. Mean pair micro F1 is `0.3128` for the preferred DistilBERT LOAO setting.
- The result is below the documented lexical global-sentiment micro-selected all-row LOAO lower bound (`0.3780` mean pair micro F1 for `example_filtered`).
- This is not mainly a sentiment problem: sentiment accuracy when the gold aspect is predicted is high, around `0.93`.
- The bottleneck is unseen-aspect relevance detection and threshold calibration under taxonomy shift.
- The dissertation should therefore present the local DistilBERT pipeline as the strongest fixed-split non-LLM baseline, and LOAO as the robustness caveat that motivates LLM-assisted candidate-label reasoning and selective deployment.

### Next Step

- Do not spend more time on small DistilBERT LOAO hyperparameter tuning unless a specific thesis gap appears.
- Use this result to motivate the LLM-centred branch: candidate-aspect descriptions, qualitative error taxonomy, hosted Gemini/local cascade framing, and later Qwen fine-tuning/evaluation on stronger GPU access.

### Validation

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git diff --check -- PROJECT_OVERVIEW.md README.md START_NEW_CHAT_PROMPT.md docs\experiment_log.md docs\generalisation_baselines.md docs\loao_heldout_aspect.md docs\non_llm_open_topic_baseline.md
```

| Check | Result |
| --- | --- |
| Unit tests | 63 tests OK |
| Compile check | passed |
| Diff whitespace check | passed, with CRLF conversion warnings only |

## 2026-07-01: Tasks 1-3 Thesis Prep Consolidation

### Purpose

- Confirm that Gemini follow-up Tasks 1, 2, and 3 are complete and GitHub-synchronised.
- Consolidate the completed evidence into one clean handoff for dissertation writing and Task 4.
- Preserve the distinction between fixed three-aspect evidence, selective-deployment evidence, and future LOAO robustness evidence.

### Code Or Protocol Changes

- Added `docs/tasks_1_to_3_thesis_prep.md`.
- Updated `PROJECT_OVERVIEW.md`, `START_NEW_CHAT_PROMPT.md`, and `docs/gemini_candidate_label_baseline.md` to point to the new handoff note.
- No model code or experiment protocol changed.

### Consolidated Tasks

| Task | Status | Main Evidence |
| --- | --- | --- |
| 1. Full Gemini Pro fixed-split validation/test | complete | Pro test pair samples F1 `0.7141`, validation+test cost about `$2.9781` |
| 2. Full Gemini Flash-Lite fixed-split validation/test | complete | Flash-Lite test pair samples F1 `0.5516`, validation+test cost about `$0.0171` |
| 3. Local-to-Gemini uncertainty cascade | complete | local -> Pro cascade test pair samples F1 `0.8102`; Pro cascade beats pure Pro through fallback/error complementarity |

### Task 4 Starting Point

- Start from `docs/tasks_1_to_3_thesis_prep.md`.
- Test candidate-aspect descriptions without using validation/test review text or validation/test labels to create descriptions.
- Keep indexed candidate IDs, `response_format=json_schema`, `temperature=0`, and `max_tokens=2048`.
- Use validation to select the description/prompt variant, then evaluate once on test.
- Start with Gemini Flash for cost efficiency; use Pro only if the description effect is promising or diagnostically ambiguous.

### Validation

```powershell
git status --short --branch
git diff --check
rg -n "sk-[A-Za-z0-9_-]{12,}" . --glob '!outputs/**' --glob '!data/**' --glob '!models/**' --glob '!checkpoints/**' --glob '!.git/**'
```

| Check | Result |
| --- | --- |
| Git status before consolidation | clean and synced with `origin/main` |
| Diff whitespace check | passed, with CRLF conversion warnings only |
| Secret scan over tracked code/docs paths | no real API key found |
| Documentation-only update | ready to commit |

## 2026-07-01: Local-To-Gemini Pro Cascade Deep-Dive

### Purpose

- Explain why the validation-selected local -> Pro cascade beats the pure Gemini Pro fixed-split baseline.
- Separate global model strength from error complementarity.
- Produce dissertation-ready diagnostics for row-level wins, Pro abstentions, false-positive/false-negative shifts, and label-level gains.

### Code Or Protocol Changes

- Added `scripts/analyse_local_gemini_cascade.py`.
- The script aligns local, pure Pro, and selected cascade test predictions by row key.
- It reconstructs the validation-selected escalation set from the cascade policy, then writes aggregate diagnostics under ignored `outputs/`.
- Raw review text is not written to tracked documentation; generated examples contain only row ids, gold labels, predictions, and F1 values.

### Setup

- Dataset: FABSA fixed held-out-aspect test rows.
- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Local baseline: candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment, `example_filtered`.
- Hosted baseline: full fixed-split `vertex_ai/gemini-2.5-pro`.
- Cascade: validation-selected Pro policy from `outputs/analysis/local_gemini_cascade_pro_grid1/`.

### Command

```powershell
python .\scripts\analyse_local_gemini_cascade.py
```

### Outputs

- Local ignored output directory: `outputs/analysis/local_gemini_cascade_pro_deep_dive/`.
- Files written:
  - `summary.json`
  - `pair_label_comparison.csv`
  - `aspect_label_comparison.csv`
  - `sentiment_label_comparison.csv`
- Documentation updated in `docs/local_gemini_cascade.md`.

### Results

System-level comparison:

| System | Pair Samples F1 | Pair Micro F1 | TP | FP | FN | Empty Prediction Rows | Exact Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local only | 0.6071 | 0.5917 | 192 | 168 | 97 | 0 | 144 |
| Pure Pro | 0.7141 | 0.7425 | 222 | 87 | 67 | 28 | 167 |
| Local -> Pro cascade | 0.8102 | 0.7955 | 249 | 88 | 40 | 0 | 194 |

Row-level comparison:

- Cascade is better than pure Pro on `29` rows, equal on `250`, and worse on `2`.
- Pure Pro has `28` empty prediction rows; the cascade recovers all `28` with local predictions.
- On Pro-empty rows, pure Pro mean row sample F1 is `0.0000`, while local fallback/cascade mean row sample F1 is `0.6905`.
- The cascade reduces Pro pair false negatives from `67` to `40`, while false positives stay nearly flat (`87` to `88`).
- Exact-match rows increase from `167` for Pro to `194` for the cascade.

Largest label-level gains versus pure Pro:

| Label | Pro F1 | Cascade F1 | F1 Delta | FP Delta | FN Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Account management: Account access | neutral` | 0.7179 | 0.8696 | +0.1516 | +1 | -6 |
| `Value: Discounts promotions | neutral` | 0.6154 | 0.7500 | +0.1346 | +1 | -2 |
| `Company brand: Competitor | positive` | 0.6933 | 0.8263 | +0.1330 | 0 | -17 |
| `Company brand: Competitor | negative` | 0.7619 | 0.8358 | +0.0739 | 0 | -4 |

Aspect-level diagnosis:

- `Company brand: Competitor` improves from `0.7407` Pro F1 to `0.8487` cascade F1, mainly through `21` fewer false negatives.
- `Value: Discounts promotions` is unchanged versus Pro.
- `Account management: Account access` drops from `0.8432` Pro F1 to `0.8168` cascade F1 because the cascade adds `6` false positives, but this is outweighed by the larger recall gains elsewhere.

### Interpretation

- The cascade beats pure Pro because of error complementarity, not because the local model is stronger overall.
- Pure Pro is much stronger than local on average, but it sometimes abstains or misses held-out labels, especially competitor labels.
- The winning policy, `gemini_nonempty_else_local`, keeps Pro's semantic gains on most escalated rows and uses the local classifier as a deterministic safety net for Pro-empty or non-escalated rows.
- This is useful dissertation evidence because it turns the hosted LLM into a selective deployment component rather than a simple full replacement.
- The deep-dive uses test labels only after the validation-selected policy has been fixed, so it explains the headline result without selecting it.

### Next Step

- Treat task 3 as complete fixed-split selective-deployment evidence.
- Use this analysis in the dissertation results/discussion chapter.
- Keep LOAO robustness separate; this result should not be over-claimed as LOAO evidence.

### Validation

```powershell
python .\scripts\analyse_local_gemini_cascade.py
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git diff --check
rg -n "sk-[A-Za-z0-9_-]{12,}" . --glob '!outputs/**' --glob '!data/**' --glob '!models/**' --glob '!checkpoints/**' --glob '!.git/**'
```

| Check | Result |
| --- | --- |
| Deep-dive script | passed |
| Unit tests | 63 tests OK |
| Compile check | passed |
| Diff whitespace check | passed, with CRLF conversion warnings only |
| Secret scan over tracked code/docs paths | no real API key found |

## 2026-07-01: Local-To-Gemini Uncertainty Cascade

### Purpose

- Test the dissertation-relevant selective deployment pattern: local model first, hosted Gemini only for locally uncertain rows.
- Compare Flash-Lite, Flash, and Pro as escalators under the same fixed held-out-aspect protocol.
- Report F1, escalation rate, latency, token-cost estimates, and budgeted trade-offs without using test labels for policy selection.

### Code Or Protocol Changes

- Added `src/msc_project/evaluation/cascade.py` with row alignment, validation-derived reliability features, prediction-combination modes, and helper metrics.
- Added `scripts/run_local_gemini_cascade.py`.
- Added `tests/test_cascade_evaluation.py`.
- Updated the cascade sweep to use 1 percentage point ranked escalation steps.
- The historical local prediction files do not store calibrated aspect probabilities, so uncertainty is based on validation-derived local reliability proxies rather than score margins.

### Setup

- Dataset: FABSA fixed held-out-aspect split.
- Held-out aspects:
  - `Account management: Account access`
  - `Company brand: Competitor`
  - `Value: Discounts promotions`
- Local baseline: candidate-aspect DistilBERT selector + DistilBERT aspect-conditioned sentiment, `example_filtered`, selector LR `3e-5`.
- Gemini escalators:
  - `vertex_ai/gemini-2.5-flash-lite`
  - `vertex_ai/gemini-2.5-flash`
  - `vertex_ai/gemini-2.5-pro`
- Policy search: `10,578` candidate policies per escalator, selected on validation pair samples F1 with pair micro/macro F1 tie-breakers.
- Combination modes included `replace`, `gemini_nonempty_else_local`, `union`, `intersection`, `agreement_or_gemini`, and `agreement_or_local`.

### Commands

```powershell
python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_034545_flash_lite_fixed_full --input-cost-per-1m 0.10 --output-cost-per-1m 0.40 --output-dir .\outputs\analysis\local_gemini_cascade_flash_lite_grid1 --rank-rate-step 1

python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full --input-cost-per-1m 0.30 --output-cost-per-1m 2.50 --output-dir .\outputs\analysis\local_gemini_cascade_flash_grid1 --rank-rate-step 1

python .\scripts\run_local_gemini_cascade.py --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_031040_pro_fixed_full --input-cost-per-1m 1.25 --output-cost-per-1m 10.00 --output-dir .\outputs\analysis\local_gemini_cascade_pro_grid1 --rank-rate-step 1
```

### Outputs

- Local ignored analysis outputs:
  - `outputs/analysis/local_gemini_cascade_flash_lite_grid1/`
  - `outputs/analysis/local_gemini_cascade_flash_grid1/`
  - `outputs/analysis/local_gemini_cascade_pro_grid1/`
- Committed documentation:
  - `docs/local_gemini_cascade.md`
  - updated Gemini/generalisation/project handoff docs.
- Generated outputs remain ignored because selected prediction files can contain review text.

### Results

Validation-selected test results:

| Escalator | Policy Summary | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Aspect Samples F1 | Calls | Call Rate | Test Gemini Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None, local only | n/a | 0.6071 | 0.5917 | 0.4890 | 0.6651 | 0 | 0.0000 | $0.0000 |
| Flash-Lite | weighted pair+sentiment reliability, 51%, Gemini non-empty else local | 0.6679 | 0.6579 | 0.5401 | 0.7259 | 143 | 0.5089 | $0.0052 |
| Flash | low minimum pair precision, 90%, Gemini non-empty else local | 0.7459 | 0.7348 | 0.6223 | 0.7993 | 253 | 0.9004 | $0.2789 |
| Pro | low minimum pair precision, 90%, Gemini non-empty else local | 0.8102 | 0.7955 | 0.6809 | 0.8493 | 253 | 0.9004 | $1.5421 |

Selected budget diagnostics:

| Escalator | Budget | Pair Samples F1 | Pair Micro F1 | Pair Macro F1 | Calls | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Lite | 20% | 0.6511 | 0.6365 | 0.5155 | 48 | $0.0019 |
| Flash-Lite | 50% | 0.6684 | 0.6547 | 0.5371 | 118 | $0.0044 |
| Flash | 20% | 0.6636 | 0.6507 | 0.5399 | 56 | $0.0760 |
| Flash | 50% | 0.7163 | 0.7038 | 0.5910 | 135 | $0.1632 |
| Flash | 80% | 0.7495 | 0.7384 | 0.6175 | 222 | $0.2438 |
| Pro | 20% | 0.6795 | 0.6646 | 0.5517 | 51 | $0.3108 |
| Pro | 50% | 0.7515 | 0.7368 | 0.6314 | 140 | $0.8390 |
| Pro | 80% | 0.8149 | 0.8013 | 0.6760 | 222 | $1.3461 |

The 80% Pro budget row has the highest observed test pair samples F1, but it is recorded only as a budget diagnostic. The headline remains the validation-selected Pro cascade at `0.8102`.

### Interpretation

- The cascade is the strongest fixed held-out-aspect system result so far.
- Flash-Lite is a very cheap escalator and improves local-only performance, but it is not the best quality point.
- Flash is the practical selective-deployment point: large F1 gain over local-only at much lower cost and latency than Pro.
- Pro is the upper hosted-quality point and reaches the best fixed-split F1.
- This remains fixed three-aspect evidence, not LOAO robustness evidence.
- A future local rerun could export DistilBERT selector probabilities and repeat the cascade with calibrated score/margin uncertainty features.

### Next Step

- Do not run full Gemini LOAO by default.
- Next dissertation-value experiments should be candidate-aspect descriptions and qualitative error taxonomy.

### Validation

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

| Check | Result |
| --- | --- |
| Unit tests | 63 tests OK |
| Compile check | passed |

## 2026-07-02: Thesis Completion Roadmap And Qwen Full LOAO Boundary

### Purpose

- Record the project-level decision that full fine-tuned Qwen LoRA LOAO remains necessary for the strongest open-weight LLM story, but should be treated as the only major compute-bound unfinished experiment.
- Prevent the project from drifting into more Gemini prompt sweeps, full Gemini LOAO, or minor DistilBERT tuning while GPU access for Qwen is being negotiated.
- Consolidate all remaining non-major work needed for the dissertation narrative.

### Code Or Protocol Changes

- Added `docs/thesis_completion_roadmap.md`.
- Registered the existing qualitative error-taxonomy pre-registration in `docs/qualitative_error_taxonomy.md`.
- Updated `PROJECT_OVERVIEW.md`, `START_NEW_CHAT_PROMPT.md`, `docs/llm_next_experiment_directions.md`, and `docs/experiment_reproducibility_register.md`.
- No model code, dataset split, metric definition, or evaluation protocol changed.

### Setup

- Current evidence base:
  - closed-topic and held-out organisation baselines complete;
  - fixed held-out-aspect local, Qwen, Gemini, and cascade evidence complete;
  - lexical, DistilBERT, and Qwen zero-shot all-row LOAO evidence complete;
  - Gemini aspect-description ablation complete.
- Pending major experiment:
  - full fine-tuned Qwen LoRA all-row LOAO over all 12 FABSA aspects.

### Commands

```powershell
git status --short --branch
git log --oneline --decorate -5
```

No model-training or API command was run for this planning update.

### Outputs

- Tracked planning and documentation files:
  - `docs/thesis_completion_roadmap.md`
  - `docs/qualitative_error_taxonomy.md`
  - `report_notes.md`
  - `PROJECT_OVERVIEW.md`
  - `START_NEW_CHAT_PROMPT.md`
  - `docs/llm_next_experiment_directions.md`
  - `docs/experiment_reproducibility_register.md`
- No generated predictions, raw review text packets, checkpoints, credentials, or `outputs/` files were added.

### Results

The active next-work order is now:

1. Finish qualitative error taxonomy using existing local ignored outputs.
2. Build thesis-ready result tables and figure data.
3. Try cascade score/margin uncertainty without new Gemini calls.
4. Prepare and smoke-test the final Qwen LoRA SFT/evaluation runner.
5. Refresh the LaTeX thesis skeleton and draft from frozen evidence.
6. Run full fine-tuned Qwen LoRA LOAO only after GPU access and resume/manifest behaviour are ready.

### Interpretation

The thesis narrative should keep LOAO as the open-topic robustness spine. Fixed held-out-aspect Gemini and cascade results are valuable selective-deployment evidence, but they do not replace LOAO. Qwen zero-shot LOAO diagnoses the open-weight LLM failure mode: strong positive-gold recognition but weak empty-gold absence calibration. Full Qwen LoRA LOAO is therefore a valuable final robustness closure, but the dissertation can and should complete all non-compute-bound analysis first.

### Next Step

Task 5 is now complete. Move next to thesis-ready result tables/figure data, cascade score or margin uncertainty, and Qwen LoRA runner readiness before any full Qwen LoRA LOAO run.

## 2026-07-02: Pre-Qwen LoRA Full LOAO Checklist Audit

### Purpose

- Re-check the current completed evidence and Qwen LoRA readiness state.
- Convert the remaining pre-full-LOAO work into a tickable checklist that can be updated as each item is completed.
- Make the next action unambiguous before starting new implementation work.

### Code Or Protocol Changes

- Updated `docs/thesis_completion_roadmap.md` with a `Pre-Qwen LoRA Full LOAO Checklist` section using Markdown task-list syntax.
- Updated `report_notes.md` with the audit summary.
- No model code, data split, metric definition, or experiment output changed.

### Current Completed Work

- Closed-topic FABSA baselines: complete.
- Held-out organisation baselines: complete.
- Strongest local fixed held-out-aspect baseline: complete.
- Lexical and DistilBERT full all-row LOAO baselines: complete.
- Qwen indexed zero-shot fixed and full all-row LOAO: complete.
- Gemini fixed Pareto, local-to-Gemini cascade, Gemini aspect descriptions: complete.
- Gemini-assisted qualitative error taxonomy with manual consolidation: complete.
- LaTeX thesis skeleton refresh: complete.

### Remaining Checklist Before Full Qwen LoRA LOAO

- Thesis-ready result tables and figure data.
- Cascade score/margin uncertainty improvement.
- Final Qwen held-out-aspect LoRA SFT runner with manifest logging.
- Resume/skip behaviour and focused runner tests.
- Tiny local Qwen LoRA held-out-aspect smoke test.
- Optional fixed held-out-aspect Qwen LoRA configuration before full LOAO.
- Full 12-fold LOAO command templates, output naming, recovery plan, and GPU environment confirmation.

### Next Step

Start with thesis-ready result tables and figure data. This is the highest-value non-GPU task and should be finished before implementing the final Qwen LoRA runner.

## 2026-07-02: Thesis-Ready Result Tables And Figure Data

### Purpose

- Convert completed experiment evidence into thesis-ready aggregate tables and plot-ready CSV inputs.
- Keep closed-topic, held-out organisation, fixed held-out aspect, all-row LOAO, positive-gold diagnostics, and Gemini/cascade deployment evidence conceptually separate.

### Code Or Protocol Changes

- Added `scripts/build_thesis_result_tables.py`.
- Generated aggregate-only tracked outputs under `docs/`.
- No model code, split definition, metric implementation, API call, or training protocol changed.

### Setup

- Data sources: existing tracked documentation and already documented aggregate metrics.
- Raw local outputs under `outputs/` were not committed and no raw review text was copied into tracked files.
- The table rows intentionally separate fixed held-out-aspect results from all-row LOAO robustness rows.

### Commands

```powershell
python .\scripts\build_thesis_result_tables.py
```

### Outputs

- Tracked summary:
  - `docs/thesis_result_tables.md`
- Tracked figure-ready aggregate CSVs:
  - `docs/thesis_figure_data/protocol_ladder.csv`
  - `docs/thesis_figure_data/loao_robustness.csv`
  - `docs/thesis_figure_data/qwen_positive_diagnostic.csv`
  - `docs/thesis_figure_data/cascade_tradeoff.csv`
  - `docs/thesis_figure_data/fixed_vs_loao_drop.csv`

### Results

| Artifact | Result |
| --- | --- |
| Thesis evidence map | Written |
| Protocol ladder table | Written |
| All-row LOAO robustness table | Written |
| Qwen positive-gold diagnostic table | Written |
| Gemini/cascade trade-off table | Written |
| Fixed split versus LOAO drop figure data | Written |

### Interpretation

- The dissertation can now cite one tracked aggregate table pack without flattening incompatible protocols into a single leaderboard.
- The fixed held-out-aspect and cascade rows remain useful deployment evidence, while all-row LOAO remains the robustness evidence.
- The positive-gold diagnostic is preserved as a Qwen absence-calibration explanation, not as an open-topic headline score.

### Limitations

- The script materialises the currently documented headline metrics rather than regenerating all upstream experiments.
- If a future Qwen LoRA fixed or LOAO result is completed, the table constants should be updated and regenerated.

### Next Step

- Continue to the cascade score/margin uncertainty check without new Gemini calls.

## 2026-07-02: Cascade Score/Margin Uncertainty Check

### Purpose

- Test whether local candidate-aspect score and margin features improve the local-to-Gemini fixed held-out-aspect cascade.
- Avoid any new Gemini API calls by reusing cached Gemini fixed-split prediction files.

### Code Or Protocol Changes

- Updated `scripts/run_aspect_label_aware_baseline.py` so prediction JSONL exports include local selector `score_features`.
- Updated `src/msc_project/evaluation/cascade.py` so cascade features include score/margin uncertainty when present.
- Updated `scripts/run_local_gemini_cascade.py` so score/margin features are included in the policy candidate search.
- Added focused cascade tests for score feature ingestion.

### Setup

- Local rerun: strongest fixed held-out-aspect baseline configuration, `example_filtered`.
- Model: candidate-aspect DistilBERT selector plus DistilBERT aspect-conditioned sentiment.
- Hardware: local NVIDIA GeForce RTX 5050 Laptop GPU, CUDA available through PyTorch `2.10.0+cu128`.
- Gemini predictions reused from completed Flash-Lite, Flash, and Pro fixed-split runs.
- No Gemini API calls were made.

### Commands

```powershell
python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702

python .\scripts\run_local_gemini_cascade.py --local-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702\example_filtered --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_034545_flash_lite_fixed_full --input-cost-per-1m 0.10 --output-cost-per-1m 0.40 --output-dir .\outputs\analysis\local_gemini_cascade_flash_lite_score_margin_20260702 --rank-rate-step 1
python .\scripts\run_local_gemini_cascade.py --local-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702\example_filtered --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full --input-cost-per-1m 0.30 --output-cost-per-1m 2.50 --output-dir .\outputs\analysis\local_gemini_cascade_flash_score_margin_20260702 --rank-rate-step 1
python .\scripts\run_local_gemini_cascade.py --local-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702\example_filtered --gemini-dir .\outputs\llm\gemini_candidate_label_20260701_031040_pro_fixed_full --input-cost-per-1m 1.25 --output-cost-per-1m 10.00 --output-dir .\outputs\analysis\local_gemini_cascade_pro_score_margin_20260702 --rank-rate-step 1
```

### Outputs

- Local ignored local rerun:
  - `outputs/baselines/aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered_score_export_20260702/`
- Local ignored cascade outputs:
  - `outputs/analysis/local_gemini_cascade_flash_lite_score_margin_20260702/`
  - `outputs/analysis/local_gemini_cascade_flash_score_margin_20260702/`
  - `outputs/analysis/local_gemini_cascade_pro_score_margin_20260702/`
- Tracked documentation:
  - `docs/local_gemini_cascade.md`

### Results

The local score-export rerun reproduced the documented strongest local fixed held-out-aspect baseline.

| Local Metric | Value |
| --- | ---: |
| Validation pair samples F1 | 0.6226 |
| Validation pair micro F1 | 0.6049 |
| Validation-selected threshold | 0.37 |
| Test pair samples F1 | 0.6071 |
| Test pair micro F1 | 0.5917 |
| Test pair macro F1 | 0.4890 |

Score/margin features increased the cascade candidate policy count from `10,578` to `20,598`, but validation selection kept the original reliability-proxy-style policies.

| Escalator | Test Pair Samples F1 | Test Pair Micro F1 | Call Rate | Selected Policy |
| --- | ---: | ---: | ---: | --- |
| Flash-Lite | 0.6679 | 0.6579 | 0.5089 | weighted pair+sentiment reliability, 51%, Gemini non-empty else local |
| Flash | 0.7459 | 0.7348 | 0.9004 | low minimum pair precision, 90%, Gemini non-empty else local |
| Pro | 0.8102 | 0.7955 | 0.9004 | low minimum pair precision, 90%, Gemini non-empty else local |

### Interpretation

- The score/margin export works and is now available for future local/cascade experiments.
- On the current fixed held-out-aspect cascade, score/margin features did not improve validation-selected F1 or reduce the Flash/Pro call rate.
- This should be reported as a negative methodological check, not as a new headline cascade.

### Next Step

- Proceed to the final Qwen held-out-aspect LoRA SFT runner with manifest logging and resume/skip behaviour.

## 2026-07-02: Final Qwen Held-Out-Aspect LoRA Runner Readiness

### Purpose

- Replace the closed-topic Qwen LoRA pilot with a final held-out-aspect candidate-label SFT runner.
- Make the runner executable for the fixed held-out-aspect adaptation check and reusable for later one-aspect LOAO fold directories.
- Add manifest logging, adapter naming, prediction resume/skip behaviour, and focused tests before any long GPU run.

### Code Or Protocol Changes

- Added `scripts/run_qwen_lora_heldout_aspect.py`.
- Extended `scripts/prepare_qwen_heldout_aspect_sft_data.py` with `--heldout-aspect` so one-aspect LOAO fold JSONL directories can be prepared without changing the default fixed three-aspect behaviour.
- Added `tests/test_qwen_lora_heldout_runner.py`.

### Setup

- Input protocol: indexed held-out-aspect SFT JSONL with train/validation/test split files and `metadata.json`.
- Model target: `Qwen/Qwen3-4B-Instruct-2507`.
- Loading target: 4-bit QLoRA, bitsandbytes NF4 double quantisation.
- Local hardware detected in dry-run manifest: NVIDIA GeForce RTX 5050 Laptop GPU, PyTorch `2.10.0+cu128`, CUDA available.

### Commands

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --limit 24 --output-dir .\outputs\qwen_heldout_aspect_sft_tiny_20260702

python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_tiny_20260702 --strategy example_filtered --output-dir .\outputs\qwen_lora_heldout_aspect_dry_run_20260702 --train-limit 8 --validation-limit 4 --test-limit 4 --epochs 1 --max-train-steps 1 --batch-size 1 --grad-accumulation-steps 1 --learning-rate 1e-4 --max-length 512 --max-input-tokens 512 --max-new-tokens 64 --dry-run

python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --heldout-aspect "Staff support: Email" --limit 3 --output-dir .\outputs\qwen_loao_sft_single_fold_check_20260702

python -m unittest tests.test_qwen_lora_heldout_runner tests.test_qwen_loao_runner tests.test_qwen_format
python -m compileall -q .\scripts\run_qwen_lora_heldout_aspect.py .\tests\test_qwen_lora_heldout_runner.py
python -m compileall -q .\scripts\prepare_qwen_heldout_aspect_sft_data.py
```

### Outputs

- Tracked code/tests:
  - `scripts/run_qwen_lora_heldout_aspect.py`
  - `scripts/prepare_qwen_heldout_aspect_sft_data.py`
  - `tests/test_qwen_lora_heldout_runner.py`
- Local ignored dry-run outputs:
  - `outputs/qwen_heldout_aspect_sft_tiny_20260702/`
  - `outputs/qwen_lora_heldout_aspect_dry_run_20260702/`
  - `outputs/qwen_loao_sft_single_fold_check_20260702/`

### Results

- The new runner reads indexed held-out-aspect SFT JSONL and validates required row fields.
- Loss masking uses the prompt-only chat text with a generation prompt, then masks prompt tokens in the full prompt+assistant answer sequence.
- The runner writes `manifest.json` before loading the model and updates `summary.json` during evaluation.
- Manifest fields include exact command, cwd, git commit, model name, quantisation/loading, LoRA parameters, prompt variant, split protocol, row scope, seed, runtime fields, hardware, output directory, and package versions.
- Resume/recovery support includes:
  - `--resume-from-adapter` for adapter-weight continuation;
  - `--skip-training-if-adapter-exists` for explicit adapter reuse;
  - stable final adapter path `adapter_final`;
  - optional per-epoch adapter checkpoints;
  - validation/test prediction JSONL resume with row-index/id prefix validation;
  - complete prediction skip when `--skip-existing-predictions` is enabled.
- Training resume limitation is explicitly recorded in the manifest: optimiser and scheduler state are not restored.
- Focused tests passed: `18` tests across the new runner, existing Qwen LOAO runner, and Qwen formatting utilities.
- A single-fold SFT data preparation check for `Staff support: Email` produced candidate lists with that aspect held out for validation/test, confirming the runner can consume later LOAO fold directories.

### Interpretation

- The final Qwen held-out-aspect LoRA runner, manifest logging, resume/skip behaviour, and focused tests are now ready for a tiny local smoke test.
- This does not yet prove model loading, LoRA training, adapter saving, generation, or metrics under actual Qwen execution; those are the next smoke-test step.

### Limitations

- The dry-run does not load Qwen or run training.
- Adapter resume restores LoRA weights only; it does not restore optimiser/scheduler state.
- Full 12-fold Qwen LoRA LOAO remains deliberately unrun.

### Next Step

- Run the tiny local Qwen LoRA held-out-aspect smoke test on a few train/validation/test rows.

## 2026-07-02: Tiny Local Qwen LoRA Held-Out-Aspect Smoke Test

### Purpose

- Verify the final held-out-aspect Qwen LoRA runner with actual local model loading, one tiny training step, adapter saving, generation, JSON parsing, candidate-ID mapping, and metrics.
- Keep the run intentionally tiny and local. This is a pipeline smoke test, not a model-quality experiment.

### Code Or Protocol Changes

- Updated `scripts/run_qwen_lora_heldout_aspect.py` so inference after training explicitly switches to evaluation mode, disables gradient checkpointing for generation where available, restores `use_cache`, and wraps generation in `torch.no_grad()`.
- This fixed an early smoke diagnostic where generation remained in the training/checkpointing path and produced invalid repeated-token JSON.

### Setup

- SFT data: `outputs/qwen_heldout_aspect_sft_tiny_20260702/example_filtered/`.
- Row scope: 4 train rows, 1 validation row, 1 test row.
- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Loading: 4-bit bitsandbytes NF4 double quantisation.
- LoRA: rank `8`, alpha `16`, dropout `0.05`, standard Qwen projection/MLP target modules.
- Hardware: local NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM.
- The successful canonical smoke disabled gradient checkpointing for this tiny local run to keep generation stable on the laptop GPU.

### Command

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_tiny_20260702 --strategy example_filtered --output-dir .\outputs\qwen_lora_heldout_aspect_tiny_smoke_evalmode_20260702 --train-limit 4 --validation-limit 1 --test-limit 1 --epochs 1 --max-train-steps 1 --batch-size 1 --grad-accumulation-steps 1 --learning-rate 1e-6 --max-length 512 --max-input-tokens 512 --max-new-tokens 96 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions
```

### Outputs

- Local ignored output directory:
  - `outputs/qwen_lora_heldout_aspect_tiny_smoke_evalmode_20260702/`
- Key local files:
  - `manifest.json`
  - `summary.json`
  - `adapter_final/adapter_model.safetensors`
  - `predictions/validation_predictions.jsonl`
  - `predictions/test_predictions.jsonl`

### Results

| Check | Result |
| --- | --- |
| Qwen model loaded | Passed |
| 4-bit QLoRA adapter attached | Passed |
| One training step completed | Passed: `global_step=1`, train loss `1.2265` |
| Adapter saved | Passed: `adapter_final/adapter_model.safetensors` written under ignored `outputs/` |
| Validation generation | Passed: 1/1 example |
| Test generation | Passed: 1/1 example |
| JSON parsing | Passed: validation/test valid JSON rate `1.0000` |
| Schema/candidate ID mapping | Passed: validation/test schema valid rate `1.0000`; validation `A1` mapped to `Account management: Account access` |
| Metrics writing | Passed: `summary.json` contains validation and test metric blocks |
| Manifest writing | Passed: command, cwd, git commit, model, quantisation, LoRA parameters, split protocol, row scope, runtime, hardware, packages, and output directory recorded |

Observed tiny metrics:

| Split | Examples | Pair Samples F1 | Pair Micro F1 | Valid JSON Rate | Schema Valid Rate | Seconds/Example |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.6617 |
| test | 1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 6.4489 |

### Interpretation

- The final runner is operational on the local laptop GPU for a tiny held-out-aspect QLoRA train/generate cycle.
- The tiny metrics must not be used as evidence of model quality because they use only one validation and one test row.
- The successful smoke mainly de-risks data loading, loss masking, LoRA attachment, adapter output, manifesting, parsing, candidate-ID mapping, and metric writing.

### Limitations

- The smoke uses a deliberately tiny row count and very low learning rate (`1e-6`) to avoid destabilising generation after a single update.
- The run does not establish a fixed held-out-aspect Qwen LoRA result and does not run any LOAO fold.
- Full 12-fold Qwen LoRA LOAO remains unrun by design.

### Next Step

- Prepare the fixed held-out-aspect Qwen LoRA command/config decision and the full 12-fold LOAO launch plan without starting a long run.

## 2026-07-02: Qwen LoRA Fixed-Split Decision And Full LOAO Launch Plan

### Purpose

- Prepare the optional fixed held-out-aspect Qwen LoRA configuration and the full 12-fold Qwen LoRA LOAO launch plan.
- Avoid starting a long GPU run without explicit confirmation.

### Code Or Protocol Changes

- Added `docs/qwen_lora_loao_launch_plan.md`.
- Fixed `scripts/run_qwen_lora_heldout_aspect.py` recovery semantics so `--skip-training-if-adapter-exists` auto-loads the existing `adapter_final` when no explicit `--resume-from-adapter` is supplied.
- Added a focused unit test for this adapter resume behaviour.

### Commands

No fixed full Qwen LoRA run and no full LOAO run were launched.

Validation commands run for the code change:

```powershell
python -m unittest tests.test_qwen_lora_heldout_runner
python -m compileall -q .\scripts\run_qwen_lora_heldout_aspect.py .\tests\test_qwen_lora_heldout_runner.py
```

### Outputs

- `docs/qwen_lora_loao_launch_plan.md`
- Updated `scripts/run_qwen_lora_heldout_aspect.py`
- Updated `tests/test_qwen_lora_heldout_runner.py`

### Results

- The optional fixed held-out-aspect Qwen LoRA configuration is documented with separate validation and test commands.
- The full 12-fold LOAO plan documents:
  - fold IDs for all 12 FABSA aspects;
  - one-aspect SFT data-preparation command template;
  - validation-first command template;
  - test-after-validation command template;
  - output directory pattern;
  - adapter checkpoint naming;
  - prediction resume/recovery behaviour;
  - adapter recovery behaviour;
  - runtime/storage assumptions;
  - data-transfer rules;
  - pre-launch validation/safety checks.
- The adapter skip/resume test now passes as part of `tests.test_qwen_lora_heldout_runner`.

### Interpretation

- The full Qwen LoRA LOAO launch plan is ready as a command/template document.
- The optional fixed held-out-aspect Qwen LoRA run was not started because full validation/test generation on the local 8 GB GPU is expected to exceed two hours and should be explicitly confirmed first.
- The target full-LOAO GPU environment is still not confirmed; the launch plan records assumptions rather than a completed launch gate.

### Limitations

- Runtime and storage estimates are based on local smoke and earlier zero-shot Qwen timings, not a completed full fine-tuned fold.
- Full 12-fold Qwen LoRA LOAO remains unrun by design.

### Next Step

- Update overview/handoff docs, run the final repository validation and safety checks, then commit only safe tracked code/docs/tests.

## 2026-07-02: Planned Fixed Held-Out-Aspect Qwen LoRA Run

### Purpose

- Run the optional fixed held-out-aspect Qwen LoRA configuration now that a long local GPU run has been approved.
- Test whether task-specific QLoRA adaptation improves the fixed three-aspect candidate-label setting before any full 12-fold LOAO launch.
- Keep this separate from full LOAO robustness evidence.

### Pre-Flight State

- Git state before planning: clean worktree, `main...origin/main [ahead 2]`.
- Remote fetched with `git fetch --prune`.
- Current commit before planning: `0e2cd3f llm: prepare qwen lora heldout runner`.
- Local CUDA available: NVIDIA GeForce RTX 5050 Laptop GPU, about 8 GB VRAM.
- PyTorch: `2.10.0+cu128`.

### Input Scope

- SFT data directory: `outputs/qwen_heldout_aspect_sft_indexed/example_filtered/`.
- Split protocol: fixed held-out-aspect, `example_filtered`, held-out labels only.
- Held-out aspects:
  - `Account management: Account access`;
  - `Company brand: Competitor`;
  - `Value: Discounts promotions`.
- Row counts:
  - train: `6,495`;
  - validation: `212`;
  - test: `281`.

### Planned Configuration

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Loading: 4-bit bitsandbytes NF4 double quantisation.
- LoRA: rank `8`, alpha `16`, dropout `0.05`.
- Epochs: `1`.
- Batch size: `1`.
- Gradient accumulation: `8`.
- Learning rate: `1e-5`.
- Weight decay: `0.0`.
- Warmup ratio: `0.05`.
- Training max length: `512`.
- Evaluation max input tokens: `1024`.
- Evaluation max new tokens: `192`.
- Seed: default runner seed `13`.
- Training progress log interval: `50` optimiser updates.
- Gradient checkpointing: disabled for this first long local run because the successful tiny smoke used the no-checkpointing path and produced valid JSON/schema output. If this OOMs, retry with `--gradient-checkpointing` and record the failure.

### Planned Commands

Validation-first training/evaluation:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Test evaluation only if validation succeeds and JSON/schema diagnostics are acceptable:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_20260702 --eval-split test --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --skip-training-if-adapter-exists --resume-predictions --skip-existing-predictions
```

### Output Scope

- Local ignored output directory:
  - `outputs/llm/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_20260702/`
- Local ignored logs:
  - `outputs/logs/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_20260702_validation.*.log`
  - `outputs/logs/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_20260702_test.*.log`
- Only aggregate metrics and documentation updates will be committed.
- Adapter/checkpoint files and raw prediction JSONL remain ignored under `outputs/`.

### Validation And Test Rule

- Use validation to decide whether the run is sane enough for test evaluation.
- Do not tune hyperparameters using test labels.
- Stop before test if validation JSON/schema validity collapses, if the run OOMs without a stable fallback, or if the adapter/prediction files are incomplete.

### Runtime And Risk

- This is expected to exceed two hours on the local RTX 5050 Laptop GPU.
- Main risks: local 8 GB VRAM pressure, long generation time, degenerate JSON after fine-tuning, or an interrupted long run.
- Resume plan: rerun the same command with `--resume-predictions --skip-existing-predictions`; for test, reuse `adapter_final` through `--skip-training-if-adapter-exists`.

### First Validation Attempt Result

The planned `1e-5` run was started and stopped before validation generation because training loss became non-finite:

| Step | Mean Loss So Far |
| ---: | ---: |
| 1 | 1.0447 |
| 50 | 0.6422 |
| 100 | 0.4185 |
| 150 | 0.3105 |
| 200 | NaN |
| 250 | NaN |

Decision:

- Stop the process rather than produce a likely invalid adapter.
- Initial interpretation was potential LR instability, but a tokenizer audit found the concrete cause: with `max_length=512`, `10` of `6,495` training rows had the assistant answer fully truncated, leaving no supervised labels and causing NaN loss.
- Add fail-fast protection to `scripts/run_qwen_lora_heldout_aspect.py` so future non-finite training losses raise an error immediately.
- Update `ChatSftDataset` to skip rows with fully truncated assistant answers and log the skipped count.

Tokenizer audit:

| Max Length | Used Rows | Skipped Fully Truncated Answer Rows |
| ---: | ---: | ---: |
| 512 | 6,485 | 10 |
| 768 | 6,493 | 2 |
| 1024 | 6,495 | 0 |

Because only `10` rows are skipped at `max_length=512`, keep the original memory-safe `512` training length and rerun the planned `lr=1e-5` configuration with the skip fix.

Skip-fix validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

### Skip-Fix Validation Result

- Output directory: `outputs/llm/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702/` (ignored).
- Log files:
  - `outputs/logs/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702_validation.out.log`
  - `outputs/logs/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702_validation.err.log`
- Training rows:
  - original: `6,495`;
  - skipped because assistant answer was fully truncated at `max_length=512`: `10`;
  - used: `6,485`.
- Optimiser steps: `811`.
- Final train loss: `0.1147`.
- Training runtime: `5,590.9` seconds.
- Full validation runtime including generation: `6,321.4` seconds.
- Validation generated rows: `212 / 212`.

Validation metrics recomputed from `validation_predictions.jsonl`:

| Metric | Value |
| --- | ---: |
| pair samples F1 | 0.5991 |
| pair micro F1 | 0.5977 |
| pair precision | 0.5991 |
| pair recall | 0.5963 |
| pair macro F1 | 0.4335 |
| pair exact match | 0.5802 |
| aspect samples F1 | 0.6635 |
| aspect micro F1 | 0.6621 |
| sentiment accuracy when gold aspect predicted | 0.9028 |
| valid JSON rate | 1.0000 |
| schema-valid rate | 1.0000 |
| seconds per example | 3.3370 |

Decision:

- Validation was stable enough to run the held-out test split.
- No hyperparameter was selected from the test set; the test command reused the validation-approved adapter and configuration.

### Fixed Held-Out-Aspect Test Result

Test command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702 --eval-split test --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --skip-training-if-adapter-exists --resume-predictions --skip-existing-predictions
```

Observed behaviour:

- The runner loaded `adapter_final` and skipped training.
- Test generated rows: `281 / 281`.
- Test runtime: `811.5` seconds overall, with `794.5` seconds spent in generation/evaluation.
- Test log files:
  - `outputs/logs/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702_test.out.log`
  - `outputs/logs/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702_test.err.log`

Test metrics recomputed from `test_predictions.jsonl`:

| Metric | Value |
| --- | ---: |
| pair samples F1 | 0.5528 |
| pair micro F1 | 0.5552 |
| pair precision | 0.5533 |
| pair recall | 0.5571 |
| pair macro F1 | 0.4393 |
| pair exact match | 0.5196 |
| aspect samples F1 | 0.6192 |
| aspect micro F1 | 0.6218 |
| sentiment accuracy when gold aspect predicted | 0.8944 |
| valid JSON rate | 1.0000 |
| schema-valid rate | 0.9964 |
| schema invalid count | 1 |
| conflicting sentiment count | 1 |
| seconds per example | 2.8275 |

### Interpretation

- This is a completed fixed held-out-aspect Qwen LoRA result, not a full LOAO result.
- QLoRA improves over Qwen indexed zero-shot on the fixed split:
  - pair samples F1: `0.5374` to `0.5528`;
  - pair micro F1: `0.5300` to `0.5552`;
  - pair macro F1: `0.4374` to `0.4393`.
- The gain is modest and does not exceed the strongest local non-LLM fixed held-out-aspect baseline (`0.6071` pair samples F1, `0.5917` pair micro F1) or the stronger Gemini/cascade fixed-split results.
- The main thesis value is therefore not a new headline score. It is evidence that:
  - the final indexed held-out-aspect QLoRA runner can train beyond smoke scale on the local GPU;
  - adapter saving/reuse works for validation-first then test evaluation;
  - structured JSON output remains reliable after fine-tuning;
  - fixed-split Qwen adaptation helps a little but still needs LOAO evaluation before any robustness claim.
- Runner reproducibility was tightened after the run:
  - rows with fully truncated assistant answers are skipped before training;
  - non-finite training loss now raises immediately;
  - progress logging records optimiser-step loss trends;
  - repeated validation/test invocations preserve existing summary split results instead of overwriting unrelated splits;
  - each invocation writes a run-specific manifest under `<output_dir>/manifests/` as well as the latest `manifest.json`.

### Limitations

- This was one fixed three-aspect configuration, not a hyperparameter sweep.
- Evaluation rows contain held-out labels only, so the result does not test all-row absence calibration.
- Full 12-fold Qwen LoRA LOAO remains unrun by design.
- The first failed attempt exposed a runner/data edge case. The final result uses the corrected dataset skip rule for fully truncated assistant answers.

### Next Step

- Update thesis tables, reproducibility register, roadmap, and handoff docs with this fixed-split adaptation result.
- Keep full Qwen LoRA LOAO gated behind target GPU/storage confirmation and the pre-launch validation checks.

## 2026-07-02: Pre-Registered Single-Fold All-Row Qwen LoRA LOAO Pilot

### Purpose

- Run a single-fold Qwen LoRA LOAO pilot before considering the full 12-fold fine-tuned LOAO.
- Test whether QLoRA improves all-row absence calibration for one held-out aspect, rather than only improving positive-row/fixed-split evaluation.
- Treat this as a go/no-go diagnostic. It is not a replacement for full 12-fold LOAO.

### Pre-Flight State

- Remote fetched with `git fetch --prune`.
- Current branch: `main`.
- Git state before code changes: clean and synced after pushing prior commits to `origin/main`.
- Latest pushed commit before this pilot: `97e7550 llm: run fixed qwen lora heldout experiment`.

### Fold And Baselines

- Held-out aspect: `Company brand: Competitor`.
- Rationale:
  - Qwen zero-shot has known competitor-boundary weaknesses.
  - The qualitative taxonomy identifies competitor-positive recall and semantic boundary ambiguity as important failure modes.
  - The fold is hard enough to be diagnostic but has enough positive validation/test rows to interpret.
- Existing all-row zero-shot Qwen validation baseline for this fold:
  - pair samples F1: `0.0360`;
  - pair micro F1: `0.2397`;
  - precision: `0.1645`;
  - recall: `0.4419`;
  - FP rows / 100: `17.7862`;
  - valid JSON/schema-valid: `1.0000 / 1.0000`.
- Existing all-row DistilBERT validation baseline for this fold:
  - pair samples F1: `0.0293`;
  - pair micro F1: `0.2490`;
  - precision: `0.1902`;
  - recall: `0.3605`;
  - FP rows / 100: `12.2990`.

### Input Scope

- SFT data preparation command:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed --heldout-aspect "Company brand: Competitor" --eval-row-scope all --output-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow
```

- Expected row counts from the split helper:
  - train: `7,314`;
  - validation: `1,057`, including `86` positive held-out rows and `971` empty-gold rows;
  - test: `1,587`, including `121` positive held-out rows and `1,466` empty-gold rows.

### Planned Primary Configuration

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Loading: 4-bit bitsandbytes QLoRA.
- LoRA: rank `8`, alpha `16`, dropout `0.05`.
- Strategy: `example_filtered`.
- Prompt variant: indexed candidate IDs.
- Eval row scope: all official validation/test rows.
- Epochs: `1`.
- Batch size: `1`.
- Gradient accumulation: `8`.
- Learning rate: `1e-5`.
- Weight decay: `0.0`.
- Warmup ratio: `0.05`.
- Max train length: `512`.
- Max input tokens: `1024`.
- Max new tokens: `192`.
- Seed: `13`.
- Gradient checkpointing: disabled for the first run, matching the successful fixed-split long run. If this OOMs, retry with gradient checkpointing and record the fallback.
- Output directory: `outputs/llm/qwen_lora_loao_single_fold_company_brand_competitor_r8_lr1e-5_ep1_allrow_20260702/` (ignored).
- Log files under `outputs/logs/` (ignored).

Validation-only command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_single_fold_company_brand_competitor_r8_lr1e-5_ep1_allrow_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

### Validation Decision Rule

- Primary selection metric: validation pair micro F1, because this is an all-row one-aspect LOAO detection problem with many empty-gold rows.
- Supporting diagnostics:
  - pair precision/recall;
  - FP rows / 100;
  - predicted labels per example;
  - valid JSON/schema-valid rates;
  - positive-gold recall and sentiment accuracy where available.
- Continue to test only if validation is interpretable and materially useful:
  - valid JSON remains `>= 0.99`;
  - schema-valid remains close to previous Qwen all-row behaviour;
  - pair micro F1 improves over Qwen zero-shot by about `0.02` absolute or gives a clear precision/FP reduction without collapsing recall.
- If validation is worse or only trivially changed, do not run test immediately. Record it as a negative pilot and consider whether an absence-aware SFT data format is needed before spending more GPU time.

### Optimisation / Stopping Rule

- First run the primary `1e-5`, 1-epoch configuration because it was stable in the fixed held-out-aspect long run.
- If it fails due to OOM, retry the same configuration with gradient checkpointing before changing modelling assumptions.
- If it fails due to non-finite loss, inspect skipped-label counts and logs before lowering LR.
- If validation improves clearly, run the same fold's test using `--skip-training-if-adapter-exists`.
- If validation does not improve, stop this pilot unless there is an obvious one-run fix. The next likely improvement would be a new absence-aware one-candidate SFT data construction, which should be pre-registered separately rather than hidden inside this run.

### Commit Scope

- Commit only code, tests, tracked docs, and aggregate metrics.
- Do not commit `outputs/`, raw predictions, raw review text, adapters, checkpoints, model weights, credentials, or private endpoints.

### Primary Validation Run: Early Stop

Observed training result:

- Training rows: `7,314` original, `7,302` used after skipping `12` fully truncated-answer rows.
- Optimiser steps completed: `913 / 913`.
- Final train loss: `0.1178`.
- Training runtime: `7,483.0` seconds.
- Adapter was saved under the ignored output directory.

The validation generation was stopped early after `107 / 1,057` rows because the partial diagnostic showed severe over-prediction:

| Metric | Partial Value |
| --- | ---: |
| examples | 107 |
| positive-gold rows | 12 |
| pair samples F1 | 0.0935 |
| pair micro F1 | 0.1681 |
| pair precision | 0.0935 |
| pair recall | 0.8333 |
| FP rows / 100 | 88.7850 |
| predicted labels / example | 1.0000 |
| gold labels / example | 0.1121 |
| valid JSON rate | 1.0000 |
| schema-valid rate | 1.0000 |

Interpretation:

- The standard indexed prompt/data setup trained successfully but produced a label for every validation row in the all-row setting.
- This is far worse than the zero-shot all-row baseline's validation FP rows / 100 (`17.7862`) and does not test the intended improvement.
- The run is a negative partial diagnostic, not a completed validation result.

### Prompt-Only Optimisation Branch

Before retraining, run a cheaper prompt-only optimisation:

- Reuse the saved adapter from the stopped standard-indexed run.
- Regenerate the same fold's validation prompts with `indexed_conservative`.
- Use `--skip-training-if-adapter-exists` from a new output directory containing a copied local adapter.
- Run full all-row validation only.

Rationale:

- The stopped run used the older `indexed` prompt text from `src/msc_project/llm/qwen_format.py`, which does not explicitly say to return `[]` when no candidate aspect is relevant.
- `indexed_conservative` adds explicit absence guidance and is the lowest-cost check for whether the over-prediction failure is prompt-sensitive.
- If this still over-predicts badly, the next useful optimisation is not another prompt-only rerun; it is a separately pre-registered absence-aware one-candidate SFT data construction.

Planned conservative data command:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed_conservative --heldout-aspect "Company brand: Competitor" --eval-row-scope all --output-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_indexed_conservative
```

Planned conservative validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_indexed_conservative --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_single_fold_company_brand_competitor_r8_lr1e-5_ep1_allrow_conservative_eval_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --skip-training-if-adapter-exists --no-resume-predictions --no-skip-existing-predictions
```

### Conservative Prompt-Only Branch: Early Stop

The conservative prompt-only evaluation was also stopped early. It reduced over-prediction compared with the standard prompt, but not enough to be competitive:

| Metric | Partial Value |
| --- | ---: |
| examples | 88 |
| positive-gold rows | 8 |
| pair samples F1 | 0.0795 |
| pair micro F1 | 0.1707 |
| pair precision | 0.0946 |
| pair recall | 0.8750 |
| FP rows / 100 | 75.0000 |
| predicted labels / example | 0.8409 |
| gold labels / example | 0.0909 |
| valid JSON rate | 1.0000 |
| schema-valid rate | 1.0000 |

Interpretation:

- Explicit conservative prompt wording helps compared with the standard prompt, but the model still over-predicts badly.
- Prompt-only optimisation is therefore not enough.

### Absence-Aware Singleton SFT Branch

Next optimisation:

- Build an absence-aware singleton training set from seen aspects.
- Each training row uses exactly one seen candidate aspect.
- Positive singleton rows contain the aspect's sentiment label.
- Sampled negative singleton rows use `[]`.
- Keep the same all-row validation/test scope for the held-out aspect.

Data command:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed_conservative --heldout-aspect "Company brand: Competitor" --eval-row-scope all --train-candidate-mode singleton --singleton-negative-ratio 1 --seed 13 --output-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg1_indexed_conservative
```

Observed generated row counts:

| Split | Rows | Non-Empty Gold Rows | Empty Gold Rows |
| --- | ---: | ---: | ---: |
| train | 24,368 | 12,185 | 12,183 |
| validation | 1,057 | 86 | 971 |
| test | 1,587 | 121 | 1,466 |

Training budget:

- Use `--max-train-steps 913` so the singleton branch has the same optimiser-step budget as the primary standard-indexed run.
- This prevents the optimisation from becoming an uncontrolled extra-compute comparison.

Planned singleton validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg1_indexed_conservative --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_single_fold_company_brand_competitor_singleton_neg1_r8_lr1e-5_steps913_allrow_20260702 --eval-split validation --epochs 1 --max-train-steps 913 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Stopping rule:

- If singleton validation still fails to beat Qwen zero-shot validation pair micro F1 (`0.2397`) or does not reduce false positives materially, stop the single-fold optimisation and record that ordinary SFT plus prompt/data alignment was insufficient.
- If it improves materially, run the same fold's test with the saved adapter before considering any additional folds.

### Absence-Aware Singleton Neg1 Result and Mid-Ratio Optimisation Plan

Observed singleton neg1 validation result:

| Metric | Value |
| --- | ---: |
| examples | 1,057 |
| positive-gold rows | 86 |
| pair samples F1 | 0.0028 |
| pair micro F1 | 0.0625 |
| pair precision | 0.3000 |
| pair recall | 0.0349 |
| FP rows / 100 | 0.1892 |
| FN rows / 100 | 7.3794 |
| predicted labels / example | 0.0095 |
| gold labels / example | 0.0814 |
| valid JSON rate | 1.0000 |
| schema-valid rate | 1.0000 |

Runtime evidence:

- Training rows: `24,368` original, `24,318` used after skipping `50` fully truncated-answer rows.
- Optimiser steps completed: `913 / 913`.
- Final train loss: `0.0703`.
- Training runtime: `5,863.1` seconds.
- Validation runtime: `436.8` seconds.
- Adapter and predictions are local-only ignored outputs.

Interpretation:

- The singleton neg1 branch fixed the original over-prediction problem, but over-corrected into severe under-prediction.
- This brackets the likely calibration problem:
  - grouped SFT predicted the held-out aspect for nearly every all-row validation row;
  - singleton neg1 SFT predicted almost no held-out aspects.
- A single mid-ratio singleton run is justified before stopping, because it directly tests whether the useful operating point lies between those two failure modes.

Pre-registered next run:

- Data variant: singleton SFT with `indexed_conservative` prompts and `--singleton-negative-ratio 0.25`.
- Held-out aspect: `Company brand: Competitor`.
- Evaluation scope: all validation rows.
- Training budget: `913` optimiser steps, matching the earlier primary run and singleton neg1 run.
- Primary metric: validation pair micro F1.
- Supporting diagnostics: precision, recall, FP rows / 100, FN rows / 100, predicted labels / example, valid/schema rates.
- Test rule: run test only if validation materially improves over the same-fold Qwen zero-shot (`0.2397`) or provides a clearly better precision/recall trade-off than the local DistilBERT all-row validation baseline.

Planned data command:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed_conservative --heldout-aspect "Company brand: Competitor" --eval-row-scope all --train-candidate-mode singleton --singleton-negative-ratio 0.25 --seed 13 --output-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg025_indexed_conservative
```

Planned validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg025_indexed_conservative --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_single_fold_company_brand_competitor_singleton_neg025_r8_lr1e-5_steps913_allrow_20260702 --eval-split validation --epochs 1 --max-train-steps 913 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Stopping rule:

- If neg0.25 still does not beat same-fold Qwen zero-shot validation pair micro F1 or still has an unusable FP/FN trade-off, stop the single-fold optimisation.
- Do not run test for a failed validation branch.
- Do not start full 12-fold Qwen LoRA LOAO from these results.

### Absence-Aware Singleton Neg0.25 Result and Final Recall-Shift Plan

Observed singleton neg0.25 validation result:

| Metric | Value |
| --- | ---: |
| examples | 1,057 |
| positive-gold rows | 86 |
| pair samples F1 | 0.0104 |
| pair micro F1 | 0.1803 |
| pair precision | 0.3056 |
| pair recall | 0.1279 |
| pair label TP / FP / FN | 11 / 25 / 75 |
| predicted labels | 36 |
| gold labels | 86 |
| FP rows / 100 | 1.6083 |
| FN rows / 100 | 6.3387 |
| predicted labels / example | 0.0341 |
| valid JSON rate | 1.0000 |
| schema-valid rate | 1.0000 |

Runtime evidence:

- Training rows: `15,296` original, `15,265` used after skipping `31` fully truncated-answer rows.
- Optimiser steps completed: `913 / 913`.
- Final train loss: `0.0681`.
- Training runtime: `5,871.9` seconds.
- Validation runtime: `510.8` seconds.
- Adapter and predictions are local-only ignored outputs.

Interpretation:

- Neg0.25 is the best Qwen LoRA all-row single-fold branch so far, but it still does not beat same-fold Qwen zero-shot (`0.2397`) or local DistilBERT (`0.2490`) on validation pair micro F1.
- Compared with neg1, it improves recall and F1 while retaining low false positives, but it is still too conservative:
  - predicted labels: `36` vs `86` gold labels;
  - recall: `0.1279`;
  - false-negative rows / 100: `6.3387`.
- Per the validation-first rule, do not run test for neg0.25.
- One final recall-shift branch is justified: reduce the singleton negative ratio from `0.25` to `0.10` while keeping the same prompt, seed, fold, optimiser-step budget, and evaluation scope.

Pre-registered final optimisation run:

- Data variant: singleton SFT with `indexed_conservative` prompts and `--singleton-negative-ratio 0.10`.
- Held-out aspect: `Company brand: Competitor`.
- Evaluation scope: all validation rows.
- Training budget: `913` optimiser steps.
- Primary metric: validation pair micro F1.
- Supporting diagnostics: precision, recall, FP rows / 100, FN rows / 100, predicted labels / example, valid/schema rates.
- Test rule: run test only if validation materially beats same-fold Qwen zero-shot or gives a clearly superior precision/recall trade-off to the local DistilBERT validation baseline.

Planned data command:

```powershell
python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy example_filtered --prompt-variant indexed_conservative --heldout-aspect "Company brand: Competitor" --eval-row-scope all --train-candidate-mode singleton --singleton-negative-ratio 0.10 --seed 13 --output-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg010_indexed_conservative
```

Planned validation command:

```powershell
python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_lora_loao_sft_20260702\02_company_brand_competitor_allrow_singleton_neg010_indexed_conservative --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_loao_single_fold_company_brand_competitor_singleton_neg010_r8_lr1e-5_steps913_allrow_20260702 --eval-split validation --epochs 1 --max-train-steps 913 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters
```

Stopping rule:

- If neg0.10 still does not beat the same-fold baselines, stop this single-fold optimisation.
- Do not run more singleton ratio sweeps in this session unless a validation result crosses the baseline threshold.
- Do not start full 12-fold Qwen LoRA LOAO from these single-fold results if neg0.10 remains below baseline.

## 2026-07-02 - Planned Local-to-Qwen LOAO Offline Cascade Diagnostic

Question:

- Test whether the stronger thesis direction is not Qwen replacing DistilBERT, but Qwen complementing DistilBERT under LOAO taxonomy shift.
- Reuse existing full all-row LOAO predictions:
  - local DistilBERT candidate-aspect selector plus DistilBERT aspect-conditioned sentiment;
  - Qwen3-4B indexed zero-shot.
- Do not call Qwen again and do not retrain DistilBERT for this first diagnostic.

Motivation:

- Gemini fixed-split evidence already showed that a local-to-LLM cascade can outperform an LLM-only deployment because each model handles different parts of the distribution.
- Qwen zero-shot has strong positive-gold recognition but weak absence calibration.
- DistilBERT is more conservative and cheaper, but weaker under some unseen-aspect semantic shifts.
- A local-to-Qwen cascade may therefore be more thesis-relevant than direct Qwen JSON SFT.

Inputs:

- Local LOAO predictions:
  `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630/`
- Qwen validation predictions:
  `outputs/llm/qwen_loao_heldout_aspect_all_rows_validation_20260701/`
- Qwen test predictions:
  `outputs/llm/qwen_loao_heldout_aspect_all_rows_test_20260701/`

Planned command:

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --output-dir .\outputs\analysis\local_qwen_loao_cascade_20260702
```

Policies:

- `local_only`;
- `qwen_only`;
- `local_nonempty_else_qwen`;
- `qwen_nonempty_else_local`;
- `pair_agreement`;
- `aspect_agreement_local_sentiment`;
- `aspect_agreement_qwen_sentiment`;
- `union_pairs`.

Selection:

- Primary diagnostic metric: validation mean pair micro F1 across the 12 LOAO folds.
- Also report pair precision/recall, false-positive rows / 100, false-negative rows / 100, and estimated Qwen call rate.
- Compare:
  - fixed single policies;
  - one global validation-selected policy applied to test;
  - an optimistic per-aspect validation-selected policy applied to test.

Important limitation:

- Existing DistilBERT LOAO prediction files do not contain row-level score/margin features.
- This run is therefore an offline prediction-combination diagnostic, not the final confidence-based uncertainty gate.
- If the result shows useful signal, the next method step is to rerun/export local LOAO score and margin features for true uncertainty routing.

Observed command:

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --output-dir .\outputs\analysis\local_qwen_loao_cascade_20260702
```

Observed results:

| Policy / Selection | Split | Pair Micro F1 Mean | Precision Mean | Recall Mean | FP Rows / 100 Mean | FN Rows / 100 Mean | Qwen Call Rate Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT only | test | 0.3128 | 0.3345 | 0.4315 | 12.7022 | 8.6746 | 0.0000 |
| Qwen only | test | 0.3378 | 0.2379 | 0.8182 | 34.4150 | 2.9235 | 1.0000 |
| Global validation-selected: `aspect_agreement_qwen_sentiment` | test | 0.3470 | 0.4116 | 0.4147 | 8.8112 | 9.0265 | 0.1868 |
| Optimistic per-aspect validation-selected mixed policy | test | 0.4131 | 0.3461 | 0.5779 | 16.1101 | 3.8332 | 0.3645 |

Interpretation:

- The global validation-selected policy improves mean test pair micro F1 over both local-only and Qwen-only while reducing Qwen calls to about `18.7%` of rows.
- The selected global policy is a confirmation gate: keep the prediction only when local predicts the aspect and Qwen also predicts the aspect, using Qwen's sentiment label.
- This sharply improves precision and false-positive control relative to Qwen-only, but gives up Qwen's high recall.
- The optimistic per-aspect validation-selected policy is much stronger (`0.4131` mean pair micro F1) and beats the documented lexical all-row LOAO lower bound (`0.3780`), but it has more selection freedom and a higher Qwen call rate.
- The result supports the model-division hypothesis:
  - Qwen is useful as a semantic complement;
  - DistilBERT remains valuable as a cheap stabilising gate;
  - the most promising next step is a true score/margin uncertainty gate, not direct Qwen JSON SFT.

Limitations:

- This is an offline combination of existing predictions, not a fresh deployment run.
- The old local LOAO predictions do not contain row-level score/margin features, so the current gate uses prediction presence/agreement rather than calibrated local uncertainty.
- Per-aspect validation selection is an optimistic diagnostic and should not be presented as a final deployed policy unless the selection rule is frozen before test in a future run.

Next step:

- Rerun or export local LOAO score features for the same predictions if feasible.
- Implement a true local-score/margin-to-Qwen uncertainty gate and compare it against:
  - local-only;
  - Qwen-only;
  - simple agreement confirmation;
  - per-aspect validation-selected diagnostic.

### Absence-Aware Singleton Neg0.10 Result and Stop Decision

Observed singleton neg0.10 validation result:

| Metric | Value |
| --- | ---: |
| examples | 1,057 |
| positive-gold rows | 86 |
| pair samples F1 | 0.0180 |
| pair micro F1 | 0.1900 |
| pair precision | 0.1667 |
| pair recall | 0.2209 |
| pair label TP / FP / FN | 19 / 95 / 67 |
| predicted labels | 114 |
| gold labels | 86 |
| FP rows / 100 | 7.2848 |
| FN rows / 100 | 4.6358 |
| predicted labels / example | 0.1079 |
| valid JSON rate | 1.0000 |
| schema-valid rate | 1.0000 |

Runtime evidence:

- Training rows: `13,420` original, `13,394` used after skipping `26` fully truncated-answer rows.
- Optimiser steps completed: `913 / 913`.
- Final train loss: `0.0546`.
- Training runtime: `6,407.7` seconds.
- Validation runtime: `736.8` seconds.
- Adapter and predictions are local-only ignored outputs.

Interpretation:

- Neg0.10 improves recall over neg0.25 and is the best singleton-ratio validation F1 in this local single-fold pilot, but it still fails the pre-registered threshold:
  - Qwen zero-shot same-fold validation pair micro F1: `0.2397`;
  - local DistilBERT same-fold validation pair micro F1: `0.2490`;
  - best Qwen LoRA singleton branch: `0.1900`.
- Lowering the negative ratio moved the model in the expected direction:
  - neg1: very low false positives, very low recall;
  - neg0.25: improved recall, still conservative;
  - neg0.10: higher recall but substantially more false positives.
- The trade-off remains below the zero-shot and local baselines. This suggests that the current singleton SFT recipe changes calibration but does not solve held-out aspect generalisation for this hard fold.

Stop decision:

- Do not run test for neg0.10.
- Do not run further singleton-ratio sweeps in this session.
- Do not launch the full 12-fold Qwen LoRA LOAO using this SFT recipe.
- Full Qwen LoRA LOAO would need a revised calibration/training objective before it is worth the 12-fold GPU cost.

Thesis contribution:

- This single-fold pilot is a negative but useful adaptation result. It shows that local QLoRA can train and evaluate real all-row LOAO data, but ordinary held-out-aspect SFT plus simple absence-aware negative sampling does not outperform the existing zero-shot/open-weight or local DistilBERT baselines on the selected hard fold.
- The dissertation should use it to justify a compute-aware limitation: full fine-tuned Qwen LOAO was not launched because the validation gate failed on a representative hard fold, not because the runner was unready.

## 2026-07-02 - Local-to-Qwen Hybrid Direction Pivot and Score Export Pre-Registration

### Purpose

- Record the modelling pivot after the Qwen LoRA validation gate failed and the first local-to-Qwen LOAO cascade diagnostic showed positive signal.
- Prevent future sessions from drifting back to full 12-fold Qwen JSON-SFT LOAO under the current recipe.
- Prepare the next experiment: rerun/export the local DistilBERT LOAO branch with score, threshold-distance, and sentiment-confidence features, then reuse existing Qwen zero-shot LOAO predictions for score/margin routing.

### Code Or Protocol Changes

- Added `docs/qwen_local_hybrid_direction.md` as the source-of-truth note for the new direction.
- Updated the project roadmap and handoff documents to state that full Qwen LoRA LOAO is deferred until a revised objective passes validation.
- Extended local sentiment helpers so feature exports can include:
  - predicted sentiment;
  - sentiment class probabilities;
  - top probability;
  - second probability;
  - probability margin;
  - entropy.
- Extended `scripts/run_aspect_label_aware_baseline.py` so prediction JSONL can contain both `score_features` and `sentiment_features`.
- Extended `scripts/analyse_local_qwen_loao_cascade.py` so it automatically adds score-distance and sentiment-margin policies when the selected local LOAO directory contains those features.

### Setup

- Dataset: public FABSA export.
- Protocol: full all-row LOAO over all 12 FABSA aspects.
- Local model to rerun/export:
  - candidate-aspect DistilBERT cross-encoder;
  - `example_filtered` training strategy;
  - all validation/test rows;
  - validation pair micro F1 threshold selection;
  - DistilBERT aspect-conditioned sentiment.
- Qwen predictions to reuse:
  - existing indexed Qwen zero-shot all-row LOAO validation/test predictions.
- No new Qwen calls are planned for the first score/margin diagnostic.

### Commands

Planned local score/sentiment-confidence export:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702
```

Planned hybrid analysis:

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_score_margin_cascade_20260702
```

### Outputs

- Planned local ignored outputs:
  - `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702/`
  - `outputs/analysis/local_qwen_loao_score_margin_cascade_20260702/`
- Tracked documentation:
  - `docs/qwen_local_hybrid_direction.md`
  - this log entry
  - updated roadmap/handoff notes.

### Results

- Not yet run at pre-registration time.
- At pre-registration time, expected outputs were not described as completed until the commands finished and metrics were read from generated summaries.

### Interpretation

- The current best hypothesis is that Qwen should be used as a selective semantic judge, not as a direct replacement for the local calibrated DistilBERT candidate-aspect pipeline.
- The score/margin export is designed to test whether validation-selected uncertainty routing can improve over the previous global agreement gate (`0.3470` mean test pair micro F1, `18.7%` Qwen call rate).

### Next Step

- Run the local LOAO score/sentiment-confidence export.
- Run the local-to-Qwen score/margin routing analysis.
- Document whether the result improves, matches, or fails to improve over the existing agreement-gate diagnostic.

## 2026-07-03 - Local-to-Qwen Score-Distance LOAO Routing Result

### Purpose

- Close the pre-registered local-to-Qwen score/margin routing diagnostic.
- Test whether local DistilBERT selector uncertainty is a better trigger for Qwen than the previous prediction-agreement-only cascade.
- Reuse existing Qwen zero-shot LOAO predictions; do not call Qwen again.

### Commands

Local DistilBERT LOAO score/sentiment-confidence export:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702
```

Hybrid analysis:

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_score_margin_cascade_20260702
```

### Inputs And Outputs

- Protocol: full all-row LOAO over all 12 FABSA aspects.
- Local model: candidate-aspect DistilBERT cross-encoder, `example_filtered`, LR `3e-5`, 3 epochs, validation pair micro F1 threshold selection, DistilBERT aspect-conditioned sentiment.
- Qwen model: existing indexed Qwen zero-shot LOAO validation/test outputs from `Qwen/Qwen3-4B-Instruct-2507`.
- Local output directory:
  `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702/`
- Analysis output directory:
  `outputs/analysis/local_qwen_loao_score_margin_cascade_20260702/`
- Local run logs:
  `outputs/logs/loao_score_export_20260702.out.log`
  and `outputs/logs/loao_score_export_20260702.err.log`.
- Feature coverage: all 12 validation prediction files contain both `score_features` and `sentiment_features`.
- Runtime/hardware evidence: local export log timestamp range 2026-07-02 23:37 to 2026-07-03 03:51; GPU available as NVIDIA GeForce RTX 5050 Laptop GPU, 8,151 MiB VRAM, driver 596.08.

### Results

| Policy / Selection | Split | Pair Micro F1 Mean | Precision Mean | Recall Mean | FP Rows / 100 Mean | FN Rows / 100 Mean | Qwen Call Rate Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT rerun only | test | 0.3158 | 0.3449 | 0.3965 | 10.4442 | 8.9582 | 0.0000 |
| Qwen zero-shot only | test | 0.3378 | 0.2379 | 0.8182 | 34.4150 | 1.9114 | 1.0000 |
| Previous global agreement gate | test | 0.3470 | 0.4116 | 0.4147 | 8.8112 | 9.0265 | 0.1868 |
| Global validation-selected score-distance gate: `score_abs_replace_le_0.05` | test | 0.3800 | 0.3323 | 0.5426 | 16.6667 | 4.0485 | 0.2239 |
| Per-aspect validation-selected mixed policy | test | 0.4186 | 0.3567 | 0.5549 | 14.9811 | 4.1168 | 0.3480 |
| Best sentiment-margin-only diagnostic: `sentiment_margin_confirm_le_0.50` | test | 0.3204 | 0.3518 | 0.3992 | 10.2394 | 8.9845 | 0.0106 |

### Interpretation

- The score-distance gate is a positive result. It improves over the local rerun, Qwen-only, and the previous global agreement gate while calling Qwen on about `22.4%` of rows.
- The selected global policy replaces local predictions with Qwen only when the local aspect score is close to the validation-selected threshold (`<= 0.05` absolute distance). This is the cleanest evidence that Qwen is useful as a selective semantic judge for locally uncertain unseen-aspect cases.
- The per-aspect validation-selected mixed policy is stronger (`0.4186`) but should be treated as an optimistic diagnostic because it selects different policies by held-out aspect.
- Sentiment-margin routing is a weak/negative sub-result. Its best test mean pair micro F1 is only `0.3204`, so the improvement is not coming from routing sentiment uncertainty alone.

### Limitations

- This remains an offline combination of existing Qwen predictions, not a fresh deployment system that makes row-by-row Qwen calls.
- The local rerun is stochastic and should not be treated as a bit-for-bit reproduction of the earlier local-only LOAO run; it is close enough for routing analysis and includes the missing score/sentiment features.
- The per-aspect result has higher selection freedom than the global validation-selected policy.
- Outputs may contain review text and remain local-only under ignored `outputs/`.

### Next Step

- Promote score-distance local-to-Qwen routing as the main Qwen follow-up method in thesis planning.
- Do not return to full 12-fold Qwen JSON-SFT LOAO with the current grouped/singleton recipe.
- If another Qwen experiment is needed, test candidate-wise Qwen semantic judgement or a revised absence-calibration objective with a validation gate first.

## 2026-07-03 - User-Prioritised Local-to-Qwen Optimisation Queue

### Purpose

- Record the user's prioritisation of possible follow-up experiments after the positive local-to-Qwen score-distance routing result.
- Keep future sessions aligned on the numbering and order of proposed methods.

### Prioritised Queue

1. Asymmetric score-distance router.
   - Immediate priority.
   - Systematically separate below-threshold rescue from above-threshold confirmation/veto.
   - Reuse existing local and Qwen LOAO outputs.

2. Cost-quality / F1-call-rate Pareto curve.
   - Immediate priority.
   - Should be run with item 1 so the thesis can report the quality/deployment trade-off rather than only one selected policy.

3. Lightweight defer router.
   - Potentially valuable but lower priority because it may not improve over the simple score-distance rule.
   - Keep any learned router simple and regularised.

4. Fair per-aspect routing.
   - Diagnostic only for now.
   - The current per-aspect diagnostic is strong but should remain an upper bound.
   - Do not make it the main method because a real new topic will not usually have enough topic-specific validation data to choose its own policy.

5. Candidate-wise Qwen semantic judge.
   - Deferred.
   - Formulation: `review + one candidate aspect -> absent / positive / negative / neutral`.
   - This is a separate Qwen-inference route, not a continuation of items 1 and 2.

6. Aspect descriptions and boundary examples.
   - Deferred with item 5.
   - Should be a small controlled ablation.

### Thesis Decision

- Items 1 and 2 are considered sufficient for the main thesis-facing local-to-Qwen method if completed cleanly.
- The headline should be a unified global router shared across held-out aspects.
- Per-aspect routing remains a diagnostic upper bound.
- Items 5 and 6 should not be run unless items 1 and 2 unexpectedly fail to provide enough evidence or a supervisor specifically requests another Qwen inference method.

### Deferred

- Qwen logits or absent-threshold calibration.
- DistilBERT shortlist plus Qwen judging for large candidate sets.
- Gatekeeper-style confidence tuning.
- Full Qwen JSON-SFT LOAO under the current grouped/singleton recipe.

### Next Step

- Implement item 1 and item 2 first.
- Do not start items 3-6 unless the systematic no-new-Qwen-call global routing analysis leaves a clear thesis gap.

## 2026-07-03 - Planned Local-to-Qwen Asymmetric Score Router And Pareto Analysis

### Purpose

- Complete user-confirmed local-to-Qwen item 1 and item 2.
- Test whether one unified global asymmetric score-distance router can improve on the previous global `score_abs_replace_le_0.05` gate by separating:
  - below-threshold Qwen rescue for local false negatives;
  - above-threshold Qwen confirmation/veto for weak local positives;
  - far-from-threshold local retention.
- Produce aggregate, plot-ready F1/call-rate data for the same validation-selected policy grid.

### Planned Setup

- Dataset and protocol: FABSA full all-row LOAO over all 12 held-out aspects.
- Local input: existing DistilBERT LOAO score export at `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702/`.
- Qwen inputs: existing zero-shot LOAO predictions at `outputs/llm/qwen_loao_heldout_aspect_all_rows_validation_20260701/` and `outputs/llm/qwen_loao_heldout_aspect_all_rows_test_20260701/`.
- No new Qwen calls will be made.
- No DistilBERT retraining will be run unless the existing score export proves unusable.
- Main method: one unified global validation-selected policy shared across all held-out aspects.
- Test protocol: select the global rule on validation aggregate mean pair micro F1, then evaluate that selected rule once on test.
- Per-aspect policy selection, if reported, remains an upper-bound diagnostic only.

### Planned Command

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_asymmetric_score_router_20260703 --public-pareto-csv .\docs\thesis_figure_data\qwen_local_qwen_loao_pareto.csv --public-selected-csv .\docs\thesis_figure_data\qwen_local_qwen_loao_selected.csv
```

### Planned Outputs

- Local ignored analysis directory:
  `outputs/analysis/local_qwen_loao_asymmetric_score_router_20260703/`
- Tracked aggregate CSVs:
  - `docs/thesis_figure_data/qwen_local_qwen_loao_pareto.csv`
  - `docs/thesis_figure_data/qwen_local_qwen_loao_selected.csv`
- Tracked documentation updates after results are observed.

### Planned Comparisons

- Local-only.
- Qwen-only.
- Previous global agreement gate.
- Existing global `score_abs_replace_le_0.05`.
- New asymmetric rescue/confirm/veto policy family.
- Per-aspect policy selection only as an upper-bound diagnostic if the same grid already produces it.

### Planned Metrics

- Mean pair micro F1.
- Mean pair samples F1.
- Mean precision.
- Mean recall.
- Mean false-positive rows / 100.
- Mean false-negative rows / 100.
- Mean Qwen call rate.

### Caveats

- This remains an offline policy analysis over existing local and Qwen predictions.
- Qwen call rate is an estimated row-level invocation rate implied by the policy, not a newly observed serving trace.
- Raw predictions and review text remain local-only under ignored `outputs/`.

### Code Or Protocol Changes

- Updated `scripts/analyse_local_qwen_loao_cascade.py`:
  - added combined asymmetric score-distance policies:
    - `score_asym_rescue_le_<below>_confirm_le_<above>`;
    - `score_asym_rescue_le_<below>_veto_le_<above>`;
  - kept far-from-threshold predictions local;
  - evaluated the full policy grid on validation, then limited test evaluation to fixed comparisons, the single global validation-selected policy, and the policies needed for the per-aspect diagnostic;
  - added aggregate Pareto CSV output with policy family, Pareto-frontier flag, selected-rule flag, and public plot-ready metric columns.
- Updated `tests/test_analyse_local_qwen_loao_cascade.py` with focused asymmetric routing and Pareto-table tests.

### Observed Command

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_asymmetric_score_router_20260703 --public-pareto-csv .\docs\thesis_figure_data\qwen_local_qwen_loao_pareto.csv --public-selected-csv .\docs\thesis_figure_data\qwen_local_qwen_loao_selected.csv
```

### Observed Outputs

- Local ignored output directory:
  `outputs/analysis/local_qwen_loao_asymmetric_score_router_20260703/`
- Local ignored files:
  - `aggregate_policy_results.csv`
  - `pareto_policy_results.csv`
  - `per_aspect_policy_results.csv`
  - `selected_policy_results.csv`
  - `summary.json`
- Tracked aggregate-only CSVs:
  - `docs/thesis_figure_data/qwen_local_qwen_loao_pareto.csv`
  - `docs/thesis_figure_data/qwen_local_qwen_loao_selected.csv`
- The public Pareto CSV contains `234` aggregate rows: `229` validation policy-grid rows plus `5` test comparison/selected rows.

### Observed Results

Test aggregate comparison on the same score-export/Qwen inputs:

| Policy / Selection | Split | Pair Micro F1 Mean | Pair Samples F1 Mean | Precision Mean | Recall Mean | FP Rows / 100 Mean | FN Rows / 100 Mean | Qwen Call Rate Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT rerun only | test | 0.3158 | 0.0527 | 0.3449 | 0.3965 | 10.4442 | 8.9582 | 0.0000 |
| Qwen zero-shot only | test | 0.3378 | 0.1212 | 0.2379 | 0.8182 | 34.4150 | 1.9114 | 1.0000 |
| Agreement gate on score-export rerun, `aspect_agreement_qwen_sentiment` | test | 0.3467 | 0.0513 | 0.4200 | 0.3815 | 7.1519 | 9.3153 | 0.1614 |
| Existing score-distance gate, `score_abs_replace_le_0.05` | test | 0.3800 | 0.0998 | 0.3323 | 0.5426 | 16.6667 | 4.0485 | 0.2239 |
| Global validation-selected asymmetric router, `score_asym_rescue_le_0.05_confirm_le_0.30` | test | 0.3900 | 0.0988 | 0.3530 | 0.5300 | 15.2752 | 4.2166 | 0.2905 |
| Expanded-grid per-aspect validation-selected diagnostic | test | 0.4178 | 0.1015 | 0.3511 | 0.5650 | 15.4012 | 4.0538 | 0.3442 |

Validation selection evidence:

- Global selected policy: `score_asym_rescue_le_0.05_confirm_le_0.30`.
- Validation mean pair micro F1: `0.4224`.
- Validation Qwen call rate: `0.2863`.
- The validation Pareto frontier contains a range of low-to-moderate call-rate policies; always-Qwen is dominated because it uses Qwen on every row while reaching lower validation pair micro F1 than selective score-distance policies.

### Interpretation

- Item 1 is complete. The best unified global router is the asymmetric rescue/confirm rule `score_asym_rescue_le_0.05_confirm_le_0.30`.
- It improves over the existing global `score_abs_replace_le_0.05` result (`0.3900` versus `0.3800` mean test pair micro F1).
- The improvement comes from allowing Qwen to rescue near-threshold local false negatives while using Qwen confirmation/veto on a broader above-threshold weak-positive band.
- The selected router also improves precision and false-positive rows relative to `score_abs_replace_le_0.05`, but has slightly lower recall and a higher Qwen call rate.
- Qwen-only still has the highest pair samples F1 in this comparison (`0.1212`), but it has much lower pair precision and far more false-positive rows. The router is selected on pair micro F1 and is primarily a cost/precision/recall trade-off result, not a pair-samples-F1 win.
- Item 2 is complete. The Pareto CSV shows that selective Qwen use is more reasonable than always-Qwen for the primary LOAO detection metric: moderate call-rate score-routing policies dominate the 100% Qwen point on validation pair micro F1.
- The per-aspect result remains diagnostic only. It uses separate validation-selected policies by held-out aspect and should not be promoted as the deployable main method.

### Limitations

- This is still an offline analysis over existing predictions, not a live router that actually invokes Qwen row by row.
- Test policy evaluation is deliberately limited to fixed comparisons and validation-selected rules; the full broad policy search is validation-only.
- Qwen call rate is estimated from the policy conditions and row counts.
- Aggregate CSVs are public-safe, but local output CSVs under `outputs/` remain ignored because they may include row-level prediction material.

### Next Step

- Promote the asymmetric global router as the main local-to-Qwen thesis method.
- Use `docs/thesis_figure_data/qwen_local_qwen_loao_pareto.csv` for the F1/call-rate curve.
- Keep per-aspect routing as an upper-bound diagnostic only.
- Do not start lightweight learned routers, candidate-wise Qwen judging, aspect descriptions, or full Qwen LoRA LOAO unless a supervisor requests more evidence beyond this completed global-router contribution.

## 2026-07-17: Pre-Registration — Protocol-Matched BoW And Sentence Embeddings

### Objective

Add the missing Bag-of-Words and frozen sentence-embedding rungs to the primary twelve-fold all-row LOAO benchmark. A strict train-only character TF-IDF rerun is included because the historical lexical baseline fitted the candidate label text together with the permitted training reviews.

### Frozen Protocol And Methods

- Primary `example_filtered` training construction.
- All official evaluation rows and held-out-only gold projection.
- Raw canonical singleton candidate aspect and empty predictions allowed.
- Validation pair micro F1 selects one threshold per method and aspect; the test stage loads frozen thresholds and never reselects them.
- Shared global word+character TF-IDF balanced Logistic Regression sentiment component.
- Registered methods: count BoW word 1–2 grams, strict train-only character TF-IDF 3–5 grams, frozen `all-MiniLM-L6-v2`, and frozen `e5-base-v2`.
- Both sentence encoders are fixed a priori and reported independently; test data does not select an encoder.
- Full parameters, revisions, threshold grid, tie-breakers, metrics, outputs, gates, and stopping rule: `configs/experiments/loao_bow_sentence_embedding_v1.json`.

### Planned Execution

1. Focused tests plus one-aspect validation smoke.
2. Full validation only; verify and freeze 48 method-aspect thresholds.
3. Single frozen test evaluation.
4. Record commands, runtime, results, limitations, and tracked aggregate/per-aspect numeric outputs in `docs/experiments/loao_bow_sentence_embedding_v1.md`.

Raw review-level outputs, model files, and cached embeddings remain local under ignored paths.

### Observed Validation Gate

The clean validation command was:

```powershell
python .\scripts\run_similarity_loao_baselines.py --stage validation --device cpu --local-files-only --output-dir .\outputs\baselines\loao_bow_sentence_embedding_v1 --public-output-dir .\docs\thesis_figure_data
```

- Git commit: `63c3e724b336137c49425e1e7b59e257c037a5e4`.
- Runtime: `313.8` seconds on CPU; the GPU was intentionally left to another process.
- Isolation: manifest contains SHA-256 fingerprints for `train.csv` and `validation.csv` only; `test.csv` was not loaded.
- Completeness audit: 4 methods × 12 aspects, 48 finite frozen thresholds, 48 fold-result rows, 50,736 validation prediction rows, no review text in prediction JSONL.
- Mean validation pair micro F1: BoW `0.3269`, strict train-only TF-IDF `0.3943`, MiniLM `0.3603`, E5 `0.3944`.
- Mean validation presence F1: BoW `0.3763`, strict train-only TF-IDF `0.4624`, MiniLM `0.4286`, E5 `0.4564`.
- Mean validation presence average precision: BoW `0.2820`, strict train-only TF-IDF `0.4027`, MiniLM `0.4029`, E5 `0.4327`.

No method, encoder, threshold, or tie-break rule was changed after observing validation. The next permitted action is the single full test-stage run using `outputs/baselines/loao_bow_sentence_embedding_v1/validation/selection_manifest.json`.
