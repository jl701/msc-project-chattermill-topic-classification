# Experiment Reproducibility Register

Last updated: 2026-07-17

This register audits whether the project experiments have enough recorded parameters for the user, Aji, or a future project session to reproduce the reported results. It complements `docs/experiment_log.md`, which remains the chronological ledger.

## Recording Standard

Every future meaningful experiment should record:

- exact command and working directory
- dataset path and split/protocol
- training/evaluation row scope and label scope
- model name and model family
- prompt variant or feature pipeline
- important hyperparameters, including learning rate, epochs, batch sizes, sequence length, class weighting, threshold-selection metric, random seed, and negative sampling
- output directory
- validation-selection rule
- headline and supporting metrics
- runtime, hardware, API endpoint family, token usage, and approximate cost when relevant
- whether outputs are local-only because they may contain review text
- caveats, such as smoke-test status, subset evaluation, or non-comparability with full validation/test results

Every future experiment must also be closed out with a safe GitHub sync:

- update the chronological record in `docs/experiment_log.md`
- update this register when a new canonical experiment, parameter set, or headline result is introduced
- commit and push safe tracked files after validation passes
- never commit `outputs/`, raw data, credentials, checkpoints, model weights, or raw prediction files containing review text
- if push cannot be completed immediately, record the reason in the experiment log and complete the push as soon as possible

## Audit Summary

The current project is mostly reproducible for all dissertation-relevant results.

Strongly recorded:

- post-2026-06-21 experiments in `docs/experiment_log.md`
- held-out aspect protocols and LOAO experiments
- strongest local non-LLM fixed-split and LOAO results
- Qwen indexed zero-shot fixed-split and full all-row LOAO results
- Gemini Flash/Flash-Lite/Pro fixed-split runs
- local-to-Gemini cascade and Pro cascade deep-dive
- Gemini-generated aspect-description ablation
- Gemini-assisted qualitative error taxonomy

Recoverable from local artifacts but less well centralised before this register:

- early closed-topic DistilBERT tuning runs
- early Qwen LoRA feasibility pilots
- first TF-IDF Logistic Regression baseline
- exploratory pair-label cross-encoder runs

Remaining limitations:

- Some early exploratory commands were not logged verbatim when they were first run. Their key parameters are recoverable from output directory names, `summary.json`, script defaults, and later documentation, but not always from a single historical command line.
- Most scripts save configurations and metrics, but not all save the exact shell command, git commit, package versions, or hardware metadata. This should be improved for future experiments if time allows.
- Generated prediction files under `outputs/` may contain review text and must remain uncommitted.

## Canonical Experiment Register

| Area | Experiment / Claim | Parameter Record Status | Canonical Command Or Parameter Source | Main Output / Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Data exploration | FABSA schema, split sizes, label distribution | Adequate | `python .\scripts\explore_fabsa.py` | `outputs/fabsa_exploration/summary.json`, `PROJECT_OVERVIEW.md` | Early exploratory result; sufficient for dataset description. |
| Split design | Held-out organisation and held-out aspect split analysis | Complete | `python .\scripts\analyse_split_candidates.py`; `python .\scripts\build_fabsa_splits.py` | `docs/evaluation_protocol.md`, `outputs/split_analysis/summary.json` | Split rows, held-out orgs/aspects, and leakage checks are documented. |
| Protocol freeze | LOAO open-topic all-row protocol v1 | Complete | n/a | `docs/evaluation_protocol.md`, `docs/loao_heldout_aspect.md`, `docs/experiment_log.md` | Freezes all-row LOAO scope, candidate set, aggregation, and metric hierarchy before Qwen zero-shot/fine-tuning LOAO. |
| Closed-topic traditional | First TF-IDF Logistic Regression baseline | Partially centralised, recoverable | `python .\scripts\run_tfidf_logreg.py --eval-split all --output-dir .\outputs\baselines\tfidf_logreg` | `outputs/baselines/tfidf_logreg/`, `PROJECT_OVERVIEW.md` | Superseded by the classical sweep; record is adequate as a first lower-bound baseline. |
| Closed-topic traditional | Full classical baseline sweep | Complete | `python .\scripts\run_classical_baselines.py --output-dir .\outputs\baselines\classical` | `docs/closed_topic_baselines.md`, `outputs/baselines/classical/summary.json`, `validation_sweep.csv` | Best config: word+char TF-IDF Linear SVM, word ngrams `1-3`, `C=0.2`, threshold `-0.38`. |
| Closed-topic encoder | DistilBERT/BERT closed-topic tuning | Complete for main runs | `python .\scripts\run_transformer_baseline.py --protocol closed-topic --epochs <N> --batch-size 16 --learning-rate <LR> --pos-weight sqrt --output-dir .\outputs\baselines\transformer\<run>` | `docs/closed_topic_baselines.md`, `outputs/baselines/transformer/*/summary.json` | Each summary stores model name, max length, batch size, LR, weight decay, epochs, warmup, seed, AMP, class weighting, history, and best validation/test metrics. |
| Held-out organisation | Traditional SVM generalisation baseline | Complete | `python .\scripts\run_generalisation_baselines.py --protocol heldout-org --refined-cross-org --output-dir .\outputs\baselines\generalisation_refined` | `docs/generalisation_baselines.md`, `outputs/baselines/generalisation_refined/summary.json` | Best refined config and validation/test metrics are documented. |
| Held-out organisation | DistilBERT generalisation baseline | Complete | `python .\scripts\run_transformer_baseline.py --protocol heldout-org --epochs 10 --batch-size 16 --learning-rate 6e-5 --pos-weight sqrt --output-dir .\outputs\baselines\transformer_heldout_org\distilbert_lr6e-5_sqrt` | `docs/generalisation_baselines.md`, `outputs/baselines/transformer_heldout_org/*/summary.json` | LR `4e-5` to `7e-5` checks are recorded in summaries and docs. |
| Held-out aspect | Lexical candidate-label baseline with global sentiment | Complete | `python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode global --output-dir .\outputs\baselines\generalisation_global_sentiment_rerun` | `docs/generalisation_baselines.md`, local output summaries | Fixed three-aspect lower-bound baseline. |
| Held-out aspect | Lexical candidate-label baseline with shallow aspect-conditioned sentiment | Complete | `python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode aspect_conditioned --output-dir .\outputs\baselines\generalisation_aspect_conditioned_sentiment` | `docs/generalisation_baselines.md`, `docs/experiment_log.md` | Methodologically cleaner than global sentiment, but weaker empirically. |
| Held-out aspect | Lexical selector with DistilBERT aspect-conditioned sentiment | Complete | `python .\scripts\run_generalisation_baselines.py --protocol heldout-aspect --strategy both --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --output-dir .\outputs\baselines\generalisation_transformer_sentiment_lr2e-5_ep3_balanced_accuracy` | `docs/generalisation_baselines.md`, `docs/non_llm_open_topic_baseline.md` | Controls the aspect selector and isolates sentiment model strength. |
| Held-out aspect | Candidate-aspect DistilBERT cross-encoder with global sentiment | Complete | `python .\scripts\run_aspect_label_aware_baseline.py --strategy both --sentiment-mode global --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_lr2e-5_ep3_neg3` | `docs/generalisation_baselines.md`, `docs/heldout_aspect_error_analysis.md`, local summaries | Important predecessor to the final local non-LLM branch. |
| Held-out aspect | Pair-label cross-encoder exploratory runs | Adequate for negative result | `python .\scripts\run_label_aware_baseline.py --strategy label_masked --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 2e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\label_aware_pair_cross_encoder_lr2e-5_ep3_neg3` | `outputs/baselines/label_aware_pair_cross_encoder_*`, `docs/generalisation_baselines.md` | Underperformed; not a headline result, but output parameters are preserved. |
| Held-out aspect | Strongest local non-LLM fixed-split baseline | Complete | `python .\scripts\run_aspect_label_aware_baseline.py --strategy example_filtered --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\aspect_label_aware_transformer_sentiment_lr3e-5_ep3_neg3_example_filtered` | `docs/non_llm_open_topic_baseline.md`, `docs/generalisation_baselines.md`, local summary | Headline fixed held-out-aspect non-LLM result: `0.6071` pair samples F1. |
| Held-out aspect tuning | Strong local non-LLM tuning attempts | Complete for important runs | Output directories named by LR/epochs/top-k plus `summary.json` | `docs/non_llm_open_topic_baseline.md`, `outputs/baselines/aspect_label_aware_transformer_sentiment_*` | Checked LR `2e-5`, `3e-5`, `4e-5`, 5 epochs, top-2 cap, and label-masked LR `3e-5`. |
| LOAO | Lexical global-sentiment LOAO, all-row and positive-row | Complete | See `docs/loao_heldout_aspect.md` reproduction commands | `docs/loao_heldout_aspect.md`, `docs/experiment_log.md` | Includes sample-F1 and micro-F1 threshold-selection variants. |
| LOAO | Protocol-matched Count BoW, strict TF-IDF, MiniLM, and E5 all-row baselines | Complete | Validation and test: `python .\scripts\run_similarity_loao_baselines.py --stage <validation|test> --device cpu --local-files-only --output-dir .\outputs\baselines\loao_bow_sentence_embedding_v1 --public-output-dir .\docs\thesis_figure_data`; paired table: `python .\scripts\analyse_similarity_loao_results.py` | `configs/experiments/loao_bow_sentence_embedding_v1.json`, `docs/experiments/loao_bow_sentence_embedding_v1.md`, `docs/thesis_figure_data/loao_bow_sentence_embedding_v1_*.csv` | Four methods × 12 aspects under the same all-row, example-filtered, raw-candidate, pair-set protocol. Validation freezes 48 thresholds before test load. Test pair micro F1: BoW `0.3225`, strict TF-IDF `0.3667`, MiniLM `0.3699`, E5 `0.3791`. Raw predictions remain ignored and omit review text. |
| LOAO | Lexical shallow aspect-conditioned sentiment LOAO | Complete | `python .\scripts\run_loao_heldout_aspect.py --baseline lexical --strategy both --sentiment-mode aspect_conditioned --selection-metric pair_micro_f1 --output-dir .\outputs\baselines\loao_heldout_aspect_lexical_aspect_conditioned_micro_selection` | `docs/loao_heldout_aspect.md` | Confirms shallow aspect-conditioned sentiment is not stronger. |
| LOAO | Strongest DistilBERT selector plus DistilBERT aspect-conditioned sentiment LOAO | Complete | `python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630` | `docs/loao_heldout_aspect.md`, `docs/experiment_log.md` | Full 12-fold run complete; LR `2e-5` tuning check also documented. |
| Error analysis | Local/Qwen held-out aspect error analysis | Complete enough | `python .\scripts\analyse_heldout_aspect_errors.py --output-dir .\outputs\analysis\heldout_aspect` | `docs/heldout_aspect_error_analysis.md`, local analysis outputs | Uses row IDs and aggregate labels in docs; raw review text remains local. |
| Qwen | Closed-topic zero-shot smoke | Adequate for smoke result | `python .\scripts\run_qwen_zero_shot.py --split validation --limit 100 --load-in-4bit --output-dir .\outputs\qwen_zero_shot_val100` | `docs/qwen_feasibility.md`, `outputs/qwen_zero_shot_*` | Smoke only, not comparable with full validation/test baselines. |
| Qwen | Closed-topic QLoRA feasibility pilots | Recoverable and now centralised | See Qwen LoRA parameter table below | `docs/qwen_feasibility.md`, `outputs/qwen_lora_*/summary.json` | Exact historical commands were not all logged, but key parameters and metrics are in summaries. |
| Qwen | Held-out-aspect indexed zero-shot baseline | Complete | `python .\scripts\run_qwen_heldout_aspect_smoke.py --split validation --limit 10000 --load-in-4bit --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_smoke\validation_indexed_full`; `python .\scripts\run_qwen_heldout_aspect_smoke.py --split test --limit 10000 --load-in-4bit --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_smoke\test_indexed_full` | `docs/qwen_feasibility.md`, `docs/generalisation_baselines.md`, local summaries | Full validation/test prompt baseline, not fine-tuned Qwen. |
| Qwen LOAO | Open-weight indexed zero-shot all-row LOAO baseline | Complete | `python .\scripts\run_qwen_loao_heldout_aspect.py --split validation --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701`; `python .\scripts\run_qwen_loao_heldout_aspect.py --split test --prompt-variant indexed --load-in-4bit --resume --output-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701`; `python .\scripts\analyse_qwen_loao_predictions.py --validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701` | `docs/qwen_feasibility.md`, `docs/generalisation_baselines.md`, local summaries under `outputs/llm/` | Full 12-fold all-row validation/test run. Model: `Qwen/Qwen3-4B-Instruct-2507`, 4-bit NF4, indexed candidate IDs. Test mean pair samples F1 `0.1212`, pair micro F1 `0.3378`, valid JSON `1.0000`. Raw predictions remain local-only because they contain review text. |
| Qwen LOAO analysis | Qwen zero-shot LOAO versus preferred local DistilBERT LOAO | Complete | `python .\scripts\analyse_qwen_loao_comparison.py --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --qwen-positive-dir .\outputs\analysis\qwen_loao_positive_diagnostic_20260701 --distilbert-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_20260630 --distilbert-lr2e5-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_lr2e-5_20260701 --output-dir .\outputs\analysis\qwen_loao_full_interpretation_20260701` | `docs/qwen_loao_experiment_analysis.md`, derived local CSV/JSON under `outputs/analysis/qwen_loao_full_interpretation_20260701/` | No new model calls. Records the 6/6 per-aspect Qwen-vs-DistilBERT split, high-recall/low-precision Qwen behaviour, positive-gold diagnostic gap, and thesis interpretation. Derived outputs remain local-only. |
| Qwen | Held-out-aspect SFT data preparation | Complete | `python .\scripts\prepare_qwen_heldout_aspect_sft_data.py --strategy both --prompt-variant indexed --output-dir .\outputs\qwen_heldout_aspect_sft_indexed` | `docs/qwen_feasibility.md`, `docs/generalisation_baselines.md` | Generated data remains local and uncommitted. |
| Gemini | Hosted API dry-run, smoke, prompt/max-token sweep | Complete | See `docs/gemini_candidate_label_baseline.md` commands | `docs/gemini_candidate_label_baseline.md`, `outputs/llm/gemini_candidate_label_*/summary.json` | Captures model, prompt variant, response format, max tokens, token usage, schema validity, latency, and cost inputs. |
| Gemini | Full fixed-split Flash baseline | Complete | `python .\scripts\run_gemini_heldout_aspect.py --split both --limit 10000 --prompt-variant indexed --response-format json_schema --response-format-fallback --max-tokens 2048 --output-dir .\outputs\llm\gemini_candidate_label_20260701_0145_fixed_full` | `docs/gemini_candidate_label_baseline.md`, local summary | Cost recomputed from stored token counts when explicit rates were not stored. |
| Gemini | Full fixed-split Pro and Flash-Lite baselines | Complete | See `docs/tasks_1_to_3_thesis_prep.md` commands with explicit cost rates | `docs/gemini_candidate_label_baseline.md`, `docs/tasks_1_to_3_thesis_prep.md`, local summaries | Full hosted Pareto comparison is reproducible except for requiring local API credentials. |
| Gemini cascade | Local-to-Gemini Flash-Lite/Flash/Pro cascade | Complete | See `docs/local_gemini_cascade.md` commands | `docs/local_gemini_cascade.md`, `outputs/analysis/local_gemini_cascade_*` | Policy search is validation-only; test labels used only for final evaluation and later explanation. |
| Gemini cascade | Score/margin uncertainty check for local-to-Gemini cascade | Complete negative check | See `docs/local_gemini_cascade.md` score/margin commands | `docs/local_gemini_cascade.md`, local score-export/cascade outputs under `outputs/` | Re-ran the strongest local fixed held-out-aspect baseline to export local selector score features. Reused existing Gemini predictions only. Score/margin features did not improve validation-selected F1 or reduce Flash/Pro call rate, so the original validation-reliability proxy remains the headline cascade uncertainty signal. |
| Gemini cascade | Pro cascade deep-dive | Complete | `python .\scripts\analyse_local_gemini_cascade.py` | `docs/local_gemini_cascade.md`, `outputs/analysis/local_gemini_cascade_pro_deep_dive/summary.json` | Explains error complementarity and Pro-empty fallback. |
| Gemini descriptions | Gemini-generated aspect-description ablation | Complete | See `docs/gemini_aspect_descriptions.md` commands and tracked configs under `configs/` | `docs/gemini_aspect_descriptions.md`, local summaries under `outputs/llm/gemini_candidate_label_20260701_desc_*`, `outputs/analysis/gemini_description_ablation_summary.json` | Uses descriptions generated from aspect names only, with no validation/test review text. Flash-Lite label-only descriptions improve test pair samples F1 from `0.5516` to `0.5925`; stronger models show mixed trade-offs. Raw predictions remain local-only. |
| Qualitative analysis | Gemini-assisted qualitative error taxonomy | Complete | `python .\scripts\analyse_qualitative_error_taxonomy.py --output-dir .\outputs\analysis\qualitative_error_taxonomy_20260702 --max-examples-per-category 10 --max-gemini-examples-per-category 2 --snippet-chars 220 --gemini-draft --gemini-model vertex_ai/gemini-2.5-pro --gemini-max-tokens 7000 --request-timeout 240` | `docs/qualitative_error_taxonomy.md`, `report_notes.md`, local packet under `outputs/analysis/qualitative_error_taxonomy_20260702/` | Reuses existing predictions and summaries. `fixed_split_cases_no_text.jsonl` omits review text; with-text packets, Gemini prompt, and raw Gemini draft remain local-only under ignored `outputs/`. |
| Qwen LoRA | Final held-out-aspect LoRA SFT runner readiness | Complete dry-run and tests | `python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_tiny_20260702 --strategy example_filtered --output-dir .\outputs\qwen_lora_heldout_aspect_dry_run_20260702 --train-limit 8 --validation-limit 4 --test-limit 4 --epochs 1 --max-train-steps 1 --batch-size 1 --grad-accumulation-steps 1 --learning-rate 1e-4 --max-length 512 --max-input-tokens 512 --max-new-tokens 64 --dry-run` | `scripts/run_qwen_lora_heldout_aspect.py`, `tests/test_qwen_lora_heldout_runner.py`, `docs/experiment_log.md`, `report_notes.md` | Runner consumes indexed held-out-aspect SFT JSONL, writes a manifest, supports adapter-weight resume and prediction resume/skip, and has focused tests. This is not the actual Qwen train/generate smoke test. |
| Qwen LoRA | Tiny held-out-aspect LoRA smoke test | Complete smoke test | `python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_tiny_20260702 --strategy example_filtered --output-dir .\outputs\qwen_lora_heldout_aspect_tiny_smoke_evalmode_20260702 --train-limit 4 --validation-limit 1 --test-limit 1 --epochs 1 --max-train-steps 1 --batch-size 1 --grad-accumulation-steps 1 --learning-rate 1e-6 --max-length 512 --max-input-tokens 512 --max-new-tokens 96 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions` | `docs/experiment_log.md`, `report_notes.md`, local ignored summary under `outputs/qwen_lora_heldout_aspect_tiny_smoke_evalmode_20260702/` | Pipeline smoke only. Qwen loaded locally with 4-bit QLoRA, one train step completed, adapter saved, validation/test generation ran, JSON/schema validity was `1.0000`, and metrics were written. Not a performance result. |
| Qwen LoRA | Fixed held-out-aspect QLoRA adaptation run | Complete fixed-split result | Validation: `python .\scripts\run_qwen_lora_heldout_aspect.py --sft-data-dir .\outputs\qwen_heldout_aspect_sft_indexed --strategy example_filtered --output-dir .\outputs\llm\qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702 --eval-split validation --epochs 1 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-5 --weight-decay 0.0 --warmup-ratio 0.05 --max-length 512 --max-input-tokens 1024 --max-new-tokens 192 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --load-in-4bit --no-gradient-checkpointing --save-adapter --resume-predictions --skip-existing-predictions --save-epoch-adapters`; Test: same output dir with `--eval-split test --skip-training-if-adapter-exists` | `docs/qwen_feasibility.md`, `docs/thesis_result_tables.md`, `docs/experiment_log.md`, `report_notes.md`, local ignored output under `outputs/llm/qwen_lora_fixed_example_filtered_r8_lr1e-5_ep1_skipfix_20260702/` | Fixed three-aspect adaptation evidence only. Test pair samples F1 `0.5528`, pair micro F1 `0.5552`, pair macro F1 `0.4393`, valid JSON `1.0000`, schema-valid `0.9964`. The result modestly improves over Qwen zero-shot fixed split but remains below the strongest local fixed baseline and does not replace full LOAO. |
| Qwen LoRA | Single-fold all-row LOAO validation-gated adaptation pilot | Complete negative validation pilot | See `docs/experiment_log.md` and `report_notes.md` for the grouped, conservative-prompt, singleton neg1, singleton neg0.25, and singleton neg0.10 commands. Final validation command used `--heldout-aspect "Company brand: Competitor"`, `--eval-row-scope all`, `--prompt-variant indexed_conservative`, `--train-candidate-mode singleton`, `--singleton-negative-ratio 0.10`, LoRA r/alpha/dropout `8/16/0.05`, LR `1e-5`, `913` optimiser steps, 4-bit QLoRA, and validation-only evaluation. | `docs/experiment_log.md`, `report_notes.md`, local ignored outputs under `outputs/llm/qwen_lora_loao_single_fold_company_brand_competitor_*_20260702/` | Validation-gated negative result. Best singleton branch was neg0.10: validation pair micro F1 `0.1900`, precision `0.1667`, recall `0.2209`, FP rows / 100 `7.2848`, valid/schema `1.0000/1.0000`. It remained below same-fold Qwen zero-shot `0.2397` and local DistilBERT `0.2490`, so test and full 12-fold Qwen LoRA LOAO were not launched under this SFT recipe. |
| Qwen/local cascade | Local-to-Qwen offline LOAO cascade diagnostic | Complete positive diagnostic | `python .\scripts\analyse_local_qwen_loao_cascade.py --output-dir .\outputs\analysis\local_qwen_loao_cascade_20260702` | `scripts/analyse_local_qwen_loao_cascade.py`, `docs/experiment_log.md`, `report_notes.md`, `docs/llm_next_experiment_directions.md`, local ignored output under `outputs/analysis/local_qwen_loao_cascade_20260702/` | No new model calls. Reused existing full all-row LOAO local DistilBERT and Qwen zero-shot predictions. Global validation-selected agreement gate reached test mean pair micro F1 `0.3470` with Qwen call rate `0.1868`, improving over local-only `0.3128` and Qwen-only `0.3378`. Optimistic per-aspect validation-selected mixed policy reached `0.4131` with Qwen call rate `0.3645`. This supports developing a true score/margin local-to-Qwen uncertainty gate. |
| Qwen/local cascade | Local-to-Qwen score/margin LOAO routing | Complete positive score-distance result | Local export: `python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702`; analysis: `python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_score_margin_cascade_20260702` | `docs/qwen_local_hybrid_direction.md`, `docs/experiment_log.md`, `report_notes.md`, local ignored outputs under `outputs/baselines/...score_export_20260702/` and `outputs/analysis/local_qwen_loao_score_margin_cascade_20260702/` | Reused existing Qwen zero-shot LOAO predictions; no new Qwen calls. Export coverage confirmed for all 12 folds with `score_features` and `sentiment_features`. Global validation-selected score-distance gate `score_abs_replace_le_0.05` reached test mean pair micro F1 `0.3800`, precision `0.3323`, recall `0.5426`, Qwen call rate `0.2239`, improving over local rerun `0.3158`, Qwen-only `0.3378`, and previous global agreement gate `0.3470`. Per-aspect validation-selected mixed diagnostic reached `0.4186` with Qwen call rate `0.3480`. Sentiment-margin-only routing was weak (`0.3204`). Raw predictions remain local-only. |
| Qwen/local cascade | Local-to-Qwen asymmetric global router and Pareto analysis | Complete thesis-facing global-router result | `python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_asymmetric_score_router_20260703 --public-pareto-csv .\docs\thesis_figure_data\qwen_local_qwen_loao_pareto.csv --public-selected-csv .\docs\thesis_figure_data\qwen_local_qwen_loao_selected.csv` | `scripts/analyse_local_qwen_loao_cascade.py`, `tests/test_analyse_local_qwen_loao_cascade.py`, `docs/qwen_local_hybrid_direction.md`, `docs/experiment_log.md`, `report_notes.md`, `docs/thesis_figure_data/qwen_local_qwen_loao_pareto.csv`, `docs/thesis_figure_data/qwen_local_qwen_loao_selected.csv`, local ignored output under `outputs/analysis/local_qwen_loao_asymmetric_score_router_20260703/` | Reused the same local score export and Qwen zero-shot LOAO predictions; no Qwen calls or DistilBERT training. The full asymmetric policy grid was evaluated on validation; test evaluation was limited to fixed comparisons and validation-selected rules. The global selected rule was `score_asym_rescue_le_0.05_confirm_le_0.30`, reaching test mean pair micro F1 `0.3900`, pair samples F1 `0.0988`, precision `0.3530`, recall `0.5300`, FP rows / 100 `15.2752`, FN rows / 100 `4.2166`, and Qwen call rate `0.2905`. This improves over `score_abs_replace_le_0.05` (`0.3800`). The expanded-grid per-aspect diagnostic reached `0.4178` and remains diagnostic only. Public CSVs contain aggregate metrics only. |
| Qwen LoRA | Full LOAO launch plan | Complete planning note | See `docs/qwen_lora_loao_launch_plan.md` | `docs/qwen_lora_loao_launch_plan.md`, `docs/thesis_completion_roadmap.md`, `report_notes.md` | Not an experiment result. Records the completed fixed-split reference run, full 12-fold LOAO fold IDs, validation/test order, output patterns, adapter recovery, prediction recovery, resource assumptions, and launch gates. Full Qwen LoRA LOAO remains unrun. |
| Thesis tables | Thesis-ready aggregate result tables and figure data | Complete | `python .\scripts\build_thesis_result_tables.py` | `docs/thesis_result_tables.md`, `docs/thesis_figure_data/*.csv` | Aggregate-only tracked table pack. Keeps closed-topic, held-out organisation, fixed held-out aspect, all-row LOAO, positive-gold diagnostics, and Gemini/cascade deployment evidence separate. |
| Roadmap | LLM next experiment directions | Complete planning note | n/a | `docs/llm_next_experiment_directions.md` | Not an experiment result; records future order and cost rationale. |
| Roadmap | Thesis completion and Qwen full LOAO boundary | Complete planning note | n/a | `docs/thesis_completion_roadmap.md`, `docs/qualitative_error_taxonomy.md`, `report_notes.md` | Not an experiment result. Records the completed pre-full-LOAO evidence package and the later validation-gated decision to defer full fine-tuned Qwen LOAO until a revised absence-calibration recipe passes a single-fold gate. |

## Supplemental Qwen LoRA Pilot Parameters

These early Qwen feasibility runs were originally recorded as a table in `docs/qwen_feasibility.md`. The detailed parameters below are recovered from local `summary.json` files.

| Output Directory | Train / Eval Examples | Epochs | Batch / Grad Accum | LR | LoRA r / alpha / dropout | Runtime | Final Pair Micro F1 | Status |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| `outputs/qwen_lora_smoke/` | 16 / 5 | 1 | 1 / 4 | `2e-4` | 8 / 16 / 0.05 | 38 s | 0.8000 | Tiny smoke only |
| `outputs/qwen_lora_train500_eval100_ep1_r8_lr2e-4/` | 500 / 100 | 1 | 1 / 8 | `2e-4` | 8 / 16 / 0.05 | 728 s | 0.6476 | Feasibility |
| `outputs/qwen_lora_train500_eval100_ep2_r8_lr2e-4/` | 500 / 100 | 2 | 1 / 8 | `2e-4` | 8 / 16 / 0.05 | 1,586 s | 0.7193 | Feasibility |
| `outputs/qwen_lora_train500_eval100_ep3_r8_lr2e-4/` | 500 / 100 | 3 | 1 / 8 | `2e-4` | 8 / 16 / 0.05 | 2,285 s | 0.7283 | Feasibility |
| `outputs/qwen_lora_train1000_eval100_ep2_r8_lr2e-4_len512/` | 1000 / 100 | 2 | 1 / 8 | `2e-4` | 8 / 16 / 0.05 | 2,346 s | 0.7405 | Feasibility |
| `outputs/qwen_lora_train1000_eval100_ep3_r8_lr2e-4_len512/` | 1000 / 100 | 3 | 1 / 8 | `2e-4` | 8 / 16 / 0.05 | 3,451 s | 0.7478 final epoch, best documented pilot `0.7586` | Feasibility |
| `outputs/qwen_lora_train1000_eval100_ep3_r8_lr1e-4_len512/` | 1000 / 100 | 3 | 1 / 8 | `1e-4` | 8 / 16 / 0.05 | 3,466 s | 0.7616 | Feasibility |

Canonical command template:

```powershell
python .\scripts\run_qwen_lora_pilot.py --train-limit 1000 --eval-limit 100 --epochs 3 --batch-size 1 --grad-accumulation-steps 8 --learning-rate 1e-4 --max-length 512 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --output-dir .\outputs\qwen_lora_train1000_eval100_ep3_r8_lr1e-4_len512
```

Do not treat these pilots as final full-split Qwen results. They are 100-row validation-slice feasibility checks on a local laptop GPU.

## Supplemental Closed-Topic Transformer Tuning Parameters

Closed-topic transformer summaries under `outputs/baselines/transformer/*/summary.json` store full configs. The main tested runs were:

| Output Directory | Model | LR | Epochs | Batch | Pos Weight | Best Epoch | Test Pair Samples F1 | Test Pair Micro F1 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `distilbert_lr2e-5_len256_bs16_ep3` | DistilBERT | `2e-5` | 3 | 16 | none | 3 | 0.6760 | 0.6787 |
| `distilbert_lr2e-5_len256_bs16_ep4_sqrtpos` | DistilBERT | `2e-5` | 4 | 16 | sqrt | 4 | 0.7285 | 0.7128 |
| `distilbert_lr3e-5_len256_bs16_ep8_sqrtpos` | DistilBERT | `3e-5` | 8 | 16 | sqrt | 7 | 0.7678 | 0.7567 |
| `distilbert_lr4e-5_len256_bs16_ep10_sqrtpos` | DistilBERT | `4e-5` | 10 | 16 | sqrt | 8 | 0.7803 | 0.7738 |
| `distilbert_lr5e-5_len256_bs16_ep10_sqrtpos` | DistilBERT | `5e-5` | 10 | 16 | sqrt | 9 | 0.7744 | 0.7663 |
| `bertbase_lr2e-5_len256_bs8ga2_ep6_sqrtpos` | BERT-base | `2e-5` | 6 | 8, grad accum 2 | sqrt | 6 | 0.7453 | 0.7312 |

Canonical command template:

```powershell
python .\scripts\run_transformer_baseline.py --protocol closed-topic --model-name distilbert-base-uncased --epochs 10 --batch-size 16 --learning-rate 4e-5 --pos-weight sqrt --output-dir .\outputs\baselines\transformer\distilbert_lr4e-5_len256_bs16_ep10_sqrtpos
```

## Supplemental Held-Out Organisation Transformer Parameters

Held-out-organisation transformer summaries under `outputs/baselines/transformer_heldout_org/*/summary.json` store full configs.

| Output Directory | LR | Epochs | Batch | Pos Weight | Best Epoch | Test Pair Samples F1 | Test Pair Micro F1 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `distilbert_lr4e-5_sqrt` | `4e-5` | 10 | 16 | sqrt | 8 | 0.7570 | 0.7541 |
| `distilbert_lr5e-5_sqrt` | `5e-5` | 10 | 16 | sqrt | 8 | 0.7565 | 0.7567 |
| `distilbert_lr6e-5_sqrt` | `6e-5` | 10 | 16 | sqrt | 8 | 0.7575 | 0.7600 |
| `distilbert_lr7e-5_sqrt` | `7e-5` | 10 | 16 | sqrt | 8 | 0.7594 | 0.7629 |

The validation-selected report row uses LR `6e-5`; LR `7e-5` has slightly higher test metrics but was not validation-selected.

## Future Logging Improvement

For future experiments, prefer scripts that write a run manifest alongside `summary.json` containing:

- `command`: exact shell command
- `cwd`: working directory
- `git_commit`: current commit hash
- `python_version` and key package versions
- `hardware`: GPU or API model/endpoint family
- `started_at` and `finished_at`
- `data_dir` and split/protocol metadata
- all parsed CLI arguments

This would make future reproduction less dependent on human-written notes.
