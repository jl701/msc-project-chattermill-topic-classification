# Qwen-Local Hybrid Direction

Last updated: 2026-07-03

This note records the current modelling pivot after the fixed held-out-aspect Qwen LoRA run, the single-fold all-row Qwen LoRA validation pilot, the first offline local-to-Qwen LOAO cascade diagnostic, and the completed local score/margin routing analysis.

## Decision

The next modelling direction is not to run the full 12-fold fine-tuned Qwen LoRA LOAO experiment with the current JSON SFT recipe. The current recipe failed the validation gate on the hard `Company brand: Competitor` all-row fold:

| System | Fold / Split | Pair Micro F1 | Precision | Recall | FP Rows / 100 |
| --- | --- | ---: | ---: | ---: | ---: |
| Local DistilBERT baseline | `Company brand: Competitor` validation | 0.2490 | 0.1902 | 0.3605 | 12.2990 |
| Qwen zero-shot | `Company brand: Competitor` validation | 0.2397 | 0.1645 | 0.4419 | 17.7862 |
| Best Qwen LoRA singleton SFT branch | `Company brand: Competitor` validation | 0.1900 | 0.1667 | 0.2209 | 7.2848 |

The most promising immediate direction is a hybrid local-to-Qwen system:

```text
DistilBERT = cheap calibrated gate and high-throughput baseline
Qwen = semantic judge for uncertain, unfamiliar, or validation-routed unseen-aspect cases
```

This is motivated by the stronger local-to-Gemini fixed-split cascade and the local-to-Qwen LOAO diagnostics. The first diagnostic used only existing prediction agreement and already improved mean test pair micro F1 over both local-only and Qwen-only:

| LOAO System | Test Mean Pair Micro F1 | Precision | Recall | Qwen Call Rate |
| --- | ---: | ---: | ---: | ---: |
| Local DistilBERT only | 0.3128 | 0.3345 | 0.4315 | 0.0000 |
| Qwen zero-shot only | 0.3378 | 0.2379 | 0.8182 | 1.0000 |
| Global validation-selected agreement gate | 0.3470 | 0.4116 | 0.4147 | 0.1868 |
| Optimistic per-aspect validation-selected mixed policy | 0.4131 | 0.3461 | 0.5779 | 0.3645 |

The completed score-distance routing diagnostic strengthens this direction. It reran the strongest local DistilBERT LOAO branch with candidate score, threshold-distance, and sentiment-confidence exports, then reused existing Qwen zero-shot LOAO predictions without new Qwen calls:

| LOAO System | Test Mean Pair Micro F1 | Precision | Recall | FP Rows / 100 | FN Rows / 100 | Qwen Call Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local DistilBERT rerun only | 0.3158 | 0.3449 | 0.3965 | 10.4442 | 8.9582 | 0.0000 |
| Qwen zero-shot only | 0.3378 | 0.2379 | 0.8182 | 34.4150 | 1.9114 | 1.0000 |
| Global validation-selected score-distance gate | 0.3800 | 0.3323 | 0.5426 | 16.6667 | 4.0485 | 0.2239 |
| Per-aspect validation-selected mixed policy | 0.4186 | 0.3567 | 0.5549 | 14.9811 | 4.1168 | 0.3480 |

The global score-distance gate is the thesis-safe selected result because the policy was chosen on validation. The per-aspect mixed policy is a useful upper-bound diagnostic because it selects a policy separately for each held-out aspect from validation.

## Why This Pivot Is Needed

The current Qwen LoRA experiments trained Qwen to emit a JSON list of all relevant candidate aspect+sentiment labels. That makes Qwen compete with DistilBERT in a conventional supervised classification setting where the smaller encoder is already strong and easier to calibrate.

Qwen's observed advantage is different:

- it recognises semantically present held-out aspects well on positive-gold LOAO rows;
- it can reason over supplied label names and label meanings;
- it is useful when the local model is uncertain or when an unseen aspect needs semantic interpretation;
- it is weaker at calibrated abstention on empty-gold all-row LOAO rows.

Therefore the next experiments should use Qwen where semantic judgement matters most, while letting DistilBERT handle cheap calibrated gating and obvious negatives.

## Candidate Hybrid Sub-Directions

### 1. Score/Margin Uncertainty Routing

Use the local DistilBERT candidate-aspect selector's score, selected threshold, and distance to threshold to decide whether to call Qwen.

Candidate policies:

- call Qwen when the local aspect score is close to the validation-selected threshold;
- let Qwen rescue local near-miss negatives just below threshold;
- let Qwen confirm or veto local near-threshold positives just above threshold;
- keep local predictions when the score is far from threshold.

This is now the strongest no-new-Qwen-call method. The validation-selected global policy was `score_abs_replace_le_0.05`: when the local selector score is within `0.05` of the validation-selected threshold, replace the local prediction with Qwen; otherwise keep local. It improved test mean pair micro F1 from the local rerun's `0.3158` and Qwen-only `0.3378` to `0.3800`, with Qwen used on `22.4%` of rows.

### 2. Sentiment-Margin Routing

Use local aspect-conditioned sentiment probability margin or entropy to identify rows where the aspect selector is confident but sentiment polarity is uncertain.

Candidate policies:

- keep local aspect detection but use Qwen's sentiment when local sentiment margin is low;
- use Qwen only when both aspect score and sentiment margin are uncertain;
- compare Qwen sentiment substitution against local sentiment on positive-gold diagnostics.

This remains useful as a diagnostic, but the completed policy grid shows that sentiment-margin alone is not the main source of improvement. The best sentiment-margin test policy reached only `0.3204` mean pair micro F1 with a `1.1%` Qwen call rate, far below the score-distance gate. The current bottleneck is still unseen-aspect relevance detection and abstention calibration, not sentiment substitution alone.

### 3. Per-Aspect Validation Routing

Some held-out aspects are better handled by Qwen-led policies, while others need local/Qwen agreement. The completed per-aspect validation-selected policy reached `0.4186` test mean pair micro F1 with a `34.8%` Qwen call rate. This is strong evidence of model complementarity, but it has more selection freedom than the global gate and should be presented as an upper-bound diagnostic unless the per-aspect routing rule is frozen before deployment.

Candidate policies:

- choose local-only, Qwen-only, agreement, rescue, or confirmation policy per held-out aspect from validation;
- report global-policy and per-aspect-policy results separately;
- avoid presenting per-aspect selection as a final deployed policy unless the selection rule is fixed before test evaluation.

### 4. Candidate-Wise Qwen Semantic Judge

Reformulate Qwen from multi-label JSON generation to candidate-wise judgement:

```text
Input: review + one candidate aspect + optional aspect description
Output: absent / positive / negative / neutral
Decision: logits or validation-calibrated threshold
```

This better matches Qwen's semantic strength and directly targets absence calibration. It is the next new-Qwen-call direction if more modelling evidence is needed after the score-distance routing result. It should not be confused with the failed grouped/singleton JSON-SFT route: the candidate-wise formulation asks Qwen to judge one candidate aspect at a time and can be calibrated with logits or validation thresholds.

### 5. Small Meta-Classifier Over Local And Qwen Features

Train a lightweight validation-only router or calibrator over non-text features:

- local aspect score;
- distance to threshold;
- local selected flag;
- local sentiment probability and margin;
- Qwen predicted/non-empty flag;
- Qwen agreement with local;
- aspect ID.

This is a later option because the validation set is small per aspect and overfitting risk is high. It should only be used with strict validation/test separation and a simple model.

## First Registered Experiment In This Direction: Completed

### Purpose

Build a true local-to-Qwen score/margin routing diagnostic for full all-row LOAO.

### Local Rerun

Rerun the strongest local DistilBERT LOAO configuration so the prediction JSONL includes:

- candidate-aspect score;
- selected threshold;
- distance to threshold;
- selected-count features;
- aspect-conditioned sentiment predicted label;
- sentiment probabilities;
- sentiment margin;
- sentiment entropy.

Canonical command:

```powershell
python .\scripts\run_loao_heldout_aspect.py --baseline cross_encoder --strategy example_filtered --eval-row-scope all --selection-metric pair_micro_f1 --sentiment-mode transformer_aspect_conditioned --sentiment-epochs 3 --sentiment-learning-rate 2e-5 --sentiment-batch-size 16 --sentiment-eval-batch-size 64 --sentiment-class-weight balanced --sentiment-selection-metric accuracy --epochs 3 --batch-size 32 --eval-batch-size 96 --learning-rate 3e-5 --negatives-per-positive 3 --output-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702
```

### Hybrid Analysis

Reuse the existing Qwen zero-shot LOAO predictions. Do not call Qwen again for this diagnostic.

Canonical command:

```powershell
python .\scripts\analyse_local_qwen_loao_cascade.py --local-loao-dir .\outputs\baselines\loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702 --qwen-validation-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_validation_20260701 --qwen-test-dir .\outputs\llm\qwen_loao_heldout_aspect_all_rows_test_20260701 --output-dir .\outputs\analysis\local_qwen_loao_score_margin_cascade_20260702
```

### Original Decision Rule

Promote the method if validation-selected score/margin routing improves over the previous global agreement gate (`0.3470` mean test pair micro F1) or provides a materially better precision/recall/Qwen-call-rate trade-off.

If it does not improve, record it as a negative methodological check and move to the candidate-wise Qwen semantic judge formulation rather than more JSON-SFT sweeps.

### Observed Result

The experiment completed successfully.

Local export evidence:

- Output directory: `outputs/baselines/loao_cross_encoder_transformer_sentiment_example_filtered_micro_selection_score_export_20260702/`
- Fold coverage: `12 / 12` LOAO folds.
- Validation prediction rows per fold: `1,057`.
- Exported features present in all folds: `score_features` and `sentiment_features`.
- Approximate local runtime: 2026-07-02 23:37 to 2026-07-03 03:51 on an NVIDIA GeForce RTX 5050 Laptop GPU.

Hybrid analysis evidence:

- Output directory: `outputs/analysis/local_qwen_loao_score_margin_cascade_20260702/`
- Primary metric: mean pair micro F1 across the 12 held-out-aspect test folds.
- Global validation-selected policy: `score_abs_replace_le_0.05`.
- Global selected test result:
  - pair micro F1 mean `0.3800`;
  - precision mean `0.3323`;
  - recall mean `0.5426`;
  - false-positive rows / 100 mean `16.6667`;
  - false-negative rows / 100 mean `4.0485`;
  - Qwen call rate mean `0.2239`.
- Per-aspect validation-selected diagnostic:
  - pair micro F1 mean `0.4186`;
  - precision mean `0.3567`;
  - recall mean `0.5549`;
  - false-positive rows / 100 mean `14.9811`;
  - false-negative rows / 100 mean `4.1168`;
  - Qwen call rate mean `0.3480`.
- Best sentiment-margin-only diagnostic:
  - test pair micro F1 mean `0.3204`;
  - Qwen call rate mean `0.0106`.

Interpretation:

- Score-distance routing is a positive methodological result and should be promoted as the main Qwen follow-up method.
- The result supports the model-division claim: DistilBERT should provide cheap calibrated gating, while Qwen should be called selectively for locally uncertain unseen-aspect cases.
- Sentiment-margin-only routing is a weak/negative sub-result. It confirms that the useful uncertainty signal is mainly the local aspect selector's distance to threshold, not the sentiment classifier's confidence margin.
- The full 12-fold Qwen JSON-SFT LOAO should remain deferred. Stronger GPU access alone is not a reason to rerun the current SFT recipe.

## Guardrails For Future Sessions

- Do not run full 12-fold Qwen LoRA LOAO with the current JSON SFT recipe.
- Do not treat stronger GPU access alone as a launch condition.
- Do not mix fixed held-out-aspect, all-row LOAO, and positive-gold LOAO results in one leaderboard.
- Do not rerun Gemini fixed/cascade/description experiments unless a new thesis question requires it.
- Do not commit `outputs/`, raw predictions, review text, checkpoints, adapters, credentials, or API keys.
- Always record commands, inputs, outputs, metrics, interpretation, limitations, and next steps in tracked documentation.
