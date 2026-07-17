# LOAO Presence Metric Correction — 17 July 2026

## Status

Complete. Approved for GitHub publication on 17 July 2026.

## Objective

Correct the singleton-candidate presence F1 calculation so that true-negative absent rows do not inflate positive-class F1. Recompute the existing TF-IDF, DistilBERT, and Qwen all-row LOAO presence metrics from their frozen prediction files without changing any model, threshold, checkpoint, prompt, parser, or test decision.

## Scope And Frozen Evidence

- Dataset: official FABSA validation and test rows.
- Protocol: twelve-fold all-row LOAO, one supplied held-out aspect per fold.
- Systems: lexical TF-IDF plus global sentiment, DistilBERT candidate cross-encoder plus aspect-conditioned sentiment, and zero-shot Qwen3-4B.
- Selection: retain the original validation-selected thresholds and checkpoints and the frozen indexed Qwen prompt.
- Execution: metric-only recomputation; no training, inference, API call, or router search.
- Local raw sources and corrected machine-readable outputs remain under ignored `outputs/`.

## Planned Metric Definition

For candidate presence, collapse every non-empty aspect-sentiment pair set to present and every empty set to absent. Report positive-class precision, recall, and F1 from row-level TP, FP, and FN counts. True negatives remain available for prevalence and exact-match diagnostics but do not enter positive-class F1.

## Planned Validation

- Add a regression case in which one true-negative row and one missed positive row previously produced `0.5` instead of the correct `0.0` presence F1.
- Confirm the generic multilabel micro and macro F1 implementations remain unchanged for genuine multi-column label matrices.
- Re-run the complete unit-test suite, bytecode compilation, and `git diff --check`.

## Implementation

- `multilabel_scores` now computes micro F1 directly as `2TP / (2TP + FP + FN)` and macro F1 as the mean positive-class F1 across label columns. This preserves standard multilabel behaviour while preventing a one-column target from being treated as binary accuracy.
- `evaluate_pair_and_aspect` now emits explicit row-level `presence_*` metrics and TP/FP/FN/TN counts.
- LOAO and Qwen runners include the new presence fields in their aggregate metric lists.
- `scripts/recompute_loao_presence_metrics.py` recomputes metrics from frozen JSONL predictions, rejects duplicate row IDs, requires all 12 aspects, and fails if any previously reported pair metric changes beyond floating-point tolerance.

## Corrected Aggregate Results

All values below are unweighted means across the twelve held-out aspects. Mean F1 is the mean of fold-level F1 values, not the harmonic mean of the displayed mean precision and recall.

### Validation

| System | Presence Precision | Presence Recall | Presence F1 | Pair Micro F1 | Legacy Aspect Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.4668 | 0.5819 | **0.4684** | 0.3988 | 0.7861 |
| DistilBERT | 0.4166 | 0.4661 | 0.3838 | 0.3620 | 0.8187 |
| Qwen | 0.2413 | **0.8574** | 0.3451 | 0.3293 | 0.6319 |

### Test

| System | Presence Precision | Presence Recall | Presence F1 | Pair Micro F1 | Legacy Aspect Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | **0.4292** | 0.5553 | **0.4300** | **0.3780** | 0.7810 |
| DistilBERT | 0.3697 | 0.4235 | 0.3390 | 0.3158 | 0.8060 |
| Qwen | 0.2489 | **0.8788** | 0.3551 | 0.3378 | 0.6367 |

The legacy aspect micro F1 values are retained only to identify the superseded field in historical outputs. They included true-negative absence decisions and must not be reported as positive-class candidate-presence F1.

## Stability Check

The frozen prediction files reproduce every previous pair metric. Across all systems, splits, and folds, the maximum absolute difference for pair samples F1, pair micro F1, pair precision, pair recall, or pair macro F1 is below `1e-16`. Therefore:

- no model ranking based on pair micro F1 changes;
- no validation-selected threshold or checkpoint changes;
- no training or inference rerun is required;
- the correction affects candidate-presence reporting and the one-column aspect micro/macro fields only.

## Interpretation

- TF-IDF gives the strongest mean candidate-presence F1 and the best balance of precision and recall among the three frozen systems.
- Qwen has the highest presence recall but much lower precision, confirming that its main all-row failure is over-predicting the supplied candidate on absent rows.
- DistilBERT is more conservative than Qwen but has lower mean presence recall and F1 than TF-IDF.
- End-to-end pair micro F1 remains TF-IDF `0.3780`, Qwen `0.3378`, and DistilBERT `0.3158` on test.

## Reproduction And Outputs

Command:

```powershell
python .\scripts\recompute_loao_presence_metrics.py
```

Local machine-readable outputs, containing no review text:

- `outputs/analysis/loao_presence_metric_correction_20260717/per_aspect_corrected.csv`
- `outputs/analysis/loao_presence_metric_correction_20260717/aggregate_corrected.csv`
- `outputs/analysis/loao_presence_metric_correction_20260717/summary.json`

Public aggregate table:

- `docs/thesis_figure_data/loao_presence_metric_correction.csv`

The original prediction JSONL files remain under ignored `outputs/` and are not proposed for GitHub.

## Validation

- `python -m pytest -q`: `142 passed`.
- `python -m compileall -q src scripts tests`: passed with an ignored output-directory bytecode cache.
- `git diff --check`: passed; line-ending conversion warnings only.
- Focused credential-pattern scan: no matches.
- Approved baseline-review files are published separately from thesis drafts and experimental unified-pair work.
