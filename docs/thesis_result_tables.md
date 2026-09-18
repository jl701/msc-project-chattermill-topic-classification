# Authoritative LOAO Baseline Table

Last generated: 23 July 2026

This is the sole active baseline table for the dissertation. It is generated from
`configs/experiments/loao_active_baseline_registry_v1.json` and is limited to the
target-calibrated, twelve-fold, all-row LOAO test benchmark. Every retained method
predicts sentiment conditional on the supplied candidate aspect.

The primary metric is the unweighted mean of the twelve fold-level pair micro-F1
values. Precision, recall, presence F1, and false-positive/false-negative rows per
100 are fold means and are diagnostic rather than pooled-corpus scores.

| Method | Sentiment formulation | Pair F1 | Precision | Recall | Presence F1 | FP rows/100 | FN rows/100 | Thesis role |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Count BoW | Shared DistilBERT aspect-conditioned sentiment | 0.314412 | 0.337343 | 0.483116 | 0.342152 | 25.404 | 4.978 | Sanity control |
| Strict train-only character TF-IDF | Shared DistilBERT aspect-conditioned sentiment | 0.385640 | 0.415569 | 0.504221 | 0.417175 | 17.617 | 4.442 | Main classical baseline |
| MiniLM-L6-v2 | Shared DistilBERT aspect-conditioned sentiment | 0.388883 | 0.347270 | 0.493440 | 0.415874 | 12.251 | 4.973 | Appendix/repository |
| E5-base-v2 | Shared DistilBERT aspect-conditioned sentiment | 0.401301 | 0.341856 | 0.541620 | 0.431933 | 12.917 | 3.896 | Main frozen sentence-embedding baseline |
| DistilBERT review-candidate cross-encoder | DistilBERT aspect-conditioned sentiment | 0.315848 | 0.344925 | 0.396534 | 0.338961 | 10.444 | 8.958 | Fine-tuned contextual baseline |
| Frozen candidate-pair Qwen | Joint candidate-conditioned pair prediction | 0.337804 | 0.244003 | 0.651851 | 0.451565 | 14.288 | 4.584 | Matched QLoRA control |
| Candidate-pair QLoRA | Joint candidate-conditioned pair prediction | 0.483158 | 0.501423 | 0.499727 | 0.500890 | 7.147 | 6.070 | Main adaptation contribution |

## Permitted comparisons

- Count, strict TF-IDF, MiniLM, and E5 form the strict representation block: they
  share the outer data, candidate scope, frozen DistilBERT aspect-conditioned
  sentiment head, threshold-selection rule, and evaluation.
- Frozen candidate-pair Qwen and candidate-pair QLoRA form the matched adaptation
  block.
- DistilBERT and cross-family values share the outer benchmark, but their internal
  model training and calibration are not identical.

## Exclusions

Global-document sentiment, shallow sentiment, legacy TF-IDF, historical JSON-prompt
Qwen, and the legacy TF-IDF-to-Qwen router are not active rows. A rebuilt router is
a deployment analysis, not a baseline, and must be reported separately.

The machine-readable companion is
`docs/thesis_figure_data/loao_active_baselines_v1.csv`.
