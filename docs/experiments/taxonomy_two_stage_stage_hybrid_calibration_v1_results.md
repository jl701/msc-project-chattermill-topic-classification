# Two-Stage Stage-Hybrid and QLoRA Calibration Results v1

Date completed: 2026-08-23

## Outcome

The fixed stage-wise hybrid is a positive validation result. Using Frozen Qwen few-shot for Stage 1 aspect selection and QLoRA for Stage 2 conditional sentiment raised the primary L2-D aspect-balanced held-out pair F1 to **0.525931**. This exceeds both source systems under the same twelve folds, 1,057 reviews, complete 36-pair grid and capped-two decoder.

The low-capacity QLoRA relative calibrator moved the primary held-out estimate only from **0.479903** to **0.482384**. Its paired interval crosses zero, so it is retained as an informative inconclusive ablation rather than promoted over the fixed hybrid.

## Primary L2-D comparison

| System | Evidence role | Held-out pair F1 | Review-bootstrap 95% CI | Overall pair F1 | Held-out presence F1 | Oracle-gated Stage-2 pair F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Frozen Qwen few-shot | Existing formal system | 0.504883 | [0.470242, 0.535321] | 0.546245 | 0.585356 | 0.765699 |
| QLoRA | Existing formal system | 0.479903 | [0.453208, 0.503958] | 0.586771 | 0.515242 | 0.885369 |
| **Few-shot Stage 1 + QLoRA Stage 2** | Fixed primary hybrid | **0.525931** | **[0.493052, 0.553766]** | 0.584098 | 0.585356 | 0.885369 |
| QLoRA Stage 1 + Few-shot Stage 2 | Fixed reverse control | 0.456888 | [0.430875, 0.481501] | 0.546057 | 0.515242 | 0.765699 |
| QLoRA relative calibrator | Review-cross-fitted primary | 0.482384 | [0.454119, 0.507613] | **0.591051** | 0.521053 | 0.885369 |
| QLoRA relative calibrator | All-validation fitted diagnostic | 0.483577 | [0.454942, 0.508648] | 0.593762 | 0.521597 | 0.885369 |

The fitted calibrator row is descriptive only. The review-cross-fitted row is its valid primary estimate because every evaluated review was excluded from model fitting and threshold selection.

## Paired differences

The 20,000-draw bootstrap resampled the same `row_uid` review clusters for every method, fold and condition.

| Comparison on L2-D | Mean F1 difference | Paired 95% CI | Fold outcomes |
| --- | ---: | --- | --- |
| Primary hybrid minus Frozen Qwen few-shot | **+0.021048** | **[+0.000775, +0.042656]** | 7 improved / 1 tied / 4 worsened |
| Primary hybrid minus QLoRA | **+0.046028** | **[+0.021611, +0.069753]** | 7 improved / 0 tied / 5 worsened |
| Relative calibrator minus QLoRA | +0.002481 | [-0.004619, +0.009603] | 6 improved / 0 tied / 6 worsened |
| Relative calibrator minus Frozen Qwen few-shot | -0.022498 | [-0.049718, +0.004767] | 6 improved / 0 tied / 6 worsened |

The primary hybrid therefore has positive paired evidence against both source systems. The calibrator has no defensible held-out improvement despite its slightly higher point estimate and overall pair F1.

## What the component swap shows

This is a stronger diagnostic than a generic ensemble result:

- The primary hybrid exactly inherits Frozen Qwen few-shot's Stage-1 presence F1 (`0.585356`) and QLoRA's oracle-gated Stage-2 pair F1 (`0.885369`).
- The reverse control exactly inherits QLoRA's weaker thresholded Stage 1 and Frozen Qwen's weaker Stage 2, and its held-out F1 falls to `0.456888`.
- The result therefore supports a specific division of labour: Frozen Qwen few-shot supplies the more effective held-out aspect gate, while QLoRA supplies the more accurate sentiment decoder after an aspect has been selected.
- Overall pair F1 for the primary hybrid (`0.584098`) is only `0.002672` below QLoRA (`0.586771`) and `0.037853` above Frozen Qwen few-shot. Its primary held-out gain is not purchased by a large overall degradation.

Under the unchanged N interface, the same primary hybrid reaches held-out pair F1 `0.514235` and overall pair F1 `0.583001`. No N-specific fitting or threshold selection was performed.

## QLoRA relative calibrator diagnostics

The cross-fitted calibrator used exactly two features: the absolute QLoRA aspect logit and its difference from the within-review median of the other eleven aspect logits. Across the 72 recorded fits (60 review-block fits plus 12 final descriptive fits):

- selected thresholds ranged from `0.67` to `0.82`;
- the mean standardised coefficient was positive for both the absolute logit (`1.0762`) and relative logit (`0.8869`);
- all fits converged within at most 11 iterations;
- review overlap in the 60 primary cross-fitted train/evaluation pairs was exactly zero;
- held-out-aspect training instances were exactly zero.

The positive relative coefficient confirms that within-review ranking carries useful information, but the fixed two-feature logistic correction does not convert that signal into a reliable held-out F1 improvement. It modestly improves overall pair F1 (`0.591051` versus QLoRA's `0.586771`) while leaving the thesis-primary held-out claim essentially unchanged. No broader calibrator family was added after observing this outcome.

## Integrity audit

The formal local run verified:

- 841 critical input files against their received SHA-256 receipts;
- the exact formal execution boundary `aa84212976a652d62cfca31ed8bf0516a216c485`;
- 96 method/fold/condition metric records;
- 72 calibrator fit records and 60 leakage-safe review-block fits;
- 20,000 synchronised review-cluster bootstrap draws with seed 13;
- complete 1,057-review, twelve-aspect, three-sentiment score grids;
- zero missing/non-finite probabilities, duplicate evidence identities, prediction collapses, cross-fit review overlaps, held-out-label training instances, failures or test contracts;
- maximum two sentiments per selected aspect;
- `official_test_opened=false`, `include_official_test=false`, `test_contract_count=0`.

The QLoRA L2-D a07/a12 seen-candidate canonicalisation was reproduced exactly. Held-out D probabilities were not changed.

## Reproduction

The formal recomputation used the same package versions as the cloud formal-v2 environment for the relevant analysis stack: Python 3.11, NumPy 2.2.6, Pandas 2.3.3 and scikit-learn 1.8.0.

```powershell
$env:PYTHONWARNINGS='error::FutureWarning'
uv run --python 3.11 --with numpy==2.2.6 --with pandas==2.3.3 --with scikit-learn==1.8.0 scripts/analyse_taxonomy_two_stage_stage_hybrid_calibration.py
```

Tracked aggregate evidence is in `docs/thesis_figure_data/taxonomy_two_stage_stage_hybrid_calibration_v1/`. The detailed row evidence, verified input hashes, coefficients, thresholds and audit manifest remain under `outputs/experimental/taxonomy_two_stage_stage_hybrid_calibration_v1/`; raw reviews, checkpoints and score shards remain outside Git.

## Decision

Promote **Frozen Qwen few-shot Stage 1 + QLoRA Stage 2** as the validation-selected L2-D two-stage system for any later, separately authorised final evaluation. Keep the relative calibrator and reverse hybrid as bounded ablations. The official test remains closed, and this result alone does not open it.
