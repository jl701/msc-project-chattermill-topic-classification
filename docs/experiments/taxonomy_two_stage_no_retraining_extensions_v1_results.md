# Validation-only no-retraining extensions: results

## Outcome

Both preregistered extensions completed on 23 August 2026, and neither passed
its promotion rule. The fixed Frozen-Qwen few-shot Stage 1 plus QLoRA Stage 2
hybrid remains the validation-selected Level 2 system.

The official test partition was not opened. Every run, analysis and compact
output records `include_official_test=false` and `test_contract_count=0`.

## Primary comparison

The primary metric is the aspect-balanced mean L2-D held-out pair micro-F1.
Intervals are synchronised 20,000-draw review-cluster bootstrap intervals for
the candidate-minus-fixed-reference difference.

| Method | D held-out pair F1 | D overall pair F1 | Gain vs fixed hybrid | Paired 95% interval | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Fixed Frozen-Qwen few-shot Stage 1 + QLoRA Stage 2 | **0.525931** | 0.584098 | reference | n/a | retain |
| E5-retrieved Frozen-Qwen few-shot Stage 1 + QLoRA Stage 2 | 0.517509 | 0.593919 | -0.008422 | [-0.034933, 0.017751] | do not promote |
| Smooth few-shot/QLoRA Stage-1 fusion + QLoRA Stage 2 | 0.513711 | **0.607851** | -0.012220 | [-0.032407, 0.008033] | do not promote |

The smooth fusion improves the easier overall full-grid metric but does not
improve the held-out-aspect primary metric. This is consistent with its
review-cross-fitted parameters being selected only from seen-aspect evidence:
better seen-grid calibration does not imply better transfer to the omitted
aspect. Its D held-out precision/recall are `0.484155/0.628265`, compared with
`0.506637/0.620757` for the fixed hybrid.

Retrieval changes the D held-out precision/recall trade-off to
`0.529163/0.567097`: precision rises, but the recall loss is larger. Its
held-out presence AP is `0.503911`, effectively unchanged from the fixed
hybrid's `0.505870`, so the retrieved demonstrations do not create a stronger
ranking signal for an unseen aspect.

## Unchanged transfer to name-only condition N

All parameters selected on D were transferred to N without N-specific
selection.

| Method | N held-out pair F1 | N overall pair F1 | N presence F1 | N presence AP |
| --- | ---: | ---: | ---: | ---: |
| Fixed hybrid | **0.514235** | 0.583001 | **0.561755** | **0.581652** |
| Smooth fusion | 0.480601 | **0.604690** | 0.510826 | 0.627871 |
| Retrieval few-shot | 0.443499 | 0.589458 | 0.475172 | 0.501759 |

These N results are secondary unchanged-transfer diagnostics, not an extra
selection surface. In particular, the larger retrieval decline under N does
not authorise a new retriever, prompt or N-specific threshold search.

## Execution and integrity

The smooth fusion reused the verified formal-v2 score grids and required no
new model inference. Its audit verified 841 critical inputs.

The retrieval study used frozen `intfloat/e5-base-v2` embeddings and the
pinned Frozen-Qwen model without weight updates. It produced exactly 304,416
validation scores across 12 folds and two conditions: 164,892 new Qwen
inferences plus 139,524 condition-N seen scores copied exactly from their
identical condition-D prompts. All 24 manifest and 24 score units completed.

Both audits independently verified 1,789 formal receipts covering 4,007
unique formal files, 45 selections and 60 review-cross-fitted parameter fits.
Across the two studies there were zero:

- official-test or non-zero test-contract accesses;
- failures, OOMs or non-finite values;
- cross-fit train/evaluation review overlaps;
- held-out-aspect parameter-selection instances;
- held-out aspects or non-training rows in retrieval demonstrations;
- D/N seen-demonstration mismatches;
- duplicate row-evidence identities or prediction collapses.

The maximum decoded sentiments per selected aspect was two. Retrieval audit
SHA-256 is
`2112aa37e711092744f90048e0c29cbdbedbe69e7be2df5a9cfc7ccf933ee739`;
smooth-fusion audit SHA-256 is
`2585e75d487fef824b03ffeab21576821d2fb9e9e20ad023992370b2eac00386`.
The complete ignored evidence tree was also copied to
`C:/Msc_DSML/Msc_Project/cloud_backups/taxonomy_two_stage_no_retraining_extensions_v1`;
all 143 source files had a corresponding backup file and an identical SHA-256.

## Interpretation and final decision

The experiment answers two useful bounded questions. Softly combining two
Stage-1 confidence sources does not outperform using the stronger few-shot
gate directly. Selecting semantically similar demonstrations with a frozen E5
retriever also does not reliably improve unseen-aspect transfer. Both point
estimates are below the fixed hybrid and both paired intervals include zero.

No post-outcome method family or hyperparameter search is introduced. The two
extensions are retained as controlled validation-only negative/inconclusive
ablations. The fixed hybrid remains the only candidate for any later,
separately authorised final test evaluation.

## Reproduction

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python scripts/run_taxonomy_two_stage_retrieval_few_shot_validation.py --local-files-only
python scripts/analyse_taxonomy_two_stage_smooth_fusion.py
python scripts/analyse_taxonomy_two_stage_retrieval_few_shot.py
python scripts/build_taxonomy_two_stage_no_retraining_extension_summary.py
```

The machine-readable protocol is
`configs/experiments/taxonomy_two_stage_no_retraining_extensions_v1.json`.
Compact thesis-facing tables are under
`docs/thesis_figure_data/taxonomy_two_stage_no_retraining_extensions_v1/`.
Raw score shards, prompt manifests, embeddings and row evidence remain in the
ignored local `outputs/` and backup trees and are not eligible for Git.
