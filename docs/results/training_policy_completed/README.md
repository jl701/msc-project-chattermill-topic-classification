# Completed training-policy comparison

**Evidence role: later validation-only robustness, not official confirmation.**

These two CSVs are byte-preserving copies of the completed 14 September 2026
analysis outputs. Each policy uses its own few-shot detector and QLoRA sentiment
component. The earlier mixed-component diagnostic is not substituted here.

- [`aggregate_metrics.csv`](aggregate_metrics.csv): 3 systems × 2 training
  policies × 2 description conditions, each aggregating twelve aspect folds.
- [`paired_contrasts.csv`](paired_contrasts.csv): registered policy/composition
  contrasts and nominal conditional intervals. Negative masked-minus-filtered
  values mean lower performance when reviews are retained.

## Reading the columns

`heldout_pair_f1` is the equal-aspect mean of held-out pair micro-F1.
`pooled_heldout_pair_f1` instead pools confusion counts before computing F1.
`overall_pair_micro_f1` includes familiar candidates. `oracle_stage2_f1` uses
gold aspect presence for diagnosis and is not an end-to-end system score.
`nominal_ci_*` conditions on the selected systems, fixed taxonomy and training
realisations. It does not include neural seed variation or sequential selection.

`N` means candidate names and `D` adds the registered minimal definition.
`review_filtered` removes entire training reviews containing the held-out aspect.
`label_masked_all_reviews` retains reviews but excludes that aspect's supervision,
including negatives. Demonstration selection uses each policy's training pool.

## Provenance and availability

The source is the author-held `QLoRA_Training_Policy_Final_20260914/analysis`
artifact bundle. It also contains per-fold outputs, the source manifest, audit
and review-confusion/bootstrap arrays. Those row-level artifacts are not
distributed here. The copied aggregates allow inspection, not independent
score-level reproduction.

The analysis entry point is
[`analyse_taxonomy_policy_qlora_completion.py`](../../../scripts/analyse_taxonomy_policy_qlora_completion.py).
The training recipe is
[`taxonomy_policy_qlora_completion_v1.json`](../../../configs/experiments/taxonomy_policy_qlora_completion_v1.json).

Review-filtered validation values here are from the aligned completion analysis,
including its canonical rescoring. Do not silently replace values in earlier
dated validation reports with this table. Neither version changes the frozen
official-test result.
