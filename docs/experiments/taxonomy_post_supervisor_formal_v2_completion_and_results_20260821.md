# Taxonomy two-stage formal-v2 completion and validation results

Date: 21 August 2026

Protocol: `taxonomy_two_stage_formal_v2`

Deployed commit: `aa84212976a652d62cfca31ed8bf0516a216c485`

Status: complete, locally replicated, hash-audited and test-blind

## Executive conclusion

The mandatory revised cloud campaign is complete and safe to use as formal
validation evidence. All 81 registered result payloads are present: 27 each
for genuine two-stage DistilBERT, Frozen-Qwen few-shot and QLoRA. The local
audit verified every synchronization receipt, all checkpoint contents, all
selected parameter records, 648 score shards and 3,082,212 validation score
rows. It found zero failures, zero test contracts, zero non-finite values, zero
resume conflicts and no globally empty, saturated or constant prediction
distribution. The official test partition was not opened.

On the central Level 2 held-out pair metric, Frozen-Qwen few-shot and QLoRA are
the leading formal methods. With minimal definitions (`D`), their mean held-out
pair micro-F1 values are 0.5049 and 0.4799 respectively, compared with 0.1054
for DistilBERT. QLoRA has the strongest full-grid overall pair micro-F1
(0.5868). The matched description effect is model-dependent and not a general
win: it is small for DistilBERT and Frozen-Qwen, and positive but uncertain for
QLoRA on end-to-end held-out pair F1.

## 1. Audit boundary and result inventory

Only the immutable local copy under
`C:/Msc_DSML/Msc_Project/cloud_backups/taxonomy_two_stage_formal_v2_r2`
was read. No FABSA source split was loaded by the analysis script.

| Evidence | Audited count | Outcome |
|---|---:|---|
| Local verified synchronization receipts | 1,789 | all referenced files matched byte counts and SHA-256 |
| Trainable campaign jobs | 150/150 | complete; failure 0; test-contract 0 |
| Frozen-Qwen few-shot scopes | 15/15 | complete; failure 0; test-contract 0 |
| Candidate checkpoint manifests | 90 | 45 DistilBERT + 45 QLoRA |
| Checkpoint-internal files | 900 | all internal SHA-256 values matched |
| Seen-validation selections | 45 | 15 per method; sealed and finite |
| Formal result payloads | 81 | exactly 27 per method |
| Score shards | 648 | exactly eight per result |
| Validation score rows | 3,082,212 | finite, within [0,1], unique identities |

The smallest per-result score diversity was 587 distinct aspect probabilities
and 7,305 distinct sentiment probabilities. All nine method/level/condition
groups produced non-empty and non-saturated held-out predictions. Stored
prediction counts were consistent with their confusion matrices. Result
thresholds and selected learning rates matched the sealed seen-validation
selection records; `N` and `D` therefore use the same selected outer-scope
model and thresholds.

## 2. Formal Level 2 model results

All values below are unweighted means over the same twelve leave-one-aspect-out
validation folds. `N` gives the canonical aspect name only; `D` adds the frozen
minimal definition. The held-out pair score is the primary generalisation
quantity. Overall pair F1 evaluates the complete twelve-aspect grid.

| Method | Condition | Held-out pair F1 | Held-out aspect F1 | Stage-1 AP | Oracle-stage-2 pair F1 | Overall pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| DistilBERT | N | 0.1057 | 0.1217 | 0.2099 | 0.8761 | 0.4969 |
| DistilBERT | D | 0.1054 | 0.1269 | 0.2381 | 0.8690 | 0.4999 |
| Frozen-Qwen few-shot | N | 0.5019 | 0.5618 | 0.5817 | 0.8299 | 0.5456 |
| Frozen-Qwen few-shot | D | **0.5049** | **0.5854** | 0.5059 | 0.7657 | 0.5462 |
| QLoRA | N | 0.4420 | 0.4639 | 0.5339 | 0.8878 | 0.5842 |
| QLoRA | D | 0.4799 | 0.5152 | **0.5834** | **0.8854** | **0.5868** |

The two-stage decomposition matters for interpretation. DistilBERT has high
oracle-gated sentiment performance but very weak end-to-end held-out F1: its
aspect detector is the bottleneck, not its conditional sentiment classifier.
Frozen-Qwen few-shot obtains the best held-out end-to-end result through the
strongest thresholded aspect F1. QLoRA has the best ranking AP under `D`, the
best oracle-stage-2 result and the best overall full-grid result, but its
thresholded held-out aspect F1 remains below Frozen-Qwen few-shot.

## 3. Matched effect of supplying a definition

These are paired `D-N` fold differences. The interval is a deterministic
20,000-draw percentile bootstrap over the twelve matched aspect folds. The
reported multiplicity-adjusted value is an exact sign-flip p-value with Holm
adjustment across the three formal methods for the same metric. These remain
validation analyses and are not a substitute for the sealed final test.

| Method | N held-out pair F1 | D held-out pair F1 | D-N | Bootstrap 95% interval | D better / worse folds |
|---|---:|---:|---:|---:|---:|
| DistilBERT | 0.1057 | 0.1054 | -0.0003 | [-0.0248, 0.0259] | 6 / 6 |
| Frozen-Qwen few-shot | 0.5019 | 0.5049 | +0.0030 | [-0.0601, 0.0553] | 7 / 5 |
| QLoRA | 0.4420 | 0.4799 | +0.0379 | [-0.0390, 0.1184] | 8 / 4 |

No formal model has a precise end-to-end held-out pair-F1 improvement from
minimal definitions at twelve folds. The stage-1 AP effects are clearer but
point in different directions:

- DistilBERT: +0.0282, interval [0.0139, 0.0447], Holm p=0.0044;
- Frozen-Qwen few-shot: -0.0758, interval [-0.1366, -0.0206], Holm p=0.0449;
- QLoRA: +0.0495, interval [0.0102, 0.0937], Holm p=0.0532.

This supports a deliberately narrow thesis claim: a definition is a
model-dependent interface intervention, not a universally beneficial source
of information. Ranking gains do not necessarily translate through a frozen
threshold and the second stage into end-to-end pair gains.

## 4. Matched model comparisons at Level 2-D

| Comparison on held-out pair F1 | Mean difference | Bootstrap 95% interval | Better / worse folds | Holm p |
|---|---:|---:|---:|---:|
| Frozen-Qwen few-shot minus DistilBERT | +0.3995 | [0.3049, 0.4878] | 12 / 0 | 0.0015 |
| QLoRA minus DistilBERT | +0.3745 | [0.2589, 0.4874] | 12 / 0 | 0.0015 |
| QLoRA minus Frozen-Qwen few-shot | -0.0250 | [-0.1188, 0.0690] | 6 / 6 | 0.6265 |

Both Qwen approaches clearly outperform the genuine two-stage DistilBERT
cross-encoder on unseen-aspect end-to-end pair prediction. The available
twelve folds do not distinguish QLoRA from Frozen-Qwen few-shot on that held-
out metric. QLoRA nevertheless leads on the overall full-grid metric, which
captures performance on seen candidates as well as the single held-out aspect.

## 5. Level 4 structured parent-group shift

Level 4 has only three parent-group folds, so every result below should remain
descriptive. The per-group table is the primary evidence; the method mean is a
compact summary rather than a strong population estimate.

| Method | Company brand | Staff support | Value | Three-group mean |
|---|---:|---:|---:|---:|
| DistilBERT | 0.0693 | 0.2131 | 0.2463 | 0.1762 |
| Frozen-Qwen few-shot | **0.5123** | **0.5805** | **0.5916** | **0.5615** |
| QLoRA | 0.4243 | 0.5423 | 0.5789 | 0.5152 |

Frozen-Qwen few-shot is higher on held-out pair F1 in all three groups. QLoRA
again has the highest full-grid overall pair F1 (0.5893 versus 0.5417 for
Frozen-Qwen and 0.4983 for DistilBERT). With only three groups, exact sign-flip
tests cannot provide useful inferential resolution; the individual group
values and the consistent ordering are more honest than a significance claim.

## 6. Experiment-level context

The broader comparison CSV also contains the audited Level 1 reference and
local Level 2 TF-IDF, E5, Frozen-Qwen zero-shot and DCWT results. Level 1 pair
micro-F1 is 0.7051, but it is a closed, fully seen-taxonomy task and must not be
ranked directly against Level 2 or Level 4. Within Level 2-D, the previously
audited local held-out pair means are 0.2773 for TF-IDF, 0.2222 for E5, 0.4513
for Frozen-Qwen zero-shot and 0.1453 for DCWT. Frozen-Qwen few-shot improves
over its zero-shot counterpart descriptively (0.5049 versus 0.4513), although
the current table does not claim a paired inferential test across those two
separate audited runs.

## 7. Thesis-safe claims and remaining decisions

The current validation evidence supports the following claims:

1. The two-stage architecture makes the source of error measurable: aspect
   detection, rather than sentiment prediction conditional on a correct
   aspect, is the dominant bottleneck for DistilBERT.
2. Frozen-Qwen few-shot is the strongest held-out end-to-end method in the
   registered Level 2-D and Level 4-D summaries.
3. QLoRA is the strongest full-grid method and has the strongest Level 2-D
   aspect ranking and oracle-gated sentiment diagnostics.
4. Minimal definitions do not deliver a universal end-to-end improvement;
   their effect depends on the scoring architecture and thresholded decoder.
5. Level 4 is useful as a structured-shift stress test, but three groups support
   descriptive conclusions only.

The next scientific decision is whether Level 3 remains appendix-only using
existing exact evidence or receives any additional models. That decision must
be made without opening the official test. Thesis integration, diagram updates
and the eventual one-pass test freeze remain separate later phases.

## 8. Reproducible outputs

- Audit receipt: `docs/experiments/taxonomy_post_supervisor_formal_v2_audit_20260821.json`
- Fold-level formal table: `docs/thesis_figure_data/taxonomy_post_supervisor_formal_v2/formal_fold_results.csv`
- Model-condition summary: `docs/thesis_figure_data/taxonomy_post_supervisor_formal_v2/formal_model_condition_summary.csv`
- Description contrasts: `docs/thesis_figure_data/taxonomy_post_supervisor_formal_v2/formal_l2_paired_description_effects.csv`
- Paired model contrasts: `docs/thesis_figure_data/taxonomy_post_supervisor_formal_v2/formal_paired_model_effects.csv`
- Two-stage diagnostic summary: `docs/thesis_figure_data/taxonomy_post_supervisor_formal_v2/formal_l2_stage_view_summary.csv`
- Level 4 group table: `docs/thesis_figure_data/taxonomy_post_supervisor_formal_v2/formal_l4_group_results.csv`
- Cross-level/model context table: `docs/thesis_figure_data/taxonomy_post_supervisor_formal_v2/experiment_level_model_comparison.csv`
- Reproducible analysis: `scripts/analyse_taxonomy_post_supervisor_formal_v2.py`

Heavy checkpoints, score shards, prompt caches and synchronization receipts
remain outside Git under the local backup root. The compact audit and tables
contain their verification hashes and are the shareable evidence layer.
